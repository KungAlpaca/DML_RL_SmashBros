"""Super Smash Bros. Melee environment for PPO training with Stable-Baselines3.

Each `SmashEnv` runs its own Slippi Dolphin process and talks to it through
libmelee. The agent controls port 1; the in-game CPU controls port 2.
Observations are read straight from game memory (positions, damage, action
states, ...), so the policy is an MLP rather than a CNN.

One episode is one full game (until either side runs out of stocks). Between
episodes the env navigates the post-game, character select and stage select
screens by itself.

With headless=True (Linux only) it uses the ExiAI Dolphin build from libmelee's
author: no window, no rendering and Melee's fast-forward mode, which is much
faster for training.
"""

import configparser
import itertools
import logging
import multiprocessing as mp
import os
import platform
import shutil
import signal
import tempfile
import time
import weakref

import gymnasium as gym
import melee
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper
from stable_baselines3.common.vec_env.subproc_vec_env import _worker

from smash_rl.external_bots import bot_character, make_bot
from smash_rl.rewards import RewardTracker, RewardWeights

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if platform.system() == "Darwin":
    DEFAULT_DOLPHIN = os.path.join(PROJECT_DIR, "dolphin", "Slippi Dolphin.app", "Contents", "MacOS", "Slippi Dolphin")
else:  # Linux: libmelee runs the Slippi Online AppImage directly
    DEFAULT_DOLPHIN = os.path.join(PROJECT_DIR, "dolphin", "Slippi_Online-x86_64.AppImage")
HEADLESS_DOLPHIN = os.path.join(PROJECT_DIR, "dolphin", "Slippi_Online-x86_64-ExiAI.AppImage")
CAN_RUN_HEADLESS = platform.system() == "Linux"
DEFAULT_ISO = os.path.join(PROJECT_DIR, "Super Smash Bros Melee v1.02.iso")

# Stick positions (x, y) in libmelee's 0..1 range, 0.5 is neutral.
N, R, L, U, D = (0.5, 0.5), (1.0, 0.5), (0.0, 0.5), (0.5, 1.0), (0.5, 0.0)
UR, UL, DR, DL = (0.85, 0.85), (0.15, 0.85), (0.85, 0.15), (0.15, 0.15)
DIRS9 = [N, R, L, U, D, UR, UL, DR, DL]
B = melee.Button

# Each action is (main stick, button or None). Pressing A or B together with a
# full stick tilt gives smash attacks / directional specials; in the air it
# gives the matching aerial.
ACTIONS = (
    [(d, None) for d in DIRS9]                      # 0-8: move / crouch / fast-fall / drift
    + [(d, B.BUTTON_A) for d in DIRS9]              # 9-17: jab, tilts, smashes, aerials
    + [(d, B.BUTTON_B) for d in (N, R, L, U, D)]    # 18-22: specials (Fox: laser, side-B, up-B, shine)
    + [(d, B.BUTTON_X) for d in (N, R, L)]          # 23-25: jump (short hop), drifting
    + [(N, B.BUTTON_Z)]                             # 26: grab
    + [(d, B.BUTTON_L) for d in (N, R, L, D)]       # 27-30: shield, roll, spot dodge
)
# Buttons released on the last frame of each action repeat so they can be
# pressed again on the next step. Shield is kept held, since dropping it costs frames.
TAP_BUTTONS = (B.BUTTON_A, B.BUTTON_B, B.BUTTON_X, B.BUTTON_Z)
# Frames each action is held (60 / frame_skip decisions per second). Models trained with another value
# store it as model.frame_skip (MK5 and MK9 from the final training: 2); older ones used this one.
DEFAULT_FRAME_SKIP = 3


def model_frame_skip(model) -> int:
    """How many frames the model holds each action, as it was trained."""
    return int(getattr(model, "frame_skip", DEFAULT_FRAME_SKIP))

NUM_ANIMATIONS = 400  # libmelee action ids are < 400
NUM_CHARACTERS = 27  # in-game character ids 0..26 (Mario .. Roy); Sheik and the Ice Climbers have their own
PLAYER_FEATURES = 20
PLAYER_SIZE = PLAYER_FEATURES + NUM_CHARACTERS + NUM_ANIMATIONS
GLOBAL_FEATURES = 4
# Opponents for opponent="RANDOM": everyone on the character select screen. The CPU
# can't be made to pick Sheik, but a CPU Zelda may transform into her mid-game.
RANDOM_OPPONENTS = tuple(c for c in melee.Character
                         if c.value < NUM_CHARACTERS and c not in (melee.Character.SHEIK, melee.Character.NANA))
IN_GAME = (melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH)
# A-button attacks (jabs, tilts, smashes, aerials) are contiguous in libmelee's Action enum.
ATTACK_ACTIONS = range(melee.Action.NEUTRAL_ATTACK_1.value, melee.Action.DAIR.value + 1)
OTHER_PORT = {1: 2, 2: 1}
_LIVE_ENVS = weakref.WeakSet()  # envs in this process, closed when a worker exits
PID_DIR_ENV = "SMASH_DOLPHIN_PID_DIR"  # where workers note their Dolphin's pid (see NonDaemonSubprocVecEnv)


class _DropMenuEventNoise(logging.Filter):
    """libmelee logs this on every frame of some menu transitions in headless mode; it is harmless."""

    def filter(self, record):
        return not str(record.msg).startswith("Got a menu event in the middle of a frame")


logging.getLogger().addFilter(_DropMenuEventNoise())


def _action_id(player) -> int:
    value = getattr(player.action, "value", player.action)
    return int(value) if 0 <= int(value) < NUM_ANIMATIONS else NUM_ANIMATIONS - 1


def player_features(p, edge: float) -> np.ndarray:
    """20 continuous features, then a character one-hot, then an action-state one-hot."""
    f = np.zeros(PLAYER_SIZE, dtype=np.float32)
    f[:PLAYER_FEATURES] = (
        p.position.x / 100.0,
        p.position.y / 100.0,
        p.percent / 100.0,
        p.stock / 4.0,
        1.0 if p.facing else -1.0,
        float(p.on_ground),
        float(p.off_stage),
        float(p.invulnerable),
        min(p.hitlag_left, 30) / 10.0,
        min(p.hitstun_frames_left, 120) / 30.0,
        p.jumps_left / 2.0,
        p.shield_strength / 60.0,
        p.speed_air_x_self / 3.0,
        p.speed_ground_x_self / 3.0,
        p.speed_y_self / 3.0,
        np.clip(p.speed_x_attack / 5.0, -5, 5),
        np.clip(p.speed_y_attack / 5.0, -5, 5),
        min(p.action_frame, 120) / 60.0,
        (edge - abs(p.position.x)) / 100.0,
        float(p.iasa) if isinstance(p.iasa, (bool, int, float)) else 0.0,
    )
    character = int(getattr(p.character, "value", p.character))
    if 0 <= character < NUM_CHARACTERS:
        f[PLAYER_FEATURES + character] = 1.0
    f[PLAYER_FEATURES + NUM_CHARACTERS + _action_id(p)] = 1.0
    return f


def observation(gs, me, op, edge: float) -> np.ndarray:
    """What a model sees: `me` (the player it controls) first, then `op`, then 4 globals."""
    dx = (op.position.x - me.position.x) / 100.0
    dy = (op.position.y - me.position.y) / 100.0
    glob = np.array([dx, dy, np.hypot(dx, dy), min(gs.frame, 28800) / 28800.0], dtype=np.float32)
    obs = np.concatenate([player_features(me, edge), player_features(op, edge), glob])
    return np.clip(obs, -10.0, 10.0).astype(np.float32)


class SmashEnv(gym.Env):
    """Gymnasium env: agent (port 1) vs the Melee CPU (port 2) on one stage.

    opponent="RANDOM" gives the CPU a new random character every game. With it, or
    with headless=True, games go through the menus instead of restarting instantly.

    opponent="HUMAN" leaves port 2 to a person with their own controller (see play.py):
    they pick their character and press Start, and Dolphin no longer waits for the
    agent on every frame, so the game runs in real time and can be paused.

    two_player=True puts a second agent on port 2 instead of the CPU (see selfplay.py):
    `opponent` is then its character (or RANDOM), step() takes one action per port, and
    observations, rewards and infos come back per port (obs and rewards stacked along a
    leading axis of size 2, infos as {"p1": ..., "p2": ...}).

    With two_player and cpu_opponent=True, port 2 is the Melee CPU (cpu_level) instead and
    its action is ignored, so CPU games fit in the same per-port interface. character="RANDOM"
    gives port 1 a new random character every game.

    With two_player and opponent_bot=(kind, path) (see smash_rl/external_bots.py), port 2 is
    another team's bot (Kepiti's Fox, the Ganondorf agent): built inside this process, it reads
    the game state and sets port 2's controller every frame; action[1] is ignored.

    reward="v1" (M1, MK0-MK2), per step and from each player's point of view:
        damage_scale * (damage dealt - damage taken) + (stocks taken - stocks lost)
        - whiff_penalty * (A attacks that ended without hitting the opponent or their shield)
    reward="v2" (MK3 / MK4): knockouts worth 5, combo rewards, anti-spam and anti-camping
    penalties and defence rewards, see smash_rl/rewards.py.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        character: str = "FOX",
        opponent: str = "FOX",
        stage: str = "FINAL_DESTINATION",
        cpu_level: int = 1,
        frame_skip: int = 3,
        damage_scale: float = 0.01,
        whiff_penalty: float = 0.0,
        reward: str = "v1",
        two_player: bool = False,
        cpu_opponent: bool = False,
        port: int = 51441,
        headless: bool = False,
        dolphin_path: str = None,
        dolphin_home: str = None,
        iso_path: str = DEFAULT_ISO,
        gfx_backend: str = "OGL",
        emulation_speed: float = 0.0,
        max_menu_frames: int = 6000,
        hang_timeout: float = 30.0,
        opponent_bot: tuple = None,
    ):
        super().__init__()
        self.random_character = character.upper() == "RANDOM"
        self.character = melee.Character.FOX if self.random_character else melee.Character[character.upper()]
        self.random_opponent = opponent.upper() == "RANDOM"
        self.human_opponent = opponent.upper() == "HUMAN"
        self.opponent = None if self.random_opponent or self.human_opponent else melee.Character[opponent.upper()]
        self.opponent_bot = opponent_bot  # (kind, path); the bot itself is built in reset(), in the worker
        self._bot = None
        if opponent_bot is not None:
            if not two_player or cpu_opponent:
                raise ValueError("opponent_bot needs two_player=True and cpu_opponent=False")
            self.random_opponent = False
            self.opponent = melee.Character[bot_character(opponent_bot)]
        # Instant Match restarts a finished game without the menus. It is off with a random
        # opponent or a person (character select every game) and headless (it crashes the ExiAI build).
        self.instant_restart = not (self.random_character or self.random_opponent or self.human_opponent or headless)
        self.stage = melee.Stage[stage.upper()]
        self.cpu_level = int(cpu_level)
        self.frame_skip = frame_skip
        self.damage_scale = damage_scale
        self.whiff_penalty = whiff_penalty
        if reward not in ("v1", "v2"):
            raise ValueError(f"unknown reward {reward!r}")
        self.reward_version = reward
        self.two_player = two_player
        self.cpu_opponent = cpu_opponent  # with two_player: port 2 is the CPU, not a second agent
        self.port = port
        self.headless = headless
        self.dolphin_path = dolphin_path or (HEADLESS_DOLPHIN if headless else DEFAULT_DOLPHIN)
        self.dolphin_home = dolphin_home  # None: a fresh temporary Dolphin user folder per start
        self.iso_path = iso_path
        self.gfx_backend = gfx_backend
        self.emulation_speed = emulation_speed
        self.max_menu_frames = max_menu_frames
        self.hang_timeout = hang_timeout  # seconds without game states before Dolphin counts as frozen
        for path in (self.dolphin_path, self.iso_path):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"{path} not found - see setup.sh")

        self.edge = melee.stages.EDGE_GROUND_POSITION[self.stage]
        self.action_space = gym.spaces.Discrete(len(ACTIONS))
        n_obs = 2 * PLAYER_SIZE + GLOBAL_FEATURES
        self.observation_space = gym.spaces.Box(-10.0, 10.0, shape=(n_obs,), dtype=np.float32)

        self.console = None
        _LIVE_ENVS.add(self)
        self._gs = None
        self._prev = None
        self._stats = {}

    # ------------------------------------------------------------------ dolphin
    def _start_dolphin(self):
        self._stop_dolphin()
        console = melee.Console(
            path=self.dolphin_path,
            dolphin_home_path=self.dolphin_home,
            tmp_home_directory=self.dolphin_home is None,
            slippi_port=self.port,
            # Blocking input makes Dolphin wait for the agent every frame, which would freeze
            # a pause by a person (no game states arrive to answer with while paused).
            blocking_input=not self.human_opponent,
            # Wait at most hang_timeout for each game state, so a frozen Dolphin is noticed and
            # restarted instead of stalling training forever. A person's pause sends nothing for
            # as long as it lasts, so human games wait without a limit.
            polling_mode=not self.human_opponent,
            polling_timeout=0 if self.human_opponent else self.hang_timeout,
            fullscreen=False,
            disable_audio=not self.human_opponent,
            save_replays=False,
            gfx_backend="Null" if self.headless else self.gfx_backend,
            log_types=[],
            instant_match_restart=self.instant_restart,
            use_exi_inputs=self.headless,
            enable_ffw=self.headless,
        )
        game_ini = os.path.join(console._get_dolphin_home_path(), "GameSettings", "GALE01r2.ini")
        with open(game_ini) as f:
            text = f.read()
        # Instant Match restarts the same match as soon as one ends, skipping the menus.
        # Its third line selects a random stage; 00000000 keeps the stage we picked.
        head, sep, tail = text.partition("$Optional: Instant Match")
        tail = tail.replace("00000001 7C0802A6", "00000000 7C0802A6", 1)
        # This Slippi build only reads EmulationSpeed from the per-game ini, not Dolphin.ini.
        with open(game_ini, "w") as f:
            f.write(head + sep + tail + f"\n[Core]\nEmulationSpeed = {self.emulation_speed}\n")
        dolphin_ini = os.path.join(console._get_dolphin_config_path(), "Dolphin.ini")
        # Lower-case keys like libmelee writes them (Dolphin ignores case): mixing both would
        # leave duplicate keys behind in a Dolphin folder that is kept between runs.
        cfg = configparser.ConfigParser()
        cfg.read(dolphin_ini)
        width, height = (1152, 864) if self.human_opponent else (480, 360)
        cfg.set("Display", "RenderWindowWidth", str(width))
        cfg.set("Display", "RenderWindowHeight", str(height))
        cfg.set("Display", "RenderWindowAutoSize", "False")
        if self.human_opponent:  # port 2 is a standard controller, mapped in GCPadNew.ini by play.py
            cfg.set("Core", "SIDevice1", melee.ControllerType.STANDARD.value)
        with open(dolphin_ini, "w") as f:
            cfg.write(f)

        self.pad = melee.Controller(console=console, port=1, type=melee.ControllerType.STANDARD)
        self.cpu_pad = None
        if not self.human_opponent:
            self.cpu_pad = melee.Controller(console=console, port=2, type=melee.ControllerType.STANDARD)
        console.run(iso_path=self.iso_path)
        self.console = console
        pid_dir = os.environ.get(PID_DIR_ENV)
        if pid_dir:  # lets the vectorized env kill this Dolphin even if this worker process dies
            with open(os.path.join(pid_dir, f"{self.port}.pid"), "w") as f:
                f.write(str(console._process.pid))
        # libmelee's connect() waits for Dolphin with no time limit; same steps, bounded.
        stream = console._slippstream
        stream._worker.start()
        if not stream._buffer.poll(timeout=60) or not stream._buffer.recv():
            self._stop_dolphin()
            raise RuntimeError(f"could not connect to dolphin on port {self.port}")
        stream.running = True
        self.pad.connect()
        if self.cpu_pad is not None:
            self.cpu_pad.connect()
        self._running_level = self.cpu_level
        self._game_over = False

    def _stop_dolphin(self):
        """Kill Dolphin and libmelee's connection process; never waits forever, even on a frozen Dolphin."""
        if self.console is None:
            return
        console, self.console = self.console, None
        proc = getattr(console, "_process", None)
        if proc is not None and proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=10)
            except Exception:
                pass
        stream = getattr(console, "_slippstream", None)
        worker = getattr(stream, "_worker", None)
        if worker is not None and worker.pid is not None:  # libmelee's shutdown() would join it with no timeout
            stream._shutdown.set()
            worker.join(timeout=5)
            if worker.is_alive():
                worker.kill()
                worker.join(timeout=5)
            stream._worker = None
        try:
            console.stop()
        except Exception:
            pass

    def _next_state(self):
        if self.human_opponent:
            for _ in range(1000):
                gs = self.console.step()
                if gs is not None:
                    return gs
            raise RuntimeError("dolphin stopped sending game states")
        gs = self.console.step()
        if gs is None:
            raise RuntimeError(f"no game state for {self.hang_timeout:.0f} s: Dolphin froze")
        return gs

    def _alive(self) -> bool:
        proc = getattr(self.console, "_process", None)
        return self.console is not None and proc is not None and proc.poll() is None

    # --------------------------------------------------------------- episodes
    def set_cpu_level(self, level: int):
        """Takes effect from the next game.

        A new level restarts Dolphin: with instant restarts the menus are never visited, and
        changing the level of a CPU already set up on the character select screen gets the
        menu helper stuck.
        """
        self.cpu_level = int(np.clip(level, 1, 9))

    def _go_to_game(self):
        """Navigate menus until a fresh game with the right setup has started."""
        need_new_game = self._game_over
        last_frame = self._gs.frame if (need_new_game and self._gs is not None) else None
        # MenuHelper remembers that it already picked a stage, so use fresh ones for every visit.
        self.menu = melee.MenuHelper()
        self.cpu_menu = melee.MenuHelper()
        # A person can take as long as they like on the character select screen.
        for _ in itertools.count() if self.human_opponent else range(self.max_menu_frames):
            gs = self._next_state()
            if gs.menu_state in IN_GAME:
                self.pad.release_all()
                if self.cpu_pad is not None:
                    self.cpu_pad.release_all()
                if need_new_game:
                    # Wait for the previous game to end and the next one to begin.
                    if gs.frame < 0 or (last_frame is not None and gs.frame < last_frame):
                        need_new_game = False
                    last_frame = gs.frame
                    continue
                p1, p2 = gs.players.get(1), gs.players.get(2)
                if p1 is None or p2 is None:
                    continue
                if gs.frame >= 0 and p1.stock > 0 and p2.stock > 0:
                    return gs
                continue
            need_new_game = False
            cpu_ready = False
            if self.cpu_pad is not None and gs.menu_state == melee.Menu.CHARACTER_SELECT:
                p2 = gs.players.get(2)
                if self.two_player and not self.cpu_opponent:  # a second agent plays port 2 as a human slot
                    cpu_ready = p2 is not None and p2.character == self.opponent and p2.coin_down
                else:
                    cpu_ready = p2 is not None and p2.cpu_level == self.cpu_level and p2.character == self.opponent
                self.cpu_menu.menu_helper_simple(gs, self.cpu_pad, self.opponent, self.stage,
                                                 cpu_level=0 if self.two_player and not self.cpu_opponent
                                                 else self.cpu_level,
                                                 autostart=False)
            elif self.cpu_pad is not None:
                self.cpu_pad.release_all()
            # Only start once the CPU slot is set up (a person on port 2 presses Start
            # themselves); autostart also makes port 1 pick the stage.
            self.menu.menu_helper_simple(
                gs, self.pad, self.character, self.stage, cpu_level=0,
                autostart=gs.menu_state != melee.Menu.CHARACTER_SELECT or cpu_ready)
        raise RuntimeError("stuck in menus")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.random_opponent:
            self.opponent = RANDOM_OPPONENTS[self.np_random.integers(len(RANDOM_OPPONENTS))]
        if self.random_character:
            self.character = RANDOM_OPPONENTS[self.np_random.integers(len(RANDOM_OPPONENTS))]
        for attempt in range(5):
            try:
                cpu_in_game = not self.human_opponent and (not self.two_player or self.cpu_opponent)
                if self._alive() and self._running_level != self.cpu_level and cpu_in_game:
                    self._stop_dolphin()
                if not self._alive():
                    self._start_dolphin()
                gs = self._go_to_game()
                break
            except Exception as e:  # dolphin crashed or hung in menus: restart it
                if self.human_opponent:
                    raise  # most likely the person closed the Dolphin window
                print(f"[port {self.port}] restarting dolphin after: {e}", flush=True)
                self._stop_dolphin()
                time.sleep(2)
        else:
            raise RuntimeError("could not start a game after 5 attempts")
        if self.human_opponent:
            self.opponent = gs.players[2].character
        if self.opponent_bot is not None:
            if self._bot is None:
                # Its network gets a couple of threads: several envs share the CPU with training.
                self._bot = make_bot(self.opponent_bot, port=2, opponent_port=1, torch_threads=2)
            self._bot.reset()
        self._gs = gs
        self._game_over = False
        self._prev = self._snapshot(gs)
        self._stats = {port: {"damage_dealt": 0.0, "damage_taken": 0.0, "kills": 0, "deaths": 0,
                              "attacks": 0, "whiffs": 0} for port in (1, 2)}
        self._attack = {1: None, 2: None}  # [action, hit so far] of the A attack each port is doing
        if self.reward_version == "v2":
            self._rewarder = RewardTracker(gs, RewardWeights())
            self._stats = self._rewarder.stats
        if self.two_player:
            return self._both_obs(gs), self._both_info(gs, done=False)
        return self._obs(gs), self._info(gs, done=False)

    @staticmethod
    def _snapshot(gs):
        """({port: percent}, {port: stocks})"""
        return ({p: float(gs.players[p].percent) for p in (1, 2)}, {p: int(gs.players[p].stock) for p in (1, 2)})

    @staticmethod
    def _press(pad, action):
        """Set `pad` to one of ACTIONS; returns the button pressed (or None)."""
        stick, button = ACTIONS[int(action)]
        pad.release_all()
        pad.tilt_analog(B.BUTTON_MAIN, *stick)
        if button is not None:
            pad.press_button(button)
        return button

    def step(self, action):
        if self.two_player and not self.cpu_opponent and self._bot is None:
            pads = [(self.pad, action[0]), (self.cpu_pad, action[1])]
        else:
            pads = [(self.pad, action[0] if self.two_player else action)]
        pressed = [(pad, self._press(pad, a)) for pad, a in pads]
        if self.reward_version == "v2":
            for port, (_, a) in enumerate(pads, start=1):
                self._rewarder.set_input(port, *ACTIONS[int(a)])

        reward = np.zeros(2, dtype=np.float32)
        terminated = crashed = False
        gs = self._gs
        try:
            for i in range(self.frame_skip):
                if i == self.frame_skip - 1 and self.frame_skip > 1:
                    for pad, button in pressed:
                        if button in TAP_BUTTONS:
                            pad.release_button(button)
                if self._bot is not None:  # the other team's bot sets port 2 for this frame
                    self._bot.act(gs, self.cpu_pad)
                gs = self._next_state()
                # Leaving the game, or the frame counter going back (a timed-out game was instantly restarted).
                if gs.menu_state not in IN_GAME or gs.frame < self._gs.frame:
                    terminated = True
                    break
                reward += self._frame_reward(gs)
                p1, p2 = gs.players.get(1), gs.players.get(2)
                if p1 is None or p2 is None or p1.stock == 0 or p2.stock == 0:
                    terminated = True
                    break
            for pad, _ in pressed:
                pad.release_all()
        except Exception as e:  # Dolphin died or froze: end this game here, reset() starts a new Dolphin
            if self.human_opponent:
                raise
            print(f"[port {self.port}] dolphin failed mid-game ({type(e).__name__}: {e}); restarting it", flush=True)
            self._stop_dolphin()
            gs, terminated, crashed = self._gs, True, True

        if gs.menu_state in IN_GAME and 1 in gs.players and 2 in gs.players and not (terminated and gs.frame < self._gs.frame):
            self._gs = gs
        self._game_over = terminated
        if self.two_player:
            info = self._both_info(self._gs, done=terminated)
            for side in info.values():
                side["crashed"] = crashed
            return self._both_obs(self._gs), reward, terminated, False, info
        info = self._info(self._gs, done=terminated)
        info["crashed"] = crashed
        return self._obs(self._gs), float(reward[0]), terminated, False, info

    def _frame_reward(self, gs) -> np.ndarray:
        """This frame's reward for port 1 and port 2; also updates their stats."""
        if self.reward_version == "v2":
            return self._rewarder.frame(gs)
        pct, stock = self._snapshot(gs)
        prev_pct, prev_stock = self._prev
        self._prev = (pct, stock)
        deaths = {p: max(0, prev_stock[p] - stock[p]) for p in (1, 2)}
        # Percent resets to 0 on death, so only count increases within a stock.
        taken = {p: 0.0 if deaths[p] else max(0.0, pct[p] - prev_pct[p]) for p in (1, 2)}

        rewards = np.zeros(2, dtype=np.float32)
        for p in (1, 2):
            o = OTHER_PORT[p]
            hit = taken[o] > 0 or deaths[o] > 0 or gs.players[o].action == melee.Action.SHIELD_STUN
            whiffs = self._track_attack(gs, p, hit)
            stats = self._stats[p]
            stats["damage_dealt"] += taken[o]
            stats["damage_taken"] += taken[p]
            stats["kills"] += deaths[o]
            stats["deaths"] += deaths[p]
            rewards[p - 1] = (self.damage_scale * (taken[o] - taken[p]) + deaths[o] - deaths[p]
                              - self.whiff_penalty * whiffs)
        return rewards

    def _track_attack(self, gs, port: int, hit: bool) -> int:
        """Follow the A attack `port` is doing; returns 1 when one ends without having hit anything."""
        action = getattr(gs.players[port].action, "value", -1)
        attack = self._attack[port]
        whiffed = 0
        if attack is not None and attack[0] != action:  # it ended, or turned into the next attack (jab 1 -> 2)
            whiffed = int(not attack[1])
            self._stats[port]["whiffs"] += whiffed
            attack = None
        if action in ATTACK_ACTIONS:
            if attack is None:
                attack = [action, False]
                self._stats[port]["attacks"] += 1
            attack[1] = attack[1] or hit
        self._attack[port] = attack
        return whiffed

    # ------------------------------------------------------------ observation
    def _obs(self, gs, port: int = 1) -> np.ndarray:
        """Observation from `port`'s point of view: its own features first."""
        return observation(gs, gs.players[port], gs.players[OTHER_PORT[port]], self.edge)

    def _info(self, gs, done: bool, port: int = 1) -> dict:
        me, op = gs.players[port], gs.players[OTHER_PORT[port]]
        info = dict(self._stats[port])
        info["cpu_level"] = self.cpu_level
        info["character"] = me.character.name
        info["opponent"] = op.character.name
        if done:
            info["won"] = float(op.stock < me.stock or (op.stock == me.stock and op.percent > me.percent))
            info["stock_diff"] = int(me.stock) - int(op.stock)
        return info

    def _both_obs(self, gs) -> np.ndarray:
        return np.stack([self._obs(gs, 1), self._obs(gs, 2)])

    def _both_info(self, gs, done: bool) -> dict:
        return {"p1": self._info(gs, done, 1), "p2": self._info(gs, done, 2)}

    def close(self):
        self._stop_dolphin()


def _exit_on_signal(signum, frame):
    raise SystemExit(1)


def _safe_worker(remote, parent_remote, env_fn_wrapper):
    """SB3 worker that leaves Ctrl+C / hangups to the main process and always closes Dolphin on exit."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, _exit_on_signal)
    try:
        _worker(remote, parent_remote, env_fn_wrapper)
    finally:
        for env in list(_LIVE_ENVS):
            env.close()


class NonDaemonSubprocVecEnv(SubprocVecEnv):
    """SubprocVecEnv whose workers are not daemons.

    libmelee starts a helper process inside each env, and daemon processes are
    not allowed to have children, so the stock SubprocVecEnv cannot be used.
    """

    def __init__(self, env_fns, start_method: str = "spawn"):
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)
        self.pid_dir = tempfile.mkdtemp(prefix="smash_dolphins_")
        os.environ[PID_DIR_ENV] = self.pid_dir  # inherited by the workers
        ctx = mp.get_context(start_method)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            process = ctx.Process(target=_safe_worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)), daemon=False)
            process.start()
            self.processes.append(process)
            work_remote.close()
        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()
        super(SubprocVecEnv, self).__init__(n_envs, observation_space, action_space)

    def close(self):
        """Ask every worker to close its Dolphin, then make sure each process is gone.

        SB3's close() waits forever for workers that died or are stuck; this never does.
        """
        if self.closed:
            return
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for process in self.processes:
            process.join(timeout=45)  # a worker blocked on a frozen Dolphin notices within hang_timeout
            if process.is_alive():
                process.terminate()  # SIGTERM: the worker still closes its Dolphin on the way out
                process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        for remote in self.remotes:
            remote.close()
        # Dolphins whose worker died without closing them (killed, out of memory, ...).
        for name in os.listdir(self.pid_dir):
            try:
                with open(os.path.join(self.pid_dir, name)) as f:
                    pid = int(f.read())
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    cmdline = f.read()
                if b"Slippi" in cmdline or b"AppRun" in cmdline:  # still that Dolphin, not a reused pid
                    os.kill(pid, signal.SIGKILL)
            except (OSError, ValueError):
                pass
        shutil.rmtree(self.pid_dir, ignore_errors=True)
        self.closed = True


class SmashMonitor(Monitor):
    """Monitor that forwards `set_cpu_level` so VecEnv.env_method can reach the env."""

    def set_cpu_level(self, level: int):
        self.env.unwrapped.set_cpu_level(level)


INFO_KEYWORDS = ("won", "stock_diff", "damage_dealt", "damage_taken", "kills", "deaths", "attacks", "whiffs",
                 "cpu_level")


def make_env(rank: int = 0, base_port: int = 51441, **env_kwargs):
    """Return a thunk that builds one monitored Smash env with its own Dolphin port."""

    def _init():
        env = SmashEnv(port=base_port + rank, **env_kwargs)
        return SmashMonitor(env, info_keywords=INFO_KEYWORDS)

    return _init


def make_vec_env(n_envs: int = 4, base_port: int = 51441, subprocess: bool = True, **env_kwargs):
    """Vectorized Smash envs (one Dolphin per env), ready for an SB3 MlpPolicy."""
    thunks = [make_env(i, base_port, **env_kwargs) for i in range(n_envs)]
    if subprocess and n_envs > 1:
        return NonDaemonSubprocVecEnv(thunks)
    return DummyVecEnv(thunks)
