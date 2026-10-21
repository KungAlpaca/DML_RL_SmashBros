"""Free-for-all on Final Destination: you against up to two trained agents, plus a Melee CPU if you want.

Usage:
    python party.py                                        # MK5 (Fox) + MK6 (random) vs you, Xbox controller
    python party.py --input keyboard
    python party.py --p1 models/alejandro/MK9.zip --p2 models/all_models/fox/MK3.zip --p2-character FOX
    python party.py --p2 none                              # only one agent
    python party.py --p1 models/all_models/fox/MK3.zip --p2 models/all_models/random/MK4.zip --learn   # both learn from every game

For the other teams' bots (Kepiti's Fox, the Ganondorf agent) use party_teams.py.

Ports: 1 = first agent (Fox), 2 = second agent (--p2-character; RANDOM gives it a new
character every game), 3 = you, 4 = free. To add a Melee CPU, click the button above
slot 4 on the character select screen until it says CPU, then pick its character and
level with your cursor. Press Start when everyone is ready; the agents pick Final
Destination, the only stage they know.

Turning the agents on and off, on the character select screen (pressed during a game,
it applies from the next one): hold View (the two-squares button) and press LB for
agent 1 or RB for agent 2, or type 1 or 2 and Enter in this terminal. An agent that is
off leaves its slot (N/A).

In a game with more than two players each agent targets the nearest opponent.

With --learn each agent samples its actions and runs one PPO update after every game,
saving to <name>_live.zip next to the model. Its reward (v2) comes from its fight with you: your
knockouts, the damage you take and its combos and whiffs on you. With more players, damage
you take from someone else still counts for it.
Close the Dolphin window or press Ctrl+C here to stop.
"""

import argparse
import configparser
import os
import queue
import random
import sys
import threading
import time
from types import SimpleNamespace

import melee
from stable_baselines3 import PPO

from play import GCPAD_INI, PLAY_HOME, map_port_to_pipe, write_keyboard_controls, KEYBOARD_LAYOUT, XBOX_LAYOUT
from smash_rl import xbox_pad
from smash_rl.env import (ACTIONS, DEFAULT_DOLPHIN, DEFAULT_ISO, IN_GAME, RANDOM_OPPONENTS, TAP_BUTTONS, SmashEnv,
                           model_frame_skip, observation)
from smash_rl.live_learning import LiveLearner, live_path
from smash_rl.reaction import ReactionDelay
from smash_rl.rewards import RewardTracker, RewardWeights
from smash_rl.xbox_pad import BTN_SELECT, BTN_TL, BTN_TR, XboxPad

HUMAN_PORT = 3
CSS = melee.Menu.CHARACTER_SELECT


class Bot:
    """A trained model playing one port."""

    def __init__(self, name, path, port, character, console, deterministic, reaction_ms, frame_skip, learn_lr=None):
        self.name, self.port, self.deterministic = name, port, deterministic
        self.model = PPO.load(path, device="cpu")
        self.frame_skip = frame_skip or model_frame_skip(self.model)
        if reaction_ms is None:
            reaction_ms = getattr(self.model, "reaction_ms", 0)
        self.delay = ReactionDelay(reaction_ms, self.frame_skip)
        self.learner = LiveLearner(self.model, live_path(path), learn_lr) if learn_lr else None
        self.tracker = None  # reward v2 against the human, while learning
        self.pending_reward = 0.0
        self.decided = False
        self.random_character = character.upper() == "RANDOM"
        self.character = melee.Character.FOX if self.random_character else melee.Character[character.upper()]
        self.pad = melee.Controller(console=console, port=port, type=melee.ControllerType.STANDARD)
        self.menu = melee.MenuHelper()
        self.enabled = True
        self.button = None

    def new_menu_visit(self):
        self.menu = melee.MenuHelper()
        if self.random_character:
            self.character = random.choice(RANDOM_OPPONENTS)

    def on_select_screen(self, gs):
        if self.enabled:
            self.menu.choose_character(self.character, gs, self.pad, cpu_level=0)
        else:
            melee.MenuHelper.change_controller_status(self.pad, gs, self.port,
                                                      melee.ControllerStatus.CONTROLLER_UNPLUGGED)

    def start_game(self, gs):
        """With --learn, follow this agent's fight with the human for the reward."""
        self.tracker, self.pending_reward, self.decided = None, 0.0, False
        self.delay.reset()
        if self.learner is not None and self.port in gs.players and HUMAN_PORT in gs.players:
            self.tracker = RewardTracker(self._duel(gs), RewardWeights())

    def _duel(self, gs):
        return SimpleNamespace(frame=gs.frame, players={1: gs.players[self.port], 2: gs.players[HUMAN_PORT]})

    def end_game(self):
        if self.tracker is not None and self.decided:
            self.learner.record(self.pending_reward)
            print(f"  {self.name}: {self.learner.end_game()}")
        self.tracker = None

    def act(self, gs, edge):
        """Pick an action every frame_skip frames, against the nearest opponent still in the game."""
        frame_skip = self.frame_skip
        me = gs.players.get(self.port)
        if self.tracker is not None and me is not None and HUMAN_PORT in gs.players:
            self.pending_reward += float(self.tracker.frame(self._duel(gs))[0])
        others = [p for port, p in gs.players.items() if port != self.port and p.stock > 0]
        if me is None or me.stock == 0 or not others:
            return
        if gs.frame % frame_skip == 0:
            target = min(others, key=lambda p: (p.position.x - me.position.x) ** 2 + (p.position.y - me.position.y) ** 2)
            obs = self.delay(observation(gs, me, target, edge))
            if self.tracker is not None:
                if self.decided:
                    self.learner.record(self.pending_reward)
                self.pending_reward, self.decided = 0.0, True
                action = self.learner.act(obs)
                self.tracker.set_input(1, *ACTIONS[action])
            else:
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
            self.button = SmashEnv._press(self.pad, action)
        elif gs.frame % frame_skip == frame_skip - 1 and self.button in TAP_BUTTONS:
            self.pad.release_button(self.button)


def _read_stdin(requests):
    for line in sys.stdin:
        requests.put(line.strip())


def configure_dolphin(console, keyboard: bool):
    """Bigger window, and the keyboard player's port plugged in as a standard controller."""
    path = os.path.join(console._get_dolphin_config_path(), "Dolphin.ini")
    cfg = configparser.ConfigParser()  # lower-cased keys like libmelee writes them
    cfg.read(path)
    cfg.set("Display", "RenderWindowWidth", "1152")
    cfg.set("Display", "RenderWindowHeight", "864")
    cfg.set("Display", "RenderWindowAutoSize", "False")
    if keyboard:
        cfg.set("Core", f"SIDevice{HUMAN_PORT - 1}", melee.ControllerType.STANDARD.value)
    with open(path, "w") as f:
        cfg.write(f)


def parse_args():
    p = argparse.ArgumentParser(description="You vs up to two trained Melee agents (and a CPU)")
    p.add_argument("--p1", default="models/alejandro/MK5.zip", help="agent on port 1, or 'none'")
    p.add_argument("--p2", default="models/all_models/random/MK6.zip", help="agent on port 2, or 'none'")
    p.add_argument("--p1-character", default="FOX")
    p.add_argument("--p2-character", default="RANDOM", help="a character, or RANDOM for a new one every game")
    p.add_argument("--input", choices=("xbox", "keyboard"), default="xbox", help="what you play with (port 3)")
    p.add_argument("--remap", action="store_true", help="keyboard: overwrite your mapping with the default one")
    p.add_argument("--smash-hold-ms", type=int, default=0, help="Xbox: >0 makes a direction + A held this long a smash (delays A; off by default)")
    p.add_argument("--no-charge-smash", action="store_true",
                   help="Xbox: the right stick is a plain C-stick (default: on the ground, holding it charges a smash)")
    p.add_argument("--frame-skip", type=int, default=None,
                   help="frames each action is held (default: each model's own, as trained; 3 for older models)")
    p.add_argument("--reaction-ms", type=float, default=None,
                   help="agent reaction time in ms: it sees the game this long ago (default: the model's own, "
                        "200 for MK7 / MK8, 0 = instant for older models)")
    p.add_argument("--stochastic", action="store_true", help="agents sample actions instead of taking the most likely one")
    p.add_argument("--autostart", action="store_true", help="the agents press Start themselves (to watch them play)")
    p.add_argument("--learn", action="store_true", help="the agents fine-tune on every game (see above)")
    p.add_argument("--learn-lr", type=float, default=5e-5, help="learning rate of those updates")
    p.add_argument("--port", type=int, default=51530)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(PLAY_HOME, exist_ok=True)
    pad = None
    if args.input == "xbox":
        name, device = xbox_pad.find_pad()
        pad = XboxPad(device, smash_hold=args.smash_hold_ms / 1000, port=HUMAN_PORT,
                      charge_smash=not args.no_charge_smash)
        map_port_to_pipe(HUMAN_PORT)
        print(f"you are port {HUMAN_PORT}: {name}\n{XBOX_LAYOUT}")
    else:
        write_keyboard_controls(args.remap, port=HUMAN_PORT)
        print(f"you are port {HUMAN_PORT}: keyboard (mapping in {GCPAD_INI})\n{KEYBOARD_LAYOUT}")

    console = melee.Console(path=DEFAULT_DOLPHIN, dolphin_home_path=PLAY_HOME, tmp_home_directory=False,
                            slippi_port=args.port, blocking_input=False, fullscreen=False, disable_audio=False,
                            save_replays=False, gfx_backend="OGL", log_types=[])
    configure_dolphin(console, keyboard=args.input == "keyboard")
    bots = [Bot(name, path, port, character, console, not args.stochastic, args.reaction_ms, args.frame_skip,
                args.learn_lr if args.learn else None)
            for name, path, port, character in (("agent 1", args.p1, 1, args.p1_character),
                                                ("agent 2", args.p2, 2, args.p2_character))
            if path.lower() != "none"]
    if not bots:
        raise SystemExit("at least one agent is needed (it also drives the menus)")
    lead = bots[0]
    edge = melee.stages.EDGE_GROUND_POSITION[melee.Stage.FINAL_DESTINATION]
    for bot, path in zip(bots, [p for p in (args.p1, args.p2) if p.lower() != "none"]):
        print(f"{bot.name}: port {bot.port}, {path}, reaction {bot.delay.ms:.0f} ms, every {bot.frame_skip} frames, "
              f"{'a random character every game' if bot.random_character else bot.character.name}")

    console.run(iso_path=DEFAULT_ISO)
    for _ in range(120):
        if console.connect():
            break
        time.sleep(0.5)
    else:
        console.stop()
        raise SystemExit("could not connect to Dolphin")
    for bot in bots:
        bot.pad.connect()
    if pad is not None:
        pad.attach(console)
        pad.start()

    requests = queue.Queue()
    threading.Thread(target=_read_stdin, args=(requests,), daemon=True).start()
    print("\nturn agents on/off: View + LB / RB on the controller, or type 1 / 2 + Enter here")
    combo_down = {BTN_TL: False, BTN_TR: False}
    last_menu, last_game, games = None, None, 0
    try:
        while True:
            gs = console.step()
            if gs is None:
                continue
            if pad is not None:
                pad.gamestate = gs
                for button, port in ((BTN_TL, 1), (BTN_TR, 2)):
                    down = BTN_SELECT in pad.keys and button in pad.keys
                    if down and not combo_down[button]:
                        requests.put(str(port))
                    combo_down[button] = down
            while not requests.empty():
                request = requests.get()
                for bot in bots:
                    if request == str(bot.port):
                        bot.enabled = not bot.enabled
                        print(f"{bot.name} (port {bot.port}) {'on' if bot.enabled else 'off'}"
                              + ("" if gs.menu_state == CSS else " from the next character select"))

            if gs.menu_state != last_menu:
                if gs.menu_state == CSS:
                    for bot in bots:
                        bot.new_menu_visit()
                    lead.menu.stage_selected = False
                if last_menu in IN_GAME and last_game is not None:
                    standing = sorted(last_game.players.items(), key=lambda item: -item[1].stock)
                    print(f"game {games} over: " + ", ".join(f"port {port} {p.character.name} {p.stock} stocks"
                                                           for port, p in standing))
                    for bot in bots:
                        bot.end_game()
                last_menu = gs.menu_state

            if gs.menu_state in IN_GAME:
                if last_game is None or gs.frame < last_game.frame:
                    games += 1
                    print(f"game {games}: " + ", ".join(f"port {port} {p.character.name}"
                                                        for port, p in sorted(gs.players.items())))
                    for bot in bots:
                        bot.start_game(gs)
                last_game = gs
                for bot in bots:
                    bot.act(gs, edge)
                continue
            last_game = None if gs.menu_state == CSS else last_game
            for bot in bots:
                bot.pad.release_all()
            if gs.menu_state == CSS:
                for bot in bots:
                    bot.on_select_screen(gs)
                ready = all(gs.players.get(b.port) is not None and gs.players[b.port].character == b.character
                            and gs.players[b.port].coin_down for b in bots if b.enabled)
                if args.autostart and ready and gs.ready_to_start == 0 and gs.frame % 2 == 0:
                    lead.pad.press_button(melee.Button.BUTTON_START)
            else:  # main menus, stage select (the first agent picks Final Destination), results screen
                lead.menu.menu_helper_simple(gs, lead.pad, lead.character, melee.Stage.FINAL_DESTINATION,
                                             cpu_level=0, autostart=True)
    except KeyboardInterrupt:
        pass
    finally:
        console.stop()
        if pad is not None:
            pad.close()


if __name__ == "__main__":
    main()
