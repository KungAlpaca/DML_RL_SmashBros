"""Free-for-all on Final Destination between the three teams' bots, plus you or a CPU on port 4.

Each port belongs to one team:

    P1  our PPO models          models/alejandro/*.zip  (Fox; each with its saved reaction time)
    P2  Kepiti's Fox            models/kepiti/*.pt      (LSTM imitation + RL, 18-frame reaction delay;
                                                         needs his repo, --kepiti-repo)
    P3  the Ganondorf agent     models/ganondorf/*.zip  (SB3 PPO, no reaction delay)
    P4  you (--p4 human), a Melee CPU (--p4 cpu) or nobody (--p4 none)

Any of P1-P3 can be 'none'. In games with three or more players every bot targets the nearest
opponent still alive (all three were trained 1 vs 1). Turn a bot on/off on the character select
screen by typing its port (1-3) and Enter in this terminal. Press Start when everyone is ready
(--autostart: the bots press it). Close the Dolphin window or press Ctrl+C here to stop.

Usage:
    python party_teams.py --p4 human                       # you vs all three (Xbox pad)
    python party_teams.py --p4 cpu --cpu-level 9 --autostart
    python party_teams.py --p1 models/alejandro/MK9.zip --p3 none --p4 human --input keyboard
"""

import argparse
import configparser
import importlib
import os
import queue
import sys
import threading
import time

import melee
from stable_baselines3 import PPO

from play import GCPAD_INI, KEYBOARD_LAYOUT, PLAY_HOME, XBOX_LAYOUT, map_port_to_pipe, write_keyboard_controls
from smash_rl import xbox_pad
from smash_rl.external_bots import import_kepiti_foxbot
from smash_rl.env import DEFAULT_DOLPHIN, DEFAULT_ISO, IN_GAME, PROJECT_DIR, TAP_BUTTONS, SmashEnv, observation
from smash_rl.reaction import ReactionDelay
from smash_rl.xbox_pad import XboxPad

CSS = melee.Menu.CHARACTER_SELECT
FD = melee.Stage.FINAL_DESTINATION
EDGE = melee.stages.EDGE_GROUND_POSITION[FD]
FRAME_SKIP = 3
HUMAN_PORT = 4


def nearest_opponent(gs, port):
    """Port of the closest other player still in the game, or None."""
    me = gs.players.get(port)
    others = [(p, pl) for p, pl in gs.players.items() if p != port and pl.stock > 0]
    if me is None or me.stock == 0 or not others:
        return None
    return min(others, key=lambda item: (item[1].position.x - me.position.x) ** 2
               + (item[1].position.y - me.position.y) ** 2)[0]


class Slot:
    """One port driven from this script."""

    character = melee.Character.FOX
    cpu_level = 0

    def __init__(self, name, port, console):
        self.name, self.port = name, port
        self.pad = melee.Controller(console=console, port=port, type=melee.ControllerType.STANDARD)
        self.menu = melee.MenuHelper()
        self.enabled = True

    def new_menu_visit(self):
        self.menu = melee.MenuHelper()

    def on_select_screen(self, gs):
        if self.enabled:
            self.menu.choose_character(self.character, gs, self.pad, cpu_level=self.cpu_level)
        else:
            melee.MenuHelper.change_controller_status(self.pad, gs, self.port,
                                                      melee.ControllerStatus.CONTROLLER_UNPLUGGED)

    def ready(self, gs):
        p = gs.players.get(self.port)
        return p is not None and p.character == self.character and (p.coin_down or self.cpu_level > 0)

    def start_game(self, gs):
        pass

    def act(self, gs):
        pass


class OurBot(Slot):
    def __init__(self, port, console, path, deterministic):
        super().__init__(f"P{port} ours ({os.path.basename(path)})", port, console)
        self.model = PPO.load(path, device="cpu")
        self.deterministic = deterministic
        self.delay = ReactionDelay(getattr(self.model, "reaction_ms", 0), FRAME_SKIP)
        self.button = None

    def start_game(self, gs):
        self.delay.reset()

    def act(self, gs):
        target = nearest_opponent(gs, self.port)
        if target is None:
            return
        if gs.frame % FRAME_SKIP == 0:
            obs = self.delay(observation(gs, gs.players[self.port], gs.players[target], EDGE))
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
            self.button = SmashEnv._press(self.pad, int(action))
        elif gs.frame % FRAME_SKIP == FRAME_SKIP - 1 and self.button in TAP_BUTTONS:
            self.pad.release_button(self.button)


class ForeignBot(Slot):
    """A bot with the act(gs, controller) / reset() interface (Kepiti's FoxBot, the Ganondorf agent)."""

    def __init__(self, name, port, console, bot, character):
        super().__init__(name, port, console)
        self.bot, self.character = bot, character

    def start_game(self, gs):
        self.bot.reset()

    def act(self, gs):
        target = nearest_opponent(gs, self.port)
        if target is not None:
            self.bot.opponent_port = target
            self.bot.act(gs, self.pad)


class CpuSlot(Slot):
    def __init__(self, port, console, character, level):
        super().__init__(f"P{port} CPU", port, console)
        self.character, self.cpu_level = melee.Character[character.upper()], level

    def act(self, gs):
        self.pad.release_all()


def configure_dolphin(console, keyboard_port=None):
    """Bigger window, and the keyboard player's port plugged in as a standard controller."""
    path = os.path.join(console._get_dolphin_config_path(), "Dolphin.ini")
    cfg = configparser.ConfigParser()  # lower-cased keys like libmelee writes them
    cfg.read(path)
    cfg.set("Display", "RenderWindowWidth", "1152")
    cfg.set("Display", "RenderWindowHeight", "864")
    cfg.set("Display", "RenderWindowAutoSize", "False")
    if keyboard_port:
        cfg.set("Core", f"SIDevice{keyboard_port - 1}", melee.ControllerType.STANDARD.value)
    with open(path, "w") as f:
        cfg.write(f)


def _read_stdin(requests):
    for line in sys.stdin:
        requests.put(line.strip())


def parse_args():
    models = os.path.join(PROJECT_DIR, "models")
    p = argparse.ArgumentParser(description="FFA between our bot (P1), Kepiti's (P2), Ganondorf (P3) and you / a CPU (P4)")
    p.add_argument("--p1", default=os.path.join(models, "alejandro", "MK5.zip"), help="our .zip model, or 'none'")
    p.add_argument("--p2", default=os.path.join(models, "kepiti", "fox_rl_3h.pt"), help="Kepiti's .pt model, or 'none'")
    p.add_argument("--p3", default=os.path.join(models, "ganondorf", "ganondorf_edge_v4.zip"),
                   help="Ganondorf .zip (meta.json and ganondorf_agent.py next to it), or 'none'")
    p.add_argument("--p4", choices=("human", "cpu", "none"), default="human")
    p.add_argument("--input", choices=("xbox", "keyboard"), default="xbox", help="--p4 human: what you play with")
    p.add_argument("--remap", action="store_true", help="keyboard: overwrite your mapping with the default one")
    p.add_argument("--no-charge-smash", action="store_true",
                   help="Xbox: the right stick is a plain C-stick (default: on the ground, holding it charges a smash)")
    p.add_argument("--cpu-character", default="FOX", help="--p4 cpu: its character")
    p.add_argument("--cpu-level", type=int, default=9, help="--p4 cpu: its level")
    p.add_argument("--kepiti-repo", default=os.path.expanduser("~/Documents/DML/super_smash_RL"),
                   help="Kepiti's repo (his smash_rl package runs his model)")
    p.add_argument("--kepiti-delay", type=int, default=18, help="Kepiti's reaction delay in frames (his training: 18)")
    p.add_argument("--stochastic", action="store_true", help="our bot and Ganondorf sample actions (default: most likely)")
    p.add_argument("--autostart", action="store_true", help="the bots press Start themselves (to watch them play)")
    p.add_argument("--port", type=int, default=51540)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(PLAY_HOME, exist_ok=True)
    pad = None
    if args.p4 == "human":
        if args.input == "xbox":
            name, device = xbox_pad.find_pad()
            pad = XboxPad(device, port=HUMAN_PORT, charge_smash=not args.no_charge_smash)
            map_port_to_pipe(HUMAN_PORT)
            print(f"you are port {HUMAN_PORT}: {name}\n{XBOX_LAYOUT}")
        else:
            write_keyboard_controls(args.remap, port=HUMAN_PORT)
            print(f"you are port {HUMAN_PORT}: keyboard (mapping in {GCPAD_INI})\n{KEYBOARD_LAYOUT}")

    console = melee.Console(path=DEFAULT_DOLPHIN, dolphin_home_path=PLAY_HOME, tmp_home_directory=False,
                            slippi_port=args.port, blocking_input=False, fullscreen=False, disable_audio=False,
                            save_replays=False, gfx_backend="OGL", log_types=[])
    configure_dolphin(console, keyboard_port=HUMAN_PORT if args.p4 == "human" and args.input == "keyboard" else None)

    slots = []
    if args.p1.lower() != "none":
        slots.append(OurBot(1, console, args.p1, not args.stochastic))
    if args.p2.lower() != "none":
        FoxBot = import_kepiti_foxbot(args.kepiti_repo)
        bot = FoxBot(args.p2, port=2, opponent_port=1, delay=args.kepiti_delay, device="cpu")
        slots.append(ForeignBot(f"P2 Kepiti ({os.path.basename(args.p2)})", 2, console, bot, melee.Character.FOX))
    if args.p3.lower() != "none":
        sys.path.insert(0, os.path.dirname(os.path.abspath(args.p3)))
        GanondorfAgent = importlib.import_module("ganondorf_agent").GanondorfAgent
        bot = GanondorfAgent(args.p3, port=3, opponent_port=1, deterministic=not args.stochastic)
        slots.append(ForeignBot(f"P3 Ganondorf ({os.path.basename(args.p3)})", 3, console, bot,
                                melee.Character.GANONDORF))
    if args.p4 == "cpu":
        slots.append(CpuSlot(4, console, args.cpu_character, args.cpu_level))
    bots = [s for s in slots if not isinstance(s, CpuSlot)]
    if not bots:
        raise SystemExit("at least one of P1-P3 is needed (it also drives the menus)")
    lead = bots[0]
    for s in slots:
        extra = f", reaction {s.delay.ms:.0f} ms" if isinstance(s, OurBot) else ""
        print(f"{s.name}: {s.character.name}{extra}" + (f" level {s.cpu_level}" if s.cpu_level else ""))
    print(f"P4: {args.p4}")

    console.run(iso_path=DEFAULT_ISO)
    for _ in range(120):
        if console.connect():
            break
        time.sleep(0.5)
    else:
        console.stop()
        raise SystemExit("could not connect to Dolphin")
    for s in slots:
        s.pad.connect()
    if pad is not None:
        pad.attach(console)
        pad.start()

    requests = queue.Queue()
    threading.Thread(target=_read_stdin, args=(requests,), daemon=True).start()
    print("\nturn bots on/off on the character select screen: type 1, 2 or 3 + Enter here")
    last_menu, last_game, games = None, None, 0
    try:
        while True:
            gs = console.step()
            if gs is None:
                continue
            if pad is not None:
                pad.gamestate = gs
            while not requests.empty():
                request = requests.get()
                for s in bots:
                    if request == str(s.port):
                        s.enabled = not s.enabled
                        print(f"{s.name} {'on' if s.enabled else 'off'}"
                              + ("" if gs.menu_state == CSS else " from the next character select"))

            if gs.menu_state != last_menu:
                if gs.menu_state == CSS:
                    for s in slots:
                        s.new_menu_visit()
                    lead.menu.stage_selected = False
                if last_menu in IN_GAME and last_game is not None:
                    standing = sorted(last_game.players.items(), key=lambda item: (-item[1].stock, item[1].percent))
                    print(f"game {games} over: " + ", ".join(f"P{port} {p.character.name} {p.stock} stocks"
                                                           for port, p in standing))
                last_menu = gs.menu_state

            if gs.menu_state in IN_GAME:
                if last_game is None or gs.frame < last_game.frame:
                    games += 1
                    print(f"game {games}: " + ", ".join(f"P{port} {p.character.name}"
                                                        for port, p in sorted(gs.players.items())))
                    for s in slots:
                        s.start_game(gs)
                last_game = gs
                for s in slots:
                    if s.port in gs.players:
                        s.act(gs)
                continue
            last_game = None if gs.menu_state == CSS else last_game
            for s in slots:
                s.pad.release_all()
            if gs.menu_state == CSS:
                for s in slots:
                    s.on_select_screen(gs)
                ready = all(s.ready(gs) for s in slots if s.enabled)
                if args.autostart and ready and gs.ready_to_start == 0 and gs.frame % 2 == 0:
                    lead.pad.press_button(melee.Button.BUTTON_START)
            else:  # main menus, stage select (the first bot picks Final Destination), results screen
                lead.menu.menu_helper_simple(gs, lead.pad, lead.character, FD, cpu_level=0, autostart=True)
    except KeyboardInterrupt:
        pass
    finally:
        console.stop()
        if pad is not None:
            pad.close()


if __name__ == "__main__":
    main()
