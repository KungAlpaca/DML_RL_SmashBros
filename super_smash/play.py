"""Play against a trained agent with an Xbox controller or the keyboard (Linux).

Usage:
    python play.py runs/<run>/final_model.zip                      # Xbox controller plugged in by USB
    python play.py runs/<run>/final_model.zip --input keyboard
    python play.py runs/<run>/final_model.zip --smash-hold-ms 150  # direction + A held = smash (off by default)
    python play.py models/alejandro/MK5.zip                        # our best Fox
    python play.py models/all_models/random/MK6.zip --character RANDOM
    python play.py models/all_models/fox/MK3.zip --learn           # the agent learns from every game

The agent is port 1 and picks Fox; you are port 2. On the character select screen
pick your character and press Start; the agent then picks Final Destination, the
only stage it knows. Games run at normal speed with sound and can be paused. Close
the Dolphin window or press Ctrl+C here to stop.

The Xbox controller is read here in Python (see smash_rl/xbox_pad.py) so that the
stick never jumps (and, with --smash-hold-ms, A held with a direction smashes). The keyboard is mapped in
Dolphin: its mapping lives in dolphin/play_home and can be changed with Controllers >
Port 2 > Configure in the Dolphin window (kept until you pass --remap).

With --learn the agent samples its actions and runs one PPO update on each game it plays
against you (reward v2), saving to <name>_live.zip next to the model (or --save-to) after every game;
the original model is not touched. Load the _live file next time to keep improving it.
"""

import argparse
import configparser
import os

import melee
from stable_baselines3 import PPO

from smash_rl import xbox_pad
from smash_rl.env import DEFAULT_DOLPHIN, PROJECT_DIR, SmashEnv
from smash_rl.live_learning import LiveLearner, live_path
from smash_rl.reaction import ReactionDelay
from smash_rl.xbox_pad import XboxPad

PLAY_HOME = os.path.join(PROJECT_DIR, "dolphin", "play_home")
GCPAD_INI = os.path.join(PLAY_HOME, "Config", "GCPadNew.ini")

XBOX_LAYOUT = """\
  Xbox             GameCube
  left stick       control stick (pushing up never jumps)
  right stick      ground: hold = charged smash that way; air: C-stick (aerials)
  A                A (smash attacks: right stick, or Melee's own flick + A)
  B                B (special)
  X / Y            X / Y (jump)
  RB               Z (grab)
  LT / RT          L / R (shield, air dodge)
  Menu             Start
  D-pad            D-pad"""

KEYBOARD_LAYOUT = """\
  Keyboard         GameCube
  W A S D          control stick
  hold Space       half tilt (walk, tilt attacks)
  arrow keys       C-stick (smash attacks, aerials)
  J                A
  K                B
  L / I            X / Y (jump)
  O                Z (grab)
  {  (right of Ñ)  R (shield, air dodge)
  Enter            Start"""


class HumanEnv(SmashEnv):
    """SmashEnv whose port 2 is fed by an XboxPad; the pad sees every game state."""

    def __init__(self, pad: XboxPad, **kwargs):
        super().__init__(**kwargs)
        self.xbox = pad

    def _start_dolphin(self):
        super()._start_dolphin()
        self.xbox.attach(self.console)
        self.xbox.start()

    def _next_state(self):
        gs = super()._next_state()
        self.xbox.gamestate = gs
        return gs


def _read_gcpad_ini():
    cfg = configparser.ConfigParser()  # lower-cases keys like libmelee does; Dolphin ignores case
    cfg.read(GCPAD_INI)
    return cfg


def map_port_to_pipe(port: int = 2):
    """Map a port to libmelee's named pipe in PLAY_HOME; Dolphin has to see it when it starts."""
    os.makedirs(os.path.dirname(GCPAD_INI), exist_ok=True)
    cfg = _read_gcpad_ini()
    if cfg.remove_section(f"GCPad{port}"):
        with open(GCPAD_INI, "w") as f:
            cfg.write(f)
    console = melee.Console(path=DEFAULT_DOLPHIN, dolphin_home_path=PLAY_HOME, tmp_home_directory=False)
    console.setup_dolphin_controller(port)


def keyboard_controls() -> dict:
    """GCPadNew.ini mapping for the keyboard through Dolphin's XInput2 backend (X key names)."""
    return {
        "Device": "XInput2/0/Virtual core pointer",
        "Buttons/A": "`J`",
        "Buttons/B": "`K`",
        "Buttons/X": "`L`",
        "Buttons/Y": "`I`",
        "Buttons/Z": "`O`",
        "Buttons/Start": "`Return`",
        "Main Stick/Up": "`W`",
        "Main Stick/Down": "`S`",
        "Main Stick/Left": "`A`",
        "Main Stick/Right": "`D`",
        "Main Stick/Modifier": "`space`",
        "Main Stick/Modifier/Range": "50.0",
        "C-Stick/Up": "`Up`",
        "C-Stick/Down": "`Down`",
        "C-Stick/Left": "`Left`",
        "C-Stick/Right": "`Right`",
        "Triggers/R": "`braceleft`",  # the { key right of Ñ on a Latin American keyboard
        "Triggers/R-Analog": "`braceleft`",
    }


def write_keyboard_controls(remap: bool, port: int = 2) -> bool:
    """Map a port to the keyboard. Returns False if an existing keyboard mapping was kept."""
    controls = keyboard_controls()
    section = f"GCPad{port}"
    os.makedirs(os.path.dirname(GCPAD_INI), exist_ok=True)
    cfg = _read_gcpad_ini()
    if not remap and cfg.has_section(section) and cfg.get(section, "Device", fallback="") == controls["Device"]:
        return False
    cfg.remove_section(section)
    cfg.add_section(section)
    for key, value in controls.items():
        cfg.set(section, key, value)
    with open(GCPAD_INI, "w") as f:
        cfg.write(f)
    return True


def parse_args():
    p = argparse.ArgumentParser(description="Play Melee against a trained PPO agent")
    p.add_argument("model", help="path to a saved .zip model")
    p.add_argument("--input", choices=("xbox", "keyboard"), default="xbox", help="what you play with")
    p.add_argument("--remap", action="store_true", help="keyboard: overwrite your mapping with the default one")
    p.add_argument("--smash-hold-ms", type=int, default=0,
                   help="Xbox: >0 makes a direction + A held this long a smash (delays A; off by default)")
    p.add_argument("--no-charge-smash", action="store_true",
                   help="Xbox: the right stick is a plain C-stick (default: on the ground, holding it charges a smash)")
    p.add_argument("--character", default="FOX", help="the agent's character, must match training")
    p.add_argument("--frame-skip", type=int, default=3, help="must match training")
    p.add_argument("--reaction-ms", type=float, default=None,
                   help="agent reaction time in ms: it sees the game this long ago (default: the model's own, "
                        "200 for MK7 / MK8, 0 = instant for older models)")
    p.add_argument("--stochastic", action="store_true", help="sample actions instead of taking the most likely one")
    p.add_argument("--learn", action="store_true", help="fine-tune the agent on every game against you")
    p.add_argument("--learn-lr", type=float, default=5e-5, help="learning rate of those updates")
    p.add_argument("--save-to", default=None, help="where --learn saves the model (default: <model>_live.zip)")
    p.add_argument("--reward", choices=("v1", "v2"), default="v2", help="reward the agent learns from with --learn")
    p.add_argument("--port", type=int, default=51520)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(PLAY_HOME, exist_ok=True)
    env_kwargs = dict(character=args.character, opponent="HUMAN", frame_skip=args.frame_skip, port=args.port,
                      emulation_speed=1.0, dolphin_home=PLAY_HOME, reward=args.reward)
    pad = None
    if args.input == "xbox":
        name, device = xbox_pad.find_pad()
        pad = XboxPad(device, smash_hold=args.smash_hold_ms / 1000, charge_smash=not args.no_charge_smash)
        map_port_to_pipe(2)
        print(f"port 2: {name} ({device})\n{XBOX_LAYOUT}")
        env = HumanEnv(pad, **env_kwargs)
    else:
        written = write_keyboard_controls(args.remap)
        print(f"port 2: keyboard ({'default mapping' if written else 'your saved mapping'})")
        if written:
            print(KEYBOARD_LAYOUT)
        env = SmashEnv(**env_kwargs)

    model = PPO.load(args.model, device="cpu")
    if model.observation_space.shape != env.observation_space.shape:
        raise SystemExit(f"{args.model} expects observations of shape {model.observation_space.shape}, "
                         f"but the game gives {env.observation_space.shape}: it was trained with an older env.py")
    learner = None
    if args.learn:
        learner = LiveLearner(model, args.save_to or live_path(args.model), args.learn_lr)
        print(f"learning mode: the agent improves after every game and is saved to {learner.save_path}")

    delay = ReactionDelay(getattr(model, "reaction_ms", 0) if args.reaction_ms is None else args.reaction_ms,
                          args.frame_skip)
    print(f"agent reaction time: {delay.ms:.0f} ms")
    results = []
    try:
        while True:
            print("\npick your character on port 2 and press Start...")
            obs, info = env.reset()
            print(f"game {len(results) + 1}: you ({info['opponent']}) vs the agent ({args.character})")
            done = False
            delay.reset()
            while not done:
                seen = delay(obs)
                if learner is not None:
                    action = learner.act(seen)
                else:
                    action, _ = model.predict(seen, deterministic=not args.stochastic)
                obs, reward, done, _, info = env.step(action)
                if learner is not None:
                    learner.record(reward)
            results.append(info)
            print(f"  {'the agent' if info['won'] else 'you'} won | your stock difference {-info['stock_diff']:+d} | "
                  f"you dealt {info['damage_taken']:.0f}%, took {info['damage_dealt']:.0f}%")
            if learner is not None:
                print("  " + learner.end_game())
    except KeyboardInterrupt:
        pass
    except Exception as e:  # closing the Dolphin window ends the connection
        print(f"\nDolphin stopped ({type(e).__name__}: {e})")
    finally:
        env.close()
        if pad is not None:
            pad.close()
        if results:
            yours = sum(1 for r in results if not r["won"])
            print(f"\nyou won {yours} of {len(results)} games")


if __name__ == "__main__":
    main()
