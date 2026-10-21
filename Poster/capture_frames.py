"""Screenshots for the poster: bots playing in a windowed Dolphin, captured when someone takes a big hit.

Run from super_smash/ with its .venv (the ports take any mix of our models, Kepiti's and the CPU):

    python ../Poster/capture_frames.py --out ../Poster/frames/ffa --seconds 150 \
        --p1 ours:models/alejandro/MK5.zip --p2 ours:models/alejandro/MK9.zip \
        --p3 kepiti:models/kepiti/fox_imitation_18h.pt --p4 kepiti:models/kepiti/fox_rl_3h.pt
    python ../Poster/capture_frames.py --out ../Poster/frames/mk5 --p1 ours:models/alejandro/MK5.zip --p2 cpu:FOX:9

The game renders at 3x the native resolution in a 1280x960 window with no on-screen messages and waits for
the bots every frame (blocking input), so no decision is skipped. Every capture is a PNG of the Dolphin
window plus a line in captures.jsonl (game frame, why, each player's damage, stocks and action).
"""

import argparse
import configparser
import json
import os
import subprocess
import sys
import time

SMASH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "super_smash")
sys.path.insert(0, os.path.abspath(SMASH))

import melee  # noqa: E402

from party_teams import FD, CpuSlot, ForeignBot, OurBot  # noqa: E402
from smash_rl.env import DEFAULT_DOLPHIN, DEFAULT_ISO, IN_GAME  # noqa: E402
from smash_rl.external_bots import bot_character, make_bot  # noqa: E402

CSS = melee.Menu.CHARACTER_SELECT


def parse_args():
    p = argparse.ArgumentParser(description="Poster screenshots of bots playing, taken on big hits")
    for port in (1, 2, 3, 4):
        p.add_argument(f"--p{port}", default=None,
                       help="ours:<.zip>, kepiti:<.pt>, ganondorf:<.zip> or cpu:<CHARACTER>:<level>")
    p.add_argument("--out", required=True, help="folder for the PNGs and captures.jsonl")
    p.add_argument("--seconds", type=float, default=90, help="wall-clock seconds of play to capture")
    p.add_argument("--min-hit", type=float, default=7.0, help="damage in one frame that triggers a capture")
    p.add_argument("--every", type=float, default=4.0, help="also capture every N seconds of play")
    p.add_argument("--gap", type=float, default=0.35, help="minimum seconds between captures")
    p.add_argument("--scale", type=int, default=3, help="internal resolution (x native)")
    p.add_argument("--port", type=int, default=51580)
    return p.parse_args()


def configure(console, scale):
    """1280x960 window, no on-screen text, scale x native resolution."""
    config = console._get_dolphin_config_path()
    os.makedirs(config, exist_ok=True)
    path = os.path.join(config, "Dolphin.ini")
    cfg = configparser.ConfigParser()  # lower-cased keys like libmelee writes them
    cfg.read(path)
    for section in ("Display", "Interface"):
        if not cfg.has_section(section):
            cfg.add_section(section)
    cfg.set("Display", "RenderWindowWidth", "1280")
    cfg.set("Display", "RenderWindowHeight", "960")
    cfg.set("Display", "RenderWindowAutoSize", "False")
    cfg.set("Interface", "OnScreenDisplayMessages", "False")
    with open(path, "w") as f:
        cfg.write(f)
    path = os.path.join(config, "GFX.ini")
    cfg = configparser.ConfigParser()
    cfg.read(path)
    for section in ("Settings", "Enhancements"):
        if not cfg.has_section(section):
            cfg.add_section(section)
    cfg.set("Settings", "InternalResolution", str(scale))
    cfg.set("Settings", "ShowFPS", "False")
    cfg.set("Settings", "MSAA", "4")
    cfg.set("Enhancements", "MaxAnisotropy", "4")
    with open(path, "w") as f:
        cfg.write(f)


def find_window():
    """X window id of Dolphin's game view: the biggest window titled "Dolphin" (the render window; the main
    window, "Faster Melee - Slippi", is the small game list). Prints the candidates."""
    tree = subprocess.run(["xwininfo", "-root", "-tree"], capture_output=True, text=True).stdout
    found = []
    for line in tree.splitlines():
        line = line.strip()
        if not line.startswith("0x") or not any(k in line for k in ("Melee", "GALE01", "Dolphin")):
            continue
        try:
            w, h = (int(v) for v in line.split(")")[-1].split()[0].split("+")[0].split("x"))
        except (ValueError, IndexError):
            continue
        found.append(('"Dolphin"' in line or "GALE01" in line, w * h, line.split()[0], line))
    for item in found:
        print("  window:", item[3][:150], flush=True)
    found.sort(reverse=True)
    return found[0][2] if found else None


def make_slot(spec, port, console):
    kind, _, rest = spec.partition(":")
    if kind == "ours":
        slot = OurBot(port, console, rest, deterministic=True)
    elif kind in ("kepiti", "ganondorf"):
        bot = make_bot((kind, rest), port=port, opponent_port=1 if port != 1 else 2)
        slot = ForeignBot(f"P{port} {kind} ({os.path.basename(rest)})", port, console, bot,
                          melee.Character[bot_character((kind, rest))])
    elif kind == "cpu":
        character, level = rest.split(":")
        slot = CpuSlot(port, console, character, int(level))
    else:
        raise SystemExit(f"unknown player {spec!r}")
    return slot


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    # A fresh Dolphin folder every run: one left over from a run with more bots keeps those ports as
    # pipes, and with blocking input Dolphin then waits for them forever.
    console = melee.Console(path=DEFAULT_DOLPHIN, tmp_home_directory=True,
                            slippi_port=args.port, blocking_input=True, fullscreen=False, disable_audio=True,
                            save_replays=False, gfx_backend="OGL", log_types=[])
    configure(console, args.scale)
    slots = [make_slot(spec, port, console) for port, spec in
             ((1, args.p1), (2, args.p2), (3, args.p3), (4, args.p4)) if spec]
    bots = [s for s in slots if not isinstance(s, CpuSlot)]
    lead = bots[0]
    print("players: " + ", ".join(f"{s.name} ({s.character.name})" for s in slots), flush=True)
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

    log = open(os.path.join(args.out, "captures.jsonl"), "a")
    window, last_menu, last_gs, playing_since, n = None, None, None, None, 0
    last_capture = last_periodic = 0.0
    try:
        while True:
            gs = console.step()
            if gs is None:
                continue
            if gs.menu_state != last_menu:
                if gs.menu_state == CSS:
                    for s in slots:
                        s.new_menu_visit()
                    lead.menu.stage_selected = False
                last_menu = gs.menu_state
            if gs.menu_state in IN_GAME:
                if last_gs is None or gs.frame < last_gs.frame:
                    for s in slots:
                        s.start_game(gs)
                    playing_since = playing_since or time.time()
                    window = window or find_window()
                    print(f"game on, window {window}", flush=True)
                now = time.time()
                why = None
                if last_gs is not None and gs.frame > 0:
                    for port, p in gs.players.items():
                        before = last_gs.players.get(port)
                        if before is not None and p.stock == before.stock and p.percent - before.percent >= args.min_hit:
                            why = f"P{port} hit for {p.percent - before.percent:.0f}%"
                        elif before is not None and p.stock < before.stock:
                            why = f"P{port} lost a stock"
                if why is None and now - last_periodic >= args.every and gs.frame > 60:
                    why, last_periodic = "periodic", now
                if why and window and now - last_capture >= args.gap:
                    path = os.path.join(args.out, f"{n:04d}_f{gs.frame:05d}.png")
                    subprocess.Popen(["import", "-silent", "-window", window, path])
                    log.write(json.dumps({"file": os.path.basename(path), "frame": int(gs.frame), "why": why, "players": {
                        int(port): {"character": p.character.name, "percent": float(p.percent), "stock": int(p.stock),
                                    "action": p.action.name} for port, p in gs.players.items()}}) + "\n")
                    log.flush()
                    n, last_capture = n + 1, now
                last_gs = gs
                for s in slots:
                    if s.port in gs.players:
                        s.act(gs)
                if now - playing_since >= args.seconds:
                    break
                continue
            last_gs = None if gs.menu_state == CSS else last_gs
            for s in slots:
                s.pad.release_all()
            if gs.menu_state == CSS:
                for s in slots:
                    s.on_select_screen(gs)
                ready = all(s.ready(gs) for s in slots)
                if ready and gs.ready_to_start == 0 and gs.frame % 2 == 0:
                    lead.pad.press_button(melee.Button.BUTTON_START)
            else:
                lead.menu.menu_helper_simple(gs, lead.pad, lead.character, FD, cpu_level=0, autostart=True)
    except KeyboardInterrupt:
        pass
    finally:
        time.sleep(1.0)  # let the last captures finish
        console.stop()
        print(f"{n} captures in {args.out}", flush=True)


if __name__ == "__main__":
    main()
