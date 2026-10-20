"""Run matches between the Ganondorf agent and another player through libmelee.

Examples:
    # headless and fast (libmelee's ExiAI Dolphin), Ganondorf (P1) vs a level-9 CPU Fox
    python match.py --dolphin /path/to/ExiAI/AppRun --iso melee.iso --p2 cpu:FOX:9 --games 3
    # watch in a window (normal Slippi Dolphin), Ganondorf (P1) vs your agent (P2)
    python match.py --dolphin /path/to/Slippi_Online-x86_64.AppImage --iso melee.iso --p2 friend --watch
    # Ganondorf on port 2 instead
    python match.py --dolphin ... --iso ... --p1 friend --p2 ganondorf

Players: "ganondorf" (this agent), "friend" (FriendAgent in friend_agent.py), or "cpu:<CHARACTER>:<level>".
"""

import argparse
import importlib
import time

import melee

from ganondorf_agent import GanondorfAgent

IN_GAME = (melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dolphin", required=True, help="Slippi Dolphin executable / AppImage (ExiAI build for headless)")
    p.add_argument("--iso", required=True, help="Super Smash Bros. Melee NTSC v1.02 ISO")
    p.add_argument("--p1", default="ganondorf")
    p.add_argument("--p2", default="cpu:FOX:9")
    p.add_argument("--model", default="model/ganondorf_edge_v4.zip")
    p.add_argument("--stage", default="FINAL_DESTINATION", help="the agent was trained on Final Destination only")
    p.add_argument("--games", type=int, default=1)
    p.add_argument("--watch", action="store_true", help="real-time speed (default: as fast as possible)")
    p.add_argument("--slippi-port", type=int, default=51460)
    return p.parse_args()


class Player:
    """One port: an agent (act() every frame) or the in-game CPU."""

    def __init__(self, spec: str, port: int, other: int, model: str):
        self.port, self.cpu_level, self.agent = port, 0, None
        if spec == "ganondorf":
            self.character = melee.Character.GANONDORF
            self.agent = GanondorfAgent(model, port=port, opponent_port=other)
        elif spec == "friend":
            mod = importlib.import_module("friend_agent")
            self.agent = mod.FriendAgent(port=port, opponent_port=other)
            self.character = melee.Character[self.agent.character]
        elif spec.startswith("cpu:"):
            _, name, level = spec.split(":")
            self.character, self.cpu_level = melee.Character[name.upper()], int(level)
        else:
            raise SystemExit(f"unknown player '{spec}'")
        self.name = spec


def main():
    args = parse_args()
    stage = melee.Stage[args.stage]
    exi_ai = melee.console.get_dolphin_version(args.dolphin).build == melee.console.DolphinBuild.EXI_AI
    fast = not args.watch
    console = melee.Console(path=args.dolphin, slippi_port=args.slippi_port, blocking_input=True,
                            fullscreen=False, disable_audio=True, save_replays=False,
                            gfx_backend="Null" if exi_ai else "", use_exi_inputs=exi_ai,
                            enable_ffw=exi_ai and fast, polling_mode=True, polling_timeout=30.0)
    players = {1: Player(args.p1, 1, 2, args.model), 2: Player(args.p2, 2, 1, args.model)}
    pads = {port: melee.Controller(console=console, port=port, type=melee.ControllerType.STANDARD) for port in (1, 2)}
    console.run(iso_path=args.iso)
    for _ in range(120):
        if console.connect():
            break
        time.sleep(0.5)
    else:
        raise SystemExit("could not connect to Dolphin")
    for pad in pads.values():
        pad.connect()
    menus = {port: melee.MenuHelper() for port in (1, 2)}
    print(f"P1 {players[1].name} ({players[1].character.name}) vs P2 {players[2].name} ({players[2].character.name}) "
          f"on {stage.name}")

    score, games, in_game, last = {1: 0, 2: 0}, 0, False, None
    try:
        while games < args.games:
            gs = console.step()
            if gs is None:
                raise SystemExit("Dolphin stopped sending frames")
            if gs.menu_state in IN_GAME:
                if not in_game:
                    in_game = True
                    for pl in players.values():
                        if pl.agent:
                            pl.agent.reset()
                for port, pl in players.items():
                    if pl.agent:
                        pl.agent.act(gs, pads[port])
                    else:
                        pads[port].release_all()
                if 1 in gs.players and 2 in gs.players:
                    last = gs  # scored when the game ends and the menus come back
                continue
            if in_game:  # a game just ended
                in_game = False
                games += 1
                a, b = last.players[1], last.players[2]
                winner = 1 if (a.stock, -a.percent) > (b.stock, -b.percent) else 2
                score[winner] += 1
                print(f"game {games}: P{winner} wins | stocks {a.stock}-{b.stock} | score P1 {score[1]} - {score[2]} P2")
                if games >= args.games:
                    break
            # Menus: port 2 sets itself up on character select only; port 1 starts the game and picks the stage
            # once port 2 is ready (two helpers would fight over the shared stage cursor otherwise).
            p2 = gs.players.get(2)
            ready = gs.menu_state != melee.Menu.CHARACTER_SELECT or (
                p2 is not None and p2.character == players[2].character and p2.cpu_level == players[2].cpu_level
                and (p2.coin_down or players[2].cpu_level > 0))
            if gs.menu_state == melee.Menu.CHARACTER_SELECT:
                menus[2].menu_helper_simple(gs, pads[2], players[2].character, stage,
                                            cpu_level=players[2].cpu_level, autostart=False)
            else:
                pads[2].release_all()
            menus[1].menu_helper_simple(gs, pads[1], players[1].character, stage, cpu_level=players[1].cpu_level,
                                        autostart=ready)
    finally:
        console.stop()
    print(f"final: P1 {players[1].name} {score[1]} - {score[2]} P2 {players[2].name}")


if __name__ == "__main__":
    main()
