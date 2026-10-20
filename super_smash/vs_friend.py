"""Our models against another team's agent, using that team's own match.py (nothing learns).

The other agent (e.g. ../friend_ganondorf/ganondorf_agent with match.py, ganondorf_agent.py and its
model) builds its own observations, so it runs in its own match loop; our model joins it as the
FriendAgent from external/friend_agent.py. Each model plays --games games as port 1 and --games as
port 2, as Fox on Final Destination, headless; --parallel series (one Dolphin each) at a time.

Usage:
    python vs_friend.py --friend-dir ../../friend_ganondorf/ganondorf_agent --friend ganondorf \\
        MK5=runs/mk56910_c/MK5.zip MK9=runs/mk56910_c/MK9.zip --games 10
"""

import argparse
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from smash_rl.env import DEFAULT_ISO, HEADLESS_DOLPHIN, PROJECT_DIR

GAME_LINE = re.compile(r"game \d+: P(\d) wins \| stocks (\d+)-(\d+)")


def parse_args():
    p = argparse.ArgumentParser(description="Our models vs another team's agent through its match.py")
    p.add_argument("models", nargs="+", help="NAME=path.zip")
    p.add_argument("--friend-dir", required=True, help="folder with the other team's match.py")
    p.add_argument("--friend", default="ganondorf", help="the other agent's player name in its match.py")
    p.add_argument("--games", type=int, default=10, help="games per model on each port")
    p.add_argument("--parallel", type=int, default=6, help="series running at the same time")
    p.add_argument("--deterministic", action="store_true", help="our models take their most likely action")
    p.add_argument("--base-port", type=int, default=51800)
    p.add_argument("--timeout-min", type=float, default=8.0, help="minutes allowed per game before a series is killed")
    return p.parse_args()


def run_series(args, name, path, our_port, slippi_port):
    """One match.py run: `name` on our_port against the friend's agent for args.games games."""
    env = dict(os.environ, SMASH_MODEL=os.path.abspath(path), SMASH_DETERMINISTIC="1" if args.deterministic else "0",
               PYTHONPATH=os.pathsep.join([os.path.join(PROJECT_DIR, "external"), PROJECT_DIR]))
    p1, p2 = ("friend", args.friend) if our_port == 1 else (args.friend, "friend")
    cmd = [sys.executable, "match.py", "--dolphin", HEADLESS_DOLPHIN, "--iso", DEFAULT_ISO, "--p1", p1, "--p2", p2,
           "--games", str(args.games), "--slippi-port", str(slippi_port)]
    try:
        out = subprocess.run(cmd, cwd=args.friend_dir, env=env, capture_output=True, text=True,
                             timeout=args.timeout_min * 60 * args.games).stdout
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
    games = []
    for winner, s1, s2 in GAME_LINE.findall(out):
        mine, theirs = (int(s1), int(s2)) if our_port == 1 else (int(s2), int(s1))
        games.append({"won": int(winner) == our_port, "stock_diff": mine - theirs})
    return name, our_port, games


def main():
    args = parse_args()
    models = dict(spec.split("=", 1) for spec in args.models)
    jobs = [(name, path, port) for name, path in models.items() for port in (1, 2)]
    start = time.time()
    results = {name: [] for name in models}
    with ThreadPoolExecutor(args.parallel) as pool:
        futures = [pool.submit(run_series, args, name, path, port, args.base_port + i)
                   for i, (name, path, port) in enumerate(jobs)]
        for f in futures:
            name, port, games = f.result()
            results[name] += games
            wins = sum(g["won"] for g in games)
            print(f"{name} as P{port}: {wins}/{len(games)} won vs {args.friend} "
                  f"({(time.time() - start) / 60:.0f} min)", flush=True)
    print(f"\n{'model':<8} {'won':>9} {'win %':>7} {'stock diff':>11}   (vs {args.friend}, both ports)")
    for name, games in results.items():
        if games:
            wins = sum(g["won"] for g in games)
            diff = sum(g["stock_diff"] for g in games) / len(games)
            print(f"{name:<8} {wins:>4}/{len(games):<4} {wins / len(games):>7.0%} {diff:>+11.2f}")
        else:
            print(f"{name:<8} no finished games")


if __name__ == "__main__":
    main()
