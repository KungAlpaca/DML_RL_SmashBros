"""Round robin between saved models, plus each one against the Melee CPU. Nothing learns or is saved.

Every model plays every other one --games times (sides alternate) and the CPU --cpu-games times,
all as --character on Final Destination, each with its own saved reaction time (model.reaction_ms).
Games run in parallel: --pair-envs Dolphins for model vs model, --cpu-envs for model vs CPU; a
Dolphin moves on to the next match-up when its game ends, without restarting.

Usage:
    python tournament.py MK3=runs/mk56910_c/MK3.zip MK5=runs/mk56910_c/MK5.zip --games 10

Writes the per-game results to --out (CSV) and prints win-rate tables.
"""

import argparse
import csv
import time
from collections import defaultdict, deque
from itertools import combinations

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.utils import obs_as_tensor

from smash_rl.env import CAN_RUN_HEADLESS, NonDaemonSubprocVecEnv, SmashEnv
from smash_rl.reaction import ReactionDelay

STATS = ("won", "stock_diff", "damage_dealt", "damage_taken", "kills", "deaths", "attacks", "whiffs", "combos",
         "smash_spam")


def parse_args():
    p = argparse.ArgumentParser(description="Round robin between saved models (no training)")
    p.add_argument("models", nargs="+", help="NAME=path.zip")
    p.add_argument("--character", default="FOX", help="character every model plays")
    p.add_argument("--cpu-character", default="FOX", help="the CPU's character (or RANDOM)")
    p.add_argument("--cpu-level", type=int, default=9)
    p.add_argument("--games", type=int, default=10, help="games per pair of models")
    p.add_argument("--cpu-games", type=int, default=10, help="games per model against the CPU")
    p.add_argument("--pair-envs", type=int, default=6)
    p.add_argument("--cpu-envs", type=int, default=2)
    p.add_argument("--deterministic", action="store_true",
                   help="take the most likely action (default: sample, as in training; identical "
                        "deterministic models can replay the same game)")
    p.add_argument("--frame-skip", type=int, default=3)
    p.add_argument("--base-port", type=int, default=51700)
    p.add_argument("--out", default="tournament_results.csv")
    return p.parse_args()


def main():
    args = parse_args()
    models, delays_ms = {}, {}
    for spec in args.models:
        name, path = spec.split("=", 1)
        models[name] = PPO.load(path, device="cpu")
        delays_ms[name] = getattr(models[name], "reaction_ms", 0)
    print("models: " + ", ".join(f"{n} ({delays_ms[n]:g} ms)" for n in models), flush=True)

    pair_queue = deque()
    for a, b in combinations(models, 2):
        for g in range(args.games):
            pair_queue.append((a, b) if g % 2 == 0 else (b, a))
    cpu_queue = deque(name for name in models for _ in range(args.cpu_games))
    total = len(pair_queue) + len(cpu_queue)

    kinds = ["pair"] * args.pair_envs + ["cpu"] * args.cpu_envs
    headless = CAN_RUN_HEADLESS

    def make(i, kind):
        def _init():
            return SmashEnv(port=args.base_port + i, two_player=True, character=args.character,
                            opponent=args.cpu_character if kind == "cpu" else args.character,
                            cpu_opponent=kind == "cpu", cpu_level=args.cpu_level, reward="v2",
                            frame_skip=args.frame_skip, headless=headless)
        return _init

    def next_match(kind):
        queue = pair_queue if kind == "pair" else cpu_queue
        if not queue:
            return None
        item = queue.popleft()
        return item if kind == "pair" else (item, "CPU")

    env = NonDaemonSubprocVecEnv([make(i, kind) for i, kind in enumerate(kinds)])
    match = [next_match(kind) for kind in kinds]
    delay = [[ReactionDelay(delays_ms.get(who, 0), args.frame_skip) for who in m] if m else None for m in match]
    obs = env.reset()
    results, done_games, start = [], 0, time.time()
    try:
        while any(m is not None for m in match):
            actions = np.zeros((len(kinds), 2), dtype=np.int64)
            by_model = defaultdict(list)
            for i, m in enumerate(match):
                if m is None:
                    continue
                for side, who in enumerate(m):
                    if who != "CPU":
                        by_model[who].append((i, side, delay[i][side](obs[i, side])))
            with torch.no_grad():
                for name, items in by_model.items():
                    seen = obs_as_tensor(np.stack([x[2] for x in items]), "cpu")
                    action, _, _ = models[name].policy(seen, deterministic=args.deterministic)
                    for (i, side, _), a in zip(items, action.cpu().numpy()):
                        actions[i, side] = a
            obs, _, dones, infos = env.step(actions)
            for i, done in enumerate(dones):
                if not done or match[i] is None:
                    continue
                m = match[i]
                if infos[i]["p1"].get("crashed"):
                    (pair_queue if kinds[i] == "pair" else cpu_queue).append(m if kinds[i] == "pair" else m[0])
                else:
                    for side, who in enumerate(m):
                        if who == "CPU":
                            continue
                        mine = infos[i][("p1", "p2")[side]]
                        results.append({"model": who, "opponent": m[1 - side], "side": side + 1,
                                        **{k: mine.get(k, 0) for k in STATS}})
                    done_games += 1
                    if done_games % 20 == 0:
                        print(f"{done_games}/{total} games | {(time.time() - start) / 60:.0f} min", flush=True)
                match[i] = next_match(kinds[i])
                delay[i] = [ReactionDelay(delays_ms.get(w, 0), args.frame_skip) for w in match[i]] if match[i] else None
    finally:
        env.close()
        if results:
            with open(args.out, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(results[0]))
                writer.writeheader()
                writer.writerows(results)

    names = list(models)
    table = defaultdict(list)
    for r in results:
        table[(r["model"], r["opponent"])].append(r)
    width = max(len(n) for n in names) + 2
    print(f"\nwin rate of the row model against the column ({args.games} games per pair, "
          f"CPU = level {args.cpu_level} {args.cpu_character.lower()})")
    print(" " * width + "".join(f"{n:>7}" for n in names + ["CPU"]) + "   overall")
    for a in names:
        cells, everyone = [], []
        for b in names + ["CPU"]:
            games = table.get((a, b), [])
            everyone += games if b != "CPU" else []
            cells.append(f"{np.mean([g['won'] for g in games]):7.0%}" if games else "      -")
        overall = np.mean([g["won"] for g in everyone]) if everyone else 0
        print(f"{a:<{width}}" + "".join(cells) + f"   {overall:6.0%}")
    print("\nper model (all its games): stock diff, damage dealt / taken, whiff rate, combos, smash spam per game")
    for a in names:
        games = [r for r in results if r["model"] == a]
        if not games:
            continue
        mean = lambda k: np.mean([g[k] for g in games])
        whiff = sum(g["whiffs"] for g in games) / max(sum(g["attacks"] for g in games), 1)
        print(f"{a:<{width}} stock {mean('stock_diff'):+.2f} | dealt {mean('damage_dealt'):.0f}% taken "
              f"{mean('damage_taken'):.0f}% | whiffs {whiff:.0%} | combos {mean('combos'):.2f} | "
              f"smash spam {mean('smash_spam'):.1f}")
    print(f"\n{done_games} games in {(time.time() - start) / 60:.0f} min; per-game results in {args.out}")


if __name__ == "__main__":
    main()
