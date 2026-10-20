"""Play two trained models against each other (no training) and report who wins.

Usage:
    python versus.py models/all_models/fox/MK5.zip models/all_models/random/MK6.zip                     # watch 3 games, MK6 a random character
    python versus.py models/all_models/fox/MK5.zip models/all_models/fox/MK3.zip --p2-character FOX --games 20 --headless
    python versus.py models/all_models/fox/M1.zip models/all_models/fox/MK1.zip --p2-character FOX --headless

Models with a reaction time (MK7-MK10) play without it here; tournament.py and play.py use it.

The first model is port 1, the second port 2.
"""

import argparse

import numpy as np
from stable_baselines3 import PPO

from smash_rl.env import SmashEnv


def parse_args():
    p = argparse.ArgumentParser(description="Play two Melee PPO models against each other")
    p.add_argument("p1_model", help="port 1 model (.zip)")
    p.add_argument("p2_model", help="port 2 model (.zip)")
    p.add_argument("--p1-character", default="FOX")
    p.add_argument("--p2-character", default="RANDOM", help="a character, or RANDOM for a new one every game")
    p.add_argument("--stage", default="FINAL_DESTINATION")
    p.add_argument("--games", type=int, default=3)
    p.add_argument("--frame-skip", type=int, default=3, help="must match training")
    p.add_argument("--stochastic", action="store_true", help="sample actions instead of taking the most likely one")
    p.add_argument("--fast", action="store_true", help="run the emulator unthrottled instead of at 60 fps")
    p.add_argument("--headless", action="store_true", help="no window and fast-forward mode (Linux only, fastest)")
    p.add_argument("--port", type=int, default=51510)
    return p.parse_args()


def main():
    args = parse_args()
    models = [PPO.load(path, device="cpu") for path in (args.p1_model, args.p2_model)]
    env = SmashEnv(two_player=True, character=args.p1_character, opponent=args.p2_character, stage=args.stage,
                   frame_skip=args.frame_skip, port=args.port, headless=args.headless,
                   emulation_speed=0.0 if args.fast or args.headless else 1.0)
    results = []
    try:
        for game in range(1, args.games + 1):
            obs, info = env.reset()
            done = False
            while not done:
                actions = [model.predict(obs[side], deterministic=not args.stochastic)[0]
                           for side, model in enumerate(models)]
                obs, _, done, _, info = env.step(actions)
            p1, p2 = info["p1"], info["p2"]
            results.append(p1)
            print(f"game {game}: {'P1' if p1['won'] else 'P2'} won | P1 {p1['character']} vs P2 {p2['character']} | "
                  f"stock diff {p1['stock_diff']:+d} | whiffed attacks P1 {p1['whiffs']}/{p1['attacks']} "
                  f"P2 {p2['whiffs']}/{p2['attacks']}")
    finally:
        env.close()
    if results:
        print(f"\nP1 ({args.p1_model}) won {np.mean([r['won'] for r in results]):.0%} of {len(results)} games | "
              f"mean stock diff {np.mean([r['stock_diff'] for r in results]):+.2f}")


if __name__ == "__main__":
    main()
