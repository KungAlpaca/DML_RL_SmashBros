"""Stage 2: MK1 and MK2 learn at the same time by playing each other.

Both start from the same weights (MK0 by default). MK1 is port 1 and keeps Fox; MK2 is
port 2 and gets a random character every game. Every update both models play the same
games (each from its own side), then each runs its own PPO update on what it saw.

Usage:
    python selfplay.py --init models/MK0.zip --updates 300 --run-name mk_selfplay
    python selfplay.py --mk1 runs/mk_selfplay/MK1.zip --mk2 runs/mk_selfplay/MK2.zip --updates 200 --run-name mk_more

Writes runs/<run>/MK1.zip and MK2.zip at the end (also on Ctrl+C), checkpoints every
--save-every updates to runs/<run>/checkpoints/, and TensorBoard logs to runs/<run>/MK1
and runs/<run>/MK2 (tensorboard --logdir runs/<run>).
"""

import argparse
import os
import signal
import time
from collections import deque

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import DummyVecEnv

from smash_rl.env import CAN_RUN_HEADLESS, NonDaemonSubprocVecEnv, SmashEnv

STAT_KEYS = ("won", "stock_diff", "kills", "deaths", "damage_dealt", "damage_taken", "attacks")


class _Spaces(gym.Env):
    """Only carries the spaces, so PPO.load can size a model's rollout buffer for n_envs games."""

    def __init__(self, observation_space, action_space):
        self.observation_space, self.action_space = observation_space, action_space


def make_env(rank: int, base_port: int, **env_kwargs):
    def _init():
        return SmashEnv(port=base_port + rank, two_player=True, **env_kwargs)

    return _init


def load_model(path: str, spaces_env: DummyVecEnv, lr: float, log_dir: str) -> PPO:
    model = PPO.load(path, env=spaces_env, device="cpu")
    model.lr_schedule = lambda progress_remaining: progress_remaining * lr
    model.set_logger(configure(log_dir, ["tensorboard"]))
    return model


def collect(models, env, obs, starts, n_steps, games):
    """Play n_steps with both models, filling their rollout buffers. Returns the last obs / episode starts."""
    for model in models:
        model.policy.set_training_mode(False)
        model.rollout_buffer.reset()
    for _ in range(n_steps):
        actions, values, log_probs = [], [], []
        with torch.no_grad():
            for side, model in enumerate(models):
                action, value, log_prob = model.policy(obs_as_tensor(obs[:, side], model.device))
                actions.append(action.cpu().numpy())
                values.append(value)
                log_probs.append(log_prob)
        new_obs, rewards, dones, infos = env.step(np.stack(actions, axis=1))
        for side, model in enumerate(models):
            model.num_timesteps += env.num_envs
            model.rollout_buffer.add(obs[:, side], actions[side].reshape(-1, 1), rewards[:, side], starts,
                                     values[side], log_probs[side])
        games.extend(info for info, done in zip(infos, dones) if done and not info["p1"].get("crashed"))
        obs, starts = new_obs, dones
    with torch.no_grad():
        for side, model in enumerate(models):
            last_values = model.policy.predict_values(obs_as_tensor(obs[:, side], model.device))
            model.rollout_buffer.compute_returns_and_advantage(last_values=last_values, dones=starts)
    return obs, starts


def log_games(model, games, side):
    """Record this update's finished games, from one side's point of view, to the model's TensorBoard."""
    for game in games:
        info = game[side]
        for key in STAT_KEYS:
            model.logger.record_mean(f"smash/{key}", float(info.get(key, 0.0)))
        model.logger.record_mean("smash/whiff_rate", info["whiffs"] / max(info["attacks"], 1))


def parse_args():
    p = argparse.ArgumentParser(description="Train MK1 (Fox, port 1) and MK2 (random character, port 2) against each other")
    p.add_argument("--init", default="models/MK0.zip", help="starting weights for both models")
    p.add_argument("--mk1", default=None, help="starting weights for MK1 only (e.g. to continue a run)")
    p.add_argument("--mk2", default=None, help="starting weights for MK2 only")
    p.add_argument("--updates", type=int, default=200,
                   help="PPO updates; each is n_envs x n_steps game steps followed by n_epochs passes over them")
    p.add_argument("--mk1-character", default="FOX")
    p.add_argument("--mk2-character", default="RANDOM", help="a character, or RANDOM for a new one every game")
    p.add_argument("--whiff-penalty", type=float, default=0.01,
                   help="reward subtracted for each A attack that ends without hitting the opponent or their shield")
    p.add_argument("--lr", type=float, default=1e-4, help="learning rate for both, decaying linearly to 0")
    p.add_argument("--n-envs", type=int, default=8, help="parallel games (one Dolphin each)")
    p.add_argument("--stage", default="FINAL_DESTINATION")
    p.add_argument("--frame-skip", type=int, default=3, help="must match how the starting model was trained")
    p.add_argument("--save-every", type=int, default=25, help="checkpoint both models every N updates")
    p.add_argument("--base-port", type=int, default=51441, help="env i uses port base-port + i")
    p.add_argument("--gui", action="store_true", help="show a Dolphin window per env (slower)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", default="mk_selfplay")
    return p.parse_args()


def _raise_interrupt(signum, frame):
    raise KeyboardInterrupt


def main():
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, _raise_interrupt)
    signal.signal(signal.SIGHUP, _raise_interrupt)
    args = parse_args()
    run_dir = os.path.join("runs", args.run_name)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    headless = CAN_RUN_HEADLESS and not args.gui

    env_kwargs = dict(character=args.mk1_character, opponent=args.mk2_character, stage=args.stage,
                      frame_skip=args.frame_skip, whiff_penalty=args.whiff_penalty, headless=headless)
    probe = SmashEnv(two_player=True, **env_kwargs)  # never started; only for the spaces
    spaces_env = DummyVecEnv([lambda: _Spaces(probe.observation_space, probe.action_space)] * args.n_envs)
    names = ("MK1", "MK2")
    models = [load_model(path or args.init, spaces_env, args.lr, os.path.join(run_dir, name))
              for path, name in zip((args.mk1, args.mk2), names)]
    steps_per_update = models[0].n_steps * args.n_envs
    print(f"run dir: {run_dir} | MK1 = {args.mk1 or args.init} as {args.mk1_character} | "
          f"MK2 = {args.mk2 or args.init} as {args.mk2_character} | {args.n_envs} "
          f"{'headless ' if headless else ''}dolphins | {args.updates} updates of {steps_per_update} steps")

    env = NonDaemonSubprocVecEnv([make_env(i, args.base_port, **env_kwargs) for i in range(args.n_envs)])
    recent = deque(maxlen=100)
    total_games = update = 0
    try:
        env.seed(args.seed)
        obs, starts = env.reset(), np.ones(args.n_envs, dtype=bool)
        start = time.time()
        for update in range(1, args.updates + 1):
            games = []
            obs, starts = collect(models, env, obs, starts, models[0].n_steps, games)
            for side, model in enumerate(models):
                model._current_progress_remaining = 1.0 - (update - 1) / args.updates
                model.train()
                log_games(model, games, ("p1", "p2")[side])
                model.logger.record("smash/games", total_games + len(games))
                model.logger.dump(step=model.num_timesteps)
            recent.extend(games)
            total_games += len(games)
            if games or update == 1:
                mk1 = [g["p1"] for g in recent]
                whiff = [sum(g[p]["whiffs"] for g in recent) / max(sum(g[p]["attacks"] for g in recent), 1)
                         for p in ("p1", "p2")]
                print(f"update {update}/{args.updates} | {models[0].num_timesteps:,} steps | "
                      f"{update * steps_per_update / (time.time() - start):.0f} steps/s | last {len(recent)} games: "
                      f"MK1 won {np.mean([g['won'] for g in mk1]) if mk1 else 0:.0%}, "
                      f"stock diff {np.mean([g['stock_diff'] for g in mk1]) if mk1 else 0:+.2f} | "
                      f"whiff rate MK1 {whiff[0]:.0%} MK2 {whiff[1]:.0%}", flush=True)
            if update % args.save_every == 0:
                for model, name in zip(models, names):
                    model.save(os.path.join(run_dir, "checkpoints", f"{name}_update{update:04d}"))
                print(f"saved checkpoints for update {update}")
    except KeyboardInterrupt:
        print(f"\nstopped at update {update} - saving current models")
    finally:
        for model, name in zip(models, names):
            model.save(os.path.join(run_dir, name))
        print(f"saved {run_dir}/MK1.zip and {run_dir}/MK2.zip")
        env.close()


if __name__ == "__main__":
    main()
