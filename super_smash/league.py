"""Train MK3 (Fox), MK4 (random), MK5 (Fox) and MK6 (random, starts from MK5's weights) together.

Every model learns, for --hours of wall clock (its learning rate decays to 0 over them).
The 8 parallel games are:

    6 games, one for each pair of models   both models learn from it
    MK6 vs CPU                              CPU level climbs from --mk6-cpu-start to 9: one level
                                            up whenever MK6 wins --promote-at of its last
                                            --promote-window games at the current level
    MK3 / MK4 / MK5 vs CPU level 9          which of the three takes turns every --rotate-every updates

--frozen MK4 keeps a model as an opponent without training it: it still plays its games (always
sampling actions), its weights never change and it leaves the CPU rotation.

CPU opponents get a random character every game. When the rotation changes, the games
are restarted with the new line-up (a few seconds).

Usage:
    python league.py --mk3 runs/mk345/MK3.zip --mk4 runs/mk345/MK4.zip --mk5 runs/mk345/MK5.zip \\
        --hours 12 --run-name mk3456

Writes runs/<run>/MK3.zip ... MK6.zip at the end (also on Ctrl+C), checkpoints every
--save-every updates, and TensorBoard logs to runs/<run>/<model>.
"""

import argparse
import os
import signal
import time
from collections import defaultdict, deque
from itertools import combinations

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import DummyVecEnv

from smash_rl.env import CAN_RUN_HEADLESS, NonDaemonSubprocVecEnv, SmashEnv
from smash_rl.rewards import STAT_KEYS

MODELS = ("MK3", "MK4", "MK5", "MK6")
CHARACTERS = {"MK3": "FOX", "MK4": "RANDOM", "MK5": "FOX", "MK6": "RANDOM", "CPU": "RANDOM"}
CPU_ROTATION = ("MK3", "MK4", "MK5")  # who plays the level 9 CPU game, in turns
CURRICULUM_GAME = 6  # index of the MK6 vs climbing-level CPU game


def layout_for(rotation_turn: int, rotation=CPU_ROTATION):
    """The 8 games as (port 1, port 2)."""
    pairs = list(combinations(MODELS, 2))
    return pairs + [("MK6", "CPU"), (rotation[rotation_turn % len(rotation)], "CPU")]


class _Spaces(gym.Env):
    """Only carries the spaces, so a PPO model can be loaded without a real env."""

    def __init__(self, observation_space, action_space):
        self.observation_space, self.action_space = observation_space, action_space


def slots_of(layout, name):
    """(game, side) pairs where `name` plays."""
    return [(game, side) for game, pair in enumerate(layout) for side, who in enumerate(pair) if who == name]


def make_env(game, layout, cpu_level, base_port, **env_kwargs):
    p1, p2 = layout[game]

    def _init():
        return SmashEnv(port=base_port + game, two_player=True, character=CHARACTERS[p1], opponent=CHARACTERS[p2],
                        cpu_opponent=p2 == "CPU", cpu_level=cpu_level, reward="v2", **env_kwargs)

    return _init


def resize_buffer(model, n_slots):
    """Give a model a rollout buffer for the number of games it plays in the current line-up."""
    model.n_envs = n_slots
    model.rollout_buffer = RolloutBuffer(model.n_steps, model.observation_space, model.action_space,
                                         device=model.device, gamma=model.gamma, gae_lambda=model.gae_lambda,
                                         n_envs=n_slots)


def parse_args():
    p = argparse.ArgumentParser(description="Train MK3, MK4, MK5 and MK6 together against each other and the CPU")
    p.add_argument("--mk3", required=True, help="MK3 weights to continue from")
    p.add_argument("--mk4", required=True, help="MK4 weights to continue from")
    p.add_argument("--mk5", required=True, help="MK5 weights to continue from")
    p.add_argument("--mk6", default=None, help="MK6 weights to continue from (default: start from MK5's weights)")
    p.add_argument("--hours", type=float, default=12.0, help="hours every model learns; learning rates decay to 0 over them")
    p.add_argument("--lr-mk3", type=float, default=4e-5)
    p.add_argument("--lr-mk4", type=float, default=4e-5)
    p.add_argument("--lr-mk5", type=float, default=1.5e-4)
    p.add_argument("--lr-mk6", type=float, default=2e-4)
    for n in ("mk3", "mk4", "mk5", "mk6"):
        p.add_argument(f"--ent-{n}", type=float, default=None,
                       help=f"{n.upper()} entropy bonus (default: keep the saved one, 0.01); higher keeps it exploring")
    p.add_argument("--frozen", nargs="+", default=[], choices=MODELS, metavar="MODEL",
                   help="models that only play as opponents and never learn, e.g. --frozen MK4")
    p.add_argument("--mk6-cpu-start", type=int, default=1, help="CPU level MK6's curriculum game starts at")
    p.add_argument("--promote-window", type=int, default=20, help="MK6 games at a level used to judge promotion")
    p.add_argument("--promote-at", type=float, default=0.7, help="MK6 win rate that moves its CPU up a level")
    p.add_argument("--rotate-every", type=int, default=50, help="updates between turns of the level 9 CPU game")
    p.add_argument("--save-every", type=int, default=100, help="checkpoint every model every N updates")
    p.add_argument("--frame-skip", type=int, default=3)
    p.add_argument("--base-port", type=int, default=51441, help="game i uses port base-port + i")
    p.add_argument("--gui", action="store_true", help="show the Dolphin windows (slower)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", default="mk3456")
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
    env_kwargs = dict(frame_skip=args.frame_skip, headless=headless)

    probe = SmashEnv(two_player=True, reward="v2", **env_kwargs)  # never started; only for the spaces
    spaces_env = DummyVecEnv([lambda: _Spaces(probe.observation_space, probe.action_space)])
    paths = {"MK3": args.mk3, "MK4": args.mk4, "MK5": args.mk5, "MK6": args.mk6 or args.mk5}
    lrs = {"MK3": args.lr_mk3, "MK4": args.lr_mk4, "MK5": args.lr_mk5, "MK6": args.lr_mk6}
    models = {}
    for name in MODELS:
        model = PPO.load(paths[name], env=spaces_env, device="cpu")
        model.lr_schedule = (lambda lr: (lambda progress_remaining: progress_remaining * lr))(lrs[name])
        ent = getattr(args, f"ent_{name.lower()}")
        if ent is not None:
            model.ent_coef = ent
        model.set_logger(configure(os.path.join(run_dir, name), ["tensorboard"]))
        models[name] = model
    n_steps = models["MK3"].n_steps
    if any(m.n_steps != n_steps for m in models.values()):
        raise SystemExit("all models must use the same n_steps")
    learners = [n for n in MODELS if n not in args.frozen]
    rotation = [n for n in CPU_ROTATION if n in learners]
    if not rotation:
        raise SystemExit("at least one of " + ", ".join(CPU_ROTATION) + " must learn (it plays the level 9 CPU)")
    print(f"run dir: {run_dir} | {args.hours:g} h | " + ", ".join(
        f"{n} ({CHARACTERS[n].lower()}, from {paths[n]}, " +
        ("frozen)" if n in args.frozen else f"lr {lrs[n]:g}, ent {models[n].ent_coef:g})") for n in MODELS), flush=True)

    cpu_level = args.mk6_cpu_start
    at_level = deque(maxlen=args.promote_window)  # MK6's results against the curriculum CPU at its current level
    recent = {name: defaultdict(lambda: deque(maxlen=50)) for name in MODELS}  # model -> opponent -> games
    env = None
    update = 0
    start = time.time()
    try:
        turn = 0
        while (time.time() - start) / 3600 < args.hours:
            layout = layout_for(turn, rotation)
            slots = {name: slots_of(layout, name) for name in MODELS}
            for name in learners:
                resize_buffer(models[name], len(slots[name]))
            print(f"\n=== games: " + ", ".join(f"{a} vs {b}" + (f" (level {cpu_level})" if i == CURRICULUM_GAME else
                                                                  " (level 9)" if b == "CPU" else "")
                                              for i, (a, b) in enumerate(layout)), flush=True)
            if env is not None:
                env.close()
            env = NonDaemonSubprocVecEnv([make_env(i, layout, cpu_level if i == CURRICULUM_GAME else 9,
                                                   args.base_port, **env_kwargs) for i in range(len(layout))])
            env.seed(args.seed + update)
            obs = env.reset()
            starts = np.ones(len(layout), dtype=bool)
            phase_start, phase_updates = time.time(), 0

            for _ in range(args.rotate_every):
                if (time.time() - start) / 3600 >= args.hours:
                    break
                update += 1
                phase_updates += 1
                games = []
                for model in models.values():
                    model.policy.set_training_mode(False)
                for name in learners:
                    models[name].rollout_buffer.reset()
                for _ in range(n_steps):
                    actions = np.zeros((len(layout), 2), dtype=np.int64)
                    step_data = {}
                    with torch.no_grad():
                        for name, where in slots.items():
                            games_i, sides = (list(x) for x in zip(*where))
                            sub_obs = obs[games_i, sides]
                            action, value, log_prob = models[name].policy(obs_as_tensor(sub_obs, "cpu"))
                            actions[games_i, sides] = action.cpu().numpy()
                            step_data[name] = (sub_obs, action.cpu().numpy(), value, log_prob, games_i, sides)
                    new_obs, rewards, dones, infos = env.step(actions)
                    for name in learners:
                        model = models[name]
                        sub_obs, action, value, log_prob, games_i, sides = step_data[name]
                        model.num_timesteps += len(games_i)
                        model.rollout_buffer.add(sub_obs, action.reshape(-1, 1), rewards[games_i, sides],
                                                 starts[games_i], value, log_prob)
                    games += [(game, infos[game]) for game, done in enumerate(dones) if done]
                    obs, starts = new_obs, dones

                elapsed_h = (time.time() - start) / 3600
                for name in learners:
                    model = models[name]
                    games_i, sides = (list(x) for x in zip(*slots[name]))
                    with torch.no_grad():
                        last = model.policy.predict_values(obs_as_tensor(obs[games_i, sides], "cpu"))
                    model.rollout_buffer.compute_returns_and_advantage(last_values=last, dones=starts[games_i])
                    model._current_progress_remaining = max(0.0, 1.0 - elapsed_h / args.hours)
                    model.train()
                for game, info in games:
                    for side, who in enumerate(layout[game]):
                        mine = info[("p1", "p2")[side]]
                        if who == "CPU" or mine.get("crashed"):  # crashed: a Dolphin froze or died
                            continue
                        opponent = layout[game][1 - side]
                        if game == CURRICULUM_GAME:
                            opponent = f"CPU{cpu_level}"
                            if mine.get("cpu_level") == cpu_level:
                                at_level.append(float(mine.get("won", 0.0)))
                        elif opponent == "CPU":
                            opponent = "CPU9"
                        recent[who][opponent].append(mine)
                        log = models[who].logger
                        for key in ("won", "stock_diff") + STAT_KEYS:
                            log.record_mean(f"smash/{key}", float(mine.get(key, 0.0)))
                        log.record_mean(f"smash/won_vs_{opponent}", float(mine.get("won", 0.0)))
                        log.record_mean("smash/whiff_rate", mine["whiffs"] / max(mine["attacks"], 1))
                models["MK6"].logger.record("smash/cpu_level", cpu_level)
                for model in models.values():
                    model.logger.dump(step=model.num_timesteps)

                if cpu_level < 9 and len(at_level) == at_level.maxlen and np.mean(at_level) >= args.promote_at:
                    cpu_level += 1
                    at_level.clear()
                    env.env_method("set_cpu_level", cpu_level, indices=[CURRICULUM_GAME])
                    print(f"\n>>> MK6 won {args.promote_at:.0%}+ of its last games: its CPU is now level {cpu_level}\n",
                          flush=True)
                if games or phase_updates == 1:
                    parts = []
                    for name in MODELS:
                        rates = ", ".join(f"vs {opp} {np.mean([g['won'] for g in hist]):.0%}"
                                          for opp, hist in sorted(recent[name].items()) if hist)
                        every = [g for hist in recent[name].values() for g in hist]
                        whiff = sum(g["whiffs"] for g in every) / max(sum(g["attacks"] for g in every), 1)
                        combos = np.mean([g["combos"] for g in every]) if every else 0.0
                        defences = np.mean([g["defences"] for g in every]) if every else 0.0
                        smash = np.mean([g.get("smash_spam", 0) for g in every]) if every else 0.0
                        spam = np.mean([g.get("spam_penalty", 0.0) for g in every]) if every else 0.0
                        parts.append(f"{name} won {rates or '-'} | whiffs {whiff:.0%} | combos {combos:.2f} | "
                                     f"defences {defences:.0f} | smash spam {smash:.1f} | spam cost {spam:.1f}")
                    speed = phase_updates * n_steps * len(layout) / (time.time() - phase_start)
                    print(f"update {update} | {elapsed_h:.2f} h | {speed:.0f} steps/s | MK6 CPU level {cpu_level} || "
                          + " || ".join(parts), flush=True)
                if update % args.save_every == 0:
                    for name, model in models.items():
                        model.save(os.path.join(run_dir, "checkpoints", f"{name}_update{update:05d}"))
                    print(f"saved checkpoints for update {update}", flush=True)
            turn += 1
    except KeyboardInterrupt:
        print(f"\nstopped at update {update} ({(time.time() - start) / 3600:.2f} h) - saving current models")
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)  # a second Ctrl+C or a closed terminal must not cut the save short
        for name, model in models.items():
            model.save(os.path.join(run_dir, name))
        print(f"saved {', '.join(f'{run_dir}/{name}.zip' for name in models)} | MK6 CPU level {cpu_level}", flush=True)
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
