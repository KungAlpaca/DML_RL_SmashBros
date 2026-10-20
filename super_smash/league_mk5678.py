"""Train MK5 (Fox), MK6 (random), MK7 (Fox) and MK8 (random) together; MK3 and MK4 are frozen opponents.

MK7 and MK8 start from random weights and have a human-like reaction time (--reaction-mk7 /
--reaction-mk8, 200 ms by default): they see the game as it was that long ago, during training
and later in play.py / party.py (the delay is saved inside the model). MK5 and MK6 react instantly,
as before. MK3 and MK4 only play: their weights never change.

Every learner learns for --hours of wall clock (its learning rate decays to 0 over them).
The 8 parallel games are:

    MK7 vs CPU, MK8 vs CPU      each with its own CPU level, from --cpu-start up to 9: one level up
                                whenever the model wins --promote-at of its last --promote-window
                                games at the current level
    MK7 vs MK8, MK5 vs MK6      always
    4 more games                a new set every --rotate-every updates (ROTATION): the learners
                                against each other, against MK3 / MK4 and MK5 / MK6 against CPU 9

Usage:
    python league_mk5678.py --mk3 runs/mk3456_g/MK3.zip --mk4 runs/mk3456_g/MK4.zip \\
        --mk5 runs/mk3456_g/MK5.zip --mk6 runs/mk3456_g/MK6.zip --hours 12 --run-name mk5678

--mk7 / --mk8 continue a saved MK7 / MK8 instead of starting from scratch (pass --mk7-cpu-start /
--mk8-cpu-start then: the CPU levels are not saved). Writes runs/<run>/MK3.zip ... MK8.zip at the
end (also on Ctrl+C), checkpoints every --save-every updates, and TensorBoard logs to runs/<run>/<model>.
"""

import argparse
import os
import signal
import time
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import DummyVecEnv

from league import _raise_interrupt, _Spaces, resize_buffer, slots_of
from smash_rl.env import CAN_RUN_HEADLESS, NonDaemonSubprocVecEnv, SmashEnv
from smash_rl.reaction import ReactionDelay
from smash_rl.rewards import STAT_KEYS

@dataclass
class Lineup:
    """Who trains, who only plays, and which games they play."""
    description: str
    characters: dict  # model -> character, plus "CPU"
    frozen: tuple  # models that only play
    new: dict  # model -> None (starts from scratch) or the model whose weights it starts from, unless --<model> is given
    curriculum: dict  # model -> game index of its climbing-level CPU game
    fixed: list  # games always played, as (port 1, port 2)
    rotation: list  # sets of games that take turns every --rotate-every updates
    lrs: dict  # learner -> default learning rate
    cpu_start: int  # default first level of the curriculum games
    run_name: str

    @property
    def models(self):
        return tuple(m for m in self.characters if m != "CPU")

    @property
    def learners(self):
        return tuple(m for m in self.models if m not in self.frozen)

    def layout_for(self, turn: int):
        """The 8 games as (port 1, port 2)."""
        return self.fixed + self.rotation[turn % len(self.rotation)]


MK5678 = Lineup(
    description="Train MK5-MK8 together against each other, frozen MK3 / MK4 and the CPU",
    characters={"MK3": "FOX", "MK4": "RANDOM", "MK5": "FOX", "MK6": "RANDOM", "MK7": "FOX", "MK8": "RANDOM",
                "CPU": "RANDOM"},
    frozen=("MK3", "MK4"),
    new={"MK7": None, "MK8": None},
    curriculum={"MK7": 0, "MK8": 1},
    fixed=[("MK7", "CPU"), ("MK8", "CPU"), ("MK7", "MK8"), ("MK5", "MK6")],
    rotation=[  # 4 games per turn; over a cycle every learner meets every other model
        [("MK5", "MK7"), ("MK6", "MK8"), ("MK5", "MK3"), ("MK6", "MK4")],
        [("MK5", "MK8"), ("MK6", "MK7"), ("MK7", "MK3"), ("MK8", "MK4")],
        [("MK5", "MK4"), ("MK6", "MK3"), ("MK5", "CPU"), ("MK6", "CPU")],
        [("MK7", "MK4"), ("MK8", "MK3"), ("MK7", "MK5"), ("MK8", "MK6")],
    ],
    lrs={"MK5": 6e-5, "MK6": 8e-5, "MK7": 3e-4, "MK8": 3e-4},
    cpu_start=1,
    run_name="mk5678",
)


def make_env(lu, game, layout, cpu_level, base_port, **env_kwargs):
    p1, p2 = layout[game]

    def _init():
        return SmashEnv(port=base_port + game, two_player=True, character=lu.characters[p1],
                        opponent=lu.characters[p2], cpu_opponent=p2 == "CPU", cpu_level=cpu_level, reward="v2",
                        **env_kwargs)

    return _init


def parse_args(lu):
    p = argparse.ArgumentParser(description=lu.description)
    for name in lu.models:
        n = name.lower()
        start = lu.new.get(name, "")
        p.add_argument(f"--{n}", required=name not in lu.new,
                       help=f"{name} weights to continue from" + (" (default: from scratch)" if start is None else
                                                                  f" (default: start from --{start.lower()}'s weights)"
                                                                  if start else ""))
        if start:
            p.add_argument(f"--{n}-init", default=None,
                           help=f"weights {name} starts from when --{n} is not given (default: --{start.lower()})")
    p.add_argument("--hours", type=float, default=12.0, help="hours every learner learns; learning rates decay to 0 over them")
    for name in lu.learners:
        n = name.lower()
        p.add_argument(f"--lr-{n}", type=float, default=lu.lrs[name])
        p.add_argument(f"--ent-{n}", type=float, default=None,
                       help=f"{name} entropy bonus (default: the saved one, 0.01 from scratch)")
    for name in lu.new:
        p.add_argument(f"--reaction-{name.lower()}", type=float, default=200,
                       help=f"{name} reaction time in ms when it starts (a continued model keeps its own)")
    for name in lu.curriculum:
        p.add_argument(f"--{name.lower()}-cpu-start", type=int, default=lu.cpu_start,
                       help=f"CPU level {name}'s curriculum game starts at")
    p.add_argument("--template", default=None,
                   help="model whose PPO settings models starting from scratch copy (default: the first learner)")
    p.add_argument("--promote-window", type=int, default=20, help="games at a level used to judge promotion")
    p.add_argument("--promote-at", type=float, default=0.7, help="win rate that moves a curriculum CPU up a level")
    p.add_argument("--rotate-every", type=int, default=50, help="updates between sets of rotating games")
    p.add_argument("--save-every", type=int, default=100, help="checkpoint every model every N updates")
    p.add_argument("--frame-skip", type=int, default=3)
    p.add_argument("--base-port", type=int, default=51441, help="game i uses port base-port + i")
    p.add_argument("--gui", action="store_true", help="show the Dolphin windows (slower)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", default=lu.run_name)
    return p.parse_args()


def levels_text(cpu_level):
    return ", ".join(f"{name} {level}" for name, level in cpu_level.items())


def from_scratch(template: str, spaces_env, seed: int, reaction_ms: float):
    """A model with the template's PPO settings and fresh random weights."""
    model = PPO.load(template, env=spaces_env, device="cpu")
    torch.manual_seed(seed)
    model._setup_model()  # new policy, optimizer and rollout buffer
    model.num_timesteps = 0
    model.ent_coef = 0.01
    model.reaction_ms = reaction_ms
    return model


def main(lu=MK5678):
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, _raise_interrupt)
    signal.signal(signal.SIGHUP, _raise_interrupt)
    args = parse_args(lu)
    MODELS, FROZEN, LEARNERS, CURRICULUM = lu.models, lu.frozen, lu.learners, lu.curriculum
    run_dir = os.path.join("runs", args.run_name)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    headless = CAN_RUN_HEADLESS and not args.gui
    env_kwargs = dict(frame_skip=args.frame_skip, headless=headless)

    probe = SmashEnv(two_player=True, reward="v2", **env_kwargs)  # never started; only for the spaces
    spaces_env = DummyVecEnv([lambda: _Spaces(probe.observation_space, probe.action_space)])
    models, origin = {}, {}
    for i, name in enumerate(MODELS):
        n = name.lower()
        path = getattr(args, n)
        if path is None and lu.new[name] is None:
            template = args.template or getattr(args, LEARNERS[0].lower())
            model = from_scratch(template, spaces_env, args.seed + 100 + i, getattr(args, f"reaction_{n}"))
            origin[name] = "scratch"
        elif path is None:
            start = getattr(args, f"{n}_init") or getattr(args, lu.new[name].lower())
            model = PPO.load(start, env=spaces_env, device="cpu")
            model.num_timesteps = 0
            model.reaction_ms = getattr(args, f"reaction_{n}")
            origin[name] = f"{start} weights"
        else:
            model = PPO.load(path, env=spaces_env, device="cpu")
            origin[name] = path
        if name in LEARNERS:
            lr = getattr(args, f"lr_{n}")
            model.lr_schedule = (lambda lr: (lambda progress_remaining: progress_remaining * lr))(lr)
            ent = getattr(args, f"ent_{n}")
            if ent is not None:
                model.ent_coef = ent
        model.reaction_ms = getattr(model, "reaction_ms", 0)
        model.set_logger(configure(os.path.join(run_dir, name), ["tensorboard"]))
        models[name] = model
    n_steps = models[LEARNERS[0]].n_steps
    if any(m.n_steps != n_steps for m in models.values()):
        raise SystemExit("all models must use the same n_steps")
    print(f"run dir: {run_dir} | {args.hours:g} h | " + ", ".join(
        f"{n} ({lu.characters[n].lower()}, from {origin[n]}, "
        + ("frozen" if n in FROZEN else f"lr {models[n].lr_schedule(1.0):g}, ent {models[n].ent_coef:g}")
        + f", reaction {models[n].reaction_ms:g} ms)" for n in MODELS), flush=True)

    cpu_level = {name: getattr(args, f"{name.lower()}_cpu_start") for name in CURRICULUM}
    at_level = {name: deque(maxlen=args.promote_window) for name in CURRICULUM}
    recent = {name: defaultdict(lambda: deque(maxlen=50)) for name in MODELS}  # model -> opponent -> games
    env = None
    update = 0
    start = time.time()
    try:
        turn = 0
        while (time.time() - start) / 3600 < args.hours:
            layout = lu.layout_for(turn)
            levels = [cpu_level[a] if CURRICULUM.get(a) == i else 9 for i, (a, b) in enumerate(layout)]
            slots = {name: slots_of(layout, name) for name in MODELS}
            playing = [name for name in MODELS if slots[name]]
            learning = [name for name in playing if name in LEARNERS]
            for name in learning:
                resize_buffer(models[name], len(slots[name]))
            # One reaction delay per game a model plays in.
            delays = {name: [ReactionDelay(models[name].reaction_ms, args.frame_skip) for _ in slots[name]]
                      for name in playing}
            print("\n=== games: " + ", ".join(f"{a} vs {b}" + (f" (level {levels[i]})" if b == "CPU" else "")
                                             for i, (a, b) in enumerate(layout)), flush=True)
            if env is not None:
                env.close()
            env = NonDaemonSubprocVecEnv([make_env(lu, i, layout, levels[i], args.base_port, **env_kwargs)
                                          for i in range(len(layout))])
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
                for name in learning:
                    models[name].rollout_buffer.reset()
                for _ in range(n_steps):
                    actions = np.zeros((len(layout), 2), dtype=np.int64)
                    step_data = {}
                    with torch.no_grad():
                        for name in playing:
                            games_i, sides = (list(x) for x in zip(*slots[name]))
                            seen = []
                            for delay, game, side in zip(delays[name], games_i, sides):
                                if starts[game]:
                                    delay.reset()
                                seen.append(delay(obs[game, side]))
                            seen = np.stack(seen)
                            action, value, log_prob = models[name].policy(obs_as_tensor(seen, "cpu"))
                            actions[games_i, sides] = action.cpu().numpy()
                            step_data[name] = (seen, action.cpu().numpy(), value, log_prob, games_i, sides)
                    new_obs, rewards, dones, infos = env.step(actions)
                    for name in learning:
                        model = models[name]
                        seen, action, value, log_prob, games_i, sides = step_data[name]
                        model.num_timesteps += len(games_i)
                        model.rollout_buffer.add(seen, action.reshape(-1, 1), rewards[games_i, sides],
                                                 starts[games_i], value, log_prob)
                    games += [(game, infos[game]) for game, done in enumerate(dones) if done]
                    obs, starts = new_obs, dones

                elapsed_h = (time.time() - start) / 3600
                for name in learning:
                    model = models[name]
                    games_i, sides = (list(x) for x in zip(*slots[name]))
                    last_seen = np.stack([delay.peek(obs[game, side], new_game=starts[game])
                                          for delay, game, side in zip(delays[name], games_i, sides)])
                    with torch.no_grad():
                        last = model.policy.predict_values(obs_as_tensor(last_seen, "cpu"))
                    model.rollout_buffer.compute_returns_and_advantage(last_values=last, dones=starts[games_i])
                    model._current_progress_remaining = max(0.0, 1.0 - elapsed_h / args.hours)
                    model.train()
                for game, info in games:
                    for side, who in enumerate(layout[game]):
                        mine = info[("p1", "p2")[side]]
                        if who == "CPU" or mine.get("crashed"):  # crashed: a Dolphin froze or died
                            continue
                        opponent = layout[game][1 - side]
                        if opponent == "CPU":
                            opponent = f"CPU{levels[game]}"
                            if CURRICULUM.get(who) == game and mine.get("cpu_level") == levels[game]:
                                at_level[who].append(float(mine.get("won", 0.0)))
                        recent[who][opponent].append(mine)
                        log = models[who].logger
                        for key in ("won", "stock_diff") + STAT_KEYS:
                            log.record_mean(f"smash/{key}", float(mine.get(key, 0.0)))
                        log.record_mean(f"smash/won_vs_{opponent}", float(mine.get("won", 0.0)))
                        log.record_mean("smash/whiff_rate", mine["whiffs"] / max(mine["attacks"], 1))
                for name in CURRICULUM:
                    models[name].logger.record("smash/cpu_level", cpu_level[name])
                for name in learning:
                    models[name].logger.dump(step=models[name].num_timesteps)

                for name, game in CURRICULUM.items():
                    results = at_level[name]
                    if cpu_level[name] < 9 and len(results) == results.maxlen and np.mean(results) >= args.promote_at:
                        cpu_level[name] += 1
                        levels[game] = cpu_level[name]
                        results.clear()
                        env.env_method("set_cpu_level", cpu_level[name], indices=[game])
                        print(f"\n>>> {name} won {args.promote_at:.0%}+ of its last games: its CPU is now level "
                              f"{cpu_level[name]}\n", flush=True)
                if games or phase_updates == 1:
                    parts = []
                    for name in LEARNERS:
                        hist_by_opp = recent[name]
                        rates = ", ".join(f"vs {opp} {np.mean([g['won'] for g in hist]):.0%}"
                                          for opp, hist in sorted(hist_by_opp.items()) if hist)
                        every = [g for hist in hist_by_opp.values() for g in hist]
                        whiff = sum(g["whiffs"] for g in every) / max(sum(g["attacks"] for g in every), 1)
                        mean = (lambda key: np.mean([g.get(key, 0.0) for g in every]) if every else 0.0)
                        parts.append(f"{name} won {rates or '-'} | whiffs {whiff:.0%} | combos {mean('combos'):.2f} | "
                                     f"defences {mean('defences'):.0f} | smash spam {mean('smash_spam'):.1f} | "
                                     f"spam cost {mean('spam_penalty'):.1f}")
                    speed = phase_updates * n_steps * len(layout) / (time.time() - phase_start)
                    print(f"update {update} | {elapsed_h:.2f} h | {speed:.0f} steps/s | CPU level {levels_text(cpu_level)} || "
                          + " || ".join(parts), flush=True)
                if update % args.save_every == 0:
                    for name in LEARNERS:
                        models[name].save(os.path.join(run_dir, "checkpoints", f"{name}_update{update:05d}"))
                    print(f"saved checkpoints for update {update}", flush=True)
            turn += 1
    except KeyboardInterrupt:
        print(f"\nstopped at update {update} ({(time.time() - start) / 3600:.2f} h) - saving current models")
    finally:
        for name, model in models.items():
            model.save(os.path.join(run_dir, name))
        print(f"saved {', '.join(f'{run_dir}/{name}.zip' for name in models)} | CPU level {levels_text(cpu_level)}",
              flush=True)
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
