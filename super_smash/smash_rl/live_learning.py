"""Fine-tune a model on the games it plays against a person (play.py / party.py --learn).

The agent samples its actions (PPO needs them on-policy), every decision and the reward
that followed it are kept, and when a game ends the model runs one PPO update (n_epochs
passes) over that game and is saved. One game is a few thousand steps against the
millions of a training run, so this nudges a model rather than retraining it.
"""

import numpy as np
import torch
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.utils import obs_as_tensor


def live_path(model_path: str) -> str:
    """Where a fine-tuned copy goes: .../MK3.zip -> .../MK3_live.zip (a _live model keeps its name)."""
    stem = model_path[:-4] if model_path.endswith(".zip") else model_path
    return f"{stem}.zip" if stem.endswith("_live") else f"{stem}_live.zip"


class LiveLearner:
    def __init__(self, model, save_path: str, lr: float):
        self.model, self.save_path = model, save_path
        model.lr_schedule = lambda progress_remaining: lr
        model._current_progress_remaining = 1.0
        model.set_logger(Logger(folder=None, output_formats=[]))
        self.games = 0
        self._reset()

    def _reset(self):
        self.obs, self.actions, self.values, self.log_probs, self.rewards, self.starts = [], [], [], [], [], []

    def act(self, obs) -> int:
        """Sample an action for `obs` and remember it; call record() with its reward before the next act()."""
        policy = self.model.policy
        policy.set_training_mode(False)
        with torch.no_grad():
            action, value, log_prob = policy(obs_as_tensor(obs[None], policy.device))
        self.starts.append(not self.obs)
        self.obs.append(np.asarray(obs, dtype=np.float32))
        self.actions.append(int(action.item()))
        self.values.append(value)
        self.log_probs.append(log_prob)
        return self.actions[-1]

    def record(self, reward: float):
        if len(self.rewards) < len(self.obs):
            self.rewards.append(float(reward))

    def end_game(self) -> str:
        """Run one PPO update on the game just played and save. Returns a one-line summary."""
        steps = min(len(self.obs), len(self.rewards))
        if steps < 2:
            self._reset()
            return "game too short to learn from"
        m = self.model
        buffer = RolloutBuffer(steps, m.observation_space, m.action_space, device=m.device,
                               gamma=m.gamma, gae_lambda=m.gae_lambda, n_envs=1)
        for i in range(steps):
            buffer.add(self.obs[i][None], np.array([[self.actions[i]]]), np.array([self.rewards[i]]),
                       np.array([self.starts[i]]), self.values[i], self.log_probs[i])
        buffer.compute_returns_and_advantage(last_values=torch.zeros(1, 1), dones=np.array([True]))
        saved_buffer, m.rollout_buffer = m.rollout_buffer, buffer
        m.train()
        m.rollout_buffer = saved_buffer
        m.num_timesteps += steps
        m.save(self.save_path)
        self.games += 1
        total = sum(self.rewards[:steps])
        self._reset()
        return f"learned from {steps} decisions (reward {total:+.2f}), saved {self.save_path}"
