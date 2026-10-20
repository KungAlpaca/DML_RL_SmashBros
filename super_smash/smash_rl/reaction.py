"""A human-like reaction time: the model sees the game as it was a moment ago.

The agents read the game's memory and decide every frame_skip frames, so without this they
react in 0 ms; people need about 200-250 ms to see something on screen and press a button.
Models trained with one (MK7, MK8) store it as model.reaction_ms; play.py and party.py use it.
"""

from collections import deque

import numpy as np


class ReactionDelay:
    def __init__(self, ms: float, frame_skip: int):
        self.steps = max(0, round(ms / 1000 * 60 / frame_skip))  # decisions the agent's view lags behind
        self.ms = self.steps * frame_skip * 1000 / 60
        self.reset()

    def reset(self):
        """Call at the start of every game."""
        self.seen = deque(maxlen=self.steps + 1)

    def __call__(self, obs):
        """Takes the observation now, returns the one from self.steps decisions ago (the oldest one early in a game)."""
        self.seen.append(np.array(obs, copy=True))
        return self.seen[0]

    def peek(self, obs, new_game: bool = False):
        """What __call__(obs) would return, without storing obs (for PPO's value bootstrap)."""
        if new_game:
            return obs
        return (list(self.seen) + [obs])[-(self.steps + 1):][0]
