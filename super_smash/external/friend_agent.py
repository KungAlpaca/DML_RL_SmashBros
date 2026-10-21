"""One of our PPO models as the `FriendAgent` of another team's libmelee match runner.

Some teams ship their agent with a match.py that loads `friend_agent.FriendAgent(port, opponent_port)`
and calls `.reset()` every game and `.act(gs, controller)` every in-game frame (e.g. the Ganondorf
agent). Put this folder first on PYTHONPATH and pick the model with environment variables:

    SMASH_MODEL=runs/.../MK9.zip     the model (its saved reaction time, model.reaction_ms, is used)
    SMASH_CHARACTER=FOX              the character it plays (default FOX)
    SMASH_DETERMINISTIC=0            1: most likely action, 0: sample as in training (default)

It plays exactly like in our env: observation() from smash_rl.env, one action every frame_skip frames (the model's),
A/B/X/Z released on the last frame of each action. vs_friend.py runs whole series with it.
"""

import os

import melee
from stable_baselines3 import PPO

from smash_rl.env import TAP_BUTTONS, SmashEnv, model_frame_skip, observation
from smash_rl.reaction import ReactionDelay


class FriendAgent:
    character = os.environ.get("SMASH_CHARACTER", "FOX").upper()

    def __init__(self, port: int, opponent_port: int):
        self.port, self.opponent_port = port, opponent_port
        self.model = PPO.load(os.environ["SMASH_MODEL"], device="cpu")
        self.deterministic = os.environ.get("SMASH_DETERMINISTIC", "0") == "1"
        self.frame_skip = model_frame_skip(self.model)
        self.delay = ReactionDelay(getattr(self.model, "reaction_ms", 0), self.frame_skip)
        self.edge = melee.stages.EDGE_GROUND_POSITION[melee.Stage.FINAL_DESTINATION]
        self.reset()

    def reset(self):
        self.frame, self.button = 0, None
        self.delay.reset()

    def act(self, gs, controller):
        if self.port not in gs.players or self.opponent_port not in gs.players:
            return
        step = self.frame % self.frame_skip
        self.frame += 1
        if step == 0:
            me, op = gs.players[self.port], gs.players[self.opponent_port]
            obs = self.delay(observation(gs, me, op, self.edge))
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
            self.button = SmashEnv._press(controller, int(action))
        elif step == self.frame_skip - 1 and self.button in TAP_BUTTONS:
            controller.release_button(self.button)
