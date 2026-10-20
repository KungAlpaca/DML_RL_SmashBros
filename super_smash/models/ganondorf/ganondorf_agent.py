"""Standalone Ganondorf agent (PPO, Stable-Baselines3) for Super Smash Bros. Melee via libmelee.

Drop-in usage in any libmelee match loop, from either port:

    agent = GanondorfAgent("model/ganondorf_edge_v4.zip", port=1, opponent_port=2)
    while True:
        gs = console.step()
        if gs is None:
            continue
        if gs.menu_state in (melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH):
            agent.act(gs, controller)   # call once per frame, before the next console.step()
        else:
            agent.reset()               # between games

The agent picks one of 31 controller inputs every 3 frames (20 decisions/s) from an 847-number
observation read from game memory. Everything here reproduces the training environment exactly
(feature scaling, left/right mirroring, opponent one-hot, button timing); do not change it.
"""

import json
import os

import melee
import numpy as np
from stable_baselines3 import PPO

B = melee.Button
# Stick positions (x, y) in libmelee's 0..1 range, 0.5 is neutral.
N, R, L, U, D = (0.5, 0.5), (1.0, 0.5), (0.0, 0.5), (0.5, 1.0), (0.5, 0.0)
UR, UL, DR, DL = (0.85, 0.85), (0.15, 0.85), (0.85, 0.15), (0.15, 0.15)
DIRS9 = [N, R, L, U, D, UR, UL, DR, DL]

# The 31 discrete actions: (main stick, button or None). The order is part of the trained network.
ACTIONS = (
    [(d, None) for d in DIRS9]                      # 0-8: move / crouch / fast-fall / drift
    + [(d, B.BUTTON_A) for d in DIRS9]              # 9-17: jab, tilts, smashes, aerials
    + [(d, B.BUTTON_B) for d in (N, R, L, U, D)]    # 18-22: Warlock Punch, Gerudo Dragon, Dark Dive, Wizard's Foot
    + [(d, B.BUTTON_X) for d in (N, R, L)]          # 23-25: jump (short hop), drifting
    + [(N, B.BUTTON_Z)]                             # 26: grab
    + [(d, B.BUTTON_L) for d in (N, R, L, D)]       # 27-30: shield, roll, spot dodge
)
# Action i with the stick flipped left <-> right.
MIRRORED_ACTION = [
    next(j for j, (st, bt) in enumerate(ACTIONS) if bt == button and round(st[0], 6) == round(1.0 - stick[0], 6)
         and round(st[1], 6) == round(stick[1], 6))
    for stick, button in ACTIONS
]
# Released on the last frame of each 3-frame action so they can be pressed again; shield stays held.
TAP_BUTTONS = (B.BUTTON_A, B.BUTTON_B, B.BUTTON_X, B.BUTTON_Z)
NUM_ANIMATIONS = 400
PLAYER_FEATURES = 20


def _action_id(player) -> int:
    value = getattr(player.action, "value", player.action)
    return int(value) if 0 <= int(value) < NUM_ANIMATIONS else NUM_ANIMATIONS - 1


class GanondorfAgent:
    """Plays Ganondorf on `port` against whoever is on `opponent_port`.

    Settings default to the ones stored next to the model (meta.json); they must match training.
    """

    def __init__(self, model_path: str, port: int, opponent_port: int, deterministic: bool = True, device: str = "cpu"):
        meta_path = os.path.join(os.path.dirname(os.path.abspath(model_path)), "meta.json")
        meta = json.load(open(meta_path)) if os.path.isfile(meta_path) else {}
        self.port, self.opponent_port = port, opponent_port
        self.deterministic = deterministic
        self.frame_skip = int(meta.get("frame_skip", 3))
        self.canonical_side = meta.get("canonical_side", "right")
        self.opponent_pool = [melee.Character[c] for c in meta.get("opponent", "FOX,MARTH,CPTFALCON").split(",")]
        self.stage = melee.Stage[meta.get("stage", "FINAL_DESTINATION")]
        self.edge = melee.stages.EDGE_GROUND_POSITION[self.stage]
        self.model = PPO.load(model_path, device=device)
        expected = 2 * (PLAYER_FEATURES + NUM_ANIMATIONS) + 4 + len(self.opponent_pool)
        if self.model.observation_space.shape != (expected,):
            raise ValueError(f"model expects {self.model.observation_space.shape}, settings give {expected}")
        self.reset()

    def reset(self):
        """Call between games."""
        self._frame = 0
        self._button = None

    # ---------------------------------------------------------------- observation
    def _player_features(self, p, mirror: bool) -> np.ndarray:
        f = np.zeros(PLAYER_FEATURES + NUM_ANIMATIONS, dtype=np.float32)
        sx = -1.0 if mirror else 1.0  # flips every left/right quantity
        f[:PLAYER_FEATURES] = (
            sx * p.position.x / 100.0,
            p.position.y / 100.0,
            p.percent / 100.0,
            p.stock / 4.0,
            sx * (1.0 if p.facing else -1.0),
            float(p.on_ground),
            float(p.off_stage),
            float(p.invulnerable),
            min(p.hitlag_left, 30) / 10.0,
            min(p.hitstun_frames_left, 120) / 30.0,
            p.jumps_left / 2.0,
            p.shield_strength / 60.0,
            sx * p.speed_air_x_self / 3.0,
            sx * p.speed_ground_x_self / 3.0,
            p.speed_y_self / 3.0,
            np.clip(sx * p.speed_x_attack / 5.0, -5, 5),
            np.clip(p.speed_y_attack / 5.0, -5, 5),
            min(p.action_frame, 120) / 60.0,
            (self.edge - abs(p.position.x)) / 100.0,
            float(p.iasa) if isinstance(p.iasa, (bool, int, float)) else 0.0,
        )
        f[PLAYER_FEATURES + _action_id(p)] = 1.0
        return f

    def observation(self, gs):
        """(847-float observation, mirrored?) for the current game state."""
        me, op = gs.players[self.port], gs.players[self.opponent_port]
        dx = (op.position.x - me.position.x) / 100.0
        dy = (op.position.y - me.position.y) / 100.0
        # The game is mirrored so the opponent always appears on the canonical side of the agent.
        mirror = (self.canonical_side == "left" and dx > 0) or (self.canonical_side == "right" and dx < 0)
        if mirror:
            dx = -dx
        glob = np.array([dx, dy, np.hypot(dx, dy), min(gs.frame, 28800) / 28800.0], dtype=np.float32)
        opp = np.zeros(len(self.opponent_pool), dtype=np.float32)
        if op.character in self.opponent_pool:  # other characters: all zeros (the agent never trained on them)
            opp[self.opponent_pool.index(op.character)] = 1.0
        obs = np.concatenate([self._player_features(me, mirror), self._player_features(op, mirror), glob, opp])
        return np.clip(obs, -10.0, 10.0).astype(np.float32), mirror

    # ---------------------------------------------------------------- acting
    def act(self, gs, controller):
        """Set `controller` for this frame. Call every in-game frame, before console.step()."""
        if self.port not in gs.players or self.opponent_port not in gs.players:
            return
        step = self._frame % self.frame_skip
        self._frame += 1
        if step == 0:
            obs, mirror = self.observation(gs)
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
            action = int(action)
            if mirror:  # the network saw a mirrored game, so flip its stick direction back
                action = MIRRORED_ACTION[action]
            stick, self._button = ACTIONS[action]
            controller.release_all()
            controller.tilt_analog(B.BUTTON_MAIN, *stick)
            if self._button is not None:
                controller.press_button(self._button)
        if step == self.frame_skip - 1 and self._button in TAP_BUTTONS and self.frame_skip > 1:
            controller.release_button(self._button)
