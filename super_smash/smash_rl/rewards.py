"""Reward "v2" (MK3 / MK4): knockouts first, combos that connect, no free spam, no camping.

Per player and per frame (weights in RewardWeights, explained in docs/COMBOS.md):

    + ko x opponent stocks taken            - ko x own stocks lost
    + damage x (damage dealt - damage taken)
    + combo reward for each attack that connects (combos.ComboTracker)
    - whiff for each attack (A or special) that hits nothing; smash_whiff for a smash attack
    SPAM (every penalty below doubles with each extra repetition, up to spam_cap = one knockout):
    - smash_spam x 2^(n-2) for the n-th smash attack (forward / up / down smash) started within
      smash_window frames (n >= 2)
    - repeat x 2^(extra-1) each time a move is used more than repeat_max times in the player's
      last repeat_window actions (Fox's laser is allowed 6 of 7, see REPEAT_LIMITS)
    - passive per second without damaging anyone (after passive_grace), x opponents still alive;
      Fox and Falco laser hits don't count as damaging here (NO_PASSIVE_RESET)
    - defense_repeat x 2^(extra-1) each time a shield, roll, spot dodge or air dodge starts while
      more than defense_max of the player's last repeat_window actions were defences
    - shield_hold per second the shield stays up past shield_grace

Shielding and dodging earn nothing by themselves: their value is the damage they avoid.
Specials started off stage (recovery moves) are never counted as whiffs. A special counts as
connecting if the opponent is hurt within projectile_grace frames after it ends (lasers).
"""

from collections import deque
from dataclasses import dataclass, field

import melee
import numpy as np

from smash_rl.combos import (DSMASH, FSMASH, NEUTRAL_B, SIDE_B, STUNNED_ACTIONS, UP_B, USMASH, ComboTracker,
                             ComboWeights, FIRST_CHARACTER_ACTION, move_of)

A = melee.Action
# Defensive options, by the action state that starts them; they share the attacks' history.
DEFENCES = {A.SHIELD_START.value: "shield", A.ROLL_FORWARD.value: "roll", A.ROLL_BACKWARD.value: "roll",
            A.SPOTDODGE.value: "spotdodge", A.AIRDODGE.value: "airdodge"}
SHIELD_ACTIONS = frozenset(a.value for a in (A.SHIELD_START, A.SHIELD, A.SHIELD_STUN, A.SHIELD_REFLECT))
OTHER_PORT = {1: 2, 2: 1}
SMASHES = (FSMASH, USMASH, DSMASH)
# (character, move) -> (window, max uses in it) for moves that are fine to use a lot.
REPEAT_LIMITS = {(melee.Character.FOX, NEUTRAL_B): (7, 6)}  # Fox fires lasers constantly in real play
# Hits that count as damage but don't stop the passive penalty: camping with lasers is still camping.
NO_PASSIVE_RESET = {(melee.Character.FOX, NEUTRAL_B), (melee.Character.FALCO, NEUTRAL_B)}
STAT_KEYS = ("damage_dealt", "damage_taken", "kills", "deaths", "attacks", "whiffs", "repeats", "combo_hits",
             "combos", "defences", "defence_repeats", "smashes", "smash_spam", "spam_penalty")


@dataclass
class RewardWeights:
    # Everything except combos was scaled down (about a third) so that combos stand out;
    # a knockout is still the largest single reward.
    ko: float = 2.0
    damage: float = 0.003
    whiff: float = 0.03
    smash_whiff: float = 0.2  # a smash attack that hits nothing
    # Spam: each penalty doubles with every extra repetition, up to spam_cap (a knockout).
    spam_cap: float = 2.0
    smash_spam: float = 0.2  # 2nd smash within smash_window, then 0.4, 0.8, 1.6, 2.0
    smash_window: int = 240  # frames (4 s)
    repeat: float = 0.1  # 1st repetition over the limit, then 0.2, 0.4 ...
    repeat_window: int = 6
    repeat_max: int = 3
    defense_repeat: float = 0.1
    defense_max: int = 3  # defences allowed in the last repeat_window actions
    shield_hold: float = 0.05  # per second
    shield_grace: int = 60  # frames
    passive: float = 0.007  # per second without dealing damage
    passive_grace: int = 120  # frames
    projectile_grace: int = 40  # frames
    b_press_window: int = 10  # a character-specific action counts as a special only this soon after pressing B
    combo: ComboWeights = field(default_factory=ComboWeights)


class _Attack:
    def __init__(self, action, move, frame, recovery):
        self.action, self.move, self.frame, self.recovery = action, move, frame, recovery
        self.hit = False  # touched the opponent or their shield
        self.combo_counted = False  # one combo step per attack, even for multi-hit moves
        self.deadline = None  # for specials that ended without hitting: when they become whiffs


class RewardTracker:
    """Keeps the per-player state reward v2 needs during one game."""

    def __init__(self, gs, weights: RewardWeights):
        self.w = weights
        self.stats = {p: dict.fromkeys(STAT_KEYS, 0) for p in (1, 2)}
        self.combos = {p: ComboTracker(gs.players[p].character, weights.combo) for p in (1, 2)}
        self.attack = {1: None, 2: None}
        self.pending = {1: [], 2: []}  # specials that ended without hitting, maybe a projectile still flying
        window = max([weights.repeat_window] + [w for w, _ in REPEAT_LIMITS.values()])
        self.recent = {p: deque(maxlen=window) for p in (1, 2)}  # the player's last attacks and defences
        self.shield_since = {1: None, 2: None}
        self.smash_frames = {1: deque(), 2: deque()}  # when each player's recent smash attacks started
        self.character = {p: gs.players[p].character for p in (1, 2)}
        self.b_direction = {1: None, 2: None}
        self.b_pressed_at = {1: -10**9, 2: -10**9}
        self.last_frame = gs.frame
        self.last_damage = {p: gs.frame for p in (1, 2)}
        self.free_since = {p: gs.frame for p in (1, 2)}  # first frame each player stopped being stunned
        self.prev_action = {p: None for p in (1, 2)}
        self.prev = ({p: float(gs.players[p].percent) for p in (1, 2)}, {p: int(gs.players[p].stock) for p in (1, 2)})

    def _spam(self, base: float, extra: int) -> float:
        """A spam penalty: base for the first repetition over the limit, doubling with each one after."""
        return min(self.w.spam_cap, base * 2 ** (extra - 1))

    def set_input(self, port: int, stick, button):
        """The stick + button this player is pressing; tells which special a B press starts."""
        if button == melee.Button.BUTTON_B:
            self.b_pressed_at[port] = self.last_frame
            x, y = stick
            self.b_direction[port] = ("up" if y > 0.7 else "down" if y < 0.3 else
                                      "side" if abs(x - 0.5) > 0.2 else "neutral")

    def frame(self, gs) -> np.ndarray:
        pct = {p: float(gs.players[p].percent) for p in (1, 2)}
        stock = {p: int(gs.players[p].stock) for p in (1, 2)}
        prev_pct, prev_stock = self.prev
        self.prev = (pct, stock)
        deaths = {p: max(0, prev_stock[p] - stock[p]) for p in (1, 2)}
        taken = {p: 0.0 if deaths[p] else max(0.0, pct[p] - prev_pct[p]) for p in (1, 2)}
        action = {p: int(getattr(gs.players[p].action, "value", -1)) for p in (1, 2)}
        rewards = np.zeros(2, dtype=np.float32)
        w = self.w
        self.last_frame = gs.frame

        for p in (1, 2):
            o = OTHER_PORT[p]
            me = gs.players[p]
            r = w.ko * (deaths[o] - deaths[p]) + w.damage * (taken[o] - taken[p])
            stats = self.stats[p]
            stats["damage_dealt"] += taken[o]
            stats["damage_taken"] += taken[p]
            stats["kills"] += deaths[o]
            stats["deaths"] += deaths[p]

            # The attack this player is doing: end the old one, start a new one.
            attack = self.attack[p]
            if attack is not None and attack.action != action[p]:
                if not attack.hit:
                    attack.deadline = gs.frame + (w.projectile_grace if attack.action >= FIRST_CHARACTER_ACTION else 0)
                    self.pending[p].append(attack)
                attack = None
            # Character-specific states are also extra jumps, floats etc.: only a B press makes them a special.
            pressed_b = gs.frame - self.b_pressed_at[p] <= w.b_press_window
            move = move_of(action[p], self.b_direction[p] if pressed_b else None)
            if move is not None and attack is None:
                recovery = move in (UP_B, SIDE_B) and bool(me.off_stage)
                attack = _Attack(action[p], move, gs.frame, recovery)
                stats["attacks"] += 1
                self.recent[p].append(move)
                window, limit = REPEAT_LIMITS.get((self.character[p], move), (w.repeat_window, w.repeat_max))
                extra = list(self.recent[p])[-window:].count(move) - limit
                if extra > 0:
                    penalty = self._spam(w.repeat, extra)
                    r -= penalty
                    stats["repeats"] += 1
                    stats["spam_penalty"] += penalty
                if move in SMASHES:
                    stats["smashes"] += 1
                    frames = self.smash_frames[p]
                    frames.append(gs.frame)
                    while gs.frame - frames[0] > w.smash_window:
                        frames.popleft()
                    if len(frames) >= 2:
                        penalty = self._spam(w.smash_spam, len(frames) - 1)
                        r -= penalty
                        stats["smash_spam"] += 1
                        stats["spam_penalty"] += penalty
            self.attack[p] = attack

            # Did it connect?
            hurt = taken[o] > 0 or deaths[o] > 0
            source = attack if attack is not None else (self.pending[p][-1] if self.pending[p] else None)
            if source is not None and (hurt or (action[o] == A.SHIELD_STUN.value and attack is not None)):
                source.hit = True
                if source in self.pending[p]:
                    self.pending[p].remove(source)
                if hurt and not source.combo_counted:
                    source.combo_counted = True
                    combo_reward, step, completed = self.combos[p].on_hit(source.move, gs.frame, self.free_since[o])
                    r += combo_reward
                    stats["combo_hits"] += int(step >= 2)
                    stats["combos"] += len(completed)
            if hurt and (source is None or (self.character[p], source.move) not in NO_PASSIVE_RESET):
                self.last_damage[p] = gs.frame

            # Attacks that are over and never touched anything.
            for missed in [a for a in self.pending[p] if gs.frame >= a.deadline]:
                self.pending[p].remove(missed)
                if missed.recovery:
                    continue
                r -= w.smash_whiff if missed.move in SMASHES else w.whiff
                stats["whiffs"] += 1
                self.combos[p].reset()

            # Defence: no reward of its own, only a cost when overused or held too long.
            defence = DEFENCES.get(action[p]) if action[p] != self.prev_action[p] else None
            if defence is not None:
                stats["defences"] += 1
                self.recent[p].append(defence)
            last = list(self.recent[p])[-w.repeat_window:]
            extra = sum(m in DEFENCES.values() for m in last) - w.defense_max
            if defence is not None and extra > 0:
                penalty = self._spam(w.defense_repeat, extra)
                r -= penalty
                stats["defence_repeats"] += 1
                stats["spam_penalty"] += penalty
            if action[p] in SHIELD_ACTIONS:
                if self.shield_since[p] is None:
                    self.shield_since[p] = gs.frame
                if gs.frame - self.shield_since[p] > w.shield_grace:
                    r -= w.shield_hold / 60.0
            else:
                self.shield_since[p] = None

            # Camping: nothing landed for a while, while there are still opponents to hit.
            if gs.frame - self.last_damage[p] > w.passive_grace:
                r -= w.passive / 60.0 * (1 if stock[o] > 0 else 0)
            rewards[p - 1] += r

        for p in (1, 2):
            if action[p] in STUNNED_ACTIONS:
                self.free_since[p] = None
            elif self.free_since[p] is None:
                self.free_since[p] = gs.frame
            self.prev_action[p] = action[p]
        return rewards
