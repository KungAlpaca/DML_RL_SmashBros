"""Per-character Melee combo lists and the tracker that rewards them (reward="v2", see rewards.py).

A combo is a sequence of moves. Every hit that connects and continues a listed combo of
the attacker's character is worth more than the last (step k of a combo pays k units);
a hit that only keeps the opponent stunned (a "true combo" hit, as the game's combo
counter sees it) pays a smaller, also growing amount; a hit outside any chain pays a
little. A move that misses pays nothing and breaks the chain. docs/COMBOS.md explains
the lists, their sources and how detection works.
"""

from dataclasses import dataclass

import melee

A = melee.Action
C = melee.Character

# Moves as combo steps. Specials are told apart by the direction of the B input that started them.
JAB, DASH, FTILT, UTILT, DTILT = "jab", "dash_attack", "ftilt", "utilt", "dtilt"
FSMASH, USMASH, DSMASH = "fsmash", "usmash", "dsmash"
NAIR, FAIR, BAIR, UAIR, DAIR = "nair", "fair", "bair", "uair", "dair"
PUMMEL, FTHROW, BTHROW, UTHROW, DTHROW = "pummel", "fthrow", "bthrow", "uthrow", "dthrow"
NEUTRAL_B, SIDE_B, UP_B, DOWN_B = "neutral_b", "side_b", "up_b", "down_b"

_ACTION_MOVES = {
    **{a: JAB for a in (A.NEUTRAL_ATTACK_1, A.NEUTRAL_ATTACK_2, A.NEUTRAL_ATTACK_3,
                        A.LOOPING_ATTACK_START, A.LOOPING_ATTACK_MIDDLE, A.LOOPING_ATTACK_END)},
    A.DASH_ATTACK: DASH,
    **{a: FTILT for a in (A.FTILT_HIGH, A.FTILT_HIGH_MID, A.FTILT_MID, A.FTILT_LOW_MID, A.FTILT_LOW)},
    A.UPTILT: UTILT, A.DOWNTILT: DTILT,
    **{a: FSMASH for a in (A.FSMASH_HIGH, A.FSMASH_MID_HIGH, A.FSMASH_MID, A.FSMASH_MID_LOW, A.FSMASH_LOW)},
    A.UPSMASH: USMASH, A.DOWNSMASH: DSMASH,
    A.NAIR: NAIR, A.FAIR: FAIR, A.BAIR: BAIR, A.UAIR: UAIR, A.DAIR: DAIR,
    A.GRAB_PUMMEL: PUMMEL,
    A.THROW_FORWARD: FTHROW, A.THROW_BACK: BTHROW, A.THROW_UP: UTHROW, A.THROW_DOWN: DTHROW,
}
_ACTION_MOVES = {action.value: move for action, move in _ACTION_MOVES.items()}
FIRST_CHARACTER_ACTION = 0x155  # ids from here on are character specific: mostly special moves
# Being hit, thrown or held: while the opponent is in one of these (or just left it), hits chain.
_STUNNED_PREFIXES = ("DAMAGE", "THROWN", "TUMBLING", "GRABBED", "GRAB_PULL", "GRAB_PUMMELED")
STUNNED_ACTIONS = frozenset(a.value for a in A if a.name.startswith(_STUNNED_PREFIXES))
B_DIRECTION_MOVES = {"neutral": NEUTRAL_B, "side": SIDE_B, "up": UP_B, "down": DOWN_B}


def move_of(action_value: int, b_direction: str = None):
    """The combo step an action state belongs to, or None if it isn't an attack."""
    if action_value in _ACTION_MOVES:
        return _ACTION_MOVES[action_value]
    if action_value >= FIRST_CHARACTER_ACTION and action_value not in STUNNED_ACTIONS:
        return B_DIRECTION_MOVES.get(b_direction)
    return None


@dataclass(frozen=True)
class Combo:
    name: str
    moves: tuple
    source: str  # where it comes from; "community" = common knowledge, not checked against a source


SW_COMBO = "SmashWiki: Combo"
SW_CHAIN = "SmashWiki: Chain grab"
SW_UTHROW = "SmashWiki: Fox (SSBM)/Up throw"
COMMUNITY = "community"

COMBOS = {
    C.FOX: [
        Combo("up throw -> up air", (UTHROW, UAIR), SW_UTHROW),
        Combo("up throw chain grab", (UTHROW, UTHROW), SW_CHAIN),
        Combo("up throw -> up smash", (UTHROW, USMASH), SW_CHAIN),
        Combo("multishine", (DOWN_B, DOWN_B), SW_COMBO),
        Combo("shine -> up smash", (DOWN_B, USMASH), SW_COMBO),
        Combo("drill -> shine -> up smash", (DAIR, DOWN_B, USMASH), COMMUNITY),
        Combo("up tilt juggle -> up air", (UTILT, UTILT, UAIR), COMMUNITY),
        Combo("nair -> shine", (NAIR, DOWN_B), COMMUNITY),
    ],
    C.FALCO: [
        Combo("pillar", (DOWN_B, DAIR, DOWN_B, DAIR), SW_COMBO),
        Combo("shine -> back air", (DOWN_B, BAIR), SW_COMBO),
        Combo("shine -> up tilt", (DOWN_B, UTILT), SW_COMBO),
        Combo("up throw chain grab", (UTHROW, UTHROW), SW_CHAIN),
        Combo("up throw -> up air", (UTHROW, UAIR), COMMUNITY),
        Combo("up tilt -> up tilt", (UTILT, UTILT), COMMUNITY),
    ],
    C.CPTFALCON: [
        Combo("sacred combo (knee -> falcon punch)", (FAIR, NEUTRAL_B), SW_COMBO),
        Combo("up throw -> knee", (UTHROW, FAIR), COMMUNITY),
        Combo("down throw -> knee", (DTHROW, FAIR), COMMUNITY),
        Combo("nair -> knee", (NAIR, FAIR), COMMUNITY),
        Combo("stomp -> knee", (DAIR, FAIR), COMMUNITY),
        Combo("up air -> up air", (UAIR, UAIR), COMMUNITY),
    ],
    C.MARTH: [
        Combo("Ken combo", (FAIR, FAIR, DAIR), SW_COMBO),
        Combo("long Ken combo", (FAIR, FAIR, FAIR, DAIR), SW_COMBO),
        Combo("up throw chain grab", (UTHROW, UTHROW), SW_CHAIN),
        Combo("up throw -> up air", (UTHROW, UAIR), "SmashWiki: Chain grab (short hop up air follow-up)"),
        Combo("forward throw -> forward air", (FTHROW, FAIR), COMMUNITY),
        Combo("down throw -> forward air", (DTHROW, FAIR), COMMUNITY),
    ],
    C.SHEIK: [
        Combo("down throw chain grab", (DTHROW, DTHROW), SW_CHAIN),
        Combo("down throw -> up smash", (DTHROW, USMASH), SW_COMBO),
        Combo("down throw -> forward air", (DTHROW, FAIR), SW_COMBO),
        Combo("forward tilt -> forward tilt", (FTILT, FTILT), COMMUNITY),
        Combo("forward tilt -> forward air", (FTILT, FAIR), COMMUNITY),
    ],
    C.JIGGLYPUFF: [
        Combo("wall of pain", (BAIR, BAIR, BAIR), SW_COMBO),
        Combo("space animal slayer", (UTHROW, UAIR, DOWN_B), SW_COMBO),
        Combo("down air -> rest", (DAIR, DOWN_B), COMMUNITY),
        Combo("up air -> rest", (UAIR, DOWN_B), COMMUNITY),
    ],
    C.PEACH: [
        Combo("up throw chain grab", (UTHROW, UTHROW), SW_CHAIN),
        Combo("down throw chain grab", (DTHROW, DTHROW), SW_CHAIN),
        Combo("up throw -> up air", (UTHROW, UAIR), COMMUNITY),
        Combo("nair -> down smash", (NAIR, DSMASH), COMMUNITY),
        Combo("down throw -> forward air", (DTHROW, FAIR), COMMUNITY),
    ],
    C.MEWTWO: [
        Combo("down throw chain grab", (DTHROW, DTHROW), SW_CHAIN),
        Combo("down throw -> forward air", (DTHROW, FAIR), COMMUNITY),
        Combo("nair -> nair", (NAIR, NAIR), COMMUNITY),
    ],
    C.DK: [
        Combo("up throw chain grab", (UTHROW, UTHROW), SW_CHAIN),
        Combo("up throw -> up air", (UTHROW, UAIR), COMMUNITY),
        Combo("down tilt -> up air", (DTILT, UAIR), COMMUNITY),
    ],
    C.MARIO: [
        Combo("up throw chain grab", (UTHROW, UTHROW), SW_CHAIN),
        Combo("down throw chain grab", (DTHROW, DTHROW), SW_CHAIN),
        Combo("up throw -> up air", (UTHROW, UAIR), COMMUNITY),
        Combo("up tilt -> up tilt -> up air", (UTILT, UTILT, UAIR), COMMUNITY),
        Combo("down throw -> forward air", (DTHROW, FAIR), COMMUNITY),
    ],
    C.DOC: [
        Combo("up throw chain grab", (UTHROW, UTHROW), SW_CHAIN),
        Combo("down throw chain grab", (DTHROW, DTHROW), SW_CHAIN),
        Combo("up throw -> up air", (UTHROW, UAIR), COMMUNITY),
        Combo("up tilt -> up tilt -> up air", (UTILT, UTILT, UAIR), COMMUNITY),
        Combo("down throw -> forward air", (DTHROW, FAIR), COMMUNITY),
    ],
    C.LUIGI: [
        Combo("up tilt -> up tilt -> up air", (UTILT, UTILT, UAIR), COMMUNITY),
        Combo("down throw -> up tilt", (DTHROW, UTILT), COMMUNITY),
        Combo("nair -> up smash", (NAIR, USMASH), COMMUNITY),
    ],
    C.POPO: [
        Combo("down throw chain grab", (DTHROW, DTHROW), SW_CHAIN),
        Combo("down throw -> forward air", (DTHROW, FAIR), COMMUNITY),
        Combo("up air -> up air", (UAIR, UAIR), COMMUNITY),
    ],
    C.ROY: [
        Combo("up throw chain grab", (UTHROW, UTHROW), SW_CHAIN),
        Combo("down throw -> forward air", (DTHROW, FAIR), COMMUNITY),
        Combo("forward air -> forward air", (FAIR, FAIR), COMMUNITY),
    ],
    C.PIKACHU: [
        Combo("up throw -> up air", (UTHROW, UAIR), COMMUNITY),
        Combo("up tilt -> up tilt", (UTILT, UTILT), COMMUNITY),
        Combo("nair -> up air", (NAIR, UAIR), COMMUNITY),
        Combo("up air -> up air", (UAIR, UAIR), COMMUNITY),
    ],
    C.PICHU: [
        Combo("up tilt chain -> up air", (UTILT, UTILT, UTILT, UAIR), COMMUNITY),
        Combo("up air -> up air", (UAIR, UAIR), COMMUNITY),
    ],
    C.SAMUS: [
        Combo("up throw -> up air", (UTHROW, UAIR), COMMUNITY),
        Combo("down throw -> up air", (DTHROW, UAIR), COMMUNITY),
        Combo("nair -> up air", (NAIR, UAIR), COMMUNITY),
    ],
    C.GANONDORF: [
        Combo("down throw chain grab", (DTHROW, DTHROW), "Smashboards: The Ganon Chaingrabbing Guide"),
        Combo("down throw -> forward air", (DTHROW, FAIR), COMMUNITY),
        Combo("up air -> up air", (UAIR, UAIR), COMMUNITY),
        Combo("stomp -> forward air", (DAIR, FAIR), COMMUNITY),
    ],
    C.LINK: [
        Combo("up throw -> up air", (UTHROW, UAIR), COMMUNITY),
        Combo("down throw -> up air", (DTHROW, UAIR), COMMUNITY),
        Combo("nair -> up air", (NAIR, UAIR), COMMUNITY),
    ],
    C.YLINK: [
        Combo("down throw -> nair", (DTHROW, NAIR), COMMUNITY),
        Combo("up throw -> up air", (UTHROW, UAIR), COMMUNITY),
        Combo("nair -> up air", (NAIR, UAIR), COMMUNITY),
    ],
    C.NESS: [
        Combo("up throw -> up air", (UTHROW, UAIR), COMMUNITY),
        Combo("down throw -> forward air", (DTHROW, FAIR), COMMUNITY),
        Combo("nair -> nair", (NAIR, NAIR), COMMUNITY),
    ],
    C.KIRBY: [
        Combo("down throw -> up air", (DTHROW, UAIR), COMMUNITY),
        Combo("up tilt -> up tilt", (UTILT, UTILT), COMMUNITY),
        Combo("forward air -> forward air", (FAIR, FAIR), COMMUNITY),
    ],
    C.BOWSER: [
        Combo("up throw -> up air", (UTHROW, UAIR), COMMUNITY),
        Combo("down throw -> up air", (DTHROW, UAIR), COMMUNITY),
    ],
    C.YOSHI: [
        Combo("up air -> up air", (UAIR, UAIR), COMMUNITY),
        Combo("up throw -> up air", (UTHROW, UAIR), COMMUNITY),
        Combo("nair -> up tilt", (NAIR, UTILT), COMMUNITY),
    ],
    C.ZELDA: [
        Combo("down throw -> forward air", (DTHROW, FAIR), COMMUNITY),
        Combo("up throw -> up air", (UTHROW, UAIR), COMMUNITY),
    ],
    C.GAMEANDWATCH: [
        Combo("down throw -> up air", (DTHROW, UAIR), COMMUNITY),
        Combo("nair -> up air", (NAIR, UAIR), COMMUNITY),
        Combo("up tilt -> up tilt", (UTILT, UTILT), COMMUNITY),
    ],
}
COMBOS[C.NANA] = COMBOS[C.POPO]


@dataclass
class ComboWeights:
    listed_step: float = 0.1  # step k >= 2 of a listed combo pays k x this (step 1 pays like any other hit)
    chain_step: float = 0.03  # hit k of an unlisted true-combo chain pays (k - 1) x this
    single_hit: float = 0.003  # a hit that starts nothing (scaled down with the other non-combo terms)
    chain_grace: int = 15  # frames the opponent may be out of stun before the chain breaks
    max_gap: int = 120  # frames between two hits after which they never chain
    max_chain_steps: int = 5  # the chain reward stops growing here (no farming endless chains)


class ComboTracker:
    """Follows one attacker's hits on one opponent and turns them into combo rewards."""

    def __init__(self, character, weights: ComboWeights):
        self.combos = COMBOS.get(character, [])
        self.w = weights
        self.reset()

    def reset(self):
        self.progress = [0] * len(self.combos)  # how many steps of each listed combo are done
        self.chain = 0  # hits in the current true-combo chain
        self.last_hit_frame = None

    def on_hit(self, move, frame: int, opponent_free_since: int):
        """A hit by `move` landed. opponent_free_since is the first frame the opponent was no
        longer stunned (None if it still is). Returns (reward, longest listed combo step, completed combo names)."""
        if move == PUMMEL:  # pummels repeat for as long as the grab lasts: a plain hit, chain untouched
            self.last_hit_frame = frame
            return self.w.single_hit, 0, []
        chained = (self.last_hit_frame is not None and frame - self.last_hit_frame <= self.w.max_gap
                   and (opponent_free_since is None or frame - opponent_free_since <= self.w.chain_grace))
        self.chain = self.chain + 1 if chained else 1
        self.last_hit_frame = frame
        best, completed = 0, []
        for i, combo in enumerate(self.combos):
            done = self.progress[i]
            if chained and 0 < done < len(combo.moves) and combo.moves[done] == move:
                done += 1
            elif combo.moves[0] == move:
                done = 1
            else:
                done = 0
            if done == len(combo.moves):
                completed.append(combo.name)
                best = max(best, done)
                done = 0
            self.progress[i] = done
            best = max(best, done)
        if best >= 2:  # continuing a listed combo; just starting one pays like any other hit
            reward = self.w.listed_step * best
        elif self.chain > 1:
            reward = self.w.chain_step * min(self.chain - 1, self.w.max_chain_steps)
        else:
            reward = self.w.single_hit
        return reward, best, completed
