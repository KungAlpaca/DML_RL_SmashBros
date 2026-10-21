"""Final training before the tournament: only MK5 (pure RL) and MK9 (RL + 200 ms reaction) learn, 1 vs 1.

Half of their games are against a frozen copy of themselves (MK5_OLD, MK9_OLD: the weights they
start from); the rest rotate through Kepiti's two Foxes (imitation, imitation + RL), the Ganondorf
agent, the level 9 CPU (random character) and other frozen MKs (MK3, MK6, MK10). Same engine as
league_mk5678.py; other teams' bots run inside the env (SmashEnv opponent_bot).

    MK5 vs MK5_OLD, MK9 vs MK9_OLD     two games each, always
    MK9 vs CPU                         its own CPU level, from --mk9-cpu-start (5) up to 9: one level up
                                       whenever it wins --promote-at of its last --promote-window games
    3 more games                       a new set every --rotate-every updates (ROTATION)

MK9 (200 ms reaction) lost almost every game against the instant opponents and stopped learning, so
it gets the CPU curriculum and, in the rotation, opponents it can beat sometimes (Kepiti's imitation
model, MK10, MK6); MK5 keeps the hard ones. Kepiti's bot runs an LSTM on the CPU every frame and the
8 games step together, so all his games are in one set: the others run at full speed.

Usage:
    python league_final.py --mk5 models/alejandro/MK5.zip --mk9 models/alejandro/MK9.zip \\
        --mk5_old models/alejandro/MK5.zip --mk9_old models/alejandro/MK9.zip \\
        --mk3 models/all_models/fox/MK3.zip --mk6 models/all_models/random/MK6.zip \\
        --mk10 models/all_models/random/MK10.zip --hours 8.5 --run-name final_mk59
"""

import os

from league_mk5678 import Lineup, main
from smash_rl.env import PROJECT_DIR

MODELS_DIR = os.path.join(PROJECT_DIR, "models")

FINAL = Lineup(
    description="Final training: MK5 and MK9 against old copies of themselves, Kepiti, Ganondorf, CPU 9 and other MKs",
    characters={"MK5": "FOX", "MK9": "FOX", "MK5_OLD": "FOX", "MK9_OLD": "FOX",
                "MK3": "FOX", "MK6": "RANDOM", "MK10": "RANDOM", "CPU": "RANDOM"},
    frozen=("MK5_OLD", "MK9_OLD", "MK3", "MK6", "MK10"),
    new={},
    curriculum={"MK9": 4},
    fixed=[("MK5", "MK5_OLD"), ("MK9", "MK9_OLD"), ("MK5", "MK5_OLD"), ("MK9", "MK9_OLD"), ("MK9", "CPU")],
    rotation=[
        [("MK5", "KEPITI_RL"), ("MK5", "KEPITI_IL"), ("MK9", "KEPITI_IL")],
        [("MK5", "GANONDORF"), ("MK5", "CPU"), ("MK9", "MK10")],
        [("MK5", "MK3"), ("MK5", "MK10"), ("MK9", "MK6")],
        [("MK5", "MK6"), ("MK5", "GANONDORF"), ("MK9", "MK10")],
    ],
    lrs={"MK5": 2e-5, "MK9": 5e-5},
    cpu_start=5,
    run_name="final_mk59",
    bots={
        "KEPITI_RL": ("kepiti", os.path.join(MODELS_DIR, "kepiti", "fox_rl_3h.pt")),
        "KEPITI_IL": ("kepiti", os.path.join(MODELS_DIR, "kepiti", "fox_imitation_18h.pt")),
        "GANONDORF": ("ganondorf", os.path.join(MODELS_DIR, "ganondorf", "ganondorf_edge_v4.zip")),
    },
)

if __name__ == "__main__":
    main(FINAL)
