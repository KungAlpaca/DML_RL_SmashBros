"""Train MK5 (Fox), MK6 (random), MK9 (Fox) and MK10 (random) together; MK3 and MK4 are frozen opponents.

MK9 starts from MK5's weights and MK10 from MK6's, both with a human-like 200 ms reaction time
(--reaction-mk9 / --reaction-mk10): they keep everything MK5 / MK6 learned and only have to adapt
to seeing the game late. MK7 and MK8 (the same delay, from scratch) stalled at CPU level 4-5 after
~25 h and are kept only as a record. Same engine and games as league_mk5678.py:

    MK9 vs CPU, MK10 vs CPU     each with its own CPU level, from --mk9-cpu-start / --mk10-cpu-start
                                (7 by default) up to 9
    MK9 vs MK10, MK5 vs MK6     always
    4 more games                a new set every --rotate-every updates

Usage:
    python league_mk56910.py --mk3 runs/mk5678_d/MK3.zip --mk4 runs/mk5678_d/MK4.zip \\
        --mk5 runs/mk5678_d/MK5.zip --mk6 runs/mk5678_d/MK6.zip --hours 12 --run-name mk56910

--mk9 / --mk10 continue a saved MK9 / MK10 (pass the CPU levels it reached then).
"""

from league_mk5678 import Lineup, main

MK56910 = Lineup(
    description="Train MK5, MK6, MK9 and MK10 together against each other, frozen MK3 / MK4 and the CPU",
    characters={"MK3": "FOX", "MK4": "RANDOM", "MK5": "FOX", "MK6": "RANDOM", "MK9": "FOX", "MK10": "RANDOM",
                "CPU": "RANDOM"},
    frozen=("MK3", "MK4"),
    new={"MK9": "MK5", "MK10": "MK6"},
    curriculum={"MK9": 0, "MK10": 1},
    fixed=[("MK9", "CPU"), ("MK10", "CPU"), ("MK9", "MK10"), ("MK5", "MK6")],
    rotation=[
        [("MK5", "MK9"), ("MK6", "MK10"), ("MK5", "MK3"), ("MK6", "MK4")],
        [("MK5", "MK10"), ("MK6", "MK9"), ("MK9", "MK3"), ("MK10", "MK4")],
        [("MK5", "MK4"), ("MK6", "MK3"), ("MK5", "CPU"), ("MK6", "CPU")],
        [("MK9", "MK4"), ("MK10", "MK3"), ("MK9", "MK5"), ("MK10", "MK6")],
    ],
    lrs={"MK5": 3e-5, "MK6": 4e-5, "MK9": 1e-4, "MK10": 1e-4},
    cpu_start=7,
    run_name="mk56910",
)

if __name__ == "__main__":
    main(MK56910)
