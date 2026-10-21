# Trained models

Final weights only (no checkpoints), in four folders:

| Folder | What | Character | Slot in `party_teams.py` |
|---|---|---|---|
| [`alejandro/`](alejandro/README.md) | Our two picks: **MK5** (best Fox, pure RL) and **MK9** (Fox with a human 200 ms reaction time) | Fox | P1 |
| [`kepiti/`](kepiti/README.md) | Kepiti's Fox: **imitation learning** on human replays, and **imitation + RL** | Fox | P2 |
| [`ganondorf/`](ganondorf/USAGE.md) | A friend's **Ganondorf** PPO agent, with its own code (`ganondorf_agent.py`, `match.py`) | Ganondorf | P3 |
| [`all_models/`](all_models/README.md) | Every model we trained (M1, MK0-MK10), split into `fox/` and `random/` | Fox / random | - |

The project compares four ways of training a Fox:

| | Approach | Model | Reaction time |
|---|---|---|---|
| 1 | Pure RL (PPO, reward shaping, league of opponents) | `alejandro/MK5.zip` | none (decides every 3 frames) |
| 2 | RL with a human-like reaction delay | `alejandro/MK9.zip` | 200 ms |
| 3 | Imitation learning on 111k human replays | `kepiti/fox_imitation_18h.pt` | 300 ms (18 frames) |
| 4 | Imitation + RL (PPO with a KL leash to the imitation policy) | `kepiti/fox_rl_3h.pt` | 300 ms (18 frames) |

## Every model

Every model here decides every 3 frames (20 times per second); MK5 and MK9 continue training at 2 frames
(30 per second). The rate is saved in each model (`frame_skip`) and all scripts use it. "Reaction" is how
late the model sees the game (saved inside our `.zip` files as `reaction_ms`; `play.py`,
`party.py`, `party_teams.py` and `tournament.py` apply it). Steps count every game a model learned from,
across its parallel games. Hours are wall-clock training time (≈: summed over several runs).

| Model | File | Plays | Reaction | How it was trained | Steps | Notes |
|---|---|---|---|---|---|---|
| **M1** | `all_models/fox/M1.zip` | Fox | 0 | `train.py`, reward v1, CPU curriculum 1 → 9, 1 h 50 min | 4.6 M | Beats CPU 9 (20/20), but 64% of its attacks hit nothing |
| **MK0** | `all_models/fox/MK0.zip` | Fox | 0 | `train.py`, reward v1 + whiff penalty, CPU curriculum until 90% of 50 games at level 9 | 2.7 M | |
| **MK1** | `all_models/fox/MK1.zip` | Fox | 0 | `selfplay.py` against MK2, from MK0 | 3.7 M | Passive: lasers all game |
| **MK2** | `all_models/random/MK2.zip` | random | 0 | `selfplay.py` against MK1, from MK0 | 3.7 M | Passive: stands still or dodges |
| **MK3** | `all_models/fox/MK3.zip` | Fox | 0 | `league.py`, reward v2 (combos, anti-spam), from scratch, ≈ 74 h; frozen since Oct 15 | 85.0 M | Strong, but the most smash-attack spam |
| **MK3_live** | `all_models/fox/MK3_live.zip` | Fox | 0 | MK3 (Oct 10) fine-tuned by playing a person (`play.py --learn`) | 13.9 M | |
| **MK4** | `all_models/random/MK4.zip` | random | 0 | `league.py`, reward v2, from scratch ≈ 62 h; collapsed, restarted from MK6's weights (2.6 h); frozen since Oct 15 | 61.7 M | Weakest of the league |
| **MK4_live** | `all_models/random/MK4_live.zip` | random | 0 | MK4 (Oct 10) fine-tuned by playing a person | 13.9 M | |
| **MK5** | `all_models/fox/MK5.zip` (= `alejandro/MK5.zip`) | Fox | 0 | `league.py` → `league_mk56910.py`, reward v2 without shield / dodge rewards, from scratch, ≈ 91 h | 89.8 M | **Best Fox**: 100% vs CPU 9 and every other model it met |
| **MK6** | `all_models/random/MK6.zip` | random | 0 | From MK5's weights, CPU curriculum 1 → 9, then the league, ≈ 82 h | 99.5 M | Best random-character model, ~70-80% vs CPU 9 |
| **MK7** | `all_models/fox/MK7.zip` | Fox | 200 ms | `league_mk5678.py`, from scratch with the delay, ≈ 23 h | 22.0 M | Stalled at CPU level 5 |
| **MK8** | `all_models/random/MK8.zip` | random | 200 ms | Same as MK7, ≈ 23 h | 22.0 M | Stalled at CPU level 4 |
| **MK9** | `all_models/fox/MK9.zip` (= `alejandro/MK9.zip`) | Fox | 200 ms | `league_mk56910.py`, from MK5's weights with the delay, 12 h | 11.7 M | Reached CPU level 8 |
| **MK10** | `all_models/random/MK10.zip` | random | 100 ms | From MK6's weights with a 100 ms delay (third try), 4.5 h | 4.4 M | Reached CPU level 5 |
| MK10_try1/2_200ms_failed | `all_models/random/` | random | 200 ms | From MK6's weights with 200 ms, 3.75 h each | 3.7 M | Got worse while training (kept as a record) |
| Kepiti imitation | `kepiti/fox_imitation_18h.pt` | Fox | 300 ms | Behaviour cloning, 18 h on 1.24 billion frames of master / diamond Fox players | - | 95% vs CPU 9, 65% vs Ganondorf (his numbers) |
| Kepiti imitation + RL | `kepiti/fox_rl_3h.pt` | Fox | 300 ms | The imitation model + 3 h of PPO with a KL leash | - | 100% vs CPU 9, 95% vs Ganondorf, 62% vs his imitation model (his numbers) |
| Ganondorf | `ganondorf/ganondorf_edge_v4.zip` | Ganondorf | 0 | PPO (Stable-Baselines3) vs level 9 CPUs and ledge-camping bots | 21.6 M | 9/9 vs CPU 9 Fox / Marth / Falcon, +3.22 stocks (its numbers) |

Reward v1: damage and stocks. Reward v2: knockouts, per-character combos, penalties for whiffs, spam
and camping ([../docs/COMBOS.md](../docs/COMBOS.md)). Models trained before the 200 ms experiments read
the game instantly, like the Ganondorf agent.

### Preliminary Fox round robin (Oct 20)

`tournament.py`, every model as Fox, Final Destination, 4 stocks, sampled actions; stopped after 367
of 720 games, so not every pair was played. Win rate of the row model against the column:

| | MK3 | MK4 | MK5 | MK6 | MK7 | MK8 | MK9 | MK10 | CPU 9 Fox |
|---|---|---|---|---|---|---|---|---|---|
| **MK5** | 100% | 100% | - | | | | | | 100% |
| **MK3** | - | 100% | 0% | 55% | 100% | 100% | 100% | 100% | 100% |
| **MK6** | 45% | 95% | | | | | | | 100% |
| MK7 | 0% | 80% | | | | | | | 0% |
| MK4 | 0% | - | 0% | 5% | 20% | 80% | 55% | 44% | 25% |
| MK10 | 0% | 56% | | | | | | | |
| MK9 | 0% | 45% | | | | | | | |
| MK8 | 0% | 20% | | | | | | | 0% |

The final tournament (all four approaches, Kepiti's models and Ganondorf included) is still to come.

## Demos

From `super_smash/` with `source .venv/bin/activate`. Kepiti's models need his repo next to this one:
`git clone https://github.com/Kepitition/super_smash_RL ~/Documents/DML/super_smash_RL` (or pass
`--kepiti-repo` / set `KEPITI_REPO`). His code runs on our `.venv` (PyTorch on the CPU is enough).

**You against one model** (you are port 2 in `play.py`, port 4 in `party_teams.py`; add `--input keyboard` for the keyboard):

```bash
python play.py models/alejandro/MK5.zip                                   # best Fox, instant
python play.py models/alejandro/MK9.zip                                   # Fox with a 200 ms reaction time
python play.py models/all_models/random/MK6.zip --character RANDOM        # any of ours; RANDOM = new character every game
python party_teams.py --p1 none --p3 none --p4 human                      # Kepiti imitation + RL
python party_teams.py --p1 none --p3 none --p4 human --p2 models/kepiti/fox_imitation_18h.pt   # Kepiti imitation
python party_teams.py --p1 none --p2 none --p4 human                      # Ganondorf
```

**Everyone at once** (P1 ours, P2 Kepiti, P3 Ganondorf, P4 you / a CPU / nobody):

```bash
python party_teams.py --p4 human                                          # you vs MK5 + Kepiti + Ganondorf
python party_teams.py --p1 models/alejandro/MK9.zip --p4 human            # with the human-speed MK9
python party_teams.py --p4 cpu --cpu-level 9 --autostart                  # watch: three bots + CPU 9
python party_teams.py --p3 none --p4 none --autostart                     # watch: MK5 vs Kepiti
python party_teams.py --p2 none --p4 none --autostart                     # watch: MK5 vs Ganondorf
python party_teams.py --p1 none --p4 none --autostart                     # watch: Kepiti vs Ganondorf
python party.py                                                           # you vs two of ours (MK5 + MK6)
```

**Bots against bots, headless, just the numbers:**

```bash
python tournament.py MK5=models/alejandro/MK5.zip MK9=models/alejandro/MK9.zip \
    MK3=models/all_models/fox/MK3.zip --games 20 --cpu-games 20       # round robin + CPU 9
python vs_friend.py MK5=models/alejandro/MK5.zip MK9=models/alejandro/MK9.zip --games 10   # vs Ganondorf
```
