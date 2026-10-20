# Kepiti's Fox (imitation learning, imitation + RL)

From [Kepitition/super_smash_RL](https://github.com/Kepitition/super_smash_RL), which started from this repo's
Dolphin environment.

| File | Approach | Reaction time | His results (Final Destination, 4 stocks) |
|---|---|---|---|
| `fox_imitation_18h.pt` | Imitation learning: an LSTM policy (9.9 M parameters) trained 18 h to predict the controller of master / diamond Fox players, 111,310 ranked games | 300 ms (18 frames), decides every frame | 95% vs CPU 9, 65% vs the Ganondorf agent |
| `fox_rl_3h.pt` | Imitation + RL: the model above after 3 h of PPO with a KL penalty that keeps it close to the human policy | 300 ms (18 frames) | 100% vs CPU 9, 95% vs Ganondorf, 62% vs the imitation model |

It uses the full controller (both sticks, analog shield, C-stick), so it wavedashes, L-cancels and shines
like a person. These files are only the weights: the code that runs them is in his repo, which has to be
cloned next to this one:

```bash
git clone https://github.com/Kepitition/super_smash_RL ~/Documents/DML/super_smash_RL
```

(or pass `--kepiti-repo PATH` to `party_teams.py`, or set `KEPITI_REPO=PATH`). His package is also called
`smash_rl`; `smash_rl/external_bots.py` loads it under another name so both work in one process. On a CPU it
takes about 6.5 ms per frame.

```bash
python party_teams.py --p1 none --p3 none --p4 human                                        # you vs imitation + RL
python party_teams.py --p1 none --p3 none --p4 human --p2 models/kepiti/fox_imitation_18h.pt # you vs imitation
python party_teams.py --p3 none --p4 none --autostart                                       # watch MK5 vs Kepiti
```
