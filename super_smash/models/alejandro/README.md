# Our picks: MK5 and MK9 (Fox)

| File | Approach | Reaction time | Trained | Result so far |
|---|---|---|---|---|
| `MK5.zip` | Pure RL: PPO with reward v2 (knockouts, combos, anti-spam), league of past models and CPUs | none: reads the game and decides every 3 frames (50 ms) | ≈ 91 h, 89.8 M steps, from scratch | 100% vs CPU level 9 and vs every model it met in the Fox round robin |
| `MK9.zip` | RL with a human-like reaction delay: MK5's weights, then trained seeing the game 200 ms late | 200 ms (saved in the file) | 12 h on top of MK5, 11.7 M steps | Reached CPU level 8; loses to the instant models |

Both are copies of the files in `../all_models/fox/` (git stores them once). They play Fox on Final
Destination, the only stage they know. Their final training runs in `../../league_final.py`.

```bash
python play.py models/alejandro/MK5.zip                        # you (port 2) vs MK5; --input keyboard
python play.py models/alejandro/MK9.zip
python play.py models/alejandro/MK5.zip --reaction-ms 200      # MK5 forced to react 200 ms late
python party_teams.py --p1 models/alejandro/MK9.zip --p4 human # P1 in the team free-for-all
```
