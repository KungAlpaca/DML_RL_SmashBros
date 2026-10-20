# Every model we trained

Final weights of each model (the checkpoints stay out of git, in `runs/`). Details of each one are in
[../README.md](../README.md#every-model).

| `fox/` (always Fox) | `random/` (a new random character every game) |
|---|---|
| `M1.zip`: first model, CPU curriculum | `MK2.zip`: self-play partner of MK1 |
| `MK0.zip`: + whiff penalty | `MK4.zip`: league, reward v2 |
| `MK1.zip`: self-play vs MK2 | `MK4_live.zip`: MK4 taught by a person |
| `MK3.zip`: league, reward v2 | `MK6.zip`: from MK5, best random model |
| `MK3_live.zip`: MK3 taught by a person | `MK8.zip`: 200 ms reaction, from scratch |
| `MK5.zip`: best Fox | `MK10.zip`: 100 ms reaction, from MK6 |
| `MK7.zip`: 200 ms reaction, from scratch | `MK10_try1_200ms_failed.zip`, `MK10_try2_200ms_failed.zip`: 200 ms from MK6, got worse |
| `MK9.zip`: 200 ms reaction, from MK5 | |

The random-character models can also play Fox (they were trained on every character).

```bash
python play.py models/all_models/fox/MK3.zip                                 # Fox models
python play.py models/all_models/random/MK6.zip --character RANDOM          # random models
python play.py models/all_models/random/MK10.zip --character RANDOM         # uses its 100 ms reaction
python play.py models/all_models/fox/MK3_live.zip --learn                    # keep teaching it (saved over MK3_live.zip)
python party.py --p1 models/all_models/fox/MK3.zip --p2 models/all_models/random/MK4.zip
```
