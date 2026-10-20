# The Ganondorf agent in this repo

`README.md`, `ganondorf_agent.py`, `match.py` and `meta.json` are the friend's package as he shared it
(trained by Venkateswarlu Reddy Konkala); only the folder layout differs: the model sits next to the code
instead of in `model/`, so his `match.py` needs `--model ganondorf_edge_v4.zip` when run by hand.

- PPO (Stable-Baselines3), 847 inputs (both players + the opponent's character), 31 actions every 3 frames,
  no reaction delay. It mirrors the game so the opponent is always on its right.
- Trained on Final Destination against level 9 CPU Fox / Marth / Captain Falcon and ledge-camping bots,
  21.6 M steps; 9/9 wins against those CPUs, +3.22 stocks on average.
- It needs only Stable-Baselines3, already in our `.venv`.

Used by `party_teams.py` (P3), `league_final.py` (as a training opponent, through
`smash_rl/external_bots.py` and `SmashEnv(opponent_bot=...)`) and `vs_friend.py` (our models join its
`match.py` as the `FriendAgent`).

```bash
python party_teams.py --p1 none --p2 none --p4 human            # you vs Ganondorf
python party_teams.py --p2 none --p4 none --autostart           # watch MK5 vs Ganondorf
python vs_friend.py MK5=models/alejandro/MK5.zip --games 10     # headless series, both ports
```
