# Ganondorf PPO agent for Super Smash Bros. Melee

A reinforcement-learning agent (PPO, Stable-Baselines3) that plays **Ganondorf** through
[libmelee](https://github.com/altf4/libmelee). Trained by Venkateswarlu Reddy Konkala for
21.6 M steps; it wins 9/9 evaluation games against level-9 CPU Fox, Marth and Captain Falcon.

```
ganondorf_agent/
├── ganondorf_agent.py    the agent: GanondorfAgent(model, port, opponent_port).act(gamestate, controller)
├── match.py              ready-made match runner (agent vs CPU / vs your agent), any port
├── friend_agent.py       template: wrap your agent here to use it in match.py
├── model/
│   ├── ganondorf_edge_v4.zip   network weights (Stable-Baselines3 PPO)
│   └── meta.json               settings the agent needs (do not change)
└── requirements.txt
```

## Setup

1. Python 3.10 and the packages: `pip install -r requirements.txt`
2. **Slippi Dolphin** (not included):
   - headless and fast on Linux: libmelee's ExiAI build
     <https://github.com/vladfi1/slippi-Ishiiruka/releases/tag/exi-ai-0.2.0> (`Slippi_Online-x86_64-ExiAI.AppImage`)
   - with a window, any OS: Slippi Ishiiruka 3.6.4 <https://github.com/project-slippi/Ishiiruka/releases/tag/v3.6.4>
3. Your own **Super Smash Bros. Melee NTSC v1.02** ISO (not included).

## Run a match

```bash
python match.py --dolphin <dolphin> --iso <melee.iso> --p2 cpu:FOX:9 --games 3     # agent (P1) vs level-9 CPU
python match.py --dolphin <dolphin> --iso <melee.iso> --p2 friend --watch          # agent (P1) vs your agent (P2)
python match.py --dolphin <dolphin> --iso <melee.iso> --p1 friend --p2 ganondorf   # swap ports
```

For `friend`, put your agent in `friend_agent.py` (class `FriendAgent` with `character`, `reset()` and
`act(gs, controller)`). `--watch` plays at normal speed; without it, the ExiAI build runs as fast as it can.
Tested: agent as P1 beat CPU Fox 9 (2-0), agent as P2 beat CPU Marth 9 (4-0).

## Use it in your own match loop

```python
from ganondorf_agent import GanondorfAgent

agent = GanondorfAgent("model/ganondorf_edge_v4.zip", port=2, opponent_port=1)
# every frame:
gs = console.step()
if gs.menu_state in (melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH):
    agent.act(gs, controller_port2)     # sets the libmelee Controller for this frame
else:
    agent.reset()                       # between games
```

Requirements for correct play:
- call `act()` on **every in-game frame** (it decides every 3rd frame and releases buttons in between),
  with `blocking_input=True` on the `melee.Console` so the game waits for both agents;
- character select: the agent plays **GANONDORF**; it was trained on **Final Destination** only.

## What the network sees and does

- **Input (847 numbers)**, read from game memory, no pixels: for itself and its opponent position, damage %,
  stocks, facing, ground/off-stage, invulnerability, hitlag, hitstun, jumps, shield, speeds, animation
  frame, distance to the ledge, IASA and a 400-way one-hot of the animation; plus relative position,
  distance, game time and a one-hot of the opponent's character (Fox / Marth / Captain Falcon; other
  characters get zeros).
- The game is **mirrored** whenever the opponent is on its left, so the network always sees the opponent on
  its right; its stick direction is flipped back. `ganondorf_agent.py` does this for you.
- **Output**: one of 31 controller inputs (stick directions, A/B attacks and specials, short-hop jumps,
  grab, shield/roll/spot dodge), held for 3 frames (20 decisions per second).

Note for a fair comparison: the agent reads exact game state and reacts within 3 frames (50 ms).
