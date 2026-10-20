# Super Smash Bros. Melee PPO Agent

Trains a PPO agent (Stable-Baselines3) to play Super Smash Bros. Melee (NTSC v1.02)
as Fox against the game's own CPU, through the Slippi Dolphin emulator and
[libmelee](https://github.com/vladfi1/libmelee). The agent reads the game state
straight from memory (positions, damage, animations...), not pixels.

The repo ships a trained model, `models/M1.zip`, that beats the level 9 CPU, and
`play.py` lets you play against it with an Xbox controller or the keyboard. A second
training pipeline (MK0, then MK1 vs MK2 self-play) adds a penalty for attacks that hit
nothing; see [Models and training stages](#models-and-training-stages).

## Results

M1 (`models/M1.zip`) was trained on Linux (Ryzen 7 7840HS, 8 cores; the GPU
is not needed) with 12 headless games in parallel. The curriculum reached CPU
level 9 after 860k steps (38 minutes); training stopped at 4.6M steps (1 h 50 min).

| Evaluated against (Final Destination) | Wins | Mean stock difference |
|---|---|---|
| CPU level 9 Fox | 20 / 20 | +3.0 |
| CPU level 9 Fox, sampled actions (`--stochastic`) | 20 / 20 | +2.6 |
| CPU level 9, a random character every game (never seen in training) | 18 / 20 | +2.1 |
| CPU level 5 Fox | 10 / 10 | +3.5 |
| CPU level 1 Fox | 10 / 10 | +3.9 |

The Melee CPU is predictable, so expect the agent to be much easier to beat for a person.
M1 also spams attacks: 64% of them hit nothing (about 80 of 120 per game), which is what
the MK models address.

## Requirements

- **Linux x86-64** for headless training and `play.py`. macOS (where the project
  started) can still train, evaluate and watch with Dolphin windows.
- Python 3.10.
- **Your own Melee NTSC v1.02 ISO**, not included in the repo. Put it in this folder as
  `Super Smash Bros Melee v1.02.iso` (game ID `GALE01`, revision 2).
- About 600 MB of RAM per parallel game (one Dolphin plus one Python worker each).

## Setup

```bash
./setup.sh                    # .venv with Python 3.10 + dependencies, downloads Dolphin
cp /path/to/your.iso "Super Smash Bros Melee v1.02.iso"
source .venv/bin/activate
python test_env.py --headless # starts one game and plays random actions
```

`setup.sh` uses `uv` when it is installed and `python3.10 -m venv` otherwise. It
downloads Slippi Dolphin 3.6.4 (the normal emulator, with a window) and, on Linux,
the headless "ExiAI" Dolphin build from libmelee's author, used for training. On
Ubuntu, AppImages need `libfuse2`. PyTorch's CUDA build is not needed; to skip its
~3 GB download, run `pip install torch --index-url https://download.pytorch.org/whl/cpu`
before `pip install -r requirements.txt`.

## Play against the agent

```bash
python play.py models/M1.zip                    # Xbox controller plugged in by USB
python play.py models/M1.zip --input keyboard
```

The agent is port 1 and plays Fox; you are port 2. Pick your character on the
character select screen and press Start; the agent picks Final Destination, the only
stage it knows. Games run at normal speed with sound and can be paused. Close the
Dolphin window or press Ctrl+C in the terminal to stop.

| Xbox controller | Does |
|---|---|
| Left stick | Move. Pushing up never jumps |
| A | A, sent to the game the instant you press it |
| B | Special (up + B: up special) |
| X / Y | Jump (the only way to jump) |
| Right stick | On the ground: hold a direction = smash attack that keeps charging until you let go (`--no-charge-smash` for a plain C-stick). In the air: aerials (C-stick) |
| RB | Z (grab) |
| LT / RT | L / R (shield, air dodge) |
| Menu | Start |

The Xbox controller is read in Python (`smash_rl/xbox_pad.py`) and replayed to the
game, which is what allows pushing up without jumping. Every press reaches the game for at
least 2 frames, so quick taps are never lost. `--smash-hold-ms 150` turns on an optional
"direction + A held = smash" mode; it is off by default because it has to hold A back until it
knows whether the press is a tap, which makes attacks feel late.

| Keyboard | Does |
|---|---|
| W A S D | Control stick (W jumps, like in Melee) |
| Hold Space | Half tilt (walk, tilt attacks) |
| Arrow keys | C-stick |
| J / K | A / B |
| L / I | X / Y (jump) |
| O | Z (grab) |
| { (right of Ñ, Latin American layout) | R (shield) |
| Enter | Start |

The keyboard mapping lives in Dolphin (`dolphin/play_home`). Change it in the Dolphin
window with Controllers > Port 2 > Configure; `--remap` restores the default.

### Teaching a model by playing it

```bash
python play.py models/MK3.zip --learn                                  # 1 vs 1
python party.py --p1 models/MK3.zip --p2 models/MK4.zip --learn        # both agents learn
python play.py models/MK3_live.zip --learn                             # keep going next time
```

With `--learn` an agent samples its actions (slightly less sharp than its best move) and, after
every game, runs one PPO update on that game with reward v2 and saves to `models/<name>_live.zip`
(`--save-to` in `play.py`); the original file is never changed. A game is a few thousand
decisions against millions in training, so expect a nudge per game, not a new model. In
`party.py` each agent's reward comes from its fight with you (port 3).

### Free-for-all against two models

```bash
python party.py                       # MK1 (port 1) + MK2 (port 2) vs you (port 3), Xbox controller
python party.py --p1 models/MK3.zip --p2 models/MK4.zip --input keyboard
```

On the character select screen, click the button above slot 4 to add a Melee CPU. Turn an agent
off or back on with View + LB (agent 1) or View + RB (agent 2), or by typing 1 or 2 + Enter in the
terminal; it leaves or rejoins at the next character select. With more than one opponent each
agent targets the nearest one.

## Watch or evaluate a model

```bash
python evaluate.py models/M1.zip --cpu-level 9 --games 3       # watch, normal speed
python evaluate.py models/M1.zip --games 20 --headless         # just the numbers, fast
python evaluate.py models/M1.zip --opponent RANDOM --headless  # other characters
python evaluate.py runs/<run>/checkpoints --games 5 --headless --cpu-level 5  # score every checkpoint
```

Passing a folder evaluates every `.zip` in it and writes `eval_results.csv` there.
Actions are deterministic (most likely action) unless `--stochastic` is given. `--reaction-ms N` makes the
agent see the game as it was N ms ago, like a person, instead of reading it instantly. By default
each model uses the reaction time it was trained with: 200 ms for MK7 and MK8, 0 (instant) for older models.

**MK9 + MK10** (`league_mk56910.py`): MK7 and MK8 stalled at CPU level 4-5 when learning from scratch
with the delay, so MK9 (Fox) starts from MK5's weights and MK10 (random) from MK6's, both with the
200 ms reaction time; MK5 and MK6 keep learning, MK3 and MK4 are frozen opponents.

```bash
python league_mk56910.py --mk3 runs/mk5678_d/MK3.zip --mk4 runs/mk5678_d/MK4.zip \
    --mk5 runs/mk5678_d/MK5.zip --mk6 runs/mk5678_d/MK6.zip --hours 12 --run-name mk56910
```

**MK7 + MK8** (`league_mk5678.py`): MK7 (Fox) and MK8 (random) learn from scratch with a 200 ms
reaction time, next to MK5 and MK6; MK3 and MK4 are frozen opponents. MK7 and MK8 each climb their
own CPU curriculum.

```bash
python league_mk5678.py --mk3 runs/mk3456_g/MK3.zip --mk4 runs/mk3456_g/MK4.zip \
    --mk5 runs/mk3456_g/MK5.zip --mk6 runs/mk3456_g/MK6.zip --hours 12 --run-name mk5678
```
`--cpu-level` defaults to 9, so set it to the level a model actually reached.

## Train

```bash
python train.py --n-envs 8 --timesteps 14000000 --save-every 50 --run-name my_run
python train.py --n-envs 8 --opponent RANDOM          # a random CPU character every game
python train.py --cpu-level 9 --no-curriculum         # always fight a level 9 CPU
python train.py --resume runs/<run>/checkpoints/<file>.zip --cpu-level 7
python train.py --gui                                 # show a Dolphin window per game (much slower)
```

- On Linux training is headless by default: about 600 steps/s with 6 games and
  740 with 12 on 8 cores, versus about 290 with windows. The bottleneck is the
  emulator, which runs on the CPU; the network takes under 1% of the time, so a GPU doesn't help.
- **Use at most one game per physical core and watch the memory.** A 12-game run on a
  14 GB laptop ran out of memory after two hours and stopped (see Known issues).
- The learning rate decays linearly to 0 over `--timesteps`, so pick the length up
  front rather than stopping early.
- Checkpoints go to `runs/<run>/checkpoints/` every `--save-every` PPO updates (one
  update is `n_envs x 1024` steps), and `runs/<run>/final_model.zip` is written at the
  end or when you press Ctrl+C (once).
- `--resume` does not remember the curriculum level; pass `--cpu-level` again.
- To keep a laptop from sleeping when the lid closes, start training with
  `systemd-inhibit --what=sleep:handle-lid-switch python train.py ...`. Closing the
  terminal stops training (and saves the model).

## Models and training stages

| Model | Plays | Trained against | Whiff penalty | How |
|---|---|---|---|---|
| **M1** (`models/M1.zip`) | Fox | CPU, curriculum 1 to 9 | none | `train.py` (results above) |
| **MK0** | Fox | CPU, curriculum 1 to 9, until it wins 90% of 50 games at level 9 | 0.01 | stage 1, `train.py` from scratch |
| **MK1** | Fox (port 1) | MK2, both learning at the same time | 0.01 | stage 2, `selfplay.py`, starts from MK0 |
| **MK2** | a random character every game (port 2) | MK1 | 0.01 | stage 2, `selfplay.py`, starts from MK0 |
| **MK3** | Fox | MK4, MK2, CPU level 9 and older copies of itself | reward v2 | stage 3, `league.py`, from scratch |
| **MK4** | a random character every game | MK3, MK2, CPU level 9 and older copies of itself | reward v2 | stage 3, `league.py`, from scratch |
| **MK5** | Fox | MK3, MK4, MK2, CPU level 9 and older copies of itself | reward v2 without any shield / dodge reward, from the start | from scratch, alongside MK3 / MK4 |
| **MK6** | a random character every game | MK3, MK4, MK5 and a CPU whose level climbs from 1 to 9 | reward v2 | `league.py`, starts from MK5's weights; MK3-MK6 all keep learning together |

**MK1 and MK2 turned out passive**: Fox fires lasers all game (specials weren't covered by the
whiff penalty) and the other agent stands still or keeps dodging. MK3 / MK4 use **reward v2**
([docs/COMBOS.md](docs/COMBOS.md)): knockouts worth 5, per-character combo lists that pay more
for every hit that connects and continues a combo, a penalty for any attack that misses (A or
special) and for repeating one move (Fox may fire 6 lasers in 7 attacks), a penalty for every
second without hurting anyone, and small rewards for blocking, powershield reflects and dodges.

**Stage 3, MK3 + MK4** (8 games: MK3 vs MK4 twice, then each against MK2, the level 9 CPU and an
older copy of itself; one update is 8 x 1024 steps, about 13 s):

```bash
# MK3, MK4, MK5 and MK6 learn together: one game per pair of models (both learn from it),
# MK6 vs a CPU that levels up as it wins, and MK3 / MK4 / MK5 taking turns against the level 9 CPU.
python league.py --mk3 runs/mk345/MK3.zip --mk4 runs/mk345/MK4.zip --mk5 runs/mk345/MK5.zip \
    --hours 12 --run-name mk3456
cp runs/mk3456/MK6.zip models/MK6.zip
```

`--frozen MK4` keeps a model as an opponent that never learns (its games still happen, it leaves
the level 9 CPU rotation, and its saved weights are unchanged). `--lr-mkN` / `--ent-mkN` set each
model's learning rate and entropy bonus.

The **whiff penalty** subtracts a fixed amount (0.01, the same as taking 1% damage) for
every A attack (jab, tilt, smash, aerial) that ends without hitting the opponent or their
shield. Specials, grabs and dodges are not penalized. `evaluate.py`, `versus.py`,
`selfplay.py` and TensorBoard (`smash/whiff_rate`) report the fraction of attacks that
hit nothing.

**Stage 1, MK0** (stops by itself once it wins 90% of 50 games against level 9; M1 reached level 9 in under an hour, the penalty may make it slower):

```bash
python train.py --whiff-penalty 0.01 --stop-win-rate 0.9 --stop-window 50 \
  --n-envs 8 --timesteps 14000000 --save-every 50 --run-name mk0
cp runs/mk0/final_model.zip models/MK0.zip
```

**Stage 2, MK1 vs MK2** (`--updates` is how long they train; one update is 8 x 1024 game
steps, so 500 updates take about 2 hours):

```bash
python selfplay.py --init models/MK0.zip --updates 500 --n-envs 8 --run-name mk_selfplay
cp runs/mk_selfplay/MK1.zip models/MK1.zip
cp runs/mk_selfplay/MK2.zip models/MK2.zip
```

Both models act on the same games every step (MK1 from port 1, MK2 from port 2, each
seeing the game from its own side), and each then runs its own PPO update, with the
learning rate decaying from `--lr` (1e-4) to 0. `--mk1` / `--mk2` continue from
earlier weights. Compare the results with:

```bash
python versus.py models/MK1.zip models/MK2.zip --games 20 --headless      # MK1 vs MK2
python versus.py models/M1.zip models/MK1.zip --p2-character FOX --headless
python evaluate.py models/MK1.zip --headless --games 20                    # vs CPU level 9
python evaluate.py models/MK2.zip --headless --games 20 --opponent RANDOM
```

`evaluate.py` and `play.py` always put the model on port 1 as `--character` (Fox by
default); MK2 can be given another character with `--character`.

## Monitor

```bash
tensorboard --logdir runs        # open http://localhost:6006
```

| Tag | Meaning |
|---|---|
| `smash/won` | Fraction of finished games the agent won |
| `smash/cpu_level` | Current curriculum level |
| `smash/stock_diff` | Agent stocks minus CPU stocks at the end of a game |
| `smash/kills`, `smash/deaths`, `smash/damage_dealt`, `smash/damage_taken` | Per game |
| `smash/whiff_rate`, `smash/attacks` | Fraction of A attacks that hit nothing, A attacks per game |
| `rollout/ep_rew_mean` | Average reward per game |
| `train/entropy_loss` | Minus the policy entropy; starts at -3.43 (31 equally likely actions) |

## How it works

- **Environment** (`smash_rl/env.py`): each env runs its own Dolphin and talks to it
  through libmelee. The agent is port 1, the CPU port 2, on Final Destination. One
  episode is one full game (4 stocks); the env goes through the menus by itself.
- **Observation:** 898 floats read from game memory, `[agent 447 | opponent 447 | 4 global]`.
  Per player: 20 continuous features (position, damage, stocks, facing, on ground,
  off stage, invulnerable, hitlag and hitstun left, jumps left, shield, self and
  knockback speeds, animation frame, distance to the edge, IASA), a 27-way character
  one-hot and a 400-way action-state one-hot. Global: dx, dy, distance and game time.
- **Actions:** 31 discrete stick + button combinations, each held for 3 frames
  (20 decisions per second): stick only in 9 directions, A + 9 directions, B + 5,
  X (jump) + 3, Z (grab), L (shield, roll left/right, spot dodge).
- **Reward:** per frame, `0.01 x (damage dealt - damage taken) + (stocks taken - stocks lost)`,
  minus `--whiff-penalty` for each A attack that ended without hitting anything (0 for M1).
- **PPO:** `MlpPolicy` with separate 256-256 ReLU actor and critic networks.
  1024-step rollouts per env, batch 512, 4 epochs, learning rate 3e-4 decaying to 0,
  gamma 0.995, GAE lambda 0.95, clip 0.2, entropy coefficient 0.01. All are flags
  of `train.py`.
- **Curriculum:** starts against CPU level 1 and moves up one level whenever the agent
  wins at least 70% of the last 20 games at the current level, up to 9.
- **Headless mode:** the ExiAI Dolphin build runs with no window and Melee's
  fast-forward mode. It crashes on the "instant match restart" code used with windows,
  so headless games go back through the menus between games (about 0.6 s).

## Files

| File | Purpose |
|---|---|
| `smash_rl/env.py` | Gymnasium env: Dolphin lifecycle, menus, observation, actions, reward; vectorized envs |
| `smash_rl/xbox_pad.py` | Reads an Xbox controller and replays it to port 2 with the custom controls |
| `train.py` | PPO training with curriculum, checkpoints and TensorBoard stats |
| `evaluate.py` | Watch a model or score a folder of checkpoints |
| `play.py` | Play against a model |
| `selfplay.py` | Stage 2: train two models against each other at the same time |
| `versus.py` | Play two models against each other (no training) |
| `league.py` | Stage 3: MK3 and MK4 from scratch against a league of opponents, reward v2 |
| `party.py` | You (port 3) against up to two models, plus a Melee CPU on port 4 if you add one |
| `smash_rl/rewards.py`, `smash_rl/combos.py` | Reward v2 and the per-character combo lists ([docs/COMBOS.md](docs/COMBOS.md)) |
| `smash_rl/live_learning.py` | `--learn` in play.py / party.py: one PPO update per game against a person |
| `test_env.py` | Environment sanity check and speed test |
| `setup.sh` | Python environment and emulator downloads |
| `models/` | Trained models shipped with the repo |

## Known issues

- **Running out of memory stops training.** When a Dolphin dies mid-game the error
  ends the whole run (the model is still saved), and the process can then hang
  instead of exiting; stop it with Ctrl+C or `kill`. Train with fewer games and close
  heavy apps.
- **Changing the CPU level gets each headless game stuck in the menus once.** The env
  notices after about 6,000 frames, restarts that Dolphin and goes on, so it costs a
  few seconds per level change.
- The agent has only seen Final Destination and plays only Fox.
- The `iasa` feature is always 0 (libmelee never fills it in).
- Models trained before the character one-hot was added (844 inputs instead of 898)
  can't be loaded by the current env.

## Credits

[libmelee](https://github.com/vladfi1/libmelee) and its headless
[ExiAI Dolphin build](https://github.com/vladfi1/slippi-Ishiiruka) by vladfi1, and
[Project Slippi](https://github.com/project-slippi/Ishiiruka). Super Smash Bros. Melee
is a Nintendo / HAL Laboratory game; this repo contains no game files.
