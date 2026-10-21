# Super Smash Bros. Melee PPO Agents

Trains PPO agents (Stable-Baselines3) to play Super Smash Bros. Melee (NTSC v1.02) through the Slippi
Dolphin emulator and [libmelee](https://github.com/vladfi1/libmelee). The agents read the game state
straight from memory (positions, damage, animations...), not pixels.

The repo ships every model we trained (M1, MK0-MK10) plus two other teams' bots to play with: Kepiti's
Fox (imitation learning, imitation + RL) and a Ganondorf PPO agent. You can play any of them with an Xbox
controller or the keyboard, alone or in a four-player free-for-all, or let them fight each other.

## Models

All in [`models/`](models/README.md), final weights only:

| Folder | What | Character |
|---|---|---|
| [`models/alejandro/`](models/alejandro/README.md) | Our picks: **MK5** (best Fox, pure RL) and **MK9** (Fox with a 200 ms human reaction time) | Fox |
| [`models/kepiti/`](models/kepiti/README.md) | Kepiti's Fox: imitation learning (18 h of human replays) and imitation + 3 h RL | Fox |
| [`models/ganondorf/`](models/ganondorf/USAGE.md) | A friend's Ganondorf PPO agent, with its own code | Ganondorf |
| [`models/all_models/`](models/all_models/README.md) | Every model we trained, in `fox/` and `random/` (a new character every game) | Fox / random |

The project compares four ways of training a Fox: **pure RL** (MK5), **RL with a reaction delay** (MK9),
**imitation learning** and **imitation + RL** (Kepiti's two models). [models/README.md](models/README.md)
has every model's character, reaction time, training and results.

## Requirements

- **Linux x86-64** for headless training and the play scripts. macOS (where the project
  started) can still train, evaluate and watch with Dolphin windows.
- Python 3.10.
- **Your own Melee NTSC v1.02 ISO**, not included in the repo. Put it in this folder as
  `Super Smash Bros Melee v1.02.iso` (game ID `GALE01`, revision 2).
- About 600 MB of RAM per parallel game (one Dolphin plus one Python worker each).
- For Kepiti's models, his repo next to this one:
  `git clone https://github.com/Kepitition/super_smash_RL ~/Documents/DML/super_smash_RL`.

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

## Play

### One against one

```bash
python play.py models/alejandro/MK5.zip                               # Xbox controller plugged in by USB
python play.py models/alejandro/MK5.zip --input keyboard
python play.py models/alejandro/MK9.zip                               # reacts 200 ms late, like a person
python play.py models/all_models/random/MK6.zip --character RANDOM    # a new character every game
python party_teams.py --p1 none --p3 none --p4 human                  # Kepiti's imitation + RL Fox
python party_teams.py --p1 none --p3 none --p4 human --p2 models/kepiti/fox_imitation_18h.pt
python party_teams.py --p1 none --p2 none --p4 human                  # the Ganondorf agent
```

In `play.py` the model is port 1 and you are port 2; in `party_teams.py` you are port 4. Pick your
character on the character select screen and press Start; the bots pick Final Destination, the only
stage they know. Games run at normal speed with sound and can be paused. Close the Dolphin window or
press Ctrl+C in the terminal to stop. Each of our models reacts with the delay it was trained with
(`--reaction-ms N` in `play.py` / `party.py` overrides it).

### Everyone at once

```bash
python party_teams.py --p4 human                                 # you vs MK5 (P1) + Kepiti (P2) + Ganondorf (P3)
python party_teams.py --p1 models/alejandro/MK9.zip --p4 human   # any of our .zip models on P1
python party_teams.py --p4 cpu --cpu-level 9 --autostart          # watch the three bots and a CPU
python party_teams.py --p1 none --p4 none --autostart            # watch Kepiti vs Ganondorf
python party.py                                                  # you (port 3) vs two of ours: MK5 + MK6
python party.py --p1 models/all_models/fox/MK3.zip --p2 models/all_models/random/MK4.zip --input keyboard
```

`party_teams.py` gives each team a port: P1 our models, P2 Kepiti's (`.pt`), P3 the Ganondorf agent,
P4 you (`--p4 human`), a Melee CPU (`--p4 cpu`, `--cpu-character`, `--cpu-level`) or nobody; any of
P1-P3 can be `none`. `--autostart` makes the bots press Start. With three or more players every bot
attacks the nearest opponent (they were all trained one against one). Type 1, 2 or 3 + Enter in the
terminal on the character select screen to turn a bot off or on.

In `party.py` you are port 3 against up to two of our models; click the button above slot 4 to add a
Melee CPU. Turn an agent off or back on with View + LB (agent 1) or View + RB (agent 2), or by typing
1 or 2 + Enter in the terminal.

### Controls

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
least 2 frames, so quick taps are never lost. Plug it in by USB: Bluetooth adds latency.
`--smash-hold-ms 150` turns on an optional "direction + A held = smash" mode; it is off by default
because it has to hold A back until it knows whether the press is a tap, which makes attacks feel late.

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
window with Controllers > Port N > Configure; `--remap` restores the default. Dolphin only reads
the keyboard while its window has the focus.

### Teaching a model by playing it

```bash
python play.py models/all_models/fox/MK3.zip --learn                  # 1 vs 1, saves MK3_live.zip next to it
python party.py --p1 models/all_models/fox/MK3.zip --p2 models/all_models/random/MK4.zip --learn
python play.py models/all_models/fox/MK3_live.zip --learn             # keep going next time
```

With `--learn` an agent samples its actions (slightly less sharp than its best move) and, after
every game, runs one PPO update on that game with reward v2 and saves to `<name>_live.zip` next to
the model (`--save-to` in `play.py`); the original file is never changed. A game is a few thousand
decisions against millions in training, so expect a nudge per game, not a new model. In
`party.py` each agent's reward comes from its fight with you (port 3).

## Bots against bots

```bash
python tournament.py MK5=models/alejandro/MK5.zip MK9=models/alejandro/MK9.zip \
    MK3=models/all_models/fox/MK3.zip --games 20 --cpu-games 20     # round robin + CPU 9, all as Fox
python vs_friend.py MK5=models/alejandro/MK5.zip MK9=models/alejandro/MK9.zip --games 10   # vs Ganondorf
python versus.py models/alejandro/MK5.zip models/all_models/random/MK6.zip --games 20 --headless
python evaluate.py models/alejandro/MK5.zip --cpu-level 9 --games 3     # watch one against the CPU
python evaluate.py models/alejandro/MK5.zip --games 20 --headless       # just the numbers, fast
python evaluate.py runs/<run>/checkpoints --games 5 --headless --cpu-level 5  # score every checkpoint
```

- `tournament.py`: every pair of our models plays `--games` games (sides alternate) and each plays the
  CPU (`--cpu-level`, `--cpu-character`) `--cpu-games` games, 8 headless games at a time, each model
  with its own reaction time. Nothing learns. Prints win-rate tables and writes every game to `--out`.
- `vs_friend.py`: our models against the Ganondorf agent through its own `match.py`
  (`models/ganondorf/`), `--games` on each port; our model joins as `external/friend_agent.py`.
- `evaluate.py`: passing a folder evaluates every `.zip` in it and writes `eval_results.csv` there.
  Actions are deterministic (most likely action) unless `--stochastic` is given.

## Results

M1, the first model, was trained on Linux (Ryzen 7 7840HS, 8 cores; the GPU is not needed) with 12
headless games in parallel. The curriculum reached CPU level 9 after 860k steps (38 minutes); training
stopped at 4.6M steps (1 h 50 min).

| Evaluated against (Final Destination) | Wins | Mean stock difference |
|---|---|---|
| CPU level 9 Fox | 20 / 20 | +3.0 |
| CPU level 9 Fox, sampled actions (`--stochastic`) | 20 / 20 | +2.6 |
| CPU level 9, a random character every game (never seen in training) | 18 / 20 | +2.1 |
| CPU level 5 Fox | 10 / 10 | +3.5 |
| CPU level 1 Fox | 10 / 10 | +3.9 |

M1 also spams attacks: 64% of them hit nothing, which is what the MK models address. The later
models' results (a preliminary round robin and the CPU levels they reached) are in
[models/README.md](models/README.md).

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
- The learning rate decays linearly to 0 over `--timesteps` (`--hours` in the league scripts), so pick
  the length up front rather than stopping early.
- Checkpoints go to `runs/<run>/checkpoints/` every `--save-every` PPO updates (one
  update is `n_envs x 1024` steps), and the final models are written at the end or when you press
  Ctrl+C.
- `--resume` does not remember the curriculum level; pass `--cpu-level` (or `--mkN-cpu-start`) again.
- To keep a laptop from sleeping when the lid closes, start training with
  `systemd-inhibit --what=sleep:handle-lid-switch python train.py ...`. Closing the
  terminal stops training (and saves the model).

### Training stages

| Model | Plays | Trained against | Reward | How |
|---|---|---|---|---|
| **M1** | Fox | CPU, curriculum 1 to 9 | v1 | `train.py` (results above) |
| **MK0** | Fox | CPU, curriculum 1 to 9, until it wins 90% of 50 games at level 9 | v1 + whiff penalty 0.01 | stage 1, `train.py` from scratch |
| **MK1** | Fox (port 1) | MK2, both learning at the same time | v1 + whiff penalty | stage 2, `selfplay.py`, starts from MK0 |
| **MK2** | random (port 2) | MK1 | v1 + whiff penalty | stage 2, `selfplay.py`, starts from MK0 |
| **MK3** | Fox | MK4, MK2, CPU level 9, later MK5 / MK6 | v2 | stage 3, `league.py`, from scratch |
| **MK4** | random | MK3, MK2, CPU level 9, later MK5 / MK6 | v2 | stage 3, `league.py`, from scratch |
| **MK5** | Fox | MK3, MK4, MK6, CPU level 9, later MK7-MK10 | v2 without shield / dodge rewards | `league.py` → `league_mk56910.py`, from scratch |
| **MK6** | random | MK3, MK4, MK5 and a CPU whose level climbs from 1 to 9 | v2 | `league.py`, starts from MK5's weights |
| **MK7** / **MK8** | Fox / random | MK5, MK6, frozen MK3 / MK4, each its own CPU curriculum | v2, 200 ms reaction | `league_mk5678.py`, from scratch |
| **MK9** / **MK10** | Fox / random | MK5, MK6, frozen MK3 / MK4, each its own CPU curriculum | v2, 200 / 100 ms reaction | `league_mk56910.py`, from MK5 / MK6's weights |
| **MK5** + **MK9** (final) | Fox | frozen copies of themselves (half the games), Kepiti's two Foxes, Ganondorf, CPU 9, MK3, MK6, MK10 | v2 | `league_final.py` |

**MK1 and MK2 turned out passive**: Fox fires lasers all game (specials weren't covered by the
whiff penalty) and the other agent stands still or keeps dodging. From MK3 on the models use
**reward v2** ([docs/COMBOS.md](docs/COMBOS.md)): knockouts first, per-character combo lists that pay
more for every hit that connects and continues a combo, penalties for any attack that misses (smash
attacks more), for repeating one move or smash attack (each repetition doubles the penalty, up to a
knockout), for overusing shields and dodges, and for every second without hurting anyone.

**MK7 and MK8** learned from scratch while seeing the game 200 ms late and stalled at CPU level 5 and
4, so **MK9 and MK10** start from MK5 / MK6's weights instead. With 200 ms MK10 (random characters) got
worse twice; with 100 ms it learned. The delay is saved in the model (`reaction_ms`) and applied by
every play script.

```bash
# stage 1 and 2
python train.py --whiff-penalty 0.01 --stop-win-rate 0.9 --stop-window 50 \
  --n-envs 8 --timesteps 14000000 --save-every 50 --run-name mk0
python selfplay.py --init models/all_models/fox/MK0.zip --updates 500 --n-envs 8 --run-name mk_selfplay

# stage 3: MK3-MK6 learn together (one game per pair of models, both learn from it), MK6 climbs a CPU
# curriculum and MK3 / MK4 / MK5 take turns against the level 9 CPU. --frozen MK4 keeps a model as a
# fixed opponent; --lr-mkN / --ent-mkN set each model's learning rate and entropy bonus.
python league.py --mk3 runs/mk345/MK3.zip --mk4 runs/mk345/MK4.zip --mk5 runs/mk345/MK5.zip \
    --hours 12 --run-name mk3456

# reaction-time models (--mkN-cpu-start when continuing: the CPU levels are not saved)
python league_mk5678.py --mk3 models/all_models/fox/MK3.zip --mk4 models/all_models/random/MK4.zip \
    --mk5 models/all_models/fox/MK5.zip --mk6 models/all_models/random/MK6.zip --hours 12 --run-name mk5678
python league_mk56910.py --mk3 models/all_models/fox/MK3.zip --mk4 models/all_models/random/MK4.zip \
    --mk5 models/all_models/fox/MK5.zip --mk6 models/all_models/random/MK6.zip --hours 12 --run-name mk56910

# final training of MK5 and MK9, 1 vs 1 (needs Kepiti's repo for his bots)
python league_final.py --mk5 models/alejandro/MK5.zip --mk9 models/alejandro/MK9.zip \
    --mk5_old models/alejandro/MK5.zip --mk9_old models/alejandro/MK9.zip \
    --mk3 models/all_models/fox/MK3.zip --mk6 models/all_models/random/MK6.zip \
    --mk10 models/all_models/random/MK10.zip --hours 8.5 --run-name final_mk59
```

The league scripts write `runs/<run>/<model>.zip` at the end (also on Ctrl+C), checkpoints every
`--save-every` updates and TensorBoard logs to `runs/<run>/<model>`. Copy a finished model into
`models/` to keep it in the repo. In `league_final.py` Kepiti's bot runs its network on the CPU every
frame, which slows the games it is in; they are grouped in one of the four rotating sets.

## Monitor

```bash
tensorboard --logdir runs        # open http://localhost:6006
```

| Tag | Meaning |
|---|---|
| `smash/won`, `smash/won_vs_<opponent>` | Fraction of finished games the agent won |
| `smash/cpu_level` | Current curriculum level |
| `smash/stock_diff` | Agent stocks minus opponent stocks at the end of a game |
| `smash/kills`, `smash/deaths`, `smash/damage_dealt`, `smash/damage_taken` | Per game |
| `smash/whiff_rate`, `smash/attacks` | Fraction of attacks that hit nothing, attacks per game |
| `smash/combos`, `smash/smash_spam`, `smash/spam_penalty` | Reward v2: finished combos, smash attacks spammed, reward lost to spam, per game |
| `train/entropy_loss` | Minus the policy entropy; starts at -3.43 (31 equally likely actions) |

## How it works

- **Environment** (`smash_rl/env.py`): each env runs its own Dolphin and talks to it
  through libmelee. The agent is port 1, the opponent port 2 (the Melee CPU, a second agent, a
  person, or another team's bot with `opponent_bot`), on Final Destination. One episode is one full
  game (4 stocks); the env goes through the menus by itself.
- **Observation:** 898 floats read from game memory, `[agent 447 | opponent 447 | 4 global]`.
  Per player: 20 continuous features (position, damage, stocks, facing, on ground,
  off stage, invulnerable, hitlag and hitstun left, jumps left, shield, self and
  knockback speeds, animation frame, distance to the edge, IASA), a 27-way character
  one-hot and a 400-way action-state one-hot. Global: dx, dy, distance and game time.
- **Actions:** 31 discrete stick + button combinations: stick only in 9 directions, A + 9 directions,
  B + 5, X (jump) + 3, Z (grab), L (shield, roll left/right, spot dodge). Each is held for `frame_skip`
  frames: 3 (20 decisions per second) for every model up to MK10; MK5 and MK9 continue at 2 (30 per
  second, `league_final.py --frame-skip 2`). The value is saved in each model (`frame_skip`) and every
  play, party, evaluation and tournament script uses it; when it changes, the league scripts convert
  gamma and GAE lambda so the horizon stays the same in seconds.
- **Reaction time** (`smash_rl/reaction.py`): MK7-MK10 see the observation from N decisions ago
  (200 ms = 4 decisions), in training and in play.
- **Reward:** v1 (M1-MK2) is `0.01 x (damage dealt - damage taken) + (stocks taken - stocks lost)`
  per frame, minus `--whiff-penalty` per missed A attack. v2 (MK3 on) is in `smash_rl/rewards.py`
  and [docs/COMBOS.md](docs/COMBOS.md).
- **PPO:** `MlpPolicy` with separate 256-256 ReLU actor and critic networks.
  1024-step rollouts per env, batch 512, 4 epochs, learning rate 3e-4 decaying to 0,
  gamma 0.995, GAE lambda 0.95, clip 0.2, entropy coefficient 0.01. The league scripts run
  several models in one rollout loop, each learning from its own side of the games.
- **Curriculum:** starts against CPU level 1 and moves up one level whenever the agent
  wins at least 70% of the last 20 games at the current level, up to 9.
- **Headless mode:** the ExiAI Dolphin build runs with no window and Melee's
  fast-forward mode. It crashes on the "instant match restart" code used with windows,
  so headless games go back through the menus between games (about 0.6 s).

## Files

| File | Purpose |
|---|---|
| `smash_rl/env.py` | Gymnasium env: Dolphin lifecycle, menus, observation, actions, reward; vectorized envs |
| `smash_rl/rewards.py`, `smash_rl/combos.py` | Reward v2 and the per-character combo lists ([docs/COMBOS.md](docs/COMBOS.md)) |
| `smash_rl/reaction.py` | Human-like reaction delay |
| `smash_rl/external_bots.py` | Other teams' bots (Kepiti's Fox, the Ganondorf agent) as opponents |
| `smash_rl/xbox_pad.py` | Reads an Xbox controller and replays it to the game with the custom controls |
| `smash_rl/live_learning.py` | `--learn` in play.py / party.py: one PPO update per game against a person |
| `train.py` | PPO against the CPU with curriculum, checkpoints and TensorBoard stats |
| `selfplay.py` | Stage 2: two models learn against each other at the same time |
| `league.py`, `league_mk5678.py`, `league_mk56910.py`, `league_final.py` | Several models learn together against each other, frozen models, bots and CPUs |
| `play.py` | You against one of our models |
| `party.py` | You (port 3) against up to two of our models, plus a Melee CPU |
| `party_teams.py` | Free-for-all: our model, Kepiti's, Ganondorf and you or a CPU |
| `tournament.py` | Round robin between our models and against the CPU (no training) |
| `vs_friend.py`, `external/friend_agent.py` | Our models against the Ganondorf agent in its own `match.py` |
| `versus.py`, `evaluate.py` | Two models against each other; one model against the CPU |
| `test_env.py` | Environment sanity check and speed test |
| `setup.sh` | Python environment and emulator downloads |
| `models/` | Every trained model and the other teams' bots ([models/README.md](models/README.md)) |

## Known issues

- **Running out of memory freezes the laptop.** Each game needs about 600 MB; with 8 games in
  training, don't open heavy apps (an editor like Unity froze it once).
- **Changing the CPU level gets each headless game stuck in the menus once.** The env
  notices after about 6,000 frames, restarts that Dolphin and goes on, so it costs a
  few seconds per level change.
- Our agents have only seen Final Destination.
- The `iasa` feature is always 0 (libmelee never fills it in).
- Models trained before the character one-hot was added (844 inputs instead of 898)
  can't be loaded by the current env.

## Credits

[libmelee](https://github.com/vladfi1/libmelee) and its headless
[ExiAI Dolphin build](https://github.com/vladfi1/slippi-Ishiiruka) by vladfi1, and
[Project Slippi](https://github.com/project-slippi/Ishiiruka). Kepiti's Fox models come from
[Kepitition/super_smash_RL](https://github.com/Kepitition/super_smash_RL); the Ganondorf agent was
trained by Venkateswarlu Reddy Konkala. Super Smash Bros. Melee is a Nintendo / HAL Laboratory game;
this repo contains no game files.
