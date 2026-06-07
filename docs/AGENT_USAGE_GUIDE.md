# Agent Usage Guide

This guide covers training RL agents and running them against the real Slay the Spire 2 game. For the full training reference (curriculum stages, policy architectures, hyperparameters), see [TRAINING_GUIDE.md](TRAINING_GUIDE.md). For observation layout details, see [SIMULATOR_ARCHITECTURE.md](SIMULATOR_ARCHITECTURE.md) and [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

---

## 1. Training a Combat Agent

The combat environment (`STS2CombatEnv`) trains an agent on single combat encounters. Default policy is a **Transformer self-attention** extractor over **57 entity tokens**; observations are **1985-dim obs v11** (full power grid, draw-pile memory, relic/potion slots, ascension, turn count).

### Recommended: Curriculum-First Training

Start on easy Act 1 fights before widening encounter tiers:

```bash
pip install -e ".[train]"

python scripts/train_combat.py \
    --curriculum easy_pair \
    --characters Ironclad,Regent \
    --total-timesteps 500000 \
    --n-envs 8 \
    --output-dir output/curriculum/s1_easy
```

Hands-off auto-promotion through all stages:

```bash
python scripts/train_combat.py --curriculum full --auto-promote \
    --characters Ironclad,Regent --total-timesteps 5000000 \
    --output-dir output/curriculum/auto
```

### Advanced Combat Training

Mixed acts and all characters (combat delegate for full-run training):

```bash
python scripts/train_combat.py \
    --acts 0,1,2 \
    --characters all \
    --policy attention \
    --total-timesteps 3000000 \
    --n-envs 8 \
    --output-dir output/combat_ppo_mixed
```

GNN policy (requires `torch-geometric`):

```bash
python scripts/train_combat.py \
    --acts 0,1,2 \
    --policy gnn \
    --output-dir output/combat_ppo_mixed_gnn
```

### What Happens During Training

1. The script creates `--n-envs` parallel combat environments.
2. With `--curriculum`, each reset samples encounters and decks from the current stage ([`combat_curriculum.py`](../sts2_env/training/combat_curriculum.py)).
3. Without curriculum, encounters are drawn from the act/tier pool for the selected character(s).
4. MaskablePPO with the chosen feature extractor (`attention`, `gnn`, or `mlp`) collects rollouts.
5. **Reward shaping** (non-linear HP penalty + Vulnerable/Weak/block micro-rewards) is on by default during training; the eval env uses sparse ±1 rewards.
6. Every `--eval-freq` steps, the agent is evaluated. Curriculum runs also log `curriculum/gate_win_rate` and `curriculum/gate_hp_ratio`.
7. The best model is saved to `output_dir/best_model/`; after training, 100 evaluation episodes print win rate.
8. Use `--resume --output-dir <same>` to continue a paused run; use `--load-model` to fine-tune into a **new** output directory (e.g. next curriculum stage).

### Training Output

```text
output/curriculum/s1_easy/
  tb_logs/                    # TensorBoard logs
  eval_logs/                  # Evaluation results
  curriculum_state.json       # Stage index for parallel workers
  best_model/best_model.zip   # Best checkpoint
  final_model.zip             # Final checkpoint
  run_config.json             # Settings for --resume
```

View training progress:

```bash
tensorboard --logdir output/curriculum/s1_easy/tb_logs
```

### Combat Training Flags (selected)

| Parameter | Default | Notes |
| --------- | ------- | ----- |
| `--policy` | `attention` | `attention`, `gnn`, or `mlp` — checkpoints not interchangeable |
| `--curriculum` | — | Stage name or `full` |
| `--auto-promote` | off | Advance when gate metrics pass |
| `--reward-shaping` | True | Disable with `--no-reward-shaping` |
| `--hp-steepness` | 3.0 | Exponential HP penalty steepness |
| `--total-timesteps` | 2M (act 0) / 500K (curriculum) | More is generally better |
| `--n-envs` | 4 | Match CPU core count |
| `--lr` | 3e-4 | Lower for stability, higher for speed |
| `--mcts` | off | Post-training eval with turn-bounded MCTS |
| `--eval-only` | off | Run MCTS/post-train eval without training (requires `--load-model`) |
| `--flawless-bonus` | 0.1 | Bonus on flawless combat wins (no HP lost) |

### Checkpoint Compatibility

- Combat `OBS_SIZE` is **1985** (obs v11). Checkpoints below obs v10 are **obsolete** — retrain on v11.
- Policy type (`mlp` / `attention` / `gnn`) must match at load time. Check `run_config.json` in the output directory.
- See [KNOWN_ISSUES.md](KNOWN_ISSUES.md) for the full obs version history.

---

## 2. Training a Full-Run Agent

Full-run training uses a **hierarchical** design: a frozen combat PPO plays fights; a separate **Navigator** PPO handles strategic decisions.

### Recommended: Navigator PPO

Requires a trained mixed-act combat model:

```bash
# Phase 0: combat delegate
python scripts/train_combat.py --acts 0,1,2 --characters all \
    --total-timesteps 3000000 --output-dir output/combat_ppo_mixed

# Phase 2: Navigator (preferred)
python scripts/train_navigator.py --preset phase1 \
    --combat-model output/combat_ppo_mixed/best_model/best_model.zip \
    --combat-value-shaping \
    --output-dir output/navigator_ppo
```

Fine-tune to full game:

```bash
python scripts/train_navigator.py --preset phase2 \
    --combat-model output/combat_ppo_mixed/best_model/best_model.zip \
    --combat-value-shaping \
    --load-model output/navigator_ppo/final_model.zip \
    --output-dir output/navigator_ppo_full
```

### Key Navigator Flags

| Flag | Description |
| ---- | ----------- |
| `--combat-model` | Single frozen combat zip for all acts |
| `--combat-models` | Per-act routing: `0:path0,1:path1,2:path2` |
| `--combat-models-by-character` | Per-character: `Ironclad:path0,Silent:path1` |
| `--combat-value-shaping` | Draft pick shaping via combat PPO critic ΔV |
| `--draft-value-scale` | 0.1 (scale for draft ΔV bonus) |
| `--flawless-combat-bonus` | 0.003 | Bonus per combat cleared without HP loss |
| `--act-count` | Acts before win (1 = Act 1, 3 = full game) |
| `--preset` | `phase1` (2M, Act 1), `phase2` (5M, full), `full` (8M) |

### Legacy: Flat Meta-Policy (`train_full_run.py`)

Deprecated in favor of `train_navigator.py`. Still available:

```bash
python scripts/train_full_run.py --preset phase1 \
    --combat-model output/combat_ppo_mixed/best_model/best_model.zip \
    --n-envs 4 --output-dir output/run_ppo
```

Optional learned card picker (alternative to combat-critic):

```bash
python scripts/train_card_value.py --collect-episodes 5000
python scripts/train_full_run.py --preset phase1 \
    --combat-model output/combat_ppo_mixed/best_model/best_model.zip \
    --card-value-model output/card_value/best_model.pt
```

### Act Count Curriculum

The `--act-count` flag controls episode length (distinct from **combat curriculum** stages):

| Value | Description | Recommended for |
| ----- | ----------- | ----------------- |
| 1 | Act 1 only | Initial Navigator training |
| 2 | Acts 1–2 | Intermediate |
| 3 | Full game (Acts 1–3) | Final training |

### Post-Training Evaluation

```bash
python scripts/eval_full_run.py \
    --load-model output/navigator_ppo/best_model/best_model.zip \
    --combat-model output/combat_ppo_mixed/best_model/best_model.zip \
    --mcts
```

---

## 2.5 Combat MCTS (Inference Only)

Turn-bounded PUCT search improves combat decisions at inference time. **Not used during PPO training.**

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--mcts` | off | Enable MCTS |
| `--mcts-sims` | 128 | Simulations per decision |
| `--mcts-c-puct` | 1.5 | PUCT exploration constant |
| `--mcts-max-depth` | 15 | Max actions within one player turn |
| `--mcts-lookahead-turns` | 1 | Extra player turns to expand after enemy phase |
| `--mcts-time-budget` | 2.0s (bridge) / None (train eval) | Wall-clock cap per decision |
| `--mcts-dirichlet-alpha` | 0.3 | Dirichlet alpha for root exploration noise |
| `--mcts-dirichlet-epsilon` | 0.25 | Root prior noise mix weight; 0 disables |

Use MCTS when you want stronger combat play and can tolerate slower decisions (especially in the live game). Reduce `--mcts-sims` or increase `--mcts-time-budget` if decisions feel too slow.

---

## 3. Running the Benchmark

```bash
python scripts/benchmark.py
```

Reports ~1,200 episodes/sec, ~28,000 steps/sec, and random-play win rate.

---

## 4. Connecting to the Real Game

### Prerequisites

1. Trained model matching **obs v11 (1985 dims)** and policy type
2. Bridge mod installed (see [MOD_BUILD_GUIDE.md](MOD_BUILD_GUIDE.md))
3. Slay the Spire 2 running with the mod loaded
4. `STS2_BRIDGE_CHARACTER` set to the same character the model was trained on

### Starting the Agent

```bash
python -m sts2_env.bridge.agent_runner \
    --model-path output/combat_ppo_mixed/best_model/best_model.zip \
    --character Ironclad \
    --mcts --mcts-time-budget 2.0 \
    --verbose
```

### Agent Runner Options

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--model-path` | (required) | Path to trained MaskablePPO model (.zip) |
| `--character` | from env | Must match `STS2_BRIDGE_CHARACTER` and training character |
| `--combat-models-by-character` | — | Per-character routing for full-run delegates |
| `--host` | 127.0.0.1 | Bridge server hostname |
| `--port` | 9002 | Bridge server port |
| `--mcts` | off | Turn-bounded MCTS for combat decisions |
| `--mcts-sims` | 128 | MCTS simulations per decision |
| `--mcts-time-budget` | 2.0 | Seconds per combat decision |
| `--mcts-lookahead-turns` | 1 | Extra player turns after enemy phase |
| `--record-replay` | — | Write bridge replay JSON for parity testing |
| `--replay-factory` | — | `module:function` to build replay recorder |
| `--deterministic` | True | Greedy action selection (no randomness) |
| `--stochastic` | False | Stochastic selection |
| `--verbose` / `-v` | False | Log every action taken |
| `--log-level` | INFO | DEBUG/INFO/WARNING/ERROR |

### Step-by-Step Walkthrough

1. **Set character** (PowerShell):

   ```powershell
   $env:STS2_BRIDGE_CHARACTER = "Ironclad"
   ```

2. **Start the game.** Launch STS2. The mod shows "Running Modded" and starts TCP on port 9002.

3. **Start the agent** in a separate terminal with a model trained for the same character.

4. **Watch the game play.** Combat uses the trained PPO (optionally wrapped with MCTS). Non-combat phases use heuristics or a Navigator model if configured.

5. **Monitor output** with `--verbose`:

   ```text
   COMBAT [HP:72/80 E:3] -> PLAY BASH (idx=4) -> NIBBIT (idx=0)
   COMBAT [HP:72/80 E:0] -> END_TURN (round 1)
   MAP: choosing node 0
   CARD_REWARD: choosing option 0
   ```

### How the Agent Handles Different Phases

| Phase | Strategy | Source |
| ----- | -------- | ------ |
| Combat | Trained MaskablePPO (+ optional MCTS) | `agent_runner.py` / combat model |
| Map navigation | Learned path preferences | Navigator PPO (`train_navigator.py`) or heuristics |
| Card rewards | Combat-critic ΔV or learned card-value net | `--combat-value-shaping` / `train_card_value.py` or rules |
| Rest sites | Rest if low HP, else upgrade | `noncombat_heuristics.py` |
| Shop | Buy relics/cards/potions before leaving | Navigator PPO or heuristics |
| Events | Pick from enabled options | Navigator PPO or heuristics |
| Treasure / Boss relics | Collect or pick relic | Navigator PPO or heuristics |

For combat-only evaluation, pass a combat model. For full-run play in the real game, wire a Navigator checkpoint or rely on the built-in heuristics for non-combat phases.

---

## 5. Interpreting Agent Output

### Training Metrics (TensorBoard)

| Metric | What It Means |
| ------ | ------------- |
| `rollout/ep_rew_mean` | Average episode reward (shaped during training) |
| `eval/mean_reward` | Sparse eval reward (should trend toward +1.0) |
| `curriculum/gate_win_rate` | Curriculum promotion win rate |
| `curriculum/gate_hp_ratio` | Mean HP retention on gate encounters |
| `eval/win_rate` | Full-run win rate (Navigator / legacy full-run) |
| `train/entropy_loss` | Policy entropy (should decrease slowly) |

### Evaluation Output

Combat (after `train_combat.py`):

```text
--- Final Evaluation ---
Episodes:    100
Win rate:    92.0%
Avg reward:  0.847
```

Navigator / full-run:

```text
Episodes:         100
Win rate:         0.0%    # full-run win rate still challenging at 1M steps
Avg floors:       8.9
Max floors:       15
```

### Real-Game Logs

- **Connection:** `Connected to STS2 bridge at 127.0.0.1:9002`
- **Combat:** `COMBAT [HP:65/80 E:2] -> PLAY INFLAME -> N/A`
- **Warnings:** `No valid actions! Defaulting to END_TURN.` (should be rare)
- **Game over:** `Game over! Result: win` or `Game over! Result: death`

---

## 6. Known Limitations

### Simulator vs Real Game

The headless simulator mirrors decompiled game logic, but edge cases may differ. See [KNOWN_ISSUES.md](KNOWN_ISSUES.md) for the full list.

### Observation and Policy Mismatch

Loading a model trained with `--policy attention` while the runner expects `mlp` (or using a pre-v11 checkpoint) produces poor play or load failures. Always match character, obs version (v11 / 1985 dims), and policy type. Navigator checkpoints must use the 166-dim obs v2 layout.

### Non-Combat Phases

The combat model only handles combat. For strategic play, train a Navigator (`train_navigator.py`) or accept heuristic non-combat behavior.

### Characters

All 5 characters are implemented. Training and bridge character must match (`--character` / `--characters` / `STS2_BRIDGE_CHARACTER`).

### Ascension

Supported by the simulator but not enabled by default in training scripts.
