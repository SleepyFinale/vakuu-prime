# Training Guide

How to train RL agents on the STS2 headless simulator.

---

## Hardware Requirements

| Component | Minimum | Recommended |
| --------- | --------- | ------------- |
| CPU | 4 cores | 8+ cores (for parallel envs) |
| RAM | 8 GB | 16 GB |
| GPU | Not required (CPU training works) | NVIDIA RTX 4070 Ti SUPER or better |
| Disk | 2 GB free | 10 GB (for logs + checkpoints) |
| Python | 3.11+ | 3.12 |

GPU is recommended for faster training but not strictly required. The simulator itself is CPU-bound (pure Python); the GPU accelerates only the neural network forward/backward passes in PyTorch.

**Tested hardware:** RTX 4070 Ti SUPER, 32 GB RAM, 12-core CPU. Combat training completes in ~27 minutes for 2M steps.

---

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd sts2-rl-agent

# Install with training dependencies (PyTorch, SB3, sb3-contrib, torch-geometric)
pip install -e ".[train]"

# Verify installation
python scripts/benchmark.py
```

On Windows, if `torch-geometric` fails to install from PyPI alone, install PyTorch
first, then follow the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
for a wheel matching your torch/CUDA version. GNN training (`--policy gnn`) requires
this package; attention and MLP policies do not.

Expected benchmark output:

```text
Episodes:       1000
Total steps:    28101
Time:           0.78s
Episodes/sec:   1276
Steps/sec:      28101
```

---

## Combat-Only Training

Train an agent to play single combat encounters. Default character is Ironclad; all five characters are supported.

### Command (Act 1 only, Ironclad)

```bash
python scripts/train_combat.py \
    --acts 0 \
    --total-timesteps 2000000 \
    --n-envs 8 \
    --output-dir output/combat_ppo
```

### Command (per-character)

```bash
python scripts/train_combat.py --character Silent --output-dir output/combat_ppo_silent
python scripts/train_combat.py --character Defect --output-dir output/combat_ppo_defect
```

### Command (mixed characters, single model)

```bash
python scripts/train_combat.py \
    --characters all \
    --total-timesteps 4000000 \
    --output-dir output/combat_ppo_mixed_chars
```

### Command (acts 0–2 mixed, recommended for full-run delegate)

```bash
python scripts/train_combat.py \
    --acts 0,1,2 \
    --total-timesteps 3000000 \
    --n-envs 8 \
    --output-dir output/combat_ppo_mixed_attn
```

### Command (self-attention policy, default)

Combat training now defaults to a **Transformer feature extractor** over entity
tokens (player, cards, enemies, relics, piles, character mechanics). Use
`--policy mlp` for the legacy flat MLP baseline.

```bash
python scripts/train_combat.py \
    --acts 0,1,2 \
    --policy attention \
    --d-model 128 \
    --n-heads 4 \
    --n-layers 2 \
    --features-dim 256 \
    --output-dir output/combat_ppo_mixed_attn
```

### Command (GNN policy)

Graph policy uses the same 268-dim obs v3 tokens with **structural edges**
(card→enemy from static target metadata, enemy→player on attack intents, relic→player).
Requires `torch-geometric`.

```bash
python scripts/train_combat.py \
    --acts 0,1,2 \
    --policy gnn \
    --d-model 128 \
    --n-heads 4 \
    --n-layers 2 \
    --features-dim 256 \
    --output-dir output/combat_ppo_mixed_gnn
```

Boss encounters are excluded from the RL pool (they are scripted in full runs).

### Flags

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--acts` | `0` | Encounter acts: `0`, `0,1,2`, or `all` |
| `--character` | `Ironclad` | Single character for all episodes |
| `--characters` | — | Mixed pool: `Ironclad,Silent` or `all` (mutually exclusive with `--character`) |
| `--act1-biome` | random | Act 1 biome for index 0: random (both biomes), overgrowth, or underdocks |
| `--total-timesteps` | 2M (act 0) / 3M (mixed acts) / 4M (mixed chars) | Total environment steps |
| `--n-envs` | 4 | Parallel environments (set to CPU core count) |
| `--lr` | 3e-4 | Learning rate |
| `--batch-size` | 256 | Minibatch size |
| `--n-steps` | 2048 | Steps per rollout per env |
| `--n-epochs` | 10 | PPO epochs per update |
| `--gamma` | 0.99 | Discount factor |
| `--ent-coef` | 0.01 | Entropy coefficient (exploration) |
| `--eval-freq` | 10,000 | Evaluate every N steps |
| `--eval-episodes` | 20 | Episodes per evaluation |
| `--output-dir` | `output/combat_ppo_attn` or `combat_ppo_mixed_attn` | Model output directory |
| `--policy` | `attention` | Feature extractor: `attention` (Transformer), `gnn` (DenseGAT), or `mlp` (SB3 default) |
| `--d-model` | 128 | Token embedding dimension (attention and GNN) |
| `--n-heads` | 4 | Attention/GAT heads |
| `--n-layers` | 2 | Transformer encoder or GAT layers |
| `--features-dim` | 256 | Pooled features fed to pi/vf MLP heads |

### Expected Results

| Timesteps | Win Rate | Training Time (8 envs, GPU) |
| --------- | -------- | -------------------------- |
| 100k | ~70% | ~1.5 min |
| 500k | ~85% | ~7 min |
| 2M | ~92% | ~27 min |

The random baseline win rate is approximately 63.4% for Act 1 Ironclad encounters. The agent significantly outperforms random within 500k steps.

### Evaluation

After training, evaluate the saved model:

```bash
python scripts/benchmark.py  # Quick throughput check

# Or use the built-in evaluation in train_combat.py:
# The script automatically runs 100 evaluation episodes after training.
```

The training script saves several model checkpoints:

- `output/combat_ppo/final_model.zip` -- model at the end of training
- `output/combat_ppo/best_model/best_model.zip` -- best model during training (based on eval callback)
- `output/combat_ppo/checkpoints/checkpoint_<steps>_steps.zip` -- periodic resumable checkpoints (every `--checkpoint-freq` steps, newest `--keep-checkpoints` retained)
- `output/combat_ppo/interrupted_checkpoint.zip` -- written when you stop training with Ctrl+C

### Pause and Resume

Long runs can be stopped and continued over several days. Both PPO trainers
(`train_combat.py` and `train_full_run.py`) support this:

```bash
# Day 1: start a run (a run_config.json records the settings)
python scripts/train_combat.py --output-dir output/combat_ppo

# Press Ctrl+C whenever you need to stop. The current progress is saved to
# output/combat_ppo/interrupted_checkpoint.zip and a resume command is printed.

# Day 2: resume from the latest checkpoint in the same output dir
python scripts/train_combat.py --resume --output-dir output/combat_ppo

# Train further than the original target by raising --total-timesteps
python scripts/train_combat.py --resume --output-dir output/combat_ppo --total-timesteps 6000000
```

`--resume` automatically loads the most recent checkpoint (preferring the
interrupted checkpoint, then the newest periodic checkpoint, then `final_model`
or `best_model`) and replays the original settings from `run_config.json`, so
you only need to pass `--output-dir`. It continues the timestep counter rather
than restarting. `--resume` and `--load-model` are mutually exclusive; keep
using `--load-model` for curriculum fine-tuning into a *new* output directory.

The same idea applies to the card-value trainer. Data collection writes a
`*.partial.npz` every 100 episodes (`--resume-collection` continues it), and
supervised training writes `training_checkpoint.pt` each epoch (`--resume`
continues from the next epoch).

### TensorBoard

Training logs are saved for TensorBoard:

```bash
tensorboard --logdir output/combat_ppo/tb_logs
```

Key metrics to watch:

- `rollout/ep_rew_mean` -- average episode reward (should trend toward +1.0)
- `rollout/ep_len_mean` -- average episode length (should decrease as agent wins faster)
- `train/entropy_loss` -- exploration (should decrease over time but not collapse to 0)
- `train/policy_gradient_loss` -- PPO policy loss
- `eval/mean_reward` -- evaluation reward (most reliable metric)

### Hyperparameter Tuning Tips

1. **Learning rate:** 3e-4 works well. Lower (1e-4) if training is unstable. Higher (1e-3) can speed up early learning but may diverge.

2. **Entropy coefficient:** 0.01 is a good default. Increase to 0.05 if the agent gets stuck in a local optimum (always ending turn). Decrease to 0.001 once the agent has learned basic play patterns.

3. **Batch size:** 256 is optimal for combat training. Larger batches (512, 1024) can stabilize gradients but slow down updates.

4. **n_steps:** 2048 per env per rollout. Since combat episodes average ~28 steps, this gives ~73 episodes per rollout per env. Shorter values (512, 1024) give more frequent updates but higher variance.

5. **n_envs:** Set to your CPU core count for maximum throughput. The simulator is CPU-bound, so more parallel envs = more samples per second.

6. **gamma:** 0.99 for combat (short episodes). Use 0.995+ for full-run training (long episodes).

---

## Full-Run Training

Train a **meta-policy** for map navigation, card rewards, events, shops, and rest sites. Combat is delegated to a pre-trained combat MaskablePPO (`STS2HierarchicalRunEnv`).

### Prerequisites

Train mixed-act combat first (recommended):

```bash
python scripts/train_combat.py --acts 0,1,2 --total-timesteps 3000000 \
    --output-dir output/combat_ppo_mixed
```

### Full-Run Command (hierarchical + heuristics)

```bash
python scripts/train_full_run.py --preset phase1 \
    --combat-model output/combat_ppo_mixed/best_model/best_model.zip \
    --n-envs 4 --output-dir output/run_ppo
```

### Key Flags

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--preset` | - | `phase1` (2M, act 1), `phase2` (5M, full), `full` (8M, full) |
| `--total-timesteps` | 2,000,000 | Meta-policy training steps |
| `--combat-model` | `output/combat_ppo_mixed/...` | Single frozen combat PPO |
| `--combat-models` | - | Per-act models: `0:path0,1:path1,2:path2` |
| `--combat-models-by-character` | - | Per-character models: `Ironclad:path0,Silent:path1` |
| `--character` | `Ironclad` | Character for the run |
| `--characters` | - | Mixed-character runs: `all` or comma-separated list |
| `--act-count` | 1 | Acts before curriculum win (1 = Act 1, 3 = full game) |
| `--act1-biome` | random | Act 1 biome: random, overgrowth, or underdocks (full-run training) |
| `--reward-shaping` | True | Floor/combat-clear/HP shaping (see `run_reward.py`) |
| `--no-reward-shaping` | - | Sparse terminal reward only |
| `--no-noncombat-heuristic` | - | Disable auto card/boss/rest picks (on by default) |
| `--card-value-model` | - | Learned card picker (`.pt` from `train_card_value.py`) |
| `--no-card-value-model` | - | Force rule-based card rewards |
| `--eval-with-heuristics` | - | Eval with assisted card/boss/rest picks |
| `--load-model` | - | Fine-tune from a saved meta-policy zip |
| `--no-combat-delegate` | - | Ablation: flat `STS2RunEnv` without combat bot |
| `--max-steps` | 10000 | Max meta-decisions per episode |
| `--gamma` | 0.995 | Higher discount for long episodes |
| `--ent-coef` | 0.02 | More exploration for complex decision space |
| `--baseline-only` | - | Only run random baseline (no training) |

The meta-policy uses a larger network (256x256 pi/vf) than default combat training.

### Reward shaping scales (defaults)

| Signal | Scale |
| ------ | ----- |
| Floor advanced | +0.05 per floor |
| Combat cleared | +0.10 |
| HP lost after combat | up to -0.20 |
| Run win / death | +1 / -1 |

### Non-combat heuristics and card-value network

During training, boss relics and rest sites use rules in [`noncombat_heuristics.py`](../sts2_env/gym_env/noncombat_heuristics.py). Card rewards can use a **learned card-value network** instead:

```bash
python scripts/train_card_value.py --collect-episodes 5000
python scripts/train_full_run.py --preset phase1 \
    --combat-model output/combat_ppo_mixed/best_model/best_model.zip \
    --card-value-model output/card_value/best_model.pt
```

Training uses outcome-weighted labels (winning runs weight 1.0, losses 0.3). Evaluation uses sparse rewards; `--eval-with-heuristics` reports assisted win rate.

### Win-rate monitoring

`RunWinRateCallback` logs `eval/win_rate`, `eval/mean_floors`, and `eval/mean_meta_steps` to TensorBoard during training. After training:

```bash
python scripts/eval_full_run.py --load-model output/run_ppo/best_model/best_model.zip \
    --combat-model output/combat_ppo_mixed/best_model/best_model.zip \
    --card-value-model output/card_value/best_model.pt
```

### Curriculum Learning

```bash
# Phase 1: Act 1 only (2M meta steps)
python scripts/train_full_run.py --preset phase1 \
    --combat-model output/combat_ppo_mixed/best_model/best_model.zip

# Phase 2: Full game fine-tune (5M meta steps)
python scripts/train_full_run.py --preset phase2 \
    --combat-model output/combat_ppo_mixed/best_model/best_model.zip \
    --load-model output/run_ppo/final_model.zip

# Optional: per-act combat specialists
python scripts/train_full_run.py --preset full \
    --combat-models "0:output/combat_act0.zip,1:output/combat_act1.zip,2:output/combat_act2.zip"
```

### Full-Run Expected Results

**Random baseline** (Act 1 only):

```text
Win rate:         0%
Avg floors:       3.9
Max floors:       8
```

**Trained agent** (1M steps, Act 1 only):

```text
Win rate:         0%
Avg floors:       8.9
Max floors:       15
```

The agent learns to progress further through Act 1 but does not yet achieve a positive win rate. Full-run training is significantly harder than combat-only due to:

1. **Sparse reward:** Only +1 for winning the entire run, -1 for death. No intermediate signal for floor progression (unless reward shaping is enabled).
2. **Long episodes:** A full Act 1 run involves 15+ floors, each with its own combat, map choice, and reward decisions. Episodes can span thousands of steps.
3. **Multi-phase action space:** The agent must learn to handle 8 different game phases with very different semantics under a single Discrete(100) action space.
4. **Compounding errors:** A bad card reward choice in floor 3 might not manifest as a loss until floor 12.

### Challenges and Mitigations

| Challenge | Current State | Future Direction |
| --------- | ------------ | ---------------- |
| Sparse reward | Shaping in `run_reward.py` | Hindsight experience replay |
| Long episodes | Meta-steps + act-count curriculum | Further episode compression |
| Multi-phase | Hierarchical combat delegate + meta PPO | Further shop/event learned policies |
| Simulation speed | ~1,200 combats/sec | Cython acceleration of core loop (target: 10k+/sec) |
| Card selection | Meta policy via combat-slice choices | Dedicated card evaluation network |

---

## Training Results Summary

| Metric | Random | Combat-Trained (2M) | Run-Trained (1M) |
| ------ | ------ | ------------------- | ---------------- |
| Combat win rate | 63.4% | 92% | N/A (not isolated) |
| Run win rate | 0% | N/A (combat only) | 0% |
| Avg floors (Act 1 run) | 3.9 | N/A | 8.9 |
| Training time | - | ~27 min (GPU) | ~2 hrs (GPU) |

---

## Future Improvements

1. **Hierarchical policy:** Separate the policy into a high-level strategy selector (which card to add, which path to take) and a low-level combat executor. Train them independently and combine.

2. **Cython acceleration:** Port `core/combat.py`, `core/hooks.py`, and `core/damage.py` to Cython for 5-10x speedup. The main bottleneck is Python interpreter overhead in the hot loop.

3. **Reward shaping:** Implemented in `sts2_env/gym_env/run_reward.py`. Possible extensions: deck-quality score, gold-efficiency metric.

4. **Population-based training (PBT):** Use Ray RLlib to train multiple agents with different hyperparameters in parallel, automatically tuning learning rate, entropy, and gamma.

5. **Self-play against harder encounters:** Start with weak encounters, progressively introduce harder ones as the agent improves.

6. **Imitation learning:** If expert human replays become available, pre-train the policy with behavioral cloning before RL fine-tuning.

7. **Multi-character support:** Implemented via `--character` / `--characters all` in `train_combat.py` and `--combat-models-by-character` in `train_full_run.py`. Combat observation is 268 dims (obs v3: mechanics + relic slots); retrain after upgrading from 148-dim or 131-dim checkpoints.

8. **Entity-based policies:** `--policy attention` (`CombatAttentionExtractor`) and `--policy gnn` (`CombatGNNExtractor`) share tokenization via `sts2_env/training/entity_tokens.py`. GNN uses structural edges from `entity_graph.py`.

---

## Live bridge evaluation

After training a character-specific combat model, evaluate it against the real game:

1. Set the bridge mod character before launching STS2 (defaults to Ironclad):

   ```powershell
   $env:STS2_BRIDGE_CHARACTER = "Defect"
   ```

2. Build and install the bridge mod ([docs/MOD_BUILD_GUIDE.md](MOD_BUILD_GUIDE.md)), then start the game.

3. Run the agent with a model trained for the same character:

   ```bash
   python -m sts2_env.bridge.agent_runner \
       --model-path output/combat_ppo_defect/best_model/best_model.zip \
       --character Defect \
       --verbose
   ```

   For full-run delegates, pass `--combat-models-by-character` with per-character
   checkpoint paths so combat phases route to the correct model.

The mod emits `character_id`, `stars`, `orb_queue`, `osty`, and `relics` in
combat JSON; the Python adapter encodes them into observation dims 131–267.
Mismatch between `STS2_BRIDGE_CHARACTER` and the loaded model produces incorrect
one-hot/mechanics/relic features and poor play.

See [docs/BRIDGE_LIVE_SMOKE.md](BRIDGE_LIVE_SMOKE.md) for the offline smoke gate
and [docs/AGENT_USAGE_GUIDE.md](AGENT_USAGE_GUIDE.md) for agent runner options.

**Observation v3 note:** Combat `OBS_SIZE` is now **268** (148 combat v2 + 120 relic slots). Existing 148-dim and 131-dim PPO checkpoints are incompatible. Retrain combat models with the updated encoder and `--policy attention` (or `--policy mlp` for a flat baseline on the new obs size).
