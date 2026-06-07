# STS2 RL Agent

A reinforcement learning agent for **Slay the Spire 2**, built on a high-performance headless combat simulator and a Gymnasium training environment. Includes a C# bridge mod for connecting the trained agent to the real game.

The RL stack supports **self-attention policies** over entity tokens, **draw-pile memory** in observations, **non-linear HP reward shaping**, **staged combat curriculum**, **hierarchical Combat + Navigator controllers**, and optional **turn-bounded MCTS** at inference.

## Architecture

```text
+-----------------------------------------------------------------------------+
|  Headless Python Simulator (sts2_env/)                                      |
|                                                                             |
|  +----------------+  +----------------+  +----------------------------------+ |
|  | Core Engine    |  | Game Content   |  | Gym Environments                 | |
|  | combat.py      |  | 578 cards      |  | combat_env.py   Discrete(115)  | |
|  | creature.py    |  | 260 powers     |  | run_env.py      Discrete(157)  | |
|  | hooks.py       |  | 121 monsters   |  | navigator_env.py  strategic obs | |
|  | damage.py      |  | 290 relics     |  | observation.py  1985-dim obs v11 | |
|  | rng.py         |  | 63 potions     |  | pile_distribution.py           | |
|  +-------+--------+  +-------+--------+  +-----------+--------------------+ |
|          |                    |                        |                      |
|  +-------+--------------------+------------------------+                      |
|  | training/  attention_extractor, gnn_extractor, combat_curriculum        | |
|  | search/    mcts_combat, mcts_agent, policy_guide (inference only)         | |
|  +---------------------------------------------------------------------------+
+-----------------------------------------------------------------------------+
           |                                              |
           v                                              v
+---------------------------+              +----------------------------------+
| Training Pipeline         |              | Bridge to Real Game              |
| MaskablePPO (SB3)         |              | bridge_mod/ (C#/Godot)             |
| train_combat.py  (tactical)|--model----->| agent_runner.py (Python)         |
| train_navigator.py (strat)|              | TCP JSON protocol, optional MCTS |
| train_full_run.py (legacy)|              +----------------------------------+
+---------------------------+
```

## Project Stats

| Metric                        | Value                                             |
| ----------------------------- | ------------------------------------------------- |
| Python source files           | 342 (tracked)                                     |
| Lines of code                 | ~50,000                                           |
| Test functions                | 5,387+ (166 test files)                           |
| Cards implemented             | 578                                               |
| Powers implemented            | 260                                               |
| Monsters implemented          | 121                                               |
| Relics implemented            | 290                                               |
| Potions implemented           | 63                                                |
| Playable characters           | 5 (Ironclad, Silent, Defect, Necrobinder, Regent) |
| Combat observation (obs v11)  | 1985 dimensions                                   |
| Default policy                | Transformer self-attention (`--policy attention`) |
| Simulation speed              | ~1,200 combats/sec, ~28,000 steps/sec             |
| Combat win rate (trained PPO) | ~92% (Act 1 Ironclad, 2M steps)                   |

## RL Feature Overview

### Self-Attention Observation Space (obs v11)

Combat observations are a **1985-dimensional** `float32` vector encoded by [`sts2_env/gym_env/observation.py`](sts2_env/gym_env/observation.py). Attention and GNN policies parse this flat vector into **57 entity tokens** via [`sts2_env/training/entity_tokens.py`](sts2_env/training/entity_tokens.py): player, piles, character mechanics, 10 hand cards (9 features each: id, cost, damage, block, is_attack, is_power, exhaust, retain, hit_count), 5 enemies, 30 relic slots (5 features each: id, rarity, enabled, is_used_up, counter_norm), and 9 potion slots.

| Version | Size | Added features |
| ------- | ---- | -------------- |
| v2 | 148 | Character mechanics (stars, orbs, Osty) |
| v3 | 268 | 30 relic slots (120 dims) |
| v4 | 294 | Draw-pile memory (26 dims replacing 3 reserved zeros) |
| v5 | 321 | 9 potion slots (27 dims: id, rarity, can_use_in_combat) |
| v6 | 1908 | All 268 `PowerId` values on player and each enemy slot |
| v7 | 1948 | Hand cards expanded to 9 features (exhaust, retain, power, hit count) |
| v8 | 1978 | Relic slots expanded to 5 features (counter_norm for count-based relics) |
| v9 | 1983 | Known draw order encodes card_id_norm + type per top-5 slot (+5 pile-memory dims) |
| v10 | 1984 | ascension/20 in player core |
| v11 | 1985 | turn_count/20 in player core |

Policy architectures (wired in [`scripts/train_combat.py`](scripts/train_combat.py)):

| `--policy` | Extractor | Notes |
| ---------- | --------- | ----- |
| `attention` (default) | `CombatAttentionExtractor` | 2-layer masked Transformer over entity tokens |
| `gnn` | `CombatGNNExtractor` | DenseGAT over structural graph; requires `torch-geometric` |
| `mlp` | SB3 default | Flat MLP on 1985-dim vector (baseline) |

**Checkpoint compatibility:** `mlp`, `attention`, and `gnn` checkpoints are **not interchangeable**. Obs v10 and earlier checkpoints are obsolete — retrain on obs v11.

### Draw Pile Memory

Dims 365–395 within the 37-dim pile block (see `TOKEN_SLICES["piles"]` in `observation.py`) are encoded by [`sts2_env/gym_env/pile_distribution.py`](sts2_env/gym_env/pile_distribution.py):

| Block | Dims | Content |
| ----- | ---- | ------- |
| Unseen composition | 5 | Attack/Skill/Power/Status/Curse fractions in draw+discard+play |
| Next-draw probabilities | 5 | P(at least one Attack/Skill/Power), expected Attack/Skill draws |
| Known draw order | 10 | card_id_norm + type encoding per top-5 draw pile slot |
| Shuffle uncertainty | 1 | Flag when next draw exceeds visible draw pile |
| High-value heuristics | 6 | Heavy attacks, powers, 0-cost, rare+, strike/defend fractions |
| Watchlist groups | 4 | Binary presence of power/finisher/setup/aoe card groups |

Plus 3 count features (draw/discard/exhaust /20) and 3 reserved padding = **37 pile dims total**. Watchlist groups load from [`docs/PILE_WATCHLIST.json`](docs/PILE_WATCHLIST.json) (regenerated by sync; see [docs/PATCH_SYNC.md](docs/PATCH_SYNC.md)). The bridge mod must serialize `draw_pile`, `discard_pile`, and `play_pile` card arrays for live-game parity (see [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md)).

### Non-Linear HP Reward Shaping and Combat Micro-Rewards

Implemented in [`sts2_env/gym_env/reward_shaping.py`](sts2_env/gym_env/reward_shaping.py) and applied by [`reward.py`](sts2_env/gym_env/reward.py) (combat) and [`run_reward.py`](sts2_env/gym_env/run_reward.py) (full run).

**Non-linear HP penalty:** damage near death costs more — weight = `exp(steepness * (1 - hp_ratio_before))`, capped at 0.2 per event (default `--hp-steepness 3.0`).

**Combat micro-rewards** (per step, capped at 0.05):

| Signal | Default scale |
| ------ | ------------- |
| Vulnerable on enemy (player-applied) | +0.02 × sublinear marginal stacks (`Δ**0.6`; first stack unchanged) |
| Weak on enemy (player-applied) | +0.02 × sublinear marginal stacks (`Δ**0.6`; first stack unchanged) |
| Block vs enemy attack | +0.001 per HP blocked |

**Run-level shaping:** floor progress (+0.01/floor), combat clear (+0.005), run-end HP efficiency bonus on win (up to +0.15), same non-linear HP penalty on macro transitions. Kill reward in run env is +0.003/enemy (combat-only training keeps +0.05).

Evaluation environments use **sparse rewards** (`--no-reward-shaping`) so `eval/mean_reward` stays near ±1.

### Hierarchical RL: Combat + Navigator Controllers

Two distinct controllers separate tactical combat from strategic run decisions:

| Controller | Role | Training script |
| ---------- | ---- | --------------- |
| **Combat Agent** | Play cards/potions in a fight | [`scripts/train_combat.py`](scripts/train_combat.py) |
| **Navigator Agent** | Map, drafting, shops, events, rest, boss relics | [`scripts/train_navigator.py`](scripts/train_navigator.py) |

Combat is always delegated to a **frozen** combat MaskablePPO (actions 0–114). The Navigator never sees the combat action slice during fights. Strategic observations are encoded separately in [`navigator_observation.py`](sts2_env/gym_env/navigator_observation.py) (166 dims, obs v2: run context, phase one-hot, map branches, path topology, combat-aligned card offers, shop gold/affordability, phase options, deck value). **Existing Navigator checkpoints trained on the prior 164-dim layout are incompatible and require retraining.**

**Combat-critic draft scoring** ([`combat_value.py`](sts2_env/gym_env/combat_value.py)): the frozen PPO value head scores hypothetical decks against sampled elite encounters. Use `--combat-value-shaping` during Navigator training for draft ΔV bonuses.

[`scripts/train_full_run.py`](scripts/train_full_run.py) trains a legacy flat meta-policy on `STS2HierarchicalRunEnv` and is **deprecated** in favor of `train_navigator.py`.

### Combat MCTS (Turn-Bounded MPC)

Inference-only PUCT search in [`sts2_env/search/`](sts2_env/search/). Not used during PPO training.

1. Root = current combat snapshot (`combat_clone.clone_combat_state()`).
2. Each simulation: PUCT child selection, expand with masked PPO action priors (root priors mixed with Dirichlet noise for policy-independent exploration), roll out with `apply_combat_action_for_search()` until `--mcts-max-depth` actions are played in the current player turn.
3. On `END_TURN`: simulate the enemy phase, then optionally expand one extra player turn (`--mcts-lookahead-turns`, default 1). Truncated turn-0 leaves fast-forward through END_TURN + enemy before critic eval (`leaf_eval="post_enemy_critic"`).
4. Leaf value from PPO critic (`predict_combat_values()`).
5. Pick highest visit-count root child (temperature 0; ties broken by Q, then non–end-turn actions).

| Entry point | Flag |
| ----------- | ---- |
| Post-training combat eval | `train_combat.py --mcts` |
| Full-run eval | `eval_full_run.py --mcts` |
| Live bridge | `agent_runner.py --mcts` (default 2s time budget) |

### Combat Curriculum Learning

Staged encounter and deck widening in [`sts2_env/training/combat_curriculum.py`](sts2_env/training/combat_curriculum.py) with tier-based HP-retention promotion gates. Early stages gate on a fixed easy encounter subset; elite-profile stages gate on Act 1 elites with relaxed thresholds.

| Stage | Encounters | Characters | Gate profile |
| ----- | ---------- | ---------- | ------------ |
| `easy_pair` | Jaw Worm + Cultists | Ironclad, Regent | Easy: 98% win, 92% HP |
| `act1_weak` | All Act 1 weak | Ironclad, Regent | Easy: 95% win, 88% HP |
| `act1_normal` | Weak + normal | Ironclad, Regent | Easy: 90% win, 82% HP |
| `act1_elite` | + elites | Ironclad, Regent | Elite: 92% win, 80% HP; 10% hard-start |
| `complex_decks` | Act 1 all tiers | + Necrobinder | Elite: 88% win, 75% HP |
| `mixed_acts` | Acts 0–2 | All five | No gate (terminal stage in `--curriculum full`) |
| `recovery` | Elites + bosses | Ironclad, Regent | No gate; 100% compromised starts; excluded from `--curriculum full` |

Parallel workers sync stage index via `output_dir/curriculum_state.json`. TensorBoard logs `curriculum/gate_win_rate`, `curriculum/gate_hp_ratio`, active thresholds (`curriculum/gate_min_*`), and stall fallback metrics (`curriculum/forced_promotion` when auto-promote force-advances after 3× the median stage budget).

**Distinct from act-count curriculum:** `--act-count` in Navigator/full-run training controls how many acts per episode (1 = Act 1 only, 3 = full game).

## Quick Start

### Prerequisites

- **Python 3.11+** (3.12 recommended)
- **pip** (included with Python)
- For training: a CUDA-capable GPU is recommended but not required
- For GNN policy: `torch-geometric` (included in `[train]` extras)
- For real-game bridge: .NET 9 SDK, Godot 4.5.1 Mono, Slay the Spire 2 (Steam)

### Install

```bash
git clone <repo-url>
cd sts2-rl-agent

# Core simulator only
pip install -e .

# With training dependencies (PyTorch, SB3, sb3-contrib, torch-geometric)
pip install -e ".[train]"

# With dev dependencies (pytest)
pip install -e ".[dev]"
```

### Run Benchmark

Measure simulation throughput with random actions:

```bash
python scripts/benchmark.py
```

Expected output on a modern CPU:

```text
Episodes:       1000
Total steps:    28101
Time:           0.78s
Episodes/sec:   1276
Steps/sec:      28101
```

### Play a Full Run in the Terminal

```bash
python -m sts2_env.cli.play_run
python -m sts2_env.cli.play_run --character Silent --seed 123 --ascension 0
```

Without `--character`, the CLI asks you to pick a character first. Shortcuts: `a` (first action), `c` (confirm/skip), `q` (quit).

### Play a Full Run in the Browser

```bash
python -m sts2_env.web.play_run --port 8765
```

Open [http://127.0.0.1:8765/](http://127.0.0.1:8765/).

### Train a Combat Agent (recommended: curriculum)

```bash
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

Key combat training flags:

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--total-timesteps` | 2M (act 0) / 500K (curriculum stage) | Total environment steps |
| `--n-envs` | 4 | Parallel environments (use CPU cores) |
| `--policy` | `attention` | `attention`, `gnn`, or `mlp` |
| `--curriculum` | — | Stage name or `full` for full sequence |
| `--auto-promote` | off | Advance when gate metrics pass |
| `--characters` | — | Mixed pool: `Ironclad,Regent` or `all` |
| `--acts` | `0` | Encounter acts: `0`, `0,1,2`, or `all` |
| `--reward-shaping` | True | Non-linear HP + combat micro-rewards |
| `--hp-steepness` | 3.0 | Exponential HP penalty steepness |
| `--resume` | — | Continue from checkpoint in same output dir |
| `--eval-only` | off | MCTS/post-train eval without training (requires `--load-model`) |
| `--flawless-bonus` | 0.1 | Bonus on flawless combat wins (no HP lost) |
| `--mcts` | off | Post-training eval with MCTS (inference only) |

See [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) for the full flag reference.

### Train a Full-Run Agent (recommended: Navigator)

```bash
# 1. Combat delegate (mixed acts, all characters)
python scripts/train_combat.py --acts 0,1,2 --characters all \
    --total-timesteps 3000000 --output-dir output/combat_ppo_mixed

# 2. Navigator (preferred)
python scripts/train_navigator.py --preset phase1 \
    --combat-model output/combat_ppo_mixed/best_model/best_model.zip \
    --combat-value-shaping --output-dir output/navigator_ppo
```

Legacy flat meta-policy (deprecated):

```bash
python scripts/train_full_run.py --preset phase1 \
    --combat-model output/combat_ppo_mixed/best_model/best_model.zip \
    --n-envs 4 --output-dir output/run_ppo
```

### Connect to Real Game

1. Build and install the bridge mod (see [docs/MOD_BUILD_GUIDE.md](docs/MOD_BUILD_GUIDE.md))
2. Set `STS2_BRIDGE_CHARACTER` to match your trained model (default `Ironclad`)
3. Start Slay the Spire 2
4. Run the agent:

```bash
python -m sts2_env.bridge.agent_runner \
    --model-path output/combat_ppo_mixed/best_model/best_model.zip \
    --mcts --mcts-time-budget 2.0 \
    --verbose

# Optional: record bridge replay for parity testing
python -m sts2_env.bridge.agent_runner \
    --model-path output/combat_ppo_mixed/best_model/best_model.zip \
    --record-replay artifacts/smoke_combat.json
```

See [docs/AGENT_USAGE_GUIDE.md](docs/AGENT_USAGE_GUIDE.md) for details.

## Recommended Training Workflow

End-to-end path from scratch to a full-run agent:

1. **Combat curriculum** — `--curriculum easy_pair` through `mixed_acts`, or `--curriculum full --auto-promote` (~5M steps). Gate thresholds are tier-based (easy stages up to 98%/92%; elite stages 92%/80% on Act 1 elites). Stall fallback force-promotes after 3× the median stage budget if gates never pass.
2. **Mixed-act combat** — `--acts 0,1,2 --characters all` for a combat delegate that handles all acts and characters (~3M steps). Target: ~92% Act 1 win rate at 2M steps.
3. **Navigator phase 1** — `train_navigator.py --preset phase1` with frozen combat model and `--combat-value-shaping` (Act 1 only, 2M meta steps).
4. **Navigator phase 2** — `train_navigator.py --preset phase2 --load-model output/navigator_ppo/final_model.zip` (full game, 5M meta steps).
5. **Bridge evaluation** — optional `--mcts` on `agent_runner.py` for stronger combat decisions in the real game.

Expected combat win rates by timesteps (8 envs, GPU):

| Timesteps | Win Rate | Training Time |
| --------- | -------- | ------------- |
| 100k | ~70% | ~1.5 min |
| 500k | ~85% | ~7 min |
| 2M | ~92% | ~27 min |

Full-run training is significantly harder than combat-only. See [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) for hyperparameter tuning, TensorBoard metrics, and pause/resume.

## Project Structure

```text
sts2-rl-agent/
|-- pyproject.toml                 # Package config, dependencies
|-- scripts/
|   |-- benchmark.py               # Throughput benchmark
|   |-- train_combat.py            # Combat-only training (curriculum, attention/GNN)
|   |-- train_navigator.py         # Navigator PPO (preferred full-run trainer)
|   |-- train_full_run.py          # Legacy hierarchical meta-policy
|   |-- train_card_value.py        # Learned card picker (alternative to combat critic)
|   +-- eval_full_run.py           # Post-training full-run evaluation sweeps
|
|-- sts2_env/                      # Python package (headless simulator)
|   |-- core/                      # Combat engine
|   |-- cards/                     # Card definitions (578 cards)
|   |-- powers/                    # Status effects (260 powers)
|   |-- monsters/                  # Monster AI (121 monsters)
|   |-- relics/                    # Relic effects (290 relics)
|   |-- potions/                   # Potion effects (63 potions)
|   |-- orbs/                      # Orb mechanics (Defect)
|   |-- characters/                # Character starting states
|   |-- encounters/                # Encounter definitions (88 encounters)
|   |-- events/                    # Event decision trees (68 events)
|   |-- map/                       # Map generation algorithm
|   |-- run/                       # Full-run state management
|   |
|   |-- training/                  # RL policy extractors and curriculum
|   |   |-- attention_extractor.py # Transformer feature extractor
|   |   |-- gnn_extractor.py       # DenseGAT feature extractor
|   |   |-- entity_tokens.py       # Shared 57-node tokenization
|   |   |-- combat_curriculum.py   # Staged encounter/deck curriculum
|   |   |-- deck_templates.py      # Starter/stripped/exhaust deck templates
|   |   +-- curriculum_callback.py # Gate eval and auto-promotion
|   |
|   |-- search/                    # Inference-only MCTS
|   |   |-- mcts_combat.py         # PUCT turn-bounded search
|   |   |-- mcts_agent.py          # CLI integration
|   |   +-- policy_guide.py        # PPO prior + critic for MCTS
|   |
|   |-- gym_env/                   # Gymnasium environments
|   |   |-- combat_env.py          # Single-combat env (Discrete(115), obs 1985)
|   |   |-- run_env.py             # Full-run env (Discrete(157), obs 2004)
|   |   |-- navigator_env.py       # Strategic env (obs 166, combat delegated)
|   |   |-- hierarchical_run_env.py  # Legacy hierarchical wrapper
|   |   |-- observation.py         # CombatState -> 1985-dim obs v11
|   |   |-- pile_distribution.py   # Draw-pile memory encoding
|   |   |-- reward_shaping.py      # HP + micro-reward math
|   |   |-- combat_value.py        # Combat-critic draft scoring
|   |   |-- action_space.py        # Action encoding + masking
|   |   +-- reward.py              # Combat step rewards
|   |
|   +-- bridge/                    # Real-game connection
|       |-- client.py              # TCP client
|       |-- state_adapter.py       # Game JSON -> 1985-dim observation
|       +-- agent_runner.py        # Main agent loop (optional MCTS)
|
|-- bridge_mod/                    # C# Bridge Mod (Godot project)
|-- tests/                         # 166 test files, 5,387+ test functions
|-- docs/                          # Documentation (see table below)
|   |-- PILE_WATCHLIST.json        # Curated draw-pile watchlist (sync-generated)
|-- RESEARCH.md                    # Research notes and prior work
+-- DECOMPILED_ARCHITECTURE.md     # Decompiled C# architecture analysis
```

## Game Content Coverage

| Content Type            | Game Total | Implemented | Coverage        |
| ----------------------- | ---------- | ----------- | --------------- |
| Cards                   | 578        | 578         | 100%            |
| Powers (status effects) | 260        | 260         | 100%            |
| Monsters                | 121        | 121         | 100%            |
| Relics                  | 290        | 290         | 100%            |
| Potions                 | 63         | 63          | 100%            |
| Encounters              | 88         | 88          | 100%            |
| Events                  | 68         | 68          | 100%            |
| Characters              | 5 + 2      | 5           | 100% (playable) |
| Acts                    | 4          | 4           | 100%            |

## How It Works

### Two-Phase Approach

Following lessons from the STS1 RL community, this project uses a two-phase strategy:

1. **Headless Simulator** (for training): A pure-Python reimplementation of STS2 combat and run mechanics, verified against the decompiled C# source. Runs at ~1,200 combats/second — fast enough for millions of training episodes.
2. **Bridge Mod** (for validation): A C# mod that hooks into the real game via Harmony, exposes state over TCP, and injects agent decisions. Includes 5–10x speed patches for faster real-game evaluation.

### RL Algorithm and Policy

- **MaskablePPO** from sb3-contrib (Stable Baselines 3)
- **Default policy:** 2-layer masked Transformer (`CombatAttentionExtractor`) over 57 entity tokens; alternatives are DenseGAT (`--policy gnn`) or flat MLP (`--policy mlp`)
- **Invalid action masking:** Each step, the environment provides a boolean mask indicating which actions are legal. Illegal actions are zeroed out before policy sampling.

### Observation and Action Spaces

| Environment | Observation | Action space |
| ----------- | ----------- | ------------ |
| Combat (`STS2CombatEnv`) | 1985-dim obs v11 | Discrete(115) |
| Full run (`STS2RunEnv`) | 2004-dim (1985 combat + 20 run state) | Discrete(157) |
| Navigator (`STS2NavigatorEnv`) | 166-dim strategic (obs v2) | Discrete(157) meta layout |

Combat action layout: end turn (0), card plays (1–60), potion uses (61–114). Run env adds map, card reward, shop, rest, event, treasure, and boss relic slices on top of the combat block.

### Reward Design

**Combat environment (sparse terminal):**

| Outcome | Reward |
| ------- | ------ |
| Win | +1.0 |
| Loss | -1.0 |

**Optional shaping** (training only; eval uses sparse):

| Signal | Scale (default) |
| ------ | --------------- |
| HP lost | Non-linear, up to -0.20/event (`--hp-steepness 3.0`) |
| Vulnerable on enemy | +0.02 × sublinear marginal stacks (`Δ**0.6`) |
| Weak on enemy | +0.02 × sublinear marginal stacks (`Δ**0.6`) |
| Block vs enemy attack | +0.001/HP blocked |

**Full-run / Navigator shaping:**

| Signal | Scale |
| ------ | ----- |
| Floor advanced | +0.01/floor |
| Combat cleared | +0.005 |
| Kill (run env) | +0.003/enemy |
| HP efficiency (win only) | up to +0.15 (`hp_ratio × win_hp_bonus_scale`) |
| HP lost after combat | Same non-linear penalty |
| Draft pick ΔV (Navigator) | `--draft-value-scale 0.1` × combat-critic ΔV |
| Run win / death | +1 / -1 |

### Inference Enhancement

Optional **turn-bounded MCTS** wraps the trained combat PPO at inference time. Search simulates one enemy phase and one lookahead player turn by default so block/attack trade-offs and setup cards can be planned ahead. PPO provides action priors and leaf values; PUCT search selects the action with the highest visit count. Enabled via `--mcts` on `train_combat.py` (post-train eval), `eval_full_run.py`, or `agent_runner.py`.

## Documentation

| Document | Description |
| -------- | ----------- |
| [README.md](README.md) | This file — project overview and quick start |
| [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) | **Authoritative training guide** — curriculum, policies, shaping, Navigator, MCTS |
| [docs/AGENT_USAGE_GUIDE.md](docs/AGENT_USAGE_GUIDE.md) | **Running agents** against the real game and bridge options |
| [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md) | **Known limitations** and fixed issues |
| [docs/SIMULATOR_ARCHITECTURE.md](docs/SIMULATOR_ARCHITECTURE.md) | **Simulator internals** — obs layout, entity tokens, env design |
| [RESEARCH.md](RESEARCH.md) | Research notes, prior work, algorithm selection |
| [DECOMPILED_ARCHITECTURE.md](DECOMPILED_ARCHITECTURE.md) | Decompiled C# analysis for simulator |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contribution guide, dev setup, adding content |
| [docs/PATCH_SYNC.md](docs/PATCH_SYNC.md) | Sync simulator from Steam install after patches |
| [docs/DECOMPILATION_GUIDE.md](docs/DECOMPILATION_GUIDE.md) | Decompile STS2 DLL and PCK resources |
| [docs/PROTOCOL.md](docs/PROTOCOL.md) | TCP bridge communication protocol |
| [docs/MOD_BUILD_GUIDE.md](docs/MOD_BUILD_GUIDE.md) | How to build and install the bridge mod |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Common problems and solutions |
| [docs/GAME_BRIDGE_REFERENCE.md](docs/GAME_BRIDGE_REFERENCE.md) | Bridge architecture and design notes |
| [docs/AUTOSLAY_BRIDGE.md](docs/AUTOSLAY_BRIDGE.md) | AutoSlay-based bridge design |
| [docs/GAME_SYSTEMS_REFERENCE.md](docs/GAME_SYSTEMS_REFERENCE.md) | Game mechanics reference |
| [docs/PARITY_COVERAGE_BACKLOG.md](docs/PARITY_COVERAGE_BACKLOG.md) | Direct-reference parity gate (complete) |
| [docs/PARITY_GAPS.md](docs/PARITY_GAPS.md) | Confirmed blockers to exact parity |
| [docs/BRIDGE_LIVE_SMOKE.md](docs/BRIDGE_LIVE_SMOKE.md) | Bridge offline + live smoke procedure |
| [docs/CARDS_REFERENCE.md](docs/CARDS_REFERENCE.md) | All 578 cards |
| [docs/PARITY_BACKLOG.md](docs/PARITY_BACKLOG.md) | Auto-generated fingerprint audit summary |
| [docs/POWERS_REFERENCE.md](docs/POWERS_REFERENCE.md) | All 260 powers |
| [docs/MONSTERS_REFERENCE.md](docs/MONSTERS_REFERENCE.md) | All 121 monsters |
| [docs/RELICS_REFERENCE.md](docs/RELICS_REFERENCE.md) | All 290 relics |

## License

This project is for research and educational purposes. Slay the Spire 2 is the property of Mega Crit Games.

## Acknowledgments

- [decapitate-the-spire](https://github.com/jahabrewer/decapitate-the-spire) — STS1 headless simulator, architectural inspiration
- [spire-codex](https://github.com/ptrlrd/spire-codex) — STS2 data extraction pipeline
- [CommunicationMod](https://github.com/ForgottenArbiter/CommunicationMod) — STS1 game bridge protocol design
- [BaseLib-StS2](https://github.com/Alchyr/BaseLib-StS2) — STS2 mod framework
- [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3) — RL training framework
