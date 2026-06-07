# Known Issues and Limitations

Current known issues, bugs, and limitations of the STS2 RL Agent project.

For the current RL architecture (obs v11 / 1985 dims, attention policies, curriculum, Navigator,
MCTS, reward shaping), see the [README RL Feature Overview](../README.md#rl-feature-overview),
[TRAINING_GUIDE.md](TRAINING_GUIDE.md), and [SIMULATOR_ARCHITECTURE.md](SIMULATOR_ARCHITECTURE.md).

---

## Fixed Issues

### 1. Energy always displayed as 3 with CardCmd.AutoPlay

**Status:** Fixed

**Problem:** The C# bridge mod initially used `CardCmd.AutoPlay()` to execute card plays. This method bypasses the normal energy deduction, so the player's energy always stayed at 3 (max) regardless of cards played. The agent could play unlimited cards per turn.

**Fix:** Switched to `PlayCardAction` which properly spends energy:

```csharp
var playAction = new PlayCardAction(card, target);
RunManager.Instance.ActionQueueSynchronizer.RequestEnqueue(playAction);
```

**Location:** `bridge_mod/RlCombatHandler.cs` line 187-188

### 2. EchoForm / modify_card_play_count was missing

**Status:** Fixed

**Problem:** The hook for modifying how many times a card is played was not implemented. Powers like EchoForm (play each card twice) had no effect.

**Fix:** Added `modify_card_play_count` to `core/hooks.py` and wired it into `CombatState.play_card()`.

**Location:** `sts2_env/core/hooks.py` lines 189-200, `sts2_env/core/combat.py` line 255

### 3. Enemy round-1 block not cleared

**Status:** Fixed

**Problem:** Enemies that gained block before their first turn (from combat-start effects) were not having their block cleared at the start of the enemy turn on round 1.

**Fix:** The enemy turn now always clears block for each alive enemy, regardless of round number.

**Location:** `sts2_env/core/combat.py` `_execute_enemy_turn()`

### 4. State adapter and action mask protocol mismatches

**Status:** Fixed

**Problem:** The Python `StateAdapter` was expecting different field names and formats than what the C# mod was actually sending. For example, target type strings like `"AnyEnemy"` vs `"ANY_ENEMY"`, and power list format differences.

**Fix:** Updated `state_adapter.py` to handle both formats:

```python
_UNTARGETED_TYPES = {TargetTypeName.SELF, TargetTypeName.NONE, TargetTypeName.ALL_ENEMIES,
                     "SELF", "NONE", "ALL_ENEMIES", "Self", "None", "AllEnemies"}
```

**Location:** `sts2_env/bridge/state_adapter.py` lines 69-71

### 5. Fogmog + Eye With Teeth combat soft-lock

**Status:** Fixed

**Problem:** After killing Fogmog, combat would not end if Eye With Teeth was dead but mid-revive (`IllusionPower.is_reviving`). The next enemy turn would run `REVIVE_MOVE` and bring Eye back, making the fight unwinnable.

**Root cause:** `_check_combat_end()` required all enemies dead (including minions) and `IllusionPower.should_stop_combat_ending()` blocked victory while reviving. The game only requires all primary (non-minion) enemies to be dead; illusion minions do not block combat end in the decompiled `IllusionPower`.

**Fix:** Added `Creature.is_primary_enemy` / `is_secondary_enemy` (minion = secondary), victory checks `alive_primary_enemies`, and removed illusion combat-end blocking.

**Location:** `sts2_env/core/creature.py`, `sts2_env/core/combat.py`, `sts2_env/powers/monster.py`

---

## Open Issues

### 6. AnimationSpeedPatch fails to apply

**Severity:** Low (affects real-game speed only)

**Problem:** The Harmony patch targeting `MegaAnimationState.SetTimeScale` fails on some game versions because the method signature changed between updates. The patch is skipped with a log message.

**Impact:** The game runs at normal animation speed instead of 5x. The `WaitSpeedPatch` (which reduces timed delays by 10x) still applies successfully, providing some speedup.

**Workaround:** None currently. The animation patch needs to be updated when the game's `MegaAnimationState` API changes.

**Location:** `bridge_mod/MainFile.cs` `AnimationSpeedPatch` class

### 7. Mod abandon-run popup path may not match all versions

**Severity:** Low

**Problem:** The Godot scene tree paths used to find the abandon-run confirmation popup (`VerticalPopup/YesButton`) may not match all game versions. If the path is wrong, the mod cannot automatically abandon an existing run before starting a new one.

**Impact:** If there is already a run in progress when the mod starts, it may fail to abandon it cleanly.

**Workaround:** Manually abandon the run from the main menu before starting the agent.

**Location:** `bridge_mod/RlAutoSlayer.cs` `PlayMainMenuAsync()` lines 455-472

### 8. Full-run training needs significantly more steps and better reward shaping

**Severity:** High (fundamental training challenge)

**Status:** Mitigated (infrastructure added; win rate still expected to need long training)

**Problem:** The full-run environment produces 0% win rate even after 1M training steps with a flat policy. The agent learns to progress further through Act 1 (avg 8.9 floors vs 3.9 for random) but cannot complete a run.

**Root causes:**

- Sparse reward: only +1 at run victory, -1 at death without shaping.
- Long episodes: a full run spans thousands of micro-steps.
- Multi-phase action space: `Discrete(157)` across combat, map, rewards, shop, rest, event, treasure, and player-selection slices.
- Compounding decisions: bad deck choices early doom later combats.

**Mitigation (implemented):**

- **Reward shaping** in [`sts2_env/gym_env/run_reward.py`](../sts2_env/gym_env/run_reward.py) and [`reward_shaping.py`](../sts2_env/gym_env/reward_shaping.py): floor progress (+0.01/floor), combat clear (+0.005), run-end HP efficiency bonus on win (up to +0.15), **non-linear HP penalties** (damage near 0 HP costs much more than at full HP), and combat **micro-rewards** for Vulnerable/Weak on enemies and block vs attacks (`--reward-shaping`, default on). Progress scales were rebalanced so cumulative dense shaping on a long death stays negative — earlier defaults (+0.05/floor, +0.10/combat clear, +0.05/kill) could make a floor-50 loss outscore a short win.
- **Act-count curriculum** via `RunManager.max_acts` and `--act-count` (train Act 1 before full game).
- **Hierarchical training** via [`STS2HierarchicalRunEnv`](../sts2_env/gym_env/hierarchical_run_env.py): frozen combat PPO handles fights; legacy meta PPO trains on map/rewards/shop (`--combat-model`, [`scripts/train_full_run.py`](../scripts/train_full_run.py)).
- **Navigator agent** via [`STS2NavigatorEnv`](../sts2_env/gym_env/navigator_env.py): dedicated strategic observations; all non-combat phases (`--combat-model`, [`scripts/train_navigator.py`](../scripts/train_navigator.py)).
- **Combat-critic draft scoring** via [`sts2_env/gym_env/combat_value.py`](../sts2_env/gym_env/combat_value.py): `--combat-value-draft` (Phase 1) or `--combat-value-shaping` (Navigator training).
- **Curriculum fine-tune:** `--load-model` to fine-tune from a Phase 1 checkpoint into a new output dir.
- **Pause/resume:** Ctrl+C saves an `interrupted_checkpoint.zip`; `--resume --output-dir <dir>` continues from the latest checkpoint (periodic checkpoints every `--checkpoint-freq` steps), so multi-day training no longer has to run in one sitting. See [`docs/TRAINING_GUIDE.md`](TRAINING_GUIDE.md) ("Pause and Resume").

**Also implemented (extensions):**

- **Training presets** (`--preset phase1|phase2|full`) with 2M / 5M / 8M meta timesteps ([`scripts/train_full_run.py`](../scripts/train_full_run.py)).
- **Non-combat heuristics** ([`sts2_env/gym_env/noncombat_heuristics.py`](../sts2_env/gym_env/noncombat_heuristics.py)): auto card-reward, boss-relic, and rest picks during training (`--no-noncombat-heuristic` to disable).
- **Act-mixed combat training** ([`sts2_env/encounters/pools.py`](../sts2_env/encounters/pools.py), `train_combat.py --acts 0,1,2`) and per-act routing (`--combat-models`).
- **Combat tier curriculum** ([`sts2_env/training/combat_curriculum.py`](../sts2_env/training/combat_curriculum.py), `train_combat.py --curriculum`): staged easy-pool bootstrapping (Jaw Worm/Cultists), HP-retention gate metrics, deck-complexity ramps (Exhaust, Necrobinder), hard-start recovery scenarios (`--curriculum recovery`), and optional `--auto-promote`.

**Card-value network and win tracking (implemented):**

- Learned card picker in [`sts2_env/gym_env/card_value.py`](../sts2_env/gym_env/card_value.py), trained via [`scripts/train_card_value.py`](../scripts/train_card_value.py) with outcome-weighted heuristic labels.
- **Combat-critic draft picker** in [`sts2_env/gym_env/combat_value.py`](../sts2_env/gym_env/combat_value.py): uses the frozen combat PPO value head to score drafts against elite encounters. Opening-hand RNG adds variance; mitigated by fixed seed ensembles (`CombatValueConfig.rng_seed`, `num_encounters`).
- [`RunWinRateCallback`](../sts2_env/training/callbacks.py) logs `eval/win_rate` during long meta training; [`scripts/eval_full_run.py`](../scripts/eval_full_run.py) for post-training sweeps.

Remaining work: achieving a positive full-run win rate still depends on running full preset schedules (2M–8M meta steps) with mixed-act combat and a trained card-value model; no guarantee without sufficient training time.

### 8. Only Ironclad combat model trained

**Status:** Mitigated (training pipeline and bridge mod support all characters)

**Problem:** The combat training pipeline only created Ironclad starter decks. All training and evaluation used the Ironclad character.

**Mitigation (implemented):**

- **Character selection** in [`STS2CombatEnv`](../sts2_env/gym_env/combat_env.py): `--character` (single) and `--characters all` (mixed) in [`scripts/train_combat.py`](../scripts/train_combat.py).
- **Starter deck, HP, and starting relic** per character via [`sts2_env/characters/all.py`](../sts2_env/characters/all.py).
- **Observation v11 (1985 dims):** obs v10 plus normalized `turn_count/20` in the player core block (1 dim). Pre-existing obs v10 checkpoints must be retrained.
- **Observation v8 (1978 dims):** obs v7 plus relic `counter_norm` (5 features per relic slot). See `TOKEN_SLICES` in [`sts2_env/gym_env/observation.py`](../sts2_env/gym_env/observation.py).
- **Observation v7 (1948 dims):** obs v6 plus 9 hand-card features per slot (exhaust, retain, is_power, hit_count).
- **Observation v6 (1908 dims):** all 268 `PowerId` values on player (amount/20) and each enemy slot (amount/10), plus pile memory, character mechanics, relic entity slots, and potion entity slots.
- **Full-run wiring:** `--character`, `--characters`, and `--combat-models-by-character` in [`scripts/train_full_run.py`](../scripts/train_full_run.py) and [`scripts/eval_full_run.py`](../scripts/eval_full_run.py).
- **Bridge mod:** character selection via `STS2_BRIDGE_CHARACTER` env var ([`bridge_mod/BridgeConfig.cs`](../bridge_mod/BridgeConfig.cs)); combat JSON includes `character_id`, `stars`, `orb_queue`, `osty`, `relics` (with `counter`), `potions`, pile card lists, and enriched hand-card fields (`keywords`, `base_damage`, `base_block`, `retain`, `hit_count`) ([`bridge_mod/RlCombatHandler.cs`](../bridge_mod/RlCombatHandler.cs)); Python adapter encodes the full **1985-dim** vector ([`sts2_env/bridge/state_adapter.py`](../sts2_env/bridge/state_adapter.py)).

**Remaining gaps:**

- Pre-existing obs v10 and earlier combat checkpoints must be retrained for **obs v11 (1985 dims)**. `mlp`, `attention`, and `gnn` checkpoints are also mutually incompatible (different `policy_kwargs` / feature extractors).
- Live bridge eval requires matching the mod's `STS2_BRIDGE_CHARACTER` to the agent's trained character and `--character` / `--combat-models-by-character` model path.

### 9. Combat potion actions were missing from the RL action space

**Status:** Fixed

**Problem:** The combat action space originally only covered card plays and end turn, so the agent could not use potions strategically during combat.

**Fix:** The combat action space now includes fixed-width potion actions, `CombatState` can execute potion uses directly, and the bridge path serializes and decodes potion actions as explicit `POTION` commands.

**Location:** `sts2_env/core/constants.py`, `sts2_env/core/combat.py`, `sts2_env/gym_env/action_space.py`, `sts2_env/gym_env/combat_env.py`, `sts2_env/bridge/state_adapter.py`, `bridge_mod/RlCombatHandler.cs`

### 9b. Potion slots were missing from the combat observation vector

**Status:** Fixed

**Problem:** The action space exposed 9 potion slots (actions 61–114), but the 294-dim observation encoded zero potion identity. The agent could infer slot usability from the action mask alone, not potion type, rarity, or combat-usability.

**Fix:** Obs v5 adds a 27-dim potion block (9 slots × 3 features: `potion_id_norm`, `rarity_norm`, `can_use_in_combat_flag`) after relics. `TOKEN_SLICES`, entity tokenization, GNN adjacency, and the bridge state adapter were updated accordingly.

**Location:** `sts2_env/gym_env/observation.py`, `sts2_env/training/entity_tokens.py`, `sts2_env/training/entity_graph.py`, `sts2_env/bridge/state_adapter.py`

### 9c. Only 6 player powers and 3 enemy powers were tracked in combat observations

**Status:** Fixed

**Problem:** The combat observation encoded only six player powers (Strength, Dexterity, Vulnerable, Weak, Frail, Artifact) and three per-enemy powers (Vulnerable, Weak, Strength). Game-critical buffs like Barricade, Corruption, Echo Form, Ritual, Poison, and Doom were invisible to the policy.

**Fix:** Obs v6 expands power encoding to all 268 `PowerId` values (excluding legacy aliases and the `GENERIC` placeholder), derived programmatically from the enum. Player powers use amount/20; enemy powers use amount/10 per slot. `ENEMY_POWERS`, `ENEMY_CORE_FEATURES`, and shared bridge tracking lists keep simulator and live-game encoders aligned.

**Location:** `sts2_env/gym_env/observation.py`, `sts2_env/bridge/state_adapter.py`, `tests/test_observation_powers.py`

### 9d. No turn counter in the combat observation vector

**Status:** Fixed

**Problem:** Relics and powers that scale with combat duration (e.g. Demon Form Strength, Malleable block requirement) were ambiguous without a turn-count signal — the agent could not distinguish turn 1 from turn 10 at the same power amounts.

**Fix:** Obs v11 adds `min(turn_count, 20) / 20.0` as the sixth player-core feature. The bridge adapter derives the same value from the serialized `round` field.

**Location:** `sts2_env/gym_env/observation.py`, `sts2_env/bridge/state_adapter.py`, `tests/test_observation_turn_count.py`

### 10. Some card effects may not match the real game exactly

**Severity:** Medium (simulator fidelity) — **mitigated (audited surface clean; live bridge pass pending)**

**Problem:** Exact parity is no longer blocked on missing core helpers or audited fingerprint deltas. The only remaining open item is recording a live-game bridge smoke pass to field-verify the simulator against the running client.

**Mitigation (implemented):**

- Behavioral audit scripts: [`scripts/audit_onplay_behavior_coverage.py`](../scripts/audit_onplay_behavior_coverage.py), [`scripts/audit_relic_hook_coverage.py`](../scripts/audit_relic_hook_coverage.py), optional [`scripts/audit_wiki_card_metadata.py`](../scripts/audit_wiki_card_metadata.py) (wiki is informational only; decompiled source wins).
- Generated backlog: [`docs/PARITY_BACKLOG.md`](PARITY_BACKLOG.md) now reports **0** card fingerprint mismatches (543/543) and **0** relic hook mismatches (287/287). Both audits accept `--fail-on-mismatch` to keep them at zero on sync.
- Every non-deprecated OnPlay card and every hook-bearing relic has at least one `Matches {Class}.cs` regression (hand-written suites plus [`tests/test_generated_onplay_smoke_parity.py`](../tests/test_generated_onplay_smoke_parity.py) and [`tests/test_generated_relic_smoke_parity.py`](../tests/test_generated_relic_smoke_parity.py)).
- Recent decompiled-backed fixes: `Stoke` now exhausts the hand and generates/upgrades that many new character cards; `Snap` now lets the player pick a hand card to gain Retain; `Permafrost` now resets its per-combat flag on combat-room entry; plus earlier `Compact`, `KnifeTrap`, `Glow` / `GuidingStar`, and `SpoilsOfBattle` fixes.
- The relic audit now resolves Python hooks through the class MRO and a broadened `PY_HOOK_ALIASES`, so equivalent-behavior naming differences no longer show as gaps.
- Live-bridge smoke pipeline: build gate [`scripts/bridge_live_smoke.ps1`](../scripts/bridge_live_smoke.ps1), procedure [`docs/BRIDGE_LIVE_SMOKE.md`](BRIDGE_LIVE_SMOKE.md), committed golden fixture, and offline + opt-in live tests in [`tests/test_bridge_live_smoke.py`](../tests/test_bridge_live_smoke.py). The C# bridge mod now compiles cleanly.

**Remaining gaps:**

- Live-game bridge smoke record-and-compare pass has not been run yet (offline gate passed 2026-06-04; no game was listening for `--live`). Run `python scripts/record_bridge_smoke.py --live` on a machine with STS2 + the mod installed (see [PARITY_GAPS.md](PARITY_GAPS.md) §3).
- Direct-reference parity gate is closed ([PARITY_COVERAGE_BACKLOG.md](PARITY_COVERAGE_BACKLOG.md)); use `scripts/audit_behavioral_edge_coverage.py --smoke-only` after adding high-impact cards.

**Impact:** Residual drift can still affect RL transfer; use bridge evaluation as ground truth for training claims.

### 11. Reconnection timing issues

**Severity:** Low

**Problem:** If the Python agent connects before the game has finished loading and the AutoSlayer has started, there can be a race condition where the first state message arrives before the agent is ready.

**Workaround:** Start the game first, wait for the main menu to appear, then start the Python agent. The agent runner has reconnection retry logic (`_reconnect_with_retry` with 10 attempts, 3s delay).

**Location:** `sts2_env/bridge/agent_runner.py` lines 288-309

### 12. `inspect.signature` on hot path

**Status:** Fixed

**Severity:** Low (performance)

**Problem:** `fire_after_card_drawn` used to call `inspect.signature(method).parameters` for every card draw to determine the parameter count of each power's `on_card_drawn` method. This was slow.

**Fix:** All power `on_card_drawn` implementations now use `(owner, card, from_hand_draw, combat)`, and the dispatcher calls that signature directly.

**Location:** `sts2_env/core/hooks.py`

### 13. `run_env` exception handling used to hide simulation bugs

**Status:** Fixed

**Problem:** `STS2RunEnv.step()` used to convert internal simulation exceptions into silent losses, which made debugging difficult.

```python
try:
    if phase == RunManager.PHASE_COMBAT:
        self._step_combat(action)
    # ...
except Exception:
    if not self._mgr.is_over:
        self._mgr.run_state.lose_run()
```

**Fix:** `STS2RunEnv.step()` now logs the exception before forcing the run to end, so failures are visible in logs instead of disappearing into episode outcomes.

**Location:** `sts2_env/gym_env/run_env.py`

### 14. Pile-summary distribution shift between simulator and bridge

**Status:** Fixed (obs v4)

**Problem:** Obs v3 kept three pile-composition slots zeroed because the bridge only sent pile counts, not card lists. The agent could not count cards in the draw pile.

**Fix:** Obs v4 adds 26 pile-memory features (unseen deck composition, next-draw probabilities, known top-of-deck order, high-value heuristics, watchlist groups). The bridge now serializes `draw_pile`, `discard_pile`, and `play_pile` card arrays; simulator and adapter share `encode_pile_memory()` from `pile_distribution.py`.

**Location:** `sts2_env/gym_env/pile_distribution.py`, `sts2_env/gym_env/observation.py`, `sts2_env/bridge/state_adapter.py`, `bridge_mod/RlCombatHandler.cs`

### 15. Multi-step enemy intents only encoded the first component

**Status:** Fixed

**Problem:** Enemy observation encoding read only `move.intents[0]`. Moves that combine attack with buff/debuff/defend (e.g. attack + gain Strength) exposed only the first intent type and could under-report damage when attack was not first. Bridge serialization had the same first-intent-only limitation.

**Fix:** All intents on the current move are folded into the enemy obs slot: tracked intent types are OR'd into the 5-bit one-hot, and ATTACK/MULTI_ATTACK damage/hits are summed. Normalization changed to total damage `/60` and hits `min(hits, 10) / 10`. Bridge payloads now include `intent_types` alongside aggregated `intent_damage` / `intent_hits`.

**Retrain required:** Existing checkpoints trained on per-hit damage `/30` and hits `/5` need retraining.

**Location:** `sts2_env/gym_env/observation.py`, `sts2_env/bridge/state_adapter.py`, `sts2_env/parity/bridge_replay.py`, `bridge_mod/RlCombatHandler.cs`, `tests/test_observation_enemy_intents.py`

### 16. Hardcoded pile watchlist groups

**Status:** Fixed

**Problem:** `WATCHLIST_GROUPS` in `pile_distribution.py` was a static dict of `CardId` values. New cards added by game patches never appeared in the four binary watchlist features (finisher/setup/aoe/power presence), degrading pile memory for new content.

**Fix:** Watchlists live in [`docs/PILE_WATCHLIST.json`](PILE_WATCHLIST.json), loaded at runtime via `load_watchlist_groups()`. The sync `docs` step regenerates the file, validates card names, auto-appends **newly introduced** `CardId` values to `finisher`/`aoe` when they match conservative metadata rules, and reports unlisted cards for manual `power`/`setup` curation. Cache clears on sync alongside card factory metadata.

**Retrain required:** After a patch sync auto-adds watchlist entries, retrain combat policies if watchlist group features change (four binary dims in the pile-memory block; see `TOKEN_SLICES["piles"]` in `observation.py`).

**Location:** `sts2_env/gym_env/pile_distribution.py`, `scripts/sync/generate_pile_watchlist.py`, `docs/PILE_WATCHLIST.json`
