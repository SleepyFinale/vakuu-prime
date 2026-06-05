# Parity Coverage Backlog

This document tracks the **direct-reference coverage gate**: every decompiled gameplay class counted by the parity inventory must have a Python implementation mention and a named test mention.

It complements [PARITY_GAPS.md](./PARITY_GAPS.md):

- `PARITY_GAPS.md` — why exact parity is not yet claimed (behavior depth, bridge field verification, full-run replay).
- this document — whether each surface is **named** in code and tests (coverage gate, not full behavior proof).

## Status: direct-reference gate closed

As of **2026-06-04**, the stricter audit (including deprecated save placeholders) reports **zero** missing implementation and **zero** missing direct-test references:

| Surface | Total | Missing implementation | Missing test references |
| --- | ---: | ---: | ---: |
| Cards | 578 | 0 | 0 |
| Encounters | 89 | 0 | 0 |
| Events | 68 | 0 | 0 |
| Modifiers | 17 | 0 | 0 |
| Monsters | 122 | 0 | 0 |
| Potions | 64 | 0 | 0 |
| Powers | 265 | 0 | 0 |
| Relics | 295 | 0 | 0 |

Reproduce:

```powershell
python scripts/parity_reference_audit.py --direct-test-references --include-deprecated --code-implementation-references --show-missing
```

Behavioral fingerprints and hook names are tracked separately in [PARITY_BACKLOG.md](./PARITY_BACKLOG.md) (543/543 OnPlay cards, 287/287 hook-bearing relics, zero mismatches at last audit).

Card `effect_vars` keys used by `@register_effect` handlers are checked by `scripts/audit_card_effect_vars.py`. Factory output merges handwritten `effect_vars` with decompiled dynamic vars in `create_card()`.

## Current remaining backlog (reference gate)

All module and event buckets are **0**. No card, relic, event, encounter, monster, potion, power, or modifier class counted by the audit lacks both implementation and direct test references.

## What remains for exact parity

The reference gate is done. Remaining work lives in [PARITY_GAPS.md](./PARITY_GAPS.md):

1. **Deep edge-case tests** — high-impact cards that only have generated `*_onplay_smoke` tests (no outcome assertions). Triage: `python scripts/audit_behavioral_edge_coverage.py --smoke-only`.
2. **Live bridge smoke** — `python scripts/record_bridge_smoke.py --live` with STS2 + the bridge mod (offline golden replay runs in CI).
3. **Full-run replay** — optional map → combat → reward slices per [BRIDGE_REPLAY_HARNESS.md](./BRIDGE_REPLAY_HARNESS.md).

## Verification commands

```powershell
python scripts/parity_reference_audit.py --direct-test-references --include-deprecated --code-implementation-references --show-missing
python scripts/audit_onplay_behavior_coverage.py --fail-on-mismatch --fail-on-missing-tests
python scripts/audit_relic_hook_coverage.py --fail-on-mismatch
python scripts/audit_card_effect_vars.py
python scripts/audit_behavioral_edge_coverage.py --smoke-only
python -m pytest tests/test_bridge_live_smoke.py tests/test_bridge_replay_harness.py -q
```

Add `tests/test_*parity*.py` locally when changing card, relic, or event behavior.

## Counting rules

### Cards

- Totals come from `sts2_env.cards.factory._factory_registry()` plus reference-backed ids.
- Direct coverage requires a test function whose docstring includes `Matches {Card}.cs` or explicit naming in a focused parity suite.

### Relics

- Direct coverage: `obtain_relic`, `create_relic_by_name`, or `relics=[...]` in a focused test.

### Events

- Direct coverage: event class instantiated or driven through `RunManager` in a focused event test.

## Recommended order

1. Keep the reference audit at zero after each `sync_from_game` run (enforced in `scripts/sync/report.py`).
2. Shrink the smoke-only high-impact list using `scripts/audit_behavioral_edge_coverage.py` and existing `*_reference_parity` / `*_edge*` suites.
3. Record live bridge smoke when a game instance is available.
4. Do not claim exact parity until [PARITY_GAPS.md](./PARITY_GAPS.md) checklist is satisfied.

## Interpretation guardrails

- Missing from this gate means missing **named proof**, not necessarily missing implementation.
- Passing the gate does not prove timing, RNG, multiplayer owner scope, or bridge serialization match the client.
- Historical baseline inventories (2026-03-17) are archived in [archive/PARITY_COVERAGE_BASELINE_2026-03-17.md](./archive/PARITY_COVERAGE_BASELINE_2026-03-17.md).

## Pass changelog (abbreviated)

Multi-agent passes added direct references and fixed logic for Ironclad/Silent/Defect/colorless/Regent/Necrobinder cards, status curses, events (act1–3 + shared), and relic hooks. Notable fixes include `Quadcast`, `Dualcast`, `Escape Plan`, `Pillage`, `DemonicShield` / `Fisticuffs` block hooks, `NORMALITY` enforcement, and extensive event/relic reward flows. Full per-pass notes are in the archive copy of this file and in git history.
