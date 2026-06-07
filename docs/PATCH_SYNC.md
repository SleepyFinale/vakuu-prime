# Patch Sync Runbook

After Slay the Spire 2 updates on Steam, refresh decompiled sources and simulator metadata with the sync pipeline.

## Prerequisites

- [ilspycmd](https://github.com/icsharpcode/ILSpy): `dotnet tool install -g ilspycmd`
- STS2 installed (default: `C:\Program Files (x86)\Steam\steamapps\common\Slay the Spire 2`)
- Optional: [GDRE Tools](https://github.com/GDRETools/gdsdecomp) on PATH for `extract-pck`

Override install path with `--game-path` or environment variable `STS2_GAME_PATH`.

## Quick sync

```powershell
# Full pipeline (decompile, docs, scaffold, static apply, audits)
python scripts/sync_from_game.py all --game-path "C:\Program Files (x86)\Steam\steamapps\common\Slay the Spire 2" --apply

# Apply scaffolds and static patches (required when the game adds new cards)
python scripts/sync_from_game.py scaffold --apply
python scripts/sync_from_game.py apply-static --apply
```

If decompile reports cards without `CardId` (e.g. `NotYet`), run `scaffold --apply` before `apply-static` completes. The `all` command runs scaffold before docs; use `--apply` to write enum entries and stubs.

```bash
pytest tests/ -q
```

Restart any long-running Python process after sync so `create_card()` reloads regenerated `docs/CARDS_REFERENCE.md` and pile watchlists reload from `docs/PILE_WATCHLIST.json` (`@lru_cache` in `factory.py` and `pile_distribution.py`).

## Commands

| Command | Description |
| ------- | ----------- |
| `decompile` | Run `ilspycmd` → `decompiled/`, update `sync_manifest.json` |
| `extract-pck` | Run GDRE on `sts2.pck` (optional localization) |
| `docs` | Regenerate `docs/CARDS_REFERENCE.md`, `POWERS_REFERENCE.md`, `RELICS_REFERENCE.md`, `MONSTERS_REFERENCE.md`, `PILE_WATCHLIST.json` |
| `report` | Write `sync_report.md` (diff vs `decompiled_prev/`, parity summary) |
| `scaffold` | Add missing `CardId` entries and card stubs (`# SYNC_STUB`) |
| `apply-static` | Update `cost`, `base_damage`, `base_block`, `effect_vars` in `make_*` factories |
| `all` | Runs decompile → scaffold → docs → apply-static → audits |

Use `--apply` to write scaffold/static changes. Without it, those steps only print planned edits.

## What is automated vs manual

**Automated:** decompilation, reference markdown, enum/stub scaffolding, optional `apply-static` patches to `make_*` factories.

**Runtime card stats:** After decompile, `create_card()` overlays cost, rarity, keywords, vars, and related fields from `decompiled/` automatically (see `factory._apply_decompiled_static_metadata`). The post-sync audits should pass without hand-editing hundreds of factories.

**Manual:** card `OnPlay` effects, power/relic hooks, monster AI state machines. Parity audit may still list missing power/relic implementations — that is expected until ported. Use `sync_report.md` and `git diff decompiled/` to prioritize.

After `docs`, review `git diff docs/PILE_WATCHLIST.json` for auto-added finisher/aoe cards and any **New cards not in any group** lines in `sync_report.md`; add high-value engine cards to `power` or `setup` manually when needed. The watchlist drives draw-pile memory **watchlist group** features in combat observations (see [README draw-pile memory](../README.md#draw-pile-memory) and [SIMULATOR_ARCHITECTURE.md](SIMULATOR_ARCHITECTURE.md)).

Mark files that must not receive auto patches with `# SYNC_NO_AUTO` at the top of the module.

## Related docs

- [DECOMPILATION_GUIDE.md](DECOMPILATION_GUIDE.md) — tools and namespace map
- [CONTRIBUTING.md](../CONTRIBUTING.md) — implementing new content after sync
