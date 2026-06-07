"""Tests for STS2 game sync pipeline generators and utilities."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_effect_summary_bash_fixture():
    from scripts.sync.effect_summary import summarize_on_play

    source = (REPO_ROOT / "tests/fixtures/decompiled_cards/Bash.cs").read_text()
    summary = summarize_on_play(source)
    assert "Deal Damage" in summary
    assert "Apply power" in summary


def test_reference_metadata_bash_fixture():
    from sts2_env.cards.reference_static_metadata import (
        reference_dynamic_vars_from_source,
        reference_metadata_from_source,
    )

    path = REPO_ROOT / "tests/fixtures/decompiled_cards/Bash.cs"
    meta = reference_metadata_from_source(path)
    assert meta.cost == 2
    vars_ = reference_dynamic_vars_from_source(path)
    assert vars_.get("damage") == 8
    assert vars_.get("vulnerable") == 2


def test_manifest_roundtrip():
    from scripts.sync.manifest import SyncManifest

    with tempfile.TemporaryDirectory() as tmp:
        dll = Path(tmp) / "sts2.dll"
        dll.write_bytes(b"test-dll-contents")
        manifest = SyncManifest.from_paths(dll, str(Path(tmp)))
        out = Path(tmp) / "sync_manifest.json"
        manifest.save(out)
        loaded = SyncManifest.load(out)
        assert loaded is not None
        assert loaded.sts2_dll_sha256 == manifest.sts2_dll_sha256
        assert loaded.surfaces["cards"] >= 0


def test_reference_metadata_skips_unmapped_decompiled_cards():
    from sts2_env.cards.reference_static_metadata import (
        clear_reference_metadata_caches,
        reference_metadata_by_card_id,
        unmapped_reference_card_classes,
    )

    clear_reference_metadata_caches()
    metadata = reference_metadata_by_card_id()
    unmapped = unmapped_reference_card_classes()
    for class_name in unmapped:
        assert all(meta.card_id.name != class_name.upper() for meta in metadata.values())
    assert isinstance(metadata, dict)


def test_apply_static_dry_run_on_ironclad_basic():
    from scripts.sync.apply_static import apply_card_static

    result = apply_card_static(apply=False)
    assert isinstance(result.patches, list)
    assert isinstance(result.skipped, list)


def test_generate_powers_writes_header(tmp_path, monkeypatch):
    from scripts.sync import generate_powers_reference as gen

    if not (REPO_ROOT / "decompiled/MegaCrit.Sts2.Core.Models.Powers").is_dir():
        pytest.skip("decompiled powers not present")
    out = tmp_path / "POWERS_REFERENCE.md"
    path = gen.generate_powers_reference(out)
    text = path.read_text(encoding="utf-8")
    assert "Powers Reference" in text
    assert "VulnerablePower" in text or "powers total" in text


def test_generate_cards_reference_pool_counts_and_markdown():
    from scripts.sync.generate_cards_reference import generate_cards_reference

    if not (REPO_ROOT / "decompiled/MegaCrit.Sts2.Core.Models.Cards").is_dir():
        pytest.skip("decompiled cards not present")
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "CARDS_REFERENCE.md"
        generate_cards_reference(out)
        text = out.read_text(encoding="utf-8")
    assert "<a id=" not in text
    assert "| Ironclad | 87 |" in text or "| Ironclad | " in text
    assert "**Target:** AnyEnemy" in text or "**Target:** Self" in text
    assert "`{Damage:" in text or "`{}`" in text
    assert "ANYENEMY" not in text


def test_game_paths_default():
    from scripts.sync.game_paths import resolve_game_paths

    paths = resolve_game_paths()
    assert paths.game_root.name == "Slay the Spire 2"
    assert paths.sts2_dll.name == "sts2.dll"


def test_generate_pile_watchlist_auto_append(tmp_path) -> None:
    from sts2_env.core.enums import CardId
    from scripts.sync.generate_pile_watchlist import generate_pile_watchlist

    known_without_thunderclap = sorted(
        name for name in CardId.__members__ if name != "THUNDERCLAP"
    )
    payload = {
        "version": 1,
        "known_card_ids": known_without_thunderclap,
        "groups": {
            "power": {"cards": [], "auto": None, "exclude": []},
            "finisher": {"cards": [], "auto": None, "exclude": []},
            "setup": {"cards": [], "auto": None, "exclude": []},
            "aoe": {
                "cards": [],
                "auto": {
                    "card_type": "ATTACK",
                    "target_type": "ALL_ENEMIES",
                    "min_base_damage": 4,
                    "rarity_min": "COMMON",
                },
                "exclude": [],
            },
        },
    }
    watchlist_path = tmp_path / "PILE_WATCHLIST.json"
    watchlist_path.write_text(json.dumps(payload), encoding="utf-8")

    _, summary = generate_pile_watchlist(output=watchlist_path)

    assert "THUNDERCLAP" in summary.added.get("aoe", [])
    written = json.loads(watchlist_path.read_text(encoding="utf-8"))
    assert "THUNDERCLAP" in written["groups"]["aoe"]["cards"]


def test_generate_pile_watchlist_setup_manual_only(tmp_path) -> None:
    from sts2_env.core.enums import CardId
    from scripts.sync.generate_pile_watchlist import generate_pile_watchlist

    known_without_claw = sorted(name for name in CardId.__members__ if name != "CLAW")
    payload = {
        "version": 1,
        "known_card_ids": known_without_claw,
        "groups": {
            "power": {"cards": [], "auto": None, "exclude": []},
            "finisher": {
                "cards": [],
                "auto": {
                    "card_type": "ATTACK",
                    "min_base_damage": 9,
                    "rarity_min": "UNCOMMON",
                },
                "exclude": [],
            },
            "setup": {"cards": [], "auto": None, "exclude": []},
            "aoe": {"cards": [], "auto": None, "exclude": []},
        },
    }
    watchlist_path = tmp_path / "PILE_WATCHLIST.json"
    watchlist_path.write_text(json.dumps(payload), encoding="utf-8")

    _, summary = generate_pile_watchlist(output=watchlist_path)

    assert "CLAW" in summary.unlisted_new
    assert "CLAW" not in summary.added.get("setup", [])
    assert "CLAW" not in summary.added.get("finisher", [])
