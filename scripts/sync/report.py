"""Diff decompiled sources vs manifest and parity audit."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from scripts.sync.common import (
    DECOMPILED_PREV_DIR,
    MODEL_SURFACES,
    REPO_ROOT,
    SYNC_REPORT_PATH,
)
from scripts.sync.generate_pile_watchlist import WatchlistSummary, format_watchlist_report_section
from scripts.sync.manifest import SyncManifest, current_surface_counts, dll_sha256


@dataclass(frozen=True)
class SurfaceDiff:
    surface: str
    added: tuple[str, ...]
    removed: tuple[str, ...]
    changed: tuple[str, ...]


def _file_hashes(directory: Path) -> dict[str, str]:
    import hashlib

    result: dict[str, str] = {}
    if not directory.is_dir():
        return result
    for path in sorted(directory.glob("*.cs")):
        result[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def diff_surfaces(prev_root: Path | None = None) -> list[SurfaceDiff]:
    prev_root = prev_root or DECOMPILED_PREV_DIR
    diffs: list[SurfaceDiff] = []
    for surface, relative in MODEL_SURFACES.items():
        current_dir = REPO_ROOT / relative
        prev_dir = prev_root / Path(relative).name
        if not prev_dir.is_dir():
            prev_hashes: dict[str, str] = {}
        else:
            prev_hashes = _file_hashes(prev_dir)
        current_hashes = _file_hashes(current_dir)
        added = tuple(sorted(set(current_hashes) - set(prev_hashes)))
        removed = tuple(sorted(set(prev_hashes) - set(current_hashes)))
        changed = tuple(
            sorted(
                name
                for name in set(current_hashes) & set(prev_hashes)
                if current_hashes[name] != prev_hashes[name]
            )
        )
        diffs.append(SurfaceDiff(surface, added, removed, changed))
    return diffs


def run_behavior_audit_json() -> dict | None:
    summaries: dict = {}
    for script, key in (
        ("scripts/audit_onplay_behavior_coverage.py", "cards_onplay"),
        ("scripts/audit_relic_hook_coverage.py", "relics"),
    ):
        path = REPO_ROOT / script
        if not path.is_file():
            continue
        try:
            result = subprocess.run(
                [sys.executable, str(path), "--json", "--no-write-backlog"],
                check=True,
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
            )
            summaries[key] = json.loads(result.stdout)
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            continue
    return summaries or None


def run_parity_audit_json(
    *,
    direct_test_references: bool = False,
    include_deprecated: bool = False,
    code_implementation_references: bool = False,
) -> list[dict] | None:
    script = REPO_ROOT / "scripts" / "parity_reference_audit.py"
    if not script.is_file():
        return None
    command = [sys.executable, str(script), "--json"]
    if direct_test_references:
        command.append("--direct-test-references")
    if include_deprecated:
        command.append("--include-deprecated")
    if code_implementation_references:
        command.append("--code-implementation-references")
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        payload = json.loads(result.stdout)
        return payload if isinstance(payload, list) else None
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return None


def parity_reference_audit_has_gaps(
    *,
    direct_test_references: bool = True,
    include_deprecated: bool = True,
    code_implementation_references: bool = True,
) -> tuple[bool, list[str]]:
    """Return (has_gaps, human-readable gap lines) for the strict reference gate."""
    audit = run_parity_audit_json(
        direct_test_references=direct_test_references,
        include_deprecated=include_deprecated,
        code_implementation_references=code_implementation_references,
    )
    if audit is None:
        return True, ["parity reference audit failed to run"]
    lines: list[str] = []
    for row in audit:
        missing_impl = row.get("missing_implementation") or []
        missing_tests = row.get("missing_tests") or []
        if missing_impl or missing_tests:
            surface = row.get("surface", "?")
            lines.append(
                f"{surface}: {len(missing_impl)} missing implementation, "
                f"{len(missing_tests)} missing tests"
            )
    return (bool(lines), lines)


def write_sync_report(
    *,
    dll_path: Path | None = None,
    scaffold_summary: list[str] | None = None,
    static_summary: list[str] | None = None,
    watchlist_summary: WatchlistSummary | None = None,
    output: Path = SYNC_REPORT_PATH,
) -> Path:
    lines: list[str] = ["# STS2 Sync Report", ""]
    manifest = SyncManifest.load()
    if manifest:
        lines.append(f"Last manifest sync: {manifest.synced_at}")
        lines.append(f"Previous DLL SHA256: `{manifest.sts2_dll_sha256[:16]}...`")
        lines.append("")
    if dll_path and dll_path.is_file():
        current_hash = dll_sha256(dll_path)
        lines.append(f"Current DLL SHA256: `{current_hash}`")
        if manifest and manifest.sts2_dll_sha256 != current_hash:
            lines.append("**DLL changed since last manifest.**")
        lines.append("")

    counts = current_surface_counts()
    if manifest:
        lines.append("## Surface file counts")
        lines.append("")
        lines.append("| Surface | Previous | Current | Delta |")
        lines.append("| ------- | -------- | ------- | ----- |")
        for surface, count in sorted(counts.items()):
            prev = manifest.surfaces.get(surface, 0)
            lines.append(f"| {surface} | {prev} | {count} | {count - prev:+d} |")
        lines.append("")

    lines.append("## Decompiled diff (vs decompiled_prev/)")
    lines.append("")
    for diff in diff_surfaces():
        if not (diff.added or diff.removed or diff.changed):
            lines.append(f"### {diff.surface}: no changes")
            continue
        lines.append(f"### {diff.surface}")
        if diff.added:
            lines.append(f"- Added ({len(diff.added)}): " + ", ".join(diff.added[:20]))
            if len(diff.added) > 20:
                lines.append(f"  - ... and {len(diff.added) - 20} more")
        if diff.removed:
            lines.append(f"- Removed ({len(diff.removed)}): " + ", ".join(diff.removed[:20]))
        if diff.changed:
            lines.append(f"- Changed ({len(diff.changed)}): " + ", ".join(diff.changed[:20]))
        lines.append("")

    has_gaps, gap_lines = parity_reference_audit_has_gaps()
    audit = run_parity_audit_json(
        direct_test_references=True,
        include_deprecated=True,
        code_implementation_references=True,
    )
    if audit:
        lines.append("## Parity audit (direct-reference gate)")
        lines.append("")
        if has_gaps:
            lines.append("**Gate failed:** missing implementation or test references detected.")
            lines.extend(f"- {line}" for line in gap_lines)
            lines.append("")
        rows = audit if isinstance(audit, list) else audit.get("summary", [])
        for row in rows:
            if isinstance(row, dict):
                missing_impl = row.get("missing_implementation", [])
                missing_tests = row.get("missing_tests", [])
                impl_count = (
                    len(missing_impl)
                    if isinstance(missing_impl, (list, tuple))
                    else missing_impl
                )
                test_count = (
                    len(missing_tests)
                    if isinstance(missing_tests, (list, tuple))
                    else missing_tests
                )
                lines.append(
                    f"- **{row.get('surface', '?')}**: "
                    f"{impl_count} missing impl, "
                    f"{test_count} missing tests"
                )
            else:
                lines.append(f"- {row}")
        lines.append("")

    behavior = run_behavior_audit_json()
    if behavior:
        lines.append("## Behavioral parity backlog")
        lines.append("")
        cards = behavior.get("cards_onplay", {})
        relics = behavior.get("relics", {})
        if cards:
            lines.append(
                f"- **cards OnPlay**: {cards.get('with_matches_test', 0)}/"
                f"{cards.get('onplay_total', 0)} with Matches tests, "
                f"{cards.get('fingerprint_mismatch', 0)} fingerprint mismatches"
            )
        if relics:
            lines.append(
                f"- **relics**: {relics.get('with_matches_test', 0)}/"
                f"{relics.get('hook_relics_total', 0)} with Matches tests"
            )
        lines.append("")
        lines.append("See [docs/PARITY_BACKLOG.md](../docs/PARITY_BACKLOG.md).")
        lines.append("")

    if scaffold_summary:
        lines.append("## Scaffold")
        lines.append("")
        lines.extend(f"- {line}" for line in scaffold_summary)
        lines.append("")

    if static_summary:
        lines.append("## Static apply")
        lines.append("")
        lines.extend(f"- {line}" for line in static_summary)
        lines.append("")

    lines.extend(format_watchlist_report_section(watchlist_summary))

    lines.append("## Manual work queue")
    lines.append("")
    lines.append("Implement `@register_effect`, power hooks, and monster AI for new/changed classes above.")
    lines.append("")

    output.write_text("\n".join(lines), encoding="utf-8")
    return output
