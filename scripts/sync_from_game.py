#!/usr/bin/env python3
"""Sync vakuu-prime simulator content from an installed Slay the Spire 2 copy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.sync.apply_static import apply_all_static, apply_static_summary  # noqa: E402
from scripts.sync.decompile import check_ilspycmd, run_decompile, run_extract_pck  # noqa: E402
from scripts.sync.game_paths import resolve_game_paths, validate_game_paths  # noqa: E402
from scripts.sync.generate_cards_reference import generate_cards_reference  # noqa: E402
from scripts.sync.generate_monsters_reference import generate_monsters_reference  # noqa: E402
from scripts.sync.generate_powers_reference import generate_powers_reference  # noqa: E402
from scripts.sync.generate_relics_reference import generate_relics_reference  # noqa: E402
from scripts.sync.manifest import SyncManifest  # noqa: E402
from scripts.sync.report import write_sync_report  # noqa: E402
from scripts.sync.scaffold import scaffold_all, scaffold_summary  # noqa: E402


def _shared_argument_parser() -> argparse.ArgumentParser:
    """Flags available on every subcommand (e.g. `all --apply`, not `--apply all`)."""
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--apply",
        action="store_true",
        help="Write scaffold/static changes (default is dry-run for those steps)",
    )
    shared.add_argument(
        "--game-path",
        type=Path,
        default=None,
        help="STS2 install root (or set STS2_GAME_PATH)",
    )
    return shared


def _clear_reference_caches() -> None:
    from sts2_env.cards.factory import _decompiled_card_static_metadata, _reference_cards

    _decompiled_card_static_metadata.cache_clear()
    _reference_cards.cache_clear()
    from sts2_env.cards.reference_static_metadata import clear_reference_metadata_caches

    clear_reference_metadata_caches()


def cmd_decompile(args: argparse.Namespace) -> int:
    paths = resolve_game_paths(args.game_path)
    errors = validate_game_paths(paths)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    err = check_ilspycmd()
    if err:
        print(err, file=sys.stderr)
        return 1
    code = run_decompile(paths, backup=not args.no_backup)
    if code != 0:
        return code
    _clear_reference_caches()
    manifest = SyncManifest.from_paths(paths.sts2_dll, str(paths.data_dir))
    manifest.save()
    print(f"Updated sync_manifest.json (DLL SHA256 {manifest.sts2_dll_sha256[:16]}...)")
    unmapped = _print_unmapped_cards()
    if unmapped:
        print("Run: python scripts/sync_from_game.py scaffold --apply")
    return 0


def _print_unmapped_cards() -> tuple[str, ...]:
    from sts2_env.cards.reference_static_metadata import unmapped_reference_card_classes

    unmapped = unmapped_reference_card_classes()
    if unmapped:
        print(
            "Decompiled cards without CardId (add via scaffold --apply): "
            + ", ".join(unmapped),
            file=sys.stderr,
        )
    return unmapped


def cmd_extract_pck(args: argparse.Namespace) -> int:
    paths = resolve_game_paths(args.game_path)
    errors = validate_game_paths(paths)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    return run_extract_pck(paths)


def cmd_docs(_args: argparse.Namespace) -> int:
    paths = [
        generate_cards_reference(),
        generate_powers_reference(),
        generate_relics_reference(),
        generate_monsters_reference(),
    ]
    for path in paths:
        print(f"Wrote {path.relative_to(REPO_ROOT)}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    paths = resolve_game_paths(args.game_path) if args.game_path else None
    dll = paths.sts2_dll if paths else None
    report_path = write_sync_report(dll_path=dll)
    print(f"Wrote {report_path.relative_to(REPO_ROOT)}")
    return 0


def cmd_scaffold(args: argparse.Namespace) -> int:
    apply = args.apply
    result = scaffold_all(apply=apply)
    for line in scaffold_summary(result):
        print(line)
    if apply:
        _clear_reference_caches()
    if not apply and (result.enum_additions or result.card_stubs):
        print("Re-run with --apply to write scaffolds")
    _print_unmapped_cards()
    return 0


def cmd_apply_static(args: argparse.Namespace) -> int:
    apply = args.apply
    result = apply_all_static(apply=apply)
    for line in apply_static_summary(result, apply=apply):
        print(line)
    if not apply and result.patches:
        print("Re-run with --apply to write patches")
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    steps: list[tuple[str, int]] = []
    if not args.skip_decompile:
        code = cmd_decompile(args)
        steps.append(("decompile", code))
        if code != 0:
            return code
    result = scaffold_all(apply=args.apply)
    steps.append(("scaffold", 0))
    for line in scaffold_summary(result):
        print(line)
    if not args.apply and (result.enum_additions or result.card_stubs):
        print("Re-run with --apply to add new CardId entries and stubs")
    code = cmd_docs(args)
    steps.append(("docs", code))
    if code != 0:
        return code
    static_result = apply_all_static(apply=args.apply)
    steps.append(("apply-static", 0))
    paths = resolve_game_paths(args.game_path)
    report_path = write_sync_report(
        dll_path=paths.sts2_dll if paths.sts2_dll.is_file() else None,
        scaffold_summary=scaffold_summary(result),
        static_summary=apply_static_summary(static_result, apply=args.apply),
    )
    print(f"Wrote {report_path.relative_to(REPO_ROOT)}")
    if not args.skip_audits:
        return _run_audits()
    return 0


def _run_audits() -> int:
    import subprocess

    scripts = [
        "scripts/audit_card_static_metadata.py",
        "scripts/audit_card_dynamic_vars.py",
        "scripts/parity_reference_audit.py",
    ]
    failed = False
    for script in scripts:
        print(f"Running {script}...")
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / script)],
            cwd=REPO_ROOT,
        )
        if result.returncode != 0:
            failed = True
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sync simulator content from Slay the Spire 2 game files",
    )
    shared = _shared_argument_parser()
    sub = parser.add_subparsers(dest="command", required=True)

    decompile_p = sub.add_parser(
        "decompile",
        help="Decompile sts2.dll to decompiled/",
        parents=[shared],
    )
    decompile_p.add_argument("--no-backup", action="store_true")
    decompile_p.set_defaults(func=cmd_decompile)

    pck_p = sub.add_parser(
        "extract-pck",
        help="Extract sts2.pck with GDRE Tools",
        parents=[shared],
    )
    pck_p.set_defaults(func=cmd_extract_pck)

    docs_p = sub.add_parser(
        "docs",
        help="Regenerate docs/*_REFERENCE.md",
        parents=[shared],
    )
    docs_p.set_defaults(func=cmd_docs)

    report_p = sub.add_parser(
        "report",
        help="Write sync_report.md",
        parents=[shared],
    )
    report_p.set_defaults(func=cmd_report)

    scaffold_p = sub.add_parser(
        "scaffold",
        help="Add missing CardId stubs",
        parents=[shared],
    )
    scaffold_p.set_defaults(func=cmd_scaffold)

    static_p = sub.add_parser(
        "apply-static",
        help="Patch card static fields in make_*",
        parents=[shared],
    )
    static_p.set_defaults(func=cmd_apply_static)

    all_p = sub.add_parser(
        "all",
        help="decompile + scaffold + docs + apply-static + report",
        parents=[shared],
    )
    all_p.add_argument("--skip-decompile", action="store_true")
    all_p.add_argument("--skip-audits", action="store_true")
    all_p.add_argument("--no-backup", action="store_true")
    all_p.set_defaults(func=cmd_all)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
