#!/usr/bin/env python3
"""Compare wiki.gg card metadata to decompiled static metadata (informational only)."""

from __future__ import annotations

import argparse
import re
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

WIKI_URL = "https://slaythespire.wiki.gg/wiki/Slay_the_Spire_2:Cards_List"


def _fetch_wiki_text() -> str:
    request = urllib.request.Request(
        WIKI_URL,
        headers={"User-Agent": "vakuu-prime-parity-audit/1.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="ignore")


def _wiki_card_names(html: str) -> set[str]:
    names: set[str] = set()
    for match in re.finditer(r"/wiki/Slay_the_Spire_2:([A-Za-z0-9_]+)", html):
        name = match.group(1)
        if name in {"Cards_List", "Main", "Cards"}:
            continue
        names.add(name)
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Skip network fetch")
    args = parser.parse_args()

    decompiled_dir = REPO_ROOT / "decompiled/MegaCrit.Sts2.Core.Models.Cards"
    decomp_names = {p.stem for p in decompiled_dir.glob("*.cs") if "Deprecated" not in p.stem}

    if args.offline:
        print("wiki audit skipped (--offline)")
        return 0

    try:
        html = _fetch_wiki_text()
    except OSError as exc:
        print(f"wiki fetch failed: {exc}")
        return 0

    wiki_names = _wiki_card_names(html)
    only_wiki = sorted(wiki_names - decomp_names)
    only_decomp = sorted(decomp_names - wiki_names)

    print(f"wiki cards linked: {len(wiki_names)}")
    print(f"decompiled cards: {len(decomp_names)}")
    print(f"wiki only: {len(only_wiki)}")
    print(f"decompiled only: {len(only_decomp)}")
    if only_decomp[:15]:
        print("sample decompiled-only:", ", ".join(only_decomp[:15]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
