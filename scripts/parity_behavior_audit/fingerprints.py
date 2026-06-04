"""Behavior fingerprints for decompiled C# and Python implementations."""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass

from scripts.sync.effect_summary import EFFECT_PATTERNS, _on_play_body

ON_PLAY_METHOD_RE = re.compile(
    r"protected\s+override\s+(?:async\s+)?Task\s+OnPlay\s*\(",
)

CS_EXTRA_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"CardCmd\.AutoPlay"), "Auto-play card"),
    (re.compile(r"CardCmd\.Transform"), "Transform card"),
    (re.compile(r"PotionCmd\."), "Potion action"),
    (re.compile(r"PlayerCmd\.GainGold"), "Gain gold"),
    (re.compile(r"GetDistinctForCombat|CombatCardGeneration"), "Combat card generation"),
    (re.compile(r"StableShuffle"), "Stable shuffle"),
    (re.compile(r"CardSelectCmd\."), "Card choice"),
    (re.compile(r"CardPileCmd\.Discard"), "Discard card(s)"),
    (re.compile(r"CardPileCmd\.Exhaust"), "Exhaust card(s)"),
    (re.compile(r"CardPileCmd\.Shuffle"), "Shuffle pile"),
    (re.compile(r"SummonCmd\."), "Summon"),
    (re.compile(r"StarCmd\.|GainStars"), "Stars"),
    (re.compile(r"SoulCmd\.|GainSoul"), "Souls"),
)

PY_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"apply_damage|_deal_damage|_deal_osty_damage|deal_damage"), "Deal Damage"),
    (re.compile(r"gain_block|_gain_resolved_block|calculate_block"), "Gain Block"),
    (re.compile(r"apply_power_to|apply_power\("), "Apply power"),
    (re.compile(r"draw_cards"), "Draw card(s)"),
    (re.compile(r"add_generated_card|generate_card|add_card_to_discard|add_card_to_hand|draw_pile\.insert|draw_pile\.append|discard_pile\.insert|discard_pile\.append"), "Add generated card(s) to pile"),
    (re.compile(r"request_.*choice|request_card_choice|pending_choice|resolve_pending"), "Card choice"),
    (re.compile(r"set_temporary_cost|set_temporary_free"), "Set card(s) to cost 0"),
    (re.compile(r"create_distinct_character_cards|combat_card_generation_rng"), "Combat card generation"),
    (re.compile(r"gain_energy"), "Gain Energy"),
    (re.compile(r"heal_creature|\.heal\("), "Heal"),
    (re.compile(
        r"_channel_orb|channel_orb|OrbType|_remove_orb_slot|_add_orb_slot|_evoke|_trigger_lightning"
    ), "Orb action"),
    (re.compile(r"forge_stars|combat\.forge|\.forge\("), "Forge"),
    (re.compile(r"summon_osty|summon_minion"), "Summon minion"),
    (re.compile(r"set_temporary_free|cost\s*=\s*0"), "Set card(s) to cost 0"),
    (re.compile(r"exhaust_card|move_card.*exhaust"), "Exhaust"),
    (re.compile(r"upgrade_card|upgrade_hand|upgrade_random_cards|\.upgraded\s*=\s*True|upgraded=card\.upgraded"), "Upgrade card(s)"),
    (re.compile(r"auto_play_card"), "Auto-play card"),
    (re.compile(r"procure_random_potion"), "Potion action"),
    (re.compile(r"gain_gold"), "Gain gold"),
    (re.compile(r"transform_card"), "Transform card"),
    (re.compile(r"combat_card_generation|create_character_cards"), "Combat card generation"),
    (re.compile(r"stable_shuffle"), "Stable shuffle"),
    (re.compile(r"discard_card|move_card.*discard|shuffle"), "Discard card(s)"),
    (re.compile(r"shuffle_draw|shuffle_discard|shuffle_rng\.shuffle|draw_pile.*shuffle"), "Shuffle pile"),
    (re.compile(r"draw_cards\(|combat\.draw_cards"), "Draw card(s)"),
    (re.compile(r"request_preview|preview_card|FromChooseACard"), "Preview card(s)"),
    (re.compile(r"gain_stars|spend_stars"), "Stars"),
    (re.compile(r"gain_soul|soul"), "Souls"),
)

HIGH_IMPACT_LABELS = frozenset({
    "Gain gold",
    "Combat card generation",
    "Auto-play card",
    "Card choice",
    "Stable shuffle",
    "Transform card",
    "Potion action",
    "Summon minion",
    "Stars",
    "Souls",
})


@dataclass(frozen=True)
class FingerprintSet:
    labels: frozenset[str]

    def missing_in(self, other: FingerprintSet) -> frozenset[str]:
        return frozenset(label for label in self.labels if label not in other.labels)


def fingerprint_cs_onplay(source: str) -> FingerprintSet:
    body = _on_play_body(source)
    labels: set[str] = set()
    for pattern, label in (*EFFECT_PATTERNS, *CS_EXTRA_PATTERNS):
        if label == "Exhaust" and not re.search(
            r"CardPileCmd\.Exhaust|ExhaustCmd",
            body,
        ):
            continue
        if label == "Upgrade card(s)" and not re.search(
            r"CardCmd\.Upgrade|\.Upgrade\(",
            body,
        ):
            continue
        if pattern.search(body):
            labels.add(label)
    return FingerprintSet(frozenset(labels))


def fingerprint_cs_hooks(source: str) -> FingerprintSet:
    labels: set[str] = set()
    for match in re.finditer(
        r"public\s+override\s+(?:async\s+)?(?:Task|void|bool|int|decimal)\s+(\w+)\s*\(",
        source,
    ):
        name = match.group(1)
        if name.startswith("get_") or name in {"ToString", "GetHashCode"}:
            continue
        labels.add(name)
    return FingerprintSet(frozenset(labels))


def fingerprint_py_source(source: str) -> FingerprintSet:
    labels: set[str] = set()
    for pattern, label in PY_PATTERNS:
        if pattern.search(source):
            labels.add(label)
    return FingerprintSet(frozenset(labels))


def fingerprint_py_callable(func) -> FingerprintSet:
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        return FingerprintSet(frozenset())
    return fingerprint_py_source(source)


def has_onplay(source: str) -> bool:
    return bool(ON_PLAY_METHOD_RE.search(source))
