"""Deck templates for combat curriculum training."""

from __future__ import annotations

from typing import Literal

from sts2_env.cards.base import CardInstance
from sts2_env.characters.all import create_starting_deck, get_character
from sts2_env.core.enums import CardId

DeckTemplateName = Literal[
    "starter",
    "stripped",
    "ironclad_exhaust",
    "necrobinder_starter",
]

SUPPORTED_DECK_TEMPLATES: tuple[str, ...] = (
    "starter",
    "stripped",
    "ironclad_exhaust",
    "necrobinder_starter",
)

# Character-specific power cards removed by the stripped template.
_STRIPPED_EXCLUDE: dict[str, frozenset[CardId]] = {
    "Ironclad": frozenset({CardId.BASH}),
    "Silent": frozenset({CardId.NEUTRALIZE, CardId.SURVIVOR}),
    "Defect": frozenset({CardId.ZAP, CardId.DUALCAST}),
    "Regent": frozenset({CardId.FALLING_STAR, CardId.VENERATE}),
    "Necrobinder": frozenset({CardId.BODYGUARD, CardId.UNLEASH}),
}


def build_deck_template(
    template: str,
    character_id: str,
) -> list[CardInstance]:
    """Build a deck for curriculum episodes."""
    if template == "starter":
        return create_starting_deck(character_id)
    if template == "necrobinder_starter":
        return create_starting_deck("Necrobinder")
    if template == "stripped":
        return _build_stripped_deck(character_id)
    if template == "ironclad_exhaust":
        if character_id != "Ironclad":
            raise ValueError(
                f"ironclad_exhaust template requires Ironclad, got {character_id!r}"
            )
        return _build_ironclad_exhaust_deck()
    raise ValueError(
        f"Unknown deck template {template!r}. "
        f"Valid: {', '.join(SUPPORTED_DECK_TEMPLATES)}"
    )


def _build_stripped_deck(character_id: str) -> list[CardInstance]:
    """Strikes and defends only — no character power card."""
    from sts2_env.cards.base import _get_next_id

    cfg = get_character(character_id)
    exclude = _STRIPPED_EXCLUDE.get(character_id, frozenset())
    starter = create_starting_deck(character_id)
    prototypes = {card.card_id: card for card in starter}
    deck: list[CardInstance] = []
    for card_id, count in cfg.starting_deck:
        if card_id in exclude:
            continue
        prototype = prototypes.get(card_id)
        if prototype is None:
            raise ValueError(f"No starter factory for {card_id} on {character_id}")
        for _ in range(count):
            deck.append(prototype.clone(_get_next_id()))
    return deck


def _build_ironclad_exhaust_deck() -> list[CardInstance]:
    from sts2_env.cards.ironclad import make_infernal_blade, make_molten_fist

    deck = create_starting_deck("Ironclad")
    deck.append(make_molten_fist())
    deck.append(make_infernal_blade())
    return deck


def sample_deck_template(
    rng,
    templates: tuple[str, ...],
    character_id: str,
) -> tuple[str, list[CardInstance]]:
    """Pick a deck template uniformly and build the deck."""
    if not templates:
        templates = ("starter",)
    index = int(rng.integers(0, len(templates)))
    name = templates[index]
    return name, build_deck_template(name, character_id)
