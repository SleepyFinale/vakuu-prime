"""Tests for combat curriculum deck templates."""

from sts2_env.core.enums import CardId
from sts2_env.training.deck_templates import build_deck_template


def test_stripped_ironclad_has_no_bash() -> None:
    deck = build_deck_template("stripped", "Ironclad")
    card_ids = {card.card_id for card in deck}
    assert CardId.BASH not in card_ids
    assert CardId.STRIKE_IRONCLAD in card_ids
    assert CardId.DEFEND_IRONCLAD in card_ids
    assert len(deck) == 9


def test_ironclad_exhaust_contains_exhaust_cards() -> None:
    deck = build_deck_template("ironclad_exhaust", "Ironclad")
    assert len(deck) == 12
    exhaust_cards = [card for card in deck if "exhaust" in card.keywords]
    assert len(exhaust_cards) == 2


def test_necrobinder_starter_matches_starter() -> None:
    starter = build_deck_template("starter", "Necrobinder")
    explicit = build_deck_template("necrobinder_starter", "Necrobinder")
    assert len(starter) == len(explicit)
    assert {card.card_id for card in starter} == {card.card_id for card in explicit}
