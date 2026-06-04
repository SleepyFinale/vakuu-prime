"""Matches NotYet.cs: placeholder card with no combat play effect."""

from sts2_env.cards.factory import create_card
from sts2_env.core.enums import CardId


def test_not_yet_is_placeholder_card():
    """Matches NotYet.cs: stub card exists for timeline unlock gating only."""
    card = create_card(CardId.NOT_YET)
    assert card.card_id == CardId.NOT_YET
