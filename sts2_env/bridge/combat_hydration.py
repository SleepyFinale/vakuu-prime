"""Hydrate a searchable CombatState from bridge JSON."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from sts2_env.cards.base import reset_instance_counter
from sts2_env.cards.factory import create_card
from sts2_env.core.combat import CombatState
from sts2_env.core.creature import Creature
from sts2_env.core.damage import apply_damage, calculate_damage
from sts2_env.core.enums import CardId, CombatSide, PowerId, ValueProp
from sts2_env.core.rng import Rng
from sts2_env.monsters.intents import Intent, IntentType, attack_intent
from sts2_env.monsters.state_machine import MonsterAI, MoveState
from sts2_env.parity.bridge_replay import combat_state_to_bridge_state
from sts2_env.potions.base import PotionInstance
from sts2_env.relics.registry import create_relic_by_name

_CARD_STR_TO_ID: dict[str, CardId] = {cid.name: cid for cid in CardId}
_POWER_STR_TO_ID: dict[str, PowerId] = {pid.name: pid for pid in PowerId}

# Known monster factories for bridge hydration (extend as needed).
_MONSTER_FACTORIES: dict[str, Any] = {}


def _register_monster_factories() -> None:
    if _MONSTER_FACTORIES:
        return
    from sts2_env.monsters.act1_weak import create_shrinker_beetle

    _MONSTER_FACTORIES["SHRINKER_BEETLE"] = create_shrinker_beetle


@dataclass
class HydrationResult:
    combat: CombatState | None
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.combat is not None


def _parse_card(raw: dict[str, Any]) -> Any:
    card_id_str = str(raw.get("id", "UNKNOWN"))
    card_enum = _CARD_STR_TO_ID.get(card_id_str)
    if card_enum is None:
        for key, value in _CARD_STR_TO_ID.items():
            if key.replace("_", "") == card_id_str.replace("_", ""):
                card_enum = value
                break
    if card_enum is None:
        raise KeyError(f"Unknown card id '{card_id_str}'")
    upgraded = bool(raw.get("upgraded", False))
    card = create_card(card_enum, upgraded=upgraded)
    cost = raw.get("cost")
    if cost is not None:
        card.cost = int(cost)
    return card


def _parse_cards(raw_cards: list[Any]) -> list[Any]:
    cards = []
    for raw in raw_cards:
        if isinstance(raw, dict):
            cards.append(_parse_card(raw))
    return cards


def _apply_powers(creature: Creature, powers_json: list[Any]) -> None:
    from sts2_env.core.creature import get_power_class

    for raw in powers_json:
        if not isinstance(raw, dict):
            continue
        pid_str = str(raw.get("id", ""))
        amount = int(raw.get("amount", 0))
        power_id = _POWER_STR_TO_ID.get(pid_str.upper())
        if power_id is None:
            continue
        cls = get_power_class(power_id)
        if cls is None:
            continue
        creature.powers[power_id] = cls(power_id, amount)


def _bridge_intent_type(intent_str: str) -> IntentType:
    mapping = {
        "ATTACK": IntentType.ATTACK,
        "MULTI_ATTACK": IntentType.MULTI_ATTACK,
        "DEFEND": IntentType.DEFEND,
        "BUFF": IntentType.BUFF,
        "DEBUFF": IntentType.DEBUFF,
        "STUN": IntentType.STUN,
    }
    return mapping.get(intent_str.upper(), IntentType.ATTACK)


def _make_bridge_intent_ai(enemy_json: dict[str, Any]) -> MonsterAI:
    move_id = str(enemy_json.get("intent_move_id") or "BRIDGE_MOVE")
    intent_str = str(enemy_json.get("intent", "ATTACK"))
    damage = int(enemy_json.get("intent_damage") or 0)
    hits = max(1, int(enemy_json.get("intent_hits") or 1))
    intent_type = _bridge_intent_type(intent_str)

    def effect(combat: CombatState) -> None:
        if intent_type not in (IntentType.ATTACK, IntentType.MULTI_ATTACK) or damage <= 0:
            return
        owner = None
        for enemy in combat.enemies:
            if enemy.monster_id == enemy_json.get("id"):
                owner = enemy
                break
        if owner is None or not owner.is_alive:
            return
        for _ in range(hits):
            target = combat.player
            if not target.is_alive:
                break
            dmg = calculate_damage(damage, owner, target, ValueProp.MOVE, combat)
            apply_damage(target, dmg, ValueProp.MOVE, combat, owner)
            combat._check_combat_end()  # noqa: SLF001
            if combat.is_over:
                break

    intents = [attack_intent(damage, hits)] if damage > 0 else [Intent(intent_type)]
    states = {
        move_id: MoveState(move_id, effect, intents, follow_up_id=move_id),
    }
    ai = MonsterAI(states, move_id)
    ai.set_forced_move(move_id)
    return ai


def _make_enemy(enemy_json: dict[str, Any], rng: Rng, warnings: list[str]) -> tuple[Creature, MonsterAI]:
    _register_monster_factories()
    monster_id = str(enemy_json.get("id", "UNKNOWN"))
    factory = _MONSTER_FACTORIES.get(monster_id)
    if factory is not None:
        creature, ai = factory(rng)
        creature.max_hp = int(enemy_json.get("max_hp", creature.max_hp))
        creature.current_hp = int(enemy_json.get("hp", creature.current_hp))
        creature.block = int(enemy_json.get("block", 0))
        move_id = enemy_json.get("intent_move_id")
        if move_id:
            ai.set_forced_move(str(move_id))
        _apply_powers(creature, enemy_json.get("powers", []))
        return creature, ai

    creature = Creature(
        max_hp=int(enemy_json.get("max_hp", 1)),
        current_hp=int(enemy_json.get("hp", 0)),
        side=CombatSide.ENEMY,
        monster_id=monster_id,
    )
    creature.block = int(enemy_json.get("block", 0))
    _apply_powers(creature, enemy_json.get("powers", []))
    warnings.append(f"Using bridge intent stub AI for monster '{monster_id}'")
    return creature, _make_bridge_intent_ai(enemy_json)


def _rng_seed_from_state(state: dict[str, Any]) -> int:
    payload = repr(sorted(state.items())).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], "big")


def combat_to_full_bridge_state(combat: CombatState) -> dict[str, Any]:
    """Export combat to bridge-like JSON including full pile lists."""
    base = combat_state_to_bridge_state(combat)
    if base.get("type") == "card_select":
        return base

    def _card_dict(card: Any) -> dict[str, Any]:
        return {
            "id": card.card_id.name,
            "cost": card.cost,
            "type": card.card_type.name.title(),
            "target": card.target_type.name,
            "playable": combat.can_play_card(card),
            "upgraded": card.upgraded or None,
            "base_damage": card.base_damage,
            "base_block": card.base_block,
        }

    base["draw_pile"] = [_card_dict(c) for c in combat.draw_pile]
    base["discard_pile"] = [_card_dict(c) for c in combat.discard_pile]
    base["play_pile"] = [_card_dict(c) for c in combat.play_pile]
    base["hand"] = [_card_dict(c) for c in combat.hand]
    base["exhaust_pile_count"] = len(combat.exhaust_pile)

    enemies_out = []
    for enemy in combat.enemies:
        enemy_data = next(
            (e for e in base.get("enemies", []) if e.get("id") == enemy.monster_id),
            None,
        )
        if enemy_data is None:
            enemy_data = {
                "id": enemy.monster_id or "UNKNOWN",
                "hp": enemy.current_hp,
                "max_hp": enemy.max_hp,
                "block": enemy.block,
                "is_alive": enemy.is_alive,
            }
        ai = combat.enemy_ais.get(enemy.combat_id)
        if ai is not None:
            enemy_data["intent_move_id"] = ai.current_move.state_id
        enemies_out.append(enemy_data)
    base["enemies"] = enemies_out
    base["player"]["character_id"] = combat.character_id
    return base


def hydrate_combat_from_bridge(
    state: dict[str, Any],
    *,
    character_id: str | None = None,
) -> HydrationResult:
    """Build a searchable ``CombatState`` from bridge combat JSON."""
    warnings: list[str] = []
    combat_json = state.get("combat_state") or state
    if "player" not in combat_json:
        return HydrationResult(None, ["Missing player in bridge state"])

    player_json = combat_json["player"]
    resolved_character = character_id or player_json.get("character_id") or "Ironclad"

    try:
        hand = _parse_cards(combat_json.get("hand", []))
        draw = _parse_cards(combat_json.get("draw_pile", []))
        discard = _parse_cards(combat_json.get("discard_pile", []))
        play = _parse_cards(combat_json.get("play_pile", []))
    except KeyError as exc:
        return HydrationResult(None, [str(exc)])

    deck = draw + discard + play + hand
    if not deck:
        return HydrationResult(None, ["No cards in bridge state"])

    rng_seed = _rng_seed_from_state(combat_json)
    rng = Rng(rng_seed)

    relics_json = combat_json.get("relics") or []
    relics = []
    for raw in relics_json:
        if isinstance(raw, dict) and raw.get("id"):
            try:
                relics.append(create_relic_by_name(str(raw["id"])))
            except KeyError:
                warnings.append(f"Unknown relic '{raw.get('id')}'")

    potions_json = combat_json.get("potions") or []
    potions: list[PotionInstance | None] = []
    for raw in potions_json:
        if raw is None:
            potions.append(None)
        elif isinstance(raw, dict) and raw.get("id"):
            from sts2_env.potions.registry import create_potion_by_name

            try:
                potions.append(create_potion_by_name(str(raw["id"])))
            except KeyError:
                potions.append(None)
                warnings.append(f"Unknown potion '{raw.get('id')}'")
        else:
            potions.append(None)

    combat = CombatState(
        player_hp=int(player_json.get("hp", 1)),
        player_max_hp=int(player_json.get("max_hp", 1)),
        deck=list(deck),
        rng_seed=rng_seed,
        relics=relics,
        character_id=resolved_character,
        potions=potions,
        max_potion_slots=max(len(potions), 3),
    )

    combat.hand.clear()
    combat.hand.extend(hand)
    combat.draw_pile.clear()
    combat.draw_pile.extend(draw)
    combat.discard_pile.clear()
    combat.discard_pile.extend(discard)
    combat.play_pile.clear()
    combat.play_pile.extend(play)

    combat.player.block = int(player_json.get("block", 0))
    combat.energy = int(player_json.get("energy", 0))
    combat.base_max_energy = int(player_json.get("max_energy", combat.base_max_energy))
    combat.stars = int(player_json.get("stars", 0))
    _apply_powers(combat.player, player_json.get("powers", []))

    combat.round_number = int(combat_json.get("round", 1))
    combat.turn_count = max(0, combat.round_number - 1)
    combat._combat_started = True
    combat.in_play_phase = True
    combat.current_side = CombatSide.PLAYER

    for enemy_json in combat_json.get("enemies", []):
        if not isinstance(enemy_json, dict):
            continue
        if not enemy_json.get("is_alive", True):
            continue
        creature, ai = _make_enemy(enemy_json, rng, warnings)
        combat.add_enemy(creature, ai)

    return HydrationResult(combat, warnings)
