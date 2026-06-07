"""Fast combat-state clone for MCTS branching."""

from __future__ import annotations

from dataclasses import replace

from sts2_env.cards.base import CardInstance
from sts2_env.core.attack import AttackContext
from sts2_env.core.combat import CardPlayFinishedEntry, CardPlayStartedEntry, CombatState
from sts2_env.core.combat_player import CombatPlayerState
from sts2_env.core.creature import Creature
from sts2_env.core.enums import PowerId
from sts2_env.core.selection import CardChoiceOption, PendingCardChoice
from sts2_env.monsters.state_machine import MonsterAI
from sts2_env.orbs.base import OrbQueue
from sts2_env.powers.base import PowerInstance


def clone_combat_state(combat: CombatState) -> CombatState:
    """Return an independent copy of *combat* suitable for simulated branching."""
    card_cache: dict[int, CardInstance] = {}
    creature_map: dict[int, Creature] = {}

    for creature in _collect_all_creatures(combat):
        _copy_creature(creature, creature_map)

    new = object.__new__(CombatState)
    new.rng = combat.rng.copy()
    new.room = combat.room
    new.ascension_level = combat.ascension_level

    new._root_player = creature_map[id(combat._root_player)]
    new.enemies = [creature_map[id(enemy)] for enemy in combat.enemies]
    new.allies = [creature_map[id(ally)] for ally in combat.allies]
    new.osty = _map_creature(combat.osty, creature_map)

    new._acting_player = _map_creature(combat._acting_player, creature_map)

    new._primary_player_state = _copy_combat_player_state(
        combat._primary_player_state,
        new._root_player,
        card_cache,
    )
    new._ally_player_states = [
        _copy_combat_player_state(
            state,
            creature_map[id(state.creature)],
            card_cache,
        )
        for state in combat._ally_player_states
    ]

    _remap_creature_links(combat, creature_map)
    _remap_power_appliers(creature_map)

    new._combat_player_state_by_creature = {
        new._root_player: new._primary_player_state,
    }
    for ally_state in new._ally_player_states:
        new._combat_player_state_by_creature[ally_state.creature] = ally_state

    new._acting_player_state = (
        new._combat_player_state_by_creature.get(new._acting_player)
        if new._acting_player is not None
        else None
    )

    new.enemy_ais = {
        combat_id: _copy_monster_ai(ai)
        for combat_id, ai in combat.enemy_ais.items()
    }

    new._ally_player_zones = {
        _map_creature(creature, creature_map): {
            zone: [_copy_card(card, card_cache) for card in cards]
            for zone, cards in zones.items()
        }
        for creature, zones in combat._ally_player_zones.items()
    }

    new.round_number = combat.round_number
    new.current_side = combat.current_side
    new.is_over = combat.is_over
    new.player_won = combat.player_won
    new.turn_count = combat.turn_count
    new._pending_retain_count = dict(combat._pending_retain_count)
    new.pending_choice = _copy_pending_choice(combat.pending_choice, card_cache)
    new._pending_play = dict(combat._pending_play) if combat._pending_play is not None else None
    new._pending_draw = dict(combat._pending_draw) if combat._pending_draw is not None else None
    new._pending_turn_setup = combat._pending_turn_setup
    new._end_turn_after_play = combat._end_turn_after_play
    new.in_play_phase = combat.in_play_phase

    new._damage_events_this_turn = _copy_damage_events(combat._damage_events_this_turn, creature_map)
    new._damage_events_combat = _copy_damage_events_combat(combat._damage_events_combat, creature_map)
    new._power_events_combat = _copy_power_events(combat._power_events_combat, creature_map)
    new._block_events_this_turn = _copy_block_events(combat._block_events_this_turn, creature_map)
    new._draw_events_this_turn = _copy_draw_events(combat._draw_events_this_turn, creature_map, card_cache)
    new._draw_events_combat = [
        _map_creature(creature, creature_map) for creature in combat._draw_events_combat
    ]
    new._exhaust_events_this_turn = [_copy_card(card, card_cache) for card in combat._exhaust_events_this_turn]
    new._discard_events_this_turn = [_copy_card(card, card_cache) for card in combat._discard_events_this_turn]
    new._stars_gained_this_turn = [
        (_map_creature(creature, creature_map), amount)
        for creature, amount in combat._stars_gained_this_turn
    ]
    new._power_events_this_turn = _copy_power_events(combat._power_events_this_turn, creature_map)
    new._generated_cards_combat = [
        (_map_creature(creature, creature_map), flag)
        for creature, flag in combat._generated_cards_combat
    ]
    new._energy_spent_this_turn = {
        _map_creature(creature, creature_map): amount
        for creature, amount in combat._energy_spent_this_turn.items()
    }
    new._after_energy_reset_owners_this_turn = {
        _map_creature(creature, creature_map)
        for creature in combat._after_energy_reset_owners_this_turn
    }
    new._orb_channel_events_combat = [
        (_map_creature(creature, creature_map), orb)
        for creature, orb in combat._orb_channel_events_combat
    ]

    new._active_card_source = combat._active_card_source
    new._active_card_target = _map_creature(combat._active_card_target, creature_map)
    new._card_being_played_for_cost = (
        _copy_card(combat._card_being_played_for_cost, card_cache)
        if combat._card_being_played_for_cost is not None
        else None
    )
    new._attack_context_stack = [
        _copy_attack_context(ctx, creature_map) for ctx in combat._attack_context_stack
    ]
    new._pending_auto_attack = (
        _copy_attack_context(combat._pending_auto_attack, creature_map)
        if combat._pending_auto_attack is not None
        else None
    )

    new._combat_started = combat._combat_started
    new.extra_card_rewards = combat.extra_card_rewards
    new._legacy_extra_card_rewards = combat._legacy_extra_card_rewards
    new._played_cards_this_turn = [
        _copy_card(card, card_cache) for card in combat._played_cards_this_turn
    ]
    new._played_cards_combat = [
        _copy_card(card, card_cache) for card in combat._played_cards_combat
    ]
    new._card_play_finished_entries_combat = [
        CardPlayFinishedEntry(
            card=_copy_card(entry.card, card_cache),
            was_ethereal=entry.was_ethereal,
            round_number=entry.round_number,
        )
        for entry in combat._card_play_finished_entries_combat
    ]
    new._card_play_starts_this_turn = [
        CardPlayStartedEntry(
            card=_copy_card(entry.card, card_cache),
            is_first_in_series=entry.is_first_in_series,
            energy_value=entry.energy_value,
        )
        for entry in combat._card_play_starts_this_turn
    ]
    new._card_play_round_counts = {
        (instance_id, _map_creature(creature, creature_map)): count
        for (instance_id, creature), count in combat._card_play_round_counts.items()
    }

    _rewire_combat_pointers(new)
    return new


def _copy_card(card: CardInstance, cache: dict[int, CardInstance]) -> CardInstance:
    key = id(card)
    cached = cache.get(key)
    if cached is not None:
        return cached

    deck_version = (
        _copy_card(card.deck_version, cache) if card.deck_version is not None else None
    )
    copied = CardInstance(
        card_id=card.card_id,
        cost=card.cost,
        card_type=card.card_type,
        target_type=card.target_type,
        rarity=card.rarity,
        base_damage=card.base_damage,
        base_block=card.base_block,
        upgraded=card.upgraded,
        keywords=card.keywords,
        tags=card.tags,
        can_be_generated_in_combat=card.can_be_generated_in_combat,
        can_be_generated_by_modifiers=card.can_be_generated_by_modifiers,
        enchantments=dict(card.enchantments),
        effect_vars=dict(card.effect_vars),
        instance_id=card.instance_id,
        has_energy_cost_x=card.has_energy_cost_x,
        star_cost=card.star_cost,
        has_star_cost_x=card.has_star_cost_x,
        has_turn_end_in_hand_effect=card.has_turn_end_in_hand_effect,
        combat_vars=dict(card.combat_vars),
        original_cost=card.original_cost,
        single_turn_retain=card.single_turn_retain,
        bound=card.bound,
        base_replay_count=card.base_replay_count,
        deck_version=deck_version,
    )
    cache[key] = copied
    return copied


def _copy_creature(creature: Creature, creature_map: dict[int, Creature]) -> Creature:
    key = id(creature)
    cached = creature_map.get(key)
    if cached is not None:
        return cached

    copied = Creature(
        max_hp=creature.max_hp,
        current_hp=creature.current_hp,
        side=creature.side,
        is_player=creature.is_player,
        monster_id=creature.monster_id,
        combat_id=creature.combat_id,
    )
    copied.block = creature.block
    copied.powers = {
        power_id: _copy_power_instance(power)
        for power_id, power in creature.powers.items()
    }
    copied.stars = creature.stars
    copied.is_pet = creature.is_pet
    copied.is_osty = creature.is_osty
    copied.escaped = creature.escaped
    copied._death_processed = creature._death_processed
    creature_map[key] = copied
    return copied


def _copy_power_instance(power: PowerInstance) -> PowerInstance:
    copied = PowerInstance(power.power_id, power.amount)
    copied.skip_next_tick = power.skip_next_tick
    copied.ignore_next_instance = power.ignore_next_instance
    copied.applier = power.applier
    return copied


def _map_creature(creature: Creature | None, creature_map: dict[int, Creature]) -> Creature | None:
    if creature is None:
        return None
    return creature_map[id(creature)]


def _add_creature_ref(refs: list[Creature], seen: set[int], creature: Creature | None) -> None:
    if creature is None:
        return
    key = id(creature)
    if key in seen:
        return
    seen.add(key)
    refs.append(creature)


def _collect_all_creatures(combat: CombatState) -> list[Creature]:
    refs: list[Creature] = []
    seen: set[int] = set()

    _add_creature_ref(refs, seen, combat._root_player)
    _add_creature_ref(refs, seen, combat._acting_player)
    for enemy in combat.enemies:
        _add_creature_ref(refs, seen, enemy)
    for ally in combat.allies:
        _add_creature_ref(refs, seen, ally)
    _add_creature_ref(refs, seen, combat.osty)
    for state in combat._ally_player_states:
        _add_creature_ref(refs, seen, state.creature)

    for dealer, target, _prop in combat._damage_events_this_turn:
        _add_creature_ref(refs, seen, dealer)
        _add_creature_ref(refs, seen, target)
    for dealer, target, _prop, _a, _b in combat._damage_events_combat:
        _add_creature_ref(refs, seen, dealer)
        _add_creature_ref(refs, seen, target)
    for owner, _power_id, _amount, applier in combat._power_events_combat:
        _add_creature_ref(refs, seen, owner)
        _add_creature_ref(refs, seen, applier)
    for owner, _prop, _source in combat._block_events_this_turn:
        _add_creature_ref(refs, seen, owner)
    for owner, _card, _flag in combat._draw_events_this_turn:
        _add_creature_ref(refs, seen, owner)
    for creature in combat._draw_events_combat:
        _add_creature_ref(refs, seen, creature)
    for creature, _amount in combat._stars_gained_this_turn:
        _add_creature_ref(refs, seen, creature)
    for owner, _power_id, _amount, applier in combat._power_events_this_turn:
        _add_creature_ref(refs, seen, owner)
        _add_creature_ref(refs, seen, applier)
    for creature, _flag in combat._generated_cards_combat:
        _add_creature_ref(refs, seen, creature)
    for creature in combat._energy_spent_this_turn:
        _add_creature_ref(refs, seen, creature)
    for creature in combat._after_energy_reset_owners_this_turn:
        _add_creature_ref(refs, seen, creature)
    for creature, _orb in combat._orb_channel_events_combat:
        _add_creature_ref(refs, seen, creature)

    _add_creature_ref(refs, seen, combat._active_card_target)
    for ctx in combat._attack_context_stack:
        _add_creature_ref(refs, seen, ctx.attacker)
        _add_creature_ref(refs, seen, ctx.target)
    if combat._pending_auto_attack is not None:
        _add_creature_ref(refs, seen, combat._pending_auto_attack.attacker)
        _add_creature_ref(refs, seen, combat._pending_auto_attack.target)

    for creature in combat._combat_player_state_by_creature:
        _add_creature_ref(refs, seen, creature)
    for _instance_id, round_creature in combat._card_play_round_counts:
        _add_creature_ref(refs, seen, round_creature)
    for power_owner in (
        *combat._root_player.powers.values(),
        *(power for enemy in combat.enemies for power in enemy.powers.values()),
        *(power for ally in combat.allies for power in ally.powers.values()),
    ):
        _add_creature_ref(refs, seen, power_owner.applier)

    return refs


def _remap_creature_links(combat: CombatState, creature_map: dict[int, Creature]) -> None:
    for old_creature in _collect_all_creatures(combat):
        new_creature = creature_map[id(old_creature)]
        new_creature.pet_owner = _map_creature(old_creature.pet_owner, creature_map)
        new_creature.owner = _map_creature(old_creature.owner, creature_map)


def _remap_power_appliers(creature_map: dict[int, Creature]) -> None:
    for new_creature in creature_map.values():
        for power in new_creature.powers.values():
            if power.applier is not None:
                mapped = creature_map.get(id(power.applier))
                if mapped is not None:
                    power.applier = mapped


def _copy_monster_ai(ai: MonsterAI) -> MonsterAI:
    copied = object.__new__(MonsterAI)
    copied.states = ai.states
    copied.state_log = list(ai.state_log)
    copied._current_state_id = ai._current_state_id
    copied._performed_first_move = ai._performed_first_move
    copied._forced_move_id = ai._forced_move_id
    return copied


def _copy_orb_queue(queue: OrbQueue | None) -> OrbQueue | None:
    if queue is None:
        return None
    copied = OrbQueue(queue.capacity)
    copied.orbs = list(queue.orbs)
    return copied


def _copy_combat_player_state(
    state: CombatPlayerState,
    creature: Creature,
    card_cache: dict[int, CardInstance],
) -> CombatPlayerState:
    hand = [_copy_card(card, card_cache) for card in state.hand]
    draw = [_copy_card(card, card_cache) for card in state.draw]
    discard = [_copy_card(card, card_cache) for card in state.discard]
    exhaust = [_copy_card(card, card_cache) for card in state.exhaust]
    play = [_copy_card(card, card_cache) for card in state.play]
    starting_deck = [_copy_card(card, card_cache) for card in state.starting_deck]

    copied = object.__new__(CombatPlayerState)
    copied.player_state = state.player_state
    copied.creature = creature
    copied.starting_deck = starting_deck
    copied.hand = hand
    copied.draw = draw
    copied.discard = discard
    copied.exhaust = exhaust
    copied.play = play
    copied.relics = list(state.relics)
    copied.potions = list(state.potions)
    copied.max_potion_slots = state.max_potion_slots
    copied.energy = state.energy
    copied.stars = state.stars
    copied.base_max_energy = state.base_max_energy
    copied.orb_queue = _copy_orb_queue(state.orb_queue)
    copied.zone_map = {
        "hand": hand,
        "draw": draw,
        "discard": discard,
        "exhaust": exhaust,
    }
    return copied


def _copy_pending_choice(
    choice: PendingCardChoice | None,
    card_cache: dict[int, CardInstance],
) -> PendingCardChoice | None:
    if choice is None:
        return None
    return PendingCardChoice(
        prompt=choice.prompt,
        options=[
            CardChoiceOption(card=_copy_card(option.card, card_cache), source_pile=option.source_pile)
            for option in choice.options
        ],
        resolver=choice.resolver,
        allow_skip=choice.allow_skip,
        min_choices=choice.min_choices,
        max_choices=choice.max_choices,
        selected_indices=set(choice.selected_indices),
    )


def _copy_attack_context(ctx: AttackContext, creature_map: dict[int, Creature]) -> AttackContext:
    return replace(
        ctx,
        attacker=_map_creature(ctx.attacker, creature_map),
        target=_map_creature(ctx.target, creature_map),
        results=list(ctx.results),
    )


def _copy_damage_events(
    events: list[tuple[Creature | None, Creature, object]],
    creature_map: dict[int, Creature],
) -> list[tuple[Creature | None, Creature, object]]:
    return [
        (_map_creature(dealer, creature_map), _map_creature(target, creature_map), prop)
        for dealer, target, prop in events
    ]


def _copy_damage_events_combat(
    events: list[tuple[Creature | None, Creature, object, int, int]],
    creature_map: dict[int, Creature],
) -> list[tuple[Creature | None, Creature, object, int, int]]:
    return [
        (
            _map_creature(dealer, creature_map),
            _map_creature(target, creature_map),
            prop,
            a,
            b,
        )
        for dealer, target, prop, a, b in events
    ]


def _copy_power_events(
    events: list[tuple[Creature, PowerId, int, Creature | None]],
    creature_map: dict[int, Creature],
) -> list[tuple[Creature, PowerId, int, Creature | None]]:
    return [
        (
            _map_creature(owner, creature_map),
            power_id,
            amount,
            _map_creature(applier, creature_map),
        )
        for owner, power_id, amount, applier in events
    ]


def _copy_block_events(
    events: list[tuple[Creature, object, object | None]],
    creature_map: dict[int, Creature],
) -> list[tuple[Creature, object, object | None]]:
    return [
        (_map_creature(owner, creature_map), prop, source)
        for owner, prop, source in events
    ]


def _copy_draw_events(
    events: list[tuple[Creature, CardInstance, bool]],
    creature_map: dict[int, Creature],
    card_cache: dict[int, CardInstance],
) -> list[tuple[Creature, CardInstance, bool]]:
    return [
        (_map_creature(owner, creature_map), _copy_card(card, card_cache), flag)
        for owner, card, flag in events
    ]


def _rewire_combat_pointers(combat: CombatState) -> None:
    """Ensure every creature points at this combat after cloning."""
    combat._root_player.combat_state = combat
    for enemy in combat.enemies:
        enemy.combat_state = combat
    for ally in combat.allies:
        ally.combat_state = combat
    if combat.osty is not None:
        combat.osty.combat_state = combat
    for creature in combat._combat_player_state_by_creature:
        creature.combat_state = combat
    acting = combat._acting_player
    if acting is not None:
        acting.combat_state = combat
