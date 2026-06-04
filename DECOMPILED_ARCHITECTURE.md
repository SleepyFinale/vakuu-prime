# STS2 Decompiled Source Architecture — Simulator Implementation Guide

This document is based on a deep analysis of decompiled C# source from `sts2.dll` and records the core game mechanics needed to build a headless simulator.

---

## 1. Key Namespace Navigation

```text
extraction/decompiled/
├── MegaCrit.Sts2.Core.Combat/              Combat manager, combat state, sides
├── MegaCrit.Sts2.Core.Commands/            Low-level commands (damage, block, cards, powers)
│   └── Builders/                           AttackCommand builder pattern
├── MegaCrit.Sts2.Core.GameActions/         High-level actions (play card, end turn)
├── MegaCrit.Sts2.Core.Entities.Creatures/  Creature base (HP, Block, Powers)
├── MegaCrit.Sts2.Core.Entities.Players/    PlayerCombatState (energy, stars, piles)
├── MegaCrit.Sts2.Core.Entities.Cards/      TargetType, CardKeyword, PileType
├── MegaCrit.Sts2.Core.Hooks/              Hook static class (central event dispatch)
├── MegaCrit.Sts2.Core.Models/             Base types: AbstractModel, CardModel, MonsterModel, PowerModel, RelicModel
├── MegaCrit.Sts2.Core.Models.Cards/       All card implementations (~576)
├── MegaCrit.Sts2.Core.Models.Monsters/    All monster AI implementations (~111)
├── MegaCrit.Sts2.Core.Models.Powers/      All power implementations (~260)
├── MegaCrit.Sts2.Core.Models.Relics/      All relic implementations (~289)
├── MegaCrit.Sts2.Core.Models.Encounters/  Encounter definitions (monster groups)
├── MegaCrit.Sts2.Core.Models.Acts/        Act definitions (Overgrowth, Underdocks, Hive, Glory)
├── MegaCrit.Sts2.Core.MonsterMoves.MonsterMoveStateMachine/  Monster AI state machine
├── MegaCrit.Sts2.Core.MonsterMoves.Intents/                  Intent types
├── MegaCrit.Sts2.Core.Map/               Map generation (StandardActMap, MapPoint)
└── MegaCrit.Sts2.Core.ValueProps/        ValueProp flags
```

---

## 2. Combat System

### 2.1 Core class relationships

```text
CombatManager (singleton)
  └── CombatState
       ├── CombatSide (Player / Enemy)
       ├── RoundNumber (starts at 1)
       ├── PlayerCombatState
       │    ├── Energy / MaxEnergy
       │    ├── Stars (STS2 new resource)
       │    ├── Hand, DrawPile, DiscardPile, ExhaustPile, PlayPile
       │    └── HasEnoughResourcesFor(card) — checks energy + stars
       └── Creatures[] (monster list)
            ├── HP / MaxHP / Block
            └── Powers[] (power list)
```

### 2.2 Turn flow

```text
StartCombat()
  └── Hook.BeforeCombatStart (relic triggers, e.g. Anchor grants 10 block)

Each round {
  ┌─ Player turn ──────────────────────────────────────────────────────────┐
  │ StartTurn()                                                            │
  │   ├── Clear block (except turn 1; controlled by Hook.ShouldClearBlock) │
  │   ├── ResetEnergy() → refill to MaxEnergy                              │
  │   ├── Hook.ModifyHandDraw → draw count (default 5)                     │
  │   └── Draw cards                                                       │
  │                                                                        │
  │ Player action phase (play cards / use potions / end turn)              │
  │                                                                        │
  │ EndPlayerTurnPhaseOne()                                                │
  │   ├── Discard Ethereal cards                                           │
  │   └── Trigger end-of-turn effects                                      │
  │ EndPlayerTurnPhaseTwo()                                                │
  │   ├── Discard hand (keep Retain cards)                                 │
  │   └── Powers tick down                                                 │
  └────────────────────────────────────────────────────────────────────────┘

  ┌─ Enemy turn ─────────────────────────────────────┐
  │ SwitchFromPlayerToEnemySide()                    │
  │ ExecuteEnemyTurn()                               │
  │   ├── Each monster clears block                  │
  │   ├── Each monster executes current move         │
  │   └── Each monster rolls next move (RollMove)    │
  │ EndEnemyTurn()                                   │
  │   └── Powers tick down (Vulnerable, Weak, Frail) │
  └──────────────────────────────────────────────────┘

  SwitchSides() → RoundNumber++
}
```

### 2.3 Pile management

```text
5 piles:
  Hand          — hand
  DrawPile      — draw pile (shuffled)
  DiscardPile   — discard pile
  ExhaustPile   — exhaust pile
  PlayPile      — cards being played (temporary)

Draw flow:
  When DrawPile empty → shuffle DiscardPile into DrawPile → continue drawing

After playing a card:
  Default → DiscardPile
  Has Exhaust keyword → ExhaustPile
  Status / Curse cards → depends on card definition
```

---

## 3. Damage Calculation

### 3.1 Attack damage pipeline (`Hook.ModifyDamageInternal`)

```python
def calculate_damage(base_damage, dealer, target, card_source, value_props):
    damage = base_damage

    # Step 1: enchantment modifiers (additive → multiplicative)
    if card_source and card_source.enchantment:
        damage += enchantment.additive_modifier
        damage *= enchantment.multiplicative_modifier

    # Step 2: additive modifiers (iterate all hook listeners)
    # e.g. StrengthPower → when attacker has Strength, damage += strength.amount
    # Only applies to "powered attack" (has Move flag and not Unpowered)
    for listener in iterate_hook_listeners():
        damage += listener.modify_damage_additive(dealer, target, value_props)

    # Step 3: multiplicative modifiers (iterate all hook listeners)
    # e.g. VulnerablePower → when target has Vulnerable, damage *= 1.5
    #       WeakPower → when attacker has Weak, damage *= 0.75
    for listener in iterate_hook_listeners():
        damage *= listener.modify_damage_multiplicative(dealer, target, value_props)

    # Step 4: damage cap
    # e.g. IntangiblePower → cap damage at 1
    for listener in iterate_hook_listeners():
        damage = listener.modify_damage_cap(damage, dealer, target)

    # Step 5: floor at 0
    damage = max(0, floor(damage))

    return damage
```

### 3.2 Damage resolution (`CreatureCmd.Damage`)

```python
def apply_damage(creature, damage, value_props):
    # Compute final damage
    final_damage = Hook.ModifyDamage(damage, dealer, creature, card_source, value_props)

    # Subtract block (unless Unblockable)
    if not value_props.has(Unblockable):
        blocked = min(creature.block, final_damage)
        creature.block -= blocked
        final_damage -= blocked

    # Osty pet redirect (STS2 new mechanic)
    final_damage = Hook.ModifyHpLostBeforeOsty(final_damage)

    # Other unblocked damage redirect
    final_damage = Hook.ModifyUnblockedDamageTarget(final_damage)

    # Final HP adjustment
    final_damage = Hook.ModifyHpLostAfterOsty(final_damage)

    # Lose HP
    creature.lose_hp(final_damage)
```

### 3.3 Block calculation (`Hook.ModifyBlock`)

```python
def calculate_block(base_block, creature, card_source, value_props):
    block = base_block

    # Additive modifiers
    # e.g. DexterityPower → block += dexterity.amount
    for listener in iterate_hook_listeners():
        block += listener.modify_block_additive(creature, value_props)

    # Multiplicative modifiers
    # e.g. FrailPower → block *= 0.75
    for listener in iterate_hook_listeners():
        block *= listener.modify_block_multiplicative(creature, value_props)

    block = max(0, floor(block))
    return block
```

### 3.4 ValueProp flags

```text
Move       (0x8)  — from card or monster move
Unpowered  (0x4)  — not affected by Strength / Weak / Vulnerable
Unblockable(0x2)  — bypasses block

IsPoweredAttack() = has Move flag AND does not have Unpowered flag
```

Strength, Weak, and Vulnerable only apply when `IsPoweredAttack()` is true.

---

## 4. Core Power Implementations

### 4.1 Damage / block modifier powers

| Power | Type | Hook method | Modifier | Notes |
| ----- | ---- | ----------- | -------- | ----- |
| **Strength** | Buff | `ModifyDamageAdditive` | `+Amount` | Only on attacker’s powered attacks; negative values allowed |
| **Dexterity** | Buff | `ModifyBlockAdditive` | `+Amount` | Negative values allowed |
| **Vulnerable** | Debuff | `ModifyDamageMultiplicative` | `×1.5` | Target takes more damage; Paper Phrog relic can add +0.25 (i.e. ×1.75) |
| **Weak** | Debuff | `ModifyDamageMultiplicative` | `×0.75` | Attacker deals less damage; can be enhanced by Paper Krane relic |
| **Frail** | Debuff | `ModifyBlockMultiplicative` | `×0.75` | Gain less block |
| **Intangible** | Buff | `ModifyDamageCap` | cap 1 | All damage capped at 1 |

### 4.2 Power lifecycle

```text
Vulnerable, Weak, Frail:
  - Type: Counter (stacking counter)
  - Decay: AfterTurnEnd on CombatSide.Enemy (-1 at end of enemy turn)
  - Effective duration: entire player turn + enemy turn; decays when enemy turn ends

Strength, Dexterity:
  - Type: Counter (stacking counter)
  - Negative values allowed
  - Permanent (no natural decay)

Intangible:
  - Usually decays each turn
```

### 4.3 Power base class structure

```csharp
abstract class PowerModel : AbstractModel
{
    PowerType Type;                    // Buff / Debuff / None
    PowerStackType StackType;          // Counter / Single / None
    bool AllowNegative;                // whether negative stacks are allowed
    int Amount;                        // current stack amount

    // ~100+ overridable hook methods (shared with AbstractModel):
    virtual ModifyDamageAdditive(...)
    virtual ModifyDamageMultiplicative(...)
    virtual ModifyDamageCap(...)
    virtual ModifyBlockAdditive(...)
    virtual ModifyBlockMultiplicative(...)
    virtual AfterSideTurnStart(...)
    virtual AfterTurnEnd(...)
    virtual BeforeAttack(...)
    virtual AfterDamageGiven(...)
    virtual OnDeath(...)
    // ... more
}
```

---

## 5. Monster AI System

### 5.1 Architecture overview

```text
MonsterModel (abstract base)
  ├── GenerateMoveStateMachine() → defines AI behavior
  ├── Properties: MinInitialHp, MaxInitialHp (separate normal / ascension values)
  ├── Damage: XxxDamage => AscensionHelper.GetValueIfAscension(level, asc, normal)
  └── Moves: PerformMove(targets) async delegate

MonsterMoveStateMachine
  └── Dictionary<string, MonsterState>
       ├── MoveState          — actual move (Intent[] and PerformMove delegate)
       ├── RandomBranchState  — weighted random choice
       └── ConditionalBranchState — conditional branch (first match)
```

### 5.2 State machine node types

**MoveState (move node)**:

```python
MoveState(
    name="Thrash",                    # move name
    intents=[SingleAttackIntent(...)], # intent display
    perform_move=lambda targets: ..., # execution logic
    follow_up_state="NextState"       # fixed next state (optional)
)
```

**RandomBranchState (random branch)**:

```python
RandomBranchState(
    branches=[
        (weight=75, state="Attack", repeat_rule=CannotRepeat),
        (weight=25, state="Defend", repeat_rule=CanRepeatForever),
    ]
)
```

Repeat rules:

- `CannotRepeat` — cannot pick the same move twice in a row
- `CanRepeatXTimes(n)` — at most n consecutive picks
- `UseOnlyOnce` — once per combat
- `CanRepeatForever` — no limit
- Cooldown — cooldown in turns

Weights are adjusted dynamically based on `StateLog` (history).

**ConditionalBranchState (conditional branch)**:

```python
ConditionalBranchState(
    branches=[
        (condition=lambda: can_summon(), state="Summon"),
        (condition=lambda: True, state="Attack"),  # fallback
    ]
)
```

### 5.3 Move selection flow (`RollMove`)

```python
def roll_move(state_machine):
    current = state_machine.current_state

    # Walk the state chain until reaching a MoveState
    while True:
        next_state = current.get_next_state()
        if isinstance(next_state, MoveState):
            state_machine.current_move = next_state
            return
        elif isinstance(next_state, RandomBranchState):
            # Weighted random select, respecting repeat rules and cooldowns
            current = weighted_random_select(next_state.branches, state_log)
        elif isinstance(next_state, ConditionalBranchState):
            # First branch with true condition
            current = first_matching_branch(next_state.branches)
```

### 5.4 Monster AI pattern examples

**Fixed cycle (Crusher Boss)**:

```text
Thrash → Enlarging Strike → Bug Sting → Adapt → Guarded Strike → (loop)
Implementation: each MoveState’s follow_up_state points to the next
```

**Alternating (Chomper)**:

```text
Clamp → Screech → Clamp → Screech → ...
Implementation: A.follow_up = B, B.follow_up = A
```

**Random (TwoTailedRat)**:

```text
RandomBranch:
  75% → Summon (condition: can_summon(), UseOnlyOnce)
  25% → Bite (CanRepeatForever)
  50% → Scratch (CannotRepeat)
```

**Conditional**:

```text
ConditionalBranch:
  HP < 50% → healing move
  minions alive → buff move
  default → normal attack
```

### 5.5 Ascension modifiers

```python
# HP uses AscensionLevel.ToughEnemies
min_hp = AscensionHelper.GetValueIfAscension(
    AscensionLevel.ToughEnemies, ascension_value, normal_value
)

# Damage uses AscensionLevel.DeadlyEnemies
damage = AscensionHelper.GetValueIfAscension(
    AscensionLevel.DeadlyEnemies, ascension_value, normal_value
)
```

### 5.6 Intent system (Intents)

```text
SingleAttackIntent(damage)        — single attack
MultiAttackIntent(damage, count)  — multi-hit attack
BlockIntent(block)                — gain block
BuffIntent                        — buff
DebuffIntent                      — debuff
StrategicIntent                   — strategic (summon, special)
EscapeIntent                      — escape
SleepIntent                       — sleep
UnknownIntent                     — unknown intent
```

---

## 6. Relic Trigger System

### 6.1 Architecture

```text
RelicModel : AbstractModel
  ├── DynamicVars (parameterized values, e.g. BlockVar(10m))
  ├── Flash() — trigger visual effect
  ├── Rarity: Starter, Common, Uncommon, Rare, Shop, Event, Ancient, None
  └── Override AbstractModel hook methods for trigger logic
```

### 6.2 Hook trigger timing

| Timing | Hook method | Example relic |
| ------ | ----------- | --------- |
| Combat start | `BeforeCombatStart` | Anchor (gain 10 block) |
| Turn start | `AfterSideTurnStart` | Akabeko (gain 8 Vigor on turn 1) |
| Before card played | `BeforeCardPlayed` | — |
| After card played | `AfterCardPlayed` | — |
| Before attack | `BeforeAttack` | — |
| After damage dealt | `AfterDamageGiven` | — |
| After damage taken | `AfterDamageTaken` | — |
| Damage additive | `ModifyDamageAdditive` | — |
| Damage multiplicative | `ModifyDamageMultiplicative` | Paper Phrog (Vulnerable ×1.75 instead of ×1.5) |
| Block modifier | `ModifyBlock` | — |
| Hand draw modifier | `ModifyHandDraw` | — |
| Block clear decision | `ShouldClearBlock` | — |
| On death | `OnDeath` | — |
| On heal | `ModifyHeal` | — |
| On gold gain | `ModifyGoldGain` | — |

### 6.3 Central event dispatch (`Hook` static class)

```python
# All AbstractModel subclasses (cards, relics, powers, monsters, modifiers)
# can register as hook listeners

def hook_modify_damage(base, dealer, target, card_source, value_props):
    damage = base
    # Iterate all registered hook listeners
    for listener in CombatState.iterate_hook_listeners():
        # listener can be any AbstractModel subclass
        damage += listener.modify_damage_additive(dealer, target, value_props)
    for listener in CombatState.iterate_hook_listeners():
        damage *= listener.modify_damage_multiplicative(dealer, target, value_props)
    return damage
```

Key insight: **cards, relics, powers, and monsters** all interact through the same Hook system. They are `AbstractModel` subclasses sharing the same ~100+ virtual methods.

---

## 7. Card Effect System

### 7.1 Card base class structure

```csharp
abstract class CardModel : AbstractModel
{
    // Constructor
    base(energyCost, CardType, CardRarity, TargetType)

    // Core properties
    CardType Type;        // Attack, Skill, Power, Status, Curse, Quest
    TargetType Target;    // None, Self, AnyEnemy, AllEnemies, RandomEnemy, AnyAlly, AllAllies...
    CardRarity Rarity;    // Basic, Common, Uncommon, Rare, Ancient, Event, Token, Status, Curse, Quest
    int EnergyCost;
    int? StarCost;        // STS2 star cost
    DynamicVars Vars;     // DamageVar, BlockVar, CardsVar, etc.

    // Core methods
    abstract OnPlay(PlayerChoiceContext ctx, CardPlay cardPlay);
    virtual OnUpgrade();
    virtual CanPlay();
    virtual IsValidTarget(target);
}
```

### 7.2 Effect implementation patterns

**Attack cards**:

```csharp
// Strike-style
OnPlay(ctx, cardPlay) {
    DamageCmd.Attack(dynamicVars.Damage)
        .FromCard(this)
        .Targeting(target)
        .WithHitFx(...)
        .Execute(ctx);
}
```

**Block cards**:

```csharp
// Defend-style
OnPlay(ctx, cardPlay) {
    CreatureCmd.GainBlock(owner.Creature, dynamicVars.Block, cardPlay);
}
```

**Power application cards**:

```csharp
// Apply Strength
OnPlay(ctx, cardPlay) {
    PowerCmd.Apply<StrengthPower>(creature, amount, applier, cardSource);
}
```

**Draw**:

```csharp
CardPileCmd.Draw(ctx, count, owner);
```

**Generate status cards**:

```csharp
// Add Dazed to enemy discard pile
CardPileCmd.AddToCombatAndPreview<Dazed>(targets, PileType.DiscardPile, count);
```

**X-cost cards**:

```csharp
int x = ResolveEnergyXValue();  // spend all energy
// use x as repeat count or effect multiplier
for (int i = 0; i < x; i++) { ... }
```

### 7.3 Full play flow (`PlayCardAction`)

```python
def play_card(card, target, ctx):
    # 1. Validate
    assert card.can_play()
    assert card.is_valid_target(target)

    # 2. Spend resources
    card.spend_resources()  # energy + stars

    # 3. Execute
    Hook.before_card_played(card, target)
    card.on_play(ctx, card_play)
    if card.enchantment:
        card.enchantment.on_play(ctx, card_play)
    Hook.after_card_played(card, target)

    # 4. Extra plays (e.g. Double Tap)
    play_count = Hook.modify_card_play_count(card)
    for i in range(1, play_count):
        card.on_play(ctx, card_play)  # extra execution

    # 5. Card destination
    if card.has_keyword(Exhaust):
        move_to(ExhaustPile)
    else:
        move_to(DiscardPile)  # default
```

### 7.4 Upgrade system

```python
def upgrade_card(card):
    # Common upgrade patterns:
    card.damage_var.upgrade_value_by(3)      # damage +3
    card.block_var.upgrade_value_by(3)        # block +3
    card.energy_cost.upgrade_by(-1)           # cost -1
    card.add_keyword(CardKeyword.Innate)      # add keyword
    card.remove_keyword(CardKeyword.Ethereal) # remove keyword
```

---

## 8. Map Generation

### 8.1 Map structure (`StandardActMap`)

```python
# 7 columns × mapLength rows
map_grid = MapPoint[7][map_length]   # map_grid[column][row]

# Special nodes
boss_point = BossMapPoint           # after final row
second_boss_point = optional        # second boss in some cases
```

### 8.2 Path generation algorithm

```python
def generate_paths(map_grid, num_paths=7):
    for i in range(num_paths):
        # Start at random column on row 1
        col = random_column(0..6)
        row = 1

        while row < map_length:
            # Create node
            map_grid[col][row] = MapPoint()

            # Connect to parent on previous row
            connect_to_parent(col, row)

            # Advance one row, random column offset (left / center / right)
            direction = random_choice([-1, 0, +1])
            new_col = clamp(col + direction, 0, 6)

            # Prevent path crossing
            if would_cross_another_path(col, new_col, row):
                new_col = col  # go straight

            col = new_col
            row += 1

        # Connect last row to Boss node
        connect_to_boss(col)
```

### 8.3 Room type assignment

**Fixed placement**:

```text
Row 1              → Monster (first encounter)
7th row from end   → Treasure (chest) or Elite (replaced in ascension)
Last row before Boss → RestSite (campfire)
```

**Random pool** (Gaussian distribution):

```text
Elite:    ~5 (×1.6 on ascension)
Shop:     3
Unknown:  ~12 (?) — can trigger event / combat / shop
RestSite: ~5
```

**Placement rules**:

- Same special room type cannot be adjacent (parent/child or siblings)
- RestSite and Elite cannot appear in first 4 rows
- RestSite cannot appear in last 3 rows

### 8.4 Room type enum

```text
Unassigned, Unknown, Shop, Treasure, RestSite, Monster, Elite, Boss, Ancient
```

### 8.5 Per-act configuration

| Act | Name | Room count | Notes |
| --- | ---- | ---------- | ----- |
| Act 1 | Overgrowth | 15 | Intro encounters |
| Act 2 | Hive | — | Mid-game challenge |
| Act 3 | Glory | — | Hard encounters |
| Act 4 | Underdocks | — | Final challenge |

Each act defines its own: boss pool, encounter pool, event pool, ancient NPCs

---

## 9. Energy and Stars

### 9.1 Energy

```python
# Start of each turn
player.energy = player.max_energy     # default 3
player.energy += Hook.modify_energy() # relic / power modifiers

# Playing cards
player.energy -= card.energy_cost

# X-cost cards
x = player.energy  # spend all remaining energy
player.energy = 0
```

### 9.2 Stars — STS2 new mechanic

```python
# Some cards require both energy and stars
def has_enough_resources(card):
    return (player.energy >= card.energy_cost and
            player.stars >= card.star_cost)
```

---

## 10. Simulator Implementation Recommendations

### 10.1 Priority order

```text
P0 — Implement first (minimal combat prototype):
  ├── Creature (HP, Block, Powers)
  ├── 5 piles (Hand, Draw, Discard, Exhaust, Play)
  ├── Energy system
  ├── Damage formula (Strength, Weak, Vulnerable)
  ├── Block formula (Dexterity, Frail)
  ├── Basic card effects (Strike, Defend style)
  ├── Turn flow (player → enemy → loop)
  └── Monster AI state machine framework

P1 — Core expansion:
  ├── Full Hook event system
  ├── All Ironclad cards (start with one character)
  ├── Act 1 monster AI
  ├── Key relics (Starter + Common)
  ├── Key powers (~20 high-frequency)
  └── Card keywords (Exhaust, Ethereal, Innate, Retain)

P2 — Full combat:
  ├── All powers (260)
  ├── All relic triggers
  ├── Potion system
  ├── Enchantment system (STS2 new)
  ├── Star resource system
  └── X-cost cards

P3 — Map and non-combat:
  ├── Map generation algorithm
  ├── Event decision trees
  ├── Shop system
  ├── Rest site (rest / upgrade / recall)
  ├── Card reward selection
  └── Boss chest relic selection

P4 — Full game:
  ├── All 5 characters
  ├── All 4 acts
  ├── Ascension system
  └── Unlock system
```

### 10.2 Workflow for extracting logic from C\#

```text
1. Locate target file
   e.g. extraction/decompiled/MegaCrit.Sts2.Core.Models.Monsters/Chomper.cs

2. Read C# source and understand logic
   - Constructor → HP, damage values
   - GenerateMoveStateMachine() → AI pattern
   - PerformMove delegate → concrete effects

3. Reimplement in Python
   - No 1:1 port required; behavior-equivalent is enough
   - Use spire-codex JSON for static values
   - Implement dynamic logic yourself (AI decisions, effect triggers)

4. Validate with unit tests
   - Compare to real game behavior
   - Verify damage formula and power interactions
```

### 10.3 Integrating with spire-codex data

```python
# spire-codex data/*.json provides static data
cards_data = load_json("cards.json")        # 576 cards’ values
monsters_data = load_json("monsters.json")  # 111 monsters’ HP/damage
powers_data = load_json("powers.json")      # 260 powers’ metadata
encounters_data = load_json("encounters.json")  # 87 encounter monster groups
characters_data = load_json("characters.json")  # 5 characters’ starting state

# Dynamic logic to extract from C# source yourself
# → Card OnPlay() effects
# → Monster GenerateMoveStateMachine() AI
# → Relic Hook overrides
# → Power Hook overrides
# → Map generation StandardActMap algorithm
```

### 10.4 Key pitfalls and notes

1. **Hook execution order**: Order of multiple listeners can affect outcomes (additive before multiplicative is hard-coded, but iteration order among listeners of the same kind must be confirmed).

2. **Vulnerable / Weak / Frail decay timing**: They decay at **end of enemy turn**, not end of player turn. One stack of Vulnerable applied by the player lasts through the full player turn and enemy turn before ticking down.

3. **Block not cleared on turn 1**: Player block is not cleared at the start of turn 1 (e.g. Anchor’s 10 block at combat start persists through turn 1).

4. **Powered vs Unpowered attacks**: Strength / Weak / Vulnerable only affect "powered attacks". Some damage sources (e.g. Thorns, Poison) are marked Unpowered and skip these modifiers.

5. **Monster AI randomness**: RandomBranchState weights and repeat constraints must match the game exactly or behavior diverges.

6. **Async execution model**: The game uses C# async/await; the simulator can use synchronous execution, but some effects depend on ordering (e.g. triggers between multi-hit attacks).
