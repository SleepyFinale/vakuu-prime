# Slay the Spire 2 RL Agent — Research Notes

## 1. STS2 Technical Architecture


| Item                 | Details                                                                      |
| -------------------- | ---------------------------------------------------------------------------- |
| Game engine          | Godot 4 (migrated from Unity due to Unity’s 2023 per-install pricing policy) |
| Programming language | C# / .NET 8                                                                  |
| Core file            | `sts2.dll` (all game logic)                                                  |
| Code obfuscation     | None — class names, method names, and game logic are fully readable          |
| Asset format         | `.pck` files (Godot standard pack format)                                    |
| Monster animations   | Spine skeletal animation (`.skel` + `.atlas` + `.png`)                       |


Comparison with STS1:


| Aspect               | STS1                                 | STS2                                |
| -------------------- | ------------------------------------ | ----------------------------------- |
| Engine               | libGDX                               | Godot 4                             |
| Language             | Java                                 | C# / .NET 8                         |
| Game files           | `.jar` (ZIP containing `.class`)     | `.pck` + `sts2.dll`                 |
| Decompilation tools  | CFR / IntelliJ / JD-GUI              | ILSpy + GDRE Tools                  |
| Mod loading          | ModTheSpire + BaseMod                | Native `mods` folder + BaseLib-StS2 |
| Automation interface | CommunicationMod (stdin/stdout JSON) | None yet                            |
| Bot frameworks       | spirecomm (Python), bottled_ai       | None yet                            |


## 2. Decompilation Methods

### 2.1 Extracting assets (.pck files)

Tool: [GDRE Tools (gdsdecomp)](https://github.com/GDRETools/gdsdecomp)

```bash
gdre_tools --headless --recover=<path-to-pck>
```

- Extracts ~9,947 files: images, Spine skeletal animations, localization data, Godot scenes/resources
- Alternative tools: [Godot PCK Explorer](https://dmitriysalnikov.itch.io/godot-pck-explorer), [godotdec](https://github.com/Bioruebe/godotdec)

### 2.2 Decompiling the C# DLL (game logic)

Tool: [ILSpy](https://github.com/icsharpcode/ILSpy) (recommended) or its CLI version `ilspycmd`

```bash
ilspycmd -p -o <output-dir> sts2.dll
```

- Produces ~3,300 readable C# source files
- Key namespaces: `MegaCrit.Sts2.Core.Models.Powers/`, card definitions, relic definitions, monster AI, etc.
- Alternative tools: dnSpy, dotPeek (JetBrains)

## 3. Existing Modding Ecosystem

### 3.1 Mod installation

- Create a `mods` folder in the game directory and place `.dll` + `.pck` files inside
- The game shows "Running Modded" and uses separate save data
- Steam Workshop support is confirmed / planned

### 3.2 Key modding tools


| Tool                                                   | Description                                                  |
| ------------------------------------------------------ | ------------------------------------------------------------ |
| [BaseLib-StS2](https://github.com/Alchyr/BaseLib-StS2) | Mod base library (similar to STS1’s BaseMod)                 |
| [Harmony](https://github.com/pardeike/Harmony)         | C# runtime method hooking library (note: Mac ARM64 has bugs) |
| GUMM                                                   | Alternative mod loading approach                             |
| R2Modman / Thunderstore                                | Community mod managers                                       |


### 3.3 Useful existing projects


| Project                                                        | Description                                                                                               |
| -------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| [spire-codex](https://github.com/ptrlrd/spire-codex)           | Extracts structured data from decompiled code (cards/relics/monsters/potions); includes 20 Python parsers |
| [BetterSpire2](https://www.nexusmods.com/slaythespire2/mods/2) | QoL mod (damage counter, auto-confirm, fast mode, etc.)                                                   |
| DevConsole mod                                                 | Enables the built-in developer console (`card`, `block`, `act`, `afflict`, etc.)                          |
| [Nexus Mods (STS2)](https://www.nexusmods.com/slaythespire2)   | Mod hosting platform                                                                                      |


## 4. RL Environment Setup

### 4.1 Route A: Mod bridge (prototype / validation)

Build a C# mod modeled on STS1’s [CommunicationMod](https://github.com/ForgottenArbiter/CommunicationMod):

1. Use **Harmony** to hook game state update methods (key classes in `sts2.dll`)
2. Serialize game state to JSON
3. Expose state to an external Python process via TCP socket / named pipe
4. Run the RL agent in Python and send action commands

**Pros**: Interacts with the real game directly; state is accurate.
**Cons**: Speed is limited by render frame rate; cannot support the millions of runs RL training requires.

### 4.2 Route B: Headless simulator (training, recommended)

Reimplement core game logic in Python from decompiled C# source, with a Gymnasium interface.

STS1 precedent projects:


| Project                                                                    | Language | Description                                                            |
| -------------------------------------------------------------------------- | -------- | ---------------------------------------------------------------------- |
| [decapitate-the-spire](https://github.com/jahabrewer/decapitate-the-spire) | Python   | Headless simulator, gym-style `step()` API; focused on Silent/Exordium |
| [conquer-the-spire](https://github.com/utilForever/conquer-the-spire)      | C++      | Cross-platform simulator                                               |
| [MiniStS](https://github.com/iambb5445/MiniSTS)                            | Python   | Simplified implementation; published at AAAI/AIIDE 2024                |
| Miles Oram’s C++ remake                                                    | C++      | Full Ironclad 4-act experience + deep RL                               |


**Key lesson** (from the decapitate-the-spire author): RL training by connecting directly to the game is not viable; you need a headless simulator to reach thousands of games per second.

### 4.3 Recommended approach

Use both routes together:

```text
Route B (Headless simulator) — large-scale RL training
         ↓ after training
Route A (Mod bridge) — validate agent behavior in the real game
```

## 5. RL Algorithm Selection

### 5.1 Recommended: PPO + invalid action masking


| Algorithm   | Fit                     | Notes                                                                                          |
| ----------- | ----------------------- | ---------------------------------------------------------------------------------------------- |
| **PPO**     | **Recommended**         | Policy gradient method; stable; validated in multiple STS projects                             |
| DQN         | Not recommended         | Unstable with large action spaces; poor results reported for UNO / mahjong / dou dizhu         |
| AlphaZero   | Not directly applicable | Designed for perfect-information games; STS has hidden information (draw order, enemy intents) |
| NFSP / CFR  | Not recommended         | High compute cost; better suited to multi-player adversarial games (e.g. poker)                |
| Two-Step RL | Worth exploring         | Splits the problem into deck-building and combat phases                                        |


Key paper: [A Closer Look at Invalid Action Masking in Policy Gradient Algorithms](https://arxiv.org/abs/2006.14171)

### 5.2 Action masking

Legal actions change every turn in STS (different hand, targets, potions, etc.), so action masking is required:

- Define a maximum possible action space
- Each step, `action_masks()` returns a boolean array; zero illegal action probabilities and renormalize
- sb3-contrib provides `MaskablePPO` out of the box

Note: action spaces above ~1400 may have numerical precision issues.

## 6. RL Framework Selection


| Framework                                                                                                                                  | Recommendation               | Notes                                                                 |
| ------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------- | --------------------------------------------------------------------- |
| **[SB3](https://github.com/DLR-RM/stable-baselines3) + [sb3-contrib**](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html) | **Best for getting started** | Built-in MaskablePPO; simple API; PyTorch backend; 95% code coverage  |
| [Ray RLlib](https://docs.ray.io/en/latest/rllib/)                                                                                          | Large-scale training         | Distributed training; native action masking; steeper learning curve   |
| [RLCard](https://rlcard.org/)                                                                                                              | Design reference             | Card-game RL toolkit; standardized state formats                      |
| [OpenSpiel](https://github.com/google-deepmind/open_spiel)                                                                                 | Research reference           | DeepMind; 70+ game environments; supports imperfect-information games |
| [Gymnasium](https://gymnasium.farama.org/)                                                                                                 | Interface standard           | All custom environments should implement the Gymnasium API            |


## 7. Observation Space Design

### 7.1 State components to encode

```text
Player: HP / max HP / block / energy
Hand: count by card type (bag-of-cards encoding)
Draw pile: known card composition (unknown order)
Discard pile: card composition
Exhaust pile: card composition
Enemies: HP / block / intent (attack/buff/debuff) / powers
Player powers: Strength, Dexterity, Vulnerable, Weak, etc.
Relics: binary vector (owned / not owned)
Potions: currently held
Floor / act
Gold
Map state (for path selection)
```

### 7.2 Encoding approaches (reference)


| Approach                 | Description                                                      | Source                      |
| ------------------------ | ---------------------------------------------------------------- | --------------------------- |
| One-hot card vector      | Each card maps to a binary vector (1×234)                        | LearnTheSpire               |
| Bag-of-cards             | Count per card type                                              | Common across ML projects   |
| Binary relic vector      | 0/1 per relic                                                    | Tilburg University research |
| Dimensionality reduction | Compress to abstract concepts (e.g. “ongoing damage capability”) | decapitate-the-spire        |


### 7.3 Known pitfalls

- **Hand index issue**: “Hand position 4” means a different card each turn; the agent must learn the mapping from index to card effect. Prefer encoding card semantics over raw indices.
- **Huge state space**: Choose features carefully to avoid the curse of dimensionality.

## 8. Action Space Design

Fixed-size discrete space covering all possible actions:

```text
In combat:
  - Play card N (target M)   — N ∈ [0, max_hand_size), M ∈ [0, max_enemies)
  - Use potion P (target M)  — P ∈ [0, max_potions)
  - End turn

Out of combat:
  - Pick card reward K       — K ∈ [0, max_card_choices) or skip
  - Map node navigation
  - Shop buy / remove
  - Event option selection
  - Use / discard potion
```

Mask unavailable actions each step with `action_masks()`.

## 9. Reward Shaping


| Signal                    | Description                                      |
| ------------------------- | ------------------------------------------------ |
| Win run                   | Large positive reward                            |
| Death                     | Large negative reward                            |
| Kill monster              | Small positive reward                            |
| Advance floor             | Small positive reward                            |
| Lose HP                   | Small negative reward (encourages flawless runs) |
| Gain relic / upgrade card | Optional small positive reward                   |


**Warning**: Excessive reward shaping can let the agent exploit reward hacks (e.g. dragging out fights for more kill rewards). Balance carefully.

## 10. Key Papers and Resources

### Academic papers


| Paper                                                        | Venue / source       | Link                                                    |
| ------------------------------------------------------------ | -------------------- | ------------------------------------------------------- |
| Language-Driven Play: LLMs as Game-Playing Agents in StS     | FDG 2024             | [ACM](https://dl.acm.org/doi/10.1145/3649921.3650013)   |
| MiniStS: A Testbed for Dynamic Rule Exploration              | AAAI AIIDE/EXAG 2024 | [GitHub](https://github.com/iambb5445/MiniSTS)          |
| Strategic Delegation: A Modular and Hybrid Agent             | agents4science 2025  | [OpenReview](https://openreview.net/pdf?id=gC3D2ESSyK)  |
| LLMs May Not Be Human-Level Players, But They Can Be Testers | arXiv 2024           | [arXiv:2410.02829](https://arxiv.org/html/2410.02829v1) |
| Predicting a Successful Run in StS                           | Tilburg University   | [Paper](http://arno.uvt.nl/show.cgi?fid=169629)         |
| A Closer Look at Invalid Action Masking                      | arXiv 2020           | [arXiv:2006.14171](https://arxiv.org/abs/2006.14171)    |
| Two-Step RL for Multistage Strategy Card Game                | arXiv 2023           | [arXiv:2311.17305](https://arxiv.org/html/2311.17305v1) |
| Playing Non-Embedded Card-Based Games with RL                | arXiv 2025           | [arXiv:2504.04783](https://arxiv.org/html/2504.04783v1) |


### Blogs


| Article                                                 | Link                                                                                                                    |
| ------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Creating an AI for Slay the Spire (PPO/A2C in practice) | [toypiper.com](https://www.toypiper.com/creating-an-ai-for-slay-the-spire/)                                             |
| Tackling UNO with RL                                    | [Towards Data Science](https://towardsdatascience.com/tackling-uno-card-game-with-reinforcement-learning-fad2fc19355c/) |
| Training an AI for Dominion (deck-building RL)          | [ianwdavis.com](https://ianwdavis.com/dominion2.html)                                                                   |


### GitHub projects


| Project                                                                    | Description                          |
| -------------------------------------------------------------------------- | ------------------------------------ |
| [decapitate-the-spire](https://github.com/jahabrewer/decapitate-the-spire) | STS1 Python headless simulator       |
| [conquer-the-spire](https://github.com/utilForever/conquer-the-spire)      | STS1 C++ simulator                   |
| [MiniStS](https://github.com/iambb5445/MiniSTS)                            | Simplified STS Python implementation |
| [spire-codex](https://github.com/ptrlrd/spire-codex)                       | STS2 decompiled data extraction      |
| [CommunicationMod](https://github.com/ForgottenArbiter/CommunicationMod)   | STS1 game–external process bridge    |
| [spirecomm](https://github.com/ForgottenArbiter/spirecomm)                 | STS1 Python communication library    |
| [bottled_ai](https://github.com/xaved88/bottled_ai)                        | STS1 bot (52% Watcher win rate)      |
| [BaseLib-StS2](https://github.com/Alchyr/BaseLib-StS2)                     | STS2 mod base library                |
| [GDRE Tools](https://github.com/GDRETools/gdsdecomp)                       | Godot decompilation tools            |


## 11. spire-codex Deep Dive

[spire-codex](https://github.com/ptrlrd/spire-codex) is a full STS2 decompilation data pipeline + REST API + frontend, live at [spire-codex.com](https://spire-codex.com).

### 11.1 Architecture overview

```text
spire-codex/
├── backend/
│   ├── app/
│   │   ├── main.py                 FastAPI app (CORS, rate limiting, static files)
│   │   ├── models/schemas.py       Pydantic models (19 entity types)
│   │   ├── parsers/                20 Python regex parsers + orchestrator
│   │   ├── routers/                22 FastAPI route modules
│   │   └── services/data_service.py  lru_cache JSON loader
│   ├── static/images/              Card/relic/potion/monster PNGs
│   └── requirements.txt
├── frontend/                       Next.js 16 + TypeScript + Tailwind CSS
├── tools/
│   ├── spine-renderer/             Headless Spine skeletal renderer (Node.js)
│   ├── diff_data.py                Version changelog generator
│   └── update.py                   Full pipeline driver script
├── data/                           Output: 20 parsed JSON files
│   ├── cards.json, monsters.json, relics.json, ...
│   └── changelogs/
└── extraction/                     Not in git; generate locally
    ├── raw/                        GDRE-extracted Godot assets (~9,947 files)
    │   ├── images/                 Card portraits, relics, potions
    │   ├── animations/             Spine .skel + .atlas + .png
    │   └── localization/eng/       JSON localization (names, descriptions, SmartFormat templates)
    └── decompiled/                 ILSpy-decompiled C# (~3,300 .cs files)
        └── MegaCrit.Sts2.Core.Models.{Cards,Characters,Monsters,...}/
```

### 11.2 Extracted game data

`data/` already contains complete structured JSON — **usable without running parsers**:


| Category     | File                | Count   | Value for RL                                                                                               |
| ------------ | ------------------- | ------- | ---------------------------------------------------------------------------------------------------------- |
| Cards        | `cards.json`        | 576     | **Core** — cost/type/rarity/damage/block/hit count/powers applied/keywords/upgrade deltas/character        |
| Monsters     | `monsters.json`     | 111     | **Core** — HP range (normal + ascension) / move list / damage per move (normal + ascension) / block values |
| Encounters   | `encounters.json`   | 87      | **Core** — monster groups / room type / act                                                                |
| Powers       | `powers.json`       | 260     | **Core** — Buff/Debuff type / stack type (Counter/Single/None)                                             |
| Relics       | `relics.json`       | 289     | **Important** — rarity / character pool / description                                                      |
| Characters   | `characters.json`   | 5       | **Important** — starting HP/gold/energy/deck/relics                                                        |
| Potions      | `potions.json`      | 63      | **Important** — rarity / character pool                                                                    |
| Acts         | `acts.json`         | 4       | **Important** — boss pool / encounter pool / event pool / ancient NPCs                                     |
| Events       | `events.json`       | 66      | **Useful** — options / outcomes / full decision trees                                                      |
| Enchantments | `enchantments.json` | 22      | **Useful** — card type restrictions / stackable or not                                                     |
| Keywords     | `keywords.json`     | 8       | Reference                                                                                                  |
| Intents      | `intents.json`      | 14      | Reference                                                                                                  |
| Orbs         | `orbs.json`         | 5       | Reference                                                                                                  |
| Afflictions  | `afflictions.json`  | 9       | Reference                                                                                                  |
| Modifiers    | `modifiers.json`    | 16      | Reference                                                                                                  |
| Achievements | `achievements.json` | 33      | Not relevant (for RL)                                                                                      |
| Ascensions   | `ascensions.json`   | 11      | Parametric difficulty                                                                                      |
| Epochs       | `epochs.json`       | several | Unlock system                                                                                              |
| Stories      | `stories.json`      | several | Unlock system                                                                                              |


Playable characters: **Ironclad, Silent, Defect, Necrobinder, Regent** (5 total).
Acts: **Overgrowth (Act 1), Hive (Act 2), Glory (Act 3), Underdocks**.

### 11.3 JSON data structure examples

**Card** (`cards.json`):

```json
{
  "id": "ABRASIVE",
  "name": "Abrasive",
  "description": "Gain 1 [gold]Dexterity[/gold].\nGain 4 [gold]Thorns[/gold].",
  "description_raw": "Gain {DexterityPower:diff()} ...",
  "cost": 3,
  "type": "Power",
  "rarity": "Rare",
  "target": "Self",
  "color": "silent",
  "damage": null,
  "block": null,
  "hit_count": null,
  "powers_applied": [
    {"power": "Thorns", "amount": 4},
    {"power": "Dexterity", "amount": 1}
  ],
  "keywords": ["Sly"],
  "tags": null,
  "vars": {"ThornsPower": 4, "Thorns": 4, "DexterityPower": 1, "Dexterity": 1},
  "upgrade": {"thornspower": "+2"}
}
```

**Monster** (`monsters.json`):

```json
{
  "id": "ASSASSIN_RUBY_RAIDER",
  "name": "Assassin Raider",
  "type": "Normal",
  "min_hp": 18, "max_hp": 23,
  "min_hp_ascension": 19, "max_hp_ascension": 24,
  "moves": [{"id": "KILLSHOT", "name": "Killshot"}],
  "damage_values": {"Killshot": {"normal": 11, "ascension": 12}},
  "block_values": null
}
```

**Rich text format**: Descriptions use tags like `[gold]text[/gold]`, `[red]`, `[energy:N]`, `[star:N]`. Strip with regex: `\[/?[a-z]+(?::\d+)?\]`

### 11.4 How parsers work

All parsers pull from two sources:

- **Decompiled C#**: `extraction/decompiled/MegaCrit.Sts2.Core.Models.{Category}/`
- **Localization JSON**: `extraction/raw/localization/eng/{category}.json`

**ID generation rule** (shared across entities):

```python
def class_name_to_id(name: str) -> str:
    # PascalCase → UPPER_SNAKE_CASE
    s = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', '_', name)
    s = re.sub(r'(?<=[A-Z])(?=[A-Z][a-z])', '_', s)
    return s.upper()
# FrogKnight → FROG_KNIGHT
```

**Core parsers (detail)**:


| Parser                    | Key regex / logic                                                                                                                                                                  |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `card_parser.py`          | Extract `base(cost, CardType, CardRarity, TargetType)` from constructors; scan `DamageVar`/`BlockVar` for values; `CanonicalKeywords` for keywords; pool files for character color |
| `monster_parser.py`       | HP from `MinInitialHp => AscensionHelper.GetValueIfAscension(level, asc, normal)`; moves from `new MoveState("NAME", ...)`; damage from `(\w+)Damage =>`                           |
| `description_resolver.py` | **Shared core**: `extract_vars_from_source()` pulls numeric vars from C#; `resolve_description()` resolves SmartFormat templates (conditionals, plurals, icons, etc.)              |
| `event_parser.py`         | Most complex — multi-page event trees, StringVar resolution, random ranges, ancient NPC dialogue                                                                                   |


**Parser dependencies**:

```text
pool_parser → must run after potion_parser (reads data/potions.json to fill pool info)
event_parser → reads data/relics.json to enrich relic descriptions
encounter_parser ↔ monster_parser → cross-reference for monster type and act assignment
```

### 11.5 API

FastAPI backend (port 8000), rate limit 60 req/min/IP:

```text
GET /api/{category}          — list (with filters)
GET /api/{category}/{id}     — single entity
GET /api/stats               — counts per entity type
```

Filter parameters:

- Cards: `color`, `type`, `rarity`, `keyword`, `search`
- Relics / potions: `rarity`, `pool`, `search`
- Monsters: `type` (Normal/Elite/Boss), `search`
- Encounters: `room_type`, `act`
- Events: `type`, `act`
- Powers: `type`, `stack_type`

Also available at [spire-codex.com/api/](https://spire-codex.com/api/)

### 11.6 Dependencies and environment

**Backend**: FastAPI + uvicorn + Pydantic (SQLAlchemy is in requirements.txt but unused; database migration planned)
**Frontend**: Next.js 16 + TypeScript + Tailwind CSS
**Extraction tools** (external): GDRE Tools v2.4.0, ILSpy CLI v9.1.0 (`dotnet tool install ilspycmd -g`), Python 3.10+, Node.js 20+

### 11.7 Value for this RL agent project

**Directly reusable**:

1. `**data/*.json`** — 576 cards, 111 monsters, 87 encounters, 260 powers as structured data for the headless simulator’s static data layer
2. **Parser code** — re-run parsers after game updates for fresh data
3. **ID convention** — UPPER_SNAKE_CASE as the simulator’s unified ID system
4. **Description resolver** — SmartFormat logic in `description_resolver.py` for understanding card effects
5. **Character starting data** — `characters.json` has full initial state (HP/gold/energy/deck/relics)

**Still to implement yourself**:

1. **Combat logic** — spire-codex is static data only; no turn flow, damage formulas, or power trigger timing
2. **Monster AI** — `monsters.json` has move names and damage only; decision logic must come from decompiled C#
3. **Relic triggers** — descriptions only; read C# for conditions and effects
4. **Map generation** — understand algorithms from decompiled code
5. **Event effects** — `events.json` has option text but not structured numeric outcomes

### 11.8 C# namespace → parser mapping

Key decompiled directories when building the simulator:


| C# namespace directory                                | Parser           | Simulator use              |
| ----------------------------------------------------- | ---------------- | -------------------------- |
| `MegaCrit.Sts2.Core.Models.Cards/`                    | card_parser      | Card effect implementation |
| `MegaCrit.Sts2.Core.Models.Monsters/`                 | monster_parser   | Monster AI logic           |
| `MegaCrit.Sts2.Core.Models.Powers/`                   | power_parser     | Power logic                |
| `MegaCrit.Sts2.Core.Models.Relics/`                   | relic_parser     | Relic trigger logic        |
| `MegaCrit.Sts2.Core.Models.Encounters/`               | encounter_parser | Encounter composition      |
| `MegaCrit.Sts2.Core.Models.Events/`                   | event_parser     | Event decision trees       |
| `MegaCrit.Sts2.Core.Models.Characters/`               | character_parser | Character starting state   |
| `MegaCrit.Sts2.Core.Models.Potions/`                  | potion_parser    | Potion effects             |
| `MegaCrit.Sts2.Core.Models.{Card,Relic,Potion}Pools/` | pool_parser      | Character → item mapping   |


## 12. Recommended Implementation Roadmap

### Phase 1: Decompilation and understanding

1. Install ILSpy and decompile `sts2.dll`
2. Study game state models (`MegaCrit.Sts2.Core.Models.`*)
3. Follow spire-codex’s data extraction approach
4. Compile full lists of cards, relics, monsters, and powers

### Phase 2: Headless simulator

1. Implement core combat logic in Python (decapitate-the-spire-style architecture)
2. Start with one character and Act 1
3. Implement Gymnasium API (`reset()`, `step()`, `action_masks()`)
4. Write unit tests and validate against real game behavior

### Phase 3: RL training

1. Use SB3 + sb3-contrib MaskablePPO
2. Design observation space (flat vector, segmented encoding)
3. Design action space (fixed size + action masking)
4. Experiment with reward shaping
5. Train on simplified scenarios first (fixed deck, single combat), then expand

### Phase 4: Real-game validation

1. Build C# mod (Harmony hooks + JSON serialization + TCP)
2. Run the trained agent in the real game
3. Compare simulator vs. real game behavior and iterate

