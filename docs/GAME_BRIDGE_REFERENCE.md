# STS2 Real-Game Bridge — Technical Design

## Overview

Connecting a trained RL agent to the real STS2 game requires two components:

1. **C# Mod** (game side) — expose game state via Harmony hooks, receive action commands
2. **Python client** (agent side) — connect to the mod, run model inference

## Architecture

```text
STS2 game process (Godot + .NET 8)          Python agent process
┌─────────────────────────────┐   TCP   ┌──────────────────┐
│  STS2BridgeMod (.dll)       │◄───────►│  sts2_client.py  │
│                             │  JSON   │                  │
│  ┌───────────────────────┐  │         │  ┌────────────┐  │
│  │ TcpListener (port 9002)│ │ ─state─►│  │ Parse state│  │
│  │ background thread     │  │         │  │ → obs vec  │  │
│  └───────────────────────┘  │         │  └────────────┘  │
│                             │         │        ↓         │
│  ┌───────────────────────┐  │         │  ┌────────────┐  │
│  │ Harmony Hooks:        │  │         │  │ MaskablePPO│  │
│  │  CombatManager        │  │         │  │ inference  │  │
│  │  PlayCardAction       │  │         │  └────────────┘  │
│  │  ActionQueueSet       │  │         │        ↓         │
│  └───────────────────────┘  │         │  ┌────────────┐  │
│                             │ ◄action─│  │ Encode act │  │
│  ┌───────────────────────┐  │         │  │ → JSON cmd │  │
│  │ Superfast Patches:    │  │         │  └────────────┘  │
│  │  3–10x animation      │  │         │                  │
│  │  skip waits           │  │         │                  │
│  └───────────────────────┘  │         │                  │
└─────────────────────────────┘         └──────────────────┘
```

## Part 1: C# Mod

### Project setup

See [lamali292/sts2_example_mod](https://github.com/lamali292/sts2_example_mod):

```xml
<!-- STS2BridgeMod.csproj -->
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>net9.0</TargetFramework>
    <PlatformTarget>x64</PlatformTarget>
  </PropertyGroup>
  <ItemGroup>
    <!-- Reference from game data_sts2_windows_x86_64/ directory -->
    <Reference Include="0Harmony">
      <HintPath>$(STS2GamePath)/data_sts2_windows_x86_64/0Harmony.dll</HintPath>
    </Reference>
    <Reference Include="GodotSharp">
      <HintPath>$(STS2GamePath)/data_sts2_windows_x86_64/GodotSharp.dll</HintPath>
    </Reference>
    <Reference Include="sts2">
      <HintPath>$(STS2GamePath)/data_sts2_windows_x86_64/sts2.dll</HintPath>
    </Reference>
  </ItemGroup>
</Project>
```

### Mod entry point

```csharp
using HarmonyLib;
using MegaCrit.Sts2.Core.Modding;

[ModInitializer("Initialize")]
public class BridgeMod
{
    public static void Initialize()
    {
        var harmony = new Harmony("sts2.bridge.rl");
        harmony.PatchAll();

        // Start TCP server
        BridgeServer.Instance.Start(port: 9002);
        Logger.Log("STS2 Bridge Mod initialized on port 9002");
    }
}
```

### TCP server

```csharp
// TcpListener runs on a background thread
public class BridgeServer
{
    private TcpListener _listener;
    private TcpClient _client;
    private bool _running;

    public void Start(int port)
    {
        _listener = new TcpListener(IPAddress.Loopback, port);
        _listener.Start();
        _running = true;
        Task.Run(AcceptLoop);
    }

    private async Task AcceptLoop()
    {
        while (_running)
        {
            _client = await _listener.AcceptTcpClientAsync();
            await HandleClient(_client);
        }
    }
}
```

### State serialization (when ActionQueue is idle)

Important: send state only when the game is waiting for player input.

```csharp
// Key hook points:
// 1. CombatManager — turn start/end
// 2. ActionQueueSet — read state when queue is idle
// 3. PlayCardAction — inject play-card actions

// State JSON format (see STS1 CommunicationMod):
{
    "type": "game_state",
    "phase": "COMBAT_PLAY",  // COMBAT_PLAY, COMBAT_END_TURN, MAP, EVENT, SHOP, REST, CARD_REWARD, ...
    "combat_state": {
        "player": {
            "hp": 70, "max_hp": 80, "block": 5,
            "energy": 3, "max_energy": 3,
            "powers": [{"id": "STRENGTH", "amount": 2}]
        },
        "hand": [
            {"id": "STRIKE_IRONCLAD", "cost": 1, "type": "Attack", "target": "AnyEnemy", "upgraded": false}
        ],
        "draw_pile_count": 5,
        "discard_pile_count": 2,
        "exhaust_pile_count": 0,
        "enemies": [
            {"id": "NIBBIT", "hp": 35, "max_hp": 44, "block": 0,
             "intent": "ATTACK", "intent_damage": 12, "intent_hits": 1,
             "powers": []}
        ],
        "round": 2
    },
    "available_actions": ["PLAY", "END_TURN", "POTION"],
    "run_state": {
        "floor": 5, "act": 1, "gold": 120,
        "deck": [...], "relics": [...], "potions": [...]
    }
}
```

### Action injection

```csharp
// After receiving an action over TCP, use CallDeferred to run on the main thread
public void HandleAction(string actionJson)
{
    var action = JsonConvert.DeserializeObject<BridgeAction>(actionJson);

    // Game actions must run on the main thread
    Godot.Callable.From(() => {
        switch (action.Type)
        {
            case "PLAY":
                // Inject play card: find hand card, find target, create PlayCardAction
                InjectPlayCard(action.CardIndex, action.TargetIndex);
                break;
            case "END_TURN":
                // Inject end turn
                InjectEndTurn();
                break;
            case "CHOOSE":
                // Out-of-combat choices (card rewards, event options, map nodes, etc.)
                InjectChoice(action.ChoiceIndex);
                break;
            case "POTION":
                InjectPotionUse(action.PotionSlot, action.TargetIndex);
                break;
        }
    }).CallDeferred();
}
```

### Superfast speed-up (STS2_Superfast_Mod approach)

```csharp
// Hook Cmd.CustomScaledWait to reduce all wait times
[HarmonyPatch]
class SpeedPatch
{
    [HarmonyPatch(typeof(Cmd), "CustomScaledWait")]
    [HarmonyPrefix]
    static void Prefix(ref float fastSeconds, ref float standardSeconds)
    {
        fastSeconds *= 0.1f;    // 10x speed-up
        standardSeconds *= 0.1f;
    }
}

// Hook Spine animation speed
[HarmonyPatch]
class AnimSpeedPatch
{
    [HarmonyPatch(typeof(MegaAnimationState), "SetTimeScale")]
    [HarmonyPrefix]
    static void Prefix(ref float timeScale)
    {
        timeScale *= 5.0f;  // 5x animation speed
    }
}
```

### Mod installation

```text
Slay the Spire 2/
└── mods/
    └── STS2BridgeMod/
        ├── STS2BridgeMod.dll      # built mod
        ├── STS2BridgeMod.pck      # Godot resource pack (minimal)
        └── mod_manifest.json      # {"pck_name":"STS2BridgeMod","name":"STS2 Bridge"}
```

## Part 2: Python client

### Connection and communication

```python
# sts2_client.py
import socket
import json

class STS2GameClient:
    def __init__(self, host="127.0.0.1", port=9002):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))
        self.buffer = b""

    def receive_state(self) -> dict:
        """Receive game state JSON."""
        while b"\n" not in self.buffer:
            self.buffer += self.sock.recv(4096)
        line, self.buffer = self.buffer.split(b"\n", 1)
        return json.loads(line)

    def send_action(self, action: dict):
        """Send action command."""
        self.sock.sendall(json.dumps(action).encode() + b"\n")

    def play_card(self, card_index: int, target_index: int = -1):
        self.send_action({"type": "PLAY", "card_index": card_index, "target_index": target_index})

    def end_turn(self):
        self.send_action({"type": "END_TURN"})

    def choose(self, choice_index: int):
        self.send_action({"type": "CHOOSE", "choice_index": choice_index})

    def use_potion(self, slot: int, target_index: int = -1):
        self.send_action({"type": "POTION", "slot": slot, "target_index": target_index})
```

### Agent run loop

```python
# run_agent.py
from sb3_contrib import MaskablePPO
from sts2_client import STS2GameClient
from sts2_env.gym_env.observation import encode_observation
from sts2_env.gym_env.action_space import decode_action, compute_action_mask

def run_agent(model_path: str, host: str = "127.0.0.1", port: int = 9002):
    model = MaskablePPO.load(model_path)
    client = STS2GameClient(host, port)

    while True:
        state = client.receive_state()

        if state["phase"] == "COMBAT_PLAY":
            # In combat: use trained model
            obs = encode_observation(state["combat_state"])
            mask = compute_action_mask(state["combat_state"])
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            decoded = decode_action(int(action), state["combat_state"])

            if decoded["type"] == "end_turn":
                client.end_turn()
            else:
                client.play_card(decoded["card_index"], decoded.get("target_index", -1))

        elif state["phase"] == "CARD_REWARD":
            # Card reward: simple heuristic for now
            client.choose(pick_best_card(state))

        elif state["phase"] == "MAP":
            # Map routing: simple heuristic for now
            client.choose(pick_map_node(state))

        elif state["phase"] in ("EVENT", "SHOP", "REST"):
            # Other phases: simple heuristic for now
            client.choose(0)  # default to first option
```

## Part 3: Implementation steps

### Phase 1: Minimal viable (combat only)

1. Create C# mod project (see sts2_example_mod)
2. Implement TCP server
3. Hook CombatManager, serialize combat state
4. Implement play card / end turn action injection
5. Python client + trained combat model
6. **Validate**: agent can play one combat in the real game

### Phase 2: Full run

1. Extend state serialization (map / events / shop / rest / card rewards)
2. Extend action injection (map choice / purchases / rest options)
3. Train full-run RunEnv model
4. **Validate**: agent can complete an entire run

### Phase 3: High performance

1. Integrate Superfast speed-up (3–10x)
2. Test `--headless` mode
3. Batch runs for data collection
4. Fine-tune model on real-game data

## Key reference projects

| Project | Description | Link |
| ------- | ----------- | ---- |
| sts2_example_mod | STS2 mod project template | [GitHub](https://github.com/lamali292/sts2_example_mod) |
| STS2_Superfast_Mod | Speed mod (Harmony patches) | [GitHub](https://github.com/jidon333/STS2_Superfast_Mod) |
| QuickRestart | State manipulation reference | [GitHub](https://github.com/freude916/sts2-quickRestart) |
| CommunicationMod (STS1) | Protocol design reference | [GitHub](https://github.com/ForgottenArbiter/CommunicationMod) |
| spirecomm (STS1) | Python client reference | [GitHub](https://github.com/ForgottenArbiter/spirecomm) |
| TelnetTheSpire (STS1) | TCP approach reference | [GitHub](https://github.com/cdaymand/TelnetTheSpire) |
| UndoAndRedo | State snapshot reference | [NexusMods](https://www.nexusmods.com/slaythespire2/mods/16) |
| BaseLib-StS2 | Mod base library | [GitHub](https://github.com/Alchyr/BaseLib-StS2) |

## Notes

1. **Thread safety**: Read game state and inject actions on the Godot main thread; TCP I/O on a background thread
2. **Stability**: Read state and inject actions only when ActionQueue is idle
3. **Separate modded saves**: Modded runs use separate save data; normal saves are unaffected
4. **Mac ARM64**: Stock `0Harmony.dll` has bugs; use a patched build
5. **Game updates**: STS2 is in Early Access; updates may require re-adapting Harmony patches
