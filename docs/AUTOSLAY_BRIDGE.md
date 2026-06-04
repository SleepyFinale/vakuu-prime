# AutoSlay Bridge — RL Agent Integration via Built-in Automation

## Discovery

STS2 includes a built-in `AutoSlay` system (`MegaCrit.Sts2.Core.AutoSlay`), a full automation framework
that can play an entire run automatically. It is gated off in release builds by `IsReleaseGame()`.

## New approach vs old approach

| Topic | Old approach (TCP + JSON) | New approach (AutoSlay hook) |
| ----- | ------------------------- | ---------------------------- |
| State serialization | Hand-written JSON serialization | Direct access to game objects |
| Action injection | Reflection + CallDeferred | Direct calls to CardCmd / PlayerCmd |
| UI handling | Manual handling of every screen | AutoSlay handles all screens |
| Error recovery | Hand-written | AutoSlay Watchdog |
| Code size | ~2500 lines C# | ~200 lines C# |

## Implementation steps

### Step 1: Patch IsReleaseGame

```csharp
[HarmonyPatch(typeof(NGame), nameof(NGame.IsReleaseGame))]
static class UnlockAutoSlay
{
    static bool Prefix(ref bool __result)
    {
        __result = false;
        return false;
    }
}
```

### Step 2: Replace CombatRoomHandler

Replace random card plays with RL agent decisions:

```csharp
public class RlCombatHandler : IRoomHandler
{
    // Receive CombatState, send to Python agent over TCP
    // Python returns card_index + target_index
    // Call CardCmd.AutoPlay(ctx, card, target)
}
```

### Step 3: Implement ICardSelector

Replace random card picks with RL agent decisions:

```csharp
public class RlCardSelector : ICardSelector
{
    // All card selection (rewards, upgrades, transforms, etc.) sent to Python over TCP
    Task<IEnumerable<CardModel>> GetSelectedCards(...)
    CardModel? GetSelectedCardReward(...)
}
```

### Step 4: Replace MapScreenHandler

Replace fixed path selection with RL agent decisions.

## Key APIs

| API | Purpose |
| --- | ------- |
| `CardCmd.AutoPlay(ctx, card, target)` | Play card (bypass UI) |
| `PlayerCmd.EndTurn(player, false)` | End turn |
| `potionModel.EnqueueManualUse(target)` | Use potion |
| `CardSelectCmd.UseSelector(selector)` | Inject card selection logic |
| `UiHelper.Click(button)` | Click UI button |
| `WaitHelper.Until(condition, ct)` | Wait until condition is met |
| `RunManager.Instance.DebugOnlyGetState()` | Get RunState |
| `CombatManager.Instance` | Get combat state |
| `NOverlayStack.Instance.Peek()` | Current screen |

## Thread safety

All operations must run on the Godot main thread. AutoSlay runs async tasks via `TaskHelper.RunSafely()`.

## Startup

After the mod loads, start AutoSlayer from the main menu automatically and replace its decision handlers with the RL agent.
Alternatively, trigger startup via a TCP signal.
