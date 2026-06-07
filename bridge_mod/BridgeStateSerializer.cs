// BridgeStateSerializer.cs -- Shared helpers for RL bridge combat JSON fields.

using System;
using System.Collections.Generic;
using System.Linq;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Orbs;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.Relics;
using MegaCrit.Sts2.Core.Models;

namespace STS2BridgeMod;

internal static class BridgeStateSerializer
{
    private const int MaxSerializedOrbSlots = 3;

    public static string NormalizeCharacterId(string? rawId)
    {
        if (string.IsNullOrWhiteSpace(rawId))
            return BridgeConfig.PreferredCharacterId;

        foreach (string supported in BridgeConfig.SupportedCharacterIds)
        {
            if (supported.Equals(rawId.Trim(), StringComparison.OrdinalIgnoreCase))
                return supported;
        }

        return rawId.Trim();
    }

    public static string NormalizeOrbType(string idEntry)
    {
        if (string.IsNullOrWhiteSpace(idEntry))
            return "UNKNOWN";

        string normalized = idEntry.Trim();
        if (normalized.EndsWith("Orb", StringComparison.OrdinalIgnoreCase))
            normalized = normalized[..^3];

        return normalized.ToUpperInvariant();
    }

    public static Dictionary<string, object> SerializeOrbQueue(OrbQueue? queue)
    {
        var result = new Dictionary<string, object>
        {
            ["capacity"] = queue?.Capacity ?? 0,
            ["count"] = queue?.Orbs.Count ?? 0,
            ["orbs"] = new List<Dictionary<string, object>>(),
        };

        if (queue == null)
            return result;

        var orbs = (List<Dictionary<string, object>>)result["orbs"];
        foreach (OrbModel orb in queue.Orbs.Take(MaxSerializedOrbSlots))
        {
            orbs.Add(new Dictionary<string, object>
            {
                ["type"] = NormalizeOrbType(orb.Id.Entry),
                ["evoke_value"] = (int)Math.Max(0m, orb.EvokeVal),
            });
        }

        return result;
    }

    public static Dictionary<string, object> SerializeOsty(Creature? osty)
    {
        if (osty == null || !osty.IsAlive)
        {
            return new Dictionary<string, object>
            {
                ["alive"] = false,
                ["hp"] = 0,
                ["max_hp"] = 0,
                ["block"] = 0,
            };
        }

        return new Dictionary<string, object>
        {
            ["alive"] = true,
            ["hp"] = osty.CurrentHp,
            ["max_hp"] = osty.MaxHp,
            ["block"] = osty.Block,
        };
    }

    public static void AddCharacterMechanics(
        Dictionary<string, object> playerObj,
        Player player,
        PlayerCombatState? pcs)
    {
        playerObj["character_id"] = NormalizeCharacterId(player.Character.Id.Entry);
        playerObj["stars"] = pcs?.Stars ?? 0;
        playerObj["orb_queue"] = SerializeOrbQueue(pcs?.OrbQueue);
        playerObj["osty"] = SerializeOsty(player.Osty);
    }

    public static List<Dictionary<string, object>> SerializeRelics(Player player)
    {
        var relics = new List<Dictionary<string, object>>();
        foreach (RelicModel relic in player.Relics)
        {
            relics.Add(new Dictionary<string, object>
            {
                ["id"] = relic.Id.Entry,
                ["rarity"] = relic.Rarity.ToString().ToUpperInvariant(),
                ["enabled"] = true,
                ["used_up"] = relic.IsUsedUp,
                ["counter"] = relic.DisplayAmount,
            });
        }
        return relics;
    }
}
