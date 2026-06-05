// BridgeConfig.cs -- Runtime configuration for the STS2 RL bridge mod.

using System;

namespace STS2BridgeMod;

public static class BridgeConfig
{
    private const string CharacterEnvVar = "STS2_BRIDGE_CHARACTER";
    private const string DefaultCharacterId = "Ironclad";

    public static readonly string[] SupportedCharacterIds =
    {
        "Ironclad",
        "Silent",
        "Defect",
        "Regent",
        "Necrobinder",
    };

    public static string PreferredCharacterId { get; private set; } = DefaultCharacterId;

    public static void Initialize()
    {
        string? raw = Environment.GetEnvironmentVariable(CharacterEnvVar);
        if (string.IsNullOrWhiteSpace(raw))
        {
            PreferredCharacterId = DefaultCharacterId;
            Logger.Log($"BridgeConfig: character={PreferredCharacterId} (default)");
            return;
        }

        foreach (string supported in SupportedCharacterIds)
        {
            if (supported.Equals(raw.Trim(), StringComparison.OrdinalIgnoreCase))
            {
                PreferredCharacterId = supported;
                Logger.Log(
                    $"BridgeConfig: character={PreferredCharacterId} (from {CharacterEnvVar})");
                return;
            }
        }

        string valid = string.Join(", ", SupportedCharacterIds);
        Logger.Log(
            $"BridgeConfig: unknown {CharacterEnvVar}='{raw}'; "
            + $"using default {DefaultCharacterId}. Valid: {valid}");
        PreferredCharacterId = DefaultCharacterId;
    }
}
