"""Build combat encounter pools across acts for RL training."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

from sts2_env.encounters.registry import EncounterSetup
from sts2_env.map.acts import BiomeId

# Act indices used by RunState (0 = Act 1, 1 = Act 2, 2 = Act 3).
SUPPORTED_TRAINING_ACTS = (0, 1, 2)


def encounter_lists_for_act(
    act_index: int,
    biome_id: BiomeId | None = None,
) -> tuple[
    list[EncounterSetup],
    list[EncounterSetup],
    list[EncounterSetup],
    list[EncounterSetup],
]:
    """Return (weak, normal, elite, boss) encounter setups for a run act index."""
    if act_index == 0:
        if biome_id == "underdocks":
            from sts2_env.encounters import act4

            return (
                list(act4.WEAK_ENCOUNTERS),
                list(act4.NORMAL_ENCOUNTERS),
                list(act4.ELITE_ENCOUNTERS),
                list(act4.BOSS_ENCOUNTERS),
            )
        from sts2_env.encounters import act1

        return (
            list(act1.WEAK_ENCOUNTERS),
            list(act1.NORMAL_ENCOUNTERS),
            list(act1.ELITE_ENCOUNTERS),
            list(act1.BOSS_ENCOUNTERS),
        )
    if act_index == 1:
        from sts2_env.encounters import act2

        return (
            list(act2.WEAK_ENCOUNTERS),
            list(act2.NORMAL_ENCOUNTERS),
            list(act2.ELITE_ENCOUNTERS),
            list(act2.BOSS_ENCOUNTERS),
        )
    if act_index == 2:
        from sts2_env.encounters import act3

        return (
            list(act3.WEAK_ENCOUNTERS),
            list(act3.NORMAL_ENCOUNTERS),
            list(act3.ELITE_ENCOUNTERS),
            list(act3.BOSS_ENCOUNTERS),
        )
    raise ValueError(f"Unsupported act index: {act_index}")


def _act_encounter_lists(act_index: int) -> tuple[
    list[EncounterSetup],
    list[EncounterSetup],
    list[EncounterSetup],
    list[EncounterSetup],
]:
    """Legacy: act 0 defaults to Overgrowth only."""
    return encounter_lists_for_act(act_index, biome_id="overgrowth")


def parse_act_indices(acts_spec: str) -> tuple[int, ...]:
    """Parse ``'0'``, ``'0,1,2'``, or ``'all'`` into act indices."""
    normalized = acts_spec.strip().lower()
    if normalized == "all":
        return SUPPORTED_TRAINING_ACTS
    indices: list[int] = []
    for part in normalized.split(","):
        part = part.strip()
        if not part:
            continue
        index = int(part)
        if index == 3:
            warnings.warn(
                "Act index 3 (legacy Underdocks) is deprecated; use act 0 with "
                "--act1-biome underdocks or random.",
                DeprecationWarning,
                stacklevel=2,
            )
            if 0 not in indices:
                indices.append(0)
            continue
        if index not in SUPPORTED_TRAINING_ACTS:
            raise ValueError(
                f"Act index {index} is not in supported training acts "
                f"{SUPPORTED_TRAINING_ACTS}"
            )
        indices.append(index)
    if not indices:
        raise ValueError(f"No act indices parsed from: {acts_spec!r}")
    return tuple(dict.fromkeys(indices))


def build_encounter_pool(
    act_indices: Sequence[int],
    *,
    include_weak: bool = True,
    include_normal: bool = True,
    include_elite: bool = True,
    include_boss: bool = False,
    act1_biome: str = "random",
    act1_biome_rng: "Rng | None" = None,
) -> list[EncounterSetup]:
    """Combine encounter setups from one or more acts."""
    from sts2_env.core.rng import Rng
    from sts2_env.map.acts import _pick_act1_biome

    pool: list[EncounterSetup] = []
    for act_index in act_indices:
        biome_id: BiomeId | None = None
        if act_index == 0:
            if act1_biome in ("overgrowth", "underdocks"):
                biome_id = act1_biome  # type: ignore[assignment]
            elif act1_biome == "random":
                rng = act1_biome_rng or Rng(0)
                biome_id = _pick_act1_biome(
                    rng,
                    underdocks_unlocked=True,
                    underdocks_discovered=True,
                    act1_override=None,
                )
            else:
                raise ValueError(
                    f"act1_biome must be 'random', 'overgrowth', or 'underdocks', got {act1_biome!r}"
                )
        weak, normal, elite, boss = encounter_lists_for_act(act_index, biome_id=biome_id)
        if include_weak:
            pool.extend(weak)
        if include_normal:
            pool.extend(normal)
        if include_elite:
            pool.extend(elite)
        if include_boss:
            pool.extend(boss)
    if not pool:
        raise ValueError(
            f"Empty encounter pool for acts={list(act_indices)} "
            f"(weak={include_weak}, normal={include_normal}, "
            f"elite={include_elite}, boss={include_boss})"
        )
    return pool


def build_mixed_act1_encounter_pool(
    act_indices: Sequence[int],
    *,
    include_weak: bool = True,
    include_normal: bool = True,
    include_elite: bool = True,
    include_boss: bool = False,
) -> list[EncounterSetup]:
    """Pool for act 0 that includes both Overgrowth and Underdocks encounters."""
    pool: list[EncounterSetup] = []
    for act_index in act_indices:
        if act_index == 0:
            for biome in ("overgrowth", "underdocks"):
                weak, normal, elite, boss = encounter_lists_for_act(0, biome_id=biome)
                if include_weak:
                    pool.extend(weak)
                if include_normal:
                    pool.extend(normal)
                if include_elite:
                    pool.extend(elite)
                if include_boss:
                    pool.extend(boss)
        else:
            weak, normal, elite, boss = encounter_lists_for_act(act_index)
            if include_weak:
                pool.extend(weak)
            if include_normal:
                pool.extend(normal)
            if include_elite:
                pool.extend(elite)
            if include_boss:
                pool.extend(boss)
    if not pool:
        raise ValueError(f"Empty mixed encounter pool for acts={list(act_indices)}")
    return pool
