"""Clone performance smoke tests."""

from __future__ import annotations

import copy
import time

import pytest

from sts2_env.search.combat_clone import _rewire_combat_pointers, clone_combat_state


def _clone_via_deepcopy(combat):
    cloned = copy.deepcopy(combat)
    _rewire_combat_pointers(cloned)
    return cloned


@pytest.mark.slow
def test_explicit_clone_faster_than_deepcopy(simple_combat):
    iterations = 1000

    start = time.perf_counter()
    for _ in range(iterations):
        _clone_via_deepcopy(simple_combat)
    deepcopy_s = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(iterations):
        clone_combat_state(simple_combat)
    explicit_s = time.perf_counter() - start

    speedup = deepcopy_s / explicit_s
    assert speedup >= 1.75, f"expected >=1.75x speedup, got {speedup:.2f}x"
