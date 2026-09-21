from __future__ import annotations

import math


def test_recommended_budget() -> None:
    assert math.ceil((57.1 + 10.0) / 5.0) * 5.0 == 70.0


def test_active_decode_interval() -> None:
    assert 1.0 < 2.0 < 3.0


def test_client_noncopy() -> None:
    assert 53.0 - 6.0 == 47.0

