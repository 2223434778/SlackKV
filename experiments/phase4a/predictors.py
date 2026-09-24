"""Leakage-safe online median predictors; not run when timing gate fails."""

from __future__ import annotations

import heapq
import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class Boundary:
    decision_ms: float
    resume_ms: float
    signature: str

    @property
    def slack_ms(self) -> float:
        return self.resume_ms - self.decision_ms


def online_predictions(boundaries: list[Boundary]) -> list[dict]:
    """Resolve completed boundaries before every decision, never on decision order alone."""
    pending: list[tuple[float, int, Boundary]] = []
    observed: list[Boundary] = []
    output: list[dict] = []
    # Python's stable sort preserves caller-provided order for simultaneous
    # decisions; never use future resume time to break a decision-time tie.
    for sequence, current in enumerate(sorted(boundaries, key=lambda item: item.decision_ms)):
        if current.resume_ms < current.decision_ms:
            raise ValueError("negative actual slack")
        while pending and pending[0][0] <= current.decision_ms:
            _, _, resolved = heapq.heappop(pending)
            observed.append(resolved)
        assert all(item.resume_ms <= current.decision_ms for item in observed)
        global_value = statistics.median(item.slack_ms for item in observed) if observed else None
        matching = [item.slack_ms for item in observed if item.signature == current.signature]
        tool_value = statistics.median(matching) if matching else global_value
        output.append({"decision_ms": current.decision_ms, "actual_slack_ms": current.slack_ms, "global_prediction_ms": global_value, "tool_prediction_ms": tool_value, "history_count": len(observed), "future_leakage_count": sum(item.resume_ms > current.decision_ms for item in observed)})
        heapq.heappush(pending, (current.resume_ms, sequence, current))
    return output
