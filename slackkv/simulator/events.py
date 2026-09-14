"""Small event definitions used to document the simulator's event boundaries."""
from dataclasses import dataclass
from enum import Enum


class EventKind(str, Enum):
    RELEASE = "release"
    TRANSFER_COMPLETE = "transfer_complete"
    IDLE_TIMER = "idle_timer"


@dataclass(frozen=True, order=True)
class Event:
    time_s: float
    kind: EventKind
    job_id: str = ""
