from dataclasses import dataclass


@dataclass(frozen=True)
class PromotionJob:
    job_id: str
    release_time_s: float
    deadline_s: float
    kv_size_bytes: int
    service_time_s: float

    def __post_init__(self) -> None:
        if self.release_time_s < 0 or self.deadline_s < self.release_time_s:
            raise ValueError("deadline must be no earlier than release")
        if self.kv_size_bytes <= 0 or self.service_time_s <= 0:
            raise ValueError("KV size and service time must be positive")

    @property
    def kv_size_gb(self) -> float:
        return self.kv_size_bytes / 1e9
