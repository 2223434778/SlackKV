"""Load measured Phase 0 P50 transfer latencies without bandwidth approximations."""
import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


SIZE_MIB = (256, 1024, 4096)
DIRECTION_BY_TIER = {"cpu": "pinned_cpu_to_gpu0", "peer": "gpu1_to_gpu0"}


@dataclass(frozen=True)
class TransferModel:
    source_tier: str
    direction: str
    p50_latency_s_by_bytes: Dict[int, float]
    phase0_commit: str

    @classmethod
    def from_phase0(cls, repository_root: Path, source_tier: str, all_payload_sizes: bool = False) -> "TransferModel":
        if source_tier not in DIRECTION_BY_TIER:
            raise ValueError("source_tier must be cpu or peer")
        path = repository_root / "results" / "phase0" / "summary_metrics.csv"
        values = {}
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row["direction"] == DIRECTION_BY_TIER[source_tier] and (all_payload_sizes or int(float(row["size_mib"])) in SIZE_MIB):
                    values[int(row["size_bytes"])] = float(row["latency_ms_p50"]) / 1000.0
        if not all_payload_sizes and len(values) != len(SIZE_MIB):
            raise RuntimeError("Phase 0 calibration CSV lacks required CPU/peer promotion sizes")
        commit = subprocess.check_output(
            ["git", "log", "-1", "--format=%H", "--", str(path.relative_to(repository_root))], cwd=repository_root, text=True
        ).strip()
        if not commit:
            raise RuntimeError("cannot identify Phase 0 calibration commit")
        return cls(source_tier, DIRECTION_BY_TIER[source_tier], values, commit)

    def service_time_s(self, size_bytes: int, multiplier: float = 1.0) -> float:
        if multiplier <= 0:
            raise ValueError("service multiplier must be positive")
        try:
            return self.p50_latency_s_by_bytes[size_bytes] * multiplier
        except KeyError as error:
            raise ValueError("unmeasured payload size") from error

    def piecewise_service_time_s(self, size_bytes: int, multiplier: float = 1.0) -> float:
        """Interpolate measured latencies; larger sizes are max-size chunks plus remainder."""
        if size_bytes <= 0 or multiplier <= 0:
            raise ValueError("size and multiplier must be positive")
        knots = sorted(self.p50_latency_s_by_bytes.items())
        maximum_size, maximum_latency = knots[-1]
        chunks, remainder = divmod(size_bytes, maximum_size)
        total = chunks * maximum_latency
        if not remainder:
            return total * multiplier
        if remainder <= knots[0][0]:
            return (total + knots[0][1] * remainder / knots[0][0]) * multiplier
        for (left_size, left_latency), (right_size, right_latency) in zip(knots, knots[1:]):
            if remainder <= right_size:
                fraction = (remainder - left_size) / (right_size - left_size)
                return (total + left_latency + fraction * (right_latency - left_latency)) * multiplier
        raise AssertionError("remainder exceeds largest measured payload")
