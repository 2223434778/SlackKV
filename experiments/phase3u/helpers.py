POLICIES = (
    "OnDemand-K2", "Blocking70-K2", "NaiveAsync40-K2",
    "ReservationLedger2-Async40-K2",
)
SESSIONS = tuple("ABCDEFGHIJKL")
DEADLINES_MS = dict(zip(SESSIONS, (300,320,340,360,900,920,940,960,1500,1520,1540,1560)))
K=2
ASYNC_MS=40
BLOCKING_MS=70
RESERVATION_BLOCKS=512
X_TOKENS=256

def h2d_metrics(native_count: int, external_tokens: float):
    native = native_count * 112.0
    external = external_tokens * 112.0 / 4096.0
    total = native + external
    return native, external, total, max(0.0, total - 1344.0)

def x_slowdown(duration_ms: float, baseline_ms: float) -> float:
    return duration_ms / baseline_ms

def promotion_outcome(promoted: bool, t6_ms, arrival_ms, remote_wait: bool):
    if not promoted:
        return "RESERVATION_SKIP"
    if float(t6_ms) > float(arrival_ms):
        return "PROMOTION_INTERNAL_READY_MISS"
    if remote_wait:
        return "PROMOTION_REMOTE_WAIT"
    return "PROMOTION_USEFUL"
