from experiments.phase3o.helpers import (
    ASYNC_SERVICE_MS,
    BLOCKING_SERVICE_MS,
    async_resume_may_send,
    configured_service_budget,
    effective_latency_ms,
    gpu_ready_residency_ms,
    internal_ready_miss,
)


def test_effective_latency_uses_original_deadline_not_send_time() -> None:
    assert effective_latency_ms(380.0, 300.0) == 80.0


def test_t6_readiness_and_residency() -> None:
    assert internal_ready_miss(299.0, 300.0)
    assert not internal_ready_miss(300.0, 300.0)
    assert gpu_ready_residency_ms(340.0, 300.0) == 40.0
    assert gpu_ready_residency_ms(280.0, 300.0) == 0.0


def test_async_does_not_wait_for_http_future() -> None:
    assert async_resume_may_send(False)
    assert async_resume_may_send(True)


def test_frozen_policy_budgets() -> None:
    assert configured_service_budget("Blocking70-K2") == BLOCKING_SERVICE_MS == 70.0
    assert configured_service_budget("Async40-K2") == ASYNC_SERVICE_MS == 40.0
    assert configured_service_budget("OnDemand-K2") is None
