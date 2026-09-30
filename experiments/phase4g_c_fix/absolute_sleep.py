"""Linux CLOCK_MONOTONIC absolute sleep backend for Phase 4G-C-FIX."""
from __future__ import annotations

import ctypes
import ctypes.util
import errno
import threading
import time
from dataclasses import dataclass
from typing import Callable

CLOCK_MONOTONIC = 1
TIMER_ABSTIME = 1
BACKEND = "CLOCK_NANOSLEEP_ABS"


class Timespec(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]


class TimingBackendError(RuntimeError):
    def __init__(self, return_code: int):
        self.return_code = int(return_code)
        super().__init__(f"clock_nanosleep failed with return code {self.return_code}")


@dataclass(frozen=True)
class SleepResult:
    return_code: int
    eintr_retry_count: int


@dataclass(frozen=True)
class WaitObservation:
    target_ns: int
    sleep_enter_ns: int
    sleep_wakeup_ns: int
    return_code: int
    eintr_retry_count: int


_libc_name = ctypes.util.find_library("c")
_libc = ctypes.CDLL(_libc_name, use_errno=True) if _libc_name else None
if _libc is not None:
    _native_clock_nanosleep = _libc.clock_nanosleep
    _native_clock_nanosleep.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.POINTER(Timespec), ctypes.c_void_p,
    ]
    _native_clock_nanosleep.restype = ctypes.c_int
else:
    _native_clock_nanosleep = None

_thread_state = threading.local()


def timespec_from_ns(target_ns: int) -> Timespec:
    if target_ns < 0:
        raise ValueError("absolute monotonic deadline must be non-negative")
    return Timespec(target_ns // 1_000_000_000, target_ns % 1_000_000_000)


def absolute_sleep_until_ns(
    target_ns: int,
    *,
    now_ns: Callable[[], int] | None = None,
    syscall: Callable[[int, int, Timespec], int] | None = None,
) -> SleepResult:
    """Sleep to one absolute monotonic deadline; EINTR retries that same target."""
    clock = now_ns or time.monotonic_ns
    call = syscall or _call_native
    if target_ns <= clock():
        return SleepResult(0, 0)
    request = timespec_from_ns(target_ns)
    retries = 0
    while True:
        result = int(call(CLOCK_MONOTONIC, TIMER_ABSTIME, request))
        if result == 0:
            return SleepResult(0, retries)
        if result == errno.EINTR:
            retries += 1
            continue
        raise TimingBackendError(result)


def _call_native(clock_id: int, flags: int, request: Timespec) -> int:
    if _native_clock_nanosleep is None:
        raise RuntimeError("CLOCK_NANOSLEEP_ABS is unavailable on this host")
    return int(_native_clock_nanosleep(
        clock_id, flags, ctypes.byref(request), None,
    ))


def phase4g_c_fix_wait_until(target_s: float) -> None:
    """Wait without Python sleeping or spinning and retain a tiny thread-local sample."""
    target_ns = round(target_s * 1_000_000_000)
    sleep_enter_ns = time.monotonic_ns()
    result = absolute_sleep_until_ns(target_ns)
    sleep_wakeup_ns = time.monotonic_ns()
    _thread_state.last_wait = WaitObservation(
        target_ns=target_ns,
        sleep_enter_ns=sleep_enter_ns,
        sleep_wakeup_ns=sleep_wakeup_ns,
        return_code=result.return_code,
        eintr_retry_count=result.eintr_retry_count,
    )


def take_wait_observation() -> WaitObservation | None:
    observation = getattr(_thread_state, "last_wait", None)
    _thread_state.last_wait = None
    return observation
