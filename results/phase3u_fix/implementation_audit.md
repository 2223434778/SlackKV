# Phase 3U-FIX implementation audit

The prior driver called `send_resume(..., ready[s] if policy == "Blocking70-K2" else None, True)`. The positional `True` selected the asynchronous branch globally, so the existing condition `if ready is not None and not async_policy: ready.wait(...)` never waited even for Blocking70.

The fixed driver removes `async_policy` from the resume API. The signature uses keyword-only `ready_event` and `block_on_prefetch`. Only Blocking70 receives its event and `True`; OnDemand, NaiveAsync40, and ReservationLedger receive `None` and `False`. Each Blocking70 resume records deadline arrival, wait interval, and client send time, then validates send-after-prefetch-return within 0.5 ms.

The controller uses 70 ms service for Blocking70 and 40 ms for both asynchronous promotion policies. Native rows report that configured value. Category summaries preserve absent promoted/fallback categories as `NA`; they no longer substitute all sessions for an empty skipped group. X slowdown uses the median of three fresh X-only decode-duration measurements.

The Phase 3S lease implementation and vLLM source are unchanged.
