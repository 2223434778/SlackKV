# Implementation notes

- `release_fix.py`: structured `SLACKKV_LEASE_TRACE` parser, incremental file-tail reader, exact episode-to-lease-key release queue, physical-before-logical timestamp validation, and frozen N1–N4/watchdog helpers.
- `integration.py`: drain exact physical lease releases before future reservation admission and after resumes; persist decision rows incrementally and persist each successful native endpoint response before awaiting the next one. The frozen scheduler/promotion policy and vLLM source were not changed.
- `liveness.py`: mechanically derives N1–N4 from the frozen jobs and contains the isolated promotion-only L0/L1 runner. It was not invoked because the Reservation integration prerequisite failed.
- `tests/test_phase4d_a_fix.py`: 35 focused regression/metric/gate tests.

The first attempts exposed a harness observability weakness: responses were only written after every promotion future completed, so a later timeout discarded earlier successful response objects. Retry2 adds immediate response persistence. The exact lease parser avoids re-reading the whole growing server log on each scheduling iteration.

No vLLM, scheduler, OffloadingConnector, physical lease, reservation policy, deadline, `P_SAFE`, K2, GPU capacity, or CPU tier changes were made.
