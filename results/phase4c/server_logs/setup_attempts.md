# Non-final setup and diagnostic attempts

- Stage A0 launch attempt 1 used unsupported `--disable-log-requests`; vLLM exited with code 2 before serving requests.
- Stage A0 launch attempt 2 omitted the active environment's `bin` directory from `PATH`; FlashInfer could not locate `ninja`, and the server exited with code 1 before serving requests. GPU inspection afterward showed no meaningful allocation and no OOM.
- Stage A1 attempt 1 met prompt, output, and cache-block checks but missed the frozen pause-error threshold because the serialized metrics snapshot was taken after the wake deadline. Instrumentation was corrected to snapshot before waiting; a fresh Stage A1 run passed without changing the threshold or workload.
- Stage B was repeated once so external-hit values came from actual serialized server metric deltas rather than an assumed value from the disabled connector. The final result uses the fresh rerun.

These attempts are documented for provenance and are not counted as valid measured requests or server crashes.
