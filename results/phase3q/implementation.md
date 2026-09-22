# Implementation

Phase 3Q locally extends the validated Phase 3P controller to A-L, fixed deadlines, 4096-token distinct prefixes, X=256 and a 16-worker outer client pool. No vLLM source change was made. The integration gate was intentionally the only workload execution after it exposed capacity retention failure.
