# Implementation

Phase 3N reuses Phase 3M T0–T7 instrumentation unchanged. It starts a 128-token ordinary X decode, submits native prefetch A in an independent thread after X's first client token, and independently sends A at fixed offsets without waiting for the endpoint return. The scheduler arrival timestamp selects the boundary category.
