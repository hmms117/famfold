# Tier 1 Plan Addendum – SaESM2 Fast-Path Guidance

This addendum supplements the Tier 1 FamilyFold plan with guidance for the new
SaESM2 fast trunk surfaced in MiniFold.

## When to Choose SaESM2 vs. ISM

- **SaESM2 Fast (`saesm2_fast`)**
  - Use for retrieval-heavy Tier 1 families where embedding throughput is the
    primary bottleneck.
  - Provides 35M-parameter SaESM2 embeddings with the same layer-normalisation
    scheme as the existing ISM fast path, keeping caches interchangeable.
  - Recommended for cold starts and large backlog clears once caches are warm.
- **ISM Fast (`ism_fast`)**
  - Prefer when absolute agreement with legacy ISM metrics is required or when
    SaESM2 validation has not yet been performed for the family.
  - Retains faesm-backed kernels and is the conservative fallback for
    high-risk launches.

## Cache Warmup Notes

- Both trunks normalise embeddings identically, allowing shared caches after an
  initial warmup run. When switching to SaESM2, populate the cache by replaying
  the pilot batch before enabling production routing.
- Monitor the retrieval cache hit rate; the SaESM2 fast path should stabilise
  above 90% once the warmup completes. If hit rates lag, schedule a targeted
  cache prefetch using the new benchmarking utilities.

## Telemetry Expectations

- Telemetry dashboards now flag `saesm2_fast` runs, exposing acceptance and
  escape rates alongside GPU memory footprints.
- Include the dashboard snapshot in Tier 1 readiness reviews, highlighting any
  divergence between SaESM2 and ISM quality metrics (RMSD/TM-score).
