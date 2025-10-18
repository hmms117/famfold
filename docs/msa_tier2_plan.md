# MSA & FoldMason escalation plan

## Purpose
Keep MSAs and FoldMason off by default; enable only when core routing flags low prior confidence or conflict patterns. (Matches current guidance to avoid standalone MSAs at inference.)

## When to escalate
- Router **ESCALATE** or persistent low prior confidence: `H_norm > 0.35` or `S_prior < 0.20`.
- Alignment conflicts: ≥25% long-range pairs disagree across top templates.
- Coverage gap: `coverage < 0.50` with `A < 0.40`.
(Thresholds mirror existing policy.)

## Inputs
- `families/<fid>/retrieval/alignments/*.aln.json` (if any)
- Templates + pLDDT from `templates/*.json.gz`
- Optional seed MSAs (`.a3m`/`.sto`) or MAFFT config

## Steps
1. **Conflict detection**
   - Compute fraction of long-range pairs (|i−j| ≥ 24) with low consensus across template priors.
   - If above threshold, mark regions for MSA refinement.

2. **MSA generation (on-demand)**
   - MAFFT quick profile for the conflict windows (bounded to ±32 residues).
   - Cap depth to N=256 (or time budget); deduplicate.

3. **FoldMason column mapping**
   - Map MSA columns ↔ template residues; persist to `meta.alignment` in template JSON if used.

4. **AIP refresh**
   - Recompute AIP with window-aware σ widening in conflict regions; update `P_prior_final`, `W`, `γ`.

5. **Re-inference**
   - Re-run `inference_routing` with updated priors; allow one recycle with uncertainty-aware prior decay.

## Interfaces & artifacts
- Write: `retrieval/alignments/<qhash>__<tid>.aln.json`
- Update (if used): `templates/<tid>.json.gz.meta.alignment`
- Log: `logs/msa_escalation.jsonl` (trigger, timings, windows, depth, effects)

## Config knobs
```yaml
msa:
  enable: false           # default
  trigger:
    conflict_frac: 0.25   # region flag
    min_gap_cov: 0.50
    min_A: 0.40
  mafft:
    max_depth: 256
    window_radius: 32
foldmason:
  enable: false           # default
```

## Tests
- Unit: column maps round-trip; σ widening applied only inside windows.
- Integration: after escalation, H_norm ↓ and S_prior ↑ on flagged families.
- Stress: depth caps respected; runtime bounded.

## Rationale for isolation
The core docs stay crisp; all heavy/expensive escalation paths live here so teams can opt in explicitly when routing deems it necessary.
