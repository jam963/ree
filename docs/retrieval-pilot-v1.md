# Two-corpus retrieval and compatibility pilot v1

Authorized by the user's 2026-09-09 request to run retrieval quality and mixed-
precision compatibility now. **This is an explicit exploratory exception to the
full-suite scoring embargo in evaluation-protocol.md, not a completed selection
gate or a replacement twelve-stratum suite.** Freeze this document, input hashes,
runner/scorer source hashes, bootstrap indices and executable hashes in the run
plan before the first real inference. No settings are chosen from pilot scores.

## Fixed scope

- Exactly the previously prepared NFCorpus and SciFact files/hashes in
  `benchmarks/datasets/small-beir-prepared.json`: 100 queries each, complete 3,633
  and 5,183 document pools. Preserve qrels/grades, text, query selection and ID order.
- All five previously pinned candidate CPU/CUDA recipes. Native dimensions first;
  Arctic 256 is derived from the **same** native 768-dimensional vector exports,
  truncated and L2-renormalized. No other dimension experiments in this pilot.
- No ArguAna repair, new queries, dropped judgments, mining, model downloads,
  artifact conversions, hyperparameter tuning or production recipe changes.
- The distributions declare CC-BY-SA-4.0. Source text stays in private local
  scratch; only hashes, IDs, metrics and diagnostics are recorded in the repository.
  This pilot does not complete the pending upstream rights/release-license review.
- No weighted 45/20/15/15/5 score, general-domain macro score, significance-based
  shipping decision, or substitution of these two scientific/biomedical tasks for
  the missing web/financial/argumentative/multilingual/code/ree strata.

## Encoding and compatibility matrix

Use ree's Rust tokenizer/ONNX runtime, 512 tokens including the already-pinned
prompts/special tokens, 8 inference threads, balanced power on AC. One process per
candidate/corpus/provider, one session load/warm-up, no speed probes. Run serially
with no simultaneous builds/provisioning. Record load, host and sampled resources;
encoding elapsed time includes audit tokenization and is **not** the speed baseline.

Documents: batch 8 in corpus ID order; queries: individual batch 1. No length
sorting, batching retries or automatic CPU fallback in these explicit-provider
runs. An inference failure is a failed/incomplete run, not a missing-score zero.
Record exact framed token-input hashes and per-role full/encoded token counts,
truncation counts and longest original input. CPU/GPU token hashes must agree.

Export native f32 little-endian vectors with ordered IDs and SHA-256, finite/unit-
vector validation, and a final metadata completion marker. Reject missing/truncated,
corrupt, misordered or semantically incompatible exports before offline scoring.
Metadata pins model/revision/tokenizer/recipe/prompts/runtime/provider/batch settings.

Score six index combinations, labeled **query provider / document index**:

1. CPU/CPU and CUDA/CUDA (homogeneous references).
2. CPU/CUDA and CUDA/CPU (cross-artifact/provider compatibility).
3. CPU/mixed and CUDA/mixed (one index containing both artifact recipes, simulating
   stored vectors spanning a fallback). For each document, CUDA is selected when
   the first SHA-256 byte of UTF-8 `ree-mixed-index-v1\0` + `20260909\0` + document
   ID is odd; otherwise CPU. This is an approximately 50/50 deterministic mixture,
   shared across all models. It is a compatibility stress case, not a simulation
   of every possible chronological fallback boundary.

Use the existing metric-version-2 policy: f32 dot product over L2 vectors, ID-
ascending tie breaks, fixed top 10, pinned self-match exclusion, binary-positive
Recall/MRR and linear-graded nDCG. Mean per-query metrics separately per corpus.
Report all six rows, per-query top-10 rankings, and top-1 agreement/top-10 overlap
against **both** homogeneous references. Do not select whichever reference makes
ranking agreement appear better.

For each mixed row report nDCG **loss** = max(CPU/CPU mean, CUDA/CUDA mean) minus
that row's mean. Flag loss >0.01 absolute. Improvements are not counted as losses.
Cross-provider and mixed-index flags are separate diagnostics; passing them on two
corpora does not qualify the full production fallback recipe. Also report native
CPU/CUDA cosine distributions across every document and query; cosine alone is
not a compatibility gate.

Batch-dependence diagnostic: rerun 32 documents individually per provider/corpus.
Select the lowest SHA-256 hex digests of UTF-8 `ree-batch-drift-v1\0` +
`20260909\0` + document ID, break digest ties by corpus position, then infer in
corpus order. Compare these exact vectors against their batch-8 counterparts.
These document-only samples report cosine drift, **not** full singleton-index
retrieval metrics or a guarantee for all production scheduler shapes.

## Paired uncertainty estimates

Use 10,000 query bootstrap replicates per corpus; all models/dimensions/combinations
share that corpus's indices. Each replicate samples 100 indices with replacement.
Derive the stream by SHA-256 over UTF-8 `ree-bootstrap-v1\0` + `20260909\0` +
corpus SHA-256 + `\0`, followed by a u64 little-endian block counter starting at 0.
Read four u64 little-endian draws per digest. Reject draws >=
`2^64 - (2^64 mod query_count)`; accepted draw modulo query_count is the index.
Persist the row-major u32 little-endian index file and its hash **before inference**.

For each row report the 2.5/97.5 percentile bootstrap interval of mean nDCG.
Percentiles use linear interpolation at index `p * (10000 - 1)`. For mixed-row
loss intervals, recompute the better homogeneous reference inside each replicate
rather than fixing the reference from the point estimate. Report paired candidate
minus Arctic deltas for matching combinations and paired Arctic-256 minus 768
deltas. These are descriptive, per-corpus, unadjusted intervals—not twelve-stratum
weighted confidence bounds or evidence of a shipping winner. Keep all comparisons,
including regressions and intervals crossing zero.

## Completion and interruptions

The plan binds the two corpus hashes, five recipes, executable/source hashes, this
protocol hash, seed/settings and bootstrap-index hashes. Outputs are new files in
a dedicated run directory; vectors/indices remain in the private scratch workspace.
A new run never overwrites another run. Resume only completed exports whose input
and output hashes match the frozen plan; partial runs are retained and require an
explicit separate retry directory, not reuse as complete caches. Preserve logs
for every failure and never change batching/precision just to finish a row.

The pilot is complete only when all 20 encoding runs and all 12 candidate/corpus/
dimension comparisons (six combinations each) have completed and validated.
Remaining full-suite tasks stay open regardless of these results.
