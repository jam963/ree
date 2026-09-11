# Retrieval pilot — completed, 2026-09-09

**All 20 encoding jobs and 12 six-way comparisons completed and verified.**
The frozen run finished at **2026-09-09T22:11:22Z**. This is still a two-corpus
exploratory pilot, **not the twelve-stratum selection gate or a shipping decision**.
Production remains provisional Arctic; no runtime, schema or embedding recipe was
changed. The preregistered scope is [pilot v1](../../../docs/retrieval-pilot-v1.md).

- `plan.json`: original corpus/recipe/source/executable/bootstrap hashes, unchanged.
- `summary.json`: all 72 metric rows, 48 compatibility checks, paired intervals and
  candidate-minus-Arctic / Arctic-256-minus-768 deltas.
- [Complete tables](metrics.md): every nDCG/Recall/MRR row, nDCG interval,
  cross-provider/mixed-index loss and loss interval, plus truncation coverage.
- `*-matrix.jsonl`: 7,200 per-query rankings/metric rows and vector-drift diagnostics.
- `completion-verification.json`: verification of all 20 exports (90,800 finite,
  unit vectors), matching CPU/CUDA token hashes, deterministic diagnostic samples,
  and independently recomputed bootstrap intervals. All **2,000 native homogeneous
  query rankings and all three metrics exactly reproduce the live encoding logs**.

## Homogeneous retrieval quality

Mean **nDCG@10 × 100**, higher is better. Labels are **query provider / index
provider**. Each corpus has 100 fixed queries; all 3,633 NFCorpus and 5,183 SciFact
documents compete. Positive grades, selected queries, ID ordering and self-match
policy are unchanged. Arctic 256 projects/renormalizes the same native 768 exports.

| Recipe | NFCorpus CPU/CPU | NFCorpus CUDA/CUDA | SciFact CPU/CPU | SciFact CUDA/CUDA |
| --- | ---: | ---: | ---: | ---: |
| Arctic 768 | 35.76 | 36.71 | 72.00 | 72.86 |
| Arctic 256 | 34.80 | 36.09 | 69.96 | 71.48 |
| Granite 384 | 30.21 | 30.21 | 64.54 | 64.54 |
| E5-small 384 | 31.98 | 32.16 | 64.97 | 66.81 |
| Jina Code 768 | 10.09 | 9.16 | 19.88 | 20.13 |
| BGE-M3 1024 | 33.83 | 33.75 | 65.58 | 65.58 |

Arctic has the highest homogeneous scores among the measured recipes on **these
scientific/biomedical tasks**. That does not establish a winner for web, financial,
argumentative, multilingual, code or ree-specific retrieval. Jina's pinned recipe
scores substantially lower on these tasks; its code-retrieval quality was not
measured, and these results do not establish the cause of the deficit. Do not tune
or substitute an export based on the pilot and present it as the original recipe.

Arctic 256 loses **2.04 CPU / 1.38 CUDA SciFact points** versus 768. The paired
intervals for 256-minus-768 are **−4.07..−0.12 CPU / −3.31..0.45 CUDA**. Do not
declare dimension equivalence or change production dimensions from this pilot.

## Mixed-provider and mixed-index compatibility

Arctic/E5/Jina compare **CPU INT8 / CUDA FP16**. Granite/BGE use the **same FP32
artifact on both providers**, not an INT8/FP16 comparison. The deterministic mixed
index selects CPU or CUDA vectors per document, approximately 50/50 and identically
across models. Both query providers are tested against it.

Loss is the better homogeneous mean minus the mixed row's mean. The preregistered
flag is **loss >1.00 nDCG point** (>0.01 on the raw scale). A negative signed loss
is an improvement, not a failure. All 48 checks are retained: **eight flag the
point-estimate cutoff**, all on SciFact. Five flags concern native recipes and
three concern Arctic 256. No NFCorpus row flags the cutoff.

### Every flagged row

Losses and descriptive paired 95% intervals are in nDCG points:

| Recipe | Query / index | Loss | 95% interval |
| --- | --- | ---: | ---: |
| Arctic 768 | CUDA / CPU | 1.19 | −0.46..3.35 |
| Arctic 256 | CPU / CUDA | 1.39 | 0.17..3.08 |
| Arctic 256 | CUDA / CPU | 1.43 | 0.08..3.34 |
| Arctic 256 | CPU / mixed | 1.19 | −0.69..3.47 |
| E5 | CPU / CUDA | 1.54 | 0.17..3.42 |
| E5 | CUDA / CPU | 1.76 | −0.37..4.35 |
| E5 | CPU / mixed | 4.70 | 2.55..7.36 |
| E5 | CUDA / mixed | 2.38 | 0.88..4.39 |

- **Arctic 768:** its SciFact cross-provider flag is statistically uncertain; the
  interval crosses zero. The CPU-query/mixed-index point loss is 0.98, just below
  the cutoff, with interval −0.71..3.05. Passing that point cutoff does not prove
  equivalence. Production fallback compatibility is not cleared.
- **E5:** all four SciFact mixed rows flag. The worst mixed-index interval is wholly
  above one point, despite high CPU/CUDA vector cosine. This is the clearest
  observed mixed-index regression, not a reason to hide the other rows.
- **Jina:** all eight point checks stay below one point; the largest is 0.74 on
  NFCorpus CUDA/mixed (0.03..1.56). SciFact passing-row intervals reach about 2.45
  points. These are not equivalence guarantees, and compatibility does not rescue
  the low absolute retrieval quality on the two measured tasks.
- **Granite:** all six combinations have identical measured nDCG/Recall/MRR on
  both corpora, with full observed top-1 agreement against both references. This
  establishes observed same-FP32 provider parity only within the pilot.
- **BGE-M3:** all combinations have identical SciFact rankings/metrics. NFCorpus
  shows small ordering differences; the worst mixed-index point loss is **0.08**
  (−0.01..0.24), below the cutoff. Do not claim bitwise/provider ranking identity.

BGE's two NFCorpus homogeneous top-10 changes involve pairs with **identical
stored document text but different IDs**. CPU f32 dot scores tie; CUDA margins are
about 1–2 × 10⁻⁷ and reverse their order. For `PLAIN-2660`, only `MED-2905` has a
positive qrel (grade 2), while `MED-3032` has none. That one ordering change lowers
mean homogeneous nDCG by 0.0812 points; the other query's metrics do not change.
The IDs/text/judgments were preserved, not deduplicated or repaired. No positive
qrel is not proof of irrelevance. See `bge-provider-diagnostics.json` for the
retained-vector f32 calculation; it ran no new inference.

## Uncertainty, truncation and batch diagnostics

All intervals use **10,000 paired query bootstrap replicates per corpus**, with
SHA-256-derived draws saved before inference. The better homogeneous reference is
recomputed inside each loss replicate. They are descriptive and **not adjusted
for multiple comparisons**, not full-suite weighted confidence bounds. The
completion check regenerated both draw files byte-for-byte and independently
recomputed all **204 intervals** using separate mean/percentile arithmetic;
maximum absolute difference was 1.12 × 10⁻¹⁶. All eight earlier partial comparisons
reproduce exactly, excluding their wall-clock scoring-duration field.

No selected queries were truncated. Jina truncates 762/3,633 NFCorpus documents
(20.97%) and 892/5,183 SciFact documents (17.21%); the other recipes truncate about
14%. Exact counts, original/encoded token totals and longest original inputs are
retained. Inputs remain limited to 512 tokens including prompts/special tokens;
this is not long-context or chunked-document qualification.

Native CPU/CUDA document cosine means are about 0.955–0.957 Arctic, 0.9943 E5 and
0.979 Jina. Minimum SciFact cosine is **0.91572 Arctic / 0.91516 Jina**. Arctic is
below the historical provisional 0.94 smoke bound, which was never a universal
qualification guarantee. The 32-document singleton-versus-batch-8 CPU cosine
means are about 0.983 Arctic, 0.998 E5 and 0.983–0.985 Jina. CUDA batch diagnostics
are approximately 0.999998 or higher; same-FP32 Granite/BGE drift is much smaller.
Cosine, batch dependence and real ranking compatibility remain distinct tests.

## Interruption history and retained artifacts

1. At **17:08:09Z**, AC disconnected and the profile changed to quiet during
   Jina's first CPU attempt. Twelve exports were complete. Historical `pause.json`
   and `interrupted-attempts/jina-code-nfcorpus-cpu/` preserve that attempt.
2. The guarded run resumed at **18:53Z** after full validation on AC/balanced. Three
   Jina jobs completed. At **19:18:53Z**, the guard stopped Jina SciFact CUDA;
   inspection found AC connected but the profile set to performance. Historical
   `resume-pause.json`, `resume-verification.json`, `resume-launch.json` and
   `resume-runner.*` preserve this second interruption with 15 exports complete.
3. After the user restored balanced mode, the run resumed at **19:23Z** with all
   15 exports revalidated. The remaining five jobs and all matrices/bootstrap
   analysis completed. `resume-2-*` records verification, launch and runner logs.
   Neither interruption is classified as a model failure.

Both incomplete attempts remain archived separately in the private workspace and
under `interrupted-attempts/`. Complete vectors and bootstrap files remain at the
absolute `workspace` path in `plan.json`; source corpora stay in private scratch.
Original source, executable, corpus and recipe hashes were unchanged throughout.
All complete jobs recorded AC/balanced before and after inference. Runtime checks
still report ONNX Runtime 1.28, CUDA driver/runtime API 13030 and cuDNN 92500.
No packages, downloads, GPU resets, power-policy changes or Rust rebuilds were
performed during continuation. Desktop/background OS work was not disabled;
encoding duration includes audit work and **is not a throughput benchmark**.

`status.json` is now **completed**. The pause records and `partial-analysis/` are
historical evidence, not current status. **Nothing remains running; do not run
`--resume` against this completed output or overwrite its summary.**

Nineteen offline Python tests now pass, including full coverage, all raw metrics,
loss flags/overlap/point deltas, live roundtrips and preservation of both pauses.
The earlier 102-passed/five-ignored Rust result remains historical; no new Rust
code was needed and the frozen binaries were not rebuilt.

## Remaining decision work

Investigate mixed-precision/index and production-batch compatibility without
relaxing the frozen thresholds. Complete the remaining web/financial/argumentative,
multilingual, code and reviewed ree strata, dataset/model license review and full
weighted paired aggregation before selecting a shipping recipe. ArguAna's five
dangling qrel document IDs remain blocked and must not be silently removed.
This completed pilot does not close those gates.
