# Model-selection evaluation protocol v1

**Beta decision update, 2026-09-10:** the user selected Arctic M v2 at 768 dimensions
and deferred broader evaluation. [Model decision v1](model-decision-v1.md)
supersedes completing this protocol as a prerequisite to choosing the beta model.
The rules/results below remain the original empirical protocol: no thresholds
were relaxed, no missing measurements filled in, and no full-suite pass or weighted
winner is claimed. Do not start further comparative evaluations unless requested.
Correctness, security, license/notice and packaging safeguards are not waived.

Historical evaluation status, 2026-09-09: **premeasurement rules specified; suite lock still pending**.
Five candidate artifacts/recipes are now pinned and CPU/CUDA speed baselines are
recorded. NFCorpus/SciFact are prepared; ArguAna is blocked by missing qrel document
IDs. See [candidate evidence](../benchmarks/candidates/README.md) and
[dataset audits](../benchmarks/datasets/README.md). A subsequent user request
explicitly authorized [pilot v1](retrieval-pilot-v1.md), a separately frozen,
two-corpus exploratory exception. That pilot is now complete for all five models:
20 encoding jobs, 12 six-way comparisons and paired per-corpus bootstrap analysis.
Its SciFact mixed-recipe/dimension regressions remain unresolved. Completion does
not replace the twelve-stratum suite or supply a weighted shipping decision.
This advances gate 1 of [the beta roadmap](beta-roadmap.md), not a model decision.
Do not use two-corpus pilot results as a full model-selection comparison until the
lock checklist below is complete. Rule changes after examining results require a new protocol version
and reruns of every affected candidate, not a favorable subset of reruns.

## 1. Required strata and local scope

The intended suite is fixed before model scoring:

| Component | Strata (one corpus/run per entry) | Split | Queries per stratum |
| --- | --- | --- | --- |
| General, 45% | BEIR NQ (web), NFCorpus (biomedical), SciFact (scientific), FiQA (financial), ArguAna (argumentation) | test | 100 |
| Multilingual, 20% | MIRACL Arabic, Japanese, Russian, Spanish | dev | 100 |
| Code/technical, 15% | CoIR CodeSearchNet, CoIR CosQA, ree-specific technical material | test for public tasks; locked local set for ree | 100 public; 50 ree |

Use all source documents when the complete upstream corpus has at most 10,000
documents; otherwise prepare 10,000-document candidate pools by the procedure
below. These are **ree local subsets**, not published BEIR/MIRACL/CoIR scores.
Do not combine documents from unrelated strata to create easy negatives.
Do not drop a stratum because a model performs poorly or an artifact is awkward.
If provenance, licensing, or capacity blocks a task, stop and version the suite
before scoring any models rather than silently redistributing its weight.

The task choices and sample sizes are ree policy, not claims of benchmark-author
endorsement. BEIR's canonical interchange uses document/query JSONL and three-
column qrel TSV. MIRACL supplies topics/qrels separately from its corpus; its
annotated negatives are actual judgments, not merely unjudged retrieval hits.
CoIR supplies multiple code-retrieval tasks. Consult the primary references below
and retain per-dataset license evidence; a benchmark repository's software license
must not be assumed to license every underlying document.

The ree set must be authored and reviewed **before** looking at candidate output:
50 non-verbatim practical queries, at least 500 source passages, with explicit
positive judgments and manually reviewed same-topic distractors. Balance source
code, SQL/storage behavior, configuration, extraction, and recovery (10 queries
each). Pin the repository snapshot, passage boundaries, authoring/review record,
and qrels. Avoid merely paraphrasing the six smoke documents. This is an in-domain
check, not an independent generalization claim.

## 2. Freeze inputs, then select

Each task needs a reviewed manifest containing the upstream URL, immutable
revision/release, split, license identifier and evidence, upstream download
SHA-256/size inventory, and the exact conversion/mining code revision and settings.
Keep upstream archives or reproducible download instructions outside Git; commit
the small manifests, hashes, and selection audit. Do not execute dataset/model-
supplied Python. No artifact or dataset downloads occur in the preparer.

Normalization produces:

- `documents.jsonl`: `{"_id":"d","title":"optional","text":"passage"}`.
  Stored evaluation text is `title + "\n" + text` when title is nonempty, otherwise
  text unchanged. No whitespace cleanup, Unicode normalization, chunking, or
  tokenizer-dependent subset selection.
- `queries.jsonl`: `{"_id":"q","text":"query"}`.
- `qrels.tsv`: exact header `query-id\tcorpus-id\tscore`, then tab-separated rows.
  Preserve integer positive grades and zero judgments. Unsupported fractional or
  negative grades require an explicit reviewed conversion policy, not coercion.
- `negatives.jsonl`, for subsets: one record per query,
  `{"query_id":"q","document_ids":["hardest","next", "..."]}`.
  A pinned best-first distractor ranking from the **upstream task corpus**, not
  from the eventual subset and not from any candidate embedding model. Pin the
  miner/tokenizer/version/parameters and input hashes. A lexical miner must have
  a documented appropriate segmentation policy per language; do not silently
  apply whitespace tokenization to Japanese. Mining adapters are still pending.
  Hand-reviewed rankings are allowed only for the ree task, with review evidence.

Selection algorithm `ree-subset-v1`, implemented by `examples/prepare_corpus.rs`:

1. Use seed string `20260909` for every stratum. Require unique nonempty IDs,
   consistent references and one judgment per query/document pair. All input
   query IDs must exist for both qrels and negative lists. Select only queries
   with at least one positive judgment; record the eligible count. Do not
   dynamically replace selected queries whose documents are inconvenient.
2. Rank eligible query IDs by SHA-256 of four UTF-8 components: algorithm version,
   seed, `query`, ID. Prefix **each** component with its byte length as a u64
   big-endian integer. Select the requested count from ascending digests, breaking
   collisions by ascending ID. Too few eligible queries is an error, not a
   silently smaller evaluation.
3. Retain **all** positive and zero-judged documents for selected queries. Preserve
   their positive grades in output `relevance`; zero judgments remain identifiable
   through the pinned source qrels and are counted in the audit. No positive is
   dropped to make the document budget fit.
4. For each query, retain the first **20** unique IDs from its pinned distractor
   ranking after removing its known positives and any excluded self-match.
   These are *unjudged distractors* unless independently judged zero. Require
   the full quota for every query unless the reviewed manifest asserts that the
   source is the complete upstream corpus and all of it fits. A small normalized
   input file alone is not evidence of a complete corpus.
5. Fill unused slots with remaining documents ranked by the same SHA-256 scheme,
   replacing component `query` with `document`. Unioning required documents must
   fit the budget; otherwise stop for a premeasurement protocol revision. Never
   reduce the positive/negative quota or skip hard queries to force a fit.
6. Sort output documents/queries by ID. Embed the entire preparation manifest,
   its byte hash, eligible/source counts, retained judgment counts, and per-query
   negative counts in provenance. Output has its own SHA-256. Input file order
   does not affect selected content; changes to input bytes still change provenance.

All documents in each query's task pool compete at scoring time, not just that
query's 20 distractors. Incomplete relevance judgments remain a limitation; do
not claim unjudged documents are known irrelevant or infer scores for omitted
parts of the upstream corpus. The tool verifies normalized inputs, **not** the
truth of upstream provenance or the quality of the supplied negative miner.
Those require review before the suite is locked.

### Offline preparer

```sh
df -h /scratch
cargo run --locked --example prepare_corpus -- \
  --manifest /scratch/ree-eval/task-manifest.json \
  --scratch-dir /scratch --output /scratch/ree-eval/task.json
```

Paths in input entries resolve relative to the manifest. The output must be new;
publication uses a same-directory temporary file and no-clobber atomic persistence
only after validation/hash checks. Temporary SQLite indexes contain IDs/qrels,
not the full document text, and are removed on ordinary success/failure. A process
kill can leave a scratch directory; never treat it as a resumable output corpus.
Reserve scratch for indexes before running; disk exhaustion fails preparation
without replacing an existing corpus. Process scratch is not a security sandbox.

The streaming input reader hashes the exact bytes parsed. Limits: 1 MiB per line,
8 GiB document input, 128 MiB each query/qrel/negative input, 100,000 source queries,
1,000 selected queries, 10,000 selected documents, 96 MiB retained text, 128 MiB
serialized corpus. Query text is materialized; document text is streamed with
bounded retained candidates. A larger source needs a reviewed preparation change,
not silent truncation. A bounded gzip adapter for the pinned small BEIR corpora
now exists; remaining source adapters and hard-negative mining are still pending.

Manifest shape (illustration only; placeholders are deliberately not usable locks):

```json
{
  "version": "ree-subset-v1",
  "dataset": "BeIR/fiqa",
  "source_url": "UPSTREAM_URL",
  "revision": "IMMUTABLE_DATASET_REVISION",
  "split": "test",
  "domain": "financial",
  "license": "REVIEWED_DATASET_LICENSE",
  "license_reference": "PINNED_LICENSE_EVIDENCE",
  "preparation": "CONVERTER_AND_MINER_REVISION_SETTINGS_AND_UPSTREAM_HASH_INVENTORY",
  "seed": "20260909",
  "query_count": 100,
  "document_count": 10000,
  "source_is_complete": true,
  "exclude_query_id": true,
  "hard_negatives_per_query": 20,
  "documents": {"path": "documents.jsonl", "sha256": "SHA256"},
  "queries": {"path": "queries.jsonl", "sha256": "SHA256"},
  "qrels": {"path": "qrels.tsv", "sha256": "SHA256"},
  "negatives": {"path": "negatives.jsonl", "sha256": "SHA256"}
}
```

`source_is_complete` describes the normalized **input**, not the smaller output.
`negatives` can be null for a complete small corpus. The tool accepts explicit
other counts/seeds for fixture/debug use; only the fixed suite settings above
qualify for this protocol. Commit a final suite inventory and audit every manifest
against it before model scoring; an accepted preparer manifest alone is not a
claim that the full model-selection protocol has been satisfied.

## 3. Scoring and aggregation rules

- Use candidate-specific pinned prompts/tokenizers, the actual ree Rust ONNX
  runtime, and 512 tokens **including prompts/special tokens** for both query and
  document inputs. Record truncation coverage before comparisons; no claim about
  long-context quality follows. Document chunking/multi-vector aggregation is not
  part of this test. Native dimensions first; Arctic 256 truncates and renormalizes
  the identical outputs/corpora as a separately identified recipe.
- Dense normalized dot product, top 10; score ties break by document ID ascending.
  Set `exclude_query_id=true` for BEIR, matching its usual self-match exclusion.
  For MIRACL/CoIR/ree, explicitly pin the setting after checking their ID namespace;
  use false unless identical IDs denote the same source passage. A positive qrel
  conflicting with exclusion is an error, not removed from the denominator.
- Recall@10 and MRR@10 consider grade >0 relevant. nDCG@10 uses **linear** gain
  `grade / log2(rank + 1)` at 1-based rank, with ideal ranking over all positive
  judgments. No exponential gain and no binarization of graded qrels. Legacy smoke
  corpora without `relevance` retain binary gain. Harness metric version 2 records
  this policy and exclusion setting; do not silently merge older tie-policy runs.
- Mean per-query nDCG within each task/language; macro-average the five general
  strata and four multilingual strata independently. Code/technical is 50% mean
  of the two code tasks and 50% ree technical. Recall/MRR and all per-domain
  results are diagnostics, never used to pick a favorable substitute metric.
- Evaluate CPU-query/CPU-document, GPU/GPU, CPU/GPU, and GPU/CPU combinations using
  the actual intended artifacts. Use the **minimum across those four combinations
  per stratum** in the weighted recipe score; report all four separately.
  Same-artifact FP16 CPU/GPU parity remains a separate runtime correctness test.
  Mixed-artifact vector export/scoring is implemented and measured for all five
  models in pilot v1; full-suite coverage and weighted aggregation remain pending.
- Throughput: three serial idle-machine process runs per artifact/device, five
  timed repeats after an untimed warm-up, lengths 64/256/512 and batches 1/8,
  fixed thread count and one recorded power mode for the entire comparison.
  For each device/length/batch cell use the median of the three within-process
  mean embeddings/second values. Divide by the fastest eligible candidate in
  that cell, then geometrically average all 12 ratios (CPU and CUDA equally).
  Larger saturation probes and quality elapsed time are diagnostics, not substitutes.
- Footprint score: smallest eligible footprint divided by candidate footprint.
  Footprint is unique required CPU+GPU model/tokenizer/sidecar bytes plus
  `1,000,000 * dimensions * 4` vector bytes. Count shared files once by hash.
  Common runtime binaries, SQLite overhead, startup, RSS, and VRAM are reported
  separately. Operational compatibility is an eligibility gate, not a subjective
  number added after seeing results.
- Final score in [0,1]: `0.45 * general + 0.20 * multilingual + 0.15 * code_technical
  + 0.15 * throughput + 0.05 * footprint`. Missing measurements are **missing**:
  no zeros, model-card substitutions, cross-hardware estimates, or redistributed
  weights. Unsupported provider/graph combinations need a documented resolution
  or explicit ineligibility decision before a winner can be frozen.

Decision thresholds fixed before scoring: flag any per-stratum nDCG loss >0.03
absolute against provisional Arctic 768 and any matched throughput cell >20%
slower. They require explicit review, even if the aggregate improves. CPU/GPU
mixed-artifact nDCG must stay within 0.01 absolute of the better homogeneous
combination in **every** stratum, or the fallback recipe needs resolution before
eligibility. Report mean top-10 overlap, top-1 agreement, and per-query metric
changes as diagnostics; cosine alone cannot satisfy this gate.

A replacement needs >0.01 absolute final-score improvement and a positive 95%
paired bootstrap lower bound on the weighted quality difference, otherwise keep
the eligible provisional recipe pending more evidence. Use 10,000 bootstrap
replicates, seed `20260909`, resample queries with replacement independently within
each stratum, share sampled query IDs across candidates, recompute the full
aggregation including four-combination minima, and use 2.5/97.5 percentile bounds.
The bootstrap implementation/RNG and replicate index hashes must be locked before
comparison. For a quality-equivalent Arctic 256 option (no stratum >0.01 lower,
weighted quality no more than 0.005 lower), permit a documented footprint-based
choice; it still needs production dimension/migration support. These are local
engineering decision thresholds, not a universal statistical significance claim.

## 4. Lock checklist / remaining implementation

Before any model-selection scoring:

- [ ] Reviewed revisions, per-dataset licenses, upstream/normalized file hashes,
  conversion code and negative-miner manifests for all twelve strata.
- [ ] Hand-reviewed ree technical corpus and qrels.
- [ ] Generated corpus hashes/selection audits, self-match settings, truncation
  coverage audit, and final versioned suite inventory committed.
- [x] Exact five-candidate CPU/GPU artifacts/manifests with model/tokenizer/sidecar
  hashes, upstream revisions and declared licenses, prompt/special-ID/output/pooling
  contracts, and local supported-provider evidence at the 512-token window.
- [ ] Review candidate export/license/batch-dependence limitations and establish
  retrieval compatibility before declaring any recipe eligible to ship.
- [x] Native export, mixed-artifact/index ranking and paired per-corpus bootstrap
  tooling implemented/tested; all five models have completed real two-corpus results.
- [ ] Complete all-model/full-suite measurements and weighted paired aggregation.
  Do not calculate the selection score from homogeneous or partial pilot output.
- [x] Capacity checked and serial idle-window speed runs recorded for all five
  candidates. No overlapping real inference, automatic installation or GPU resets.
  Recheck capacity/idle conditions when scheduling further full-suite quality runs.

Implemented now: deterministic offline preparation, streamed input hash checking,
judgment/negative preservation and validation, atomic new output, graded nDCG,
explicit self-match handling, deterministic ranking ties, pinned five-candidate
CPU/CUDA capability checks, and thirty repeated same-build speed runs. Two real
small corpora have passed preparation; ArguAna failed referential integrity and
published no corpus. The separately frozen pilot now supplies complete retrieval
and compatibility evidence for all five models on the two prepared corpora, with
all 20 exports and 12 six-way comparisons validated. Both power interruptions and
the earlier partial analysis remain preserved as historical evidence.
**The full suite, shipping compatibility and candidate winner remain unresolved.**

## Primary references

- BEIR input schema: <https://github.com/beir-cellar/beir/wiki/Load-your-custom-dataset>
- BEIR tasks and dataset-license disclaimer: <https://github.com/beir-cellar/beir>
- BEIR self-match policy: <https://github.com/beir-cellar/beir/blob/main/beir/retrieval/evaluation.py>
- MIRACL topics/qrels and negative judgments: <https://huggingface.co/datasets/miracl/miracl/blob/main/README.md>
- CoIR task definitions: <https://github.com/CoIR-team/coir>
- Linear-gain nDCG reference: <https://github.com/usnistgov/trec_eval/blob/main/m_ndcg_cut.c>

These reference pages explain formats/methodology, **not immutable data locks**.
Exact revisions and hashes must accompany the prepared suite and candidate files.
