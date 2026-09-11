# Beta model decision v1 — Arctic M v2, 768 dimensions

**Decision date: 2026-09-10. Status: selected for the beta by explicit user decision.**
The user chose to ship Arctic rather than spend more time on broader model
selection. Keep the existing pinned Arctic recipe; do not run additional
comparative model/corpus evaluations unless requested. This closes the beta's
model-choice question by a product decision, **not by passing the full evaluation
protocol or declaring every compatibility check successful**.

This decision supersedes the requirement to finish the twelve-stratum comparison
before choosing the beta model in [the roadmap](beta-roadmap.md) and
[evaluation protocol v1](evaluation-protocol.md). The measured results, thresholds,
failed checks and missing measurements are unchanged. Ordinary correctness,
security, license/notice and packaging work is not waived by this model decision.
No release has been built, installed, tagged or published by recording it.

## Frozen beta recipe

The implementation already uses this recipe, so selection requires no model
replacement, database migration or re-embedding:

| Field | Selected value |
| --- | --- |
| Model | `Snowflake/snowflake-arctic-embed-m-v2.0` |
| Revision | `95c2741480856aa9666782eb4afe11959938017f` |
| Model key | `arctic-m-v2-95c27414-cls-l2-768-ort128-strict-v1` |
| Dimensions / storage | 768, normalized f32 vectors |
| Pooling / normalization | CLS / L2 |
| Document prompt | Empty |
| External query-vector prompt | `query: ` |
| CLS / PAD / SEP IDs | 0 / 1 / 2 |
| CPU artifact | `onnx/model_int8.onnx`, INT8 |
| CUDA artifact | `onnx/model_fp16.onnx`, FP16 |
| Runtime | ONNX Runtime 1.28.0; TF32 disabled, strict skip-layer normalization |
| Default chunking | 512 total model tokens including special tokens; 64 content-token overlap |
| Provider behavior | Existing `auto` selection/fallback retained; explicit CUDA refuses CPU fallback |

Artifact locks, matching `src/model/download.rs` and the completed pilot:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| Tokenizer | 17,083,009 | `f1cc44ad7faaeec47241864835473fd5403f2da94673f3f764a77ebcb0a803ec` |
| CPU INT8 weights | 310,916,060 | `03d923bb1850ebdccb068e2f3abd8aa43fe81c50d07d037ef103fe3d0fb78e3b` |
| CUDA FP16 weights | 613,266,244 | `f27ab40ab6e230265ba49a202a37f1ad031556256cbbc105d0ca9c0bdc7ec42e` |

Both artifacts plus the shared tokenizer total 941,265,313 bytes (897.7 MiB),
excluding runtime libraries and stored document vectors. Arctic 256 is **not**
selected. Existing configurable chunk limits are unchanged; the pilot evaluated
only the 512-token window, not the metadata's 8,192-token supported maximum.

The serialized recipe's existing `qualification: "provisional"` remains unchanged:
it describes incomplete empirical qualification, not an undecided beta model.
Do not relabel it as fully qualified or change the model key merely to record this
product choice. No production code, schema or frozen pilot executable is changed.

## Evidence supporting the choice

The [completed pilot](../benchmarks/results/retrieval-pilot-20260909/README.md)
contains 20 validated encoding jobs, 12 six-way comparisons, 7,200 query rows and
10,000 paired query bootstrap replicates per corpus. Arctic has the highest
homogeneous nDCG@10 of the five tested candidates on those two corpora:

| Corpus | CPU / CPU | CUDA / CUDA |
| --- | ---: | ---: |
| NFCorpus | 35.76 | 36.71 |
| SciFact | 72.00 | 72.86 |

Values are nDCG points on a 0–100 display, not published full-dataset benchmarks.
The [same-build speed evidence](../benchmarks/results/README.md) and measured
footprint inform the decision, but no missing quality stratum is replaced by a
speed or size score. No full-suite weighted winner was calculated.

## Known limitations carried into the beta decision

- **Mixed INT8/FP16 compatibility is not cleared.** SciFact CUDA-query/CPU-index
  lost 1.19 nDCG points against the better homogeneous reference, above the frozen
  1.00-point cutoff. The paired 95% interval is −0.46..3.35: uncertain, not a pass.
  CPU-query/mixed-index lost 0.98 points (−0.71..3.05); its passing point estimate
  is not an equivalence guarantee. Provider fallback can leave mixed-artifact
  vectors in an index. Retaining it is a beta product tradeoff, not a claim of
  identical rankings or precision-independent quality.
- **Batch dependence remains.** Native Arctic CPU/CUDA document cosine reached
  0.91572, below the historical 0.94 smoke bound; singleton-versus-batch CPU
  vectors also drift. Functional fallback/atomicity and retrieval compatibility
  are separate properties. Do not hide these findings or relax their thresholds.
- **Broader quality is unmeasured locally.** Web/financial/argumentative,
  multilingual, code and reviewed ree-specific retrieval strata remain incomplete.
  The user deliberately deferred them for this beta selection; they are not
  completed checklist items and are no longer prerequisites to choosing its model.
- **No postmeasurement repair or tuning.** Preserve the pilot's corpora, qrels,
  artifact recipes, bootstrap draws, raw results and interrupted attempts. ArguAna's
  five dangling qrel IDs remain a documented source issue, not silently dropped
  judgments. Deferred evaluation may be revisited under an explicit new plan.

## Effect on the remaining work

Stop comparative model selection and use the pinned Arctic 768 recipe for the
remaining beta work. The roadmap's selected-model performance work and remaining
large-document, extraction, security, notice and installation work retain their
existing scope; this decision does not initiate another benchmark run or waive
unrelated release safeguards. Historical September 9 reports retain their
predecision qualification status and must not be rewritten as proof of a full
model-selection win.
