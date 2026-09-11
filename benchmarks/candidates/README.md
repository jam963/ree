# Pinned candidate inputs — 2026-09-09

**Historical candidate-qualification evidence.** On 2026-09-10 the user selected
Arctic M v2 at 768 dimensions for beta and deferred further comparative evaluation;
see [the decision and exact pins](../../docs/model-decision-v1.md). The choice does
not turn these limited measurements into full qualification. Five candidates
have pinned, downloaded and verified artifacts, tokenizer/pooling/special-ID checks,
and actual CPU/CUDA capability tests through ree's Rust ONNX implementation.
Repeated throughput baselines pass. The subsequent [completed retrieval pilot](../results/retrieval-pilot-20260909/README.md)
covers all five candidates on NFCorpus/SciFact, with all 20 encodings and 12 six-way
comparisons verified. SciFact flags some Arctic/E5 mixed-recipe losses and Arctic-256
regressions. Granite/BGE same-FP32 mixed losses stay below the point cutoff; Jina's
mixed checks also pass, but absolute quality on these two tasks is low and code
retrieval remains unmeasured. Passing the point cutoff does not prove equivalence.
Full-suite retrieval quality and universal CPU/GPU artifact compatibility remain
unproven; further comparative evaluation is deferred for this beta choice. License/
notice review remains release work. The beta model choice itself is now settled.

| Candidate | Exact revision | CPU artifact | CUDA artifact | Dense recipe |
| --- | --- | --- | --- | --- |
| Arctic M v2 | `95c2741480856aa9666782eb4afe11959938017f` | INT8 | FP16 | CLS, 768; query prefix `query: ` |
| Granite 107M multilingual | `d6cffd338414d6a1c1f5decfad5fec62eebc90d5` | FP32 | same FP32 | CLS, 384; no prefixes |
| multilingual-e5-small | `614241f622f53c4eeff9890bdc4f31cfecc418b3` | official AVX512-VNNI INT8 export | O4 FP16 export | masked mean, 384; `query: ` / `passage: ` |
| BGE-M3 | `5617a9f61b028005a4858fdac845db406aefb181` | FP32 | same FP32 | CLS, 1024; no prefixes; dense only |
| Jina v2 Base Code | `516f4baf13dec4ddddda8631e019b5737c8bc250` | INT8 | FP16 | masked mean, 768; no prefixes |

All five tokenizers use CLS/PAD/SEP IDs 0/1/2. In particular, do not substitute
E5/Jina architecture config `pad_token_id=0` for their actual tokenizer PAD ID 1.
All vectors are L2-normalized in ree. Mean pooling includes the unmasked special
and prompt tokens. The fixed evaluation window is 512 tokens, not each model's
advertised maximum. These are concrete exported artifact choices, not claims
that the best possible precision/export has been found for each model.

## Inventory and evidence

- `artifacts.json`: complete verified download lock, exact source URLs/revisions,
  sizes and SHA-256 values. Small Git-tracked files also retain upstream Git blob
  IDs. Contains 4,537,649,722 bytes including model cards/configs and two BGE
  tokenizer variants; not all downloaded evidence files are required at runtime.
- `metadata-lock.json`, `weights-lock.json`: original pre-download upstream size/
  hash inventories. These document the bootstrap from immutable Git/LFS metadata;
  subsequent provisioning can use the complete SHA-256 lock above.
- `recipes.json`: executable candidate manifests nested under each candidate's
  `cpu`/`cuda` keys; all artifact paths are relative to a materialized manifest.
- `graph-inspection.json`: observed token input/output names, element types,
  dimensions, operators and external-data dependencies. Generated without loading
  any model-supplied Python by `benchmarks/inspect-onnx.py`. This narrow inspector
  is not a full ONNX validator or a security sandbox. Unsupported training,
  function or sparse-tensor constructs fail inspection rather than being ignored.
- `../results/candidate-preflight-20260909/`: actual Rust tokenizer/ONNX/provider
  checks, including mixed-length and individual inference. Ten combinations pass
  load/shapes/finite-unit-vector checks, **not full retrieval correctness gates**.

Granite's reviewed token output is named `logits`, but is a rank-3 384-dimensional
hidden-state tensor, not a vocabulary prediction. The qualification recipe names
it explicitly and the runtime verifies its full shape. Production output selection
is unchanged. BGE's graph references `model.onnx_data`; the candidate manifest
hashes that sidecar before session creation. The separate upstream
`Constant_7_attr__value` file is inventoried but not referenced by this graph.
No automatic output-name guessing, graph rewriting, export or quantization was done.

## Reproduce local provisioning and runs

From the repository root, choose an existing or new private SSD scratch path:

```sh
scratch="$HOME/.cache/ree-eval/my-candidate-run"
python3 benchmarks/fetch-artifacts.py --lock benchmarks/candidates/artifacts.json \
  --scratch "$scratch" --max-bytes 8589934592
python3 - "$scratch" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
for candidate in json.loads(pathlib.Path('benchmarks/candidates/recipes.json').read_text())['candidates']:
    for device in ['cpu', 'cuda']:
        with (root / f"{candidate['id']}-{device}.json").open('x') as f:
            json.dump(candidate[device], f, indent=2, sort_keys=True)
            f.write('\n')
PY
cargo build --release --locked --example qualify
# Run only on the development host, on AC, while otherwise idle.
# --power-mode records/checks the existing mode; it does not change it.
python3 benchmarks/run-baseline.py --output /path/to/new-results-directory \
  --power-mode balanced --cpu-manifest "$scratch/granite-cpu.json" \
  --cuda-manifest "$scratch/granite-cuda.json"
```

The fetcher uses only the Python standard library and executes no model code.
It checks scratch capacity, exact byte counts and hashes, uses bounded streaming
reads and no-clobber publication, and reuses only verified existing files. The
explicit byte cap covers the files in the supplied lock; it is not a filesystem
quota. Failed downloads remove their temporary files. Do not run provisioning,
builds or other measurements concurrently with timed inference.

For capability checks without speed/quality scoring:

```sh
target/release/examples/qualify --manifest "$scratch/granite-cuda.json" \
  --device cuda:0 --power-mode balanced --load-repeats 1 --validate-only
```

The harness verifies declared sidecars, not arbitrary unknown external graph
references automatically. Review the static dependency inventory against every
manifest before loading a new export. Offline tests check that the five committed
recipes declare exactly the external files found by inspection.

## Important observed limitation

Mixed-length batching versus individual inference changed INT8 vectors in the
capability fixture: minimum cosine approximately 0.9754 Arctic, 0.9891 E5 and
0.9825 Jina. The fixture includes an empty string as well as multilingual/code
inputs, so these minima are not estimates of normal-document retrieval drift.
FP16/FP32 combinations were approximately 0.999998 or higher. Identical-input,
identical-shape repeats in the throughput runs showed no vector changes.

Batch dependence must be assessed with real rankings and the eventual batching/
fallback policy. Do not confuse same-shape repeatability, cross-batch compatibility,
CPU/GPU same-artifact parity, and cross-artifact quantization compatibility.
See the [evaluation protocol](../../docs/evaluation-protocol.md).
