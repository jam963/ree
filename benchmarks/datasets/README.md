# First real corpus preparation — 2026-09-09

These are preparation artifacts, **not retrieval results or a locked full suite**.
Text stays in local scratch; only source locks, selected IDs and audits are recorded
here. Declared distribution licenses are retained as evidence; upstream passage
rights/redistribution review remains pending rather than assuming every underlying
source inherits a benchmark repository's license.

| Dataset | Preparation | Selected queries | Competing documents |
| --- | --- | ---: | ---: |
| NFCorpus | Passed, full small corpus, graded qrels retained | 100 | 3,633 |
| SciFact | Passed, full small corpus | 100 | 5,183 |
| ArguAna | **Blocked: five qrel document IDs missing from the source corpus** | No output published | Not qualified |

The query seed is `20260909`. No positive/zero judgments were removed to fit the
budget. These complete small corpora need no mined subset distractors. Full
selection manifests, eligible counts, selected query IDs, output hashes and the
five missing ArguAna judgments are in `small-beir-prepared.json`.

`small-beir-lock.json` pins the original compressed JSONL distributions and
separate test-qrel revisions. The pre-Parquet revisions were deliberately chosen
to permit byte-preserving gzip decoding without installing conversion libraries
or running dataset-provided scripts; they are **not claimed to be the newest
revisions**. `small-beir-verified-lock.json` adds observed SHA-256 values for the
small Git-tracked files. Original repository revision/size/blob identities are
preserved. No undocumented content transformation or qrel filtering took place.

Reproduce (repository root; private scratch; no inference):

```sh
scratch="$HOME/.cache/ree-eval/my-candidate-run"
python3 benchmarks/fetch-artifacts.py --lock benchmarks/datasets/small-beir-lock.json \
  --scratch "$scratch" --max-bytes 33554432
cargo build --release --locked --example prepare_corpus
python3 benchmarks/prepare-small-beir.py --scratch "$scratch" \
  --output-dir "$scratch/prepared-small-beir-v1"
```

The final command currently returns **1** because ArguAna fails referential
integrity, while retaining valid NFCorpus/SciFact outputs and a structured error
audit. Its output directory must be new. Gzip decoding is limited to 128 MiB per
file and uses temporary no-clobber publication. The Rust preparer owns query
selection and final corpus validation/publication.

Do not repair ArguAna by silently discarding the five judgments or choosing easier
queries. Reconcile upstream distributions/versions first; if the source issue is
unavoidable, document and version an explicit premeasurement policy. A later,
separately preregistered [completed retrieval pilot](../results/retrieval-pilot-20260909/README.md)
scored all five candidates on the prepared NFCorpus/SciFact files without changing
these hashes or judgments. BGE's tiny provider ordering differences exposed
identical-text documents with different IDs and ID-specific judgments; no
postmeasurement deduplication or qrel repair was applied. The remaining NQ/FiQA,
MIRACL and code inputs still need pins/adapters/mining, plus the reviewed ree-specific set and the
full [suite lock](../../docs/evaluation-protocol.md).
