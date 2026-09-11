# Arctic CLS-copy removal: offline implementation evidence

Date: **2026-09-10**. First patch from
[the optimization plan](../../../docs/arctic-optimization-plan-v1.md), not a
performance benchmark. No real model inference, download, release build, scheduler
policy change or retrieval-pilot rerun was performed. **No speedup is claimed.**

## Change

`src/model/onnx.rs` now passes borrowed f32/f16 output slices to a private pooling
helper. CLS copies/converts only each item's first-token values into its final
vector instead of materializing the full token-output tensor. Mean pooling reads
its contributing values directly, retaining f16 conversion, per-token division,
accumulation order and the unchanged `normalize_dense` implementation. Output
name selection, shape checks, finite/nonzero validation and error propagation
remain in place. Runtime-owned outputs and GPU transfers are not removed.

No model pins, prompts, tokenizer, precision artifacts, provider options, storage
contract, input packing or scheduler source were changed. Even this copy-only
change can affect timing-based automatic calibration; future comparisons must fix
actual inference batch shapes rather than assume identical `auto` choices.

## Verification sequence

1. Rechecked cwd `/home/jacob/Projects/ree`, branch `main`, ancestor/project
   instruction files (no `AGENTS.md` found), actual dirty state and source. All
   48 pilot-mapped source hashes and both release executable hashes matched.
2. Preserved 83 source/config/test/documentation files in a source archive and
   copied both frozen release examples before editing. The archive excludes
   benchmark datasets/results; existing pilot evidence was not modified.
3. Baseline: `cargo test --locked --offline --all-targets` — **102 passed, five
   ignored**. Python offline evidence tests — **19 passed**.
4. Extracted the old full-materialization algorithm into a private helper and
   ran seven fixtures before removing materialization: **seven passed**.
5. Removed full-tensor materialization; added a counted-conversion assertion and
   explicit f16 subnormal/extreme fixture. Full Rust suite — **111 passed, five
   ignored**. Python suite — **19 passed**. Clippy with warnings denied, formatting
   and diff whitespace checks passed.

Nine tests in `src/model/onnx/tests.rs` cover bitwise f32/f16 CLS and mean results
against the historical algorithm; singleton and multi-item shapes up to
8×512×768; unequal input lengths/padding; known normalized answers; wrong output
rank/batch/length/dimensions; zero/nonfinite failures after a valid item; signed
zero, subnormal and extreme finite values; and conversion of only retained CLS
values. They require no model/session or GPU. These fixture results do not prove
real-model equivalence or measure practical runtime benefit.

## Retained files and checkpoint

- `baseline-manifest.json`: pre-edit source hashes, git state, source archive hash,
  original release binary hashes and the local checkpoint path.
- Source archive and independent release executable copies:
  `/home/jacob/.local/state/ree/checkpoints/cls-copy-20260910T150822Z/`.
  This is a local checkpoint, not a commit or isolated worktree.
- `baseline-rust.log`, `baseline-python.log`: fresh pre-edit tests.
- `fixtures-before.log`, `fixtures-before-sha256.json`: seven fixtures against
  the extracted full-copy algorithm (intermediate hashes, not a separate archive).
- `after-rust.log`, `after-python.log`, `clippy.log`: post-change checks.
- `change.patch`: this continuation's source/test/documentation changes relative
  to the checkpoint, excluding pre-existing work and these evidence files.
- `verification.json`: final source/artifact hashes, checkpoint verification,
  unchanged frozen binaries and the exact pilot-source difference.

Commands used (ordinary ignored tests remain ignored):

```sh
cargo test --locked --offline --all-targets
cargo test --locked --offline --lib model::onnx::tests
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s benchmarks -p 'test_*.py' -v
cargo clippy --locked --offline --all-targets -- -D warnings
cargo fmt --all -- --check
git diff --check
```

Next work remains separate: injected cache/calibration and automatic-growth
regressions, then policy fixes, minimal stage counters and a short controlled
Arctic-only before/after measurement. Existing historical scheduler replay
assertions are observations, not desired policy tests. Do not overwrite frozen
pilot evidence or infer a speed improvement from test durations.
