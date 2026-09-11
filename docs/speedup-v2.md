# Speedup v2: reusable queries, startup, and gated experiments

Implemented September 10, 2026, after the user confirmed the other coding session
had finished. Existing uncommitted work was retained; builds use the normal target.
No model-selection run, new model download, installation, migration of personal
storage, CUDA Graphs, or TensorRT change is part of this work.

## Available now

```sh
# Persistent stdin/stdout session; no socket or mandatory daemon:
printf '%s\n' \
  '{"v":1,"id":"q1","op":"search","query":"writer lock timeout"}' \
  '{"v":1,"id":"q2","op":"search","query":"SQLITE_BUSY","mode":"lexical"}' \
  | ree search --stream

# Optional foreground worker. Default endpoint requires a safe XDG_RUNTIME_DIR:
ree worker --idle-timeout 300s
# In another terminal, explicitly route to it:
ree search "writer lock timeout" --socket "$XDG_RUNTIME_DIR/ree/worker.sock"
```

`ree worker --socket /absolute/private/directory/worker.sock` is the alternative
when XDG_RUNTIME_DIR is unavailable. Its parent must already exist, be user-owned,
and have no group/other permissions. The default creates a private `ree` directory
inside a verified private runtime directory. Socket mode is 0600; peer UID is
checked. A startup lock serializes stale-socket cleanup; regular files, symlinks,
foreign sockets, and live listeners are not overwritten. Exit removes only the
socket inode created by that worker. The lock file can remain.

Standalone search remains the default. There is no auto-start, automatic routing,
TCP listener, mandatory service, query history, or resident ingestion endpoint.
Socket CLI clients use the worker's fixed configuration, do not read local TOML,
and reject explicit database/device/config/batch/memory overrides, including their
REE_* environment forms. Use `./worker` to ingest a file named `worker`.

### Query ownership and idle release

Both front ends share the same bounded supervisor and query-only service. Exactly
one disposable engine child owns the tokenizer/model/runtime. It is reused for
queries, never shared concurrently or with the ingestion scheduler. Each request
opens a fresh read-only database connection, validates scope/recipe, releases its
initial snapshot before encoding, then revalidates and ranks/hydrates in one fresh
snapshot. Compatible generation activation is allowed; incompatible recipes fail.
No writer lock, long-lived idle read transaction, or stale connection survives
between requests. Replacing a database path becomes visible on the next request.

The default idle timeout is 300 seconds; `--idle-timeout` accepts 1..86400 seconds
with an optional `s` suffix. It applies to streaming as well as sockets. Connected
clients and lexical traffic do not renew dense-model residency. Idle expiry exits
and reaps the **whole child**, releasing CUDA context/arena allocations rather than
assuming session Drop releases VRAM. The socket/stdin supervisor stays alive;
next use reloads and re-verifies artifacts/device activation. Lexical and empty
scopes never initialize the model. Failed runtime calls invalidate their scheduler.

Cancellation of executing work terminates/reaps the child; next execution reloads.
This is process isolation, not an assertion that ONNX calls can be safely preempted.
Queued work is cancelled without executing it. Full disconnect cancels the active
consumer's work; half-close/EOF drains accepted input. SIGINT/SIGTERM stop admission,
cancel work, drain bounded output for up to ten seconds, and clean up the endpoint.
A private Linux parent-death signal prevents orphan engine processes after a hard
supervisor exit. No library/GPU reset or daemon installation is performed.

### Version 1 JSONL protocol

One newline-terminated UTF-8 object per frame. `v`, `id`, and `op` are required.
Search fields are `query`, optional `mode` (semantic default), `limit` (10 default),
`root`, and `media_type`. Unknown fields/operations are rejected. Dense/lexical
query validation remains the existing retrieval contract.

```json
{"v":1,"id":"q1","op":"search","query":"writer lock timeout","mode":"hybrid","limit":10}
{"v":1,"id":"cancel-1","op":"cancel","target":"q1"}
```

Events carry `v`, `id`, and `type`; child events also report `engine_pid`. Search
hits/completion retain existing fields, including provider/generation/warnings.
Accepted execution requests are FIFO within each connection and round-robin across
connections. Result frames for an individual search stay contiguous. Control,
validation, overload, and cancellation responses can precede earlier search
completion and must be correlated by ID, not line position.

A healthy connection gets one terminal `search_completed` or `request_failed` per
accepted search. `cancel_completed` terminates a cancellation operation and reports
`cancelled` or `not_pending`; it never adds a second terminal result to a completed
search. IDs are unique while outstanding. No exactly-once receipt across disconnects
or cross-connection deduplication is promised. Request failures leave the connection
usable; malformed/oversized/partial-final frames emit `connection_failed` where
possible and close it. Earlier accepted work can still drain. Duplicate outstanding
IDs are connection errors rather than ambiguous second completions.

Bounds:

- 128 KiB including LF per frame; existing 64-KiB query bound still applies.
- IDs: 1..128 UTF-8 bytes, no control characters. Filters: at most 4096 bytes each.
- Eight clients, two queued searches each, one globally executing search. Stdin
  admission backpressures; full socket request queues return retryable `busy`.
  One bounded look-ahead frame allows cancellation with a full stdin search queue.
  An extra search stays buffered; controls behind over-pipelined search frames
  naturally wait for byte-stream admission.
- At most 1 MiB of raw hydrated text/metadata per request, checked against SQLite
  bytes **before** copying/JSON decoding; at most 8 MiB serialized response.
- Per-client and aggregate output credits include retained packets even after a
  prefix has been written. Reserve one response for the child; at most 16 MiB of
  retained serialized response payload across supervisor and child, plus bounded
  framing/JSON/object overhead. This is not a whole-process RSS quota.
- Ten-second incomplete-frame/blocked-output timeout. Excessive slow-consumer
  buffering disconnects that consumer. Thirty-minute hard execution deadline,
  including first-use provisioning; shorter cancellation is available.

An oversized response fails with `response_limit`; it is never silently truncated.
These transport limits do not change standalone search's existing output bounds.
Protocol modes reject `--quiet` and `--progress` so terminal frames cannot disappear.
Explicit socket CLI queries retain ordinary quiet/output/error behavior. Streaming
exit: 0 all successful, 1 request failures/cancellation, 2 framing/startup arguments,
3 unrecoverable IO/service error. A graceful worker signal exit is 0; request errors
are reported to clients. Diagnostics remain on stderr.

## Default startup, allocation, and scheduler changes

- Explicit `cuda`/`cuda:N` never provisions/hashes CPU rescue weights. `cpu` and
  `auto` retain eager CPU provisioning; a later auto fallback re-verifies weights.
  CUDA preflight-before-download and real CUDA verification warmup remain intact.
- Input arrays move into ONNX tensors rather than being cloned. Only the requested
  token output is fetched. Existing CLS host-copy removal remains unchanged.
- Successful document vectors are borrowed by the writer; per-document error
  retries retain their own vectors. Insert statements are cached. Transaction,
  synchronization, FTS, pruning, failure, and document atomicity semantics remain.
- Blind post-success growth is removed on CPU and GPU. Interactive queries stay
  singleton and calibration/profile-free. Ingestion calibrates within a bounded
  candidate envelope; cached item/token hints are constrained to an actual probe.
  Splits/OOM/low-memory reductions cannot restore an unvalidated larger limit.
  Profiles use `bounded-calibration-v2`, including runtime graph variant identity;
  old profiles are not trusted under the new policy. Explicit caps and OOM
  reductions remain. CPU grouping changes beyond the old growth threshold remain
  part of the already documented batch-dependence limitation, not vector neutrality.

`REE_TIMING=1` adds bounded aggregate `stage_metrics` with integer nanoseconds and
call counts. Scopes identify command/request/preparation threads; nested spans and
concurrent service times **overlap**, so do not sum them as wall time. Timed stages
include config/DB open, artifact verification/download, tokenizer load/tokenization,
device discovery/preflight, graph/session load, verification warmup/profile,
calibration/workload embedding, tensor packing, runtime call including transfers,
pooling, hydration and document writes. `queue_timing` reports transport queue wait.
No query/token text is added to telemetry. The legacy inclusive/truncated
`inference_ms` is retained; new scheduler counters separate calibration/workload
nanoseconds/calls and attempted real/padded tokens (including retries).

## Experiments: disabled by default

### `REE_PIPELINE_OVERLAP=1`

Only explicit CUDA, 512-token windows and overlap <=64 qualify. A shared lazy
immutable tokenizer can prepare changed documents in the two existing extraction
workers while the consumer embeds/writes. Workers stage chunks only for extracted
text <=16 KiB and retain at most 256 KiB of prepared chunk/token payload per result;
larger cases keep consumer tokenization. Queues stay at two entries each, pending
inference groups at 32 documents/~8 MiB, with existing file/extractor limits. Hash
no-ops never initialize the tokenizer. No new worker pool or concurrent runtime
is introduced; writer ownership remains single-threaded.

**CPU and `auto` explicitly cannot enable this path.** A CPU trial changed group
composition and failed the frozen equivalence gate (minimum cosine 0.98349,
maximum component difference 0.02659). CUDA comparisons passed the numerical
fixture gate, but sustained-throughput timing was unstable. Therefore this remains
an experiment, not a faster default or a universal hard large-document memory bound.
Separate disk-backed large-document staging and a writer-queue redesign are deferred.

### `REE_CLS_OUTPUT=1`

Creates/reuses `derived/arctic-cls-gather-v1/<parent-sha>/model.onnx` under the cache.
A bounded streaming wire transform preserves the pinned graph, replaces its outputs,
and appends scalar-zero Gather on `token_embeddings`, returning **unnormalized**
f32 `[batch,768]`. Host L2 arithmetic stays unchanged. The graph's existing
`sentence_embedding` is not substituted: it includes graph-side normalization.

Both parent and deterministic derivative digests are verified; output publication
is locked/atomic and has a manifest. CUDA warmup additionally requires the new
Gather to execute on CUDA. Original artifacts are untouched. `runtime_artifact`
events identify the derivative separately from the unchanged logical model recipe:

| Parent | Derivative SHA-256 |
|---|---|
| CPU INT8 | `27b7e290193aa20738c81753b1f82a0c8d44ac03fe4bd9cab019eb52218be366` |
| CUDA FP16 | `e02d9d888ac67d028720b904afd49f04c54d9d0bcc8a88c7ae95a0412ab113c8` |

`benchmarks/derive-cls.py` independently reproduces the byte-exact derivative without
installing ONNX. `examples/qualify_cls.rs` checks real same-provider vectors at
batches 1/8/16, lengths 64/256/512, including padding, on the development host only.
These CPU/CUDA checks were bitwise equal. They do not qualify every supported long
shape, corpus, or provider combination. Warm short-query end-to-end latency did not
improve; cold start paid an additional derivative checksum. Runtime traces verified
CUDA Gather placement but did not expose physical transfer-byte counts (reported
as unmeasured, not inferred from empty transfer-event lists). Thus default enablement
is **no-go** pending a reproducible affected-workload benefit and complete transfer
qualification. Do not call 512-fold fewer output elements a 512-fold speedup.

Preoptimized graph caching, shorter activation warmup, and I/O binding were **not
implemented speculatively**. The checksum/trust and practical-benefit gates have
not been cleared. Full writer/pipeline restructuring is likewise not a default
change justified by the unstable sustained-ingestion measurements.

## Evidence and remaining qualification

See `benchmarks/results/speedup-20260910/README.md` for measured results, frozen
build identity, limitations, and go/no-go decisions. Offline tests cover framing,
reuse/idle child exit, queue pressure, cancellation, crash/reload, socket ownership,
stale/live sockets, shutdown, hydration limits, repeated DB observations, scheduler
cache/growth regressions, preparation overlap/no-op, and CLS wire/pooling contracts.
Existing retrieval, ingestion, durability/crash, and fallback tests remain required.
Real-model measurements are serial/local, separate from ignored CI tests.

The selected Arctic recipe remains provisional: none of these changes clears
mixed INT8/FP16 ranking compatibility, every possible batch shape, hostile helper
sandboxing, million-chunk latency, or beta release/packaging requirements. Further
contention/slow-client soak, multi-chunk/large-document stress, full resource
telemetry, and stable sustained throughput remain explicit qualification work.
