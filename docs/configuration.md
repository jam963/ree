# Configuration

Use `--config PATH` / `REE_CONFIG` to select a TOML file. An explicitly selected
missing file is an error; a missing default file is normal. Unknown TOML fields
are rejected. Relative XDG environment paths are ignored per the XDG spec.

```toml
# ~/.config/ree/config.toml
# db = "/absolute/path/ree.db"
device = "auto"                # cpu, cuda, cuda:N
chunk_size = 512                # 3..8192, includes two special tokens
overlap = 64                    # strictly less than chunk_size - 2
# batch_size = 8                # positive; disables growth, not OOM reduction
gpu_memory_fraction = 0.85      # >0 and <=0.95
max_file_size = "64MiB"         # positive integer; B/KiB/MiB/GiB (KB/MB/GB aliases)

[metadata]
project = "notes"

[extractors.epub]
extensions = ["epub"]
command = ["pandoc", "{path}", "-t", "plain"]
timeout_seconds = 60
max_output_bytes = 67108864
```

CLI > environment > file > defaults. Supported option environment variables:
`REE_DB`, `REE_CONFIG`, `REE_DEVICE`, `REE_BATCH_SIZE`,
`REE_GPU_MEMORY_FRACTION`, `REE_CHUNK_SIZE`, `REE_OVERLAP`, `REE_MAX_FILE_SIZE`.

Metadata precedence: extractor metadata, remote-source metadata, TOML metadata,
`--metadata-json` object, then repeated `--metadata KEY=VALUE`. `KEY=VALUE`
values are strings; use JSON to preserve numbers, booleans, arrays, and objects.
A processing recipe hash includes configured metadata and extractor settings,
so changes invalidate the unchanged fast path.

`--extractor text|html|notebook` forces a native extractor. Other names select a
configured extractor or built-in `pdf`, `docx`, `odt`, `epub`, or image extractor.
Commands are arrays, with literal `{path}` replacement in each argument. There
is no shell interpolation. Helpers receive a private immutable file snapshot,
not a source that can change during conversion. stdout/stderr are drained in
parallel; output size and wall-clock limits terminate the process group.

Current fixed resource bounds: two extraction workers, discovery/result queues
of two items each, up to 32 pending documents or approximately 8 MiB of token/text
payload before inference, 30-second HTTP requests, five redirects, 120-second Git
commands, 512 MiB Git checkouts, 60-second built-in extractors, and 64 MiB helper
stdout. One large document is still held in memory and is bounded by the file or
extractor limit. These are not yet all configurable; see implementation status.

`--allow-private-network` is an explicit per-invocation exception to URL/Git IP
protection. Ambient HTTP proxies are disabled for untrusted source fetches.
Model downloads use HTTPS with pinned hashes; standard proxy settings apply to
those trusted artifact hosts.

## Scheduled rebuild

```cron
# Adjust executable path and schedule to your installation.
0 3 * * 0 /home/me/.local/bin/ree rebuild >> /home/me/ree-rebuild.jsonl 2>> /home/me/ree-rebuild.stderr
```

Or a user systemd service:

```ini
[Unit]
Description=Rebuild local ree embeddings
[Service]
Type=oneshot
ExecStart=%h/.local/bin/ree rebuild
```

Pair with a user timer using `OnCalendar=weekly` and `Persistent=true`. A rebuild
resumes an incomplete generation on its next invocation; a competing writer
returns exit code 4 after a short wait. Rebuild does not reopen sources. Use an
explicit ingestion with `--force` to extract/rechunk sources under a new policy.
