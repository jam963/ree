#!/usr/bin/env python3
"""Exercise an installed beta with isolated data; optional serial development-host CPU/CUDA checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new evidence directory, never an existing index")
    parser.add_argument("--models", type=Path, help="existing ree models directory; copy it, then test CPU/CUDA serially")
    args = parser.parse_args()
    binary = (args.prefix / "bin/ree").absolute()
    real_binary = binary.resolve().parent / "ree"
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    if args.models:
        if os.environ.get("CI") or "AMD Ryzen 9 8945HS" not in Path("/proc/cpuinfo").read_text():
            parser.error("real inference smoke is restricted to the development host, never CI")
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True)
        if "RTX 4070 Laptop" not in gpu:
            parser.error("development GPU required")
    # Logs/summary persist, all generated sources, models and databases are disposable.
    with tempfile.TemporaryDirectory(prefix=".work-", dir=output) as temporary:
        work = Path(temporary)
        env = {k: v for k, v in os.environ.items() if not k.startswith(("REE_", "ORT_", "LD_", "CUDA_"))}
        env.update({"HOME": str(work / "home"), "XDG_CONFIG_HOME": str(work / "config"),
                    "XDG_DATA_HOME": str(work / "data"), "XDG_CACHE_HOME": str(work / "cache"),
                    "XDG_RUNTIME_DIR": str(work / "runtime"), "CUDA_VISIBLE_DEVICES": "",
                    # Fail instead of silently provisioning missing model artifacts.
                    "HTTPS_PROXY": "http://127.0.0.1:9", "HTTP_PROXY": "http://127.0.0.1:9",
                    "ALL_PROXY": "http://127.0.0.1:9", "NO_PROXY": ""})
        (work / "runtime").mkdir(mode=0o700)
        results = []

        def invoke(label, arguments, expected=0, exe=binary, extra=None, stdin=None):
            result = subprocess.run([str(exe), *map(str, arguments)], env={**env, **(extra or {})},
                                    cwd=work, input=stdin, capture_output=True, text=True, timeout=240)
            (output / f"{label}.stdout").write_text(result.stdout)
            (output / f"{label}.stderr").write_text(result.stderr)
            if result.returncode != expected:
                raise RuntimeError(f"{label}: exit {result.returncode}, expected {expected}; inspect {output}")
            results.append({"label": label, "exit": result.returncode})
            if arguments == ["--version"]:
                return result.stdout.strip()
            return [json.loads(line) for line in result.stdout.splitlines() if line]

        version = invoke("version", ["--version"])
        invoke("doctor-empty", ["doctor"])
        invoke("migrate", ["migrate"])
        empty = invoke("empty-lexical", ["search", "nothing", "--mode", "lexical"])
        assert not any(e["type"] == "search_result" for e in empty)
        invoke("stream-empty", ["search", "--stream"], stdin='{"v":1,"id":"one","op":"search","query":"nothing","mode":"lexical"}\n')
        # No model artifacts or CUDA providers adjacent to this binary: CPU-only startup.
        cpu_dir = work / "cpu-only"
        cpu_dir.mkdir()
        cpu_binary = cpu_dir / "ree"
        shutil.copy2(real_binary, cpu_binary)
        invoke("cpu-only-doctor", ["doctor"], exe=cpu_binary)
        if args.models:
            models = work / "cache/ree/models"
            models.parent.mkdir(parents=True)
            subprocess.run(["cp", "-a", "--reflink=auto", str(args.models.absolute()), str(models)], check=True)
            before = {str(p.relative_to(models)): sha(p) for p in models.rglob("*") if p.is_file()}
            for device, exe, extra in [
                ("cpu", cpu_binary, {"CUDA_VISIBLE_DEVICES": ""}),
                ("cuda", binary, {"CUDA_VISIBLE_DEVICES": "0"}),
            ]:
                docs = work / f"documents-{device}"
                docs.mkdir()
                (docs / "a.txt").write_text("Oranges are citrus fruit. The orchard grows oranges.")
                (docs / "b.md").write_text("SQLite stores local documents and vectors atomically.")
                db = work / f"{device}.db"
                common = ["--db", db, "--device", device, "--batch-size", "2"]
                initial = invoke(f"{device}-ingest", [*common, docs], exe=exe,
                                 extra={**extra, **({"LD_DEBUG": "libs"} if device == "cuda" else {})})
                assert any(e["type"] == "runtime" and e["provider"] == device for e in initial), initial
                assert not any(e["type"] == "device_fallback" for e in initial)
                if device == "cuda":
                    trace = (output / "cuda-ingest.stderr").read_text()
                    expected_library = str(binary.resolve().parent / "libonnxruntime_providers_cuda.so")
                    assert f"calling init: {expected_library}" in trace, "CUDA provider not loaded from installed release"
                noop = invoke(f"{device}-noop", [*common, docs], exe=exe, extra=extra)
                assert noop[-1]["unchanged"] == 2 and not any(e["type"] == "runtime" for e in noop)
                if device == "cpu":
                    # Actual model inference from the provider-free copy with default auto policy.
                    auto = invoke("cpu-only-auto", ["--db", db, "search", "oranges", "--device", "auto"],
                                  exe=exe, extra=extra)
                    assert any(e["type"] == "runtime" and e["provider"] == "cpu" for e in auto)
                    invoke("cpu-only-strict-cuda-refused", ["--db", db, "search", "oranges", "--device", "cuda"],
                           expected=3, exe=exe, extra=extra)
                for mode in ["semantic", "lexical", "hybrid"]:
                    events = invoke(f"{device}-{mode}", [*common, "search", "oranges", "--mode", mode], exe=exe, extra=extra)
                    assert any(e["type"] == "search_result" and "oranges" in e.get("text", "").lower() for e in events), events
                streaming = invoke(f"{device}-stream", [*common, "search", "--stream"], exe=exe, extra=extra,
                                   stdin='{"v":1,"id":"one","op":"search","query":"oranges"}\n{"v":1,"id":"two","op":"search","query":"SQLite"}\n')
                terminal = [e for e in streaming if e["type"] == "search_completed"]
                assert len(terminal) == 2 and terminal[0]["engine_pid"] == terminal[1]["engine_pid"]
                socket = work / "runtime" / f"{device}.sock"
                with (output / f"{device}-worker.stdout").open("w") as stdout, (output / f"{device}-worker.stderr").open("w") as stderr:
                    worker = subprocess.Popen([str(exe), *map(str, common), "worker", "--socket", str(socket)],
                                              env={**env, **extra}, cwd=work, stdout=stdout, stderr=stderr)
                    try:
                        deadline = time.monotonic() + 10
                        while not socket.exists():
                            if worker.poll() is not None or time.monotonic() >= deadline:
                                raise RuntimeError("installed worker did not create its socket")
                            time.sleep(0.02)
                        for query in ["oranges", "SQLite"]:
                            response = invoke(f"{device}-socket-{query}", ["search", query, "--socket", socket], exe=exe, extra=extra)
                            assert any(e["type"] == "search_result" for e in response)
                    finally:
                        worker.terminate()
                        try:
                            worker.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            worker.kill()
                            worker.wait(timeout=5)
                            raise RuntimeError("installed worker shutdown timed out") from None
                    assert worker.returncode == 0 and not socket.exists(), "unclean worker shutdown"
                (docs / "a.txt").write_bytes(b"\0binary replacement")
                (docs / "b.md").unlink()
                invoke(f"{device}-failed-file-and-delete", [*common, docs], expected=1, exe=exe, extra=extra)
                retained = invoke(f"{device}-preserved", [*common, "search", "oranges", "--mode", "lexical"], exe=exe, extra=extra)
                assert any(e["type"] == "search_result" for e in retained)
                status = invoke(f"{device}-status", [*common, "status"], exe=exe, extra=extra)
                assert status[-1]["documents"] == 1, status
                invoke(f"{device}-rebuild", [*common, "rebuild"], exe=exe, extra=extra)
                invoke(f"{device}-remove", [*common, "remove", docs], exe=exe, extra=extra)
                status = invoke(f"{device}-removed-status", [*common, "status"], exe=exe, extra=extra)
                assert status[-1]["documents"] == 0, status
            after = {str(p.relative_to(models)): sha(p) for p in models.rglob("*") if p.is_file()}
            assert before == after, "model files unexpectedly changed"
        summary = {"version": version, "binary_sha256": sha(real_binary),
                   "installed_release": str(binary.resolve().parent), "real_models": bool(args.models),
                   "checks": results, "isolated_data": True, "models_downloaded": False,
                   "personal_database_modified": False}
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Passed {len(results)} installed-binary checks; evidence: {output}")


if __name__ == "__main__":
    main()
