#!/usr/bin/env python3
"""Fetch a reviewed immutable HTTPS artifact lock into explicit temporary scratch.

Standard-library transport/hash verification only. Does not execute model code,
install packages, unpack archives, or run inference. Existing verified files are
reused; mismatched files are never replaced. URLs/hashes are reviewed lock inputs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import urllib.parse
import urllib.request


def digest_file(path, size, git_blob=False):
    digest = hashlib.sha1() if git_blob else hashlib.sha256()
    if git_blob:
        digest.update(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_file(path, entry):
    if path.is_symlink() or not path.is_file() or path.stat().st_size != entry["bytes"]:
        raise ValueError(f"not a regular file of expected size: {path}")
    sha256 = digest_file(path, entry["bytes"])
    if entry.get("sha256") and sha256 != entry["sha256"]:
        raise ValueError(f"SHA-256 mismatch: {path}")
    if entry.get("git_blob_sha1") and digest_file(path, entry["bytes"], True) != entry["git_blob_sha1"]:
        raise ValueError(f"Git blob hash mismatch: {path}")
    return sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--scratch", required=True, type=Path)
    parser.add_argument("--max-bytes", required=True, type=int, help="cap on total locked artifact bytes")
    args = parser.parse_args()
    lock_bytes = args.lock.read_bytes()
    if len(lock_bytes) > 16 * 1024 * 1024:
        parser.error("artifact lock exceeds 16 MiB")
    lock = json.loads(lock_bytes)
    if lock["version"] != 1 or not lock["files"]:
        parser.error("unsupported/empty artifact lock")
    args.scratch.mkdir(parents=True, exist_ok=True)
    root = args.scratch.resolve()
    paths = set()
    for entry in lock["files"]:
        path = PurePosixPath(entry["path"])
        url = urllib.parse.urlsplit(entry["url"])
        if path.is_absolute() or ".." in path.parts or not path.parts or str(path) in paths:
            parser.error("unsafe/duplicate artifact path")
        paths.add(str(path))
        if url.scheme != "https" or url.hostname != "huggingface.co" or url.username or url.password:
            parser.error("lock URLs must be public HTTPS huggingface.co artifact URLs")
        # Resolve must contain an exact revision, not a moving branch.
        parts = url.path.split("/")
        revision = parts[parts.index("resolve") + 1] if "resolve" in parts else ""
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            parser.error("artifact URL must resolve an exact commit")
        if type(entry["bytes"]) is not int or entry["bytes"] <= 0:
            parser.error("positive exact artifact size required")
        if not (entry.get("sha256") or entry.get("git_blob_sha1")):
            parser.error("expected SHA-256 or Git blob ID required")
        for key, length in [("sha256", 64), ("git_blob_sha1", 40)]:
            value = entry.get(key)
            if value is not None and (len(value) != length or any(c not in "0123456789abcdef" for c in value)):
                parser.error(f"invalid {key}")
        current = root
        for part in path.parts:
            current = current / part
            if current.is_symlink():
                parser.error("scratch artifact paths must not traverse symlinks")
    total = sum(e["bytes"] for e in lock["files"])
    if total > args.max_bytes:
        parser.error(f"locked files need {total} bytes, above requested cap")
    missing = sum(e["bytes"] for e in lock["files"] if not (root / e["path"]).exists())
    if shutil.disk_usage(root).free < missing + 512 * 1024 * 1024:
        parser.error("not enough scratch capacity plus 512 MiB headroom")
    print(f"Preflight: {total} locked bytes; {missing} bytes not present", flush=True)
    records = []
    for entry in lock["files"]:
        destination = root / entry["path"]
        if destination.exists():
            sha256 = validate_file(destination, entry)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temp_path = None
            print(f"Fetching {entry['path']} ({entry['bytes']} bytes)", flush=True)
            try:
                with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temp:
                    temp_path = Path(temp.name)
                    request = urllib.request.Request(entry["url"], headers={"User-Agent": "ree-local-qualification/0.1"})
                    with urllib.request.urlopen(request, timeout=120) as response:
                        if urllib.parse.urlsplit(response.url).scheme != "https":
                            raise ValueError("artifact redirected away from HTTPS")
                        remaining = entry["bytes"]
                        while remaining:
                            block = response.read(min(1024 * 1024, remaining))
                            if not block:
                                raise ValueError("short artifact response")
                            temp.write(block)
                            remaining -= len(block)
                        if response.read(1):
                            raise ValueError("oversized artifact response")
                    temp.flush()
                    os.fsync(temp.fileno())
                sha256 = validate_file(temp_path, entry)
                os.link(temp_path, destination)  # no-clobber publication on the same filesystem
            finally:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)
        records.append({**entry, "sha256": sha256, "verified": True})
        print(f"Verified {entry['path']}", flush=True)
    receipt = root / (args.lock.stem + "-verified.json")
    payload = {"version": 1, "input_lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
               "total_bytes": total, "files": records}
    if receipt.exists():
        if receipt.is_symlink() or json.loads(receipt.read_text()) != payload:
            raise ValueError("existing verification receipt differs; refusing replacement")
    else:
        with receipt.open("x") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
    print(f"Verified inventory: {receipt}", flush=True)


if __name__ == "__main__":
    main()
