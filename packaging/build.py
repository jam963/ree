#!/usr/bin/env python3
"""Build a local x86_64 Linux beta archive from locked sources; no installation."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib

sys.dont_write_bytecode = True
from install import digest, inventory, verify  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TARGET = "x86_64-unknown-linux-gnu"
PROVIDERS = ("libonnxruntime_providers_shared.so", "libonnxruntime_providers_cuda.so")


def run(*args, **kwargs):
    return subprocess.check_output(args, cwd=ROOT, text=True, **kwargs).strip()


def copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)  # Deliberately dereference Cargo's provider symlinks.
    destination.chmod(0o755 if destination.name in {"ree", "ree-launcher", "install.py"} else 0o644)


def notices(metadata, stage):
    """Ship a conservative superset of dependency notices AND unmodified crate sources."""
    out = stage / "licenses"
    shutil.copytree(ROOT / "packaging/notices", out / "upstream")
    for entry in json.loads((out / "upstream/provenance.json").read_text()):
        if digest(out / "upstream" / entry["path"]) != entry["sha256"]:
            raise ValueError(f"upstream notice checksum mismatch: {entry['path']}")
    lock = tomllib.loads((ROOT / "Cargo.lock").read_text())
    checksums = {(p['name'], p['version']): p.get('checksum') for p in lock['package']}
    records = []
    for package in sorted(metadata["packages"], key=lambda p: (p["name"], p["version"])):
        if not package["source"]:
            continue
        name, ver = package["name"], package["version"]
        source = Path(package["manifest_path"]).parent
        if not package["source"].startswith("registry+"):
            raise ValueError(f"source archive collection requires registry dependency: {name}")
        archive = source.parents[1] / ".." / "cache" / source.parent.name / f"{name}-{ver}.crate"
        expected = checksums[(name, ver)]
        if not expected or digest(archive) != expected:
            raise ValueError(f"crate source checksum mismatch: {name}-{ver}")
        copy(archive, stage / "sources/crates" / archive.name)
        files = []
        for path in sorted(source.rglob("*")):
            if path.is_file() and path.name.lower().startswith(
                    ("license", "licence", "copying", "copyright", "notice", "thirdpartynotices")):
                relative = path.relative_to(source)
                destination = out / "crates" / f"{name}-{ver}" / relative
                copy(path, destination)
                files.append(destination.relative_to(stage).as_posix())
        records.append({"name": name, "version": ver, "license": package["license"],
                        "repository": package["repository"], "authors": package["authors"],
                        "source": package["source"], "source_sha256": expected,
                        "source_archive": f"sources/crates/{archive.name}", "notice_files": files})
    (out / "dependencies.json").write_text(json.dumps(records, indent=2) + "\n")
    copy(ROOT / "packaging/THIRD-PARTY.md", out / "README.md")


def archive_tree(stage, destination, epoch):
    # Reproducible archive container for identical payloads; not a bit-reproducible Rust build claim.
    with destination.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=epoch) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as tar:
                for path in [stage, *sorted(stage.rglob("*"))]:
                    info = tar.gettarinfo(str(path), arcname=str(path.relative_to(stage.parent)))
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = epoch
                    info.mode = 0o755 if path.is_dir() or path.name in {"ree", "ree-launcher", "install.py"} else 0o644
                    if path.is_file():
                        with path.open("rb") as stream:
                            tar.addfile(info, stream)
                    else:
                        tar.addfile(info)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist")
    parser.add_argument("--allow-dirty", action="store_true", help="development packages only")
    parser.add_argument("--online", action="store_true", help="allow Cargo dependency downloads")
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        parser.error("native x86_64 Linux builds only")
    dirty = bool(run("git", "status", "--porcelain", "--untracked-files=all"))
    if dirty and not args.allow_dirty:
        parser.error("commit source changes first (or explicitly use --allow-dirty for testing)")
    # Reject alternate build/link settings that would invalidate the documented artifact layout.
    forbidden = [key for key in os.environ if key.startswith("ORT_") or key in {
        "CARGO_BUILD_TARGET", "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS"}]
    if forbidden:
        parser.error(f"unset custom build/link settings: {', '.join(forbidden)}")
    flags = ["--locked"] + ([] if args.online else ["--offline"])
    metadata = json.loads(run("cargo", "metadata", *flags, "--format-version", "1",
                              "--filter-platform", TARGET))
    ver = tomllib.loads((ROOT / "Cargo.toml").read_text())["package"]["version"]
    commit = run("git", "rev-parse", "HEAD")
    epoch = int(os.environ.get("SOURCE_DATE_EPOCH", run("git", "show", "-s", "--format=%ct", "HEAD")))
    name = f"ree-{ver}-{TARGET}"
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f"{name}.tar.gz"
    checksum = output / f"{name}.tar.gz.sha256"
    if archive.exists() or checksum.exists():
        parser.error("output already exists; choose another directory or a new version")
    subprocess.run(["cargo", "build", "--release", *flags, "--bin", "ree"], cwd=ROOT, check=True)
    release = Path(metadata["target_directory"]) / "release"
    if run(str(release / "ree"), "--version") != f"ree {ver}":
        raise ValueError("release executable version mismatch")
    with tempfile.TemporaryDirectory(prefix=".ree-package-", dir=output) as temp:
        stage = Path(temp) / name
        stage.mkdir()
        for file in ("ree", *PROVIDERS):
            copy(release / file, stage / file)
        if "$ORIGIN" not in run("readelf", "-d", str(stage / "ree")):
            raise ValueError("executable lacks adjacent-provider runtime search path")
        subprocess.run(["cargo", "run", *flags, "--example", "generate_docs", "--",
                        str(stage / "cli-docs")], cwd=ROOT, check=True)
        copy(ROOT / "packaging/install.py", stage / "install.py")
        copy(ROOT / "packaging/ree-launcher", stage / "ree-launcher")
        shutil.copytree(ROOT / "docs", stage / "docs")
        for file in ("README.md", "LICENSE-MIT", "LICENSE-APACHE", "Cargo.lock"):
            copy(ROOT / file, stage / file)
        notices(metadata, stage)
        # Include ree source, including explicit dirty tracked/untracked additions for dev packages.
        sources = stage / "sources/ree"
        paths = run("git", "ls-files", "-z", "--cached", "--others", "--exclude-standard").split("\0")
        for relative in paths:
            if relative and (ROOT / relative).is_file():
                copy(ROOT / relative, sources / relative)
        build = {
            "source_commit": commit, "dirty": dirty, "rustc": run("rustc", "-Vv"),
            "cargo": run("cargo", "--version"), "host_kernel": platform.release(),
            "libc": platform.libc_ver(), "source_date_epoch": epoch,
            "dynamic_dependencies": run("readelf", "-d", str(stage / "ree")),
            "required_glibc_versions": sorted(set(re.findall(r"GLIBC_[0-9.]+",
                run("readelf", "--version-info", str(stage / "ree"))))),
            "ort": "1.28.0", "cuda_provider": "CUDA 13; system NVIDIA libraries not bundled",
            "qualification": "local development-host beta; not a portable Linux release qualification",
        }
        (stage / "build-info.json").write_text(json.dumps(build, indent=2) + "\n")
        manifest = {"format": 1, "version": ver, "target": TARGET,
                    "source_commit": commit, "dirty": dirty, "files": inventory(stage)}
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        verify(stage)
        # Detect source changes during the build rather than attributing them to a clean commit.
        if run("git", "rev-parse", "HEAD") != commit or (
                not args.allow_dirty and run("git", "status", "--porcelain", "--untracked-files=all")):
            raise ValueError("source changed during packaging; commit and rerun")
        pending = Path(temp) / archive.name
        archive_tree(stage, pending, epoch)
        sha = digest(pending)
        # No-clobber publication even if another packaging process raced this one.
        os.link(pending, archive)
        with checksum.open("x") as stream:
            stream.write(f"{sha}  {archive.name}\n")
    print(archive)
    print(checksum)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f"ree packaging: {error}", file=sys.stderr)
        sys.exit(1)
