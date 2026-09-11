#!/usr/bin/env python3
"""Offline Linux installer for a trusted, extracted ree release (Python 3.11+)."""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import stat
import sys
import tempfile

VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+(?:-[A-Za-z0-9.-]+)?\Z")
LINKS = {
    "bin/ree": "ree-launcher",
    "share/man/man1/ree.1": "cli-docs/ree.1",
    "share/bash-completion/completions/ree": "cli-docs/ree.bash",
    "share/zsh/site-functions/_ree": "cli-docs/_ree",
    "share/fish/vendor_completions.d/ree.fish": "cli-docs/ree.fish",
    "share/doc/ree": "docs",
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def version(value):
    if not isinstance(value, str) or not VERSION.fullmatch(value):
        raise ValueError("invalid release version")
    return value


def inventory(root):
    files = {}
    for path in sorted(root.rglob("*")):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
            raise ValueError(f"release contains a symlink/special file: {path}")
        if stat.S_ISREG(mode) and path != root / "manifest.json":
            files[path.relative_to(root).as_posix()] = digest(path)
    return files


def verify(root):
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"not a regular release directory: {root}")
    manifest = root / "manifest.json"
    if manifest.is_symlink() or not manifest.is_file():
        raise ValueError("missing regular manifest.json")
    data = json.loads(manifest.read_text())
    version(data["version"])
    if data["format"] != 1 or data["target"] != "x86_64-unknown-linux-gnu":
        raise ValueError("unsupported release format/target")
    for name, sha in data["files"].items():
        path = PurePosixPath(name)
        if (not name or path.is_absolute() or ".." in path.parts
                or path.as_posix() != name or not re.fullmatch(r"[a-f0-9]{64}", sha)):
            raise ValueError("invalid manifest file entry")
    if inventory(root) != data["files"]:
        raise ValueError("release file inventory/checksum mismatch")
    required = {"ree", "ree-launcher", "install.py", "cli-docs/ree.1", "cli-docs/ree.bash",
                "cli-docs/_ree", "cli-docs/ree.fish", "docs/installation.md"}
    if not required.issubset(data["files"]):
        raise ValueError("release is missing required files")
    return data


def directory(path):
    """Reject symlinked paths. The prefix must be controlled by the installer UID."""
    if path.is_symlink():
        raise ValueError(f"refusing symlinked directory: {path}")
    if not path.exists():
        directory(path.parent)
        path.mkdir(mode=0o755, exist_ok=True)
    if not path.is_dir():
        raise ValueError(f"not a directory: {path}")


def managed_directory(prefix, path):
    directory(prefix)
    for part in [prefix, *reversed(list(path.parents)), path]:
        if part != prefix and prefix not in part.parents:
            continue
        directory(part)
        info = part.stat()
        if info.st_uid != os.geteuid() or info.st_mode & 0o022:
            raise ValueError(f"directory must be owned by installer UID and not group/world writable: {part}")


def link_target(prefix, relative, dest):
    return os.path.relpath(prefix / "lib/ree/current" / dest, (prefix / relative).parent)


def check_links(prefix):
    for name, dest in LINKS.items():
        link = prefix / name
        managed_directory(prefix, link.parent)
        if os.path.lexists(link) and not (
            link.is_symlink() and os.readlink(link) == link_target(prefix, name, dest)
        ):
            raise ValueError(f"refusing to overwrite unrelated path: {link}")


def current_version(base):
    current = base / "current"
    if not os.path.lexists(current):
        return None
    if not current.is_symlink():
        raise ValueError("refusing non-symlink current path")
    target = os.readlink(current)
    parts = PurePosixPath(target).parts
    if len(parts) != 2 or parts[0] != "releases":
        raise ValueError("refusing unrelated current symlink")
    return version(parts[1])


def activate(prefix, base, selected):
    release = base / "releases" / version(selected)
    if verify(release)["version"] != selected:
        raise ValueError("release directory/version mismatch")
    old = current_version(base)
    check_links(prefix)
    # Public links always follow one current pointer. Switching that pointer is atomic.
    for name, dest in LINKS.items():
        link = prefix / name
        if not os.path.lexists(link):
            link.symlink_to(link_target(prefix, name, dest))
    with tempfile.TemporaryDirectory(prefix=".activate-", dir=base) as temp:
        pointer = Path(temp) / "current"
        pointer.symlink_to(f"releases/{selected}")
        os.replace(pointer, base / "current")
    print(f"Active: {selected} ({prefix / 'bin/ree'})")
    if old and old != selected:
        print(f"Previous: {old}; rollback: python3 {release / 'install.py'} activate {old} --prefix {prefix}")


@contextlib.contextmanager
def locked(prefix):
    base = prefix / "lib/ree"
    managed_directory(prefix, base / "releases")
    fd = os.open(base / ".install.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_nlink != 1:
            raise ValueError("unsafe install lock")
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield base
    finally:
        os.close(fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["install", "activate", "list", "uninstall", "verify"])
    parser.add_argument("version", nargs="?", help="installed version for activate")
    parser.add_argument("--prefix", type=Path, default=Path.home() / ".local",
                        help="absolute prefix (default: ~/.local; system-wide: /usr/local)")
    args = parser.parse_args(argv)
    if args.command == "activate" and not args.version:
        parser.error("activate requires a version")
    if args.command != "activate" and args.version:
        parser.error("only activate accepts a version")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        parser.error("this beta supports x86_64 Linux only")
    if not args.prefix.is_absolute() or ".." in args.prefix.parts or args.prefix == Path("/"):
        parser.error("prefix must be an absolute non-root path without '..'")
    prefix = args.prefix
    # Reject symlink ancestors too, including for a not-yet-created prefix.
    for parent in [*prefix.parents, prefix]:
        if parent.is_symlink():
            raise ValueError(f"refusing symlinked prefix component: {parent}")
    source = Path(__file__).resolve().parent
    if args.command == "verify":
        print(f"Verified: {verify(source)['version']}")
        return
    data = verify(source) if args.command == "install" else None
    with locked(prefix) as base:
        current_version(base)
        if args.command == "install":
            check_links(prefix)
            dest = base / "releases" / data["version"]
            if os.path.lexists(dest):
                if verify(dest) != data:
                    raise ValueError("version already installed with different contents; use a new version")
            else:
                with tempfile.TemporaryDirectory(prefix=".install-", dir=base) as temp:
                    stage = Path(temp) / "release"
                    shutil.copytree(source, stage, symlinks=True)
                    if verify(stage) != data:
                        raise ValueError("release changed during copy")
                    for path in stage.rglob("*"):
                        path.chmod(0o755 if path.is_dir() or path.name in {"ree", "ree-launcher", "install.py"} else 0o644)
                    stage.chmod(0o755)
                    os.rename(stage, dest)
            activate(prefix, base, data["version"])
        elif args.command == "activate":
            activate(prefix, base, args.version)
        elif args.command == "list":
            active = current_version(base)
            for release in sorted((base / "releases").iterdir()):
                print(f"{'*' if release.name == active else ' '} {release.name}")
        elif args.command == "uninstall":
            check_links(prefix)
            for name in LINKS:
                link = prefix / name
                if link.is_symlink():
                    link.unlink()
            if (base / "current").is_symlink():
                (base / "current").unlink()
            print(f"Uninstalled command/docs links. Retained release files in {base / 'releases'}.")
            print("Databases, configuration, caches and running processes are unchanged.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"ree installer: {error}", file=sys.stderr)
        sys.exit(1)
