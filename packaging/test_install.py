"""Offline installation contracts, using tiny fake packages and isolated prefixes."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ree_installer", HERE / "install.py")
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class InstallationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ree-install-test-")
        self.root = Path(self.temp.name)
        self.prefix = self.root / "prefix with spaces"
        self.bundle = self.package("0.1.0-beta.1")

    def tearDown(self):
        self.temp.cleanup()

    def package(self, version):
        bundle = self.root / f"ree-{version}"
        bundle.mkdir()
        shutil.copyfile(HERE / "install.py", bundle / "install.py")
        (bundle / "ree").write_text(f"#!/bin/sh\nprintf 'ree {version}\\n'\n")
        (bundle / "ree").chmod(0o755)
        shutil.copyfile(HERE / "ree-launcher", bundle / "ree-launcher")
        (bundle / "ree-launcher").chmod(0o755)
        for name in [*installer.LINKS.values(), "docs/installation.md", "libonnxruntime_providers_cuda.so"]:
            if name in {"docs", "ree", "ree-launcher"}:
                continue
            path = bundle / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"fixture: {name}\n")
        self.manifest(bundle, version)
        return bundle

    def manifest(self, bundle, version="0.1.0-beta.1"):
        (bundle / "manifest.json").write_text(json.dumps({
            "format": 1, "target": "x86_64-unknown-linux-gnu", "version": version,
            "source_commit": "fixture", "files": installer.inventory(bundle)}))

    def run_installer(self, *args, bundle=None, ok=True, prefix=None):
        result = subprocess.run([sys.executable, str((bundle or self.bundle) / "install.py"),
                                 *args, "--prefix", str(prefix or self.prefix)],
                                text=True, capture_output=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        self.assertEqual(result.returncode == 0, ok, result.stdout + result.stderr)
        return result

    def test_install_is_relocatable_idempotent_and_data_independent(self):
        data = self.prefix / "share/ree/ree.db"
        data.parent.mkdir(parents=True)
        data.write_text("personal database sentinel")
        self.run_installer("verify")
        self.run_installer("install")
        self.run_installer("install")
        moved = self.root / "moved-prefix"
        shutil.move(self.prefix, moved)
        shutil.rmtree(self.bundle)  # The extracted package is not needed after installation.
        self.assertEqual(subprocess.check_output([str(moved / "bin/ree")], text=True).strip(),
                         "ree 0.1.0-beta.1")
        self.assertTrue((moved / "share/doc/ree/installation.md").is_file())
        provider = moved / "lib/ree/current/libonnxruntime_providers_cuda.so"
        self.assertFalse(provider.is_symlink())
        self.assertEqual((moved / "share/ree/ree.db").read_text(), "personal database sentinel")

    def test_upgrade_rollback_uninstall_reinstall(self):
        self.run_installer("install")
        other = self.package("0.1.0-beta.2")
        self.run_installer("install", bundle=other)
        self.assertEqual(os.readlink(self.prefix / "lib/ree/current"), "releases/0.1.0-beta.2")
        self.run_installer("activate", "0.1.0-beta.1")
        self.assertIn("* 0.1.0-beta.1", self.run_installer("list").stdout)
        self.run_installer("uninstall")
        self.run_installer("uninstall")
        for name in installer.LINKS:
            self.assertFalse(os.path.lexists(self.prefix / name))
        self.assertTrue((self.prefix / "lib/ree/releases/0.1.0-beta.2/ree").exists())
        self.run_installer("install")
        self.assertTrue((self.prefix / "bin/ree").exists())

    def test_checksum_failure_never_activates(self):
        self.run_installer("install")
        other = self.package("0.1.0-beta.2")
        (other / "ree").write_text("tampered")
        self.run_installer("install", bundle=other, ok=False)
        self.assertEqual(os.readlink(self.prefix / "lib/ree/current"), "releases/0.1.0-beta.1")

    def test_conflicting_same_version_rejected(self):
        self.run_installer("install")
        (self.bundle / "ree").write_text("changed")
        self.manifest(self.bundle)
        self.run_installer("install", ok=False)
        self.assertNotEqual((self.prefix / "lib/ree/current/ree").read_text(), "changed")

    def test_foreign_file_and_symlink_not_overwritten(self):
        link = self.prefix / "bin/ree"
        link.parent.mkdir(parents=True)
        link.write_text("another application")
        self.run_installer("install", ok=False)
        self.assertEqual(link.read_text(), "another application")
        link.unlink()
        link.symlink_to("/nonexistent/foreign")
        self.run_installer("install", ok=False)
        self.assertEqual(os.readlink(link), "/nonexistent/foreign")

    def test_symlinked_package_and_prefix_rejected(self):
        payload = self.bundle / "ree"
        payload.unlink()
        payload.symlink_to("/bin/true")
        self.run_installer("install", ok=False)
        payload.unlink()
        payload.write_text("ree")
        self.manifest(self.bundle)
        elsewhere = self.root / "elsewhere"
        elsewhere.mkdir()
        self.prefix.symlink_to(elsewhere)
        self.run_installer("install", ok=False)
        self.assertEqual(list(elsewhere.iterdir()), [])

    def test_manifest_traversal_and_extra_files_rejected(self):
        file = self.bundle / "manifest.json"
        original = file.read_text()
        data = json.loads(original)
        data["files"]["../outside"] = "0" * 64
        file.write_text(json.dumps(data))
        self.run_installer("verify", ok=False)
        file.write_text(original)
        (self.bundle / "extra-file").write_text("unexpected")
        self.run_installer("install", ok=False)

    def test_unsafe_prefix_and_lock_rejected(self):
        self.prefix.mkdir(mode=0o777)
        self.prefix.chmod(0o777)
        self.run_installer("install", ok=False)
        self.prefix.chmod(0o755)
        base = self.prefix / "lib/ree"
        base.mkdir(parents=True, exist_ok=True)
        (base / ".install.lock").symlink_to(self.root / "victim")
        self.run_installer("install", ok=False)
        self.assertFalse((self.root / "victim").exists())

    def test_uninstall_refuses_foreign_replacements_and_keeps_data(self):
        self.run_installer("install")
        data = self.prefix / "share/ree/ree.db"
        data.parent.mkdir(parents=True)
        data.write_text("keep")
        link = self.prefix / "bin/ree"
        link.unlink()
        link.write_text("replacement")
        self.run_installer("uninstall", ok=False)
        self.assertEqual(link.read_text(), "replacement")
        self.assertEqual(data.read_text(), "keep")
        self.assertTrue((self.prefix / "lib/ree/current").is_symlink())

    def test_corrupt_installed_version_and_invalid_activation_rejected(self):
        self.run_installer("install")
        self.run_installer("activate", "../../outside", ok=False)
        self.run_installer("activate", "0.1.0-beta.9", ok=False)
        (self.prefix / "lib/ree/current/ree").write_text("tampered")
        self.run_installer("activate", "0.1.0-beta.1", ok=False)

    def test_concurrent_installs_serialize(self):
        # Precreate the prefix so this also works with restrictive test umasks.
        self.prefix.mkdir()
        command = [sys.executable, str(self.bundle / "install.py"), "install", "--prefix", str(self.prefix)]
        processes = [subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                     for _ in range(3)]
        for process in processes:
            stdout, stderr = process.communicate(timeout=30)
            self.assertEqual(process.returncode, 0, stdout + stderr)
        self.assertEqual(os.readlink(self.prefix / "lib/ree/current"), "releases/0.1.0-beta.1")


class ArchiveTests(unittest.TestCase):
    def test_archive_is_deterministic_and_contains_regular_payload(self):
        sys.path.insert(0, str(HERE))
        try:
            spec = importlib.util.spec_from_file_location("ree_builder", HERE / "build.py")
            build = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(build)
        finally:
            sys.path.pop(0)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stage = root / "ree-fixture"
            stage.mkdir()
            (stage / "ree").write_text("binary fixture")
            first, second = root / "first.tgz", root / "second.tgz"
            build.archive_tree(stage, first, 1234)
            os.utime(stage / "ree", (9999, 9999))
            build.archive_tree(stage, second, 1234)
            self.assertEqual(installer.digest(first), installer.digest(second))
            with tarfile.open(first) as archive:
                self.assertTrue(all(f.isfile() or f.isdir() for f in archive.getmembers()))
                self.assertEqual(archive.getmember("ree-fixture/ree").mode, 0o755)


if __name__ == "__main__":
    unittest.main()
