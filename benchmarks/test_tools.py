"""Offline tests for benchmark orchestration; no real downloads, models or GPUs."""
import contextlib
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parent


def module(name):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), ROOT / (name + ".py"))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


fetch = module("fetch-artifacts")
inspector = module("inspect-onnx")
prepare = module("prepare-small-beir")
runner = module("run-baseline")
pilot = module("run-retrieval-pilot")


def varint(value):
    result = bytearray()
    while value > 127:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def field(number, value):
    if isinstance(value, int):
        return varint(number << 3) + varint(value)
    if isinstance(value, str):
        value = value.encode()
    return varint((number << 3) | 2) + varint(len(value)) + value


class Tools(unittest.TestCase):
    def test_cls_derivative_wire_contract(self):
        derivative = module("derive-cls")
        graph = field(7, derivative.additions())
        info = inspector.inspect(graph)
        self.assertEqual(info["graph"]["outputs"], [{"name": "ree_cls_embedding_v1", "element_type": 1, "shape": ["batch", 768]}])
        self.assertEqual(info["operators"], {"::Constant": 1, "::Gather": 1})
        self.assertEqual(info["external_data"], [])

    def test_artifact_digests_and_symlink_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact"
            path.write_bytes(b"hello")
            entry = {"bytes": 5, "sha256": hashlib.sha256(b"hello").hexdigest(),
                     "git_blob_sha1": hashlib.sha1(b"blob 5\0hello").hexdigest()}
            self.assertEqual(fetch.validate_file(path, entry), entry["sha256"])
            with self.assertRaises(ValueError):
                fetch.validate_file(path, {**entry, "bytes": 6})
            with self.assertRaises(ValueError):
                fetch.validate_file(path, {**entry, "sha256": "0" * 64})
            link = path.parent / "link"
            link.symlink_to(path)
            with self.assertRaises(ValueError):
                fetch.validate_file(link, entry)

    def run_fetch(self, directory, content=b"hello", expected=b"hello", path="models/model.onnx", cap=100):
        root = Path(directory)
        lock = root / "lock.json"
        lock.write_text(json.dumps({"version": 1, "files": [{"path": path, "bytes": len(expected),
            "sha256": hashlib.sha256(expected).hexdigest(),
            "url": "https://huggingface.co/example/model/resolve/" + "0" * 40 + "/model.onnx"}]}))
        response = io.BytesIO(content)
        response.url = "https://cdn.example/weights"
        args = ["fetch", "--lock", str(lock), "--scratch", str(root / "scratch"), "--max-bytes", str(cap)]
        with mock.patch("sys.argv", args), mock.patch.object(fetch.urllib.request, "urlopen", return_value=response) as request:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                fetch.main()
            return request.call_count

    def test_download_publication_and_verified_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(self.run_fetch(directory), 1)
            self.assertEqual(self.run_fetch(directory), 0)
            output = Path(directory) / "scratch/models/model.onnx"
            self.assertEqual(output.read_bytes(), b"hello")
            output.write_bytes(b"other")
            with self.assertRaises(ValueError):
                self.run_fetch(directory)
            self.assertEqual(output.read_bytes(), b"other")

    def test_short_oversized_and_bad_checksum_never_publish(self):
        for data in [b"short"[:2], b"helloextra", b"other"]:
            with self.subTest(data=data), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    self.run_fetch(directory, content=data)
                self.assertEqual(list((Path(directory) / "scratch").rglob("model.onnx")), [])
                self.assertEqual([p for p in (Path(directory) / "scratch").rglob("*") if p.is_file()], [])

    def test_path_and_budget_guards_precede_network(self):
        for path, cap in [("../escape", 100), ("/absolute", 100), ("", 100), ("model", 4)]:
            with self.subTest(path=path), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(SystemExit):
                    self.run_fetch(directory, path=path, cap=cap)

    def test_gzip_bounds_crc_and_no_clobber(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source.gz", root / "output"
            source.write_bytes(gzip.compress(b"hello"))
            self.assertEqual(prepare.unpack(source, output), hashlib.sha256(b"hello").hexdigest())
            with self.assertRaises(FileExistsError):
                prepare.unpack(source, output)
            self.assertEqual(output.read_bytes(), b"hello")
            with mock.patch.object(prepare, "MAX_BYTES", 4), self.assertRaises(ValueError):
                prepare.unpack(source, root / "oversized")
            self.assertFalse((root / "oversized").exists())
            source.write_bytes(gzip.compress(b"hello")[:-3])
            with self.assertRaises(EOFError):
                prepare.unpack(source, root / "truncated")
            self.assertFalse((root / "truncated").exists())

    def test_onnx_io_and_external_references(self):
        shape = field(1, field(2, "batch")) + field(1, field(1, 384))
        info = field(1, "logits") + field(2, field(1, field(1, 1) + field(2, shape)))
        tensor = field(2, 1) + field(13, field(1, "location") + field(2, "model.data")) + field(14, 1)
        model = field(1, 8) + field(7, field(5, tensor) + field(12, info))
        result = inspector.inspect(model)
        self.assertEqual(result["external_data"], ["model.data"])
        self.assertEqual(result["graph"]["outputs"], [{"name": "logits", "element_type": 1, "shape": ["batch", 384]}])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.onnx"
            path.write_bytes(model)
            self.assertEqual(inspector.inspect_path(path), result)

    def test_inspector_rejects_unsupported_and_malformed_graphs(self):
        tensor = field(13, field(1, "location") + field(2, "../outside")) + field(14, 1)
        for model in [b"\x00", b"\x3a\x7f", field(7, field(5, tensor)), field(7, field(15, b"")),
                      field(7, b"") + field(25, b""), field(7, b"") + field(20, b"")]:
            with self.subTest(model=model), self.assertRaises(ValueError):
                inspector.inspect(model)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.onnx"
            path.write_bytes(b"\x3a\x7f")
            with self.assertRaisesRegex(ValueError, "truncated"):
                inspector.inspect_path(path)

    def test_baseline_refuses_ci_before_host_probes(self):
        with mock.patch.dict("os.environ", {"CI": "1"}), mock.patch("sys.argv", ["run", "--output", "unused", "--power-mode", "balanced"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner.main()

    def test_bootstrap_stream_is_fixed_and_paired(self):
        indices = pilot.bootstrap_indices("0" * 64, 3, 4)
        self.assertEqual(list(indices), [1, 2, 0, 0, 0, 2, 2, 2, 0, 0, 2, 2])
        self.assertEqual(hashlib.sha256(pilot.index_bytes(indices)).hexdigest(),
                         "ea772a6b5b27b46cc8e831db7c48cc9ff538073f3b58ef20b325f220744e0023")
        values = pilot.bootstrap_means([0., 0.5, 1.], indices)
        shifted = pilot.bootstrap_means([0.1, 0.6, 1.1], indices)
        for a, b in zip(values, shifted):
            self.assertAlmostEqual(b-a, 0.1)
        self.assertEqual(pilot.interval([0., 1., 2., 3.]), [0.07500000000000001, 2.925])
        with self.assertRaises(ValueError):
            pilot.bootstrap_indices("hash", 0)
        with self.assertRaises(ValueError):
            pilot.bootstrap_means([1.], [1])

    def test_bootstrap_modulo_rejection_and_index_integrity(self):
        import struct
        first, second = mock.Mock(), mock.Mock()
        first.digest.return_value = struct.pack("<4Q", *([2**64-1] * 4))
        second.digest.return_value = bytes(32)
        with mock.patch.object(pilot.hashlib, "sha256", side_effect=[first, second]) as digest:
            self.assertEqual(list(pilot.bootstrap_indices("hash", 3, 1)), [0, 0, 0])
            self.assertEqual(digest.call_count, 2)
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(pilot, "REPLICATES", 4):
            path = Path(directory) / "indices"
            path.write_bytes(pilot.index_bytes(pilot.bootstrap_indices("hash", 3, 4)))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(len(pilot.read_indices(path, digest, 3)), 12)
            with self.assertRaises(ValueError):
                pilot.read_indices(path, "0" * 64, 3)
            with self.assertRaises(ValueError):
                pilot.read_indices(path, digest, 2)

    def test_ranking_overlap_and_pilot_ci_guard(self):
        self.assertEqual(pilot.overlap([["a", "b"]], [["b", "a"]]),
                         {"top1_agreement": 0., "top10_overlap": 1.})
        with self.assertRaises(ValueError):
            pilot.overlap([["a", "a"]], [["a", "b"]])
        with mock.patch.dict("os.environ", {"CI": "1"}), mock.patch("sys.argv", ["pilot", "--output", "unused", "--scratch", "unused"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                pilot.main()

    def test_pinned_recipes_match_static_external_data_inventory(self):
        lock = json.loads((ROOT / "candidates/artifacts.json").read_text())
        files = {f["path"]: f for f in lock["files"]}
        inspected = {g["path"]: g for g in json.loads((ROOT / "candidates/graph-inspection.json").read_text())["models"]}
        for candidate in json.loads((ROOT / "candidates/recipes.json").read_text())["candidates"]:
            for provider in ["cpu", "cuda"]:
                recipe = candidate[provider]
                for artifact in [recipe["model"], recipe["tokenizer"], *recipe["sidecars"]]:
                    self.assertEqual(artifact["sha256"], files[artifact["path"]]["sha256"])
                graph = inspected[recipe["model"]["path"]]
                parent = Path(recipe["model"]["path"]).parent
                self.assertEqual({a["path"] for a in recipe["sidecars"]}, {str(parent / p) for p in graph["external_data"]})
                output = recipe["recipe"].get("token_output")
                names = {o["name"] for o in graph["graph"]["outputs"]}
                self.assertTrue(output in names if output else names.intersection({"token_embeddings", "last_hidden_state"}))


if __name__ == "__main__":
    unittest.main()
