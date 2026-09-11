#!/usr/bin/env python3
"""Independent, offline reproduction of arctic-cls-gather-v1. Pinned parents only.
No ONNX package, model download or inference. Never overwrites an existing output.
The result is an experimental runtime artifact, not a newly qualified model.
"""
import argparse
import hashlib
import importlib.util
import json
import mmap
from pathlib import Path

spec = importlib.util.spec_from_file_location("onnx_inspector", Path(__file__).with_name("inspect-onnx.py"))
inspector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inspector)
PARENTS = {"03d923bb1850ebdccb068e2f3abd8aa43fe81c50d07d037ef103fe3d0fb78e3b", "f27ab40ab6e230265ba49a202a37f1ad031556256cbbc105d0ca9c0bdc7ec42e"}
OUTPUT = "ree_cls_embedding_v1"


def varint(n):
    result = bytearray()
    while n >= 128:
        result.append((n & 127) | 128)
        n >>= 7
    return bytes(result + bytes([n]))


def field(n, value):
    if isinstance(value, str):
        value = value.encode()
    if isinstance(value, int):
        return varint(n << 3) + varint(value)
    return varint((n << 3) | 2) + varint(len(value)) + value


def additions():
    def attr(name, value):
        return field(1, name) + field(3, value) + field(20, 2)
    constant = field(2, "ree_cls_index_v1") + field(3, "ree/cls-index-v1") + field(4, "Constant") + field(5, attr("value_int", 0))
    gather = field(1, "token_embeddings") + field(1, "ree_cls_index_v1") + field(2, OUTPUT) + field(3, "ree/cls-gather-v1") + field(4, "Gather") + field(5, attr("axis", 1))
    shape = field(1, field(2, "batch")) + field(1, field(1, 768))
    tensor = field(1, 1) + field(2, shape)
    output = field(1, OUTPUT) + field(2, field(1, tensor))
    return field(1, constant) + field(1, gather) + field(12, output)


def spans(data, start=0, end=None):
    end = len(data) if end is None else end
    offset = start
    while offset < end:
        start = offset
        key, offset = inspector.varint(data, offset)
        number, wire = key >> 3, key & 7
        if wire == 0:
            _, offset = inspector.varint(data, offset)
            size = 0
        elif wire == 2:
            size, offset = inspector.varint(data, offset)
        elif wire in (1, 5):
            size = 8 if wire == 1 else 4
        else:
            raise ValueError("unsupported wire type")
        if not number or offset + size > end:
            raise ValueError("invalid/truncated field")
        yield number, start, offset, offset + size
        offset += size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.parent.open("rb") as source:
        parent_hash = hashlib.file_digest(source, "sha256").hexdigest()
        if parent_hash not in PARENTS:
            raise ValueError("parent is not a pinned Arctic artifact")
        with mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as data:
            root = list(spans(data))
            graphs = [f for f in root if f[0] == 7]
            if len(graphs) != 1:
                raise ValueError("expected one graph")
            graph = graphs[0]
            contents = list(spans(data, graph[2], graph[3]))
            outputs = [f for f in contents if f[0] == 12]
            if len(outputs) != 2:
                raise ValueError("expected two Arctic outputs")
            extra = additions()
            length = graph[3] - graph[2] - sum(f[3] - f[1] for f in outputs) + len(extra)
            with args.output.open("xb") as out:
                def copy(start, end):
                    for offset in range(start, end, 1024 * 1024):
                        out.write(data[offset:min(end, offset + 1024 * 1024)])
                for number, start, payload, end in root:
                    if number != 7:
                        copy(start, end)
                        continue
                    out.write(varint((7 << 3) | 2) + varint(length))
                    for number, start, payload, end in contents:
                        if number != 12:
                            copy(start, end)
                    out.write(extra)
    with args.output.open("rb") as result:
        digest = hashlib.file_digest(result, "sha256").hexdigest()
    print(json.dumps({"transform": "arctic-cls-gather-v1", "parent_sha256": parent_hash, "sha256": digest, "bytes": args.output.stat().st_size, "output": OUTPUT}, indent=2))


if __name__ == "__main__":
    main()
