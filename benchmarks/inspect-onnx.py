#!/usr/bin/env python3
"""Read ONNX I/O and external-data references without installing ONNX or executing it.

A deliberately narrow wire-format inspector, NOT a full ONNX validator or sandbox.
Rejects training/functions/sparse tensors rather than claiming complete coverage.
Schema reference: https://github.com/onnx/onnx/blob/main/onnx/onnx.proto
"""
import argparse
from collections import Counter
import json
import mmap
from pathlib import Path, PurePosixPath


def varint(data, offset):
    value = 0
    for shift in range(0, 70, 7):
        if offset >= len(data):
            raise ValueError("truncated varint")
        byte = data[offset]
        offset += 1
        value |= (byte & 127) << shift
        if byte < 128:
            if value > (1 << 64) - 1:
                raise ValueError("varint overflow")
            return value, offset
    raise ValueError("overlong varint")


def fields(data):
    offset = 0
    while offset < len(data):
        key, offset = varint(data, offset)
        number, wire = key >> 3, key & 7
        if number == 0:
            raise ValueError("invalid field zero")
        if wire == 0:
            value, offset = varint(data, offset)
        elif wire in (1, 2, 5):
            if wire == 2:
                size, offset = varint(data, offset)
            else:
                size = 8 if wire == 1 else 4
            if size > len(data) - offset:
                raise ValueError("truncated field")
            value = data[offset:offset + size]
            offset += size
        else:
            raise ValueError("unsupported protobuf wire type")
        yield number, wire, value


def text(data):
    return bytes(data).decode("utf-8")


def value_info(data):
    result = {}
    for number, _, value in fields(data):
        if number == 1:
            result["name"] = text(value)
        elif number == 2:
            types = list(fields(value))
            if len(types) != 1 or types[0][0] != 1:
                raise ValueError("only tensor I/O supported by inspector")
            for field, _, tensor in fields(types[0][2]):
                if field == 1:
                    result["element_type"] = tensor
                elif field == 2:
                    result["shape"] = []
                    for dim_field, _, dim in fields(tensor):
                        if dim_field != 1:
                            raise ValueError("unexpected shape field")
                        dimensions = [(n, v) for n, _, v in fields(dim) if n in (1, 2)]
                        if len(dimensions) > 1:
                            raise ValueError("ambiguous dimension")
                        result["shape"].append(None if not dimensions else
                                               dimensions[0][1] if dimensions[0][0] == 1 else text(dimensions[0][1]))
    return result


def inspect(data):
    references = set()
    operators = Counter()
    tensor_types = Counter()

    def tensor(data):
        location = 0
        external = {}
        for number, _, value in fields(data):
            if number == 2:
                tensor_types[value] += 1
            elif number == 13:
                entry = {n: text(v) for n, _, v in fields(value)}
                if set(entry) != {1, 2} or entry[1] in external:
                    raise ValueError("invalid external-data entry")
                external[entry[1]] = entry[2]
            elif number == 14:
                location = value
        if location == 1:
            path = PurePosixPath(external["location"])
            if path.is_absolute() or ".." in path.parts or not path.parts:
                raise ValueError("external data escapes graph directory")
            references.add(str(path))
        elif location != 0 or external:
            raise ValueError("inconsistent external-data location")

    def graph(data, depth=0):
        if depth > 64:
            raise ValueError("graph nesting limit")
        result = {"inputs": [], "outputs": []}
        for number, _, value in fields(data):
            if number == 5:
                tensor(value)
            elif number == 15:
                raise ValueError("sparse tensors require a fuller inspector")
            elif number in (11, 12):
                result["inputs" if number == 11 else "outputs"].append(value_info(value))
            elif number == 1:
                op, domain = None, ""
                for field, _, node in fields(value):
                    if field == 4:
                        op = text(node)
                    elif field == 7:
                        domain = text(node)
                    elif field == 5:
                        for attr, _, payload in fields(node):
                            if attr in (5, 10):
                                tensor(payload)
                            elif attr in (6, 11):
                                graph(payload, depth + 1)
                            elif attr in (22, 23):
                                raise ValueError("sparse attributes require a fuller inspector")
                operators[f"{domain}::{op}"] += 1
        return result

    result = {}
    for number, _, value in fields(data):
        if number == 1:
            result["ir_version"] = value
        elif number in (2, 3):
            result["producer" if number == 2 else "producer_version"] = text(value)
        elif number == 7:
            if "graph" in result:
                raise ValueError("multiple model graphs")
            result["graph"] = graph(value)
        elif number in (20, 25):
            raise ValueError("training/functions require a fuller inspector")
    if "graph" not in result:
        raise ValueError("missing model graph")
    result.update(external_data=sorted(references), operators=dict(sorted(operators.items())),
                  tensor_element_types=dict(sorted(tensor_types.items())))
    return result


def inspect_path(path):
    with Path(path).open("rb") as file, mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
        view = memoryview(mapped)
        error = None
        try:
            result = inspect(view)
        except Exception as exc:
            # Clear traceback-held child views before closing the mmap, so malformed
            # input reports its real error rather than an unrelated BufferError.
            error = f"{type(exc).__name__}: {exc}"
        finally:
            view.release()
    if error is not None:
        raise ValueError(error)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect_path(args.model), indent=2, sort_keys=True))
