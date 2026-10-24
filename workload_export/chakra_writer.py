"""Writer (and reader) for the Chakra execution trace format, with no dependencies.

Chakra (MLCommons) is the open graph format ASTRA-sim and other simulators consume. A
.et file is a length delimited protobuf stream: one GlobalMetadata message followed by
one Node message per operator, each prefixed by its size as a varint. The schema is
schema/protobuf/et_def.proto in github.com/mlcommons/chakra; the field numbers below
are taken from it. Attribute names and value types follow what ASTRA-sim's Chakra
feeder reads (src/feeder/et_feeder_node.cpp):

  is_cpu_op    bool_val      num_ops      int64_val     tensor_size  uint64_val
  comm_type    int64_val     comm_size    int64_val     pg_name      string_val
  comm_src     int32_val     comm_dst     int32_val     comm_tag     int32_val

The reader exists so the tests can round trip a file; workload_export/check_chakra.py
also decodes the output with the official protobuf library to confirm the encoding.
"""

import struct

# NodeType enum
METADATA_NODE, MEM_LOAD_NODE, MEM_STORE_NODE, COMP_NODE = 1, 2, 3, 4
COMM_SEND_NODE, COMM_RECV_NODE, COMM_COLL_NODE = 5, 6, 7
# CollectiveCommType enum
ALL_REDUCE, REDUCE, ALL_GATHER, GATHER, SCATTER, BROADCAST, ALL_TO_ALL, REDUCE_SCATTER = range(8)

SCHEMA_VERSION = "0.0.4"

# AttributeProto oneof field numbers and wire types for the value kinds used here
_ATTR_FIELDS = {
    "double_val": (3, 1), "int32_val": (7, 0), "int64_val": (9, 0), "uint64_val": (13, 0),
    "bool_val": (27, 0), "string_val": (29, 2), "bool_list": (28, 2), "int64_list": (10, 2),
}


# --------------------------------------------------------------------------
# wire format
# --------------------------------------------------------------------------

def _varint(n):
    if n < 0:
        n += 1 << 64            # proto3 encodes negative int32/int64 as 10 byte two's complement
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _key(field, wire):
    return _varint((field << 3) | wire)


def _len_field(field, payload):
    return _key(field, 2) + _varint(len(payload)) + payload


def _str_field(field, s):
    return _len_field(field, s.encode("utf-8"))


def _attr(name, kind, value):
    field, wire = _ATTR_FIELDS[kind]
    body = _str_field(1, name)
    if kind in ("int32_val", "int64_val", "uint64_val"):
        body += _key(field, 0) + _varint(int(value))
    elif kind == "bool_val":
        body += _key(field, 0) + _varint(1 if value else 0)
    elif kind == "double_val":
        body += _key(field, 1) + struct.pack("<d", float(value))
    elif kind == "string_val":
        body += _str_field(field, value)
    elif kind == "bool_list":
        body += _len_field(field, _len_field(1, b"".join(_varint(1 if v else 0) for v in value)))
    elif kind == "int64_list":
        body += _len_field(field, _len_field(1, b"".join(_varint(int(v)) for v in value)))
    else:
        raise ValueError(kind)
    return body


def encode_node(node):
    """node: dict with id, name, type, data_deps, ctrl_deps, duration_micros, inputs and
    outputs (dicts of values/shapes/types strings), attrs (list of (name, kind, value))."""
    b = _key(1, 0) + _varint(node["id"]) + _str_field(2, node["name"]) + _key(3, 0) + _varint(node["type"])
    for field, deps in ((4, node.get("ctrl_deps", ())), (5, node.get("data_deps", ()))):
        if deps:
            b += _len_field(field, b"".join(_varint(d) for d in deps))     # packed repeated uint64
    if node.get("start_time_micros"):
        b += _key(6, 0) + _varint(node["start_time_micros"])
    if node.get("duration_micros"):
        b += _key(7, 0) + _varint(node["duration_micros"])
    for field, io in ((8, node.get("inputs")), (9, node.get("outputs"))):
        if io:
            body = b"".join(_str_field(i, io[k]) for i, k in ((1, "values"), (2, "shapes"), (3, "types")) if io.get(k))
            b += _len_field(field, body)
    for name, kind, value in node.get("attrs", ()):
        b += _len_field(10, _attr(name, kind, value))
    return b


def encode_metadata(attrs=()):
    b = _str_field(1, SCHEMA_VERSION)
    for name, kind, value in attrs:
        b += _len_field(2, _attr(name, kind, value))
    return b


def write_et(path, nodes, metadata_attrs=()):
    """Write a Chakra .et file: GlobalMetadata, then every node, each length prefixed."""
    with open(path, "wb") as f:
        for msg in [encode_metadata(metadata_attrs)] + [encode_node(n) for n in nodes]:
            f.write(_varint(len(msg)))
            f.write(msg)


# --------------------------------------------------------------------------
# reader (for round trip tests)
# --------------------------------------------------------------------------

def _read_varint(buf, i):
    shift = result = 0
    while True:
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, i
        shift += 7


def _fields(buf):
    i = 0
    while i < len(buf):
        key, i = _read_varint(buf, i)
        field, wire = key >> 3, key & 7
        if wire == 0:
            v, i = _read_varint(buf, i)
        elif wire == 1:
            v = buf[i:i + 8]
            i += 8
        elif wire == 2:
            n, i = _read_varint(buf, i)
            v = buf[i:i + n]
            i += n
        elif wire == 5:
            v = buf[i:i + 4]
            i += 4
        else:
            raise ValueError(f"wire type {wire}")
        yield field, wire, v


def _packed(buf):
    out, i = [], 0
    while i < len(buf):
        v, i = _read_varint(buf, i)
        out.append(v)
    return out


def _decode_attr(buf):
    name, value = None, None
    for field, wire, v in _fields(buf):
        if field == 1:
            name = v.decode()
        elif field in (7, 9, 13):
            value = v if v < (1 << 63) else v - (1 << 64)
        elif field == 27:
            value = bool(v)
        elif field == 29:
            value = v.decode()
        elif field == 3:
            value = struct.unpack("<d", v)[0]
        elif field in (28, 10):
            inner = [x for f, _, x in _fields(v) if f == 1]
            vals = _packed(inner[0]) if inner else []
            value = [bool(x) for x in vals] if field == 28 else vals
    return name, value


def decode_node(buf):
    n = {"data_deps": [], "ctrl_deps": [], "attrs": {}, "duration_micros": 0}
    for field, wire, v in _fields(buf):
        if field == 1:
            n["id"] = v
        elif field == 2:
            n["name"] = v.decode()
        elif field == 3:
            n["type"] = v
        elif field in (4, 5):
            vals = _packed(v) if wire == 2 else [v]
            n["ctrl_deps" if field == 4 else "data_deps"] += vals
        elif field == 7:
            n["duration_micros"] = v
        elif field in (8, 9):
            io = {}
            for f2, _, v2 in _fields(v):
                io[{1: "values", 2: "shapes", 3: "types"}[f2]] = v2.decode()
            n["inputs" if field == 8 else "outputs"] = io
        elif field == 10:
            k, val = _decode_attr(v)
            n["attrs"][k] = val
    return n


def read_et(path):
    """Returns (metadata attrs dict, list of node dicts)."""
    data = open(path, "rb").read()
    i, msgs = 0, []
    while i < len(data):
        n, i = _read_varint(data, i)
        msgs.append(data[i:i + n])
        i += n
    meta = {}
    for field, _, v in _fields(msgs[0]):
        if field == 2:
            k, val = _decode_attr(v)
            meta[k] = val
    return meta, [decode_node(m) for m in msgs[1:]]
