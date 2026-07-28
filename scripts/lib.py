"""Shared utilities for minfer diagnostic scripts."""

import os
import struct
import numpy as np
from gguf import GGUFReader, GGUFWriter, GGUFValueType


# ─── GGUF field helpers ──────────────────────────────────────

def extract_field_value(field):
    """Extract a Python value from a GGUFReader field."""
    ftype = field.types
    parts = field.parts

    if len(ftype) == 1:
        t = ftype[0]
        val_bytes = parts[-1].tobytes()
        if t == GGUFValueType.UINT32:
            return struct.unpack("<I", val_bytes)[0]
        elif t == GGUFValueType.INT32:
            return struct.unpack("<i", val_bytes)[0]
        elif t == GGUFValueType.UINT16:
            return struct.unpack("<H", val_bytes)[0]
        elif t == GGUFValueType.INT16:
            return struct.unpack("<h", val_bytes)[0]
        elif t == GGUFValueType.UINT8:
            return val_bytes[0]
        elif t == GGUFValueType.INT8:
            return int(np.int8(val_bytes[0]))
        elif t == GGUFValueType.FLOAT32:
            return struct.unpack("<f", val_bytes)[0]
        elif t == GGUFValueType.FLOAT64:
            return struct.unpack("<d", val_bytes)[0]
        elif t == GGUFValueType.BOOL:
            return bool(val_bytes[0])
        elif t == GGUFValueType.STRING:
            return val_bytes.decode("utf-8", errors="replace").strip("\x00")
        else:
            raise ValueError(f"Unknown scalar type {t}")

    elif len(ftype) == 2 and ftype[0] == GGUFValueType.ARRAY:
        elem_type = ftype[1]
        array_len = struct.unpack("<Q", parts[4].tobytes())[0]
        result = []
        if elem_type == GGUFValueType.STRING:
            for i in range(array_len):
                str_idx = 5 + i * 2
                if str_idx + 1 < len(parts):
                    s = (
                        parts[str_idx + 1]
                        .tobytes()
                        .decode("utf-8", errors="replace")
                        .strip("\x00")
                    )
                    result.append(s)
        elif elem_type == GGUFValueType.INT32:
            for i in range(array_len):
                idx = 5 + i
                if idx < len(parts):
                    result.append(struct.unpack("<i", parts[idx].tobytes())[0])
        elif elem_type == GGUFValueType.UINT32:
            for i in range(array_len):
                idx = 5 + i
                if idx < len(parts):
                    result.append(struct.unpack("<I", parts[idx].tobytes())[0])
        elif elem_type == GGUFValueType.FLOAT32:
            for i in range(array_len):
                idx = 5 + i
                if idx < len(parts):
                    result.append(struct.unpack("<f", parts[idx].tobytes())[0])
        else:
            raise ValueError(f"Unknown array element type {elem_type}")
        return result

    raise ValueError(f"Unexpected types {ftype}")


def add_field_to_writer(writer, name, value, ftype):
    """Add a properly typed field to GGUFWriter."""
    t = ftype[0]
    if len(ftype) == 2:  # ARRAY
        writer.add_array(name, value)
    elif t == GGUFValueType.UINT32:
        writer.add_uint32(name, value)
    elif t == GGUFValueType.INT32:
        writer.add_int32(name, value)
    elif t == GGUFValueType.UINT16:
        writer.add_uint16(name, value)
    elif t == GGUFValueType.INT16:
        writer.add_int16(name, value)
    elif t == GGUFValueType.UINT8:
        writer.add_uint8(name, value)
    elif t == GGUFValueType.INT8:
        writer.add_int8(name, value)
    elif t == GGUFValueType.FLOAT32:
        writer.add_float32(name, value)
    elif t == GGUFValueType.FLOAT64:
        writer.add_float64(name, value)
    elif t == GGUFValueType.BOOL:
        writer.add_bool(name, value)
    elif t == GGUFValueType.STRING:
        writer.add_string(name, value)
    else:
        print(f"  [WARN] unknown type {t} for field {name}, skipping")


# ─── GGUF truncation ─────────────────────────────────────────

def create_truncated_model(src_path, dest_path, target_layers, arch_str):
    """Create a zero-copy truncated GGUF with target_layers using GGUFWriter."""
    import numpy as _np
    from gguf import GGMLQuantizationType
    reader = GGUFReader(src_path)
    writer = GGUFWriter(dest_path, arch=arch_str)
    writer.add_architecture()

    for field in reader.fields.values():
        name = field.name
        if name.startswith("GGUF."):
            continue
        if name == "block_count" or name == f"{arch_str}.block_count":
            writer.add_uint32("block_count", target_layers)
            continue
        try:
            value = extract_field_value(field)
            add_field_to_writer(writer, name, value, field.types)
        except Exception as e:
            print(f"  [WARN] failed to copy field {name}: {e}")
            continue

    written = 0
    for tensor in reader.tensors:
        skip = False
        if "blk." in tensor.name:
            parts = tensor.name.split(".")
            if len(parts) >= 2:
                try:
                    layer_idx = int(parts[1])
                    if layer_idx >= target_layers:
                        skip = True
                except ValueError:
                    pass
        if not skip:
            # Map GGUF tensor type to numpy dtype + raw_dtype for quantized types
            tt = tensor.tensor_type
            raw = GGMLQuantizationType(tt) if tt > 1 else None
            if raw is not None:
                # Quantized: pass raw_dtype, use F32 as placeholder np.dtype
                writer.add_tensor_info(
                    tensor.name, tensor.shape, _np.float32,
                    tensor.n_bytes, raw_dtype=raw,
                )
            else:
                # F32/F16: standard path
                writer.add_tensor_info(
                    tensor.name, tensor.shape, _np.float32,
                    tensor.n_bytes,
                )
            written += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    return written


# ─── Math helpers ────────────────────────────────────────────

def load_f32(path):
    """Load raw f32 dump file."""
    return np.fromfile(path, dtype=np.float32)


def cosine_sim(a, b):
    """Cosine similarity between two 1D arrays."""
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-30 or nb < 1e-30:
        return 0.0
    return float(dot / (na * nb))
