#!/usr/bin/env python3
"""
Corrects pruned Gemma 3 GGUF models to be loadable by llama-cpp-python.

Fixes applied:
  1. tokenizer.ggml.pre: 'gemma' → 'default'  (BPE pre-tokenizer not recognized in 0.3.x)
  2. tokenizer.ggml.tokens array: truncated to match embedding vocab size
  3. tokenizer.ggml.token_type array: same truncation

The pruning process sometimes removes the last N embeddings while keeping vocab metadata intact.
This script makes both consistent.

Usage: python3 fix_pruned_gemma.py <input.gguf> <output.gguf>
"""

import sys
import os
import numpy as np
from gguf import GGUFReader, GGUFWriter, GGUFValueType, GGMLQuantizationType


def fix_pruned_gemma(src_path: str, dst_path: str) -> None:
    print(f"Reading: {src_path}")
    reader = GGUFReader(src_path, "r")

    # Determine actual vocab size from embedding tensor
    embd_tensor = next((t for t in reader.tensors if t.name == "token_embd.weight"), None)
    if embd_tensor is None:
        raise ValueError("token_embd.weight tensor not found")

    # Shape is stored reversed in GGUF: [embed_dim, vocab_size]
    embd_shape = embd_tensor.shape.tolist()
    actual_vocab = int(embd_shape[-1])  # last dimension = vocab size
    print(f"  Embedding shape: {embd_shape}  →  actual vocab = {actual_vocab}")

    tokens_field = reader.fields.get("tokenizer.ggml.tokens")
    declared_vocab = len(tokens_field.data) if tokens_field else actual_vocab
    print(f"  Declared vocab (tokens array): {declared_vocab}")

    if declared_vocab != actual_vocab:
        print(f"  Vocab mismatch! Truncating arrays from {declared_vocab} → {actual_vocab}")

    # Architecture
    arch_field = reader.fields["general.architecture"]
    architecture = bytes(arch_field.parts[-1]).decode()
    print(f"  Architecture: {architecture}")

    writer = GGUFWriter(dst_path, architecture)

    # --- Metadata pass-through with fixes ---
    skip = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture"}

    for name, field in reader.fields.items():
        if name in skip:
            continue

        vtype = field.types[0]

        # Fix 1: pre-tokenizer
        if name == "tokenizer.ggml.pre":
            old = bytes(field.parts[-1]).decode()
            print(f"  Fixing {name}: {old!r} → 'default'")
            writer.add_string(name, "default")
            continue

        if vtype == GGUFValueType.STRING:
            # field.data[0] is an INDEX into field.parts for string fields too
            val = bytes(field.parts[field.data[0]]).decode()
            writer.add_string(name, val)

        elif vtype == GGUFValueType.BOOL:
            # field.data[0] is an INDEX into field.parts — not the value itself
            writer.add_bool(name, bool(field.parts[field.data[0]][0]))

        elif vtype == GGUFValueType.UINT8:
            writer.add_uint8(name, int(field.parts[field.data[0]][0]))
        elif vtype == GGUFValueType.INT8:
            writer.add_int8(name, int(field.parts[field.data[0]][0]))
        elif vtype == GGUFValueType.UINT16:
            writer.add_uint16(name, int(field.parts[field.data[0]][0]))
        elif vtype == GGUFValueType.INT16:
            writer.add_int16(name, int(field.parts[field.data[0]][0]))
        elif vtype == GGUFValueType.UINT32:
            writer.add_uint32(name, int(field.parts[field.data[0]][0]))
        elif vtype == GGUFValueType.INT32:
            writer.add_int32(name, int(field.parts[field.data[0]][0]))
        elif vtype == GGUFValueType.FLOAT32:
            writer.add_float32(name, float(field.parts[field.data[0]][0]))
        elif vtype == GGUFValueType.UINT64:
            writer.add_uint64(name, int(field.parts[field.data[0]][0]))
        elif vtype == GGUFValueType.INT64:
            writer.add_int64(name, int(field.parts[field.data[0]][0]))
        elif vtype == GGUFValueType.FLOAT64:
            writer.add_float64(name, float(field.parts[field.data[0]][0]))

        elif vtype == GGUFValueType.ARRAY:
            elem_type = field.types[1]
            full_count = len(field.data)

            # Fix 2 & 3: truncate vocab arrays
            if name in ("tokenizer.ggml.tokens", "tokenizer.ggml.token_type"):
                count = min(full_count, actual_vocab)
                if count < full_count:
                    print(f"  Truncating {name}: {full_count} → {count}")
            else:
                count = full_count

            if elem_type == GGUFValueType.STRING:
                vals = []
                for i in range(count):
                    idx = field.data[i]
                    vals.append(bytes(field.parts[idx]).decode(errors="replace"))
                writer.add_array(name, vals)
            else:
                # Numeric array: field.data[i] is an INDEX into field.parts.
                # Access actual values via parts and convert to Python list
                # (numpy arrays are not instances of Sequence as required by add_array).
                vals = np.concatenate([field.parts[field.data[i]] for i in range(count)])
                writer.add_array(name, vals.tolist())

        else:
            print(f"  WARNING: unhandled field type {vtype} for {name}, skipping")

    # --- Tensors: register shapes, then write raw bytes ---
    print(f"\n  Registering {len(reader.tensors)} tensors...")
    tensor_raw_data = []

    for i, tensor in enumerate(reader.tensors):
        # Read raw bytes from mmap
        raw = bytes(reader.data[tensor.data_offset: tensor.data_offset + tensor.n_bytes])
        tensor_raw_data.append(raw)

        # Use dtype=int8 (not uint8) to prevent GGUFWriter from triggering
        # quant_shape_from_byte_shape, which would corrupt the shape.
        # The raw bytes are reinterpreted without conversion.
        raw_np = np.frombuffer(raw, dtype=np.int8).copy()

        # GGUFWriter.write_ti_data_to_file reverses shape when writing (shape[n_dims-1-j]).
        # GGUFReader gives shape in GGUF order [ne0, ne1, ...] (ne0 = fastest dim).
        # To get correct output we must reverse here so the double-reversal lands correctly.
        shape = list(reversed(tensor.shape.tolist()))

        writer.add_tensor(
            tensor.name,
            raw_np,
            raw_shape=shape,
            raw_dtype=tensor.tensor_type,
        )

        if (i + 1) % 50 == 0 or i + 1 == len(reader.tensors):
            print(f"  Registered {i + 1}/{len(reader.tensors)} tensors", end="\r")

    print()

    print(f"\n  Writing header + KV metadata...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    print(f"  Writing tensor info + data...")
    writer.write_tensors_to_file(progress=False)
    writer.close()

    out_size = os.path.getsize(dst_path) / 1024**2
    print(f"\n  Done! Output: {dst_path} ({out_size:.1f} MB)")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: fix_pruned_gemma.py <input.gguf> <output.gguf>")
        sys.exit(1)
    fix_pruned_gemma(sys.argv[1], sys.argv[2])
