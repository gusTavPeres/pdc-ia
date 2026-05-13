#!/usr/bin/env python3
"""
Patches GGUF files to change tokenizer.ggml.pre from 'gemma' to 'default'.

llama-cpp-python <= 0.3.20 does not recognize the BPE 'gemma' pre-tokenizer
used by pruned Gemma 3 models. Changing it to 'default' makes the model loadable
with a minor tokenization quality trade-off.

Usage: python3 patch_gemma_tokenizer.py <input.gguf> <output.gguf>
"""

import sys
import os
import struct
import mmap

GGUF_MAGIC = b"GGUF"
GGUF_TYPE_STRING = 8


def patch_gguf_pre_tokenizer(src_path, dst_path, old_value="gemma", new_value="default"):
    print(f"Patching: {os.path.basename(src_path)}")
    print(f"  {old_value!r} -> {new_value!r}")

    with open(src_path, "rb") as f:
        data = f.read()

    # Build search pattern:
    # uint64(key_len) + key_bytes + uint32(TYPE_STRING) + uint64(val_len) + val_bytes
    key = b"tokenizer.ggml.pre"
    key_len_bytes = struct.pack("<Q", len(key))
    type_bytes = struct.pack("<I", GGUF_TYPE_STRING)
    old_val_len_bytes = struct.pack("<Q", len(old_value))
    old_val_bytes = old_value.encode()

    pattern = key_len_bytes + key + type_bytes + old_val_len_bytes + old_val_bytes

    pos = data.find(pattern)
    if pos < 0:
        print(f"  Pattern not found — file may already be patched or use different format")
        return False

    print(f"  Found field at offset 0x{pos:x}")

    # Build replacement
    new_val_len_bytes = struct.pack("<Q", len(new_value))
    new_val_bytes = new_value.encode()
    replacement = key_len_bytes + key + type_bytes + new_val_len_bytes + new_val_bytes

    # The replacement has a different length — we need to rebuild the file
    # GGUF structure: header | kv_metadata | tensor_info | [alignment_padding] | tensor_data
    # Tensor data offsets are relative to data_offset, so we only need to update alignment padding.

    old_len = len(pattern)
    new_len = len(replacement)
    delta = new_len - old_len

    print(f"  Size delta: {delta:+d} bytes")

    # Reconstruct: data before pattern + replacement + data after pattern
    before = data[:pos]
    after = data[pos + old_len:]
    patched = before + replacement + after

    # GGUF alignment: find alignment value from metadata (gguf.general.alignment key)
    # Default alignment is 32 if not specified
    alignment = 32
    align_key = b"general.alignment"
    align_pattern = struct.pack("<Q", len(align_key)) + align_key + struct.pack("<I", 6)  # UINT32
    align_pos = data.find(align_pattern)
    if align_pos >= 0:
        val_pos = align_pos + len(align_pattern)
        alignment = struct.unpack("<I", data[val_pos:val_pos + 4])[0]
        print(f"  GGUF alignment: {alignment}")

    # Find where tensor data starts in original file.
    # We need to find where tensor data actually begins by scanning for it.
    # Approach: parse tensor count from header, parse all KV pairs, then all tensor infos
    # to find the data_offset.
    # Simpler: use GGUFReader to get the data_offset.
    from gguf import GGUFReader
    reader = GGUFReader(src_path, "r")
    original_data_offset = reader.data_offset
    print(f"  Original data_offset: 0x{original_data_offset:x}")

    # Calculate new header size (everything before tensor data in original, adjusted for delta)
    original_header_end = original_data_offset  # data starts here (post-alignment)
    # But we need the actual content size before padding:
    # The metadata section size changed by delta, so new_header_content_size = original_header_end - padding + delta
    # We'll compute new data_offset by finding the new aligned position

    # First, figure out where the old header content ended (before alignment padding)
    # data_offset = align(header_content_size, alignment)
    # So header_content_size <= data_offset, and data_offset % alignment == 0
    # With delta, new_header_content_size = old_header_content_size + delta
    # new_data_offset = align(new_header_content_size, alignment)

    # Estimate old header content size: it's <= original_data_offset
    # The actual size: original_data_offset - padding = original_data_offset - (original_data_offset % alignment or alignment)
    # But alignment is already a multiple, so padding = original_data_offset - actual_header_end
    # We can compute: actual_header_end from gguf structure

    # Simpler: just write to file and use numpy to fill tensor data
    # The patched bytes up to original_data_offset + delta might have wrong alignment padding
    # We need to:
    # 1. Write metadata (bytes 0 to data_offset+delta, roughly)
    # 2. Add padding to reach next aligned offset
    # 3. Copy tensor data from original file

    # Get tensor data from original
    tensor_data_size = os.path.getsize(src_path) - original_data_offset
    print(f"  Tensor data size: {tensor_data_size / 1024**2:.1f} MB")

    # Compute new header content end (before padding)
    # Original: bytes 0..original_data_offset contains header+metadata+padding
    # After patch: bytes 0..original_data_offset+delta contains header+metadata(modified)+?padding
    new_content_end = original_data_offset + delta
    # Round up to alignment
    new_data_offset = ((new_content_end + alignment - 1) // alignment) * alignment
    new_padding = new_data_offset - new_content_end
    print(f"  New data_offset: 0x{new_data_offset:x} (padding: {new_padding} bytes)")

    # Build final file:
    # - patched metadata (first original_data_offset + delta bytes, then re-padded)
    metadata_section = patched[:new_content_end] + b"\x00" * new_padding

    print(f"  Writing {dst_path}...")
    with open(dst_path, "wb") as out:
        out.write(metadata_section)
        # Copy tensor data from original file
        with open(src_path, "rb") as orig:
            orig.seek(original_data_offset)
            copied = 0
            chunk_size = 64 * 1024 * 1024  # 64MB chunks
            while True:
                chunk = orig.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)
                copied += len(chunk)
                pct = copied / tensor_data_size * 100
                print(f"\r  Copying tensors: {pct:.1f}% ({copied/1024**2:.0f}/{tensor_data_size/1024**2:.0f} MB)", end="", flush=True)

    print(f"\n  Done! Output: {dst_path}")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: patch_gemma_tokenizer.py <input.gguf> <output.gguf>")
        sys.exit(1)

    ok = patch_gguf_pre_tokenizer(sys.argv[1], sys.argv[2])
    sys.exit(0 if ok else 1)
