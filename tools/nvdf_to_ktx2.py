#!/usr/bin/env python3
"""Converts a Nubis voxel cloud (NVDF, Guerrilla's Nubis Voxel Cloud Pack, SIGGRAPH 2023) from its
TGA slices to one 3D KTX2 texture that `VoxelCloud` (bevy_pbr atmosphere) loads.

Input: a directory with `modeling_data.NNN.tga` (RGB: dimensional profile, detail type, density
scale) and `field_data.NNN.tga` (R: signed distance), one slice per height, slice 001 at the
bottom; 512 x 512 x 64 in the pack. The slices are z-up (Houdini), image rows along -Y.

The volume is cropped to the cloud (the voxels with a profile > 0) plus a margin of
`--margin` voxels (default 4), each side rounded up to a multiple of 8 for the mip chain; the pack
holds one cloud in a 4 km x 500 m x 4 km domain. `--no-crop` keeps the whole domain. The voxel size
is unchanged (the pack's 7.8125 m), so `VoxelCloud` sizes its box from the texture.

Output: RGBA8 UNORM (linear) 3D texture, R profile, G type, B density scale, A signed distance
(-256 m .. 4096 m mapped to 0..1, as in the pack). Y is up: width = X, height = the slices
(row 0 = the bottom slice, so v is the normalized height), depth = the image rows. Full mip
chain, 2x2x2 box filter per level (the shader picks the level by distance). Written
uncompressed, then supercompressed with `ktx deflate --zstd` (KTX-Software) when available.

Usage: nvdf_to_ktx2.py [--no-crop] [--margin N] <slice_dir> <out.ktx2>
"""
import pathlib
import shutil
import struct
import subprocess
import sys

import numpy as np
from PIL import Image

VK_FORMAT_R8G8B8A8_UNORM = 37


def load(slice_dir: pathlib.Path) -> np.ndarray:
    def stack(kind):
        files = sorted(slice_dir.glob(f"{kind}.*.tga"), key=lambda p: int(p.suffixes[-2][1:]))
        if not files:
            sys.exit(f"no {kind}.*.tga in {slice_dir}")
        return np.stack([np.asarray(Image.open(f).convert("RGBA")) for f in files])

    modeling = stack("modeling_data")  # (slices, rows, cols, 4)
    field = stack("field_data")
    if modeling.shape != field.shape:
        sys.exit(f"modeling {modeling.shape} and field {field.shape} differ")
    volume = modeling.copy()
    volume[..., 3] = field[..., 0]
    # (slice = height, row, col) -> texture (depth = row, height = slice, width = col).
    return np.ascontiguousarray(volume.transpose(1, 0, 2, 3))


def crop(volume: np.ndarray, margin: int) -> np.ndarray:
    """Crops (depth, height, width, 4) to the voxels with a profile > 0 plus `margin`, each side
    rounded up to a multiple of 8 (centred on the cloud, within the domain)."""
    occupied = np.nonzero(volume[..., 0] > 0)
    if not occupied[0].size:
        sys.exit("empty volume: no voxel has a profile > 0")
    slices = []
    for axis, idx in enumerate(occupied):
        n = volume.shape[axis]
        lo, hi = max(int(idx.min()) - margin, 0), min(int(idx.max()) + 1 + margin, n)
        size = min(-(-(hi - lo) // 8) * 8, n)
        lo = max(min(lo - (size - (hi - lo)) // 2, n - size), 0)
        slices.append(slice(lo, lo + size))
    return np.ascontiguousarray(volume[tuple(slices)])


def downsample(level: np.ndarray) -> np.ndarray:
    """2x2x2 box filter; an axis already at 1 texel stays 1."""
    v = level.astype(np.float32)
    for axis in range(3):
        n = v.shape[axis]
        if n > 1:
            v = v.take(range(0, n - n % 2, 2), axis=axis) * 0.5 + v.take(range(1, n - n % 2 + 1, 2), axis=axis) * 0.5
    return np.clip(np.rint(v), 0, 255).astype(np.uint8)


def dfd_rgba8_linear() -> bytes:
    samples = b"".join(
        struct.pack("<HBBIII", bit_offset, 7, channel, 0, 0, 255)
        for bit_offset, channel in ((0, 0), (8, 1), (16, 2), (24, 15))
    )
    block_size = 24 + len(samples)
    block = struct.pack(
        "<IIBBBB4B8B",
        0,  # vendor 0, descriptor type 0 (basic)
        2 | (block_size << 16),  # version 2, block size
        1,  # color model RGBSDA
        1,  # primaries BT.709
        1,  # transfer linear
        0,  # flags: straight alpha
        0, 0, 0, 0,  # texel block 1x1x1x1
        4, 0, 0, 0, 0, 0, 0, 0,  # 4 bytes per texel
    ) + samples
    return struct.pack("<I", 4 + len(block)) + block


def write_ktx2(path: pathlib.Path, levels: list) -> None:
    depth, height, width, _ = levels[0].shape
    dfd = dfd_rgba8_linear()
    header_size = 80
    index_size = 24 * len(levels)
    dfd_offset = header_size + index_size
    data_start = dfd_offset + len(dfd)
    # Level data runs from the smallest level to the largest, each 4-byte aligned.
    offsets = [0] * len(levels)
    cursor = data_start
    for i in reversed(range(len(levels))):
        cursor = (cursor + 3) & ~3
        offsets[i] = cursor
        cursor += levels[i].nbytes
    out = bytearray(cursor)
    out[0:12] = b"\xabKTX 20\xbb\r\n\x1a\n"
    struct.pack_into(
        "<9I", out, 12, VK_FORMAT_R8G8B8A8_UNORM, 1, width, height, depth, 0, 1, len(levels), 0
    )
    struct.pack_into("<4I2Q", out, 48, dfd_offset, len(dfd), 0, 0, 0, 0)
    for i, level in enumerate(levels):
        struct.pack_into("<3Q", out, header_size + 24 * i, offsets[i], level.nbytes, level.nbytes)
        out[offsets[i] : offsets[i] + level.nbytes] = level.tobytes()
    out[dfd_offset : dfd_offset + len(dfd)] = dfd
    path.write_bytes(out)


def main() -> None:
    args = sys.argv[1:]
    cropped, margin = True, 4
    if "--no-crop" in args:
        args.remove("--no-crop")
        cropped = False
    if "--margin" in args:
        i = args.index("--margin")
        margin = int(args[i + 1])
        del args[i : i + 2]
    if len(args) != 2:
        sys.exit(__doc__)
    slice_dir, out = pathlib.Path(args[0]), pathlib.Path(args[1])
    volume = load(slice_dir)
    levels = [crop(volume, margin) if cropped else volume]
    while max(levels[-1].shape[:3]) > 1:
        levels.append(downsample(levels[-1]))
    depth, height, width, _ = levels[0].shape
    print(f"{slice_dir}: {width} x {height} x {depth}, {len(levels)} levels")
    out.parent.mkdir(parents=True, exist_ok=True)
    write_ktx2(out, levels)
    ktx = shutil.which("ktx")
    if ktx:
        subprocess.run([ktx, "validate", str(out)], check=True)
        subprocess.run([ktx, "deflate", "--zstd", "19", str(out), str(out)], check=True)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
