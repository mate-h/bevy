#!/usr/bin/env python3
"""Export an EsotericaFilterFitter FTBL binary into Bevy's raw coefficient payload.

Esoterica writes a 144-byte header then float4 coeffs. Bevy embeds only the payload.

Example:

    python3 crates/bevy_pbr/assets/export_esoterica_filter_table.py \\
        ~/git/EsotericaFilterFitter/ReflectionProbeTable_ggx_esoterica_cube.bin \\
        crates/bevy_pbr/assets/manson_sloan_poly_table.bin
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

FTBL_MAGIC = 0x4C425446
HEADER_SIZE = 144


def export(src: Path, dst: Path) -> None:
    raw = src.read_bytes()
    if len(raw) < HEADER_SIZE:
        raise SystemExit(f"{src}: file too small ({len(raw)} bytes)")

    (
        magic,
        version,
        num_levels,
        num_parameters,
        num_active_coefficients,
        num_taps_per_axis,
        _projection,
        num_axes,
        num_indices,
        num_float4,
        payload_offset,
        payload_bytes,
    ) = struct.unpack_from("<12I", raw, 0)

    if magic != FTBL_MAGIC:
        raise SystemExit(f"{src}: bad magic {magic:#x}, expected FTBL")
    if version != 4:
        raise SystemExit(f"{src}: unsupported FTBL version {version}")

    shape = raw[48:64].split(b"\0", 1)[0].decode("ascii", errors="replace")
    widths = struct.unpack_from("<7f", raw, 112)

    expected_indices = (num_axes * num_taps_per_axis) // 4
    expected_float4 = num_levels * num_parameters * num_active_coefficients * num_indices
    if num_indices != expected_indices or num_float4 != expected_float4:
        raise SystemExit(
            f"{src}: shape fields disagree "
            f"(indices={num_indices} expected {expected_indices}, "
            f"float4={num_float4} expected {expected_float4})"
        )

    # Bevy currently embeds the Esoterica const_8 cubemap layout.
    if (
        num_levels,
        num_parameters,
        num_active_coefficients,
        num_taps_per_axis,
        num_axes,
        num_indices,
    ) != (7, 5, 1, 8, 3, 6):
        raise SystemExit(
            f"{src}: unsupported shape '{shape}' "
            f"(levels={num_levels} params={num_parameters} coeffs={num_active_coefficients} "
            f"taps={num_taps_per_axis} axes={num_axes} indices={num_indices}); "
            "Bevy expects const_8 cubemap (7,5,1,8,3,6)"
        )

    if payload_offset != HEADER_SIZE or payload_bytes != num_float4 * 16:
        raise SystemExit(
            f"{src}: unexpected payload layout "
            f"(offset={payload_offset} bytes={payload_bytes})"
        )

    payload = raw[payload_offset : payload_offset + payload_bytes]
    if len(payload) != payload_bytes:
        raise SystemExit(f"{src}: truncated payload")

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(payload)

    perceptual = [w**0.5 for w in widths]
    print(f"shape={shape} float4={num_float4} wrote {dst} ({len(payload)} bytes)")
    print("alpha widths:", ", ".join(f"{w:.6g}" for w in widths))
    print("perceptual:  ", ", ".join(f"{r:.6g}" for r in perceptual))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path, help="Esoterica ReflectionProbeTable_*.bin")
    parser.add_argument(
        "dst",
        type=Path,
        nargs="?",
        default=Path(__file__).with_name("manson_sloan_poly_table.bin"),
        help="Bevy payload path (default: next to this script)",
    )
    args = parser.parse_args(argv)
    export(args.src, args.dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
