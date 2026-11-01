"""Tiny RGB PNG writer so practice datasets do not need Pillow."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np


def write_png(path: Path, image: np.ndarray) -> None:
    rgb = np.ascontiguousarray(image, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected HxWx3 uint8 image, got {rgb.shape}")
    height, width, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    blob = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    Path(path).write_bytes(blob)


def hstack(images: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(images, axis=1)


def vstack(images: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(images, axis=0)
