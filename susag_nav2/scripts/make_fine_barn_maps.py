#!/usr/bin/env python3
"""Regenerate BARN_dataset/scaled_1/map_files_fine from map_files (2026-09-15).

navigation.launch.py loads maps from map_files_fine. BARN_dataset/ is
gitignored, so run this once on any new checkout:
    python3 make_fine_barn_maps.py [src_dir] [dst_dir]

Every 0.15 m map is resampled to 0.05 m by turning each cell into a 3x3 block.
This is exact: BARN obstacles are 0.15 m cells aligned to the map grid
(origin -4.5, 0.0), so no geometry changes. Maps that are not 0.15 m
(junction_split, open_dynamic at 0.05 m) are copied unchanged.

Why: the global costmap takes the map's resolution. At 0.15 m a single lidar
mark next to an obstacle face widened it by 15 cm and closed world 48's
7 cm-margin bottleneck (planner failures, spin recovery, collision).
"""
import pathlib
import re
import sys

import numpy as np

ROOT = pathlib.Path('/home/saran/robohouse_ws/src/BARN_dataset/scaled_1')


def read_pgm(path):
    data = path.read_bytes()
    parts = data.split(maxsplit=4)
    assert parts[0] == b'P5' and parts[3] == b'255', f'{path}: unexpected PGM header {parts[:4]}'
    w, h = int(parts[1]), int(parts[2])
    return np.frombuffer(parts[4][:w * h], dtype=np.uint8).reshape(h, w)


def write_pgm(path, img):
    h, w = img.shape
    path.write_bytes(f'P5\n{w} {h}\n255\n'.encode() + img.astype(np.uint8).tobytes())


def main():
    src = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'map_files'
    dst = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / 'map_files_fine'
    dst.mkdir(parents=True, exist_ok=True)
    upsampled = copied = 0
    for yaml_path in sorted(src.glob('yaml_*.yaml')):
        text = yaml_path.read_text()
        res = float(re.search(r'^resolution:\s*([\d.]+)', text, re.M).group(1))
        image = re.search(r'^image:\s*(\S+)', text, re.M).group(1)
        img = read_pgm(src / image)
        if abs(res - 0.15) < 1e-9:
            fine = np.repeat(np.repeat(img, 3, axis=0), 3, axis=1)
            assert np.array_equal(fine[1::3, 1::3], img)
            assert (fine == 0).sum() == 9 * (img == 0).sum()
            write_pgm(dst / image, fine)
            (dst / yaml_path.name).write_text(
                re.sub(r'^resolution:.*$', 'resolution: 0.050000', text, flags=re.M))
            upsampled += 1
        else:
            write_pgm(dst / image, img)
            (dst / yaml_path.name).write_text(text)
            copied += 1
    print(f'{dst}: upsampled {upsampled} maps 0.15 -> 0.05 m, copied {copied} unchanged')


if __name__ == '__main__':
    main()
