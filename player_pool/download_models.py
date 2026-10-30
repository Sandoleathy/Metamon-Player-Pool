"""Download every Gen9OU RL checkpoint used by the local opponent pool.

Usage:
    python player_pool/download_models.py [--models Abra Kadabra3 ...]

Checks the HuggingFace cache first; only missing checkpoints are downloaded.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings

warnings.filterwarnings("ignore")

from player_pool.compat import patch_mkdir_permissions

patch_mkdir_permissions()

from metamon.rl.pretrained import get_pretrained_model

# name -> checkpoint epoch used by the pool
GEN9_POOL = {
    "SmallRLGen9Beta": 24,
    "Minikazam": 40,
    "Abra": 40,
    "Kadabra": 46,
    "Kadabra2": 44,
    "Kadabra3": 20,
    "Kadabra4": 50,
    "Alakazam": 8,
    "Superkazam": 50,
    "Kakuna": 34,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None)
    args = parser.parse_args()

    wanted = args.models or list(GEN9_POOL)
    unknown = [m for m in wanted if m not in GEN9_POOL]
    if unknown:
        print(f"unknown models: {unknown}", file=sys.stderr)
        return 2

    failures = []
    for i, name in enumerate(wanted, start=1):
        ckpt = GEN9_POOL[name]
        t0 = time.time()
        try:
            model = get_pretrained_model(name)
            path = model.get_path_to_checkpoint(ckpt)
            print(f"[{i}/{len(wanted)}] {name} epoch {ckpt} -> {path} ({time.time() - t0:.1f}s)")
        except Exception as exc:  # noqa: BLE001
            failures.append((name, repr(exc)))
            print(f"[{i}/{len(wanted)}] {name} FAILED: {exc!r}", file=sys.stderr)
        sys.stdout.flush()
    if failures:
        print(f"\n{len(failures)} failed:", failures, file=sys.stderr)
        return 1
    print("\nall checkpoints ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
