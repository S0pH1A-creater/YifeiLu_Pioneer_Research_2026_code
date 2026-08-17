#!/usr/bin/env python3
"""V3 Heston-only notebooks are option-implied; do not clone from Method A.

Use `patch_v3_heston_option_implied.py` to refresh V3 Heston / Heston–Merton
estimation. Method A cloning remains in V1/V2 `clone_heston_from_heston_merton.py`.
"""
from __future__ import annotations


def main() -> int:
    raise SystemExit(
        "Refusing to clone: V3 Heston notebooks are option-implied. "
        "Use patch_v3_heston_option_implied.py. Method A cloning stays in V1/V2."
    )


if __name__ == "__main__":
    raise SystemExit(main())
