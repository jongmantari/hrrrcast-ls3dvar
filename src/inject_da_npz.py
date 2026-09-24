#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path

import yaml


def build_analysis_path(
    experiment_root: Path,
    experiment_name: str,
    cycle: datetime,
) -> Path:

    yyyymmdd = cycle.strftime("%Y%m%d")
    hh = cycle.strftime("%H")

    return (
        experiment_root
        / "runs"
        / experiment_name
        / f"{yyyymmdd}T{hh}Z"
        / "hrrrcast"
        / f"hrrr_{yyyymmdd}_{hh}_da.npz"
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--fcst-root",
        required=True,
    )

    parser.add_argument(
        "--init-time",
        required=True,
    )

    args = parser.parse_args()

    #
    # configuration
    #
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    da_cfg = cfg.get("da", {})

    #
    # DA disabled
    #
    if not da_cfg.get("enabled", False):
        print("DA injection disabled")
        return

    mode = da_cfg.get(
        "mode",
        "replace",
    )

    if mode != "replace":
        raise ValueError(
            f"Unsupported mode: {mode}"
        )

    preserve_original = da_cfg.get(
        "preserve_original",
        True,
    )

    #
    # cycle time
    #
    cycle = datetime.strptime(
        args.init_time,
        "%Y-%m-%dT%H",
    )

    yyyymmdd = cycle.strftime("%Y%m%d")
    hh = cycle.strftime("%H")

    #
    # DA experiment settings
    #
    experiment_root = Path(
        da_cfg["experiment_root"]
    )

    experiment_name = da_cfg[
        "experiment_name"
    ]

    analysis_npz = build_analysis_path(
        experiment_root,
        experiment_name,
        cycle,
    )

    if not analysis_npz.exists():
        raise FileNotFoundError(
            f"JEDI NPZ not found:\n"
            f"{analysis_npz}"
        )

    #
    # HRRRCast target NPZ
    #
    target_npz = (
        Path(args.fcst_root)
        / yyyymmdd
        / hh
        / f"hrrr_{yyyymmdd}_{hh}.npz"
    )

    if not target_npz.exists():
        raise FileNotFoundError(
            f"HRRRCast NPZ not found:\n"
            f"{target_npz}"
        )

    backup_npz = target_npz.with_suffix(
        ".npz.orig"
    )

    print()
    print("=" * 60)
    print("DA NPZ Injection")
    print("=" * 60)

    print()
    print("Source:")
    print(analysis_npz)

    print()
    print("Target:")
    print(target_npz)

    #
    # preserve original file
    #
    if preserve_original:

        if not backup_npz.exists():

            print()
            print(
                "Preserving original NPZ:"
            )
            print(backup_npz)

            shutil.copy2(
                target_npz,
                backup_npz,
            )

    #
    # replace target
    #
    if target_npz.exists() or target_npz.is_symlink():
        target_npz.unlink()

    #
    # inject analysis
    #
    os.symlink(
        analysis_npz,
        target_npz,
    )

    print()
    print("Injected DA analysis:")
    print(
        f"{target_npz} -> {analysis_npz}"
    )

    print()
    print("DONE")
    print()


if __name__ == "__main__":
    main()
