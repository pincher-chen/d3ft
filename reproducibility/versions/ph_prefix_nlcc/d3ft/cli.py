"""Command-line interface for D3FT.

Usage::

    d3ft -m {energy,band} -c config.yaml

Mirrors Jrystal's CLI so that benchmarking scripts can swap one binary for the other.
"""
from __future__ import annotations

import argparse
import sys

from absl import logging

from .config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="d3ft",
        description="Direct Differentiable Density Functional Theory",
    )
    parser.add_argument(
        "-m", "--mode", choices=["energy", "band"], required=True,
        help="Calculation mode."
    )
    parser.add_argument(
        "-c", "--config", required=True,
        help="Path to a YAML config file (see config.yaml at the project root)."
    )
    parser.add_argument(
        "-o", "--output_dir", default=None,
        help="Directory to write results into (defaults to ./<crystal>_<mode>)."
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.output_dir is not None:
        config.save_dir = args.output_dir

    logging.set_verbosity(logging.INFO)
    logging.info("D3FT %s mode | crystal=%s", args.mode, config.crystal)

    if args.mode == "energy":
        from .calc.ground_state import calc as gs_calc
        gs_calc(config)
    elif args.mode == "band":
        from .calc.band_structure import calc as band_calc
        band_calc(config)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
