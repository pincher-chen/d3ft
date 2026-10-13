"""Locate Jrystal-bundled geometry files.

D3FT does not vendor its own ``geometry/`` directory. Instead it resolves crystal
names through :func:`jrystal.get_pkg_path` so that ``config.crystal = "si"`` finds
``<jrystal_install>/geometry/si.xyz``.
"""
from __future__ import annotations

from pathlib import Path

import jrystal
from jrystal._src.crystal import Crystal


def find_geometry(name: str) -> Path:
    """Return the path to a ``geometry/<name>.xyz`` file shipped with Jrystal."""
    candidate = Path(jrystal.get_pkg_path()) / "geometry" / f"{name}.xyz"
    if not candidate.exists():
        raise FileNotFoundError(
            f"Geometry {name!r} not found at {candidate}. Available files: "
            f"{sorted(p.name for p in candidate.parent.glob('*.xyz'))}"
        )
    return candidate


def load_crystal(name: str, spin: int = 0) -> Crystal:
    return Crystal.create_from_file(str(find_geometry(name)), spin=spin)
