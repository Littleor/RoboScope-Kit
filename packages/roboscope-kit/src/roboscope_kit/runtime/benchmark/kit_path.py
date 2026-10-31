# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""The directory simulator workers put on ``PYTHONPATH`` to import the kit."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

KIT_PACKAGE = Path(__file__).resolve().parents[2]


def kit_import_root() -> Path:
    """Return a directory whose only importable package is ``roboscope_kit``.

    Workers run in the simulator's own Python. A source tree's ``src`` holds only the kit and is
    used as is. A pip install's parent is a ``site-packages`` whose other packages would shadow the
    simulator's, so the kit is linked into a directory of its own instead.
    """

    parent = KIT_PACKAGE.parent
    if all(entry.name.startswith(("roboscope_kit", "__pycache__", ".")) for entry in parent.iterdir()):
        return parent
    digest = hashlib.sha256(str(KIT_PACKAGE).encode("utf-8")).hexdigest()[:12]
    root = Path(tempfile.gettempdir()) / f"roboscope-kit-import-{digest}"
    link = root / KIT_PACKAGE.name
    if not (link.is_symlink() and link.resolve() == KIT_PACKAGE):
        root.mkdir(parents=True, exist_ok=True)
        staging = root / f".{KIT_PACKAGE.name}.{os.getpid()}"
        staging.unlink(missing_ok=True)
        staging.symlink_to(KIT_PACKAGE, target_is_directory=True)
        os.replace(staging, link)
    return root


__all__ = ["KIT_PACKAGE", "kit_import_root"]
