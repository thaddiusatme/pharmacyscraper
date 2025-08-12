"""
Site customizations to ensure third-party packages are prioritized over the
project root when Python resolves imports. This prevents accidental shadowing
by local folders named like real packages (e.g., a local `hypothesis/`).

Python automatically imports this module at startup if it is importable.
Since the current working directory (project root) is typically first on
sys.path, this runs early enough to adjust import order before pytest plugins
load.
"""
from __future__ import annotations

import os
import sys
import site
from typing import List

try:
    # Prevent pytest from auto-loading external plugins (e.g., Hypothesis),
    # which may import before our path adjustments and get shadowed by local dirs.
    # Tests can still opt-in per-module using pytest.importorskip or manual plugin loading.
    os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")

    root = os.path.abspath(os.getcwd())

    # Collect site-packages paths (system + user) that are present in sys.path
    site_paths: List[str] = []
    if hasattr(site, "getsitepackages"):
        for p in site.getsitepackages() or []:
            if isinstance(p, str) and p in sys.path:
                site_paths.append(p)
    if hasattr(site, "getusersitepackages"):
        usp = site.getusersitepackages()
        if isinstance(usp, str) and usp in sys.path:
            site_paths.append(usp)

    # Move site-packages to the very front (preserving internal order)
    for p in reversed(list(dict.fromkeys(site_paths))):
        try:
            sys.path.remove(p)
        except ValueError:
            pass
        sys.path.insert(0, p)

    # Move project root to the end so it cannot shadow third-party packages
    if root in sys.path:
        try:
            sys.path.remove(root)
        except ValueError:
            pass
        sys.path.append(root)
except Exception:
    # Best-effort: never block interpreter startup due to path tweaks
    pass
