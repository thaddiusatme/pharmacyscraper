"""
Business Scraper shim package

This package provides a compatibility alias for the existing
`pharmacy_scraper` package. Any import of `business_scraper.*` is
transparently forwarded to `pharmacy_scraper.*` so both namespaces
remain compatible during the refactor.

Implementation details:
- Registers a MetaPathFinder/Loader that maps module names from
  `business_scraper` to `pharmacy_scraper`.
- When a `business_scraper.*` submodule is imported, the corresponding
  `pharmacy_scraper.*` module is imported and then aliased in
  `sys.modules` under the `business_scraper.*` name.
- This keeps relative imports inside the original package working, while
  giving callers the new top-level namespace.
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
from types import ModuleType
from typing import Optional


_OLD_NS = "pharmacy_scraper"
_NEW_NS = "business_scraper"


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, fullname: str, target_name: str) -> None:
        self.fullname = fullname
        self.target_name = target_name

    def create_module(self, spec):  # type: ignore[override]
        # Defer to default module creation; we'll alias in exec_module
        return None

    def exec_module(self, module: ModuleType) -> None:  # type: ignore[override]
        # Import the real target module and alias it under the new name
        target_mod = importlib.import_module(self.target_name)
        # Ensure the target is in sys.modules
        sys.modules[self.target_name] = target_mod
        # Alias the new name to the target module
        sys.modules[self.fullname] = target_mod
        # Update basic dunder attributes for friendlier introspection
        try:
            target_mod.__name__ = self.fullname  # type: ignore[attr-defined]
        except Exception:
            pass


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: Optional[list[str]],
        target: Optional[ModuleType] = None,
    ) -> Optional[importlib.machinery.ModuleSpec]:
        if not (fullname == _NEW_NS or fullname.startswith(_NEW_NS + ".")):
            return None
        # Map to the pharmacy_scraper module
        mapped = fullname.replace(_NEW_NS, _OLD_NS, 1)
        # If the target module exists or is findable, return a spec with our loader
        try:
            target_spec = importlib.util.find_spec(mapped)
        except (ImportError, ModuleNotFoundError):
            target_spec = None
        if target_spec is None:
            # No matching module in the old namespace; skip
            return None
        loader = _AliasLoader(fullname, mapped)
        return importlib.machinery.ModuleSpec(
            name=fullname,
            loader=loader,
            origin=target_spec.origin,
            is_package=target_spec.submodule_search_locations is not None,
        )


# Install the finder at high priority (just after builtins), but avoid duplicates
if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())


# Also make the top-level package immediately alias to the old one
try:
    _old = importlib.import_module(_OLD_NS)
    # Replace the current module object with the old namespace so that
    # `sys.modules['business_scraper'] is sys.modules['pharmacy_scraper']`
    # evaluates True. Preserve any attributes already placed on the shim
    # module by copying them onto the target if they don't exist there.
    _current = sys.modules.get(_NEW_NS)
    sys.modules[_NEW_NS] = _old
    if _current is not None and _current is not _old:
        for k, v in vars(_current).items():
            if not hasattr(_old, k):
                setattr(_old, k, v)
except Exception:
    # If the old namespace isn't importable at init time, the finder above
    # will still handle submodule imports when they occur.
    pass
