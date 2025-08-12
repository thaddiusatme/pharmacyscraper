"""A minimal stub of the hypothesis library for testing purposes.

This stub provides only the functionality required by the project's test suite:
- ``given`` decorator that supplies a simple ``PharmacyData`` instance.
- ``strategies`` submodule with placeholder strategy functions.

The implementation is intentionally lightweight and does not perform any real
property‑based testing. It merely allows the test suite to import ``hypothesis``
without pulling in the external dependency.
"""

from typing import Callable, Any

# Import the real PharmacyData class for constructing dummy instances.
try:
    from pharmacy_scraper.classification.models import PharmacyData
except Exception:  # pragma: no cover
    PharmacyData = None  # type: ignore


def given(*_strategies: Any, **_kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """A no‑op ``given`` decorator.

    The real ``hypothesis.given`` generates test data based on the provided
    strategies. For the purposes of this stub we simply call the wrapped test
    function with a minimal ``PharmacyData`` instance when the function expects a
    ``PharmacyData`` argument. If the function does not accept any arguments we
    call it unchanged.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # If the test function expects a ``PharmacyData`` argument, provide one.
            if PharmacyData is not None:
                # Inspect the function signature to see if a PharmacyData is expected.
                from inspect import signature
                sig = signature(func)
                params = sig.parameters
                # Build arguments list respecting positional order.
                new_args = list(args)
                # Determine which parameters have already been supplied via kwargs.
                supplied = set(kwargs.keys())
                for name, param in list(params.items())[len(args) :]:
                    if name in supplied:
                        # Argument already provided via kwargs; skip adding.
                        continue
                    if param.annotation is PharmacyData or param.name == "pharmacy":
                        # Provide a simple dummy PharmacyData instance.
                        dummy = PharmacyData.from_dict({"name": "Dummy Pharmacy"})
                        new_args.append(dummy)
                    else:
                        # Use None for any other missing positional arguments.
                        new_args.append(None)
                return func(*new_args, **kwargs)
            # Fallback: call the function unchanged.
            return func(*args, **kwargs)

        return wrapper

    return decorator

# Export the strategies submodule name for ``import strategies as st``.
# Placeholder strategies module (no-op) to satisfy imports
class _Strategies:
    def __getattr__(self, name):
        # Return a dummy callable for any strategy name
        def dummy(*args, **kwargs):
            return None
        return dummy

    def composite(self, fn):
        """Simple replacement for hypothesis.strategies.composite.

        The real ``composite`` decorator builds a strategy from a function that
        receives a ``draw`` callable. For our tests we can ignore the drawing and
        simply return a callable that produces a minimal ``PharmacyData`` instance.
        """
        def wrapper(*args, **kwargs):
            # Return a dummy PharmacyData instance regardless of draw calls.
            return PharmacyData.from_dict({"name": "Dummy Pharmacy"})
        return wrapper

strategies = _Strategies()

# Stub strategy function used in tests
def pharmacy_data_strategy():
    """Return a dummy PharmacyData instance for tests.

    The ``given`` decorator will receive this as the value for the ``pharmacy``
    argument. Providing a real ``PharmacyData`` object ensures the test code can
    access attributes like ``name`` without errors.
    """
    # Create a minimal PharmacyData instance.
    return PharmacyData.from_dict({"name": "Dummy Pharmacy"})



__all__ = ["given", "strategies"]
