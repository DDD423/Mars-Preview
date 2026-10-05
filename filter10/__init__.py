"""filter1.0: local context-only span extraction."""

__version__ = "1.0.0"

def __getattr__(name):
    if name == "FilterHarness":
        from .harness import FilterHarness
        return FilterHarness
    raise AttributeError(name)
