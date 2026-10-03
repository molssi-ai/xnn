"""Shared test set-up."""
# xnn.gnn imports e3nn so that e3nn 0.4.4 also loads under torch >= 2.6; test
# modules import e3nn directly, so do it before any of them
try:
    import xnn.gnn  # noqa: F401
except ImportError:
    pass
