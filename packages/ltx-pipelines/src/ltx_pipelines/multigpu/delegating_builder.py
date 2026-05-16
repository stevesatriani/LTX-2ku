"""Stub for single-GPU environments. DelegatingBuilder is used only as a type
hint in blocks.py; no methods are called on single-GPU inference."""
from typing import Generic, TypeVar

_M = TypeVar("_M")


class DelegatingBuilder(Generic[_M]):
    """Placeholder — real implementation lives in the multi-GPU plugin."""
    pass
