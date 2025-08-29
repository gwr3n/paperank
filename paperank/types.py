"""Common type aliases for the paperank package."""
from typing import Union, Literal

# Progress indicator type used across the package:
# - False: no progress
# - True: basic progress (or fallback)
# - 'tqdm': explicitly request tqdm progress bar
# - int: print every N iterations/steps
ProgressType = Union[Literal[False, True, 'tqdm'], int]

__all__ = ["ProgressType"]
