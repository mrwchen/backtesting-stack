"""Mutable process-local runtime state for one backtest worker."""

from types import ModuleType
from typing import Optional

CURRENT_MODEL_FILE = ""
MODEL_MODULE: Optional[ModuleType] = None
