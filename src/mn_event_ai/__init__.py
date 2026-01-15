"""
mn_event_ai

RSS → normalize/dedupe → weak labels → TextCNN classifier → incident clustering → forecasting + retrieval.

This package is deliberately region-agnostic. Region focus should be configured via feeds/configs,
not hard-coded into the core logic.
"""

from __future__ import annotations

__all__ = [
    "__version__",
]

__version__ = "0.1.0"
