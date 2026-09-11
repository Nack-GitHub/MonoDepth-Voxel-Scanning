"""Tiny string->class registry. Avoids importing torch just to run the GT pipeline."""

from __future__ import annotations

import importlib


def load_class(registry: dict[str, str], name: str, *, kind: str):
    try:
        target = registry[name]
    except KeyError:
        raise ValueError(f"Unknown {kind} '{name}'. Known: {sorted(registry)}") from None
    module_name, _, cls_name = target.partition(":")
    return getattr(importlib.import_module(module_name), cls_name)
