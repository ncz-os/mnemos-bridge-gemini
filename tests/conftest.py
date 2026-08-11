"""Test-only Google SDK types for environments without the mandatory dependency."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any


try:
    from google.genai import types as _installed_types  # noqa: F401
except ImportError:
    class _GeminiValue:
        def __init__(self, **values: Any) -> None:
            for key, value in values.items():
                setattr(self, key, value)

        def model_dump(self, *, exclude_none: bool = False) -> dict[str, Any]:
            values = vars(self)
            if exclude_none:
                values = {key: value for key, value in values.items() if value is not None}
            return {key: _dump(value, exclude_none) for key, value in values.items()}

    def _dump(value: Any, exclude_none: bool) -> Any:
        if isinstance(value, _GeminiValue):
            return value.model_dump(exclude_none=exclude_none)
        if isinstance(value, list):
            return [_dump(item, exclude_none) for item in value]
        if isinstance(value, dict):
            return {key: _dump(item, exclude_none) for key, item in value.items()}
        return value

    fake_types = SimpleNamespace(
        FunctionDeclaration=_GeminiValue,
        FunctionResponse=_GeminiValue,
        Part=_GeminiValue,
        Schema=_GeminiValue,
        Tool=_GeminiValue,
    )
    google_module = sys.modules.get("google", ModuleType("google"))
    genai_module = ModuleType("google.genai")
    genai_module.types = fake_types
    google_module.genai = genai_module
    sys.modules["google"] = google_module
    sys.modules["google.genai"] = genai_module
