from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def to_jsonable(value: Any) -> Any:
    """Convert numpy/torch-style values into JSON-native Python types."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return to_jsonable(item())
        except (TypeError, ValueError):
            pass

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return to_jsonable(tolist())
        except (TypeError, ValueError):
            pass

    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item_value) for key, item_value in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]

    return value
