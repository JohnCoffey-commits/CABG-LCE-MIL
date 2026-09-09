#!/usr/bin/env python3

import math


def has_finite_nonzero_element(record) -> bool:
    """Use max-absolute value so an L2-norm square cannot underflow to zero."""

    try:
        max_abs = float(record.get("max_abs"))
    except (TypeError, ValueError):
        return False
    return bool(
        record.get("gradient_present")
        and record.get("finite")
        and math.isfinite(max_abs)
        and max_abs > 0.0
    )
