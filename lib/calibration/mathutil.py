from __future__ import annotations

import math
from typing import Iterable


def filtered_average(values: Iterable[float], sigma_limit: float) -> float:
    values_list = list(values)
    if not values_list:
        return 0.0
    ex2sum = sum(val * val for val in values_list)
    exsum = sum(values_list)
    ex = exsum / len(values_list)
    ex2 = ex2sum / len(values_list)
    sigma = math.sqrt(max(ex2 - ex * ex, 0.0))
    limit = sigma_limit * sigma
    filt = [val for val in values_list if abs(val - ex) <= limit]
    if not filt:
        return ex
    return sum(filt) / len(filt)


def sgn(val: float) -> int:
    return (val > 0) - (val < 0)
