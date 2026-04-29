"""脏污连接器检测。"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from .context import DetectionContext

import numpy as np

from ..models import DetectedEvent
from .bend import check_step_drop


def detect_dirty_connector(
    ctx: DetectionContext,
    y: np.ndarray,
    valid_peaks: List[dict],
    peak_indices: List[int],
    events: List[DetectedEvent],
    processed_peak_indices: set,
) -> bool:
    """第二阶段：脏污检测。"""
    cfg = ctx.new_cfg
    for peak in valid_peaks:
        peak_idx = peak["index"]
        if peak_idx in processed_peak_indices:
            continue

        peak_height = peak["peak_height_db"]
        z_km = float(ctx.z[peak_idx])

        if peak_height <= cfg.peak_high_threshold_db:
            continue

        check_idx = peak.get("right_base_index", peak_idx + 5)
        has_drop, drop_db = check_step_drop(ctx, y, check_idx, peak_indices)
        events.append(
            DetectedEvent(
                kind="dirty_connector",
                z_km=z_km,
                magnitude_db=float(drop_db) if has_drop else 0.0,
                reflect_db=float(peak_height),
                index=peak_idx,
            )
        )
        processed_peak_indices.add(peak_idx)
        return True

    return False
