"""检测子模块汇总导出。"""

from .baseline import fit_linear_baseline
from .bend import check_step_drop, detect_bend
from .context import DetectionContext
from .dirty_connector import detect_dirty_connector
from .fiber_break import (
    detect_normal_break,
    detect_severe_break,
    detect_small_peak_break,
    find_descent_end,
)
from .peaks import find_dense_peak_cluster, find_peaks, identify_reflection_peaks
from .range_finder import (
    filter_peaks_before,
    find_effective_end,
    find_effective_start,
    find_last_end_peak,
    get_scan_end,
    has_strong_noise_after_peak,
)

__all__ = [
    "DetectionContext",
    "fit_linear_baseline",
    "find_peaks",
    "find_dense_peak_cluster",
    "identify_reflection_peaks",
    "find_effective_end",
    "find_effective_start",
    "find_last_end_peak",
    "has_strong_noise_after_peak",
    "get_scan_end",
    "filter_peaks_before",
    "check_step_drop",
    "detect_bend",
    "detect_severe_break",
    "detect_normal_break",
    "detect_small_peak_break",
    "find_descent_end",
    "detect_dirty_connector",
]
