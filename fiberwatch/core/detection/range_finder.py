"""有效检测范围计算。"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Optional

import numpy as np
from scipy.signal import find_peaks as _scipy_find_peaks, peak_widths

if TYPE_CHECKING:
    from ..models import DetectedEvent
    from .context import DetectionContext

from .peaks import find_dense_peak_cluster


def find_effective_end(
    ctx: DetectionContext, y: np.ndarray, peaks: List[dict]
) -> int:
    """找到有效检测的结束点。"""
    n = len(y)
    cfg = ctx.new_cfg
    end_region_start = int(n * (1 - cfg.end_region_ratio))

    # 密集峰群检测
    if peaks is not None:
        end_region_peaks = [p for p in peaks if p["index"] >= end_region_start]
        if len(end_region_peaks) >= 2:
            end_region_peaks.sort(key=lambda p: p["index"])
            cluster_start_idx = find_dense_peak_cluster(ctx, y, end_region_peaks)
            if cluster_start_idx is not None:
                last_peak = end_region_peaks[-1]
                noise_check_start = min(
                    last_peak.get("right_base_index", last_peak["index"]) + ctx.noise_check_offset_samples,
                    n - 1,
                )
                if check_enters_noise_region(ctx, y, noise_check_start):
                    return max(end_region_start, cluster_start_idx - 10)

    # 单峰 + 噪声检测
    end_region = y[end_region_start:]
    if len(end_region) == 0:
        return n - 1

    local_max_idx = np.argmax(end_region)
    global_max_idx = end_region_start + local_max_idx

    check_start = min(global_max_idx + ctx.noise_check_offset_samples, n)
    if check_start < n:
        tail = y[check_start:]
        if len(tail) >= ctx.noise_check_window_samples:
            tail_mean = np.mean(tail)
            tail_std = np.std(tail)
            if (
                tail_mean < cfg.noise_floor_db
                and tail_std > cfg.noise_std_threshold
            ):
                return global_max_idx

    return end_region_start


def check_enters_noise_region(
    ctx: DetectionContext, y: np.ndarray, start_idx: int
) -> bool:
    """检查从 start_idx 开始是否进入噪声区域。"""
    n = len(y)
    cfg = ctx.new_cfg
    check_end = min(start_idx + ctx.noise_check_window_samples, n)
    if check_end - start_idx < cfg.min_noise_segment_count:
        return False
    segment = y[start_idx:check_end]
    seg_mean = float(np.mean(segment))
    seg_std = float(np.std(segment))
    return seg_mean < cfg.noise_floor_db and seg_std > cfg.noise_std_threshold


def find_effective_start(ctx: DetectionContext, peaks: List[dict]) -> int:
    """找到有效检测的起点（跳过 skip_start_km 内的所有峰）。"""
    skip_km = ctx.new_cfg.skip_start_km
    start_idx = int(math.ceil(skip_km / ctx.sample_spacing_km))
    for p in peaks:
        if p["index"] * ctx.sample_spacing_km < skip_km:
            right = p.get("right_base_index", p["index"] + 1) + 1
            if right > start_idx:
                start_idx = right
    return start_idx


def get_scan_end(
    events: List[DetectedEvent],
    default_end: int,
    effective_start: int,
    lookback: int,
) -> int:
    """根据已有事件计算扫描终点。无事件时返回 default_end。"""
    if not events:
        return default_end
    earliest_idx = min(e.index for e in events)
    return max(effective_start, earliest_idx - lookback)


def filter_peaks_before(peaks: List[dict], cutoff: int) -> List[dict]:
    """只保留 index < cutoff 的峰。"""
    return [p for p in peaks if p["index"] < cutoff]


def find_last_end_peak(y: np.ndarray) -> Optional[int]:
    """
    查找光纤结束点：
    从 5% 之后开始检测，从后往前找第一个相对周围明显更高的峰。
    直接在 dB 曲线上检测，不使用平滑曲线，避免平滑造成假平台峰。
    高台阶只用于限制向前搜索的范围，避免过度搜索。

    参数:
        y: 已转换为 dB 的 OTDR 曲线（如 ``5 * log10(|centered_signal|)``）。

    返回:
        结束点在原始序列中的索引；若未检测到则返回 None。
    """
    signal_len = len(y)
    if signal_len < 10:
        return None

    # 高台阶末端可能落在 5% 附近，向前留一小段缓冲，避免把末端峰切掉。
    search_start = min(signal_len - 1, int(signal_len * 0.03))
    raw_signal = np.asarray(y, dtype=float)[search_start:]
    smooth_window = max(9, signal_len // 350)
    if smooth_window % 2 == 0:
        smooth_window += 1
    search_signal = raw_signal

    edge_margin = min(max(3, len(search_signal) // 20), smooth_window * 4)
    if len(search_signal) <= edge_margin * 2:
        return None

    tail_start = max(edge_margin, int(len(search_signal) * 0.65))
    tail_signal = search_signal[tail_start:]
    tail_baseline = np.median(tail_signal)
    tail_noise = np.median(np.abs(tail_signal - tail_baseline)) / 0.6745
    signal_range = max(np.percentile(search_signal, 98) - np.percentile(search_signal, 20), 1e-12)

    high_step_threshold = tail_baseline + max(tail_noise * 2.8, signal_range * 0.05)
    high_points = np.where(search_signal[edge_margin:len(search_signal) - edge_margin] >= high_step_threshold)[0]
    if len(high_points) == 0:
        return None

    high_points = high_points + edge_margin
    groups = np.split(high_points, np.where(np.diff(high_points) > 1)[0] + 1)
    high_step_group = groups[-1]
    terminal_width = max(80, signal_len // 25)
    step_left = max(edge_margin, int(high_step_group[-1] - terminal_width))
    step_right = min(len(search_signal) - edge_margin, int(high_step_group[-1] + max(6, signal_len // 120)))
    step_signal = search_signal[step_left:step_right]
    if len(step_signal) == 0:
        return None

    drop_boundary = int(high_step_group[-1] - step_left + 1)
    drop_boundary = max(1, min(drop_boundary, len(step_signal)))
    before_drop_signal = step_signal[:drop_boundary]

    step_baseline = np.median(step_signal)
    step_noise = np.median(np.abs(step_signal - step_baseline)) / 0.6745
    min_height = max(high_step_threshold, step_baseline + max(step_noise * 2.0, signal_range * 0.08))
    min_prominence = max(step_noise * 2.5, signal_range * 0.10)

    peaks, properties = _scipy_find_peaks(
        before_drop_signal,
        height=min_height,
        prominence=min_prominence,
        distance=max(3, signal_len // 180),
    )
    widths = peak_widths(before_drop_signal, peaks, rel_height=0.5)[0] if len(peaks) > 0 else []
    local_radius = max(50, signal_len // 70)
    qualified_peaks: List[int] = []

    def select_later_noise_peak(candidate_peak: int) -> int:
        """高台阶末端之后若进入噪声段且段中存在明显高峰，则取噪声段高峰。"""
        after_start = candidate_peak + max(20, signal_len // 100)
        after_end = len(search_signal) - edge_margin
        if after_start >= after_end - 10:
            return candidate_peak

        after_signal = search_signal[after_start:after_end]
        after_baseline = np.median(after_signal)
        after_noise = np.median(np.abs(after_signal - after_baseline)) / 0.6745
        after_range = max(np.percentile(after_signal, 98) - np.percentile(after_signal, 50), 1e-12)
        after_height = after_baseline + max(after_noise * 3.0, after_range * 0.55)
        after_prominence = max(after_noise * 3.0, after_range * 0.50)

        after_peaks, after_properties = _scipy_find_peaks(
            after_signal,
            height=after_height,
            prominence=after_prominence,
            distance=max(5, signal_len // 120),
        )
        if len(after_peaks) == 0:
            return candidate_peak

        after_widths = peak_widths(after_signal, after_peaks, rel_height=0.5)[0]
        valid_after_peaks: List[int] = []
        for peak_index, peak_height, prominence, width in zip(
            after_peaks,
            after_properties["peak_heights"],
            after_properties["prominences"],
            after_widths,
        ):
            if width < max(1, smooth_window // 8):
                continue
            if peak_height < after_height or prominence < after_prominence:
                continue
            valid_after_peaks.append(peak_index)

        if len(valid_after_peaks) == 0:
            return candidate_peak

        valid_values = after_signal[valid_after_peaks]
        return int(after_start + valid_after_peaks[int(np.argmax(valid_values))])

    def select_last_peak_before_drop() -> int:
        fallback_peaks, _ = _scipy_find_peaks(
            before_drop_signal,
            distance=max(3, signal_len // 180),
        )
        if len(fallback_peaks) > 0:
            return int(step_left + fallback_peaks[-1])
        return int(step_left + drop_boundary - 1)

    for peak_index, prominence, width in zip(peaks, properties["prominences"], widths):
        if width < max(1, smooth_window // 8):
            continue

        abs_peak_index = step_left + peak_index
        left = max(0, abs_peak_index - local_radius)
        right = min(len(search_signal), abs_peak_index + local_radius + 1)
        local_segment = search_signal[left:right]
        local_max = np.max(local_segment)
        local_baseline = np.median(local_segment)
        local_noise = np.median(np.abs(local_segment - local_baseline)) / 0.6745
        relative_height = search_signal[abs_peak_index] - local_baseline

        if search_signal[abs_peak_index] < local_max - max(local_noise * 0.8, signal_range * 0.05):
            continue
        if relative_height < max(min(local_noise * 3.0, signal_range * 0.65), signal_range * 0.08):
            continue
        if prominence < max(min(local_noise * 4.0, signal_range * 0.8), signal_range * 0.12):
            continue

        qualified_peaks.append(abs_peak_index)

    if len(qualified_peaks) == 0:
        candidate_peak = select_last_peak_before_drop()
        candidate_peak = select_later_noise_peak(candidate_peak)
        return int(search_start + candidate_peak)

    candidate_peak = qualified_peaks[-1]
    candidate_peak = select_later_noise_peak(candidate_peak)
    return int(search_start + candidate_peak)


def has_strong_noise_after_peak(
    y: np.ndarray, end_peak_index: Optional[int]
) -> bool:
    """
    判断结束峰之后是否存在剧烈波动，仅用于提示，不参与结束点选择。

    参数:
        y: 已转换为 dB 的 OTDR 曲线。
        end_peak_index: 结束峰索引；为 None 时直接返回 False。
    """
    if end_peak_index is None:
        return False

    before_start = max(0, end_peak_index - 300)
    before = y[before_start:end_peak_index]
    after = y[end_peak_index + 1:]

    if len(before) < 10 or len(after) < 10:
        return False

    before_std = np.std(np.diff(before))
    after_std = np.std(np.diff(after))
    return after_std > before_std * 2.0
