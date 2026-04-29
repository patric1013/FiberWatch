from typing import List

import numpy as np

from ..config.settings import DetectionConfig as DetectorConfig
from .detection import (
    DetectionContext,
    detect_bend,
    detect_dirty_connector,
    detect_severe_break,
    filter_peaks_before,
    find_effective_end,
    find_effective_start,
    find_last_end_peak,
    find_peaks,
    fit_linear_baseline,
    get_scan_end,
    identify_reflection_peaks,
)
from .models import DetectedEvent, DetectionResult


class Detector:
    """OTDR 事件检测器。

    初始化时构建 DetectionContext，detect() 编排各阶段检测。
    """

    def __init__(
        self,
        trace_db: np.ndarray,
        baseline: np.ndarray | None = None,
        config: DetectorConfig | None = None,
        *,
        sample_spacing_km: float,
    ) -> None:
        if config is None:
            config = DetectorConfig()

        self.ctx = DetectionContext(
            trace_db=trace_db,
            baseline=baseline,
            config=config,
            sample_spacing_km=sample_spacing_km,
        )

        # 向后兼容的便捷属性
        self.z = self.ctx.z
        self.distance_km = self.ctx.distance_km
        self.sample_spacing_km = self.ctx.sample_spacing_km
        self.cfg = self.ctx.cfg
        self.config = self.ctx.config
        self.trace_db = self.ctx.trace_db
        self.baseline = self.ctx.baseline
        self._new_cfg = self.ctx.new_cfg
        self._step_window_samples = self.ctx.step_window_samples

    # ── 暴露给外部（如 visualize.py 的 CNN 弯折检测）的内部方法 ──

    def _find_peaks(self, y: np.ndarray) -> List[dict]:
        return find_peaks(self.ctx, y)

    def _fit_linear_baseline(self, z: np.ndarray, y: np.ndarray) -> np.ndarray:
        return fit_linear_baseline(z, y)

    def _detect_bend(
        self,
        y: np.ndarray,
        effective_start: int,
        bend_end: int,
        step: int,
        peak_indices: List[int],
        all_peaks: List[dict],
        events: List[DetectedEvent],
        drop_fraction: float = 3.0,
    ) -> bool:
        return detect_bend(
            self.ctx,
            y,
            effective_start,
            bend_end,
            step,
            peak_indices,
            all_peaks,
            events,
            drop_fraction,
        )

    # ── 核心检测编排 ─────────────────────────────────────────────

    def detect(
        self,
        trace_db: np.ndarray,
        _fiber_index: float | None = None,
        _sample_rate_mhz: float | None = None,
    ) -> DetectionResult:
        """
        执行检测。

        检测顺序：
        0. 严重断纤
        1. 弯折（无反射峰的阶梯下降）
        2. 脏污
        补充：反射峰识别
        """
        ctx = self.ctx
        y = np.asarray(trace_db, dtype=float)
        cfg = ctx.new_cfg
        LOOKBACK = ctx.lookback_samples

        # 基线处理
        if ctx.baseline is None:
            baseline = fit_linear_baseline(ctx.z, y)
        else:
            baseline = ctx.baseline.copy()
            if len(baseline) != len(y):
                baseline = np.interp(
                    ctx.z,
                    np.linspace(0, ctx.z[-1], len(baseline)),
                    baseline,
                )

        residual = y - baseline

        # 1. 找所有反射峰
        all_peaks = find_peaks(ctx, y)
        peak_indices = [p["index"] for p in all_peaks]

        # 2. 确定有效检测范围
        # 优先使用结束峰检测；未识别到时回落到旧的范围估计逻辑
        end_peak_index = find_last_end_peak(y)
        if end_peak_index is None:
            effective_end = find_effective_end(ctx, y, all_peaks)
        else:
            effective_end = end_peak_index
        effective_start = find_effective_start(ctx, all_peaks)
        offset_samples = int(cfg.offset_samples_km / ctx.sample_spacing_km)

        # 3. 筛选有效峰
        valid_peaks = [
            p
            for p in all_peaks
            if effective_start <= p["index"] <= effective_end - offset_samples
            and p["peak_height_db"] >= cfg.peak_min_prominence_db
        ]

        events: List[DetectedEvent] = []
        processed_peak_indices: set[int] = set()
        bend_end = max(effective_start, effective_end - offset_samples)
        step = max(1, ctx.step_window_samples // 4)

        # ── 阶段0：严重断纤 ──
        detect_severe_break(ctx, y, valid_peaks, events, processed_peak_indices)

        # ── 阶段1：弯折 ──
        cur_end = get_scan_end(events, bend_end, effective_start, LOOKBACK)
        if cur_end > effective_start:
            detect_bend(
                ctx,
                y,
                effective_start,
                cur_end,
                step,
                peak_indices,
                all_peaks,
                events,
            )

        # ── 阶段2：脏污 ──
        cur_end = get_scan_end(events, bend_end, effective_start, LOOKBACK)
        if cur_end > effective_start:
            cur_peaks = filter_peaks_before(valid_peaks, cur_end)
            if cur_peaks:
                detect_dirty_connector(
                    ctx,
                    y,
                    cur_peaks,
                    peak_indices,
                    events,
                    processed_peak_indices,
                )

        events.sort(key=lambda e: e.index)

        # ── 补充阶段：反射峰识别 ──
        reflection_peaks = identify_reflection_peaks(
            ctx,
            all_peaks,
            events,
            baseline,
            effective_end,
        )

        return DetectionResult(
            events=events,
            distance_km=ctx.z,
            trace_db=y,
            baseline_db=baseline,
            residual_db=residual,
            reflection_peaks=reflection_peaks,
        )
