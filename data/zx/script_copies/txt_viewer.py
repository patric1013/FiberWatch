#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TXT数据交互式查看器
功能：
- 鼠标左键点击：添加标注点
- 鼠标左键拖动：框选放大
- 鼠标右键单击：重置视图
- 键盘左右箭头：切换上一个/下一个文件
- 键盘 R：重置视图
- 键盘 Delete/Backspace：删除最近的标注点
- ESC：退出
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


class TXTViewer:
    def __init__(self, txt_pairs):
        """
        txt_pairs: [(before_txt, after_txt, name), ...]
        """
        self.txt_pairs = list(txt_pairs)
        self.current_index = 0

        # 框选相关
        self.press_pos = None
        self.rect = None
        self.active_ax = None
        self.is_dragging = False

        # 悬停光标相关
        self.hover_lines = [None, None]
        self.hover_annotation = None
        self.current_hover_ax = None

        # 标注点相关
        self.annotations = [[], []]

        # 数据缓存
        self.data_cache = {}

        # 设置中文字体
        plt.rcParams["font.sans-serif"] = [
            "SimHei",
            "Microsoft YaHei",
            "Arial Unicode MS",
        ]
        plt.rcParams["axes.unicode_minus"] = False

        # 创建图形
        self.fig, self.axes = plt.subplots(2, 1, figsize=(14, 9))
        self.fig.canvas.manager.set_window_title(
            "TXT查看器 - 左键框选放大 | 右键重置 | 左右键切换"
        )

        # 绑定事件
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        # 加载第一个文件
        self.load_txt()

    def load_txt(self):
        """加载当前TXT文件并绘图"""
        # 清除旧的标注和悬停线
        self.clear_annotations()
        self.clear_hover_elements()

        for ax in self.axes:
            ax.clear()

        if not self.txt_pairs:
            self.axes[0].text(
                0.5, 0.5, "没有找到TXT文件", ha="center", va="center", fontsize=20
            )
            self.fig.canvas.draw()
            return

        before_txt, after_txt, name = self.txt_pairs[self.current_index]

        try:
            # 读取数据（txt文件只有dB值，每行一个数值）
            values_before = np.loadtxt(before_txt)
            values_after = np.loadtxt(after_txt)

            # 自动生成索引
            indices = np.arange(len(values_before))

            # 缓存数据用于查找最近点
            self.data_cache = {0: (indices, values_before), 1: (indices, values_after)}

            # 绘制降噪前
            self.axes[0].plot(indices, values_before, "b-", linewidth=0.5, alpha=0.8)
            self.axes[0].set_title(f"降噪前 (Before Denoising)", fontsize=12)
            self.axes[0].set_xlabel("采样点", fontsize=10)
            self.axes[0].set_ylabel("dB", fontsize=10)
            self.axes[0].grid(True, alpha=0.3)

            # 绘制降噪后
            self.axes[1].plot(indices, values_after, "r-", linewidth=0.5, alpha=0.8)
            self.axes[1].set_title(f"降噪后 (After Denoising)", fontsize=12)
            self.axes[1].set_xlabel("采样点", fontsize=10)
            self.axes[1].set_ylabel("dB", fontsize=10)
            self.axes[1].grid(True, alpha=0.3)

            # 设置总标题
            self.fig.suptitle(
                f"[{self.current_index + 1}/{len(self.txt_pairs)}] {name}\n"
                f"左键点击:标注 | 左键拖动:放大 | 右键:重置 | ←→:切换 | Del:删除标注 | R:重置 | ESC:退出",
                fontsize=11,
            )

            # 保存原始视图范围
            self.original_xlims = [ax.get_xlim() for ax in self.axes]
            self.original_ylims = [ax.get_ylim() for ax in self.axes]

        except Exception as e:
            self.axes[0].text(
                0.5, 0.5, f"无法加载TXT:\n{e}", ha="center", va="center", fontsize=12
            )

        plt.tight_layout()
        self.fig.canvas.draw()

    def get_active_ax(self, event):
        """获取鼠标所在的坐标轴"""
        for ax in self.axes:
            if event.inaxes == ax:
                return ax
        return None

    def get_ax_index(self, ax):
        """获取坐标轴索引"""
        for i, a in enumerate(self.axes):
            if a == ax:
                return i
        return None

    def find_nearest_point(self, ax, x_click, y_click):
        """在数据中找到最接近点击位置的点"""
        ax_idx = self.get_ax_index(ax)
        if ax_idx is None or ax_idx not in self.data_cache:
            return None, None

        indices, values = self.data_cache[ax_idx]

        # 归一化距离计算
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]

        # 计算归一化距离
        x_norm = (indices - x_click) / x_range
        y_norm = (values - y_click) / y_range
        distances = np.sqrt(x_norm**2 + y_norm**2)

        # 找到最近点
        min_idx = np.argmin(distances)

        # 只有当距离足够近时才返回
        if distances[min_idx] < 0.05:
            return indices[min_idx], values[min_idx]

        return None, None

    def add_annotation(self, ax, x, y):
        """添加标注点"""
        ax_idx = self.get_ax_index(ax)
        if ax_idx is None:
            return

        # 创建标注
        annotation = ax.annotate(
            f"({x:.1f}, {y:.2f})",
            xy=(x, y),
            xytext=(10, 10),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.7),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", color="red"),
            fontsize=9,
        )

        # 标记点
        marker = ax.plot(x, y, "ro", markersize=6, zorder=5)[0]

        # 保存标注信息
        self.annotations[ax_idx].append(
            {"annotation": annotation, "marker": marker, "x": x, "y": y}
        )

        self.fig.canvas.draw()

    def remove_last_annotation(self):
        """删除最近添加的标注"""
        for ax_idx in reversed(range(len(self.annotations))):
            if self.annotations[ax_idx]:
                ann_info = self.annotations[ax_idx].pop()
                ann_info["annotation"].remove()
                ann_info["marker"].remove()
                self.fig.canvas.draw()
                return

    def clear_annotations(self):
        """清除所有标注"""
        for ax_idx in range(len(self.annotations)):
            for ann_info in self.annotations[ax_idx]:
                ann_info["annotation"].remove()
                ann_info["marker"].remove()
            self.annotations[ax_idx] = []

    def update_hover_display(self, ax, x, y):
        """更新悬停显示（十字线和坐标）"""
        ax_idx = self.get_ax_index(ax)
        if ax_idx is None:
            return

        # 移除旧的悬停元素
        if self.hover_lines[ax_idx] is not None:
            for line in self.hover_lines[ax_idx]:
                line.remove()

        # 保存当前坐标轴限制
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # 绘制十字线
        vline = ax.plot([x, x], ylim, "k--", linewidth=0.5, alpha=0.5)[0]
        hline = ax.plot(xlim, [y, y], "k--", linewidth=0.5, alpha=0.5)[0]

        # 恢复坐标轴限制
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        self.hover_lines[ax_idx] = [vline, hline]

        # 更新或创建坐标显示
        if self.hover_annotation is not None and self.current_hover_ax == ax:
            self.hover_annotation.remove()

        self.hover_annotation = ax.text(
            0.02,
            0.98,
            f"X: {x:.1f}, Y: {y:.2f}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            fontsize=9,
        )
        self.current_hover_ax = ax

        self.fig.canvas.draw()

    def clear_hover_elements(self):
        """清除悬停元素"""
        for ax_idx in range(len(self.hover_lines)):
            if self.hover_lines[ax_idx] is not None:
                for line in self.hover_lines[ax_idx]:
                    line.remove()
                self.hover_lines[ax_idx] = None

        if self.hover_annotation is not None:
            self.hover_annotation.remove()
            self.hover_annotation = None
            self.current_hover_ax = None

    def on_press(self, event):
        """鼠标按下"""
        ax = self.get_active_ax(event)
        if ax is None or event.xdata is None or event.ydata is None:
            return

        # 右键重置视图
        if event.button == 3:
            for i, a in enumerate(self.axes):
                a.set_xlim(self.original_xlims[i])
                a.set_ylim(self.original_ylims[i])
            self.fig.canvas.draw()
            return

        # 左键按下
        if event.button == 1:
            self.press_pos = (event.xdata, event.ydata)
            self.active_ax = ax
            self.is_dragging = False

            # 创建矩形框
            self.rect = Rectangle(
                (event.xdata, event.ydata),
                0,
                0,
                linewidth=1,
                edgecolor="green",
                facecolor="green",
                alpha=0.3,
            )
            ax.add_patch(self.rect)

    def on_motion(self, event):
        """鼠标移动"""
        ax = self.get_active_ax(event)

        # 如果在框选中
        if self.press_pos is not None and self.rect is not None:
            if event.inaxes != self.active_ax:
                return
            if event.xdata is None or event.ydata is None:
                return

            self.is_dragging = True

            # 更新矩形框
            x0, y0 = self.press_pos
            x1, y1 = event.xdata, event.ydata

            self.rect.set_x(min(x0, x1))
            self.rect.set_y(min(y0, y1))
            self.rect.set_width(abs(x1 - x0))
            self.rect.set_height(abs(y1 - y0))

            self.fig.canvas.draw()
            return

        # 显示悬停坐标
        if ax is not None and event.xdata is not None and event.ydata is not None:
            self.update_hover_display(ax, event.xdata, event.ydata)
        else:
            self.clear_hover_elements()
            self.fig.canvas.draw()

    def on_release(self, event):
        """鼠标释放"""
        if self.press_pos is None or self.rect is None:
            return

        # 移除矩形框
        self.rect.remove()
        self.rect = None

        if event.xdata is None or event.ydata is None:
            self.press_pos = None
            self.active_ax = None
            self.is_dragging = False
            self.fig.canvas.draw()
            return

        # 判断是点击还是拖动
        if self.is_dragging:
            # 拖动：进行缩放
            x0, y0 = self.press_pos
            x1, y1 = event.xdata, event.ydata

            min_width = (
                abs(self.original_xlims[0][1] - self.original_xlims[0][0]) * 0.01
            )
            min_height = (
                abs(self.original_ylims[0][1] - self.original_ylims[0][0]) * 0.01
            )

            if abs(x1 - x0) > min_width and abs(y1 - y0) > min_height:
                self.active_ax.set_xlim(min(x0, x1), max(x0, x1))
                self.active_ax.set_ylim(min(y0, y1), max(y0, y1))
        else:
            # 点击：添加标注
            x, y = self.find_nearest_point(self.active_ax, event.xdata, event.ydata)
            if x is not None and y is not None:
                self.add_annotation(self.active_ax, x, y)

        self.press_pos = None
        self.active_ax = None
        self.is_dragging = False
        self.fig.canvas.draw()

    def on_key(self, event):
        """键盘事件"""
        if event.key == "right" or event.key == "down":
            self.current_index = (self.current_index + 1) % len(self.txt_pairs)
            self.load_txt()
        elif event.key == "left" or event.key == "up":
            self.current_index = (self.current_index - 1) % len(self.txt_pairs)
            self.load_txt()
        elif event.key == "escape":
            plt.close(self.fig)
        elif event.key == "r":
            for i, ax in enumerate(self.axes):
                ax.set_xlim(self.original_xlims[i])
                ax.set_ylim(self.original_ylims[i])
            self.fig.canvas.draw()
        elif event.key == "delete" or event.key == "backspace":
            self.remove_last_annotation()

    def show(self):
        """显示查看器"""
        plt.show()


def main():
    current_dir = Path(__file__).parent
    output_dir = current_dir / "output"

    # 查找所有 before/after TXT 对
    txt_pairs = []

    if output_dir.exists():
        before_files = sorted(output_dir.glob("*_before_denoise.txt"))
        for before_txt in before_files:
            name = before_txt.stem.replace("_before_denoise", "")
            after_txt = output_dir / f"{name}_after_denoise.txt"
            if after_txt.exists():
                txt_pairs.append((before_txt, after_txt, name))

    if not txt_pairs:
        print("没有找到 TXT 文件对！")
        print(f"查找目录: {output_dir}")
        print("需要先运行 run_with_plot.py 生成数据")
        return

    print(f"找到 {len(txt_pairs)} 组数据")
    print("\n操作说明:")
    print("  - 鼠标悬停：显示坐标")
    print("  - 鼠标左键点击：添加标注点（会自动吸附到最近的数据点）")
    print("  - 鼠标左键拖动：框选放大到选中区域")
    print("  - 鼠标右键单击：重置视图")
    print("  - 键盘 ←/→：切换文件")
    print("  - 键盘 R：重置视图")
    print("  - 键盘 Delete/Backspace：删除最近添加的一个标注")
    print("  - 键盘 ESC：退出")
    print("\n正在打开查看器...")

    viewer = TXTViewer(txt_pairs)
    viewer.show()


if __name__ == "__main__":
    main()
