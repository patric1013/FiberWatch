#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OTDR数据处理程序
功能：读取bin文件，使用chuli7方法降噪，保存降噪前后文件，并绘制对比图
"""

import struct
import numpy as np
import pandas as pd
import pywt
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def read_bin_as_word32_array(bin_file_path, byte_order='big', auto_detect=True):
    """
    读取bin文件，按照32位无符号整数（WORD32）数组格式解析
    """
    values = []

    with open(bin_file_path, 'rb') as f:
        data = f.read()

    num_values = len(data) // 4

    if num_values == 0:
        return values

    if auto_detect:
        values_little = []
        values_big = []
        for i in range(min(10, num_values)):
            offset = i * 4
            values_little.append(struct.unpack('<I', data[offset:offset+4])[0])
            values_big.append(struct.unpack('>I', data[offset:offset+4])[0])

        def reasonable_score(vals):
            score = 0
            for v in vals[:5]:
                if 0 < v < 100000:
                    score += 1
            return score

        score_little = reasonable_score(values_little)
        score_big = reasonable_score(values_big)
        byte_order = 'big' if score_big >= score_little else 'little'

    fmt = '<I' if byte_order == 'little' else '>I'

    for i in range(num_values):
        offset = i * 4
        value = struct.unpack(fmt, data[offset:offset+4])[0]
        values.append(value)

    return values


def bin_to_csv(values):
    """
    将bin数据转换为处理后的数值数组
    剔除前22个数据，根据第3、4个值选择公式处理
    """
    if len(values) < 23 or len(values) < 4:
        return None, None

    third_value = values[2]
    fourth_value = values[3]

    processed_values = []
    indices = []

    if third_value == 0:
        for idx in range(22, len(values)):
            processed_values.append(values[idx])
            indices.append(idx - 22)
    elif fourth_value > 10000:
        for idx in range(22, len(values)):
            divided = values[idx] / third_value
            result = divided * 1.75 / 16384 - 0.875
            processed_values.append(result)
            indices.append(idx - 22)
    else:
        for idx in range(22, len(values)):
            divided = values[idx] / third_value
            result = divided * 1.75 / 4096 - 0.875
            processed_values.append(result)
            indices.append(idx - 22)

    return indices, processed_values


def wavelet_denoise_hard_advanced(signal, wavelet='db8', level=6):
    """
    高级小波降噪函数 - 使用自适应硬阈值 (来自chuli7)
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    denoised_coeffs = [coeffs[0]]

    for i in range(1, len(coeffs)):
        sigma = np.median(np.abs(coeffs[i])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(coeffs[i])))
        denoised_coeff = pywt.threshold(coeffs[i], threshold, mode='hard')
        denoised_coeffs.append(denoised_coeff)

    denoised_signal = pywt.waverec(denoised_coeffs, wavelet)

    if len(denoised_signal) != len(signal):
        if len(denoised_signal) > len(signal):
            denoised_signal = denoised_signal[:len(signal)]
        else:
            denoised_signal = np.pad(denoised_signal, (0, len(signal) - len(denoised_signal)), 'edge')

    return denoised_signal


def denoise_with_tail_preservation(signal_data, wavelet='db8', level=6):
    """
    使用chuli7的方法进行降噪：检测尾端反射峰，只降噪前部分
    """
    n = min(200, len(signal_data))
    mean_val = np.mean(signal_data[-n:])
    centered_signal = signal_data - mean_val

    tail_start_index = int(0.7 * len(centered_signal))
    tail_signal = centered_signal[tail_start_index:]

    peak_indices, _ = find_peaks(tail_signal, height=np.max(tail_signal) * 0.6)

    if len(peak_indices) > 0:
        tail_end_index = peak_indices[0] + tail_start_index
    else:
        tail_end_index = len(centered_signal)

    signal_for_denoising = centered_signal[:tail_end_index]

    denoised_signal = wavelet_denoise_hard_advanced(signal_for_denoising, wavelet, level)

    signal_power = np.mean(signal_for_denoising**2)
    noise = signal_for_denoising - denoised_signal
    noise_power = np.mean(noise**2)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')

    final_signal = np.concatenate((denoised_signal, centered_signal[tail_end_index:]))

    db_before = 5 * np.log10(np.abs(centered_signal) + 1e-10)
    db_after = 5 * np.log10(np.abs(final_signal) + 1e-10)

    return centered_signal, final_signal, db_before, db_after, snr


def plot_comparison(indices, db_before, db_after, snr, output_path):
    """
    绘制降噪前后对比图
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(indices, db_before, 'b-', linewidth=0.5, alpha=0.8)
    axes[0].set_title('降噪前 (Before Denoising)', fontsize=14)
    axes[0].set_xlabel('采样点', fontsize=12)
    axes[0].set_ylabel('dB', fontsize=12)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(indices, db_after, 'r-', linewidth=0.5, alpha=0.8)
    axes[1].set_title(f'降噪后 (After Denoising) - SNR: {snr:.2f} dB', fontsize=14)
    axes[1].set_xlabel('采样点', fontsize=12)
    axes[1].set_ylabel('dB', fontsize=12)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"对比图已保存: {output_path}")


def process_single_file(bin_file_path, output_dir=None):
    """
    处理单个bin文件
    """
    bin_path = Path(bin_file_path)
    if output_dir is None:
        output_dir = bin_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    base_name = bin_path.stem

    print(f"\n{'='*60}")
    print(f"处理文件: {bin_file_path}")
    print(f"{'='*60}")

    # 1. 读取bin文件
    print("1. 读取bin文件...")
    values = read_bin_as_word32_array(bin_file_path)
    print(f"   读取到 {len(values)} 个数据点")

    # 2. 转换数据
    print("2. 转换数据...")
    indices, processed_values = bin_to_csv(values)
    if indices is None:
        print("   错误: 数据不足")
        return False
    print(f"   处理后 {len(processed_values)} 个数据点")

    # 3. 执行降噪 (使用chuli7方法)
    print("3. 执行小波降噪 (chuli7方法: 保留尾端反射峰)...")
    signal_array = np.array(processed_values)
    centered, denoised, db_before, db_after, snr = denoise_with_tail_preservation(signal_array)
    print(f"   信噪比 (SNR): {snr:.2f} dB")

    # 4. 保存降噪前的数据（仅dB值，无标题无序号，保存为txt）
    before_txt = output_dir / f"{base_name}_before_denoise.txt"
    np.savetxt(before_txt, db_before, fmt='%.10f')
    print(f"4. 降噪前数据已保存: {before_txt}")

    # 5. 保存降噪后的数据（仅dB值，无标题无序号，保存为txt）
    after_txt = output_dir / f"{base_name}_after_denoise.txt"
    np.savetxt(after_txt, db_after, fmt='%.10f')
    print(f"5. 降噪后数据已保存: {after_txt}")

    # 6. 绘制对比图
    print("6. 绘制对比图...")
    plot_path = output_dir / f"{base_name}_comparison.png"
    plot_comparison(indices, db_before, db_after, snr, plot_path)

    print(f"\n处理完成!")
    return True


def main():
    """
    主函数 - 处理当前目录下的所有bin文件
    """
    current_dir = Path(__file__).parent
    bin_files = list(current_dir.glob("*.bin"))

    if not bin_files:
        print("当前目录下没有找到 .bin 文件")
        return

    print(f"找到 {len(bin_files)} 个 .bin 文件")
    print("-" * 60)

    output_dir = current_dir / "output"
    output_dir.mkdir(exist_ok=True)

    success_count = 0
    for bin_file in sorted(bin_files):
        try:
            if process_single_file(bin_file, output_dir):
                success_count += 1
        except Exception as e:
            print(f"处理 {bin_file} 时出错: {e}")

    print(f"\n{'='*60}")
    print(f"全部完成! 成功处理 {success_count}/{len(bin_files)} 个文件")
    print(f"输出目录: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
