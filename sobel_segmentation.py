#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI Agent屏幕截图背景处理：
1) smart_resize（与 Qwen2VL 对齐）
2) 对每张输入图片：仅输出基于 28x28 patch 的前景标注图（在前景 patch 内写入字符 '1'）

说明：不再输出 *_magnitude.png；仅输出 *_patchmap.png。
"""

import cv2
import numpy as np
import os
from pathlib import Path
import math


def smart_resize(
        height: int,
        width: int,
        factor: int = 28,
        min_pixels: int = 200704,
        max_pixels: int = 1003520,
):
    """将图像尺寸调整到更适合 MLLM 处理的范围：

    1) 高度和宽度均为 factor 的倍数；
    2) 总像素数在 [min_pixels, max_pixels] 范围内；
    3) 尽可能保持纵横比；

    若 height/width 小于 factor 或纵横比超过 200，将抛出 ValueError。
    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )

    # 四舍五入到最近的 factor 倍数
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor

    # 超过最大像素，按比例缩小到不超过上限（向下取整到 factor 倍数）
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor

    # 低于最小像素，按比例放大到不低于下限（向上取整到 factor 倍数）
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar


# ===================== Patch map 相关实现 =====================

def build_patch_map_overlay(
        color_bgr: np.ndarray,
        magnitude_u8: np.ndarray,
        patch_size: int = 28,
        edge_thr: int = 50,
        ratio_thr: float = 0.01,
        draw_grid: bool = True,
) -> np.ndarray:
    """
    在 color_bgr 上基于 magnitude 的 28x28 patch 做前景/背景可视化：
    - 前景 patch：在 patch 中央绘制字符 '1'（白字+黑描边）
    - 背景 patch：不标注
    - 可选绘制白色网格线
    """
    assert color_bgr.ndim == 3 and color_bgr.shape[2] == 3, "color_bgr must be HxWx3"
    assert magnitude_u8.ndim == 2, "magnitude_u8 must be HxW (grayscale)"
    H, W = magnitude_u8.shape[:2]
    assert (H, W) == color_bgr.shape[:2], "color and magnitude size must match"

    rows, cols = H // patch_size, W // patch_size
    total = rows * cols

    out = color_bgr.copy()

    # 文本可见性设置：28 像素 patch 对应 fontScale ~ 0.6
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, patch_size / 46.0)
    thick_inner = max(1, patch_size // 24)  # 白字细描
    thick_outer = max(2, patch_size // 18)  # 黑色外描边

    for i in range(rows):
        y0 = i * patch_size
        y1 = y0 + patch_size
        for j in range(cols):
            x0 = j * patch_size
            x1 = x0 + patch_size
            patch = magnitude_u8[y0:y1, x0:x1]
            # 判定是否含边缘
            edge_pixels = np.count_nonzero(patch >= edge_thr)
            ratio = edge_pixels / (patch_size * patch_size)
            if ratio >= ratio_thr:
                # 在 patch 中央写入 '1'（先黑描边后白字）
                text = "1"
                (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thick_inner)
                cx = x0 + (patch_size - tw) // 2
                cy = y0 + (patch_size + th) // 2
                cv2.putText(out, text, (cx, cy), font, font_scale, (0, 0, 0), thick_outer, cv2.LINE_AA)
                cv2.putText(out, text, (cx, cy), font, font_scale, (255, 255, 255), thick_inner, cv2.LINE_AA)

            else:
                pass

    # 画网格线（白色）
    if draw_grid:
        for i in range(1, rows):
            y = i * patch_size
            cv2.line(out, (0, y), (W, y), (255, 255, 255), 1, cv2.LINE_AA)
        for j in range(1, cols):
            x = j * patch_size
            cv2.line(out, (x, 0), (x, H), (255, 255, 255), 1, cv2.LINE_AA)

    # print(f"Patch grid: {rows} x {cols} = {total}")
    return out


def compute_sobel_uigraph(
        image_path: str,
        preprocessed_pil=None,  # PIL.Image already resized by Qwen2VL pipeline (fetch_image_with_resize)
        preprocessed_bgr=None,  # alternatively, provide cv2 BGR ndarray already resized

        edge_thr: int = 50,
        ratio_thr: float = 0.01,
        patch_size: int = 28,
        min_pixels: int = 200704,
        max_pixels: int = 1003520,
):
    """
    为截图生成基于 Sobel 边缘检测的 patch 级前景/背景信息。
    - 对每个 28x28 的 patch 判断其是否包含边缘，作为前景（is_edge=True）或背景（is_edge=False）。
    - 不再进行区域合并（Union-Find）或计算面积比。
    返回：{"is_edge": List[bool]}；失败返回 None。
    """
    try:
        import cv2
        import numpy as np
        # 读图 / 使用预处理后的图像
        # 优先级：preprocessed_bgr > preprocessed_pil > 从 image_path 读取原图并做 sobel_segmentation.smart_resize
        if preprocessed_bgr is not None:
            img = preprocessed_bgr
            if img is None:
                return None
            h0, w0 = img.shape[:2]
            H, W = h0, w0

        elif preprocessed_pil is not None:
            # PIL (W,H) -> BGR
            try:
                import numpy as np
                pil_img = preprocessed_pil.convert("RGB")
                np_img = np.array(pil_img)
                img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
            except Exception:
                return None
            h0, w0 = img.shape[:2]
            H, W = h0, w0

        else:
            # 读原图
            img = cv2.imread(image_path)
            if img is None:
                return None
            h0, w0 = img.shape[:2]
            # smart_resize 與縮放（注意：这条路径不会包含 Qwen2VL 的 512 长边预缩放）
            h_bar, w_bar = smart_resize(h0, w0, factor=patch_size, min_pixels=min_pixels, max_pixels=max_pixels)

            if (h_bar, w_bar) != (h0, w0):
                interp = cv2.INTER_AREA if (h_bar < h0 or w_bar < w0) else cv2.INTER_CUBIC
                img = cv2.resize(img, (w_bar, h_bar), interpolation=interp)
            H, W = h_bar, w_bar
        # Sobel 幅值
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
        sx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(sx * sx + sy * sy)
        max_val = float(np.max(mag))
        if max_val <= 0:
            mag_u8 = np.zeros_like(gray, dtype=np.uint8)
        else:
            mag_u8 = np.uint8(255.0 * (mag / max_val))
        # 以 patch 為單位做邊緣判定
        rows, cols = H // patch_size, W // patch_size
        N = rows * cols

        if N <= 0:
            return None
        is_edge = np.zeros((rows, cols), dtype=np.bool_)
        for i in range(rows):
            y0 = i * patch_size
            y1 = y0 + patch_size
            for j in range(cols):
                x0 = j * patch_size
                x1 = x0 + patch_size
                patch = mag_u8[y0:y1, x0:x1]
                cnt = int(np.count_nonzero(patch >= edge_thr))
                ratio = cnt / float(patch_size * patch_size)
                if ratio >= ratio_thr:
                    is_edge[i, j] = True

        # 返回 is_edge 数组（展平为一维列表）
        is_edge_list = is_edge.flatten().tolist()
        return {
            "is_edge": is_edge_list,
        }
    except Exception:
        return None


def process_directory(
        input_dir: str,
        output_dir: str,
        edge_thr: int = 50,
        ratio_thr: float = 0.01,
        patch_size: int = 28,
        draw_grid: bool = True,
):
    """
    批量处理：对输入目录中的所有图片仅输出 Patch 标注图（不输出 magnitude 图片）。
    步骤：smart_resize -> Sobel 幅值计算 -> 构建 patch 叠加 -> 保存 *_patchmap.png
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    if not os.path.exists(input_dir):
        print(f"错误: 输入目录不存在 {input_dir}")
        return

    image_files = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in image_extensions]
    if not image_files:
        print(f"警告: 输入目录中没有找到图片文件 {input_dir}")
        return

    print(f"找到 {len(image_files)} 张GUI截图，开始处理...")

    success_count = 0
    for image_file in image_files:
        input_path = os.path.join(input_dir, image_file)

        # 读取与 smart_resize
        img = cv2.imread(input_path)
        if img is None:
            print(f"跳过: 无法读取图片 {image_file}")
            continue
        h0, w0 = img.shape[:2]
        try:
            h_bar, w_bar = smart_resize(h0, w0, factor=28, min_pixels=200704, max_pixels=1003520)
        except ValueError as e:
            print(f"跳过: {image_file} 因 smart_resize 失败 -> {e}")
            continue
        if (h_bar, w_bar) != (h0, w0):
            interp = cv2.INTER_AREA if (h_bar < h0 or w_bar < w0) else cv2.INTER_CUBIC
            img = cv2.resize(img, (w_bar, h_bar), interpolation=interp)
            print(f"smart_resize: ({h0},{w0}) -> ({h_bar},{w_bar}), factor=28")
        else:
            print(f"smart_resize: 尺寸已满足约束，保持不变 ({h0},{w0})")

        # 计算 Sobel 幅值（不落盘）
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        # 防止全零导致除零
        max_val = np.max(magnitude)
        magnitude_u8 = np.zeros_like(gray, dtype=np.uint8) if max_val == 0 else np.uint8(255 * magnitude / max_val)

        # 构建 patch 叠加并保存
        overlay = build_patch_map_overlay(
            img, magnitude_u8, patch_size=patch_size, edge_thr=edge_thr, ratio_thr=ratio_thr, draw_grid=draw_grid
        )
        basename = os.path.splitext(os.path.basename(input_path))[0]
        out_path = os.path.join(output_dir, f"{basename}_patchmap.png")
        cv2.imwrite(out_path, overlay)
        print(f"[patchmap] saved: {out_path}")
        success_count += 1
        print()

    print("=" * 70)
    print(f"处理完成！成功处理 {success_count}/{len(image_files)} 张图片")
    print(f"结果已保存到: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    INPUT_DIR = "/data_sdc/data/ldq/Sobel/input"
    OUTPUT_DIR = "/data_sdc/data/ldq/Sobel/output"

    print("=" * 70)
    print("GUI Agent屏幕截图 - Patch 标注（仅输出 *_patchmap.png）")
    print("=" * 70)
    print(f"输入目录: {INPUT_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 70 + "")

    # 批处理：仅输出 patch 标注图
    process_directory(INPUT_DIR, OUTPUT_DIR, edge_thr=50, ratio_thr=0.01, patch_size=28, draw_grid=True)
