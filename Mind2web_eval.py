import json
import os
import re

import torch
from torch.profiler import profile, ProfilerActivity
from PIL import Image, ImageDraw, ImageFont
from typing import List
from src.training.my_qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoConfig, AutoModelForVision2Seq
import importlib

def load_model_class(model_path):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if config.model_type == "qwen2_vl":
        module_path = "src.Qwen2.model_file.LLM_compression_v2_action.modeling_qwen2vl"
        class_name = "Qwen2VLForConditionalGeneration"
    elif config.model_type == "qwen2_5_vl":
        module_path = "src.Qwen2_5.model_file.LLM_compression_v2_5_action.modeling_qwen2_5_vl"
        class_name = "Qwen2_5_VLForConditionalGeneration"
    else:
        raise ValueError(f"Unsupported model type: {config.model_type}")
    
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

IGNORE_INDEX = -100

# Optional Sobel support (used by sobel_enable path)
try:
    from sobel_segmentation import compute_sobel_uigraph
    _HAS_SOBEL = True
except Exception:
    _HAS_SOBEL = False

DEFAULT_IM_START_TOKEN = "<|im_start|>"
DEFAULT_IM_END_TOKEN = "<|im_end|>"
DEFAULT_IMAGE_TOKEN = "<|image_pad|>"
DEFAULT_VIDEO_TOKEN = "<|video_pad|>"
LLAVA_IMAGE_TOKEN = "<image>"
LLAVA_VIDEO_TOKEN = "<video>"
VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"

SYSTEM_MESSAGE = "You are a helpful assistant."
import pdb
import argparse

def extract_total_flops_from_prof(prof) -> int:
    total_flops = 0.0
    try:
        for evt in prof.key_averages():
            evt_flops = getattr(evt, "flops", None)
            if evt_flops is not None:
                total_flops += float(evt_flops)
    except Exception:
        return 0
    return int(total_flops)


def write_json(data, file_path, task):
    os.makedirs(file_path, exist_ok=True)
    file_path = os.path.join(file_path, task + '.json')
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)

# 创建解析器
parser = argparse.ArgumentParser(description='Specify paths for saving and loading models.')

# 添加参数
parser.add_argument('--save_path', type=str, default="/home/Simpagent-main/eval_results/Mind2Web",
                    help='The path where the model will be saved')
parser.add_argument('--model_path', type=str, default="/data_sdc/data/SimpAgent-main/save/Qwen2/Mind2Web/Baseline2AO_removedistill_removedata_lora",
                    help='The path where the model is loaded from')
parser.add_argument('--his_num', type=int, default=2,
                    help='The path where the model is loaded from')
parser.add_argument('--drop_k', type=int, default=3,
                    help='The path where the model is loaded from')
parser.add_argument('--task', type=str, default="domain",
                    help='The path where the model is loaded from')

parser.add_argument('--fastv_enable', action='store_true', default=False,
                    help='Enable FastV token-level pruning at drop_k (training-free).')
parser.add_argument('--fastv_m', type=float, default=0.5,
                    help='FastV prune ratio m in [0,1], fraction of historical screenshot tokens to remove.')
parser.add_argument('--fastv_segment_ratios', type=str, default="0.9,0.9,0.9,0.9",
                    help='Comma-separated ratios for segmented FastV per history image from most recent to 4th.')

parser.add_argument('--pdrop_enable', action='store_true', default=False,
                    help='Enable PDrop token-level pruning at drop_k (training-free).')
parser.add_argument('--pdrop_debug', action='store_true', default=False,
                    help='Enable minimal PDrop debug logs (effective ratios and per-step keep stats).')
parser.add_argument('--pdrop_keep_ratio', type=str, default="0.5,0.5,0.5,0.5",
                    help='Comma-separated keep ratios for segmented PDrop per history image from most recent to 4th (e.g., 0.8,0.5,0.3,0.2).')

parser.add_argument('--sobel_edge_thr', type=int, default=50,
                    help='Sobel magnitude threshold (0~255) to count an edge pixel within a 28x28 patch.')
parser.add_argument('--sobel_ratio_thr', type=float, default=0.01,
                    help='Per-patch edge pixel ratio threshold in [0,1]; >= ratio means an edge patch.')

parser.add_argument('--DivPrune_enable', action='store_true', default=False,
                    help='Enable sparse greedy submodular pruning (training-free).')
parser.add_argument('--DivPrune_ratio', type=str, default="0.5,0.5",
                    help='Comma-separated keep ratios for DivPrune per history image from most recent to 4th.')

parser.add_argument('--sobel_enable', action='store_true', default=False,
                    help='Enable Sobel edge-only pruning at drop_k (training-free).')
parser.add_argument('--sobel_vis_enable', action='store_true', default=False,
                    help='Enable visualization of Sobel pruning results when sobel_enable is also True.')

parser.add_argument('--random_enable', action='store_true', default=False,
                    help='Enable Random token-level pruning on historical screenshots at drop_k (training-free).')
parser.add_argument('--random_ratio', type=str, default="0.5,0.5,0.5,0.5",
                    help='Comma-separated prune ratios for Random pruning per history image from most recent to 4th.')

parser.add_argument('--sparsevlm_enable', action='store_true', default=False,
                    help='Enable SparseVLM pruning (Stage1 raters + Stage2 global visual pruning).')
parser.add_argument('--sparsevlm_keep_ratio', type=float, default=0.4,
                    help='SparseVLM keep ratio for historical visual tokens (e.g., 0.4 keeps top 40% and prunes 60%).')

parser.add_argument('--fastv_vis_enable', action='store_true', default=False,
                    help='Enable visualization of FastV pruning results when fastv_enable is also True.')

parser.add_argument('--dart_enable', action='store_true', default=True,
                    help='Enable DART token-level pruning on historical screenshots at drop_k (training-free).')
parser.add_argument('--dart_ratios', type=str, default="0.5,0.5,0.5,0.5",
                    help='Comma-separated keep ratios for DART per history image from most recent to 4th.')
parser.add_argument('--dart_num_pivots', type=int, default=10,
                    help='Number of pivot tokens (top-L2) per history image for DART.')
parser.add_argument('--dart_debug', action='store_true', default=True,
                    help='Enable minimal DART debug logs (effective ratios, mode hit, per-step keep stats).')

parser.add_argument('--profile_total_flops', action='store_true', default=False,
                    help='Enable torch profiler to measure per-step total FLOPs of full generate call.')
parser.add_argument('--profile_every_n_steps', type=int, default=30,
                    help='Profile once every N valid evaluation steps when --profile_total_flops is enabled.')

# 解析参数
args = parser.parse_args()
args.save_path = os.path.join(args.save_path, args.model_path.split('/')[-1])


def process_string(s):
    # 使用正则表达式匹配所有坐标点
    pattern = r'\((\d+),(\d+)\)'
    
    # 替换所有匹配的坐标点
    def replace(match):
        # 将匹配到的数字除以1000，并四舍五入到两位小数
        x = round(float(match.group(1)) / 1000, 2)
        y = round(float(match.group(2)) / 1000, 2)
        return f"({x:.2f},{y:.2f})"
    
    return re.sub(pattern, replace, s)
    
# Default: Load the model on the available device(s)
# processor = AutoProcessor.from_pretrained("OS-Copilot/OS-Atlas-Base-7B")
#processor = AutoProcessor.from_pretrained("/home/wentao/project/gui_ads/LLaMA-Factory-main/debug_output_v2/checkpoint-1056/")
min_pixels = 200704
max_pixels = 1003520
#processor = AutoProcessor.from_pretrained("/home/wentao/project/gui_ads/LLaMA-Factory-main/debug_output_v5", min_pixels=min_pixels, max_pixels=max_pixels)
# processor = AutoProcessor.from_pretrained("/home/wentao/project/gui_ads/OS-Atlas-Base-7B/")
# processor = AutoProcessor.from_pretrained("/nas_sh/wentao/Qwen2-VL-7B-Instruct/")
processor = AutoProcessor.from_pretrained(
    args.model_path,
    trust_remote_code=True,
    min_pixels=min_pixels,
    max_pixels=max_pixels
)

# Sobel cache for historical screenshots: key=image path, value={'is_edge': List[bool]}
sobel_cache = {}

# Default: Load the model on the available device(s)
ModelClass = load_model_class(args.model_path)
model = ModelClass.from_pretrained(
    args.model_path,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2"
)

def safe_setattr(obj, attr, value):
    if hasattr(obj, attr):
        setattr(obj, attr, value)

# NOTE: disable all pruning/visualization logic for baseline evaluation with official transformers models
model.drop_k = args.drop_k
model.model.drop_k = args.drop_k


# if args.fastv_enable:
#     model.model.enable_fastv_pruning = True
if args.fastv_enable:
    safe_setattr(model.model, "enable_fastv_pruning", True)
# model.model.fastv_prune_ratio_m = float(args.fastv_m)
safe_setattr(model.model, "fastv_prune_ratio_m", float(args.fastv_m))
try:
    ratios = [float(x) for x in str(args.fastv_segment_ratios).split(',') if str(x).strip() != '']
    ratios = [0.0 if r < 0.0 else (1.0 if r > 1.0 else r) for r in ratios]
    # model.model.fastv_segment_ratios = ratios
    safe_setattr(model.model, "fastv_segment_ratios", ratios)
except Exception:
    pass

if args.pdrop_enable:
    safe_setattr(model.model, "enable_pdrop_pruning", True)
try:
    pdrop_ratios = [float(x) for x in str(args.pdrop_keep_ratio).split(',') if str(x).strip() != '']
    pdrop_ratios = [0.0 if r < 0.0 else (1.0 if r > 1.0 else r) for r in pdrop_ratios]
    if len(pdrop_ratios) == 0:
        raise ValueError("pdrop_keep_ratio is empty after parsing")
    safe_setattr(model.model, "pdrop_keep_ratios", pdrop_ratios)
    safe_setattr(model.model, "pdrop_keep_ratio_m", float(pdrop_ratios[0]))
except Exception as e:
    raise ValueError(f"Invalid --pdrop_keep_ratio: {args.pdrop_keep_ratio!r}, err={e}")

if args.pdrop_debug:
    print("[PDROP-DEBUG] cli.pdrop_enable=", bool(args.pdrop_enable))
    print("[PDROP-DEBUG] cli.pdrop_keep_ratio(raw)=", repr(args.pdrop_keep_ratio))
    print("[PDROP-DEBUG] parsed_keep_ratios=", pdrop_ratios if 'pdrop_ratios' in locals() else None)
    print("[PDROP-DEBUG] model.enable_pdrop_pruning=", bool(getattr(model.model, 'enable_pdrop_pruning', False)))
    print("[PDROP-DEBUG] model.pdrop_keep_ratios(effective)=", getattr(model.model, 'pdrop_keep_ratios', None))

# if args.sobel_enable:
#     model.model.sobel_enable = True
if args.sobel_enable:
    safe_setattr(model.model, "sobel_enable", True)

# if args.sparse_greedy_enable:
#     model.model.enable_sparse_greedy_pruning = True
if args.DivPrune_enable:
    safe_setattr(model.model, "enable_DivPrune_pruning", True)
try:
    dpratios = [float(x) for x in str(args.DivPrune_ratio).split(',') if str(x).strip() != '']
    dpratios = [0.0 if r < 0.0 else (1.0 if r > 1.0 else r) for r in dpratios]
    safe_setattr(model.model, "DivPrune_ratios", dpratios)
except Exception:
    pass

# if args.random_enable:
#     model.model.enable_random_pruning = True
if args.random_enable:
    safe_setattr(model.model, "enable_random_pruning", True)
try:
    rratios = [float(x) for x in str(args.random_ratio).split(',') if str(x).strip() != '']
    rratios = [0.0 if r < 0.0 else (1.0 if r > 1.0 else r) for r in rratios]
    # Mind2Web only uses up to 2 history images; keep the first two ratios (recent->older)
    # model.model.random_prune_ratios = rratios[:2]
    safe_setattr(model.model, "random_prune_ratios", rratios[:2])
except Exception:
    pass
# model.model.random_prune_seed = 0
safe_setattr(model.model, "random_prune_seed", 0)

# _enabled = int(bool(args.fastv_enable)) + int(bool(args.sparse_greedy_enable)) + int(bool(args.sobel_enable)) + int(bool(args.random_enable)) + int(bool(args.sparsevlm_enable))
# if _enabled > 1:
#     raise ValueError("Only one of fastv_enable / sparse_greedy_enable / sobel_enable / random_enable / sparsevlm_enable can be True.")

# if args.sparsevlm_enable:
#     setattr(model.model, 'enable_sparsevlm_pruning', True)
#     setattr(model.model, 'sparsevlm_keep_ratio', float(args.sparsevlm_keep_ratio))
if args.sparsevlm_enable:
    safe_setattr(model.model, 'enable_sparsevlm_pruning', True)
    safe_setattr(model.model, 'sparsevlm_keep_ratio', float(args.sparsevlm_keep_ratio))

# DART pruning
if args.dart_enable:
    safe_setattr(model.model, 'enable_dart_pruning', True)
_dart_parse_ok = False
_dart_parse_err = None
_parsed_dratios = None
try:
    dratios = [float(x) for x in str(args.dart_ratios).split(',') if str(x).strip() != '']
    dratios = [0.0 if r < 0.0 else (1.0 if r > 1.0 else r) for r in dratios]
    safe_setattr(model.model, 'dart_keep_ratios', dratios)
    _dart_parse_ok = True
    _parsed_dratios = dratios
except Exception as e:
    _dart_parse_err = repr(e)
safe_setattr(model.model, 'dart_num_pivots', int(args.dart_num_pivots))
safe_setattr(model.model, 'dart_debug', bool(args.dart_debug))

if args.dart_debug:
    print("[DART-DEBUG] cli.dart_enable=", bool(args.dart_enable))
    print("[DART-DEBUG] cli.dart_ratios(raw)=", repr(args.dart_ratios))
    print("[DART-DEBUG] parse_ok=", _dart_parse_ok)
    if _dart_parse_ok:
        print("[DART-DEBUG] parsed_keep_ratios=", _parsed_dratios)
    else:
        print("[DART-DEBUG] parse_error=", _dart_parse_err)
    print("[DART-DEBUG] model.enable_dart_pruning=", bool(getattr(model.model, 'enable_dart_pruning', False)))
    print("[DART-DEBUG] model.dart_keep_ratios(effective)=", getattr(model.model, 'dart_keep_ratios', None))
    print("[DART-DEBUG] model.dart_num_pivots=", getattr(model.model, 'dart_num_pivots', None))
    print("[DART-DEBUG] mode_flags=", {
        'DivPrune': bool(getattr(model.model, 'enable_DivPrune_pruning', False)),
        'sobel': bool(getattr(model.model, 'sobel_enable', False)),
        'uigraph': bool(getattr(model.model, 'enable_uigraph_pruning', False)),
        'dart': bool(getattr(model.model, 'enable_dart_pruning', False)),
        'fastv': bool(getattr(model.model, 'enable_fastv_pruning', False)),
        'pdrop': bool(getattr(model.model, 'enable_pdrop_pruning', False)),
        'random': bool(getattr(model.model, 'enable_random_pruning', False)),
        'sparsevlm': bool(getattr(model.model, 'enable_sparsevlm_pruning', False)),
    })

def visualize_fastv_keep_mask(
    *,
    keep_mask_1d: torch.Tensor,  # [L] True=keep
    oas_index_1: list,           # one sample's OAS_index (list of [s,e) segments)
    hist_img_paths_old_to_new: List[str],
    ep_id: str,
    step_id: int,
    out_root: str = "/data_sdc/data/SimpAgent-main/output/visual_result/FastV",
    max_hist: int = 4,
    patch_size: int = 14,
    spatial_merge_size: int = 2,
    min_pixel: int = None,
    max_pixel: int = None,
):
    try:
        import numpy as np
    except Exception:
        np = None

    if keep_mask_1d is None or oas_index_1 is None:
        return

    if not isinstance(oas_index_1, list) or len(oas_index_1) <= 1:
        return

    # 历史段解析逻辑增强
    hist_segments_old_to_new = []
    if len(oas_index_1) > 1:
        # 正常情况：多个 segment，最后一个是当前图
        for seg in oas_index_1[:-1]:
            if isinstance(seg, list) and len(seg) >= 2:
                s, e = int(seg[0]), int(seg[1])
                if e > s:
                    hist_segments_old_to_new.append((s, e))
    elif len(oas_index_1) == 1 and len(hist_img_paths_old_to_new) > 0:
        # 异常情况：所有图片 token 被连在了一个 segment 里
        full_s, full_e = oas_index_1[0]
        total_tokens = full_e - full_s
        num_all_imgs = len(hist_img_paths_old_to_new) + 1
        tokens_per_img = total_tokens // num_all_imgs

        for i in range(len(hist_img_paths_old_to_new)):
            s = full_s + i * tokens_per_img
            e = s + tokens_per_img
            hist_segments_old_to_new.append((s, e))

    if len(hist_segments_old_to_new) == 0 or len(hist_img_paths_old_to_new) == 0:
        return

    n_hist = min(len(hist_segments_old_to_new), len(hist_img_paths_old_to_new))
    hist_segments_old_to_new = hist_segments_old_to_new[-n_hist:]
    hist_img_paths_old_to_new = hist_img_paths_old_to_new[-n_hist:]

    # 只画最近 max_hist 张
    k = min(max_hist, n_hist)
    segs_recent = list(reversed(hist_segments_old_to_new[-k:]))
    imgs_recent = list(reversed(hist_img_paths_old_to_new[-k:]))

    out_dir = os.path.join(out_root, str(ep_id), str(step_id))
    os.makedirs(out_dir, exist_ok=True)

    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        try:
            font = ImageFont.truetype("Arial.ttf", 14)
        except Exception:
            font = ImageFont.load_default()

    for rank, (img_path, (s, e)) in enumerate(zip(imgs_recent, segs_recent)):
        try:
            _minp = min_pixel if min_pixel is not None else min_pixels
            _maxp = max_pixel if max_pixel is not None else max_pixels
            pil = get_image_info(img_path, min_pixel=_minp, max_pixel=_maxp).convert("RGB")
        except Exception:
            continue

        W, H = pil.size
        grid_h = H // (patch_size * spatial_merge_size)
        grid_w = W // (patch_size * spatial_merge_size)
        if grid_h <= 0 or grid_w <= 0:
            continue

        num_tokens = int(grid_h * grid_w)
        seg_len = int(e - s)
        if seg_len != num_tokens:
            continue

        mask_seg = keep_mask_1d[s:e]
        if isinstance(mask_seg, torch.Tensor):
            mask_seg = mask_seg.detach().to("cpu").to(torch.bool)

        draw = ImageDraw.Draw(pil)
        cell_w = W / float(grid_w)
        cell_h = H / float(grid_h)

        # FastV：标注被剪掉的 token（keep=False）为 "1"
        for idx in range(num_tokens):
            if bool(mask_seg[idx].item() if isinstance(mask_seg, torch.Tensor) else mask_seg[idx]):
                continue
            r = idx // grid_w
            c = idx % grid_w
            cx = c * cell_w + cell_w / 2.0
            cy = r * cell_h + cell_h / 2.0
            try:
                draw.text((cx, cy), "1", fill="red", font=font, anchor="mm")
            except Exception:
                draw.text((cx, cy), "1", fill="red", font=font)

        out_path = os.path.join(out_dir, f"history_{rank}_recent.png")
        try:
            pil.save(out_path)
        except Exception:
            continue


def visualize_sobel_keep_mask(
    *,
    keep_mask_1d: torch.Tensor,  # [L] True=keep
    oas_index_1: list,
    hist_img_paths_old_to_new: List[str],
    ep_id: str,
    step_id: int,
    out_root: str = "/data_sdc/data/SimpAgent-main/output/visual_result/Sobel",
    max_hist: int = 4,
    patch_size: int = 14,
    spatial_merge_size: int = 2,
    min_pixel: int = None,
    max_pixel: int = None,
):
    if keep_mask_1d is None or oas_index_1 is None:
        return

    if not isinstance(oas_index_1, list) or len(oas_index_1) <= 1:
        return

    # 历史段（不含最后一段“当前图”）
    hist_segments_old_to_new = []
    for seg in oas_index_1[:-1]:
        if isinstance(seg, list) and len(seg) >= 2:
            s, e = int(seg[0]), int(seg[1])
            if e > s:
                hist_segments_old_to_new.append((s, e))

    if len(hist_segments_old_to_new) == 0 or len(hist_img_paths_old_to_new) == 0:
        return

    n_hist = min(len(hist_segments_old_to_new), len(hist_img_paths_old_to_new))
    hist_segments_old_to_new = hist_segments_old_to_new[-n_hist:]
    hist_img_paths_old_to_new = hist_img_paths_old_to_new[-n_hist:]

    k = min(max_hist, n_hist)
    segs_recent = list(reversed(hist_segments_old_to_new[-k:]))
    imgs_recent = list(reversed(hist_img_paths_old_to_new[-k:]))

    out_dir = os.path.join(out_root, str(ep_id), str(step_id))
    os.makedirs(out_dir, exist_ok=True)

    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        try:
            font = ImageFont.truetype("Arial.ttf", 14)
        except Exception:
            font = ImageFont.load_default()

    for rank, (img_path, (s, e)) in enumerate(zip(imgs_recent, segs_recent)):
        try:
            _minp = min_pixel if min_pixel is not None else min_pixels
            _maxp = max_pixel if max_pixel is not None else max_pixels
            pil = get_image_info(img_path, min_pixel=_minp, max_pixel=_maxp).convert("RGB")
        except Exception:
            continue

        W, H = pil.size
        grid_h = H // (patch_size * spatial_merge_size)
        grid_w = W // (patch_size * spatial_merge_size)
        if grid_h <= 0 or grid_w <= 0:
            continue

        num_tokens = int(grid_h * grid_w)
        seg_len = int(e - s)
        if seg_len != num_tokens:
            continue

        mask_seg = keep_mask_1d[s:e]
        if isinstance(mask_seg, torch.Tensor):
            mask_seg = mask_seg.detach().to("cpu").to(torch.bool)

        draw = ImageDraw.Draw(pil)
        cell_w = W / float(grid_w)
        cell_h = H / float(grid_h)

        # Sobel：标注被剪掉的 token（keep=False）为 "1"
        for idx in range(num_tokens):
            if bool(mask_seg[idx].item() if isinstance(mask_seg, torch.Tensor) else mask_seg[idx]):
                continue
            r = idx // grid_w
            c = idx % grid_w
            cx = c * cell_w + cell_w / 2.0
            cy = r * cell_h + cell_h / 2.0
            try:
                draw.text((cx, cy), "1", fill="red", font=font, anchor="mm")
            except Exception:
                draw.text((cx, cy), "1", fill="red", font=font)

        out_path = os.path.join(out_dir, f"history_{rank}_recent.png")
        try:
            pil.save(out_path)
        except Exception:
            continue
    try:
        import numpy as np
    except Exception:
        np = None

    if keep_mask_1d is None or oas_index_1 is None:
        return

    if not isinstance(oas_index_1, list) or len(oas_index_1) <= 1:
        return

    # 历史段解析逻辑增强
    hist_segments_old_to_new = []
    if len(oas_index_1) > 1:
        # 正常情况：多个 segment，最后一个是当前图
        for seg in oas_index_1[:-1]:
            if isinstance(seg, list) and len(seg) >= 2:
                s, e = int(seg[0]), int(seg[1])
                if e > s:
                    hist_segments_old_to_new.append((s, e))
    elif len(oas_index_1) == 1 and len(hist_img_paths_old_to_new) > 0:
        # 异常情况：所有图片 token 被连在了一个 segment 里
        # 这种情况下，我们需要根据 token 总数和图片数量平分（Qwen2-VL 默认每张图 token 数一致，除非 resize 不同）
        full_s, full_e = oas_index_1[0]
        total_tokens = full_e - full_s
        num_all_imgs = len(hist_img_paths_old_to_new) + 1
        tokens_per_img = total_tokens // num_all_imgs
        
        for i in range(len(hist_img_paths_old_to_new)):
            s = full_s + i * tokens_per_img
            e = s + tokens_per_img
            hist_segments_old_to_new.append((s, e))

    if len(hist_segments_old_to_new) == 0 or len(hist_img_paths_old_to_new) == 0:
        return

    n_hist = min(len(hist_segments_old_to_new), len(hist_img_paths_old_to_new))
    hist_segments_old_to_new = hist_segments_old_to_new[-n_hist:]
    hist_img_paths_old_to_new = hist_img_paths_old_to_new[-n_hist:]

    # 只画最近 max_hist 张
    k = min(max_hist, n_hist)
    segs_recent = list(reversed(hist_segments_old_to_new[-k:]))
    imgs_recent = list(reversed(hist_img_paths_old_to_new[-k:]))

    out_dir = os.path.join(out_root, str(ep_id), str(step_id))
    os.makedirs(out_dir, exist_ok=True)

    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        try:
            font = ImageFont.truetype("Arial.ttf", 14)
        except Exception:
            font = ImageFont.load_default()

    for rank, (img_path, (s, e)) in enumerate(zip(imgs_recent, segs_recent)):
        try:
            _minp = min_pixel if min_pixel is not None else min_pixels
            _maxp = max_pixel if max_pixel is not None else max_pixels
            pil = get_image_info(img_path, min_pixel=_minp, max_pixel=_maxp).convert("RGB")
        except Exception as ex:
            continue

        W, H = pil.size
        grid_h = H // (patch_size * spatial_merge_size)
        grid_w = W // (patch_size * spatial_merge_size)
        if grid_h <= 0 or grid_w <= 0:
            continue

        num_tokens = int(grid_h * grid_w)
        seg_len = int(e - s)
        if seg_len != num_tokens:
            continue

        mask_seg = keep_mask_1d[s:e]
        if isinstance(mask_seg, torch.Tensor):
            mask_seg = mask_seg.detach().to("cpu").to(torch.bool)

        draw = ImageDraw.Draw(pil)
        cell_w = W / float(grid_w)
        cell_h = H / float(grid_h)

        # FastV：标注被剪掉的 token（keep=False）为 "1"
        for idx in range(num_tokens):
            if bool(mask_seg[idx].item() if isinstance(mask_seg, torch.Tensor) else mask_seg[idx]):
                continue
            r = idx // grid_w
            c = idx % grid_w
            cx = c * cell_w + cell_w / 2.0
            cy = r * cell_h + cell_h / 2.0
            try:
                draw.text((cx, cy), "1", fill="red", font=font, anchor="mm")
            except Exception:
                draw.text((cx, cy), "1", fill="red", font=font)

        out_path = os.path.join(out_dir, f"history_{rank}_recent.png")
        try:
            pil.save(out_path)
        except Exception:
            continue


def get_image_info(image_path, min_pixel=256 * 28 * 28, max_pixel=1280 * 28 * 28):
    # Using this because of process_vision_info function
    # Need to fix this in the future    
    
    messages = [
        {"role": "user", 
         "content": [
             {
                "type": "image", 
                "image": image_path,
                "min_pixels": min_pixel,
                "max_pixels": max_pixel,
            }
            ]
        }
    ]

    image_input, _ = process_vision_info(messages)

    return image_input[0]


def generate_grounding(image_path, query, vis_meta=None):
    # TODO
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text",
                 "text": query},
                #  "text": f"{query}\n请告诉我怎么操作同时输出坐标。"},
            ],
        }
    ]

    images = []
    for image_file in image_path:
        images.append(get_image_info(image_file))

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    text = text.replace(LLAVA_IMAGE_TOKEN, VISION_START_TOKEN+DEFAULT_IMAGE_TOKEN+VISION_END_TOKEN)
    inputs = processor(
        text=[text],
        images=images,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # 暴露本次 prefill 的 input_ids，供外层 debug 统计使用
    try:
        model.model._last_eval_input_ids = inputs.input_ids.detach().clone()
    except Exception:
        pass

    # ===== FastV 文本 Query 区间硬编码（用于“文本 -> 历史截图token”的注意力聚合） =====
    # 背景：我们希望 FastV / SparseVLM 的文本侧 query 覆盖“通用指令 + 具体目标(goal)”。
    # 通过 debug 已验证 apply_chat_template 输出格式固定：
    #   Please generate the next move ... current ui screenshot. Instruction: {goal}\n
    # 因此这里使用硬编码定位：
    # - 通用指令段：固定 token span [14, 35)
    # - goal 段：从 35 开始，直到遇到第一个 token_id == 1906（该 token 在当前模板下对应 goal 后的换行边界）
    # 最终 query span = [14, goal_end)
    try:
        ids = inputs.input_ids[0].tolist()

        general_instr_start = 14
        general_instr_len = 21
        general_instr_end = general_instr_start + general_instr_len  # =35

        goal_start = general_instr_end
        goal_end = None
        for idx in range(goal_start, len(ids)):
            if ids[idx] == 1906:
                goal_end = idx
                break
        if goal_end is None:
            goal_end = len(ids)

        # 写入模型侧，让 FastV 的注意力聚合仅使用该 query span（内部会把 p1+p2 合并）
        try:
            model.model._prompt_query_ranges = [{
                'p1': (int(general_instr_start), int(general_instr_end)),
                'p2': (int(goal_start), int(goal_end)),
            }]
            if args.pdrop_enable:
                model.model._pdrop_query_index = int(max(general_instr_start, goal_end - 1))
        except Exception:
            pass
    except Exception as e:
        print("[DEBUG] Compute FastV query boundary failed:", e)
    # ===== 计算结束 =====

    # Sobel attach (Batch size assumed 1)
    # - 仅 sobel_enable 需要 Sobel 先验
    if args.sobel_enable and len(image_path) > 1:
        hist_imgs = image_path[:-1]
        info_list = []

        if not _HAS_SOBEL:
            raise RuntimeError("sobel_segmentation not available but sobel_enable is True")

        for p in hist_imgs:
            if p not in sobel_cache:
                try:
                    _pil = get_image_info(p, min_pixel=min_pixels, max_pixel=max_pixels)
                except Exception:
                    _pil = None
                if _pil is None:
                    raise RuntimeError(f"Sobel preprocessed_pil is None for image: {p}")
                try:
                    sobel_info = compute_sobel_uigraph(
                        p,
                        preprocessed_pil=_pil,
                        edge_thr=int(args.sobel_edge_thr),
                        ratio_thr=float(args.sobel_ratio_thr),
                        min_pixels=min_pixels,
                        max_pixels=max_pixels,
                    )
                except Exception as ex:
                    raise RuntimeError(f"compute_sobel_uigraph failed for image: {p}, err={ex}")
                if not (isinstance(sobel_info, dict) and 'is_edge' in sobel_info):
                    raise RuntimeError(f"compute_sobel_uigraph returned invalid result for image: {p}, got={type(sobel_info)}")
                sobel_cache[p] = {'is_edge': sobel_info['is_edge']}
            info_list.append(sobel_cache.get(p, None))

        if not all(x is not None and isinstance(x, dict) and 'is_edge' in x for x in info_list):
            raise RuntimeError("Sobel info_list contains invalid entries")

        model.model.sobel_enable = True
        model.model._sobel_info = [info_list]

        # disable others to avoid conflict
        model.model.enable_uigraph_pruning = False
        model.model.enable_fastv_pruning = False
    else:
        model.model._sobel_info = None

    generated_ids = model.generate(**inputs, max_new_tokens=128)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )[0][0:-10]

    # ===== Sobel：可视化最近历史截图（被剪掉的patch标"1"） =====
    try:
        if args.sobel_enable and args.sobel_vis_enable and len(image_path) > 1 and getattr(model.model, 'sobel_enable', False):
            keep_mask = getattr(model.model, '_sobel_keep_mask', None)
            oas_index = getattr(model.model, '_last_oas_index', None)

            if keep_mask is not None and oas_index is not None and isinstance(oas_index, list) and len(oas_index) > 0:
                km_1d = keep_mask[0].detach() if hasattr(keep_mask, '__getitem__') else keep_mask
                idx_list_1 = oas_index[0]
                ep_id = "unknown"
                step_id = 0
                if isinstance(vis_meta, dict):
                    if 'ep_id' in vis_meta:
                        ep_id = vis_meta['ep_id']
                    if 'step_id' in vis_meta:
                        step_id = vis_meta['step_id']
                
                visualize_sobel_keep_mask(
                    keep_mask_1d=km_1d,
                    oas_index_1=idx_list_1,
                    hist_img_paths_old_to_new=image_path[:-1],
                    ep_id=str(ep_id),
                    step_id=int(step_id),
                    out_root="/data_sdc/data/SimpAgent-main/output/visual_result/Sobel",
                    max_hist=4,
                    patch_size=14,
                    spatial_merge_size=2,
                    min_pixel=min_pixels,
                    max_pixel=max_pixels,
                )
    except Exception:
        pass

    # ===== FastV：可视化最近历史截图（被剪掉的patch标"1"） =====
    try:
        if args.fastv_enable and args.fastv_vis_enable and len(image_path) > 1 and getattr(model.model, 'enable_fastv_pruning', False):
            keep_mask = getattr(model.model, '_fastv_keep_mask', None)
            oas_index = getattr(model.model, '_last_oas_index', None)

            if keep_mask is not None and oas_index is not None and isinstance(oas_index, list) and len(oas_index) > 0:
                # batch size assumed 1
                km_1d = keep_mask[0].detach() if hasattr(keep_mask, '__getitem__') else keep_mask
                idx_list_1 = oas_index[0]
                ep_id = "unknown"
                step_id = 0
                if isinstance(vis_meta, dict):
                    if 'ep_id' in vis_meta:
                        ep_id = vis_meta['ep_id']
                    if 'step_id' in vis_meta:
                        step_id = vis_meta['step_id']
                
                visualize_fastv_keep_mask(
                    keep_mask_1d=km_1d,
                    oas_index_1=idx_list_1,
                    hist_img_paths_old_to_new=image_path[:-1],
                    ep_id=str(ep_id),
                    step_id=int(step_id),
                    out_root="/data_sdc/data/SimpAgent-main/output/visual_result/FastV",
                    max_hist=4,
                    patch_size=14,
                    spatial_merge_size=2,
                    min_pixel=min_pixels,
                    max_pixel=max_pixels,
                )
    except Exception:
        pass

    return output_text



# evaluation on mind2web
import os
import random
import torch
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM
from transformers.generation import GenerationConfig
import re
import logging
import ast
import argparse
from PIL import Image
import numpy as np

# logging.basicConfig(level=logging.INFO)


# convert action to prediction format (and return the groundtruth bbox)
def action2step(action, image_size, return_bbox=False):
    action_type = action["operation"]["original_op"]
    assert action_type in ['CLICK', 'TYPE', 'SELECT', 'HOVER', 'ENTER']

    point_x = action["bbox"]["x"] + (action["bbox"]["width"] / 2)
    point_y = action["bbox"]["y"] + (action["bbox"]["height"] / 2)
    click_point = [point_x / image_size[0], point_y / image_size[1]]
    click_point = [round(item, 3) for item in click_point]
    click_point = [f"{int(1000*item)}" for item in click_point]
    click_point = "({},{})".format(click_point[0], click_point[1])

    if return_bbox:
        bbox = [action["bbox"]["x"], action["bbox"]["y"], action["bbox"]["x"] + action["bbox"]["width"],
                action["bbox"]["y"] + action["bbox"]["height"]]
        bbox = [bbox[0] / image_size[0], bbox[1] / image_size[1], bbox[2] / image_size[0], bbox[3] / image_size[1]]
        bbox = [round(item, 3) for item in bbox]

    if action_type in ['CLICK', 'HOVER', 'ENTER']:
        action_step = "{{\"action_type\": {}, \"click_point\": {}}}".format(4, click_point)
    elif action_type == 'SELECT':
        select_value = action["operation"]["value"]
        action_step = "{{\"action_type\": {}, \"click_point\": {}, \"value\": \"{}\"}}".format(2, click_point,
                                                                                               select_value)
    elif action_type == 'TYPE':
        typed_text = action["operation"]["value"]
        action_step = "{{\"action_type\": {}, \"click_point\": {}, \"value\": \"{}\"}}".format(3, click_point,
                                                                                               typed_text)

    if return_bbox:
        return action_step, bbox
    else:
        return action_step


# calculate action f1 following mind2web
def calculate_f1(pred, label):
    pred = set(pred.strip().split())
    label = set(label.strip().split())
    if len(pred) == 0 and len(label) == 0:
        return 1
    if len(pred) == 0 or len(label) == 0:
        return 0

    tp = len(pred & label)
    fp = len(pred - label)
    fn = len(label - pred)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision == 0 or recall == 0:
        return 0
    f1 = 2 * precision * recall / (precision + recall)
    return f1


torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(0)

task_list = [args.task]
for args_task in task_list:

    # Profiler total FLOPs list: 每个 step 的完整 generate 总FLOPs
    profiler_total_flops_list = []
    # Prefill time list (ms): 每个 step 记录一次，最终取全数据集平均
    prefill_time_list = []

    mind2web_imgs_dir = "/data_sdc/data/ldq/Dataset/Mind2Web/ming2web_images"
    mind2web_test = json.load(open('/data_sdc/data/ldq/Dataset/Mind2Web/' + 'mind2web_data_test_' + args_task + '.json', 'r'))
    prompt_origin = "Please generate the next move according to the instruction, previous actions, previous ui screenshot and current ui screenshot. Instruction: {}\n"
    results = []
    global_valid_step_idx = 0
    for episode in tqdm(mind2web_test):
        goal = episode["confirmed_task"]
        annot_id = episode["annotation_id"]
        previous_actions = []
        results_actions = []
        previous_imgs = []

        for j, step in enumerate(episode["actions"]):
            if "bbox" not in step:
                print("action not found")
                continue

            filename = annot_id + '-' + step["action_uid"] + '.jpg'
            img_path = os.path.join(mind2web_imgs_dir, filename)
            if not os.path.exists(img_path):
                print("img not found")
                continue
            image = Image.open(img_path)

            prompt = prompt_origin.format(goal)
            cur_all_imgs = []

            previous_step = ""
            cur_step_idx = len(previous_imgs[-args.his_num:])
            for i, action in enumerate(previous_actions[-args.his_num:]):
                    prompt += 'Image_' + str(i) + ":<image>\n"
                    prompt += 'Step_' + str(i) + ': ' + action + " .\n"
                    cur_all_imgs.append(previous_imgs[-args.his_num:][i])

            action_step = action2step(step, image.size)
            previous_actions.append(action_step)
            previous_imgs.append(img_path)
            cur_all_imgs.append(img_path)

            prompt += 'Image_' + str(cur_step_idx) + ":<image>\n"

            action_step_ref, bbox_ref = action2step(step, image.size, return_bbox=True)
            try:
                action_step_ref = ast.literal_eval(action_step_ref)
            except:
                continue

            should_profile_this_step = False
            if args.profile_total_flops:
                n = int(args.profile_every_n_steps) if int(args.profile_every_n_steps) > 0 else 1
                should_profile_this_step = (global_valid_step_idx % n == 0)

            if should_profile_this_step:
                _activities = [ProfilerActivity.CPU]
                if torch.cuda.is_available():
                    _activities.append(ProfilerActivity.CUDA)
                with profile(
                    activities=_activities,
                    with_flops=True,
                    record_shapes=False,
                    profile_memory=False,
                ) as prof:
                    response = generate_grounding(cur_all_imgs, prompt, vis_meta={'ep_id': annot_id, 'step_id': j})
                step_total_prof_flops = extract_total_flops_from_prof(prof)
                profiler_total_flops_list.append(int(step_total_prof_flops))
                print(f"[PROFILER-FLOPs][STEP] ep={annot_id} step={j} global_step={global_valid_step_idx} total_flops={int(step_total_prof_flops)}")
            else:
                response = generate_grounding(cur_all_imgs, prompt, vis_meta={'ep_id': annot_id, 'step_id': j})
            response = process_string(response)
            global_valid_step_idx += 1

            prefill_ms = getattr(model.model, "_last_prefill_time_ms", None)
            if prefill_ms is not None:
                prefill_time_list.append(float(prefill_ms))

            n_before_dbg = getattr(model.model, "_n_before_prefill", None)
            n_after_dbg = getattr(model.model, "_n_after_hook", None)

            if args.dart_debug and args.dart_enable:
                _dart_mask = getattr(model.model, '_dart_keep_mask', None)
                _oas = getattr(model.model, '_last_oas_index', None)
                print(f"[DART-DEBUG][STEP] ep={annot_id} step={j} n_before={n_before_dbg} n_after={n_after_dbg}")
                print("[DART-DEBUG][STEP] dart_keep_ratios(effective)=", getattr(model.model, 'dart_keep_ratios', None))
                print("[DART-DEBUG][STEP] has_oas=", bool(_oas is not None), "dart_keep_mask_exists=", bool(_dart_mask is not None))
                if _dart_mask is not None:
                    try:
                        print("[DART-DEBUG][STEP] dart_keep_mask_shape=", tuple(_dart_mask.shape))
                    except Exception:
                        pass
                # batch-size=1 for this eval loop
                if (_dart_mask is not None) and isinstance(_oas, list) and len(_oas) > 0 and isinstance(_oas[0], list):
                    try:
                        km_1d = _dart_mask[0].detach()
                        idx_list_1 = _oas[0]
                        _dbg_input_ids = getattr(model.model, '_last_eval_input_ids', None)
                        if isinstance(idx_list_1, list) and len(idx_list_1) > 1 and (_dbg_input_ids is not None):
                            hist_list = idx_list_1[:-1]  # old -> new
                            k = min(4, len(hist_list))
                            hist_recent = list(reversed(hist_list[-k:]))  # recent -> older
                            for rank, seg in enumerate(hist_recent):
                                if not (isinstance(seg, list) and len(seg) >= 2):
                                    continue
                                s, e = int(seg[0]), int(seg[1])
                                s = max(0, min(s, km_1d.shape[0]))
                                e = max(0, min(e, km_1d.shape[0]))
                                if e <= s:
                                    continue
                                seg_ids = _dbg_input_ids[0, s:e]
                                seg_valid = (seg_ids != 151643)
                                total_n = int(seg_valid.sum().item())
                                if total_n <= 0:
                                    print(f"[DART-DEBUG][STEP] hist_rank={rank} seg=[{s},{e}) total_valid=0 keep=0 keep_ratio_actual=0.0000")
                                    continue
                                keep_n = int((km_1d[s:e] & seg_valid).sum().item())
                                keep_r = float(keep_n) / float(total_n)
                                print(f"[DART-DEBUG][STEP] hist_rank={rank} seg=[{s},{e}) keep={keep_n} total_valid={total_n} keep_ratio_actual={keep_r:.4f}")
                    except Exception as e:
                        print("[DART-DEBUG][STEP] per-history stats failed:", repr(e))

            if args.pdrop_debug and args.pdrop_enable:
                _pdrop_mask = getattr(model.model, '_pdrop_keep_mask', None)
                _oas = getattr(model.model, '_last_oas_index', None)
                print(f"[PDROP-DEBUG][STEP] ep={annot_id} step={j} n_before={n_before_dbg} n_after={n_after_dbg}")
                print("[PDROP-DEBUG][STEP] pdrop_keep_ratios(effective)=", getattr(model.model, 'pdrop_keep_ratios', None))
                print("[PDROP-DEBUG][STEP] has_oas=", bool(_oas is not None), "pdrop_keep_mask_exists=", bool(_pdrop_mask is not None))
                if _pdrop_mask is not None:
                    try:
                        print("[PDROP-DEBUG][STEP] pdrop_keep_mask_shape=", tuple(_pdrop_mask.shape))
                    except Exception:
                        pass

            # 重置 hook 避免干扰下一个 step
            model.model._n_after_hook = None
            model.model._n_before_prefill = None


            step_result = {"annot_id": annot_id, "img_path": img_path, "instruction": goal, "sentence": response,
                        "Op_match": False, "Ele_match": False, "Op_F1": [0, action_step_ref["action_type"]]}
            try:
                action_pred = ast.literal_eval(response)

                if action_pred["action_type"] == action_step_ref["action_type"]:
                    step_result["Op_match"] = True

                click_point = action_pred["click_point"]

                if (bbox_ref[0] <= click_point[0] <= bbox_ref[2]) and (bbox_ref[1] <= click_point[1] <= bbox_ref[3]):
                    step_result["Ele_match"] = True

                # 按照mind2web的方式，把action转换成一个字符串，即如果是TYPE需要考虑字符间的F1
                pred_str = str(action_pred["action_type"])
                if action_pred["action_type"] == 3 or action_pred["action_type"] == 2:
                    pred_str += ' '
                    pred_str += action_pred["value"].lower()
                ref_str = str(action_step_ref["action_type"])
                if action_step_ref["action_type"] == 3 or action_step_ref["action_type"] == 2:
                    ref_str += ' '
                    ref_str += action_step_ref["value"].lower()

                op_f1 = calculate_f1(pred_str, ref_str)
                step_result["Op_F1"][0] = op_f1

            except:
                logging.info("format wrong")

            # logging.info(step_result)

            results_actions.append(step_result)    

        results.append(results_actions)


    # calculate metrics
    num_step = 0
    num_episode = 0
    num_op = 0
    num_ele = 0
    op_f1 = {4: [], 2: [], 3: []}
    macro_ele_acc = {}
    macro_step_acc = {}
    macro_action_f1 = {}
    num_step_success = 0
    num_episode_success = 0
    for i, item in enumerate(results):
        macro_ele_acc[i] = []
        macro_step_acc[i] = []
        macro_action_f1[i] = []
        num_episode += 1
        episode_success = True
        for step_result in item:
            num_step += 1

            if step_result["Op_match"]:
                num_op += 1

            if step_result["Ele_match"]:
                num_ele += 1
                macro_ele_acc[i].append(1)
            else:
                macro_ele_acc[i].append(0)

            if step_result["Op_F1"][1] in op_f1:
                op_f1[step_result["Op_F1"][1]].append(step_result["Op_F1"][0])
            macro_action_f1[i].append(step_result["Op_F1"][0])

            if step_result["Op_F1"][0] == 1.0 and step_result["Ele_match"]:
                num_step_success += 1
                macro_step_acc[i].append(1)
            else:
                macro_step_acc[i].append(0)
                episode_success = False

        if episode_success:
            num_episode_success += 1

    marco_op_f1 = np.mean([np.mean(x) for x in op_f1.values()])

    logging.info("Operation F1: " + str(marco_op_f1))
    logging.info("Element Acc: " + str(num_ele / num_step))
    logging.info("Step Success: " + str(num_step_success / num_step))
    logging.info("Episode Success: " + str(num_episode_success / num_episode))
    logging.info("Operation F1 cate: " + str([np.mean(x) for x in op_f1.values()]))

    macro_ele_acc = np.mean([np.mean(x) for x in macro_ele_acc.values()])
    macro_step_acc = np.mean([np.mean(x) for x in macro_step_acc.values()])
    macro_action_f1 = np.mean([np.mean(x) for x in macro_action_f1.values()])
    logging.info("Macro Ele Acc: " + str(macro_ele_acc))
    logging.info("Macro Op F1: " + str(macro_action_f1))
    logging.info("Macro Step SR: " + str(macro_step_acc))

    results = {
        "Operation_F1": float(marco_op_f1),
        "Element_Accuracy": float(num_ele / num_step) if num_step != 0 else 0.0,
        "Step_Success_Rate": float(num_step_success / num_step) if num_step != 0 else 0.0,
        "Episode_Success_Rate": float(num_episode_success / num_episode) if num_episode != 0 else 0.0,
        "Operation_F1_Categories": [float(np.mean(x)) for x in op_f1.values()],
        "Macro_Element_Accuracy": float(macro_ele_acc),
        "Macro_Operation_F1": float(macro_action_f1),
        "Macro_Step_Success_Rate": float(macro_step_acc),
        "Average_FLOPs_Profiler_Total": float(np.mean(profiler_total_flops_list)) if profiler_total_flops_list else 0.0,
        "profiled_steps_count": int(len(profiler_total_flops_list)),
        "profile_total_flops_enabled": bool(args.profile_total_flops),
        "profile_every_n_steps": int(args.profile_every_n_steps),
        "avg_prefill_ms_per_step": float(np.mean(prefill_time_list)) if prefill_time_list else 0.0,
        "prefill_steps_counted": int(len(prefill_time_list)),
    }
    print(args_task)
    print(results)
    print(f"Average FLOPs (profiler total): {results['Average_FLOPs_Profiler_Total']}")
    print(f"Profiled steps count: {results['profiled_steps_count']} (every {results['profile_every_n_steps']} steps)")
    print(f"Average prefill time per step (ms): {results['avg_prefill_ms_per_step']}")


    write_json(results, args.save_path, args_task)