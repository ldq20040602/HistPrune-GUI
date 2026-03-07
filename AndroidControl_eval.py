import json
import os
import re
from torch.profiler import profile, ProfilerActivity
IGNORE_INDEX = -100

DEFAULT_IM_START_TOKEN = "<|im_start|>"
DEFAULT_IM_END_TOKEN = "<|im_end|>"
DEFAULT_IMAGE_TOKEN = "<|image_pad|>"
DEFAULT_VIDEO_TOKEN = "<|video_pad|>"
LLAVA_IMAGE_TOKEN = "<image>"
LLAVA_VIDEO_TOKEN = "<video>"
VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"

SYSTEM_MESSAGE = "You are a helpful assistant."
from PIL import Image, ImageDraw, ImageFont
from typing import List

from src.training.my_qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoConfig
import importlib

def load_model_class(model_path: str):
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

# Optional Sobel prior for foreground/background split
try:
    from sobel_segmentation import compute_sobel_uigraph
    _HAS_SOBEL = True
except Exception:
    _HAS_SOBEL = False

import pdb

def write_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)


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

import argparse

# 创建解析器
parser = argparse.ArgumentParser(description='Specify paths for saving and loading models.')

# 添加参数
parser.add_argument('--save_path', type=str, default="/data_sdc/data/SimpAgent-main/eval_results/AndroidControl/Qwen2.5/Baseline2AO_FastV",
                    help='The path where the model will be saved')
parser.add_argument('--model_path', type=str, default="/data_sdc/data/SimpAgent-main/save/Qwen2/AndroidControl/baseline_lora_215",
                    help='The path where the model is loaded from')
parser.add_argument('--his_num', type=int, default=2,
                    help='The path where the model is loaded from')
parser.add_argument('--drop_k', type=int, default=3,
                    help='The path where the model is loaded from')
parser.add_argument('--alpha', type=int, default=1,
                    help='The path where the model is loaded from')

# ===== Token pruning controls (aligned with AITW_eval.py) =====
parser.add_argument('--fastv_enable', action='store_true', default=False,
                    help='Enable FastV token-level pruning at drop_k (training-free).')
parser.add_argument('--fastv_segment_keep_ratios', type=str, default="0.5,0.5,0.5,0.5",
                    help='Comma-separated keep ratios for segmented FastV per history image from most recent to 4th (e.g., 0.8,0.5,0.3,0.2).')
parser.add_argument('--pdrop_enable', action='store_true', default=False,
                    help='Enable PDrop token-level pruning at drop_k (training-free).')
parser.add_argument('--pdrop_debug', action='store_true', default=False,
                    help='Enable minimal PDrop debug logs (effective ratios and per-step keep stats).')
parser.add_argument('--pdrop_keep_ratio', type=str, default="0.5,0.5,0.5,0.5",
                    help='Comma-separated keep ratios for segmented PDrop per history image from most recent to 4th (e.g., 0.8,0.5,0.3,0.2).')

parser.add_argument('--sobel_edge_thr', type=int, default=50,
                    help='Sobel magnitude threshold (0~255) to count an edge pixel within a 28x28 patch.')
parser.add_argument('--sobel_ratio_thr', type=float, default=0.01,
                    help='Per-patch edge pixel ratio threshold in [0,1]; >= ratio means an edge patch (won\'t merge).')

parser.add_argument('--DivPrune_enable', action='store_true', default=False,
                    help='Enable DivPrune submodular pruning (training-free).')
parser.add_argument('--DivPrune_ratio', type=str, default="0.5,0.5",
                    help='Comma-separated keep ratios for DivPrune per history image from most recent to 4th.')

parser.add_argument('--sobel_enable', action='store_true', default=False,
                    help='Enable Sobel edge-only pruning at drop_k (training-free).')
parser.add_argument('--sobel_vis_enable', action='store_true', default=False,
                    help='Enable visualization of Sobel pruning results when sobel_enable is also True.')

parser.add_argument('--random_enable', action='store_true', default=False,
                    help='Enable Random token-level pruning on historical screenshots at drop_k (training-free).')
parser.add_argument('--random_keep_ratio', type=str, default="0.5,0.5,0.5,0.5",
                    help='Comma-separated keep ratios for Random pruning per history image from most recent to 4th.')

parser.add_argument('--sparsevlm_enable', action='store_true', default=False,
                    help='Enable SparseVLM pruning (Stage1 raters + Stage2 global visual pruning).')
parser.add_argument('--sparsevlm_keep_ratio', type=float, default=0.4,
                    help='SparseVLM keep ratio for historical visual tokens (e.g., 0.4 keeps top 40% and prunes 60%).')

parser.add_argument('--fastv_vis_enable', action='store_true', default=False,
                    help='Enable visualization of FastV pruning results when fastv_enable is also True.')
parser.add_argument('--fastv_debug', action='store_true', default=False,
                    help='Enable minimal FastV debug logs (effective ratios and per-step keep stats).')
parser.add_argument('--dart_debug', action='store_true', default=False,
                    help='Enable minimal DART debug logs (effective ratios and per-step keep stats).')

parser.add_argument('--dart_enable', action='store_true', default=False,
                    help='Enable DART token-level pruning at drop_k (training-free).')
parser.add_argument('--dart_ratios', type=str, default="0.5,0.5,0.5,0.5",
                    help='Comma-separated keep ratios for DART per history image from most recent to 4th.')
parser.add_argument('--dart_num_pivots', type=int, default=10,
                    help='Number of pivot tokens (top-L2) per history image for DART.')

parser.add_argument('--profile_total_flops', action='store_true', default=False,
                    help='Enable torch profiler to measure per-step total FLOPs of full generate call.')
parser.add_argument('--profile_every_n_steps', type=int, default=20,
                    help='Profile once every N valid evaluation steps when --profile_total_flops is enabled.')
parser.add_argument('--flops_max_steps', type=int, default=100,
                    help='Only compute FLOPs statistics on the first N valid steps.')

# 解析参数
args = parser.parse_args()
args.save_path = args.save_path + '.json'

import torch
# Default: Load the model on the available device(s)
ModelClass = load_model_class(args.model_path)
model = ModelClass.from_pretrained(
    # "OS-Copilot/OS-Atlas-Base-7B", torch_dtype="auto", device_map="auto"
    args.model_path, torch_dtype=torch.bfloat16, device_map="auto"
    #"/home/wentao/project/gui_ads/OS-Atlas-Base-7B", torch_dtype="auto", device_map="auto"
)
min_pixels = 200704
max_pixels = 1003520
processor = AutoProcessor.from_pretrained(
    args.model_path,
    trust_remote_code=True,
    min_pixels=min_pixels,
    max_pixels=max_pixels,
)

# Profiler total FLOPs list: 每个 step 的完整 generate 总FLOPs
profiler_total_flops_list = []
# Prefill time list (ms): 每个 step 记录一次，最终取全数据集平均
prefill_time_list = []
# 全局有效step计数（用于profile采样）
global_valid_step_idx = 0
# FLOPs统计计数（只统计前N个有效step）
flops_counted_steps = 0
flops_early_summary_printed = False


def safe_setattr(obj, attr, value):
    if hasattr(obj, attr):
        setattr(obj, attr, value)


model.drop_k = args.drop_k
model.model.drop_k = args.drop_k
model.alpha = args.alpha

# ===== Enable pruning logic (aligned with AITW_eval.py) =====
if args.fastv_enable:
    safe_setattr(model.model, "enable_fastv_pruning", True)

if args.pdrop_enable:
    safe_setattr(model.model, "enable_pdrop_pruning", True)
try:
    pdrop_ratios = [float(x) for x in str(args.pdrop_keep_ratio).split(',') if str(x).strip() != '']
    pdrop_ratios = [0.0 if r < 0.0 else (1.0 if r > 1.0 else r) for r in pdrop_ratios]
    if len(pdrop_ratios) == 0:
        raise ValueError("pdrop_keep_ratio is empty after parsing")
    setattr(model.model, "pdrop_keep_ratios", pdrop_ratios)
    safe_setattr(model.model, "pdrop_keep_ratio_m", float(pdrop_ratios[0]))
except Exception as e:
    raise ValueError(f"Invalid --pdrop_keep_ratio: {args.pdrop_keep_ratio!r}, err={e}")

if args.pdrop_debug:
    print("[PDROP-DEBUG] cli.pdrop_enable=", bool(args.pdrop_enable))
    print("[PDROP-DEBUG] cli.pdrop_keep_ratio(raw)=", repr(args.pdrop_keep_ratio))
    print("[PDROP-DEBUG] parsed_keep_ratios=", pdrop_ratios if 'pdrop_ratios' in locals() else None)
    print("[PDROP-DEBUG] model.enable_pdrop_pruning=", bool(getattr(model.model, 'enable_pdrop_pruning', False)))
    print("[PDROP-DEBUG] model.pdrop_keep_ratios(effective)=", getattr(model.model, 'pdrop_keep_ratios', None))
try:
    ratios = [float(x) for x in str(args.fastv_segment_keep_ratios).split(',') if str(x).strip() != '']
    ratios = [0.0 if r < 0.0 else (1.0 if r > 1.0 else r) for r in ratios]
    if len(ratios) == 0:
        raise ValueError("fastv_segment_keep_ratios is empty after parsing")
    setattr(model.model, "fastv_segment_keep_ratios", ratios)
except Exception as e:
    raise ValueError(f"Invalid --fastv_segment_keep_ratios: {args.fastv_segment_keep_ratios!r}, err={e}")

if args.fastv_debug:
    print("[FASTV-DEBUG] cli.fastv_enable=", bool(args.fastv_enable))
    print("[FASTV-DEBUG] cli.fastv_segment_keep_ratios(raw)=", repr(args.fastv_segment_keep_ratios))
    print("[FASTV-DEBUG] parsed_keep_ratios=", ratios if 'ratios' in locals() else None)
    print("[FASTV-DEBUG] model.enable_fastv_pruning=", bool(getattr(model.model, 'enable_fastv_pruning', False)))
    print("[FASTV-DEBUG] model.fastv_segment_keep_ratios(effective)=", getattr(model.model, 'fastv_segment_keep_ratios', None))

if args.sobel_enable:
    safe_setattr(model.model, "sobel_enable", True)

if args.DivPrune_enable:
    safe_setattr(model.model, "enable_DivPrune_pruning", True)
try:
    dpratios = [float(x) for x in str(args.DivPrune_ratio).split(',') if str(x).strip() != '']
    dpratios = [0.0 if r < 0.0 else (1.0 if r > 1.0 else r) for r in dpratios]
    setattr(model.model, "DivPrune_ratios", dpratios)
except Exception as e:
    raise ValueError(f"Invalid --DivPrune_ratio: {args.DivPrune_ratio!r}, err={e}")

if args.random_enable:
    safe_setattr(model.model, "enable_random_pruning", True)
try:
    rratios = [float(x) for x in str(args.random_keep_ratio).split(',') if str(x).strip() != '']
    rratios = [0.0 if r < 0.0 else (1.0 if r > 1.0 else r) for r in rratios]
    setattr(model.model, "random_keep_ratios", rratios[:4])
except Exception as e:
    raise ValueError(f"Invalid --random_keep_ratio: {args.random_keep_ratio!r}, err={e}")
safe_setattr(model.model, "random_prune_seed", 0)

if args.sparsevlm_enable:
    safe_setattr(model.model, "enable_sparsevlm_pruning", True)
    safe_setattr(model.model, "sparsevlm_keep_ratio", float(args.sparsevlm_keep_ratio))

# DART pruning controls via CLI (strict setattr, no fallback)
if args.dart_enable:
    setattr(model.model, "enable_dart_pruning", True)

dart_parse_ok = False
dart_parse_err = None
parsed_dart_ratios = None
try:
    dratios = [float(x) for x in str(args.dart_ratios).split(',') if str(x).strip() != '']
    dratios = [0.0 if r < 0.0 else (1.0 if r > 1.0 else r) for r in dratios]
    if len(dratios) == 0:
        raise ValueError("dart_ratios is empty after parsing")
    setattr(model.model, "dart_keep_ratios", dratios)
    dart_parse_ok = True
    parsed_dart_ratios = dratios
except Exception as e:
    dart_parse_err = repr(e)

try:
    setattr(model.model, "dart_num_pivots", int(args.dart_num_pivots))
except Exception as e:
    raise ValueError(f"Invalid --dart_num_pivots: {args.dart_num_pivots!r}, err={e}")

if args.dart_enable and (not dart_parse_ok):
    raise ValueError(f"[DART] invalid --dart_ratios={args.dart_ratios!r}, err={dart_parse_err}")
if args.dart_enable and (not isinstance(getattr(model.model, 'dart_keep_ratios', None), list) or len(getattr(model.model, 'dart_keep_ratios', [])) == 0):
    raise RuntimeError("[DART] dart_keep_ratios not effective on model; DART would not be used")
if args.dart_enable and (not bool(getattr(model.model, 'enable_dart_pruning', False))):
    raise RuntimeError("[DART] enable_dart_pruning not effective on model")

if args.dart_debug:
    print("[DART-DEBUG] cli.dart_enable=", bool(args.dart_enable))
    print("[DART-DEBUG] cli.dart_ratios(raw)=", repr(args.dart_ratios))
    print("[DART-DEBUG] parse_ok=", dart_parse_ok)
    if dart_parse_ok:
        print("[DART-DEBUG] parsed_keep_ratios=", parsed_dart_ratios)
    else:
        print("[DART-DEBUG] parse_error=", dart_parse_err)
    print("[DART-DEBUG] model.enable_dart_pruning=", bool(getattr(model.model, 'enable_dart_pruning', False)))
    print("[DART-DEBUG] model.dart_keep_ratios(effective)=", getattr(model.model, 'dart_keep_ratios', None))
    print("[DART-DEBUG] model.dart_num_pivots=", getattr(model.model, 'dart_num_pivots', None))

_enabled = int(bool(args.fastv_enable)) + int(bool(args.pdrop_enable)) + int(bool(args.DivPrune_enable)) + int(bool(args.sobel_enable)) + int(bool(args.random_enable)) + int(bool(args.sparsevlm_enable)) + int(bool(args.dart_enable))
if _enabled > 1:
    raise ValueError("Only one of fastv_enable / pdrop_enable / DivPrune_enable / sobel_enable / random_enable / sparsevlm_enable / dart_enable can be True.")

# Sobel cache for historical screenshots: key=image path, value={'is_edge': List[bool]}
sobel_cache = {}


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
    if keep_mask_1d is None or oas_index_1 is None:
        return

    if not isinstance(oas_index_1, list) or len(oas_index_1) <= 1:
        return

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

        for idx in range(num_tokens):
            if bool(mask_seg[idx].item() if isinstance(mask_seg, torch.Tensor) else mask_seg[idx]):
                continue
            r = idx // grid_w
            c = idx % grid_w
            cx = c * cell_w + cell_w / 2.0
            cy = r * cell_h + cell_h / 2.0
            try:
                draw.text((cx, cy), "x", fill="red", font=font, anchor="mm")
            except Exception:
                draw.text((cx, cy), "x", fill="red", font=font)

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

        for idx in range(num_tokens):
            if bool(mask_seg[idx].item() if isinstance(mask_seg, torch.Tensor) else mask_seg[idx]):
                continue
            r = idx // grid_w
            c = idx % grid_w
            cx = c * cell_w + cell_w / 2.0
            cy = r * cell_h + cell_h / 2.0
            try:
                draw.text((cx, cy), "x", fill="red", font=font, anchor="mm")
            except Exception:
                draw.text((cx, cy), "x", fill="red", font=font)

        out_path = os.path.join(out_dir, f"history_{rank}_recent.png")
        try:
            pil.save(out_path)
        except Exception:
            continue


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

    # ===== FastV 文本 Query 区间硬编码（用于“文本 -> 历史截图token”的注意力聚合） =====
    try:
        ids = inputs.input_ids[0].tolist()

        general_instr_start = 14
        general_instr_len = 21
        general_instr_end = general_instr_start + general_instr_len

        goal_start = general_instr_end
        goal_end = None
        for _idx in range(goal_start, len(ids)):
            if ids[_idx] == 1906:
                goal_end = _idx
                break
        if goal_end is None:
            goal_end = len(ids)

        try:
            model.model._prompt_query_ranges = [{
                'p1': (int(general_instr_start), int(general_instr_end)),
                'p2': (int(goal_start), int(goal_end)),
            }]
            # PDrop: 使用文本部分最后一个 token 作为 query（沿用同样硬编码文本边界）
            model.model._pdrop_query_index = int(max(general_instr_start, goal_end - 1))
        except Exception:
            pass
    except Exception:
        pass
    # ===== 计算结束 =====

    # DivPrune 不再依赖 Sobel 先验
    model.model._DivPrune_info = None

    # Sobel prior attach (Batch size assumed 1)
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
def android_action2step(action_str, img_width, img_height):
    """
    将AndroidControl的动作字符串转换为标准格式
    支持所有8种动作类型: click, scroll, input_text, wait, open_app, navigate_back, long_press, navigate_home
    使用图片的实际尺寸进行坐标归一化
    """
    try:
        action_data = json.loads(action_str)
        action_type = action_data["action_type"]
        
        if action_type == "click" or action_type == "long_press":
            if action_type == "click":
               action_type_id = 4
            else:
                action_type_id = 11
                
            x = action_data["x"]
            y = action_data["y"]
            # 使用图片实际尺寸进行归一化到0-1000范围
            if img_width > 0 and img_height > 0:
                x_norm = int(1000 * x / img_width)
                y_norm = int(1000 * y / img_height)
                # 确保坐标在有效范围内
                x_norm = max(0, min(1000, x_norm))
                y_norm = max(0, min(1000, y_norm))
            else:
                # 如果尺寸无效，使用默认归一化
                x_norm = int(1000 * x / 2000)
                y_norm = int(1000 * y / 2000)
                
            return f'{{"action_type": {action_type_id}, "click_point": ({x_norm},{y_norm})}}'
        
        elif action_type == "input_text":
            action_type_id = 3
            text = action_data["text"]
            return f'{{"action_type": {action_type_id}, "typed_text": "{text}"}}'
        
        elif action_type == "scroll":
            direction = action_data["direction"]
            if direction == "down":
                action_type_id = 0
            elif direction == "up":
                action_type_id = 1
            elif direction == "left":
                action_type_id = 8
            elif direction == "right":
                action_type_id = 9
            else:
                action_type_id = 0
            return f'{{"action_type": {action_type_id}}}'
        
        elif action_type == "wait":
            action_type_id = 2
            return f'{{"action_type": {action_type_id}}}'
        
        elif action_type == "navigate_back":
            action_type_id = 5
            return f'{{"action_type": {action_type_id}}}'
        
        elif action_type == "open_app":
            action_type_id = 7
            app_name = action_data["app_name"]
            return f'{{"action_type": {action_type_id}, "app_name": "{app_name}"}}'
        
        elif action_type == "navigate_home":
            action_type_id = 6
            return f'{{"action_type": {action_type_id}}}'
        
        elif action_type == "finish":
            action_type_id = 10
            return f'{{"action_type": {action_type_id}}}'
        
        else:
            print(f"未知动作类型: {action_type}")
            return f'{{"action_type": 99}}'
            
    except Exception as e:
        print(f"Error parsing action: {action_str}, error: {e}")
        return f'{{"action_type": 99}}'

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
import numpy as np

import json
import os
from PIL import Image
from tqdm import tqdm
prompt_origin = "Please generate the next move according to the instruction, previous actions, previous ui screenshot and current ui screenshot. Instruction: {}\n"
with open('/data_sdc/data/ldq/Dataset/AndroidControl/android_control_splits.json', 'r') as file:
    splits=json.load(file)
    test_ids=splits['test']
    test_ids = list(set(test_ids))
    # test_ids = test_ids[0:len(test_ids) // 4]
outputs = []
with open('/data_sdc/data/ldq/Dataset/AndroidControl/androidcontrol_data_test.json', 'r') as file:
    data = json.load(file)


img_not_found = 0
llava_format_data = []
img_dir = "/data_sdc/data/ldq/Dataset/AndroidControl/images/"
for episode in tqdm(data):
    screenshot_widths = episode.get("screenshot_widths", [])
    screenshot_heights=episode.get("screenshot_heights", [])
    previous_imgs = []
    previous_actions = []
    flag = 0
    if episode['episode_id'] not in test_ids:
        # print(episode['episode_id'])
        continue
    for idx in range(len(episode['actions'])):
        step_data = {}

        img_filename = episode["images"][idx]
        img_path = os.path.join(img_dir, img_filename)
        if not os.path.exists(img_path):
            print('image not found')
            flag = 1
            continue
        img_width = screenshot_widths[idx] if idx < len(screenshot_widths) else 0
        img_height = screenshot_heights[idx] if idx < len(screenshot_heights) else 0


        goal = episode["goal"]

        prompt = prompt_origin.format(goal)

        cur_all_imgs = []
        hist_imgs = previous_imgs[-args.his_num:]
        hist_actions = previous_actions[-args.his_num:]
        cur_step_idx = len(hist_imgs)
        for i, action in enumerate(hist_actions):
            prompt += 'Image_' + str(i) + ":<image>\n\n"
            prompt += 'Step_' + str(i) + ':' + action + " .\n"
            cur_all_imgs.append(hist_imgs[i])

        action_str = episode['actions'][idx]
        action_step = android_action2step(action_str, img_width, img_height)

        previous_actions.append(action_step)
        previous_imgs.append(img_path)

        conversations = []
        prompt += 'Image_' + str(cur_step_idx) + ":<image>\n\n"
        # print(cur_all_imgs)
        cur_all_imgs.append(img_path)

        global_valid_step_idx += 1
        should_count_flops_this_step = (flops_counted_steps < max(1, int(args.flops_max_steps)))
        do_profile_this_step = (
            should_count_flops_this_step
            and bool(args.profile_total_flops)
            and (global_valid_step_idx % max(1, int(args.profile_every_n_steps)) == 0)
        )
        if do_profile_this_step:
            _activities = [ProfilerActivity.CPU]
            if torch.cuda.is_available():
                _activities.append(ProfilerActivity.CUDA)
            with profile(
                activities=_activities,
                with_flops=True,
                record_shapes=False,
                profile_memory=False,
            ) as prof:
                response = generate_grounding(cur_all_imgs, prompt, vis_meta={'ep_id': episode['episode_id'], 'step_id': idx})
            step_total_prof_flops = extract_total_flops_from_prof(prof)
            profiler_total_flops_list.append(int(step_total_prof_flops))
            print(f"[PROFILER-FLOPs][STEP] ep={episode.get('episode_id','?')} step={idx} global_step={global_valid_step_idx} flops_step={flops_counted_steps + 1}/{max(1, int(args.flops_max_steps))} total_flops={int(step_total_prof_flops)}")
        else:
            response = generate_grounding(cur_all_imgs, prompt, vis_meta={'ep_id': episode['episode_id'], 'step_id': idx})

        if should_count_flops_this_step:
            prefill_ms = getattr(model.model, "_last_prefill_time_ms", None)
            if prefill_ms is not None:
                prefill_time_list.append(float(prefill_ms))

            flops_counted_steps += 1

            if (not flops_early_summary_printed) and (flops_counted_steps >= max(1, int(args.flops_max_steps))):
                early_avg_prof = float(np.mean(profiler_total_flops_list)) if profiler_total_flops_list else 0.0
                early_avg_prefill = float(np.mean(prefill_time_list)) if prefill_time_list else 0.0
                print("\n" + "=" * 80)
                print(f"[FLOPS-EARLY-SUMMARY] reached flops_max_steps={int(args.flops_max_steps)}")
                print(f"Average FLOPs (profiler total, first {int(args.flops_max_steps)} steps): {early_avg_prof}")
                print(f"FLOPs counted steps: {int(flops_counted_steps)}")
                print(f"Profiled steps count: {int(len(profiler_total_flops_list))} (every {int(args.profile_every_n_steps)} steps)")
                print(f"Average prefill time per step (ms, first {int(args.flops_max_steps)} steps): {early_avg_prefill}")
                print("=" * 80 + "\n")
                flops_early_summary_printed = True

        n_before_dbg = getattr(model.model, "_n_before_prefill", None)
        n_after_dbg = getattr(model.model, "_n_after_hook", None)

        # FastV 每步调试：检查 FastV keep-mask 是否生效
        if args.fastv_debug and args.fastv_enable:
            try:
                keep_mask_dbg = getattr(model.model, '_fastv_keep_mask', None)
                oas_dbg = getattr(model.model, '_last_oas_index', None)
                print(f"[FASTV-DEBUG][STEP] ep={episode.get('episode_id','?')} step={idx} n_before={n_before_dbg} n_after={n_after_dbg}")
                print("[FASTV-DEBUG][STEP] fastv_segment_keep_ratios(effective)=", getattr(model.model, 'fastv_segment_keep_ratios', None))
                print(f"[FASTV-DEBUG][STEP] has_oas={oas_dbg is not None} fastv_keep_mask_exists={keep_mask_dbg is not None}")
                if keep_mask_dbg is not None and oas_dbg is not None and isinstance(oas_dbg, list) and len(oas_dbg) > 0:
                    km = keep_mask_dbg[0].detach().to('cpu').to(torch.bool)
                    print("[FASTV-DEBUG][STEP] fastv_keep_mask_shape=", tuple(keep_mask_dbg.shape))
                    idx_list = oas_dbg[0]
                    if isinstance(idx_list, list) and len(idx_list) > 1:
                        hist_segments = [seg for seg in idx_list[:-1] if isinstance(seg, list) and len(seg) >= 2]
                        k = min(4, len(hist_segments))
                        segs_recent = list(reversed(hist_segments[-k:]))  # rank0=最近
                        seg_keep = []
                        for seg in segs_recent:
                            s, e = int(seg[0]), int(seg[1])
                            s = max(0, min(s, km.numel())); e = max(0, min(e, km.numel()))
                            if e <= s:
                                seg_keep.append((0, 0, 0.0))
                                continue
                            seg_mask = km[s:e]
                            n = int(seg_mask.numel())
                            keep_n = int(seg_mask.sum().item())
                            keep_r = (keep_n / n) if n > 0 else 0.0
                            seg_keep.append((keep_n, n, keep_r))
                        print(f"[FASTV-DEBUG][STEP] seg_keep(rank0->older)={seg_keep}")
            except Exception as e:
                print("[FASTV-DEBUG] per-step debug failed:", repr(e))

        # DART 每步调试：检查 DART keep-mask 是否生效
        if args.dart_debug and args.dart_enable:
            try:
                keep_mask_dbg = getattr(model.model, '_dart_keep_mask', None)
                oas_dbg = getattr(model.model, '_last_oas_index', None)
                print(f"[DART-DEBUG][STEP] ep={episode.get('episode_id','?')} step={idx} n_before={n_before_dbg} n_after={n_after_dbg}")
                print("[DART-DEBUG][STEP] dart_keep_ratios(effective)=", getattr(model.model, 'dart_keep_ratios', None))
                print(f"[DART-DEBUG][STEP] has_oas={oas_dbg is not None} dart_keep_mask_exists={keep_mask_dbg is not None}")
                if keep_mask_dbg is not None and oas_dbg is not None and isinstance(oas_dbg, list) and len(oas_dbg) > 0:
                    km = keep_mask_dbg[0].detach().to('cpu').to(torch.bool)
                    print("[DART-DEBUG][STEP] dart_keep_mask_shape=", tuple(keep_mask_dbg.shape))
                    idx_list = oas_dbg[0]
                    if isinstance(idx_list, list) and len(idx_list) > 1:
                        hist_segments = [seg for seg in idx_list[:-1] if isinstance(seg, list) and len(seg) >= 2]
                        k = min(4, len(hist_segments))
                        segs_recent = list(reversed(hist_segments[-k:]))  # rank0=最近
                        seg_keep = []
                        for seg in segs_recent:
                            s, e = int(seg[0]), int(seg[1])
                            s = max(0, min(s, km.numel())); e = max(0, min(e, km.numel()))
                            if e <= s:
                                seg_keep.append((0, 0, 0.0))
                                continue
                            seg_mask = km[s:e]
                            n = int(seg_mask.numel())
                            keep_n = int(seg_mask.sum().item())
                            keep_r = (keep_n / n) if n > 0 else 0.0
                            seg_keep.append((keep_n, n, keep_r))
                        print(f"[DART-DEBUG][STEP] seg_keep(rank0->older)={seg_keep}")
            except Exception as e:
                print("[DART-DEBUG] per-step debug failed:", repr(e))

        # PDrop 每步调试：检查 PDrop keep-mask 是否生效
        if args.pdrop_debug and args.pdrop_enable:
            try:
                keep_mask_dbg = getattr(model.model, '_pdrop_keep_mask', None)
                oas_dbg = getattr(model.model, '_last_oas_index', None)
                print(f"[PDROP-DEBUG][STEP] ep={episode.get('episode_id','?')} step={idx} n_before={n_before_dbg} n_after={n_after_dbg}")
                print("[PDROP-DEBUG][STEP] pdrop_keep_ratios(effective)=", getattr(model.model, 'pdrop_keep_ratios', None))
                print(f"[PDROP-DEBUG][STEP] has_oas={oas_dbg is not None} pdrop_keep_mask_exists={keep_mask_dbg is not None}")
                if keep_mask_dbg is not None and oas_dbg is not None and isinstance(oas_dbg, list) and len(oas_dbg) > 0:
                    km = keep_mask_dbg[0].detach().to('cpu').to(torch.bool)
                    print("[PDROP-DEBUG][STEP] pdrop_keep_mask_shape=", tuple(keep_mask_dbg.shape))
                    idx_list = oas_dbg[0]
                    if isinstance(idx_list, list) and len(idx_list) > 1:
                        hist_segments = [seg for seg in idx_list[:-1] if isinstance(seg, list) and len(seg) >= 2]
                        k = min(4, len(hist_segments))
                        segs_recent = list(reversed(hist_segments[-k:]))  # rank0=最近
                        seg_keep = []
                        for seg in segs_recent:
                            s, e = int(seg[0]), int(seg[1])
                            s = max(0, min(s, km.numel())); e = max(0, min(e, km.numel()))
                            if e <= s:
                                seg_keep.append((0, 0, 0.0))
                                continue
                            seg_mask = km[s:e]
                            n = int(seg_mask.numel())
                            keep_n = int(seg_mask.sum().item())
                            keep_r = (keep_n / n) if n > 0 else 0.0
                            seg_keep.append((keep_n, n, keep_r))
                        print(f"[PDROP-DEBUG][STEP] seg_keep(rank0->older)={seg_keep}")
            except Exception as e:
                print("[PDROP-DEBUG] per-step debug failed:", repr(e))

        # 重置 hook 避免干扰下一个 step
        model.model._n_after_hook = None
        model.model._n_before_prefill = None

        outputs.append({
                    'episode_id' : episode['episode_id'],
                    'pred': response,
                    'gt': action_step,
                })
        #print(outputs[-1])



avg_profiler_total_flops = float(np.mean(profiler_total_flops_list)) if profiler_total_flops_list else 0.0
avg_prefill_ms = float(np.mean(prefill_time_list)) if prefill_time_list else 0.0
print(f"Average FLOPs (profiler total, first {int(args.flops_max_steps)} steps): {avg_profiler_total_flops}")
print(f"FLOPs counted steps: {int(flops_counted_steps)}")
print(f"Profiled steps count: {int(len(profiler_total_flops_list))} (every {int(args.profile_every_n_steps)} steps)")
print(f"Average prefill time per step (ms, first {int(args.flops_max_steps)} steps): {avg_prefill_ms}")

print(f"Saving predict result ...")
# time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
savefile = os.path.join("/data_sdc/data/SimpAgent-main/eval_results/AndroidControl/Qwen2.5/Baseline2AO_FastV","AndroidControl_results.json")
json.dump({
    'outputs': outputs,
    'Average_FLOPs_Profiler_Total': avg_profiler_total_flops,
    'flops_max_steps': int(args.flops_max_steps),
    'flops_counted_steps': int(flops_counted_steps),
    'profiled_steps_count': int(len(profiler_total_flops_list)),
    'profile_total_flops_enabled': bool(args.profile_total_flops),
    'profile_every_n_steps': int(args.profile_every_n_steps),
    'avg_prefill_ms_per_step': avg_prefill_ms,
    'prefill_steps_counted': int(len(prefill_time_list)),
}, open(savefile, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)



