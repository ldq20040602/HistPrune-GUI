import json
import os
import re
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
from transformers import AutoProcessor, AutoConfig, AutoModelForVision2Seq
from torch.profiler import profile, ProfilerActivity
import importlib
from src.training.my_qwen_vl_utils import process_vision_info_with_resize
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

# Optional Sobel prior for foreground/background split
try:
    from sobel_segmentation import compute_sobel_uigraph
    _HAS_SOBEL = True
except Exception:
    _HAS_SOBEL = False
    print("[IMPORT sobel_segmentation FAILED]", repr(e))

import pdb
import sys

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


def _build_inputs_for_generation(image_path, query):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": query},
            ],
        }
    ]

    images = []
    for image_file in image_path:
        images.append(get_image_info(image_file))

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    text = text.replace(LLAVA_IMAGE_TOKEN, VISION_START_TOKEN + DEFAULT_IMAGE_TOKEN + VISION_END_TOKEN)

    inputs = processor(
        text=[text],
        images=images,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)
    return inputs


def profile_prefill_flops_only(image_path, query) -> int:
    """Profile FLOPs of one prefill forward pass (LLM path inside model.forward)."""
    inputs = _build_inputs_for_generation(image_path, query)
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    with torch.inference_mode():
        with profile(
            activities=activities,
            with_flops=True,
            record_shapes=False,
            profile_memory=False,
        ) as prof:
            _ = model(
                **inputs,
                use_cache=True,
                return_dict=True,
            )
    return extract_total_flops_from_prof(prof)


import argparse

# 创建解析器
parser = argparse.ArgumentParser(description='Specify paths for saving and loading models.')

# 添加参数
parser.add_argument('--save_path', type=str, default="/data_sdc/data/SimpAgent-main/eval_results/AITW",
                    help='The path where the result will be saved')
parser.add_argument('--model_path', type=str, default="/data_sdc/data/SimpAgent-main/save/Qwen2/AITW/Baseline4AO_lora",
                    help='The path where the model is loaded from')
parser.add_argument('--his_num', type=int, default=4,
                    help='The path where the model is loaded from')
parser.add_argument('--drop_k', type=int, default=28,
                    help='The path where the model is loaded from')
                    
parser.add_argument('--fastv_enable', action='store_true', default=False,
                    help='Enable FastV token-level pruning at drop_k (training-free).')
parser.add_argument('--fastv_segment_keep_ratios', type=str, default="0.5,0.25,0.125,0.0625",
                    help='Comma-separated keep ratios for segmented FastV per history image from most recent to 4th (e.g., 0.8,0.5,0.3,0.2).')
parser.add_argument('--pdrop_enable', action='store_true', default=False,
                    help='Enable PDrop token-level pruning at drop_k (training-free).')
parser.add_argument('--pdrop_keep_ratio', type=str, default="0.5,0.5,0.5,0.5",
                    help='Comma-separated keep ratios for segmented PDrop per history image from most recent to 4th (e.g., 0.8,0.5,0.3,0.2).')

parser.add_argument('--sobel_edge_thr', type=int, default=50,
                    help='Sobel magnitude threshold (0~255) to count an edge pixel within a 28x28 patch.')
parser.add_argument('--sobel_ratio_thr', type=float, default=0.01,
                    help='Per-patch edge pixel ratio threshold in [0,1]; >= ratio means an edge patch (won\'t merge).')

parser.add_argument('--DivPrune_enable', action='store_true', default=False,
                    help='Enable DivPrune submodular pruning (training-free).')
parser.add_argument('--DivPrune_ratio', type=str, default="0.5,0.25,0.125,0.0625",
                    help='Comma-separated keep ratios for DivPrune per history image from most recent to 4th (e.g., 0.8,0.64,0.512,0.4096).')

parser.add_argument('--sobel_enable', action='store_true', default=False,
                    help='Enable Sobel edge-only pruning at drop_k (training-free).')
parser.add_argument('--sobel_vis_enable', action='store_true', default=False,
                    help='Enable visualization of Sobel pruning results when sobel_enable is also True.')

parser.add_argument('--random_enable', action='store_true', default=True,
                    help='Enable Random token-level pruning on historical screenshots at drop_k (training-free).')
parser.add_argument('--random_keep_ratio', type=str, default="1.0,1.0,1.0,1.0",
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
parser.add_argument('--profile_prefill_flops', action='store_true', default=False,
                    help='Enable torch profiler to measure per-step prefill-only FLOPs (without full-generate total FLOPs).')
parser.add_argument('--profile_every_n_steps', type=int, default=20,
                    help='Profile once every N valid evaluation steps when profiler FLOPs options are enabled.')


# 解析参数
args = parser.parse_args()
args.save_path = args.save_path + args.model_path.split('/')[-1] + "_drop_" + str(args.drop_k) + '.json'
from transformers import AutoConfig# Default: Load the model on the available device(s)
# config = AutoConfig.from_pretrained(args.model_path)
# config.drop_k = args.drop_k
import torch
device='cuda:0'

# Default: Load the model on the available device(s)
ModelClass = load_model_class(args.model_path)
model = ModelClass.from_pretrained(
    args.model_path,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2",
)

def safe_setattr(obj, attr, value):
    if hasattr(obj, attr):
        setattr(obj, attr, value)

model.model.drop_k = args.drop_k
model.drop_k = args.drop_k
#safe_setattr(model.model, "drop_k", args.drop_k)
#safe_setattr(model, "drop_k", args.drop_k)
print("drop_k:", getattr(model, "drop_k", "N/A"))

# FastV pruning controls via CLI
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
    # backward compatibility for any legacy code path
    safe_setattr(model.model, "pdrop_keep_ratio_m", float(pdrop_ratios[0]))
except Exception as e:
    raise ValueError(f"Invalid --pdrop_keep_ratio: {args.pdrop_keep_ratio!r}, err={e}")

# Sobel edge-only pruning controls via CLI
if args.sobel_enable:
    safe_setattr(model.model, "sobel_enable", True)

# 解析分段 FastV 比例（只保留 segmented FastV，不再使用全局 fastv_keep_ratio）
fastv_segment_parse_ok = False
fastv_segment_parse_err = None
parsed_fastv_segment_ratios = None
try:
    ratios = [float(x) for x in str(args.fastv_segment_keep_ratios).split(',') if str(x).strip() != '']
    ratios = [0.0 if r < 0.0 else (1.0 if r > 1.0 else r) for r in ratios]
    if len(ratios) == 0:
        raise ValueError("fastv_segment_keep_ratios is empty after parsing")
    # 这里使用直接 setattr，避免被 safe_setattr 的 hasattr 保护静默跳过
    setattr(model.model, "fastv_segment_keep_ratios", ratios)
    fastv_segment_parse_ok = True
    parsed_fastv_segment_ratios = ratios
except Exception as e:
    fastv_segment_parse_err = repr(e)

# 强校验：启用 FastV 时，分段比例必须成功生效
if args.fastv_enable and (not fastv_segment_parse_ok):
    raise ValueError(f"[FASTV] invalid --fastv_segment_keep_ratios={args.fastv_segment_keep_ratios!r}, err={fastv_segment_parse_err}")
if args.fastv_enable and (not isinstance(getattr(model.model, 'fastv_segment_keep_ratios', None), list) or len(getattr(model.model, 'fastv_segment_keep_ratios', [])) == 0):
    raise RuntimeError("[FASTV] fastv_segment_keep_ratios not effective on model; segmented FastV would not be used")

if args.fastv_debug:
    print("[FASTV-DEBUG] cli.fastv_enable=", bool(args.fastv_enable))
    print("[FASTV-DEBUG] cli.fastv_segment_keep_ratios(raw)=", repr(args.fastv_segment_keep_ratios))
    print("[FASTV-DEBUG] model.enable_fastv_pruning=", bool(getattr(model.model, 'enable_fastv_pruning', False)))
    print("[FASTV-DEBUG] parse_ok=", fastv_segment_parse_ok)
    if fastv_segment_parse_ok:
        print("[FASTV-DEBUG] parsed_segment_keep_ratios=", parsed_fastv_segment_ratios)
    else:
        print("[FASTV-DEBUG] parse_error=", fastv_segment_parse_err)
    print("[FASTV-DEBUG] model.fastv_segment_keep_ratios(effective)=", getattr(model.model, 'fastv_segment_keep_ratios', None))

# DivPrune pruning controls via CLI
if args.DivPrune_enable:
    safe_setattr(model.model, "enable_DivPrune_pruning", True)

# Random pruning controls via CLI
if args.random_enable:
    safe_setattr(model.model, "enable_random_pruning", True)
try:
    rratios = [float(x) for x in str(args.random_keep_ratio).split(',') if str(x).strip() != '']
    rratios = [0.0 if r < 0.0 else (1.0 if r > 1.0 else r) for r in rratios]
    setattr(model.model, "random_keep_ratios", rratios[:4])
except Exception as e:
    raise ValueError(f"Invalid --random_keep_ratio: {args.random_keep_ratio!r}, err={e}")
safe_setattr(model.model, "random_prune_seed", 0)

# SparseVLM pruning controls via CLI
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

try:
    dpratios = [float(x) for x in str(args.DivPrune_ratio).split(',') if str(x).strip() != '']
    dpratios = [0.0 if r < 0.0 else (1.0 if r > 1.0 else r) for r in dpratios]
    setattr(model.model, "DivPrune_ratios", dpratios)
except Exception as e:
    raise ValueError(f"Invalid --DivPrune_ratio: {args.DivPrune_ratio!r}, err={e}")

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
)

# Sobel cache for historical screenshots: key=image path, value={'is_edge': List[bool]}
sobel_cache = {}

'''
# ===== 调试输出：核对特殊ID与token的映射（1906/198/16448 等）=====
try:
    tokenizer = processor.tokenizer
    print("=== Known special ids from config ===")
    print("config.image_token_id =", getattr(model.config, "image_token_id", None))
    print("config.video_token_id =", getattr(model.config, "video_token_id", None))

    print("\n=== Decode id -> token ===")
    ids_to_check = [1906, 198, 16448]
    for i in ids_to_check:
        try:
            toks = tokenizer.convert_ids_to_tokens([i])
            print(i, "->", toks)
        except Exception as e:
            print(i, "-> decode error:", e)

    print("\n=== Encode token -> ids (no special-added) ===")
    tokens_to_check = ["<|vision_start|>", "<|image_pad|>", "<|vision_end|>", "<image>", "<|im_start|>", "<|im_end|>"]
    for tok in tokens_to_check:
        try:
            ids = tokenizer.encode(tok, add_special_tokens=False)
            print(f"{tok} -> encode (no special):", ids)
        except Exception as e:
            print(f"{tok} -> encode error:", e)

    print("\n=== Direct convert_tokens_to_ids ===")
    for tok in tokens_to_check:
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
            print(f"{tok} -> id {tid}")
        except Exception as e:
            print(f"{tok} -> convert error:", e)
except Exception as e:
    print("Token mapping debug failed:", e)
# ===== 调试输出结束 =====
'''

def visualize_DivPrune_keep_mask(
    *,
    keep_mask_1d: torch.Tensor,  # [L]
    oas_index_1: list,           # one sample's OAS_index (list of [s,e) segments)
    hist_img_paths_old_to_new: List[str],
    ep_id: str,
    step_id: int,
    out_root: str = "/home/ldq/SimpAgent-main/output/greedy",
    max_hist: int = 4,
    patch_size: int = 14,
    spatial_merge_size: int = 2,
    min_pixel: int = None,
    max_pixel: int = None,
):
    """把 DivPrune 的 keep_mask 可视化到每张历史截图上。

    规则：
    - 每个 step 保存到 out_root/ep_id/step_id/
    - 只可视化最近 max_hist 张历史截图（不足则全画）
    - 被剪掉的 patch 在格子中心标注红色 "1"

    说明：
    - hist_img_paths_old_to_new: 历史图路径，按时间从旧到新（与你当前 cur_all_imgs[:-1] 的顺序一致）
    - 输出命名：history_0_recent.png 表示最近一张
    """


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
            print(f"[FastV-VIS] load image failed: {img_path}, err={ex}")
            continue

        W, H = pil.size
        grid_h = H // (patch_size * spatial_merge_size)
        grid_w = W // (patch_size * spatial_merge_size)
        if grid_h <= 0 or grid_w <= 0:
            continue

        num_tokens = int(grid_h * grid_w)
        seg_len = int(e - s)
        if seg_len != num_tokens:
            # 不一致则跳过（保守）
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
        except Exception as ex:
            print(f"[FastV-VIS] save failed: {out_path}, err={ex}")


    """把 Sparse Greedy 的 keep_mask 可视化到每张历史截图上。

    规则：
    - 每个 step 保存到 out_root/ep_id/step_id/
    - 只可视化最近 max_hist 张历史截图（不足则全画）
    - 被剪掉的 patch 在格子中心标注红色 "1"

    说明：
    - hist_img_paths_old_to_new: 历史图路径，按时间从旧到新（与你当前 cur_all_imgs[:-1] 的顺序一致）
    - 输出命名：history_0_recent.png 表示最近一张
    """
    """把 Sparse Greedy 的 keep_mask 可视化到每张历史截图上。

    规则：
    - 每个 step 保存到 out_root/ep_id/step_id/
    - 只可视化最近 max_hist 张历史截图（不足则全画）
    - 被剪掉的 patch 在格子中心标注红色 "1"

    说明：
    - hist_img_paths_old_to_new: 历史图路径，按时间从旧到新（与你当前 cur_all_imgs[:-1] 的顺序一致）
    - 输出命名：history_0_recent.png 表示最近一张
    """

    try:
        import numpy as np
    except Exception:
        np = None

    if keep_mask_1d is None or oas_index_1 is None:
        return

    if not isinstance(oas_index_1, list) or len(oas_index_1) <= 1:
        # 没历史或解析失败
        return

    # 取历史段（不含最后一段“当前图”）
    hist_segments_old_to_new = []
    for seg in oas_index_1[:-1]:
        if isinstance(seg, list) and len(seg) >= 2:
            s, e = int(seg[0]), int(seg[1])
            if e > s:
                hist_segments_old_to_new.append((s, e))

    if len(hist_segments_old_to_new) == 0 or len(hist_img_paths_old_to_new) == 0:
        return

    # 对齐：只取两边共同的历史长度
    n_hist = min(len(hist_segments_old_to_new), len(hist_img_paths_old_to_new))
    hist_segments_old_to_new = hist_segments_old_to_new[-n_hist:]
    hist_img_paths_old_to_new = hist_img_paths_old_to_new[-n_hist:]

    # 只可视化最近 max_hist 张：最近在末尾，所以取尾部再 reverse
    k = min(max_hist, n_hist)
    segs_recent = list(reversed(hist_segments_old_to_new[-k:]))
    imgs_recent = list(reversed(hist_img_paths_old_to_new[-k:]))

    # 创建输出目录：out_root/ep_id/step_id
    out_dir = os.path.join(out_root, str(ep_id), str(step_id))
    os.makedirs(out_dir, exist_ok=True)

    # 字体：优先 Arial，否则用默认
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        try:
            font = ImageFont.truetype("Arial.ttf", 14)
        except Exception:
            font = ImageFont.load_default()

    for rank, (img_path, (s, e)) in enumerate(zip(imgs_recent, segs_recent)):
        try:
            # 取与模型一致的 resize 后图片
            _minp = min_pixel if min_pixel is not None else min_pixels
            _maxp = max_pixel if max_pixel is not None else max_pixels
            pil = get_image_info(img_path, min_pixel=_minp, max_pixel=_maxp).convert("RGB")
        except Exception as ex:
            print(f"[VIS] load image failed: {img_path}, err={ex}")
            continue

        W, H = pil.size
        grid_h = H // (patch_size * spatial_merge_size)
        grid_w = W // (patch_size * spatial_merge_size)
        if grid_h <= 0 or grid_w <= 0:
            print(f"[VIS] invalid grid for {img_path}: size=({W},{H})")
            continue

        num_tokens = int(grid_h * grid_w)
        seg_len = int(e - s)
        if seg_len != num_tokens:
            # 你说两者严格一致；若不一致，先打印出来方便排查
            print(f"[VIS] token/grid mismatch for {img_path}: seg_len={seg_len}, grid={grid_h}x{grid_w}={num_tokens}")
            continue

        mask_seg = keep_mask_1d[s:e]
        if isinstance(mask_seg, torch.Tensor):
            mask_seg = mask_seg.detach().to("cpu").to(torch.bool)

        draw = ImageDraw.Draw(pil)
        cell_w = W / float(grid_w)
        cell_h = H / float(grid_h)

        # 标注被剪掉的 token
        for idx in range(num_tokens):
            if not bool(mask_seg[idx].item() if isinstance(mask_seg, torch.Tensor) else mask_seg[idx]):
                r = idx // grid_w
                c = idx % grid_w
                cx = c * cell_w + cell_w / 2.0
                cy = r * cell_h + cell_h / 2.0
                # 小“1”写在中心
                try:
                    draw.text((cx, cy), "1", fill="red", font=font, anchor="mm")
                except Exception:
                    # 兼容老 PIL 没有 anchor 的情况
                    draw.text((cx, cy), "1", fill="red", font=font)

        out_path = os.path.join(out_dir, f"history_{rank}_recent.png")
        try:
            pil.save(out_path)
        except Exception as ex:
            print(f"[VIS] save failed: {out_path}, err={ex}")




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

    image_input, _ = process_vision_info_with_resize(messages)

    # (debug removed)
    return image_input[0]


def generate_grounding(image_path, query):

    # Preparation for inference
    inputs = _build_inputs_for_generation(image_path, query)


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
            # PDrop: 使用文本部分最后一个 token 作为 query（沿用同样硬编码文本边界）
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

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )[0][0:-10]

    return output_text

    # print(output_text)

#     image = Image.open(image_path)
#     image_w, image_h = image.size

#     # 创建绘制对象
#     draw = ImageDraw.Draw(image)

#     try:
#         coordinates = re.findall(r"\((\d+),(\d+)\)", output_text[0])

#         # 将坐标转换为整数并保存为元组
#         points = [(int(int(x) / 1000 * image_w), int(int(y) / 1000 * image_h)) for x, y in coordinates]

#         # 绘制矩形框
#         draw.rectangle(points, outline="red", width=3)
#         draw.rectangle([tuple(truth[0]), tuple(truth[1])], outline="blue", width=3)

#         save_path = "/home/wentao/project/gui_ads/test_output/" + query + str(coordinates) + ".jpg"
#         # 显示图像
#         image = image.convert("RGB")
#         image.save(save_path)
#     except:
#         print("!!!!!!!!!!!!")

# _count = 0

from tqdm import tqdm
import action_matching 

def action2step(step_data):
    action_type = step_data["action_type_id"]

    if action_type == 4:
        if step_data["action_type_text"] == 'click':  # for click action, we calculate midpoint of touch and lift as the click point
            touch_point = step_data["touch"]
            lift_point = step_data["lift"]
            action_type_new = 4
            click_point = [int(1000 * (touch_point[0] + lift_point[0]) / 2), int(1000* (touch_point[1] + lift_point[1]) / 2)]
            click_point = [item for item in click_point]
            click_point = "({},{})".format(click_point[0], click_point[1])
            action = "{{\"action_type\": {}, \"click_point\": {}}}".format(action_type_new, click_point)
        else:  # for scroll action, we assign an action_type_id for each scroll
            if step_data["action_type_text"] == 'scroll down':
                action_type_new = 0
            elif step_data["action_type_text"] == 'scroll up':
                action_type_new = 1
            elif step_data["action_type_text"] == 'scroll left':
                action_type_new = 8
            elif step_data["action_type_text"] == 'scroll right':
                action_type_new = 9
            action = "{{\"action_type\": {}}}".format(action_type_new)
    elif action_type == 3:
        typed_text = step_data["type_text"]
        action_type_new = action_type
        action = "{{\"action_type\": {}, \"typed_text\": \"{}\"}}".format(action_type_new, typed_text)
    else:
        action_type_new = action_type
        action = "{{\"action_type\": {}}}".format(action_type_new)

    return action
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
import time
from datetime import datetime

aitw_imgs_dir = "/data_sdc/data/ldq/Dataset/AITW/images"
aitw_test = json.load(open('/data_sdc/data/ldq/Dataset/AITW/aitw_data_test.json', 'r'))
# aitw_test = json.load(open('/home/wentao/GUIAgent-Data/aitw_seeclick/aitw_data_train.json', 'r'))

prompt_origin = "Please generate the next move according to the instruction, previous actions, previous ui screenshot and current ui screenshot. Instruction: {}.\n"
score_average = 0
all_save_results = []
all_eval_results = []

# Profiler total FLOPs list: 每个 step 的完整 generate 总FLOPs
profiler_total_flops_list = []
# Profiler prefill FLOPs list: 每个被采样 step 的 prefill-only FLOPs
profiler_prefill_flops_list = []
# Prefill time list (ms): 每个 step 记录一次，最终取全数据集平均
prefill_time_list = []
# Lightweight debug: print FLOPs/prefill stats only once (first successful step)
_printed_flops_prefill_debug_once = False

processed_episodes = 0
global_valid_step_idx = 0

# ===== DivPrune 保留比例统计（按历史长度分组） =====
# 对每个 step：统计历史截图段(hist_segments)内 keep_mask=True 的token数，
# 并按“最近/第二近/第三近/最远”分组计算比例（分母为该step历史段保留token总数）。
# 再按历史长度k=1/2/3/4分别在所有steps上做平均。
dp_stats = {
    1: {"sum": np.zeros(1, dtype=np.float64), "cnt": 0},
    2: {"sum": np.zeros(2, dtype=np.float64), "cnt": 0},
    3: {"sum": np.zeros(3, dtype=np.float64), "cnt": 0},
    4: {"sum": np.zeros(4, dtype=np.float64), "cnt": 0},
}

# ===== 记录程序开始时间 =====
start_time = time.time()
start_datetime = datetime.now()
print("=" * 80)
print(f"程序开始时间: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
for task, episodes in aitw_test.items():
    print("Task: " + task)

    corr_action = 0
    corr_type = 0
    num_text = 0
    corr_text = 0
    num_scroll = 0
    corr_scroll = 0
    num_click = 0
    corr_click = 0
    num_both_click = 0
    corr_both_click = 0
    num_wrong_format = 0
    num = 0
    
    # episodes = episodes[0:20]

    print("sample num:", len(episodes))

    for j, episode in tqdm(enumerate(episodes)):

        previous_actions = []
        previous_imgs = []
        episode_imgs = set()

        for step in episode:
            step_json = {'task': task, 'episode': step['ep_id'], 'correct': 'no'}

            img_filename = step["img_filename"] + '.png'
            img_path = os.path.join(aitw_imgs_dir, img_filename)
            if not os.path.exists(img_path):
                print('image not found')
                continue
            if len(img_filename) > 100:     # several image with long filename lead to error in linux, just jump it
                continue
            image = Image.open(img_path)

            goal = step["goal"]

            prompt = prompt_origin.format(goal)

            cur_all_imgs = []
            cur_step_idx = len(previous_imgs[-args.his_num:])
            for i, action in enumerate(previous_actions[-args.his_num:]):
                prompt += 'Image_' + str(i) + ":<image>\n"
                prompt += 'Step_' + str(i) + ': ' + action + " .\n"
                cur_all_imgs.append(previous_imgs[-args.his_num:][i])

            prompt += 'Image_' + str(cur_step_idx) + ":<image>\n"
            cur_all_imgs.append(img_path)
            action_step = action2step(step)

            # 收集本episode所有截图路径（历史+当前）
            for _im in cur_all_imgs:
                episode_imgs.add(_im)


            previous_actions.append(action_step)
            previous_imgs.append(img_path)

            # print(cur_all_imgs)
            # print(repr(prompt))

            action_ref = action_matching.action_2_format(step)

            global_valid_step_idx += 1
            do_profile_this_step = (global_valid_step_idx % max(1, int(args.profile_every_n_steps)) == 0)
            should_profile_total = bool(args.profile_total_flops) and do_profile_this_step
            should_profile_prefill = bool(args.profile_prefill_flops) and do_profile_this_step

            if should_profile_total:
                _activities = [ProfilerActivity.CPU]
                if torch.cuda.is_available():
                    _activities.append(ProfilerActivity.CUDA)
                with profile(
                    activities=_activities,
                    with_flops=True,
                    record_shapes=False,
                    profile_memory=False,
                ) as prof:
                    response = generate_grounding(cur_all_imgs, prompt)
                step_total_prof_flops = extract_total_flops_from_prof(prof)
                profiler_total_flops_list.append(int(step_total_prof_flops))
            else:
                response = generate_grounding(cur_all_imgs, prompt)
                step_total_prof_flops = None

            if should_profile_prefill:
                try:
                    step_prefill_prof_flops = int(profile_prefill_flops_only(cur_all_imgs, prompt))
                except Exception as _prefill_prof_e:
                    step_prefill_prof_flops = 0
                    print(f"[PROFILER-FLOPs][STEP] prefill profile failed: {_prefill_prof_e}")
                profiler_prefill_flops_list.append(int(step_prefill_prof_flops))
            else:
                step_prefill_prof_flops = None

            if should_profile_total or should_profile_prefill:
                print(
                    f"[PROFILER-FLOPs][STEP] global_step={global_valid_step_idx} task={task} ep={step.get('ep_id','?')} "
                    f"total_flops={int(step_total_prof_flops) if step_total_prof_flops is not None else 'NA'} "
                    f"prefill_flops={int(step_prefill_prof_flops) if step_prefill_prof_flops is not None else 'NA'}"
                )

            prefill_ms = getattr(model.model, "_last_prefill_time_ms", None)
            if prefill_ms is not None:
                prefill_time_list.append(float(prefill_ms))

            n_before_dbg = getattr(model.model, "_n_before_prefill", None)
            n_after_dbg = getattr(model.model, "_n_after_hook", None)

            if not _printed_flops_prefill_debug_once:
                print(
                    f"[PREFILL-DEBUG][first-step] drop_k={args.drop_k}, "
                    f"n_before={n_before_dbg}, n_after={n_after_dbg}, "
                    f"prefill_ms={float(prefill_ms) if prefill_ms is not None else None}"
                )
                _printed_flops_prefill_debug_once = True

            # FastV 最小化调试：检查分段比例是否真的影响 keep_mask
            if args.fastv_debug and args.fastv_enable:
                try:
                    keep_mask_dbg = getattr(model.model, '_fastv_keep_mask', None)
                    oas_dbg = getattr(model.model, '_last_oas_index', None)
                    if keep_mask_dbg is not None and oas_dbg is not None and isinstance(oas_dbg, list) and len(oas_dbg) > 0:
                        km = keep_mask_dbg[0].detach().to('cpu').to(torch.bool)
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
                            print(f"[FASTV-DEBUG][step ep={step.get('ep_id','?')} idx={len(previous_actions)-1}] n_before={n_before_dbg} n_after={n_after_dbg} seg_keep(rank0->older)={seg_keep} ratios_eff={getattr(model.model, 'fastv_segment_keep_ratios', None)}")
                except Exception as e:
                    print("[FASTV-DEBUG] per-step debug failed:", repr(e))

            # DART 每步调试：检查 DART keep-mask 是否生效
            if args.dart_debug and args.dart_enable:
                try:
                    keep_mask_dbg = getattr(model.model, '_dart_keep_mask', None)
                    oas_dbg = getattr(model.model, '_last_oas_index', None)
                    print(f"[DART-DEBUG][STEP] ep={step.get('ep_id','?')} idx={len(previous_actions)-1} n_before={n_before_dbg} n_after={n_after_dbg}")
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

            # 重置 hook 避免干扰下一个 step
            model.model._n_after_hook = None
            model.model._n_before_prefill = None

            # ===== DivPrune：可视化最近4张历史截图（被剪掉的patch标"1"） =====
            try:
                if args.DivPrune_enable and len(cur_all_imgs) > 1 and getattr(model.model, 'enable_DivPrune_pruning', False):
                    keep_mask = getattr(model.model, '_DivPrune_keep_mask', None)
                    oas_index = getattr(model.model, '_last_oas_index', None)
                    if keep_mask is not None and oas_index is not None and isinstance(oas_index, list) and len(oas_index) > 0:
                        # batch size assumed 1
                        km_1d = keep_mask[0].detach()
                        idx_list_1 = oas_index[0]
                        # ep_id 用 step['ep_id']，step_id 用 episode 内序号（从0开始）
                        ep_id = step.get('ep_id', 'unknown')
                        step_id = int(len(previous_actions) - 1)
                        visualize_DivPrune_keep_mask(
                            keep_mask_1d=km_1d,
                            oas_index_1=idx_list_1,
                            hist_img_paths_old_to_new=cur_all_imgs[:-1],
                            ep_id=str(ep_id),
                            step_id=step_id,
                            out_root="/home/ldq/SimpAgent-main/output/greedy",
                            max_hist=4,
                            patch_size=14,
                            spatial_merge_size=2,
                            min_pixel=min_pixels,
                            max_pixel=max_pixels,
                        )
            except Exception as _vis_e:
                # 可视化不影响主流程
                print(f"[VIS] failed: ep={step.get('ep_id','?')}, step={len(previous_actions)-1}, err={_vis_e}")

            # ===== Sobel：可视化最近4张历史截图（被剪掉的patch标"1"） =====
            try:
                if args.sobel_enable and args.sobel_vis_enable and len(cur_all_imgs) > 1 and getattr(model.model, 'sobel_enable', False):
                    keep_mask = getattr(model.model, '_sobel_keep_mask', None)
                    oas_index = getattr(model.model, '_last_oas_index', None)
                    if keep_mask is not None and oas_index is not None and isinstance(oas_index, list) and len(oas_index) > 0:
                        km_1d = keep_mask[0].detach()
                        idx_list_1 = oas_index[0]
                        ep_id = step.get('ep_id', 'unknown')
                        step_id = int(len(previous_actions) - 1)
                        visualize_sobel_keep_mask(
                            keep_mask_1d=km_1d,
                            oas_index_1=idx_list_1,
                            hist_img_paths_old_to_new=cur_all_imgs[:-1],
                            ep_id=str(ep_id),
                            step_id=step_id,
                            out_root="/data_sdc/data/SimpAgent-main/output/visual_result/Sobel",
                            max_hist=4,
                            patch_size=14,
                            spatial_merge_size=2,
                            min_pixel=min_pixels,
                            max_pixel=max_pixels,
                        )
            except Exception as _vis_e:
                pass

            # ===== FastV：可视化最近4张历史截图（被剪掉的patch标"1"） =====
            try:
                if args.fastv_enable and args.fastv_vis_enable and len(cur_all_imgs) > 1 and getattr(model.model, 'enable_fastv_pruning', False):
                    keep_mask = getattr(model.model, '_fastv_keep_mask', None)
                    oas_index = getattr(model.model, '_last_oas_index', None)
                    if keep_mask is not None and oas_index is not None and isinstance(oas_index, list) and len(oas_index) > 0:
                        km_1d = keep_mask[0].detach()
                        idx_list_1 = oas_index[0]
                        ep_id = step.get('ep_id', 'unknown')
                        step_id = int(len(previous_actions) - 1)
                        visualize_fastv_keep_mask(
                            keep_mask_1d=km_1d,
                            oas_index_1=idx_list_1,
                            hist_img_paths_old_to_new=cur_all_imgs[:-1],
                            ep_id=str(ep_id),
                            step_id=step_id,
                            out_root="/data_sdc/data/SimpAgent-main/output/visual_result/FastV",
                            max_hist=4,
                            patch_size=14,
                            spatial_merge_size=2,
                            min_pixel=min_pixels,
                            max_pixel=max_pixels,
                        )
            except Exception as _vis_e:
                print(f"[FastV-VIS] failed: ep={step.get('ep_id','?')}, step={len(previous_actions)-1}, err={_vis_e}")

            # ===== DivPrune：统计本step各历史截图段保留token比例（按最近->最远） =====
            try:
                if args.DivPrune_enable and len(cur_all_imgs) > 1 and getattr(model.model, 'enable_DivPrune_pruning', False):
                    keep_mask = getattr(model.model, '_DivPrune_keep_mask', None)
                    oas_index = getattr(model.model, '_last_oas_index', None)
                    if keep_mask is not None and oas_index is not None and isinstance(oas_index, list) and len(oas_index) > 0:
                        # batch size 假设为1
                        km = keep_mask[0].detach().to('cpu')
                        idx_list = oas_index[0]
                        if isinstance(idx_list, list) and len(idx_list) > 1:
                            # hist_segments 严格对应历史截图视觉token段（不含动作文本/prompt/特殊token）
                            hist_segments = []
                            for seg in idx_list[:-1]:
                                if isinstance(seg, list) and len(seg) >= 2:
                                    s, e = int(seg[0]), int(seg[1])
                                    if e > s:
                                        hist_segments.append((s, e))
                            k = len(hist_segments)
                            if 1 <= k <= 4:
                                kept_counts = []
                                for (s, e) in hist_segments:
                                    kept_counts.append(int(km[s:e].sum().item()))
                                total_kept = int(sum(kept_counts))
                                if total_kept > 0:
                                    # 转为“最近->最远”顺序
                                    kept_counts_recent_to_far = list(reversed(kept_counts))
                                    ratios = np.array(kept_counts_recent_to_far, dtype=np.float64) / float(total_kept)
                                    dp_stats[k]["sum"] += ratios
                                    dp_stats[k]["cnt"] += 1
            except Exception:
                # 统计不应影响评测主流程
                pass

            # print(prompt)
            # print(cur_all_imgs)
            # print(response)
            raw_response = response
            '''
            try:
                print("\n[DEBUG] 原始模型输出（未处理）:")
                print(raw_response)
                print("[DEBUG] 原始输出repr:", repr(raw_response))
            except Exception:
                pass
            '''
            response = process_string(raw_response)
            '''
            try:
                print("[DEBUG] 归一化后的输出（已process_string）:")
                print(response)
            except Exception:
                pass
            '''
            num += 1
            
            try:
                action_pred = action_matching.pred_2_format(ast.literal_eval(response))
                annot_position = np.array(
                    [step["annot_position"][i:i + 4] for i in range(0, len(step["annot_position"]), 4)])
                check_match = action_matching.check_actions_match(action_pred["touch_point"], action_pred["lift_point"],
                                                                  action_pred["action_type"], action_ref["touch_point"],
                                                                  action_ref["lift_point"], action_ref["action_type"],
                                                                  annot_position)
                # step accuracy
                if check_match == True:
                    corr_action += 1
                    match_label = 1
                    step_json['correct'] = 'yes'
                    # print("Step: " + str(j) + " right")
                else:
                    match_label = 0
                    # print("Step: " + str(j) + " wrong")

                # type accuracy
                if action_pred["action_type"] == action_ref["action_type"]:
                    corr_type += 1

                # text accuracy
                if action_ref["action_type"] == 3:
                    num_text += 1
                    if (action_pred["typed_text"] == action_ref["typed_text"]) or (
                            action_pred["typed_text"] in action_ref["typed_text"]) or (
                            action_ref["typed_text"] in action_pred["typed_text"]):
                        corr_text += 1

                if action_ref["action_type"] == 4:
                    # click accuracy
                    if action_matching.is_tap_action(action_ref["touch_point"], action_ref["lift_point"]):
                        num_click += 1
                        if match_label:
                            corr_click += 1
                    # scroll accuracy
                    else:
                        num_scroll += 1
                        if match_label:
                            corr_scroll += 1
                    if (action_pred["action_type"] == 4) and action_matching.is_tap_action(action_ref["touch_point"],
                                                                                           action_ref[
                                                                                               "lift_point"]) and action_matching.is_tap_action(
                            action_pred["touch_point"], action_pred["lift_point"]):
                        num_both_click += 1
                        if match_label:
                            corr_both_click += 1

            except:
                num_wrong_format += 1
                # print("Step: " + str(j) + " wrong format")
            all_save_results.append(step_json)

    task_action_acc = corr_action / num if num else 0.0
    task_type_acc = corr_type / num if num else 0.0
    task_text_acc = corr_text / num_text if num_text else 0.0
    task_click_acc = corr_click / num_click if num_click else 0.0
    task_scroll_acc = corr_scroll / num_scroll if num_scroll else 0.0
    task_both_click_acc = corr_both_click / num_both_click if num_both_click else 0.0

    score_average += task_action_acc

    print("Action Acc: " + str(task_action_acc))
    print("Type Acc: " + str(task_type_acc))
    print("Text Acc: " + str(task_text_acc))
    print("Click Acc: " + str(task_click_acc))
    print("Scroll Acc: " + str(task_scroll_acc))
    print("Both Click Acc: " + str(task_both_click_acc))
    print("Num Both Click: " + str(num_both_click))
    print("Num wrong format: " + str(num_wrong_format))

    all_eval_results.append({
        "task": task,
        "sample_num": len(episodes),
        "num": num,
        "num_wrong_format": num_wrong_format,
        "action_acc": task_action_acc,
        "type_acc": task_type_acc,
        "text_acc": task_text_acc,
        "click_acc": task_click_acc,
        "scroll_acc": task_scroll_acc,
        "both_click_acc": task_both_click_acc,
        "num_both_click": num_both_click,
        "num_click": num_click,
        "num_scroll": num_scroll,
        "num_text": num_text
    })

average_score = score_average / 5
print("Average score: " + str(average_score))
print("Average FLOPs (profiler total): " + str(float(np.mean(profiler_total_flops_list)) if profiler_total_flops_list else 0.0))
print("Average FLOPs (profiler prefill-only): " + str(float(np.mean(profiler_prefill_flops_list)) if profiler_prefill_flops_list else 0.0))
print("Profiled total steps count: " + str(int(len(profiler_total_flops_list))) + f" (every {int(args.profile_every_n_steps)} steps)")
print("Profiled prefill steps count: " + str(int(len(profiler_prefill_flops_list))) + f" (every {int(args.profile_every_n_steps)} steps)")
print("Average prefill time per step (ms): " + str(float(np.mean(prefill_time_list)) if prefill_time_list else 0.0))

# ===== 打印 DivPrune：按历史长度分组的保留token分配比例（最近->最远） =====
try:
    print("\n" + "=" * 80)
    print("DivPrune 历史截图保留token分配比例（最近->最远），按历史长度k分组统计：")
    for k in [1, 2, 3, 4]:
        cnt = int(dp_stats[k]["cnt"])
        if cnt <= 0:
            print(f"k={k}: no samples")
            continue
        avg = dp_stats[k]["sum"] / float(cnt)
        avg_str = ", ".join([f"{v:.4f}" for v in avg.tolist()])
        print(f"k={k}, steps={cnt}: [{avg_str}]")
    print("=" * 80 + "\n")
except Exception:
    pass

final_log = {
    "args": vars(args),
    "task_results": all_eval_results,
    "average_score": average_score,
    "average_flops_profiler_total": float(np.mean(profiler_total_flops_list)) if profiler_total_flops_list else 0.0,
    "average_flops_profiler_prefill": float(np.mean(profiler_prefill_flops_list)) if profiler_prefill_flops_list else 0.0,
    "profiled_steps_count": int(len(profiler_total_flops_list)),
    "profiled_prefill_steps_count": int(len(profiler_prefill_flops_list)),
    "profile_total_flops_enabled": bool(args.profile_total_flops),
    "profile_prefill_flops_enabled": bool(args.profile_prefill_flops),
    "profile_every_n_steps": int(args.profile_every_n_steps),
    "avg_prefill_ms_per_step": float(np.mean(prefill_time_list)) if prefill_time_list else 0.0,
    "prefill_steps_counted": int(len(prefill_time_list)),
    "dp_stats": {
        str(k): {
            "cnt": int(dp_stats[k]["cnt"]),
            "avg": (dp_stats[k]["sum"] / float(dp_stats[k]["cnt"])).tolist() if int(dp_stats[k]["cnt"]) > 0 else None,
        }
        for k in [1, 2, 3, 4]
    },
    "steps": all_save_results,
}
write_json(final_log, args.save_path)

# ===== 记录程序结束时间并计算总运行时间 =====
end_time = time.time()
end_datetime = datetime.now()
total_time = end_time - start_time

print("\n" + "=" * 80)
print(f"程序结束时间: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"总运行时间: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
print("=" * 80)

# 将时间信息保存到JSON文件
time_info = {
    "start_time": start_datetime.strftime('%Y-%m-%d %H:%M:%S'),
    "end_time": end_datetime.strftime('%Y-%m-%d %H:%M:%S'),
    "total_seconds": round(total_time, 2),
    "total_minutes": round(total_time / 60, 2),
    "total_hours": round(total_time / 3600, 2),
    "avg_prefill_ms_per_step": float(np.mean(prefill_time_list)) if prefill_time_list else 0.0,
    "prefill_steps_counted": int(len(prefill_time_list)),
}

# 保存时间信息到单独的文件
time_log_path = args.save_path.replace('.json', '_time_log.json')
write_json(time_info, time_log_path)
print(f"\n时间日志已保存到: {time_log_path}")