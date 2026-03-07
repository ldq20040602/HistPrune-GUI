import json
import os
import re
import torch
from torch.profiler import profile, ProfilerActivity
import numpy as np
from tqdm import tqdm
from transformers import AutoProcessor, AutoConfig
from src.training.my_qwen_vl_utils import process_vision_info_with_resize
import importlib
import argparse


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
import time
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from typing import List

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

# Use the same prompt format as in GUIOdyssey_process.py
PROMPT_ORIGIN = "Please generate the next move according to the instruction, previous actions, previous ui screenshot and current ui screenshot. Instruction: {}.\n"

def write_json(data, file_path):
    dir_name = os.path.dirname(os.path.abspath(file_path))
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
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


# Optional Sobel prior for foreground/background split
try:
    from sobel_segmentation import compute_sobel_uigraph
    _HAS_SOBEL = True
except Exception as e:
    _HAS_SOBEL = False
    print("[IMPORT sobel_segmentation FAILED]", repr(e))

# 创建解析器
parser = argparse.ArgumentParser(description='Specify paths for saving and loading models.')

# 添加参数
parser.add_argument('--save_path', type=str, default="/data_sdc/data/SimpAgent-main/eval_results/GUIOdyssey/Baseline4AO_withhistory",
                    help='The path where the result will be saved')
parser.add_argument('--model_path', type=str, default="/data_sdc/data/SimpAgent-main/save/Qwen2/GUIOdyssey/Baseline4AO_lora_225",
                    help='The path where the model is loaded from')
parser.add_argument('--his_num', type=int, default=4,
                    help='History length for evaluation')
parser.add_argument('--drop_k', type=int, default=0,
                    help='Drop k for model')
parser.add_argument('--test_file', type=str, 
                    default="/data_sdc/data/SimpAgent-main/data/guiodyssey_standard_test_high_llavaformat.json",
                    help='Path to test dataset file')
parser.add_argument('--alpha', type=int, default=1,
                    help='Alpha parameter for model')

# --- 剪枝相关参数 (与 AITW_eval.py 对齐) ---
parser.add_argument('--fastv_enable', action='store_true', default=False,
                    help='Enable FastV token-level pruning at drop_k (training-free).')
parser.add_argument('--fastv_m', type=float, default=0.5,
                    help='FastV prune ratio m in [0,1], fraction of historical screenshot tokens to remove.')
parser.add_argument('--fastv_segment_ratios', type=str, default="0.5,0.5,0.5,0.5",
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
                    help='Per-patch edge pixel ratio threshold in [0,1]; >= ratio means an edge patch (won\'t merge).')

parser.add_argument('--sparse_greedy_enable', action='store_true', default=False,
                    help='Enable sparse greedy submodular pruning (training-free).')
parser.add_argument('--sparse_greedy_lambda', type=float, default=0.8,
                    help='Time decay factor (0~1).')
parser.add_argument('--sparse_greedy_step_k', type=int, default=1,
                    help='Number of tokens to select in each parallel step of sparse greedy.')
parser.add_argument('--sparse_greedy_ratios', type=str, default="",
                    help='Comma-separated keep ratios for Sparse Greedy per history image.')

parser.add_argument('--sobel_enable', action='store_true', default=False,
                    help='Enable Sobel edge-only pruning at drop_k (training-free).')
parser.add_argument('--sobel_vis_enable', action='store_true', default=False,
                    help='Enable visualization of Sobel pruning results.')

parser.add_argument('--random_enable', action='store_true', default=False,
                    help='Enable Random token-level pruning on historical screenshots.')
parser.add_argument('--random_ratio', type=str, default="0.5,0.5,0.5,0.5",
                    help='Comma-separated prune ratios for Random pruning per history image.')

parser.add_argument('--sparsevlm_enable', action='store_true', default=False,
                    help='Enable SparseVLM pruning (Stage1 raters + Stage2 global visual pruning).')
parser.add_argument('--sparsevlm_keep_ratio', type=float, default=0.4,
                    help='SparseVLM keep ratio for historical visual tokens.')

parser.add_argument('--fastv_vis_enable', action='store_true', default=False,
                    help='Enable visualization of FastV pruning results.')
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

# 处理 save_path
if args.save_path.endswith('.json'):
    pass 
else:
    os.makedirs(args.save_path, exist_ok=True)
    args.save_path = os.path.join(args.save_path, args.model_path.split('/')[-1] + "_drop_" + str(args.drop_k) + '.json')

# GUI Odyssey data directories
DATA_DIR = "/data_sdc/data/ldq/Dataset/GUI-Odyssey"
pic_base = "/data_sdc/data/ldq/Dataset/GUI-Odyssey"

# Load model
ModelClass = load_model_class(args.model_path)
model = ModelClass.from_pretrained(
    args.model_path,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2",
)

min_pixels = 200704
max_pixels = 1003520
processor = AutoProcessor.from_pretrained("/home/hibug/.cache/huggingface/hub/models--Qwen--Qwen2-VL-2B-Instruct/snapshots/895c3a49bc3fa70a340399125c650a463535e71c")

# Profiler total FLOPs list: 每个 step 的完整 generate 总FLOPs
profiler_total_flops_list = []

def safe_setattr(obj, attr, value):
    if hasattr(obj, attr):
        setattr(obj, attr, value)

# 配置模型剪枝参数
model.model.drop_k = args.drop_k
model.drop_k = args.drop_k
model.alpha = args.alpha

if args.fastv_enable:
    safe_setattr(model.model, "enable_fastv_pruning", True)
safe_setattr(model.model, "fastv_prune_ratio_m", float(args.fastv_m))

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

if args.sobel_enable:
    safe_setattr(model.model, "sobel_enable", True)

try:
    ratios = [float(x) for x in str(args.fastv_segment_ratios).split(',') if str(x).strip() != '']
    ratios = [0.0 if r < 0.0 else (1.0 if r > 1.0 else r) for r in ratios]
    safe_setattr(model.model, "fastv_segment_ratios", ratios)
except Exception:
    pass

if args.sparse_greedy_enable:
    safe_setattr(model.model, "enable_sparse_greedy_pruning", True)

if args.random_enable:
    safe_setattr(model.model, "enable_random_pruning", True)
try:
    rratios = [float(x) for x in str(args.random_ratio).split(',') if str(x).strip() != '']
    rratios = [0.0 if r < 0.0 else (1.0 if r > 1.0 else r) for r in rratios]
    safe_setattr(model.model, "random_prune_ratios", rratios[:4])
except Exception:
    pass
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

safe_setattr(model.model, "sparse_greedy_lambda", float(args.sparse_greedy_lambda))
safe_setattr(model.model, "sparse_greedy_step_k", int(args.sparse_greedy_step_k))
try:
    sgratios = [float(x) for x in str(args.sparse_greedy_ratios).split(',') if str(x).strip() != '']
    sgratios = [0.0 if r < 0.0 else (1.0 if r > 1.0 else r) for r in sgratios]
    safe_setattr(model.model, "sparse_greedy_ratios", sgratios)
except Exception:
    pass

# Sobel cache for historical screenshots
sobel_cache = {}

def get_image_info(image_path, min_pixel=200704, max_pixel=1003520):
    """Process image for model input"""
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
    return image_input[0]

# --- 可视化函数 (从 AITW_eval.py 迁移) ---
def visualize_sparse_greedy_keep_mask(
    *,
    keep_mask_1d: torch.Tensor,
    oas_index_1: list,
    hist_img_paths_old_to_new: List[str],
    ep_id: str,
    step_id: int,
    out_root: str = "/data_sdc/data/SimpAgent-main/output/visual_result/greedy",
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
                draw.text((cx, cy), "1", fill="red", font=font, anchor="mm")
            except Exception:
                draw.text((cx, cy), "1", fill="red", font=font)

        out_path = os.path.join(out_dir, f"history_{rank}_recent.png")
        try:
            pil.save(out_path)
        except Exception:
            continue

def visualize_fastv_keep_mask(
    *,
    keep_mask_1d: torch.Tensor,
    oas_index_1: list,
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
    visualize_sparse_greedy_keep_mask(
        keep_mask_1d=keep_mask_1d,
        oas_index_1=oas_index_1,
        hist_img_paths_old_to_new=hist_img_paths_old_to_new,
        ep_id=ep_id,
        step_id=step_id,
        out_root=out_root,
        max_hist=max_hist,
        patch_size=patch_size,
        spatial_merge_size=spatial_merge_size,
        min_pixel=min_pixel,
        max_pixel=max_pixel
    )

def visualize_sobel_keep_mask(
    *,
    keep_mask_1d: torch.Tensor,
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
    visualize_sparse_greedy_keep_mask(
        keep_mask_1d=keep_mask_1d,
        oas_index_1=oas_index_1,
        hist_img_paths_old_to_new=hist_img_paths_old_to_new,
        ep_id=ep_id,
        step_id=step_id,
        out_root=out_root,
        max_hist=max_hist,
        patch_size=patch_size,
        spatial_merge_size=spatial_merge_size,
        min_pixel=min_pixel,
        max_pixel=max_pixel
    )


def generate_grounding(image_paths, query):
    """Generate action prediction from model"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": query},
            ],
        }
    ]

    images = []
    for image_file in image_paths:
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
    
    # --- FastV 文本 Query 区间硬编码 (同步 AITW_eval.py) ---
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

        model.model._prompt_query_ranges = [{
            'p1': (int(general_instr_start), int(general_instr_end)),
            'p2': (int(goal_start), int(goal_end)),
        }]

        if args.pdrop_enable:
            model.model._pdrop_query_index = int(max(general_instr_start, goal_end - 1))
    except Exception as e:
        print("[DEBUG] Compute FastV query boundary failed:", e)

    # --- Sobel/Sparse Greedy prior (同步 AITW_eval.py) ---
    if (args.sparse_greedy_enable or args.sobel_enable) and len(image_paths) > 1:
        hist_imgs = image_paths[:-1]
        info_list = []
        if _HAS_SOBEL:
            for p in hist_imgs:
                if p not in sobel_cache:
                    try:
                        _pil = get_image_info(p, min_pixel=min_pixels, max_pixel=max_pixels)
                        sobel_info = compute_sobel_uigraph(
                            p,
                            preprocessed_pil=_pil,
                            edge_thr=int(args.sobel_edge_thr),
                            ratio_thr=float(args.sobel_ratio_thr),
                            min_pixels=min_pixels,
                            max_pixels=max_pixels,
                        )
                        sobel_cache[p] = {'is_edge': sobel_info['is_edge']}
                    except Exception as ex:
                        print(f"Sobel compute failed for {p}: {ex}")
                        sobel_cache[p] = None
                info_list.append(sobel_cache.get(p, None))
        
        if all(x is not None and isinstance(x, dict) and 'is_edge' in x for x in info_list):
            if args.sparse_greedy_enable:
                model.model.enable_sparse_greedy_pruning = True
                model.model._sparse_greedy_info = [info_list]
            if args.sobel_enable:
                model.model.sobel_enable = True
                model.model._sobel_info = [info_list]
        
        model.model.enable_uigraph_pruning = False
        model.model.enable_fastv_pruning = False

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )[0][0:-10]

    return output_text

def main():
    """Main evaluation function for GUI Odyssey dataset"""
    # 统计信息初始化
    sg_stats = {
        1: {"sum": np.zeros(1, dtype=np.float64), "cnt": 0},
        2: {"sum": np.zeros(2, dtype=np.float64), "cnt": 0},
        3: {"sum": np.zeros(3, dtype=np.float64), "cnt": 0},
        4: {"sum": np.zeros(4, dtype=np.float64), "cnt": 0},
    }
    start_time = time.time()
    start_datetime = datetime.now()

    print(f"Starting GUI Odyssey evaluation...")
    print(f"Model path: {args.model_path}")
    print(f"Test file: {args.test_file}")
    print(f"History length: {args.his_num}")
    
    # Load test dataset
    if not os.path.exists(args.test_file):
        raise FileNotFoundError(f"Test file not found: {args.test_file}")
    
    with open(args.test_file, 'r') as f:
        test_data = json.load(f)
    
    print(f"Loaded {len(test_data)} test samples")
    
    outputs = []
    img_not_found = 0
    episode_step_counts = {}
    prefill_time_list = []  # ms, one value per evaluated step
    global_valid_step_idx = 0
    flops_counted_steps = 0
    flops_early_summary_printed = False
    
    # Process each test sample
    for sample_idx, sample in enumerate(tqdm(test_data, desc="Evaluating")):
        try:
            conversations = sample.get("conversations", [])
            image_paths = sample.get("image", [])
            
            if len(conversations) < 2 or len(image_paths) < 1:
                continue
            
            user_content = conversations[0].get("value", "")
            gt_action = conversations[1].get("value", "")

            resolved_image_paths = [os.path.join(pic_base, p) for p in image_paths]

            # Check images
            missing = False
            for p in resolved_image_paths:
                if not os.path.exists(p):
                    missing = True
                    img_not_found += 1
                    break
            if missing: continue

            # Get model prediction
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
                    pred_action = generate_grounding(resolved_image_paths, user_content)
                step_total_prof_flops = extract_total_flops_from_prof(prof)
                profiler_total_flops_list.append(int(step_total_prof_flops))
                print(f"[PROFILER-FLOPs][STEP] sample={sample_idx} id={sample.get('id','?')} global_step={global_valid_step_idx} flops_step={flops_counted_steps + 1}/{max(1, int(args.flops_max_steps))} total_flops={int(step_total_prof_flops)}")
            else:
                pred_action = generate_grounding(resolved_image_paths, user_content)

            prefill_ms = getattr(model.model, "_last_prefill_time_ms", None)
            if prefill_ms is not None:
                prefill_time_list.append(float(prefill_ms))

            # ===== FLOPs 统计：理论 Prefill FLOPs（仅前N步） =====
            if should_count_flops_this_step:
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

            # DART 每步调试：检查 DART keep-mask 是否生效
            if args.dart_debug and args.dart_enable:
                try:
                    keep_mask_dbg = getattr(model.model, '_dart_keep_mask', None)
                    oas_dbg = getattr(model.model, '_last_oas_index', None)
                    print(f"[DART-DEBUG][STEP] sample={sample_idx} id={sample.get('id','?')}")
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

            if args.pdrop_debug and args.pdrop_enable:
                try:
                    keep_mask_dbg = getattr(model.model, '_pdrop_keep_mask', None)
                    oas_dbg = getattr(model.model, '_last_oas_index', None)
                    print(f"[PDROP-DEBUG][STEP] sample={sample_idx} id={sample.get('id','?')}")
                    print("[PDROP-DEBUG][STEP] pdrop_keep_ratios(effective)=", getattr(model.model, 'pdrop_keep_ratios', None))
                    print(f"[PDROP-DEBUG][STEP] has_oas={oas_dbg is not None} pdrop_keep_mask_exists={keep_mask_dbg is not None}")
                    if keep_mask_dbg is not None and oas_dbg is not None and isinstance(oas_dbg, list) and len(oas_dbg) > 0:
                        print("[PDROP-DEBUG][STEP] pdrop_keep_mask_shape=", tuple(keep_mask_dbg.shape))
                except Exception as e:
                    print("[PDROP-DEBUG] per-step debug failed:", repr(e))
            
            # --- 可视化与统计 (同步 AITW_eval.py) ---
            ep_id = sample.get('id', str(sample_idx))
            
            # 维护每个 episode 的 step 序号，避免可视化路径覆盖
            if ep_id not in episode_step_counts:
                episode_step_counts[ep_id] = 0
            step_id = episode_step_counts[ep_id]
            episode_step_counts[ep_id] += 1

            # Sparse Greedy Vis
            if args.sparse_greedy_enable and len(resolved_image_paths) > 1 and getattr(model.model, 'enable_sparse_greedy_pruning', False):
                try:
                    keep_mask = getattr(model.model, '_sparse_greedy_keep_mask', None)
                    oas_index = getattr(model.model, '_last_oas_index', None)
                    if keep_mask is not None and oas_index is not None:
                        km_1d = keep_mask[0].detach()
                        idx_list_1 = oas_index[0]
                        visualize_sparse_greedy_keep_mask(
                            keep_mask_1d=km_1d, oas_index_1=idx_list_1,
                            hist_img_paths_old_to_new=resolved_image_paths[:-1],
                            ep_id=str(ep_id), step_id=step_id,
                            min_pixel=min_pixels, max_pixel=max_pixels
                        )
                        # Stats
                        km = km_1d.to('cpu')
                        hist_segments = []
                        for seg in idx_list_1[:-1]:
                            if isinstance(seg, list) and len(seg) >= 2:
                                s, e = int(seg[0]), int(seg[1])
                                if e > s: hist_segments.append((s, e))
                        k_len = len(hist_segments)
                        if 1 <= k_len <= 4:
                            kept_counts = [int(km[s:e].sum().item()) for (s, e) in hist_segments]
                            total_kept = sum(kept_counts)
                            if total_kept > 0:
                                ratios = np.array(list(reversed(kept_counts)), dtype=np.float64) / float(total_kept)
                                sg_stats[k_len]["sum"] += ratios
                                sg_stats[k_len]["cnt"] += 1
                except Exception: pass

            # Sobel Vis
            if args.sobel_enable and args.sobel_vis_enable and len(resolved_image_paths) > 1:
                try:
                    keep_mask = getattr(model.model, '_sobel_keep_mask', None)
                    oas_index = getattr(model.model, '_last_oas_index', None)
                    if keep_mask is not None and oas_index is not None:
                        visualize_sobel_keep_mask(
                            keep_mask_1d=keep_mask[0].detach(), oas_index_1=oas_index[0],
                            hist_img_paths_old_to_new=resolved_image_paths[:-1],
                            ep_id=str(ep_id), step_id=step_id,
                            min_pixel=min_pixels, max_pixel=max_pixels
                        )
                except Exception: pass

            # FastV Vis
            if args.fastv_enable and args.fastv_vis_enable and len(resolved_image_paths) > 1:
                try:
                    keep_mask = getattr(model.model, '_fastv_keep_mask', None)
                    oas_index = getattr(model.model, '_last_oas_index', None)
                    if keep_mask is not None and oas_index is not None:
                        visualize_fastv_keep_mask(
                            keep_mask_1d=keep_mask[0].detach(), oas_index_1=oas_index[0],
                            hist_img_paths_old_to_new=resolved_image_paths[:-1],
                            ep_id=str(ep_id), step_id=step_id,
                            min_pixel=min_pixels, max_pixel=max_pixels
                        )
                except Exception: pass

            outputs.append({
                'sample_id': ep_id,
                'pred': pred_action,
                'gt': gt_action,
            })
            
        except Exception as e:
            print(f"Error processing sample {sample_idx}: {e}")
            continue
    
    # 打印统计与保存结果
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\nEvaluation completed! Processed: {len(outputs)}, Missing images: {img_not_found}")
    print(f"Average FLOPs (profiler total, first {int(args.flops_max_steps)} steps): {float(np.mean(profiler_total_flops_list)) if profiler_total_flops_list else 0.0}")
    print(f"FLOPs counted steps: {int(flops_counted_steps)}")
    print(f"Profiled steps count: {int(len(profiler_total_flops_list))} (every {int(args.profile_every_n_steps)} steps)")
    print(f"Average prefill time per step (ms): {float(np.mean(prefill_time_list)) if prefill_time_list else 0.0}")
    
    if args.sparse_greedy_enable:
        print("\nSparse Greedy Retention Ratios (Recent -> Far):")
        for k in [1, 2, 3, 4]:
            cnt = sg_stats[k]["cnt"]
            if cnt > 0:
                avg = sg_stats[k]["sum"] / cnt
                print(f"k={k}, samples={cnt}: {avg.tolist()}")

    final_log = {
        "args": vars(args),
        "results": outputs,
        "average_flops_profiler_total": float(np.mean(profiler_total_flops_list)) if profiler_total_flops_list else 0.0,
        "flops_max_steps": int(args.flops_max_steps),
        "flops_steps_counted": int(flops_counted_steps),
        "profiled_steps_count": int(len(profiler_total_flops_list)),
        "profile_total_flops_enabled": bool(args.profile_total_flops),
        "profile_every_n_steps": int(args.profile_every_n_steps),
        "time_info": {
            "start": start_datetime.strftime('%Y-%m-%d %H:%M:%S'),
            "total_seconds": total_time,
            "avg_prefill_ms_per_step": float(np.mean(prefill_time_list)) if prefill_time_list else 0.0,
            "prefill_steps_counted": int(len(prefill_time_list)),
        },
        "sg_stats": {str(k): {"cnt": int(sg_stats[k]["cnt"]), "avg": (sg_stats[k]["sum"]/sg_stats[k]["cnt"]).tolist() if sg_stats[k]["cnt"]>0 else None} for k in [1, 2, 3, 4]}
    }
    write_json(final_log, args.save_path)
    print(f"Results saved to: {args.save_path}")

if __name__ == "__main__":
    main()
