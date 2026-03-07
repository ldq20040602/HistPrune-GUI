import os
import argparse
import numpy as np
from tqdm import tqdm

from src.Qwen2.my_qwen_vl_utils import process_vision_info_with_resize
from src.Qwen2.my_qwen_vl_utils_single import process_vision_info
from sobel_segmentation import compute_sobel_uigraph


# 与现有评测口径保持一致（固定值）
MIN_PIXELS = 200704
MAX_PIXELS = 1003520


DATASET_DEFAULT_PATHS = {
    "aitw": "/data_sdc/data/ldq/Dataset/AITW/images",
    "odyssey": "/data_sdc/data/ldq/Dataset/Odyssey/images",
    "mind2web": "/data_sdc/data/ldq/Dataset/Mind2Web/ming2web_images",
    "androidcontrol": "/data_sdc/data/ldq/Dataset/AndroidControl/images",
}

RESIZE_DATASETS = {"aitw", "odyssey"}
NO_RESIZE_DATASETS = {"mind2web", "androidcontrol"}
ALL_DATASETS = RESIZE_DATASETS | NO_RESIZE_DATASETS


def get_images(root_dir):
    """递归获取目录下所有图片路径"""
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    image_paths = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_paths.append(os.path.join(root, file))
    return image_paths


def infer_dataset_from_path(data_root):
    path_l = data_root.lower().replace("\\", "/")
    hits = []

    if "mind2web" in path_l or "ming2web" in path_l:
        hits.append("mind2web")
    if "androidcontrol" in path_l:
        hits.append("androidcontrol")
    if "odyssey" in path_l:
        hits.append("odyssey")
    if "aitw" in path_l:
        hits.append("aitw")

    if len(hits) == 1:
        return hits[0]
    if len(hits) == 0:
        return None
    raise ValueError(
        f"Cannot uniquely infer dataset from path: {data_root}. Candidates={hits}. Please pass --dataset explicitly."
    )


def preprocess_image(img_path, dataset):
    """按数据集策略进行视觉预处理，返回 resized PIL 图像"""
    if dataset in RESIZE_DATASETS:
        messages = [{"role": "user", "content": [{"type": "image", "image": img_path}]}]
        image_inputs, _ = process_vision_info_with_resize(messages)
        if not image_inputs:
            raise RuntimeError(f"Failed to preprocess image (resize strategy): {img_path}")
        return image_inputs[0], {"strategy": "resize"}

    if dataset in NO_RESIZE_DATASETS:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img_path,
                        "min_pixels": MIN_PIXELS,
                        "max_pixels": MAX_PIXELS,
                    }
                ],
            }
        ]
        image_inputs, _ = process_vision_info(messages)
        if not image_inputs:
            raise RuntimeError(f"Failed to preprocess image (no-resize strategy): {img_path}")
        return image_inputs[0], {
            "strategy": "no_resize",
            "min_pixels": MIN_PIXELS,
            "max_pixels": MAX_PIXELS,
        }

    raise ValueError(f"Unsupported dataset: {dataset}")


def resolve_dataset_and_path(dataset, data_root):
    if dataset:
        dataset = dataset.lower()
        if dataset not in ALL_DATASETS:
            raise ValueError(f"Unsupported --dataset={dataset}. Supported: {sorted(ALL_DATASETS)}")

    if data_root and not dataset:
        inferred = infer_dataset_from_path(data_root)
        if not inferred:
            raise ValueError(
                f"Unable to infer dataset from --data-root={data_root}. "
                f"Please pass --dataset explicitly: {sorted(ALL_DATASETS)}"
            )
        dataset = inferred

    if not dataset:
        raise ValueError(
            f"You must provide --dataset, or provide --data-root that can be inferred. "
            f"Supported datasets: {sorted(ALL_DATASETS)}"
        )

    if not data_root:
        data_root = DATASET_DEFAULT_PATHS[dataset]

    return dataset, data_root


def main():
    parser = argparse.ArgumentParser(description="Unified Sobel edge-token statistics for multiple datasets.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name: aitw | odyssey | mind2web | androidcontrol",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Dataset image root path. If omitted, uses built-in default path for the selected dataset.",
    )
    parser.add_argument("--patch-size", type=int, default=28)
    parser.add_argument("--edge-thr", type=int, default=50)
    parser.add_argument("--ratio-thr", type=float, default=0.01)
    args = parser.parse_args()

    dataset, dataset_path = resolve_dataset_and_path(args.dataset, args.data_root)

    image_paths = get_images(dataset_path)

    if not image_paths:
        print(f"No images found in {dataset_path}")
        return

    print(f"Dataset: {dataset}")
    print(f"Path: {dataset_path}")
    print(f"Total images to process: {len(image_paths)}")

    if dataset in NO_RESIZE_DATASETS:
        print(f"Using Pixel Range: [{MIN_PIXELS}, {MAX_PIXELS}]")
        print("Preprocessing strategy: process_vision_info (No 512-resize)")
    else:
        print("Preprocessing strategy: process_vision_info_with_resize (with 512-resize)")

    all_ratios = []

    for img_path in tqdm(image_paths):
        try:
            resized_pil, meta = preprocess_image(img_path, dataset)

            sobel_kwargs = {
                "image_path": img_path,
                "preprocessed_pil": resized_pil,
                "patch_size": args.patch_size,
                "edge_thr": args.edge_thr,
                "ratio_thr": args.ratio_thr,
            }
            if meta["strategy"] == "no_resize":
                sobel_kwargs["min_pixels"] = MIN_PIXELS
                sobel_kwargs["max_pixels"] = MAX_PIXELS

            sobel_result = compute_sobel_uigraph(**sobel_kwargs)

            if sobel_result is None or "is_edge" not in sobel_result:
                raise RuntimeError(f"Sobel computation failed for: {img_path}")

            is_edge_list = sobel_result["is_edge"]
            total_tokens = len(is_edge_list)
            edge_tokens = sum(is_edge_list)

            current_ratio = edge_tokens / total_tokens if total_tokens > 0 else 0
            all_ratios.append(current_ratio)

        except Exception as e:
            print(f"\nError processing {img_path}: {e}")
            raise

    if all_ratios:
        mean_edge_ratio = np.mean(all_ratios)
        print("\n" + "=" * 50)
        print(f"Dataset Statistics for {dataset} (Images: {len(all_ratios)})")
        print(f"Average Edge Token Ratio (Mean of per-image ratios): {mean_edge_ratio:.2%}")
        print(f"Average Non-Edge Token Ratio: {1 - mean_edge_ratio:.2%}")
        print("=" * 50)


if __name__ == "__main__":
    main()
