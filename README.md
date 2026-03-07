## 📌 Introduction

This repository provides a practical implementation of **token pruning on historical screenshots** for **GUI Visual Agent** tasks, with the goal of **reducing computational cost** during inference/evaluation.  
Our pruning is applied **only to historical screenshots**; we do **not** prune the **current screenshot**, **historical actions**, or **user instructions**.

## ⚙️ Install Dependencies

### Python

This project uses **Python 3.10**.

### Pip

```shell
python -m pip install -r requirements.txt
```

> Note: We intentionally do **not** include `flash-attn` in `requirements.txt`.  
> Please download the wheel that matches your local **PyTorch** and **CUDA** versions from:  
> https://github.com/Dao-AILab/flash-attention/releases  
> and then install it with `pip` in your environment. This is usually faster and more stable than building from source during dependency installation.

## 🚀 Core Contributions in this repo

The main contributions of this repository center on **history-screenshot token pruning during evaluation**.

###  1) Evaluation scripts extended with pruning controls

The following evaluation scripts are extended to support pruning historical visual tokens:

- `AITW_eval.py`
- `AndroidControl_eval.py`
- `Mind2web_eval.py`
- `Odssey_eval.py`

These scripts add CLI switches and runtime hooks for multiple **training-free** pruning strategies on historical screenshots, including:

- FastV (ECCV24)
- SparseVLM (ICML25)
- DART (EMNLP25)
- PDrop (CVPR2025)
- DivPrune (CVPR25)
- Sobel (retain only foreground or background tokens)
- Random pruning


All strategies support per-history-image keep ratios (from most recent history image to older ones), making it possible to allocate more tokens to recent context and fewer tokens to distant history.


To avoid conflicting pruning paths in a single run, enable **one pruning method at a time** in evaluation.

###  2) Model-side pruning implementation

The pruning logic is implemented in model internals:

- `src/Qwen2/model_file/LLM_compression_v2_action/modeling_qwen2vl.py`
- `src/Qwen2_5/model_file/LLM_compression_v2_5_action/modeling_qwen2_5_vl.py`

If you want to modify existing pruning methods or add your own pruning strategy, these are the primary files to edit and extend.

###  3) Sobel token statistics utility

We provide `dataset_sobel_stats.py` to compute **dataset-level Sobel edge-token statistics** across GUI screenshots.

Common arguments:
- `--dataset`: `aitw | odyssey | mind2web | androidcontrol`
- `--data-root`: image root path (optional if you use built-in default paths)
- `--patch-size`: patch size for tokenization (default: `28`)
- `--edge-thr`: Sobel edge threshold (default: `50`)
- `--ratio-thr`: edge-pixel ratio threshold per patch (default: `0.01`)

## 🎯 SFT training scripts

This repo provides supervised fine-tuning (SFT) scripts for all four datasets:

- `scripts/finetune_aitw.sh`
- `scripts/finetune_androidcontrol.sh`
- `scripts/finetune_mind2web.sh`
- `scripts/finetune_odyssey.sh`

You can launch SFT by running the corresponding script, for example:

```shell
bash scripts/finetune_aitw.sh
bash scripts/finetune_androidcontrol.sh
bash scripts/finetune_mind2web.sh
bash scripts/finetune_odyssey.sh
```

> Note: Please update paths (e.g., `MODEL_NAME`, `data_path`, `output_dir`, environment-specific deepspeed/python paths) before running on your machine.

## 📦 Download datasets
You need to download and prepare the AITW, Mind2Web, AndroidControl, and Odyssey datasets on your own, and then replace the image paths in the dataset JSON files we provide accordingly.

- AITW & Mind2Web: https://github.com/njucckevin/SeeClick/blob/main/agent_tasks/readme_agent.md

- AndroidControl: https://github.com/google-research/google-research/tree/master/android_control

- Odyssey: https://huggingface.co/datasets/OpenGVLab/GUI-Odyssey


## 🙏 Acknowledgement
We thank the SimpAgent for providing the foundational codebase for our work.