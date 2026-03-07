## Install Dependencies

### Python

This project uses **Python 3.10**.

### Pip

```shell
python -m pip install -r requirements.txt
```

If you are using the Qwen2.5-VL + FlashAttention2 setup, we recommend installing `flash-attn` from a pre-built wheel (download it separately and install it via `pip install /path/to/flash_attn-*.whl`) for faster and more stable installation.


## Core modifications in this repo

The main contribution of this repository is **history-screenshot token pruning during evaluation**.

### 1) Evaluation scripts extended with pruning controls

The following evaluation scripts are extended to support pruning historical visual tokens:

- `AITW_eval.py`
- `AndroidControl_eval.py`
- `Mind2web_eval.py`
- `Odssey_eval.py`

These scripts add CLI switches and runtime hooks for multiple **training-free** pruning strategies on historical screenshots, including:

- FastV
- PDrop
- DivPrune
- Sobel (edge-only)
- Random pruning
- SparseVLM
- DART

Most strategies support per-history-image keep ratios (from most recent history image to older ones), making it possible to allocate more tokens to recent context and fewer tokens to distant history.

### 2) Model-side pruning implementation

The pruning logic is implemented in model internals:

- `src/Qwen2/model_file/LLM_compression_v2_action/modeling_qwen2vl.py`
- `src/Qwen2_5/model_file/LLM_compression_v2_5_action/modeling_qwen2_5_vl.py`

Both files include:

- pruning enable flags (e.g., `enable_fastv_pruning`, `enable_pdrop_pruning`, `enable_DivPrune_pruning`, `enable_dart_pruning`, etc.)
- strategy-specific keep-ratio settings (e.g., `*_keep_ratios`)
- keep-mask construction for history-image token segments
- integration into decoding at `drop_k` layers


### 3) Reproducibility note

To avoid conflicting pruning paths in a single run, enable **one pruning method at a time** in evaluation.

## SFT training scripts

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

Each script includes:

- DeepSpeed training command (`src/training/train.py` or `src/training/train_resize.py`)
- LoRA-based fine-tuning configuration
- A post-training LoRA merge step via `src/Qwen2/merge_lora_weights.py`

> Note: Please update paths (e.g., `MODEL_NAME`, `data_path`, `output_dir`, environment-specific deepspeed/python paths) before running on your machine.

## Download datasets
You need to download and prepare the AITW, Mind2Web, AndroidControl, and Odyssey datasets on your own, and then replace the image paths in the dataset JSON files we provide accordingly.

- AITW & Mind2Web: https://github.com/njucckevin/SeeClick/blob/main/agent_tasks/readme_agent.md

- AndroidControl: https://github.com/google-research/google-research/tree/master/android_control

- Odyssey: https://huggingface.co/datasets/OpenGVLab/GUI-Odyssey

- SimpAgent training files: https://huggingface.co/datasets/Minuskid/SimpAgent-data

## Acknowledgement
We thank the SimpAgent for providing the foundational codebase for our work.