#!/bin/bash

# You can use phi3 instead of phi3.5
MODEL_NAME="/home/noah/zhouxurui/Qwen2-VL-finetune-code/models/Qwen2-VL"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
# MODEL_NAME="/home/wentao/project/Qwen2-VL-Finetune-master/src/model_file/sequnce_visiontoken_compression"
export PYTHONPATH=src:$PYTHONPATH

python src/merge_lora_weights.py \
    --model-path /home/noah/lvyibo/Training/augment/AndroidControl_3e-4_2AI_consine_distill_alpha1_mask557_no_terminate_bs128_lora \
    --model-base $MODEL_NAME  \
    --save-model-path /home/noah/lvyibo/Models/AndroidControl_3e-4_2AI_consine_distill_alpha1_mask557_no_terminate_bs128_lora \
    --safe-serialization