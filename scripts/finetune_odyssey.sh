export PYTHONPATH=.:$PYTHONPATH

MODEL_NAME="/home/hibug/.cache/huggingface/hub/models--Qwen--Qwen2-VL-2B-Instruct/snapshots/895c3a49bc3fa70a340399125c650a463535e71c"

## Our methods
PYTHONNOUSERSITE=1 /home/hibug/anaconda3/envs/simpagent-qwen25/bin/deepspeed --include localhost:0,1 --master_port 29973 src/training/train_resize.py \
    --lora_enable True \
    --lora_namespan_exclude "['model.embed_tokens', 'lm_head']" \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.0 \
    --num_lora_modules -1 \
    --deepspeed scripts/zero2.json \
    --model_id $MODEL_NAME \
    --data_path "/data_sdc/data/SimpAgent-main/data/guiodyssey_standard_train_high_llavaformat.json" \
    --image_folder "/data_sdc/data/ldq/Dataset/GUI-Odyssey" \
    --freeze_vision_tower True \
    --freeze_llm False \
    --tune_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir /data_sdc/data/SimpAgent-main/output/Qwen2/GUIOdyssey/Baseline4AO_lora_219 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8\
    --min_pixels $((256 * 28 * 28)) \
    --max_pixels $((1280 * 28 * 28)) \
    --learning_rate 3e-5 \
    --merger_lr 0.0 \
    --vision_lr 0.0 \
    --weight_decay 0. \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "constant_with_warmup" \
    --logging_steps 10 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 10000 \
    --save_total_limit 11 \
    --seed 42 \
    --save_only_model True \
    --dataloader_num_workers 12 
   
PYTHONNOUSERSITE=1 PYTHONPATH=. python src/Qwen2/merge_lora_weights.py \
    --model-path /data_sdc/data/SimpAgent-main/output/Qwen2/GUIOdyssey/Baseline4AO_lora_219 \
    --model-base $MODEL_NAME  \
    --save-model-path /data_sdc/data/SimpAgent-main/save/Qwen2/GUIOdyssey/Baseline4AO_lora_219 \
    --safe-serialization

