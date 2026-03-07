export PYTHONPATH=.:$PYTHONPATH
MODEL_NAME="/home/hibug/.cache/huggingface/hub/models--Qwen--Qwen2-VL-2B-Instruct/snapshots/895c3a49bc3fa70a340399125c650a463535e71c"

PYTHONNOUSERSITE=1 /home/hibug/anaconda3/envs/simpagent-qwen25/bin/deepspeed --include localhost:0,1 --master_port 29473 src/training/train.py \
    --lora_enable True \
    --lora_namespan_exclude "['model.embed_tokens', 'lm_head']" \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.0 \
    --num_lora_modules -1 \
    --deepspeed scripts/zero2.json \
    --model_id $MODEL_NAME \
    --data_path "/data_sdc/data/ldq/SimpAgent-main/data/mind2web_train_sft_all_sequence_llavaformat.json" \
    --image_folder "" \
    --freeze_vision_tower True \
    --freeze_llm False \
    --tune_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir "/data_sdc/data/ldq/SimpAgent-main/output/Mind2Web/4AO_baseline_lora" \
    --num_train_epochs 5 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --min_pixels $((256 * 28 * 28)) \
    --max_pixels $((1280 * 28 * 28)) \
    --learning_rate 5e-4 \
    --merger_lr 0.0 \
    --vision_lr 0.0 \
    --weight_decay 0. \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 483  \
    --save_total_limit 11 \
    --seed 42 \
    --save_only_model True \
    --dataloader_num_workers 12 \

PYTHONNOUSERSITE=1 PYTHONPATH=. python src/Qwen2/merge_lora_weights.py \
    --model-path /data_sdc/data/ldq/SimpAgent-main/output/Mind2Web/4AO_baseline_lora \
    --model-base $MODEL_NAME  \
    --save-model-path /data_sdc/data/ldq/SimpAgent-main/save/Mind2Web/4AO_baseline_lora \
    --safe-serialization