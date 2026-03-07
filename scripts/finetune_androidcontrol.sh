export PYTHONPATH=.:$PYTHONPATH

MODEL_NAME="/mnt/data-4/users/zhangzeyu/SimpAgent-main/Model/Qwen2.5VL-3B"


PYTHONNOUSERSITE=1 /home/hibug/anaconda3/envs/simpagent-qwen25/bin/deepspeed --include localhost:0,1 --master_port 29473 src/training/train.py \
    --lora_enable True \
    --lora_namespan_exclude "['model.embed_tokens', 'lm_head']" \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.0 \
    --num_lora_modules -1 \
    --deepspeed scripts/zero2.json \
    --model_id $MODEL_NAME \
    --data_path "/mnt/data-4/users/zhangzeyu/SimpAgent-main/data/androidcontrol_train_llavaformat_no_finish_add_longpress.json" \
    --image_folder "" \
    --freeze_vision_tower True \
    --freeze_llm False \
    --tune_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir "/mnt/data-4/users/zhangzeyu/SimpAgent-main/output/Qwen2.5/AndroidControl/Baseline2AO_lora" \
    --num_train_epochs 5 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --min_pixels $((256 * 28 * 28)) \
    --max_pixels $((1280 * 28 * 28)) \
    --learning_rate 3e-4 \
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
    --save_steps 10000  \
    --save_total_limit 11 \
    --seed 42 \
    --save_only_model True \
    --dataloader_num_workers 12 
   
PYTHONNOUSERSITE=1 PYTHONPATH=. python src/Qwen2/merge_lora_weights.py \
    --model-path /mnt/data-4/users/zhangzeyu/SimpAgent-main/output/Qwen2.5/AndroidControl/Baseline2AO_lora \
    --model-base $MODEL_NAME  \
    --save-model-path /mnt/data-4/users/zhangzeyu/SimpAgent-main/save/Qwen2.5/AndroidControl/Baseline2AO_lora \
    --safe-serialization