apt install tmux

conda create --name loki --clone torch
conda activate loki
pip install -e .
pip install transformers==4.50.3


conda activate torch
pip install -e .
llamafactory-cli version


unzip SynthScars.zip

cp -r /data/LLaMA-Factory/SynthScars/train/images/ /data/LLaMA-Factory/data/synthscars/



llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path /data/Qwen2.5-VL-7B-Instruct \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template qwen2_vl \
    --flash_attn auto \
    --dataset_dir data \
    --dataset synthscars \
    --cutoff_len 2048 \
    --learning_rate 0.0001 \
    --num_train_epochs 3.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 0 \
    --packing False \
    --report_to none \
    --output_dir saves/Qwen2.5-VL-7B-Instruct/lora/train_2025-04-08-11-21-31 \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --lora_target all