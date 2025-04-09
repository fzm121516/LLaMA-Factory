apt install tmux

conda create --name loki --clone torch
conda activate loki
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip install -e .
pip install transformers==4.51.0

conda activate torch
pip install -e .
pip install -U datasets
llamafactory-cli version

pip install modelscope

modelscope download --model Qwen/Qwen2.5-VL-7B-Instruct --local_dir ./

modelscope download --dataset AI-ModelScope/OpenVid-1M OpenVid_part100.zip --local_dir ./
unzip OpenVid_part100.zip
cp -r /data/mnt /data/LLaMA-Factory/data/OpenVid_part100/

unzip SynthScars.zip
cp -r /data/SynthScars/train/images /data/LLaMA-Factory/data/synthscars

unzip media_data.zip
cp -r /data/loki_media_aggregate /data/LOKI/media_data

unzip impossible_videos.zip
cp -r /data/impossible_videos /data/LLaMA-Factory/data/impossible_videos




llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path /data/Qwen2.5-VL-7B-Instruct \
    --finetuning_type lora \
    --template qwen2_vl \
    --flash_attn auto \
    --dataset_dir data \
    --dataset ipv,OpenVid_part100 \
    --cutoff_len 2048 \
    --learning_rate 0.0001 \
    --num_train_epochs 3.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 500 \
    --warmup_steps 300 \
    --packing False \
    --report_to none \
    --output_dir saves/Qwen2.5-VL-7B-Instruct/lora/train_2025-04-10-02-26-66 \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --optim adamw_torch \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.2 \
    --pissa_init True \
    --pissa_convert True \
    --lora_target all \
    --video_max_pixels 16384 \
    --preprocessing_num_workers 16 \
    --preprocessing_batch_size 1 \
    --tokenized_path /data/LLaMA-Factory/tokenized



llamafactory-cli train examples/train_lora/qwen25vl_lora_sft.yaml



llamafactory-cli export examples/merge_lora/qwen2_5vl_lora_sft.yaml
