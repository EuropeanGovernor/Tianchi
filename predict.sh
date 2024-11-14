#!/bin/bash

# topk="$1"
# output="$2"
# adapter="$3"

llamafactory-cli train \
    --stage sft \
    --model_name_or_path Qwen/Qwen2-VL-7B-Instruct \
    --preprocessing_num_workers 16 \
    --quantization_method bitsandbytes \
    --template qwen2_vl \
    --flash_attn auto \
    --dataset_dir data \
    --eval_dataset mire_test \
    --cutoff_len 2560 \
    --max_samples 10000 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate True \
    --max_new_tokens 128 \
    --temperature 0.1 \
    --do_train False \
    --do_predict True \
    --top_k 1 \
    --output_dir ./saves/pred/baseline_lora_nohid_augment_epoch10 \
    --finetuning_type lora \
    --adapter_name_or_path ./saves/baseline_lora_nohid_augment_epoch10/ \
    # --top_k "$topk" \
    # --output_dir "$output" \
    # --adapter_name_or_path "$adapter" \
