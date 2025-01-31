export PYTHONPATH=$PYTHONPATH:$(pwd)

set -e
set -x



# CUDA_VISIBLE_DEVICES=0,1,2 python finetune_med.py \
#     --base_model llama-3.2-1b \
#     --save_model \
#     --model_path models/llama-3.2-1b \
#     --output_dir models/llama-3.2-1b/finetuned/medmcqa/ \
#     --batch_size 512 \
#     --micro_batch_size 32

CUDA_VISIBLE_DEVICES=0,1,2,3 python finetune_medmcqa.py \
    --base_model vicuna-7b-v1.5 \
    --save_model \
    --model_path models/vicuna-7b-v1.5 \
    --output_dir models/vicuna-7b-v1.5/finetuned/medmcqa/ \
    --num_epochs 2 \
    --batch_size 128 \
    --micro_batch_size 16

# CUDA_VISIBLE_DEVICES=0,1 python finetune.py \
#     --base_model llama-3.2-1b \
#     --save_model \
#     --model_path models/llama-3.2-1b \
#     --data_path datasets/GBaker--MedQA-USMLE-4-options-hf/ \
#     --output_dir models/llama-3.2-1b/finetuned/