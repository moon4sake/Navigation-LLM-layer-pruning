export PYTHONPATH=$PYTHONPATH:$(pwd)

set -e
set -x

CAL_DATA="mmlu gsm8k truthfulqa"
MODEL="llama-3.2-1b"

# CUDA_VISIBLE_DEVICES=1,2 python prune_llm.py \
#                             --base_model llama-3.2-1b \
#                             --model_path models/llama-3.2-1b \
#                             --save_model \
#                             --pr_method tail \
#                             --remove_layer 4

# CUDA_VISIBLE_DEVICES=1,2 python partial_fine-tuning.py \
#                             --base_model llama-3.2-1b \
#                             --model_path models/llama-3.2-1b/pruned/oneshot/pruned_llama-3.2-1b_tail_4 \
#                             --save_model  \
#                             --data_path datasets/yahma--alpaca-cleaned/ \
#                             --partial_layer_name last

##### gsm8k: 14 13 12 1 2 11 15 0 3 10 5 4 7 8 6 9
##### gsm8k-COT: 13 14 12 11 15 1 2 0 10 3 9 4 5 8 7 6
##### mmlu: 14 13 12 2 1 3 0 11 5 15 4 7 10 6 8 9
##### mmlu-COT: 13 14 12 11 15 10 1 9 2 0 4 8 5 7 6 3
##### truthfulqa: 14 13 12 11 7 10 5 6 8 2 15 3 1 4 9 0
##### truthfulqa-COT: 14 13 12 11 15 10 9 8 7 6 5 4 1 2 3 0

CUDA_VISIBLE_DEVICES=2,3 python prune_llm.py \
                            --base_model llama-3.2-1b \
                            --model_path models/llama-3.2-1b \
                            --save_model \
                            --pr_method taylor_truthfulqa \
                            --blocks "14 13 12 11 7 10 5 6 8 2 15 3 1 4 9 0" \
                            --remove_layer 2

CUDA_VISIBLE_DEVICES=2,3 python partial_fine-tuning.py \
                            --base_model llama-3.2-1b \
                            --model_path models/llama-3.2-1b/pruned/pruned_llama-3.2-1b_taylor_truthfulqa_2 \
                            --save_model  \
                            --data_path datasets/yahma--alpaca-cleaned/ \
                            --partial_layer_name last3

CUDA_VISIBLE_DEVICES=2,3 python prune_llm.py \
                            --base_model llama-3.2-1b \
                            --model_path models/llama-3.2-1b \
                            --save_model \
                            --pr_method taylor_truthfulqa_cot \
                            --blocks "13 14 12 11 15 10 1 9 2 0 4 8 5 7 6 3" \
                            --remove_layer 2

CUDA_VISIBLE_DEVICES=2,3 python partial_fine-tuning.py \
                            --base_model llama-3.2-1b \
                            --model_path models/llama-3.2-1b/pruned/pruned_llama-3.2-1b_taylor_truthfulqa_cot_2 \
                            --save_model  \
                            --data_path datasets/yahma--alpaca-cleaned/ \
                            --partial_layer_name last3

CUDA_VISIBLE_DEVICES=2,3 python prune_llm.py \
                            --base_model llama-3.2-1b \
                            --model_path models/llama-3.2-1b \
                            --save_model \
                            --pr_method taylor_truthfulqa \
                            --blocks "14 13 12 11 7 10 5 6 8 2 15 3 1 4 9 0" \
                            --remove_layer 6

CUDA_VISIBLE_DEVICES=2,3 python partial_fine-tuning.py \
                            --base_model llama-3.2-1b \
                            --model_path models/llama-3.2-1b/pruned/pruned_llama-3.2-1b_taylor_truthfulqa_6 \
                            --save_model  \
                            --data_path datasets/yahma--alpaca-cleaned/ \
                            --partial_layer_name last3

CUDA_VISIBLE_DEVICES=2,3 python prune_llm.py \
                            --base_model llama-3.2-1b \
                            --model_path models/llama-3.2-1b \
                            --save_model \
                            --pr_method taylor_truthfulqa_cot \
                            --blocks "13 14 12 11 15 10 1 9 2 0 4 8 5 7 6 3" \
                            --remove_layer 6

CUDA_VISIBLE_DEVICES=2,3 python partial_fine-tuning.py \
                            --base_model llama-3.2-1b \
                            --model_path models/llama-3.2-1b/pruned/pruned_llama-3.2-1b_taylor_truthfulqa_cot_6 \
                            --save_model  \
                            --data_path datasets/yahma--alpaca-cleaned/ \
                            --partial_layer_name last3
