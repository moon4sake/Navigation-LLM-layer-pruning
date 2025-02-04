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

##### gsm8k: 24 23 25 22 26 21 20 0 2 27 19 1 18 17 16 3 4 15 5 6 7 8 9 11 12 10 14 13
##### gsm8k-COT: 24 25 23 22 21 20 26 19 18 17 16 2 0 27 1 3 15 4 14 5 13 12 11 6 7 8 10 9
##### mmlu: 24 23 25 26 22 21 20 19 27 18 17 16 15 12 13 11 9 14 10 8 2 0 7 3 6 4 5 1
##### mmlu-COT: 24 22 23 25 21 20 26 19 2 18 17 1 16 0 3 4 27 15 5 14 6 13 12 7 10 11 9 8
##### truthfulqa: 23 25 24 22 26 21 20 19 18 17 16 27 2 15 9 12 8 11 0 3 10 7 13 6 1 14 5 4
##### truthfulqa-COT: 23 22 24 25 21 20 26 19 18 17 16 27 15 14 13 12 11 10 5 8 9 4 7 2 6 0 3 1



CUDA_VISIBLE_DEVICES=0 python prune_llm.py \
                            --base_model llama-3.2-3b-instruct \
                            --model_path models/llama-3.2-3b-instruct \
                            --save_model \
                            --pr_method taylor_gsm8k \
                            --blocks "24 23 25 22 26 21 20 0 2 27 19 1 18 17 16 3 4 15 5 6 7 8 9 11 12 10 14 13" \
                            --remove_layer 7 

CUDA_VISIBLE_DEVICES=0 python partial_fine-tuning.py \
                            --base_model llama-3.2-3b-instruct \
                            --model_path models/llama-3.2-3b-instruct/pruned/pruned_llama-3.2-3b-instruct_taylor_gsm8k_7 \
                            --save_model  \
                            --data_path datasets/yahma--alpaca-cleaned/ \
                            --partial_layer_name last3

CUDA_VISIBLE_DEVICES=0 python prune_llm.py \
                            --base_model llama-3.2-3b-instruct \
                            --model_path models/llama-3.2-3b-instruct \
                            --save_model \
                            --pr_method taylor_gsm8k_cot \
                            --blocks "24 25 23 22 21 20 26 19 18 17 16 2 0 27 1 3 15 4 14 5 13 12 11 6 7 8 10 9" \
                            --remove_layer 7 &

CUDA_VISIBLE_DEVICES=0 python partial_fine-tuning.py \
                            --base_model llama-3.2-3b-instruct \
                            --model_path models/llama-3.2-3b-instruct/pruned/pruned_llama-3.2-3b-instruct_taylor_gsm8k_cot_7 \
                            --save_model  \
                            --data_path datasets/yahma--alpaca-cleaned/ \
                            --partial_layer_name last3



CUDA_VISIBLE_DEVICES=0 python prune_llm.py \
                            --base_model llama-3.2-3b-instruct \
                            --model_path models/llama-3.2-3b-instruct \
                            --save_model \
                            --pr_method taylor_truthfulqa \
                            --blocks "23 25 24 22 26 21 20 19 18 17 16 27 2 15 9 12 8 11 0 3 10 7 13 6 1 14 5 4" \
                            --remove_layer 7 

CUDA_VISIBLE_DEVICES=0 python partial_fine-tuning.py \
                            --base_model llama-3.2-3b-instruct \
                            --model_path models/llama-3.2-3b-instruct/pruned/pruned_llama-3.2-3b-instruct_taylor_truthfulqa_7 \
                            --save_model  \
                            --data_path datasets/yahma--alpaca-cleaned/ \
                            --partial_layer_name last3

CUDA_VISIBLE_DEVICES=0 python prune_llm.py \
                            --base_model llama-3.2-3b-instruct \
                            --model_path models/llama-3.2-3b-instruct \
                            --save_model \
                            --pr_method taylor_truthfulqa_cot \
                            --blocks "23 22 24 25 21 20 26 19 18 17 16 27 15 14 13 12 11 10 5 8 9 4 7 2 6 0 3 1" \
                            --remove_layer 7 &

CUDA_VISIBLE_DEVICES=0 python partial_fine-tuning.py \
                            --base_model llama-3.2-3b-instruct \
                            --model_path models/llama-3.2-3b-instruct/pruned/pruned_llama-3.2-3b-instruct_taylor_truthfulqa_cot_7 \
                            --save_model  \
                            --data_path datasets/yahma--alpaca-cleaned/ \
                            --partial_layer_name last3