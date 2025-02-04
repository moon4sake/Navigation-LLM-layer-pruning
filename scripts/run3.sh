export PYTHONPATH=$PYTHONPATH:$(pwd)

set -e
set -x

CAL_DATA="mmlu gsm8k truthfulqa"
MODEL="llama-3.2-1b"

NUM_LAYER="3 7 11 14"

for i in $NUM_LAYER
do 
CUDA_VISIBLE_DEVICES=3 python prune_llm.py \
                            --base_model llama-3.2-3b-instruct \
                            --model_path models/llama-3.2-3b-instruct \
                            --save_model \
                            --pr_method taylor_truthfulqa_cot \
                            --blocks "23 22 24 25 21 20 26 19 18 17 16 27 15 14 13 12 11 10 5 8 9 4 7 2 6 0 3 1" \
                            --remove_layer ${i}

CUDA_VISIBLE_DEVICES=3 python partial_fine-tuning.py \
                            --base_model llama-3.2-3b-instruct \
                            --model_path models/llama-3.2-3b-instruct/pruned/pruned_llama-3.2-3b-instruct_taylor_truthfulqa_cot_${i} \
                            --save_model  \
                            --data_path datasets/yahma--alpaca-cleaned/ \
                            --partial_layer_name last3
done


CUDA_VISIBLE_DEVICES=3 python prune_llm.py \
                            --base_model llama-3.2-3b-instruct \
                            --model_path models/llama-3.2-3b-instruct \
                            --save_model \
                            --pr_method tail \
                            --remove_layer 14

CUDA_VISIBLE_DEVICES=3 python partial_fine-tuning.py \
                            --base_model llama-3.2-3b-instruct \
                            --model_path models/llama-3.2-3b-instruct/pruned/pruned_llama-3.2-3b-instruct_tail_14 \
                            --save_model  \
                            --data_path datasets/yahma--alpaca-cleaned/ \
                            --partial_layer_name last3