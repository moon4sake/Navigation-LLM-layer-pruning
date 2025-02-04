export PYTHONPATH=$PYTHONPATH:$(pwd)

set -e
set -x

CAL_DATA="mmlu gsm8k truthfulqa"
MODEL="llama-3.2-1b"

NUM_LAYER="3 7 11 14"

for i in $NUM_LAYER
do 
CUDA_VISIBLE_DEVICES=0 python prune_llm.py \
                            --base_model llama-3.2-3b-instruct \
                            --model_path models/llama-3.2-3b-instruct \
                            --save_model \
                            --pr_method taylor_gsm8k \
                            --blocks "24 23 25 22 26 21 20 0 2 27 19 1 18 17 16 3 4 15 5 6 7 8 9 11 12 10 14 13" \
                            --remove_layer ${i}

CUDA_VISIBLE_DEVICES=0 python partial_fine-tuning.py \
                            --base_model llama-3.2-3b-instruct \
                            --model_path models/llama-3.2-3b-instruct/pruned/pruned_llama-3.2-3b-instruct_taylor_gsm8k_${i} \
                            --save_model  \
                            --data_path datasets/yahma--alpaca-cleaned/ \
                            --partial_layer_name last3
done

CUDA_VISIBLE_DEVICES=0 python prune_llm.py \
                            --base_model llama-3.2-3b-instruct \
                            --model_path models/llama-3.2-3b-instruct \
                            --save_model \
                            --pr_method tail \
                            --remove_layer 3

CUDA_VISIBLE_DEVICES=0 python partial_fine-tuning.py \
                            --base_model llama-3.2-3b-instruct \
                            --model_path models/llama-3.2-3b-instruct/pruned/pruned_llama-3.2-3b-instruct_tail_3 \
                            --save_model  \
                            --data_path datasets/yahma--alpaca-cleaned/ \
                            --partial_layer_name last3