export PYTHONPATH=$PYTHONPATH:$(pwd)

set -e
set -x

CAL_DATA="mmlu gsm8k truthfulqa"
MODEL="llama-3.2-1b"

for CD in $CAL_DATA
do
    CUDA_VISIBLE_DEVICES=2,3 python prune_llm.py \
                                --base_model llama-3.2-1b \
                                --model_path models/llama-3.2-1b \
                                --save_model \
                                --pr_method "taylor_medmcqa_cot" \
                                --blocks "14 12 13 11 7 10 8 9 6 5 15 4 2 3 1 0" \
                                --remove_layer 4

    CUDA_VISIBLE_DEVICES=2,3 python partial_fine-tuning.py \
                                --base_model llama-3.2-1b \
                                --model_path models/llama-3.2-1b/pruned/oneshot/pruned_llama-3.2-1b_taylor_medmcqa_cot_4 \
                                --save_model  \
                                --data_path datasets/yahma--alpaca-cleaned/ \
                                --partial_layer_name last
done