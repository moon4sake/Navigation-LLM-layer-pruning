export PYTHONPATH=$PYTHONPATH:$(pwd)

set -e
set -x

CAL_DATA="mmlu gsm8k truthfulqa"
MODEL="llama-3.2-1b"

for CD in $CAL_DATA
do
    CUDA_VISIBLE_DEVICES=0,1 python prune_llm.py \
                                --base_model llama-3.2-1b \
                                --model_path models/llama-3.2-1b \
                                --save_model \
                                --pr_method taylor_${CD} \
                                --remove_layer 8

    CUDA_VISIBLE_DEVICES=0,1 python partial_fine-tuning.py \
                                --base_model llama-3.2-1b \
                                --model_path models/llama-3.2-1b \
                                --save_model  \
                                --data_path datasets/yahma--alpaca-cleaned/ \
                                --partial_layer_name last3 
done