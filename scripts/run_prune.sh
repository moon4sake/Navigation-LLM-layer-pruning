export PYTHONPATH=$PYTHONPATH:$(pwd)

set -e
set -x

CAL_DATA="bookcorpus medqa medmcqa medmcqa_exp"
MODEL="llama-3.2-1b-instruct"

for M in $MODEL
do 
     for CD in $CAL_DATA
     do
     CUDA_VISIBLE_DEVICES=0 python pruning_method.py \
          --base_model $M \
          --save_model \
          --model_path models/$M/pretrained \
          --tokenizer_path models/$M/pretrained/ \
          --pruning_method taylor \
          --cal_data $CD \
          --output_dir models/$M/pretrained/pruned/
     done
done