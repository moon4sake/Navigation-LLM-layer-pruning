export PYTHONPATH=$PYTHONPATH:$(pwd)

set -e
set -x

# CUDA_VISIBLE_DEVICES=3 python pruning_method.py \
#      --base_model llama-3.2-1b \
#      --save_model \
#      --model_path models/llama-3.2-1b/pretrained/ \
#      --pruning_method BI \
#      --cal_data medmcqa \
#      --output_dir models/llama-3.2-1b/pruned/

# CUDA_VISIBLE_DEVICES=3 python pruning_method.py \
#      --base_model vicuna-7b-v1.5 \
#      --save_model \
#      --model_path models/vicuna-7b-v1.5/pretrained/ \
#      --pruning_method BI \
#      --cal_data bookcorpus \
#      --output_dir models/vicuna-7b-v1.5/pruned/

# CUDA_VISIBLE_DEVICES=3 python pruning_method.py \
#      --base_model vicuna-7b-v1.5 \
#      --save_model \
#      --model_path models/vicuna-7b-v1.5/pretrained/ \
#      --pruning_method BI \
#      --cal_data medmcqa_exp \
#      --output_dir models/vicuna-7b-v1.5/pruned/

# CUDA_VISIBLE_DEVICES=3 python pruning_method.py \
#      --base_model vicuna-7b-v1.5 \
#      --save_model \
#      --model_path models/vicuna-7b-v1.5/pretrained/ \
#      --pruning_method BI \
#      --cal_data medqa \
#      --output_dir models/vicuna-7b-v1.5/pruned/

CAL_DATA="gsm8k mmlu truthfulqa"

for CD in $CAL_DATA
do
     CUDA_VISIBLE_DEVICES=0 python pruning_method.py \
          --base_model llama-3.2-3b-instruct \
          --save_model \
          --model_path models/llama-3.2-3b-instruct/pretrained \
          --tokenizer_path models/llama-3.2-3b-instruct/pretrained/ \
          --pruning_method taylor \
          --cal_data ${CD} \
          --output_dir models/llama-3.2-3b-instruct/pretrained/pruned/ &     
     CUDA_VISIBLE_DEVICES=1 python pruning_method.py \
          --base_model llama-3.2-3b-instruct \
          --save_model \
          --model_path models/llama-3.2-3b-instruct/pretrained \
          --tokenizer_path models/llama-3.2-3b-instruct/pretrained/ \
          --pruning_method taylor \
          --cal_data ${CD} --data_cot \
          --output_dir models/llama-3.2-3b-instruct/pretrained/pruned/
done

################################################################################
# CAL_DATA=medmcqa_cot # "bookcorpus medqa medmcqa" # medmcqa_exp"
# MODEL_PATH="llama-3.2-3b-instruct/pretrained" #"llama-3.2-3b-instruct/finetuned/medqa/checkpoint-140"
# MODEL="llama-3.2-3b-instruct"

# for M in $MODEL
# do 
#      for CD in $CAL_DATA
#      do
#      CUDA_VISIBLE_DEVICES=0 python pruning_method.py \
#           --base_model $M \
#           --merge_model \
#           --save_model \
#           --model_path models/$MODEL_PATH \
#           --tokenizer_path models/$M/pretrained/ \
#           --pruning_method taylor \
#           --cal_data $CD \
#           --output_dir models/$MODEL_PATH/pruned/
#      done
# done
################################################################################

# CUDA_VISIBLE_DEVICES=3 python pruning_method.py \
#      --base_model llama-3.2-1b \
#      --save_model \
#      --model_path models/llama-3.2-1b/finetuned/medmcqa/checkpoint-600/ \
#      --tokenizer_path models/llama-3.2-1b/pretrained/ \
#      --pruning_method BI \
#      --output_dir models/llama-3.2-1b/pruned/