export PYTHONPATH=$PYTHONPATH:$(pwd)

set -e
set -x

CUDA_VISIBLE_DEVICES=0 lm_eval --model hf \
     --model_args pretrained=models/llama-3.2-3b-instruct/pretrained,trust_remote_code=True,parallelize=True \
     --tasks pubmedqa \
     --device cuda \
     --batch_size auto \
     --num_fewshot 0 \
     --output_path models/llama-3.2-3b-instruct/pretrained/

# CUDA_VISIBLE_DEVICES=3 lm_eval --model hf \
#      --model_args pretrained=models/llama-3.2-1b/pretrained,trust_remote_code=True,peft=models/llama-3.2-1b/finetuned/medmcqa/checkpoint-600,parallelize=True \
#      --tasks mmlu,cmmlu,piqa,openbookqa,winogrande,hellaswag,arc_easy,arc_challenge \
#      --device cuda \
#      --batch_size auto \
#      --num_fewshot 0 \
#      --output_path models/llama-3.2-1b/finetuned/medmcqa/checkpoint-600/