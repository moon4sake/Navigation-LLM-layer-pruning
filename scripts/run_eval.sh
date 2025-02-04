export PYTHONPATH=$PYTHONPATH:$(pwd)

set -e
set -x


# CUDA_VISIBLE_DEVICES=3 lm_eval --model hf \
#      --model_args pretrained=models/llama-3.2-1b/pruned/oneshot/pruned_llama-3.2-1b_taylor_medmcqa_cot_4,trust_remote_code=True,parallelize=True \
#      --tasks mmlu,cmmlu,piqa,openbookqa,winogrande,hellaswag,arc_easy,arc_challenge \
#      --device cuda \
#      --batch_size auto \
#      --num_fewshot 0 \
#      --output_path models/llama-3.2-1b/pruned/oneshot/pruned_llama-3.2-1b_taylor_medmcqa_cot_4/


CAL_DATA="tail taylor_gsm8k taylor_gsm8k_cot taylor_mmlu taylor_mmlu_cot taylor_truthfulqa taylor_truthfulqa_cot"

# for CD in $CAL_DATA
# do
#      CUDA_VISIBLE_DEVICES=0 lm_eval --model hf \
#           --model_args pretrained=models/llama-3.2-1b/pruned/pruned_llama-3.2-1b_${CD}_2/pruned/partial_tuning_taylor8/llama-3.2-1b_partial_tuning_alpaca-cleaned_last3,trust_remote_code=True,parallelize=True \
#           --tasks mmlu,cmmlu,piqa,openbookqa,winogrande,hellaswag,arc_easy,arc_challenge\
#           --device cuda \
#           --batch_size auto \
#           --num_fewshot 0 \
#           --output_path models/llama-3.2-1b/pruned/oneshot/pruned_llama-3.2-1b_${CD}_2/pruned/partial_tuning_taylor8/llama-3.2-1b_partial_tuning_alpaca-cleaned_last3/ &
#      CUDA_VISIBLE_DEVICES=1 lm_eval --model hf \
#           --model_args pretrained=models/llama-3.2-1b/pruned/pruned_llama-3.2-1b_${CD}_4/pruned/partial_tuning_taylor8/llama-3.2-1b_partial_tuning_alpaca-cleaned_last3,trust_remote_code=True,parallelize=True \
#           --tasks mmlu,cmmlu,piqa,openbookqa,winogrande,hellaswag,arc_easy,arc_challenge\
#           --device cuda \
#           --batch_size auto \
#           --num_fewshot 0 \
#           --output_path models/llama-3.2-1b/pruned/oneshot/pruned_llama-3.2-1b_${CD}_4/pruned/partial_tuning_taylor8/llama-3.2-1b_partial_tuning_alpaca-cleaned_last3/ &
#      CUDA_VISIBLE_DEVICES=2 lm_eval --model hf \
#           --model_args pretrained=models/llama-3.2-1b/pruned/pruned_llama-3.2-1b_${CD}_6/pruned/partial_tuning_taylor8/llama-3.2-1b_partial_tuning_alpaca-cleaned_last3,trust_remote_code=True,parallelize=True \
#           --tasks mmlu,cmmlu,piqa,openbookqa,winogrande,hellaswag,arc_easy,arc_challenge\
#           --device cuda \
#           --batch_size auto \
#           --num_fewshot 0 \
#           --output_path models/llama-3.2-1b/pruned/oneshot/pruned_llama-3.2-1b_${CD}_6/pruned/partial_tuning_taylor8/llama-3.2-1b_partial_tuning_alpaca-cleaned_last3/ &
#      CUDA_VISIBLE_DEVICES=3 lm_eval --model hf \
#           --model_args pretrained=models/llama-3.2-1b/pruned/pruned_llama-3.2-1b_${CD}_8/pruned/partial_tuning_taylor8/llama-3.2-1b_partial_tuning_alpaca-cleaned_last3,trust_remote_code=True,parallelize=True \
#           --tasks mmlu,cmmlu,piqa,openbookqa,winogrande,hellaswag,arc_easy,arc_challenge\
#           --device cuda \
#           --batch_size auto \
#           --num_fewshot 0 \
#           --output_path models/llama-3.2-1b/pruned/oneshot/pruned_llama-3.2-1b_${CD}_8/pruned/partial_tuning_taylor8/llama-3.2-1b_partial_tuning_alpaca-cleaned_last3/
# done


# for CD in $CAL_DATA
# do
#      CUDA_VISIBLE_DEVICES=0 lm_eval --model hf \
#           --model_args pretrained=models/llama-3.2-1b/pruned/pruned_llama-3.2-1b_${CD}_2,trust_remote_code=True,parallelize=True \
#           --tasks mmlu,cmmlu,piqa,openbookqa,winogrande,hellaswag,arc_easy,arc_challenge\
#           --device cuda \
#           --batch_size auto \
#           --num_fewshot 0 \
#           --output_path models/llama-3.2-1b/pruned/oneshot/pruned_llama-3.2-1b_${CD}_2/ &
#      CUDA_VISIBLE_DEVICES=1 lm_eval --model hf \
#           --model_args pretrained=models/llama-3.2-1b/pruned/pruned_llama-3.2-1b_${CD}_4,trust_remote_code=True,parallelize=True \
#           --tasks mmlu,cmmlu,piqa,openbookqa,winogrande,hellaswag,arc_easy,arc_challenge\
#           --device cuda \
#           --batch_size auto \
#           --num_fewshot 0 \
#           --output_path models/llama-3.2-1b/pruned/oneshot/pruned_llama-3.2-1b_${CD}_4/ &
#      CUDA_VISIBLE_DEVICES=2 lm_eval --model hf \
#           --model_args pretrained=models/llama-3.2-1b/pruned/pruned_llama-3.2-1b_${CD}_6,trust_remote_code=True,parallelize=True \
#           --tasks mmlu,cmmlu,piqa,openbookqa,winogrande,hellaswag,arc_easy,arc_challenge\
#           --device cuda \
#           --batch_size auto \
#           --num_fewshot 0 \
#           --output_path models/llama-3.2-1b/pruned/oneshot/pruned_llama-3.2-1b_${CD}_6/ &
#      CUDA_VISIBLE_DEVICES=3 lm_eval --model hf \
#           --model_args pretrained=models/llama-3.2-1b/pruned/pruned_llama-3.2-1b_${CD}_8,trust_remote_code=True,parallelize=True \
#           --tasks mmlu,cmmlu,piqa,openbookqa,winogrande,hellaswag,arc_easy,arc_challenge\
#           --device cuda \
#           --batch_size auto \
#           --num_fewshot 0 \
#           --output_path models/llama-3.2-1b/pruned/oneshot/pruned_llama-3.2-1b_${CD}_8/
# done

source activate && conda activate pruning2 && clear
# for CD in $CAL_DATA
# do
#      CUDA_VISIBLE_DEVICES=0 lm_eval --model hf \
#           --model_args pretrained=models/llama-3.2-1b/pruned/pruned_llama-3.2-1b_${CD}_2/pruned/partial_tuning_taylor8/llama-3.2-1b_partial_tuning_alpaca-cleaned_last3,trust_remote_code=True,parallelize=True \
#           --tasks mmlu,cmmlu,piqa,openbookqa,winogrande,hellaswag,arc_easy,arc_challenge\
#           --device cuda \
#           --batch_size auto \
#           --num_fewshot 5 \
#           --output_path models/llama-3.2-1b/pruned/oneshot/pruned_llama-3.2-1b_${CD}_2/pruned/partial_tuning_taylor8/llama-3.2-1b_partial_tuning_alpaca-cleaned_last3/5_shot/ &
#      CUDA_VISIBLE_DEVICES=1 lm_eval --model hf \
#           --model_args pretrained=models/llama-3.2-1b/pruned/pruned_llama-3.2-1b_${CD}_4/pruned/partial_tuning_taylor8/llama-3.2-1b_partial_tuning_alpaca-cleaned_last3,trust_remote_code=True,parallelize=True \
#           --tasks mmlu,cmmlu,piqa,openbookqa,winogrande,hellaswag,arc_easy,arc_challenge\
#           --device cuda \
#           --batch_size auto \
#           --num_fewshot 5 \
#           --output_path models/llama-3.2-1b/pruned/oneshot/pruned_llama-3.2-1b_${CD}_4/pruned/partial_tuning_taylor8/llama-3.2-1b_partial_tuning_alpaca-cleaned_last3/5_shot/ &
#      CUDA_VISIBLE_DEVICES=2 lm_eval --model hf \
#           --model_args pretrained=models/llama-3.2-1b/pruned/pruned_llama-3.2-1b_${CD}_6/pruned/partial_tuning_taylor8/llama-3.2-1b_partial_tuning_alpaca-cleaned_last3,trust_remote_code=True,parallelize=True \
#           --tasks mmlu,cmmlu,piqa,openbookqa,winogrande,hellaswag,arc_easy,arc_challenge\
#           --device cuda \
#           --batch_size auto \
#           --num_fewshot 5 \
#           --output_path models/llama-3.2-1b/pruned/oneshot/pruned_llama-3.2-1b_${CD}_6/pruned/partial_tuning_taylor8/llama-3.2-1b_partial_tuning_alpaca-cleaned_last3/5_shot/ &
#      CUDA_VISIBLE_DEVICES=3 lm_eval --model hf \
#           --model_args pretrained=models/llama-3.2-1b/pruned/pruned_llama-3.2-1b_${CD}_8/pruned/partial_tuning_taylor8/llama-3.2-1b_partial_tuning_alpaca-cleaned_last3,trust_remote_code=True,parallelize=True \
#           --tasks mmlu,cmmlu,piqa,openbookqa,winogrande,hellaswag,arc_easy,arc_challenge\
#           --device cuda \
#           --batch_size auto \
#           --num_fewshot 5 \
#           --output_path models/llama-3.2-1b/pruned/oneshot/pruned_llama-3.2-1b_${CD}_8/pruned/partial_tuning_taylor8/llama-3.2-1b_partial_tuning_alpaca-cleaned_last3/5_shot/
# done


# for CD in $CAL_DATA
# do
#      CUDA_VISIBLE_DEVICES=0 lm_eval --model hf \
#           --model_args pretrained=models/llama-3.2-1b/pruned/pruned_llama-3.2-1b_${CD}_2,trust_remote_code=True,parallelize=True \
#           --tasks mmlu,cmmlu,piqa,openbookqa,winogrande,hellaswag,arc_easy,arc_challenge\
#           --device cuda \
#           --batch_size auto \
#           --num_fewshot 5 \
#           --output_path models/llama-3.2-1b/pruned/oneshot/pruned_llama-3.2-1b_${CD}_2/5_shot/ &
#      CUDA_VISIBLE_DEVICES=1 lm_eval --model hf \
#           --model_args pretrained=models/llama-3.2-1b/pruned/pruned_llama-3.2-1b_${CD}_4,trust_remote_code=True,parallelize=True \
#           --tasks mmlu,cmmlu,piqa,openbookqa,winogrande,hellaswag,arc_easy,arc_challenge\
#           --device cuda \
#           --batch_size auto \
#           --num_fewshot 5 \
#           --output_path models/llama-3.2-1b/pruned/oneshot/pruned_llama-3.2-1b_${CD}_4/5_shot/ &
#      CUDA_VISIBLE_DEVICES=2 lm_eval --model hf \
#           --model_args pretrained=models/llama-3.2-1b/pruned/pruned_llama-3.2-1b_${CD}_6,trust_remote_code=True,parallelize=True \
#           --tasks mmlu,cmmlu,piqa,openbookqa,winogrande,hellaswag,arc_easy,arc_challenge\
#           --device cuda \
#           --batch_size auto \
#           --num_fewshot 5 \
#           --output_path models/llama-3.2-1b/pruned/oneshot/pruned_llama-3.2-1b_${CD}_6/5_shot/ &
#      CUDA_VISIBLE_DEVICES=3 lm_eval --model hf \
#           --model_args pretrained=models/llama-3.2-1b/pruned/pruned_llama-3.2-1b_${CD}_8,trust_remote_code=True,parallelize=True \
#           --tasks mmlu,cmmlu,piqa,openbookqa,winogrande,hellaswag,arc_easy,arc_challenge\
#           --device cuda \
#           --batch_size auto \
#           --num_fewshot 5 \
#           --output_path models/llama-3.2-1b/pruned/oneshot/pruned_llama-3.2-1b_${CD}_8/5_shot/
# done


CUDA_VISIBLE_DEVICES=0 lm_eval --model hf \
     --model_args pretrained=models/llama-3.2-1b/pretrained,trust_remote_code=True,parallelize=True \
     --tasks mmlu,cmmlu,piqa,openbookqa,winogrande,hellaswag,arc_easy,arc_challenge\
     --device cuda \
     --batch_size auto \
     --num_fewshot 5 \
     --output_path models/llama-3.2-1b/pretrained/5_shot/


# CUDA_VISIBLE_DEVICES=0 lm_eval --model hf \
#      --model_args pretrained=models/llama-3.2-3b-instruct/pretrained,trust_remote_code=True,parallelize=True \
#      --tasks pubmedqa \
#      --device cuda \
#      --batch_size auto \
#      --num_fewshot 0 \
#      --output_path models/llama-3.2-3b-instruct/pretrained/

# CUDA_VISIBLE_DEVICES=3 lm_eval --model hf \
#      --model_args pretrained=models/llama-3.2-1b/pretrained,trust_remote_code=True,peft=models/llama-3.2-1b/finetuned/medmcqa/checkpoint-600,parallelize=True \
#      --tasks mmlu,cmmlu,piqa,openbookqa,winogrande,hellaswag,arc_easy,arc_challenge \
#      --device cuda \
#      --batch_size auto \
#      --num_fewshot 0 \
#      --output_path models/llama-3.2-1b/finetuned/medmcqa/checkpoint-600/