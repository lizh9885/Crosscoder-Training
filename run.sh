export HF_TOKEN="your token here"
huggingface-cli login --token $HF_TOKEN

export WANDB_API_KEY='your token here' 


samples_per_file=50000
gpu_ids="0,1"
gpu_nums=2


# python3 id_store.py --modelA_name meta-llama/Llama-3.1-8B --modelB_name google/gemma-2-9b --num_cuts 100 --num_samples 50000

# python3 act_store.py --model meta-llama/Llama-3.1-8B --batch_size 8 --layer_idx 29 --save_every $samples_per_file --gpu_ids $gpu_ids
# python3 act_store.py --model google/gemma-2-9b --batch_size 8 --layer_idx 39 --save_every $samples_per_file --gpu_ids $gpu_ids


python3 ddp_train.py --modelA_name meta-llama/Llama-3.1-8B\
                     --modelA_size 4096\
                     --modelB_name google/gemma-2-9b\
                     --modelB_size 3584\
                     --gpu_nums $gpu_nums\
                     --samples_per_file $samples_per_file\
                     --wandb_project crosscoder_train\
                     --wandb_name llama8b-gemma9b-test\
                     --batch_size 64\
                     