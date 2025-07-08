# %%
from trainer import Trainer
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import torch
device = 'cuda:0'

parser = argparse.ArgumentParser()

# parser.add_argument("--modelA_name", type = str, default = "meta-llama/Llama-3.1-8B")
# parser.add_argument("--modelA_size", type = int, default = 4096)
# parser.add_argument("--modelB_name", type = str, default = "google/gemma-2-9b")
# parser.add_argument("--modelB_size", type = int, default = 3584)



parser.add_argument("--modelA_name", type = str, default = "google/gemma-2-2b-it")
parser.add_argument("--modelA_size", type = int, default = 2304)
parser.add_argument("--modelB_name", type = str, default = "google/gemma-2-2b")
parser.add_argument("--modelB_size", type = int, default = 2304)

args = parser.parse_args()



# modelA = AutoModelForCausalLM.from_pretrained(
#     args.modelA_name, output_hidden_states = True,
#     device=device, 
# )
# modelA_tokenizer = AutoTokenizer.from_pretrained(args.modelA_name)
# modelB = AutoModelForCausalLM.from_pretrained(
#     args.modelB_name, output_hidden_states = True,
#     device=device, 
# )
# modelB_tokenizer = AutoTokenizer.from_pretrained(args.modelB_name)
# # %%
# # all_tokens = load_pile_lmsys_mixed_tokens()
# modelA_tokens, modelB_tokens = load_pile_lmsys_mixed_tokens_v2(args.modelA_name,args.modelB_name)

# %%
modelA_save_name = args.modelA_name.split("/")[-1]
modelB_save_name = args.modelB_name.split("/")[-1]
default_cfg = {
    "seed": 49,
    "batch_size": 32,
    # "buffer_mult": 128,
    "lr": 5e-5,
    # "num_tokens": 400_000_000,
    "l1_coeff": 2,
    "x_dir":f"act/{modelA_save_name}/",
    "y_dir":f"act/{modelB_save_name}/",
    "num_workers":4,
    "beta1": 0.9,
    "beta2": 0.999,
    "A_in": args.modelA_size,
    "B_in": args.modelB_size,
    "dict_size": 2**14,
    # "seq_len": 1024,
    "enc_dtype": "fp32",
    "device": "cuda:0",
    # "model_batch_size": 4,
    "log_every": 10,
    "save_every": 200,
    "dec_init_norm": 0.08,
    "wandb_project": "crosscoder_train",
    "wandb_name": "crosscoder_gemma2-2b_gemma2-2b-it",
}


trainer = Trainer(default_cfg)
trainer.train()
# %%