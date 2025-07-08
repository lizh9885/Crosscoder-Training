# %%
from ddp_trainer import DDPTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import torch
from torch.utils.data import DataLoader, DistributedSampler
from aligner import MLPAligner
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os


def train_ddp(rank, world_size, cfg):
    trainer = DDPTrainer(cfg, rank, world_size)
    trainer.train()
def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--modelA_name", type = str, default = "meta-llama/Llama-3.1-8B")
    parser.add_argument("--modelA_size", type = int, default = 4096)
    parser.add_argument("--modelB_name", type = str, default = "google/gemma-2-9b")
    parser.add_argument("--modelB_size", type = int, default = 3584)
    parser.add_argument("--gpu_nums",type = int, default = 2)
    parser.add_argument("--samples_per_file",type = int, default = 50000)
    parser.add_argument("--wandb_project",type = str, default = "crosscoder_train")
    parser.add_argument("--wandb_name",type = str, required = True)
    parser.add_argument("--batch_size", type = int, required =True)
    # parser.add_argument("--modelA_name", type = str, default = "google/gemma-2-2b-it")
    # parser.add_argument("--modelA_size", type = int, default = 2304)
    # parser.add_argument("--modelB_name", type = str, default = "google/gemma-2-2b")
    # parser.add_argument("--modelB_size", type = int, default = 2304)
    
    args = parser.parse_args()
    
    
    
    modelA_save_name = args.modelA_name.split("/")[-1]
    modelB_save_name = args.modelB_name.split("/")[-1]
    default_cfg = {
        "seed": 49,
        "batch_size": args.batch_size,
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
        "samples_per_file":args.samples_per_file,
        # "device": "cuda:0",
        # "model_batch_size": 4,
        "log_every": 10,
        "save_every": 200,
        "dec_init_norm": 0.08,
        "wandb_project": args.wandb_project,
        "wandb_name": args.wandb_name,
    }
    # 设置环境变量
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    mp.spawn(train_ddp, args=(args.gpu_nums, default_cfg), nprocs=args.gpu_nums, join=True)
    

    



if __name__ == "__main__":
    
    main()