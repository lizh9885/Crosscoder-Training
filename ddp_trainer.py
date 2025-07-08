# from utils import *
from crosscoder import CrossCoder
# from buffer import Buffer
import tqdm
from loader import PairedActivationDataset
import numpy as np
import torch
import wandb
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, DistributedSampler
from aligner import MLPAligner
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os


class DDPTrainer:
    def __init__(self, cfg, rank, world_size):
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f'cuda:{rank}')
        
        # 初始化进程组
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        
        # 设置当前设备
        torch.cuda.set_device(rank)

        # 加载变换器
        self.MLP = MLPAligner(cfg)
        self.MLP.load_state_dict(torch.load("./aligner_weight/aligner_model.pt", map_location=self.device))
        self.MLP.to(self.device)
        self.MLP.eval()
        
        # 加载crosscoder
        self.crosscoder = CrossCoder(cfg)
        self.crosscoder.to(self.device)
        
        # 用DDP包装模型
        self.crosscoder = DDP(self.crosscoder, device_ids=[rank], output_device=rank)
        
        # 数据集和数据加载器
        self.dataset = PairedActivationDataset(
            x_dir=cfg["x_dir"],
            y_dir=cfg["y_dir"],
            samples_per_file=cfg["samples_per_file"]
        )
        
        # 使用DistributedSampler确保每个进程处理不同的数据
        self.sampler = DistributedSampler(
            self.dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=cfg["batch_size"],
            sampler=self.sampler,  # 使用DistributedSampler而不是shuffle
            num_workers=cfg["num_workers"],
            pin_memory=True,
            persistent_workers=True
        )
        
        # clip norm
        estimated_norm_scaling_factor_A, estimated_norm_scaling_factor_B = self.estimate_norm_scaling_factor(
            cfg["batch_size"], n_batches_for_norm_estimate=1
        )
        
        self.normalisation_factor = torch.tensor(
            [estimated_norm_scaling_factor_B, estimated_norm_scaling_factor_B],
            device=self.device,
            dtype=torch.float32,
        )

        # optimizer
        print("**********",len(self.dataset),cfg["batch_size"] , world_size)
        self.total_steps = len(self.dataset) // (cfg["batch_size"] * world_size)
        
        self.optimizer = torch.optim.Adam(
            self.crosscoder.parameters(),
            lr=cfg["lr"],
            betas=(cfg["beta1"], cfg["beta2"]),
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, self.lr_lambda
        )
        self.step_counter = 0
        
        # 只在主进程初始化wandb
        if rank == 0:
            wandb.init(project=cfg["wandb_project"], name=cfg["wandb_name"])

    def lr_lambda(self, step):
        if step < 0.8 * self.total_steps:
            return 1.0
        else:
            return 1.0 - (step - 0.8 * self.total_steps) / (0.2 * self.total_steps)

    def get_l1_coeff(self):
        if self.step_counter < 0.05 * self.total_steps:
            return self.cfg["l1_coeff"] * self.step_counter / (0.05 * self.total_steps)
        else:
            return self.cfg["l1_coeff"]

            
    def transfom_with_MLP(self,x_batch,y_batch):
        x_batch = x_batch.cuda()
        y_batch = y_batch.cuda()
        X_mean = torch.load("mean_var/modelA_mean.pt").cuda()
        X_std = torch.load("mean_var/modelA_std.pt").cuda()
        Y_mean = torch.load("mean_var/modelB_mean.pt").cuda()
        Y_std = torch.load("mean_var/modelB_std.pt").cuda()

        x_batch_s = (x_batch - X_mean)/X_std
        # y_batch_s = (y_batch - Y_mean)/Y_std

        x_batch_s2y = self.MLP(x_batch_s)
        x_batch_2y = x_batch_s2y *Y_std + Y_mean
        assert x_batch_2y.size()[-1] == y_batch.size()[-1]
        acts = torch.stack([x_batch_2y,y_batch],dim=0) #(2,batch,B_in)
        acts = acts.permute(1,0,2) #(batch,2,B_in)
        acts = acts[torch.randperm(acts.shape[0]).to(self.device)]
        acts = acts * self.normalisation_factor[None, :, None]
        return acts
    def step(self):
        x_batch, y_batch = next(self.iter)
        x_batch = x_batch.float()
        y_batch = y_batch.float()
        acts = self.transfom_with_MLP(x_batch, y_batch)
        losses = self.crosscoder.module.get_losses(acts)
        loss = 0.02*losses.l2_loss + self.get_l1_coeff() * losses.l1_loss
        loss.backward()
        clip_grad_norm_(self.crosscoder.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        loss_dict = {
            "loss": loss.item(),
            "l2_loss": losses.l2_loss.item(),
            "l1_loss": losses.l1_loss.item(),
            "l0_loss": losses.l0_loss.item(),
            "l1_coeff": self.get_l1_coeff(),
            "lr": self.scheduler.get_last_lr()[0],
            "explained_variance": losses.explained_variance.mean().item(),
            "explained_variance_A": losses.explained_variance_A.mean().item(),
            "explained_variance_B": losses.explained_variance_B.mean().item(),
        }
        self.step_counter += 1
        return loss_dict
    def log(self, loss_dict):
        # 只在主进程记录日志
        if self.rank == 0:
            wandb.log(loss_dict, step=self.step_counter)
            print(loss_dict)

    def save(self):
        # 只在主进程保存模型
        pass
        # if self.rank == 0:
            # 保存DDP模型时需要使用module属性
            # torch.save(self.crosscoder.module.state_dict(), 
            #           f"checkpoints/crosscoder_step_{self.step_counter}.pt")
            # self.crosscoder.module.save()

    def train(self):
        self.step_counter = 0
        self.iter = iter(self.dataloader)
        
        try:
            for i in tqdm.trange(self.total_steps, disable=self.rank != 0):  # 只在主进程显示进度条
                # 每个epoch开始时设置sampler的epoch
                if i % len(self.dataloader) == 0:
                    self.sampler.set_epoch(i // len(self.dataloader))
                
                loss_dict = self.step()
                
                if i % self.cfg["log_every"] == 0:
                    self.log(loss_dict)
                if (i + 1) % self.cfg["save_every"] == 0:
                    self.save()
        finally:
            self.save()
            # 清理进程组
            dist.destroy_process_group()

    def estimate_norm_scaling_factor(self, batch_size, n_batches_for_norm_estimate=100):
        loader = DataLoader(
            self.dataset,
            batch_size=self.cfg["batch_size"],
            sampler=self.sampler,
            num_workers=self.cfg["num_workers"],
            pin_memory=True,
            persistent_workers=True
        )

        x_norms_per_batch = []
        y_norms_per_batch = []
        
        for batch_idx, (x_batch, y_batch) in enumerate(loader):
            if batch_idx >= n_batches_for_norm_estimate:
                break

            x_norms_per_batch.append(x_batch.norm(dim=-1).mean().item())
            y_norms_per_batch.append(y_batch.norm(dim=-1).mean().item())

        x_mean_norm = np.mean(x_norms_per_batch)
        x_scaling_factor = np.sqrt(self.cfg["A_in"]) / x_mean_norm

        y_mean_norm = np.mean(y_norms_per_batch)
        y_scaling_factor = np.sqrt(self.cfg["B_in"]) / y_mean_norm

        return x_scaling_factor, y_scaling_factor
