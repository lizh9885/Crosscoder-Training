import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from tqdm import tqdm

# class PairedActivationDataset(Dataset):
#     def __init__(self, x_dir, y_dir, samples_per_file, max_files=None):
#         """
#         Args:
#             x_dir: 存放X分片文件的目录
#             y_dir: 存放Y分片文件的目录
#             max_files: 最大加载文件数（用于调试）
#         """
#         self.x_dir = x_dir
#         self.y_dir = y_dir
#         self.samples_per_file = samples_per_file
#         # 获取所有分片文件并按序号排序
#         self.x_files = sorted([f for f in os.listdir(x_dir) if f.endswith('.pt')], 
#                              key=lambda x: int(x.split('_')[-1].split('.')[0]))
#         self.y_files = sorted([f for f in os.listdir(y_dir) if f.endswith('.pt')], 
#                              key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
#         if max_files:
#             self.x_files = self.x_files[:max_files]
#             self.y_files = self.y_files[:max_files]
        
#         # 预计算总数据量
#         self.total_samples = len(self.x_files) * samples_per_file  # 假设每个文件100万条
        
#         # 当前加载的文件索引和缓存
#         self.current_file_idx = -1
#         self.x_cache = None
#         self.y_cache = None

#     def __len__(self):
#         return self.total_samples

#     def _load_file(self, file_idx):
#         """按需加载单个分片文件"""
#         if file_idx != self.current_file_idx:
#             x_path = os.path.join(self.x_dir, self.x_files[file_idx])
#             y_path = os.path.join(self.y_dir, self.y_files[file_idx])
            
#             self.x_cache = torch.load(x_path)  # (1M, 4096)
#             self.y_cache = torch.load(y_path)  # (1M, 3084)
#             self.current_file_idx = file_idx

#     def __getitem__(self, idx):
#         """计算应访问的文件和在文件中的偏移量"""
#         file_idx = idx // self.samples_per_file
#         in_file_idx = idx % self.samples_per_file
        
#         self._load_file(file_idx)
        
#         x = self.x_cache[in_file_idx]  # (4096,)
#         y = self.y_cache[in_file_idx]  # (3084,)
        
#         return x, y
class PairedActivationDataset(Dataset):
    def __init__(self, x_dir, y_dir, samples_per_file=50000, max_files=None):
        self.x_dir = x_dir
        self.y_dir = y_dir
        self.samples_per_file = samples_per_file

        self.x_files = sorted([f for f in os.listdir(x_dir) if f.endswith('.pt')],
                              key=lambda x: int(x.split('_')[-1].split('.')[0]))
        self.y_files = sorted([f for f in os.listdir(y_dir) if f.endswith('.pt')],
                              key=lambda x: int(x.split('_')[-1].split('.')[0]))

        if max_files:
            self.x_files = self.x_files[:max_files]
            self.y_files = self.y_files[:max_files]

        assert len(self.x_files) == len(self.y_files), "Mismatch between X and Y files"

        self.total_samples = len(self.x_files) * samples_per_file

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        file_idx = idx // self.samples_per_file
        sample_idx = idx % self.samples_per_file

        x_path = os.path.join(self.x_dir, self.x_files[file_idx])
        y_path = os.path.join(self.y_dir, self.y_files[file_idx])

        # 加载一个样本，不缓存整块
        x_tensor = torch.load(x_path, mmap=True)[sample_idx]
        y_tensor = torch.load(y_path, mmap=True)[sample_idx]

        return x_tensor, y_tensor


def get_dataloader(x_dir, y_dir, batch_size=1024, num_workers=4, max_files=None):
    dataset = LargeAlignmentDataset(x_dir, y_dir, max_files)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,  # 如需全局shuffle需自定义采样器
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    return loader

if __name__ == "__main__":
    x_dir = "/path/to/X_pt_files"
    y_dir = "/path/to/Y_pt_files"
    
    loader = get_dataloader(x_dir, y_dir, batch_size=2048, max_files=10)  # 测试时限制文件数
    
    for x_batch, y_batch in tqdm(loader):
        # x_batch: (2048, 4096), y_batch: (2048, 3084)
        pass