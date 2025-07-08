from itertools import islice
import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
from tqdm import tqdm
import os
import argparse
import glob
import os
import glob
    
def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="google/gemma-2-2b")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--layer_idx", type=int, default=15)
    parser.add_argument("--save_every", type=int, default=10000)
    parser.add_argument("--gpu_ids", type=str, default="0,1", help="GPU IDs to use, separated by comma")
    return parser.parse_args()

def output_mask(batch_ids, pad_token_id):
    attention_mask = (batch_ids != pad_token_id).long()
    seq_lengths = attention_mask.sum(dim=1)  # 每个样本的有效长度
    last_non_pad_pos = seq_lengths - 1  # 最后一个有效 token 的位置 = length - 1
    return attention_mask, last_non_pad_pos

def load_and_split_data(model_save_name, gpu_rank, num_gpus):
    """加载数据并按GPU数量分割"""
    # 加载input_id
    file1 = f"./input_ids/{model_save_name}/{model_save_name}_pile-uncopyrighted.pt"
    file2 = f"./input_ids/{model_save_name}/{model_save_name}_lmsys-chat-1m.pt"
    
    data1 = torch.load(file1)
    data2 = torch.load(file2)
    combined_data = torch.cat([data1, data2], dim=0)
    
    # 加载index映射文件
    map_file1 = f"./input_ids/{model_save_name}/{model_save_name}_pile-uncopyrighted_map.pt"
    map_file2 = f"./input_ids/{model_save_name}/{model_save_name}_lmsys-chat-1m_map.pt"  # 修正了这里的文件名
    
    data1_list = torch.load(map_file1)
    data2_list = torch.load(map_file2)
    data_list = data1_list + data2_list
    
    # 按GPU数量分割数据
    total_samples = combined_data.shape[0]
    samples_per_gpu = total_samples // num_gpus
    
    start_idx = gpu_rank * samples_per_gpu
    if gpu_rank == num_gpus - 1:  # 最后一个GPU处理剩余的所有数据
        end_idx = total_samples
    else:
        end_idx = (gpu_rank + 1) * samples_per_gpu
    
    gpu_data = combined_data[start_idx:end_idx]
    gpu_data_list = data_list[start_idx:end_idx]
    
    return gpu_data, gpu_data_list

def save_activations_single_gpu(gpu_id, gpu_rank, num_gpus, model_name, layer_idx, save_dir, batch_size, save_every):
    """单个GPU的激活提取函数"""
    # 设置当前GPU
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    
    # 创建保存目录
    os.makedirs(save_dir,exist_ok = True)
    # gpu_save_dir = os.path.join(save_dir, f"gpu_{gpu_id}")
    # os.makedirs(gpu_save_dir, exist_ok=True)
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True,torch_dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pad_id = int(tokenizer.eos_token_id)
    model_save_name = model_name.split("/")[-1]
    
    # 加载并分割数据
    combined_data, data_list = load_and_split_data(model_save_name, gpu_rank, num_gpus)
    
    all_activations = []
    save_iter = 0
    act_count = 0
    
    with torch.no_grad():
        pbar = tqdm(range(0, combined_data.shape[0], batch_size), 
                   desc=f"GPU {gpu_id} Processing batches")
        
        for i in pbar:
            batch_ids = combined_data[i:i+batch_size].to(device)
            index_list = data_list[i:i+batch_size]
            
            mask, last_non_pad_pos = output_mask(batch_ids, pad_id)
            outputs = model(input_ids=batch_ids, attention_mask=mask)  # 修正了参数名
            acts = outputs.hidden_states[layer_idx+1]
            
            selected_acts = []
            for b_idx, indices in enumerate(index_list):
                if torch.is_tensor(indices):
                    indices = indices.to(device)
                token_acts = acts[b_idx, indices]
                selected_acts.append(token_acts)
            
            patched_acts = torch.cat(selected_acts, dim=0)
            all_activations.append(patched_acts.cpu())
            act_count += patched_acts.size()[0]
            
            # 定期保存
            if act_count >= save_every:
                test_final_activations = torch.cat(all_activations, dim=0)
                to_save = test_final_activations[:save_every]
                rest = test_final_activations[save_every:]
                
                save_path = os.path.join(save_dir, f"gpu_{gpu_id}_act_{model_save_name}_iter_{save_iter+1}.pt")
                torch.save(to_save, save_path)
                
                all_activations = [rest] if rest.numel() > 0 else []
                act_count = rest.shape[0]
                save_iter += 1
                
                del test_final_activations
    
    # 保存最后的激活
    # if all_activations:
    #     final_iter_activations = torch.cat(all_activations, dim=0)
    #     save_path = os.path.join(gpu_save_dir, f"act_{model_save_name}_iter_{save_iter+1}.pt")
    #     torch.save(final_iter_activations, save_path)
    #     del final_iter_activations
    
    print(f"GPU {gpu_id} finished processing")

def main():
    args = setup_args()
    
    
    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",")]
    num_gpus = len(gpu_ids)
    
    model_save_name = args.model.split("/")[-1]
    save_dir = f"./act/{model_save_name}/"
    
    mp.set_start_method('spawn', force=True)
    

    processes = []
    
    for gpu_rank, gpu_id in enumerate(gpu_ids):
        p = mp.Process(
            target=save_activations_single_gpu,
            args=(gpu_id, gpu_rank, num_gpus, args.model, args.layer_idx, 
                  save_dir, args.batch_size, args.save_every)
        )
        p.start()
        processes.append(p)
    
    # 等待所有进程完成
    for p in processes:
        p.join()
    
    print("All GPUs finished processing")
    
    # 获取当前目录下所有符合条件的 .pt 文件（按修改时间排序）
    files = sorted(glob.glob(os.path.join(save_dir, f"gpu_*_act_{model_save_name}_iter_*.pt")), key=os.path.getmtime)
    
    
    files.sort(key=lambda x: (int(x.split('_')[1]), int(x.split('_')[-1].split('.')[0])))
    
    # 重命名文件
    for idx, old_name in enumerate(files, start=1):
        new_name = f"{save_dir}act_{model_save_name}_iter_{idx}.pt"
        os.rename(old_name, new_name)
        print(f"Renamed: {old_name} -> {new_name}")
    
    print("All files renamed successfully!")
        

if __name__ == "__main__":
    main()