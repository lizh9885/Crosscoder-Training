from itertools import islice
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
from tqdm import tqdm
import random
import os
import argparse
import glob
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial

device = 'cuda:0'

parser = argparse.ArgumentParser()
# parser.add_argument("--modelA_name", type=str, default="meta-llama/Llama-3.1-8B")
parser.add_argument("--modelA_name", type=str, default="google/gemma-2-2b-it")
parser.add_argument("--modelB_name", type=str, default="google/gemma-2-2b")
parser.add_argument("--num_cuts", type=int, default=50)
parser.add_argument("--num_samples", type=int, default=1000)
parser.add_argument("--num_workers", type=int, default=cpu_count())  # 默认使用所有CPU核心
args = parser.parse_args()

splits = ["monology/pile-uncopyrighted", "lmsys/lmsys-chat-1m"]

def find_cut_points(text, num_cuts=args.num_cuts, seed=42, include_full=True):
    num_cuts = num_cuts - 1
    if seed is not None:
        random.seed(seed)
    
    spaces = [i for i, ch in enumerate(text) if ch == " "]
    if not spaces:
        return [len(text)] if include_full else []
    
    num_selected = min(num_cuts, len(spaces))
    cut_points = random.sample(spaces, num_selected)
    
    if include_full:
        cut_points.append(len(text))
    
    cut_points.sort()
    return cut_points

def verify_token_alignment(tokenizer, full_text, prefix_text):
    full_tokens = tokenizer.tokenize(full_text)
    prefix_tokens = tokenizer.tokenize(prefix_text)
    
    if full_tokens[:len(prefix_tokens)] == prefix_tokens:
        return len(prefix_tokens) 
    return None

def process_one_sentence(text, tokenizer_llama, tokenizer_gemma, num_splits=args.num_cuts):
    cut_points = find_cut_points(text, num_splits)
    if not cut_points:
        return None

    result = {"llama_splits": [], "gemma_splits": []}
    for idx in cut_points:
        prefix = text[:idx]
        llama_len = verify_token_alignment(tokenizer_llama, text, prefix)
        gemma_len = verify_token_alignment(tokenizer_gemma, text, prefix)
        if llama_len is not None and gemma_len is not None:
            if llama_len < 1024 and gemma_len < 1024:
                result["llama_splits"].append(llama_len)
                result["gemma_splits"].append(gemma_len)
    
    if result["llama_splits"] and result["gemma_splits"]:
        try:
            assert len(result["llama_splits"]) == len(result["gemma_splits"])
            return result
        except:
            return None
    else:
        return None

def process_batch(batch, tokenizerA, tokenizerB, subset):
    bufferA = []
    bufferB = []
    map_A = []
    map_B = []
    
    for item in batch:
        if "pile" in subset:
            text = item["text"]
        else:
            instruct = item["conversation"][0]["content"]
            completion = item["conversation"][1]["content"]
            text = f"User: {instruct}\nAssistant: {completion}"
        
        cut_result = process_one_sentence(text, tokenizerA, tokenizerB)
        if not cut_result:
            continue
            
        tokensA = tokenizerA(
            text,
            truncation=True,
            max_length=1024,
            padding='max_length',
            add_special_tokens=True,
        )
        tokensB = tokenizerB(
            text,
            truncation=True,
            max_length=1024,
            padding='max_length',
            add_special_tokens=True,
        )
        
        bufferA.append(tokensA["input_ids"])
        bufferB.append(tokensB["input_ids"])
        map_A.append(cut_result["llama_splits"])
        map_B.append(cut_result["gemma_splits"])
    
    return bufferA, bufferB, map_A, map_B

def save_input_ids(modelA_name, modelB_name, subset, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    tokenizerA = AutoTokenizer.from_pretrained(modelA_name)
    tokenizerB = AutoTokenizer.from_pretrained(modelB_name)

    tokenizerA.pad_token = tokenizerA.eos_token
    tokenizerA.padding_side = "right" 
    tokenizerB.pad_token = tokenizerB.eos_token
    tokenizerB.padding_side = "right" 

    sub_dataset = load_dataset(subset, split="train", streaming=True)
    dataset_iter = iter(sub_dataset)
    
    bufferA = []
    bufferB = []
    map_A = []
    map_B = []
    
    # 使用多进程池
    with Pool(args.num_workers) as pool:
        # 分批处理数据
        batch_size = 1000
        batches = []
        current_batch = []
        
        for i, item in enumerate(dataset_iter):
            current_batch.append(item)
            if len(current_batch) >= batch_size:
                batches.append(current_batch)
                current_batch = []
            if i >= args.num_samples:
                break
        
        if current_batch:
            batches.append(current_batch)
        
        # 创建部分函数以便传递tokenizers和subset
        process_func = partial(process_batch, tokenizerA=tokenizerA, tokenizerB=tokenizerB, subset=subset)
        
        # 使用tqdm显示进度
        results = list(tqdm(pool.imap(process_func, batches), total=len(batches), desc=f"Processing {subset}"))
    
    # 合并结果
    for result in results:
        batch_A, batch_B, batch_map_A, batch_map_B = result
        bufferA.extend(batch_A)
        bufferB.extend(batch_B)
        map_A.extend(batch_map_A)
        map_B.extend(batch_map_B)
        
        # 检查是否达到所需样本数
        if len(bufferA) >= args.num_samples:
            bufferA = bufferA[:args.num_samples]
            bufferB = bufferB[:args.num_samples]
            map_A = map_A[:args.num_samples]
            map_B = map_B[:args.num_samples]
            break

    # 保存结果
    tensor_data_A = torch.tensor(bufferA, dtype=torch.long)
    tensor_data_B = torch.tensor(bufferB, dtype=torch.long)
    
    split_name = subset.split("/")[-1]
    modelA_save_name = modelA_name.split("/")[-1]
    modelB_save_name = modelB_name.split("/")[-1]
    
    modelA_save_dir = f"{save_dir}/{modelA_save_name}/"
    modelB_save_dir = f"{save_dir}/{modelB_save_name}/"
    os.makedirs(modelA_save_dir,exist_ok = True)
    os.makedirs(modelB_save_dir,exist_ok = True)
    torch.save(tensor_data_A, os.path.join(modelA_save_dir, f"{modelA_save_name}_{split_name}.pt"))
    print(f"save {modelA_save_name}_{split_name}.pt successful")
    torch.save(tensor_data_B, os.path.join(modelB_save_dir, f"{modelB_save_name}_{split_name}.pt"))
    print(f"save {modelB_save_name}_{split_name}.pt successful")
    
    torch.save(map_A, os.path.join(modelA_save_dir, f"{modelA_save_name}_{split_name}_map.pt"))
    print(f"save {modelA_save_name}_{split_name}_map.pt successful")
    torch.save(map_B, os.path.join(modelB_save_dir, f"{modelB_save_name}_{split_name}_map.pt"))
    print(f"save {modelB_save_name}_{split_name}_map.pt successful")

if __name__ == "__main__":

    save_dir = "./input_ids"
    # os.makedirs(save_dir, exist_ok=True)
    for split in splits:
        save_input_ids(args.modelA_name, args.modelB_name, split, save_dir)