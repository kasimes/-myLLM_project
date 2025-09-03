

# from transformers import AutoTokenizer, AutoModelForCausalLM
# tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
# model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it")n

import json

import torch
from ders_kodları.kasim_tokenizer import Tokenizer
from torch.utils.data import DataLoader
from text_dataset import TextDataset

stride = 1

def create_dataloader(token_ids:list,
                       context_length:int, 
                       batch_size:int,
                       stride:int,
                       shuffle:bool=True,
                      device:str="cpu"):
    dataset = TextDataset(token_ids, context_length, stride)
    dataloader = DataLoader(dataset,
                             batch_size=batch_size, 
                             shuffle=shuffle,
                             generator=torch.Generator(device=device))
    return dataloader




# JSON dosyasından tokenizer yükle
tokenizer = Tokenizer.from_file("hf_tokenizer.json")

prompt = "yapay zeka "
token_ids = tokenizer.encode(prompt)
 

# print("Token ID'leri:", token_ids)
train_dataloader = create_dataloader(token_ids=token_ids.ids,
                                      context_length=10,
                                      stride=1,
                                      batch_size=2)

print(len(train_dataloader))