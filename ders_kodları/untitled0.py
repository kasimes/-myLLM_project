# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 11:42:19 2025

@author: kasim
"""

import numpy as np
import tiktoken  as tik 

text1="The cat chased the dog "

vocab={
       "the" :0,
       "cat" :1,
       "dog" :2,
       "<unk>" :8
       
    
       
       }

def tokenization (text):
    parts =  text.split()
    ids=[]
    
    for part in parts:
        if part in vocab:
            value = vocab[part]
        else:
            value = vocab["<unk>"]
        ids.append(value)
        
    return ids
    
    
    
token_ids= tokenization(text1)

reverse_vocab = {id : part for part , id in vocab.items()}

reverse_vocab 

def reverse_tokenization(ids):
 
    text= ""
    
    for id in ids:
        part = reverse_vocab[id]
        text += part + " "
    text = text.strip()
        
    return text      

result=reverse_tokenization(token_ids)

enc =  tik.get_encoding("o200k_base")
enc.n_vocab



from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("dbmdz/bert-base-turkish-cased")

text = "Merhaba, nasılsın bugün?"

# Tokenize + encode
enc = processor(
    text,
    return_tensors="pt",     # PyTorch tensörleri
    padding=True,
    truncation=True
)






    
    