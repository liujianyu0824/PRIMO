# !/usr/bin/env python3
import torch
from rich import print
from transformers import AutoTokenizer

device = 'cpu'
tokenizer = AutoTokenizer.from_pretrained('./checkpoints/rewardmodel_best/')
model = torch.load('./checkpoints/rewardmodel_best/model.pt')
model.to(device).eval()

texts = [
    'If A is a Restaurant,B is a Hotel,and A is the restaurants_in_hotel of B, we can get A is the restaurants_in_hotel of B',
    "If A is a Restaurant,B is a Hotel,and A is the restaurants_in_hotel of B, we can get B benefits from having A's restaurant services within their premises as it adds value to their hotel experience."
]
inputs = tokenizer(
    texts, 
    max_length=128,
    padding='max_length', 
    return_tensors='pt'
)
r = model(**inputs)
print(r)
