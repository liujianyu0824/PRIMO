# !/usr/bin/env python3
"""
==== No Bugs in code, just some Random Unexpected FEATURES ====
┌─────────────────────────────────────────────────────────────┐
│┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐│
││Esc│!1 │@2 │#3 │$4 │%5 │^6 │&7 │*8 │(9 │)0 │_- │+= │|\ │`~ ││
│├───┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴───┤│
││ Tab │ Q │ W │ E │ R │ T │ Y │ U │ I │ O │ P │{[ │}] │ BS  ││
│├─────┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴─────┤│
││ Ctrl │ A │ S │ D │ F │ G │ H │ J │ K │ L │: ;│" '│ Enter  ││
│├──────┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴────┬───┤│
││ Shift  │ Z │ X │ C │ V │ B │ N │ M │< ,│> .│? /│Shift │Fn ││
│└─────┬──┴┬──┴──┬┴───┴───┴───┴───┴───┴──┬┴───┴┬──┴┬─────┴───┘│
│      │Fn │ Alt │         Space         │ Alt │Win│   HHKB   │
│      └───┴─────┴───────────────────────┴─────┴───┘          │
└─────────────────────────────────────────────────────────────┘

调用本地利用PPO训练好的GPT模型。

Author: pankeyu
Date: 2023/2/21
"""
import torch

from transformers import AutoTokenizer
from trl.gpt2 import GPT2HeadWithValueModel

model_path = 'C:/Users/86183/Desktop/RLHF/checkpoints/generate/'                  # 模型存放地址
device = torch.device("cpu")
G_model = GPT2HeadWithValueModel.from_pretrained(model_path).to(device)
G_tokenizer = AutoTokenizer.from_pretrained(model_path)
G_tokenizer.eos_token = G_tokenizer.pad_token

E_model = GPT2HeadWithValueModel.from_pretrained('C:/Users/86183/Desktop/RLHF/checkpoints/extract/').to(device)
E_tokenizer = AutoTokenizer.from_pretrained('C:/Users/86183/Desktop/RLHF/checkpoints/extract/')
E_tokenizer.eos_token = E_tokenizer.pad_token

gen_len = 256
gen_kwargs = {
    "min_length":-1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": G_tokenizer.eos_token_id
}


def inference(prompt: str, model, tokenizer):
    """
    根据prompt生成内容。

    Args:
        prompt (str): _description_
    """
    inputs = tokenizer(prompt, return_tensors='pt')
    response = model.generate(inputs['input_ids'].to(device),
                                max_new_tokens=gen_len, **gen_kwargs)
    r = response.squeeze()[-gen_len:]
    return tokenizer.decode(r)


if __name__ == '__main__':
    from rich import print

    gen_times = 1

    prompt = 'If A is a Human Language,B is a Film,and A is the language of B,then what other relationships can we derive between A and B?'
    print(f'prompt: {prompt}')
    
    # for i in range(gen_times):                                          # 对同一个 prompt 连续生成 10 次答案
    res = inference(prompt,G_model,G_tokenizer)
    print(f'res : ', res)

    E_prompt = "If A is a Human Language,B is a Film,and A is the language of B,then which relationships between A and B can we extract from the passage'{}'? You only need to output relations without other information,where the relation must be enclosed in brackets, such as (A is lover of B).".format(res)

    res = inference(E_prompt, E_model, E_tokenizer)
    print(f'E_res : ', res)