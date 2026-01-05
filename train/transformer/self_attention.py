# https://transformers.run/c1/attention/
import os, torch
torch.set_grad_enabled(False)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch.nn.functional as F
import numpy as np
np.set_printoptions(precision=4, suppress=True)

from math import sqrt
from torch import nn
from transformers import AutoConfig
from transformers import AutoTokenizer


model_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

text = "time flies like an arrow"
inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
print(inputs.input_ids)

config = AutoConfig.from_pretrained(model_ckpt)
token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
print(token_emb)

inputs_embeds = token_emb(inputs.input_ids)
print(inputs_embeds.size())

def scaled_dot_product_attention(query, key, value, query_mask=None, key_mask=None, mask=None):
    dim_k = query.size(-1)
    scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
    if query_mask is not None and key_mask is not None:
        mask = torch.bmm(query_mask.unsqueeze(-1), key_mask.unsqueeze(1))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -float("inf"))
    weights = F.softmax(scores, dim=-1)
    print(weights.size(), '\n', weights.numpy())
    return torch.bmm(weights, value)

if 0:
    Q = K = V = inputs_embeds
    dim_k = K.size(-1)
    scores = torch.bmm(Q, K.transpose(1,2)) / sqrt(dim_k)
    weights = F.softmax(scores, dim=-1)
    attn_outputs = torch.bmm(weights, V)
    print(attn_outputs.shape)
    print(weights.size(), attn_outputs.shape, '\n', weights.numpy())
if 0:
    print(inputs_embeds.shape, inputs_embeds.numpy())
    attn_outputs = scaled_dot_product_attention(inputs_embeds, inputs_embeds, inputs_embeds)
    print(attn_outputs.shape, attn_outputs.numpy())

