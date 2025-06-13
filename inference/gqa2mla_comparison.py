import numpy as np
from prettytable import PrettyTable
import plotly
from utils import *

# for 40B GQA model
head_dim = 128
q_heads = 40
kv_heads = 8
hidden_dim = head_dim * q_heads
kv_hidden_dim = head_dim * kv_heads
num_layers = 80
ffn_dim = 27648

context_lens = [4096]

tp_size = [2]

kvcache_size_per_token = num_layers * kv_heads * head_dim * 2 * 2
model_stable_size = 0
model_dist_size = 40 * 2 * (10 ** 9)

epsilon = 1e-9
for i, context_len in enumerate(context_lens):
    for j, hw in enumerate(hws):
        tgs = []
        for tp in tp_size:
            activation_size = max(2 * context_len * ffn_dim * 6, 2 * context_len * (hidden_dim + kv_hidden_dim) * 2)
            max_bsz = max(0, int((hw.hbm_size - activation_size - model_stable_size - model_dist_size / tp) / (kvcache_size_per_token * context_len)))
            print(f"{max_bsz=}")
            time_qkv_projection = \
                gemm_time_ms(max_bsz, (hidden_dim + kv_hidden_dim) / tp, hidden_dim, hw.peak_flops, hw.memory_bandwidth)
            print(f"{time_qkv_projection=}")
            time_core_attn = \
                attn_time_ms(max_bsz, context_len, q_heads, kv_heads, 0, head_dim, hw.peak_flops / 2, hw.memory_bandwidth)
            print(f"{time_core_attn=}")
            time_o_projection = \
                gemm_time_ms(max_bsz, (hidden_dim) / tp, hidden_dim, hw.peak_flops, hw.memory_bandwidth)
            print(f"{time_o_projection=}")
            time_ffn = \
                shared_expert_time_ms(max_bsz, hidden_dim, 1, ffn_dim / tp, 1, hw.peak_flops, hw.memory_bandwidth)
            print(f"{time_ffn=}")
            time_ar = 2 * 2 * max_bsz * hidden_dim / hw.nvl_bandwidth
            print(f"{time_ar=}")
            if max_bsz == 0:
                time_ms = epsilon
            else:
                time_ms = (time_qkv_projection + time_core_attn + time_o_projection + time_ffn + time_ar) * num_layers
            print(f"{time_ms=}")
            tokens_total = 1 / time_ms * max_bsz / 1000
            print(f"{tokens_total=}")
            tokens_per_card = tokens_total / tp
            print("gqa model")
            print(f"{context_len}: {tokens_per_card=}")

#for 40B MLA model

for i, context_len in enumerate(context_lens):
    for j, hw in enumerate(hws):
        tgs = []
        for tp in tp_size:
            time_ar = 2 * max_bsz * hidden_dim / hw.nvl_bandwidth
            activation_size = max(2 * context_len * ffn_dim * 6, 2 * context_len * (hidden_dim + kv_hidden_dim) * 2)
            max_bsz = max(0, int((hw.hbm_size - activation_size - model_stable_size - model_dist_size / tp) / (kvcache_size_per_token * context_len)))
            

