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

context_lens = [4096 + 512]

tp_size = [2]

kvcache_size_per_token = num_layers * kv_heads * head_dim * 2 * 2
model_stable_size = 0
model_dist_size = 40 * (10 ** 9)

epsilon = 1e-9
for i, context_len in enumerate(context_lens):
    for j, hw in enumerate(hws):
        tgs = []
        for tp in tp_size:
            activation_size = max(2 * context_len * ffn_dim * 6, 2 * context_len * (hidden_dim + kv_hidden_dim) * 2)
            max_bsz = max(0, int((hw.hbm_size - activation_size - model_stable_size - model_dist_size / tp) / (kvcache_size_per_token / tp * context_len)))
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
print("MLA")
h_q = 1536
h_d_r = 64
h_c = 512
n_h = q_heads
h_d = head_dim
kvcache_size_per_token = 576 * num_layers * 2
model_stable_size = num_layers * (hidden_dim * (h_q + h_d_r + h_c) + h_q * n_h * (h_d_r + h_d) + 2 * h_d * n_h * h_c)
model_dist_size = num_layers * (ffn_dim * hidden_dim * 3)
for i, context_len in enumerate(context_lens):
    for j, hw in enumerate(hws):
        tgs = []
        for tp in tp_size:
            activation_size = max(2 * context_len * ffn_dim * 6, 2 * context_len * (hidden_dim + kv_hidden_dim) * 2)
            max_bsz = max(0, int((hw.hbm_size - activation_size - model_stable_size - model_dist_size / tp) / (kvcache_size_per_token * context_len) * tp))
            print(f"{max_bsz=}")
            time_ffn = \
                shared_expert_time_ms(max_bsz, hidden_dim, 1, ffn_dim / tp, 1, hw.peak_flops, hw.memory_bandwidth)
            print(f"{time_ffn=}")
            time_mla0 = \
                mla0_time_ms(max_bsz / tp, hidden_dim, h_q, h_c, h_d, h_d_r, n_h, hw.peak_flops, hw.memory_bandwidth)
            print(f"{time_mla0=}")
            time_mla1 = \
                mla1_time_ms(max_bsz / tp, context_len, n_h, h_d, h_d_r, h_c, hidden_dim, hw.peak_flops / 2, hw.memory_bandwidth)
            print(f"{time_mla1=}")
            time_ar = 2 * max_bsz * hidden_dim / hw.nvl_bandwidth
            print(f"{time_ar=}")

            if max_bsz == 0:
                time_ms = epsilon
            else:
                time_ms = (time_mla0 + time_mla1 + time_ffn + time_ar) * num_layers
            print(f"{time_ms=}")
            tokens_total = 1 / time_ms * max_bsz / 1000
            print(f"{tokens_total=}")
            tokens_per_card = tokens_total / tp
            print("mla model")
            print(f"{context_len}: {tokens_per_card=}")


