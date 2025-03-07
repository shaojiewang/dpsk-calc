import numpy as np
from prettytable import PrettyTable
from utils import *

# 设置模型参数
hidden_dim = 7168
h_q = 1536
h_c = 512
n_h = 128
h_d = 128
h_d_r = 64
tp = 1
expert_hidden_dim = 2048
num_experts = 256
topk = 8
num_shared_experts = 1
num_layers = 61

context_lens = [4096, 5120, 6144, 8192, 16384, 32768, 65536, 102400]

eps = [8, 16, 32, 64, 128]

table = PrettyTable()

model_stable_size = 14.11 * 1024 ** 3
model_dist_size = 609 * 1024 ** 3
kvcache_size_per_token = 70272

table.field_names = ["ai chip name", "context length", "ep", "max_batch_size", "min_time_s_per_batch", "decode_tps", "decode_tps_per_node", "decode_tps_per_card"]


for context_len in context_lens:
    for ep in eps:
        for hw in hws:
            comm_bw = hw.nvl_bandwidth if ep <= hw.nvl_num else hw.ib_bandwidth
            max_bsz = int((hw.hbm_size - model_stable_size - model_dist_size / ep) / (kvcache_size_per_token * context_len)) * ep
            time_ms_mla0_s_dispatch = max(
                comm_time_ms(max_bsz, hidden_dim, ep, topk, comm_bw, 1),
                shared_expert_time_ms(max_bsz, hidden_dim, ep, expert_hidden_dim, num_shared_experts, hw.peak_flops, hw.memory_bandwidth) + 
                mla0_time_ms(max_bsz / ep, hidden_dim, h_q, h_c, h_d, h_d_r, n_h, hw.peak_flops, hw.memory_bandwidth)
            )
            print(time_ms_mla0_s_dispatch)
            time_ms_moe = moe_time_ms(max_bsz, hidden_dim, expert_hidden_dim, topk, ep, num_experts, hw.peak_flops, hw.memory_bandwidth)
            print(time_ms_moe)
            time_ms_mla1_combine = max(
                comm_time_ms(max_bsz, hidden_dim, ep, topk, comm_bw, 2),
                mla1_time_ms(max_bsz / ep, context_len, n_h, h_d, h_d_r, h_c, hidden_dim, hw.peak_flops / 2, hw.memory_bandwidth)
            )
            print(time_ms_mla1_combine)
            time_ms = (time_ms_mla0_s_dispatch + time_ms_moe + time_ms_mla1_combine) * num_layers
            tokens_total = 1 / time_ms * max_bsz
            tokens_per_node = tokens_total / max(1, (ep / hw.nvl_num))
            tokens_per_card = tokens_total / ep
            table.add_row([hw.hw_name, context_len, ep, max_bsz, f"{time_ms:.3f}", int(tokens_total), int(tokens_per_node), int(tokens_per_card)])

table.title = "performance comparison"
print(table)

