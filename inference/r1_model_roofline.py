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

context_lens = [4096, 5120, 6144, 8192, 16384, 65536]
best_tgs_lists = []
best_price_lists = []
for _ in hws:
    l = []
    p = []
    for idx, _ in enumerate(context_lens):
        l.append(0)
        p.append(0)
    best_tgs_lists.append(l)
    best_price_lists.append(p)

eps = [8, 16, 32, 64, 128]

table = PrettyTable()

model_stable_size = 14.11 * 1024 ** 3
model_dist_size = 609 * 1024 ** 3
kvcache_size_per_token = 70272

table.field_names = ["ai chip name", "context", "ep", "max_bsz", "latency", "tps", "tps/node", "tps/card"]

epsilon = 1e-9
for i, context_len in enumerate(context_lens):
    plt.figure(figsize=(10, 6))
    for j, hw in enumerate(hws):
        tgs = []
        for ep in eps:
            comm_bw = hw.nvl_bandwidth if ep <= hw.nvl_num else hw.ib_bandwidth
            max_bsz = max(0, int((hw.hbm_size - model_stable_size - model_dist_size / ep) / (kvcache_size_per_token * context_len)) * ep)
            time_ms_mla0_s_dispatch = max(
                comm_time_ms(max_bsz, hidden_dim, ep, topk, comm_bw, 1),
                shared_expert_time_ms(max_bsz, hidden_dim, ep, expert_hidden_dim, num_shared_experts, hw.peak_flops, hw.memory_bandwidth) + 
                mla0_time_ms(max_bsz / ep, hidden_dim, h_q, h_c, h_d, h_d_r, n_h, hw.peak_flops, hw.memory_bandwidth)
            )
            #print(time_ms_mla0_s_dispatch)
            time_ms_moe = moe_time_ms(max_bsz, hidden_dim, expert_hidden_dim, topk, ep, num_experts, hw.peak_flops, hw.memory_bandwidth)
            #print(time_ms_moe)
            time_ms_mla1_combine = max(
                comm_time_ms(max_bsz, hidden_dim, ep, topk, comm_bw, 2),
                mla1_time_ms(max_bsz / ep, context_len, n_h, h_d, h_d_r, h_c, hidden_dim, hw.peak_flops / 2, hw.memory_bandwidth)
            )
            #print(time_ms_mla1_combine)
            if max_bsz == 0:
                time_ms = epsilon
            else:
                time_ms = (time_ms_mla0_s_dispatch + time_ms_moe + time_ms_mla1_combine) * num_layers
            tokens_total = 1 / time_ms * max_bsz / 1000
            tokens_per_node = tokens_total / max(1, (ep / hw.nvl_num))
            tokens_per_card = tokens_total / ep
            table.add_row([hw.hw_name, context_len, ep, max_bsz, f"{time_ms:.3f}", f"{tokens_total:.1f}k", f'{tokens_per_node:.1f}k', f'{tokens_per_card:.2f}k'])
            tgs.append(tokens_per_card)
            best_tgs_lists[j][i] = max(best_tgs_lists[j][i], tokens_per_card)
            

        plt.plot(
            eps,
            tgs,
            marker="o",         # 数据点标记
            linewidth=2,        # 线宽
            label=hw.hw_name        # 图例标签（使用列名）
        )

        best_price_lists[j][i] = hw.total_price_per_hour / (3.6 * best_tgs_lists[j][i])

    # 添加标题和标签
    plt.title(f"context len: {context_len}", fontsize=14)
    plt.xlabel("ep size", fontsize=12)
    plt.ylabel("kilo-tokens per card", fontsize=12)

    # 添加图例和网格
    plt.legend(loc="lower right")  # 图例位置
    plt.grid(True, linestyle="--", alpha=0.6)

# 显示图表
# plt.show()

plt.figure(figsize=(10, 6))
for i, hw in enumerate(hws):
    plt.plot(
        context_lens,
        best_tgs_lists[i],
        marker="o",         # 数据点标记
        linewidth=2,        # 线宽
        label=hw.hw_name        # 图例标签（使用列名）
    )

# 添加标题和标签
plt.title("best perf", fontsize=14)
plt.xlabel("context len", fontsize=12)
plt.ylabel("kilo-tokens per card", fontsize=12)

# 添加图例和网格
plt.legend(loc="upper right")  # 图例位置
plt.grid(True, linestyle="--", alpha=0.6)

plt.figure(figsize=(10, 6))
for i, hw in enumerate(hws):
    plt.plot(
        context_lens,
        best_price_lists[i],
        marker="o",         # 数据点标记
        linewidth=2,        # 线宽
        label=hw.hw_name        # 图例标签（使用列名）
    )

# 添加标题和标签
plt.title("best cost/M tokens", fontsize=14)
plt.xlabel("context len", fontsize=12)
plt.ylabel("dollars/Mtokens", fontsize=12)

# 添加图例和网格
plt.legend(loc="lower right")  # 图例位置
plt.grid(True, linestyle="--", alpha=0.6)

plt.show()


table.title = "deepseek r1 671B decode performance comparison"
print(table)

print(best_price_lists)

