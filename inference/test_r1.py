import numpy as np
from prettytable import PrettyTable
import json
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

context_lens = [750, 1024, 4096, 5120, 6144, 8192, 10 * 1024, 12 * 1024, 16384, 32*1024, 64 * 1024, 100 * 1024]
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

with open('./inference/prod-data-6-5-deepseek-r1-reasoning.json', "r", encoding="utf-8") as f:
    datas = json.load(f)

sum_input = 0
sum_output = 0
max_input = 0
max_output = 0
for data in datas:
    sum_input += data["prompt_tokens"]
    sum_output += data["completion_tokens"]
    max_input = max(max_input, data["prompt_tokens"])
    max_output = max(max_output, data["completion_tokens"])
print(f"{sum_input=}")
print(f"{sum_output=}")
avg_input = sum_input / len(datas)
avg_output = sum_output / len(datas)
print(f"{avg_input=}")
print(f"{avg_output=}")

print(f"{max_input=}")
print(f"{max_output=}")

print(f"{len(datas)}")