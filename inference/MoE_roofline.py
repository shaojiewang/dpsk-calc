import numpy as np

from utils import *

def moe_compute_intensity(b, h, e, tp, topk, num_experts, num_shared_experts):
    shared_gemm_up_ops = ops_per_gemm(num_shared_experts, b, e * 2 / tp, h)
    shared_gemm_down_ops = ops_per_gemm(num_shared_experts, b, h, e / tp)
    routed_gemm_up_ops = ops_per_gemm(topk, b, e * 2 / tp, h)
    routed_gemm_down_ops = ops_per_gemm(topk, b, h, e / tp)
    shared_gemm_up_mem = mem_acc_size_per_gemm(num_shared_experts, b, e * 2 / tp, h, 1)
    shared_gemm_down_mem = mem_acc_size_per_gemm(num_shared_experts, b, h, e / tp, 1)
    routed_gemm_up_mem = mem_acc_size_per_grouped_gemm(topk, num_experts, b, e * 2 / tp, h, 1)
    routed_gemm_down_mem = mem_acc_size_per_grouped_gemm(topk, num_experts, b, h, e / tp, 1)
    return (shared_gemm_up_ops + shared_gemm_down_ops + routed_gemm_up_ops + routed_gemm_down_ops) / (shared_gemm_up_mem + shared_gemm_down_mem + routed_gemm_up_mem + routed_gemm_down_mem)

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


max_bsz = 18000

bsz = np.linspace(1, max_bsz, max_bsz)
#real_OI = mla_compute_intensity(bsz, hidden_dim, h_q, h_c, n_h, h_d, h_d_r, tp)
real_OI = moe_compute_intensity(bsz, hidden_dim, expert_hidden_dim, tp, topk, num_experts, num_shared_experts)
print(real_OI)

points = [tuple(item) for item in zip(bsz, real_OI)]

# 生成运算强度范围（对数坐标）
# OI = np.logspace(-1, 2, 500)  # 从0.1到100 FLOP/byte
# OI = np.linspace(0.1, 100, 50)  # 从0.1到100 FLOP/byte

# 计算理论性能
# performance = np.minimum(memory_bandwidth * OI, peak_flops)
# performance = np.minimum(memory_bandwidth * real_OI, peak_flops)

perfs = []
passing_points = []
for ele in hws:
    performance = np.minimum(ele.memory_bandwidth * real_OI, ele.peak_flops)
    perfs.append(performance)

    passing_bsz = 1
    for item in points:
        if item[1] > ele.critical_OI:
            passing_bsz = item[0]
            break

    passing_points.append((passing_bsz, ele.peak_flops))  # 临界点


plot_roofline(bsz, perfs, passing_points, hws)

draw_table(passing_points, hws)
