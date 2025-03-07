import numpy as np
from utils import *

def mla_compute_intensity(b, hidden_dim, h_q, h_c, n_h, h_d, h_dr, tp):
    gemm_0_ops = ops_per_gemm(1, b, hidden_dim, h_q + h_c)
    gemm_0_mem = mem_acc_size_per_gemm(1, b, hidden_dim, h_q + h_c, 1)
    gemm_1_ops = ops_per_gemm(1, b, h_q + h_c, n_h // tp * (h_d + h_dr))
    gemm_1_mem = mem_acc_size_per_gemm(1, b, h_q + h_c, n_h // tp * (h_d + h_dr), 1)
    gemm_2_ops = ops_per_gemm(16, b, h_d, h_c)
    gemm_2_mem = mem_acc_size_per_gemm(16, b, h_d, h_c, 2)
    gemm_3_ops = ops_per_gemm(1, b, h_c, h_dr)
    gemm_3_mem = mem_acc_size_per_gemm(1, b, h_c, h_dr, 1)
    gemm_4_ops = ops_per_gemm(16, b, h_c, h_d)
    gemm_4_mem = mem_acc_size_per_gemm(16, b, h_c, h_d, 2)
    gemm_5_ops = ops_per_gemm(1, b, n_h // tp * h_d, hidden_dim)
    gemm_5_mem = mem_acc_size_per_gemm(1, b, n_h // tp * h_d, hidden_dim, 1)
    return (gemm_0_ops + gemm_1_ops + gemm_2_ops + gemm_3_ops + gemm_4_ops + gemm_5_ops) / (gemm_0_mem + gemm_1_mem + gemm_2_mem + gemm_3_mem + gemm_4_mem + gemm_5_mem)


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

max_bsz = 1000

bsz = np.linspace(1, max_bsz, max_bsz)
real_OI = mla_compute_intensity(bsz, hidden_dim, h_q, h_c, n_h, h_d, h_d_r, tp)
# print(real_OI)

points = [tuple(item) for item in zip(bsz, real_OI)]

# 生成运算强度范围（对数坐标）
# OI = np.logspace(-1, 2, 500)  # 从0.1到100 FLOP/byte
# OI = np.linspace(0.1, 100, 50)  # 从0.1到100 FLOP/byte

# 计算理论性能
# performance = np.minimum(memory_bandwidth * OI, peak_flops)
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
