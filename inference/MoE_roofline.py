import numpy as np
import matplotlib.pyplot as plt
from si_prefix import si_format

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


# 设置硬件参数
peak_flops = 1300e12       # 1 TFLOP/s
memory_bandwidth = 4800e9  # 200 GB/s

# 设置模型参数
hidden_dim = 7168
h_q = 1536
h_c = 512
n_h = 128
h_d = 128
h_d_r = 64
tp = 8
expert_hidden_dim = 2048
num_experts = 256
topk = 8
num_shared_experts = 1


max_bsz = 10000

bsz = np.linspace(1, max_bsz, max_bsz)
#real_OI = mla_compute_intensity(bsz, hidden_dim, h_q, h_c, n_h, h_d, h_d_r, tp)
real_OI = moe_compute_intensity(bsz, hidden_dim, expert_hidden_dim, tp, topk, num_experts, num_shared_experts)
print(real_OI)

points = [tuple(item) for item in zip(bsz, real_OI)]

# 计算临界运算强度
critical_OI = peak_flops / memory_bandwidth  # FLOP/byte

passing_bsz = 1
for item in points:
    if item[1] > critical_OI:
        passing_bsz = item[0]
        break

points = [
    (passing_bsz, peak_flops)  # 临界点
]

# 生成运算强度范围（对数坐标）
# OI = np.logspace(-1, 2, 500)  # 从0.1到100 FLOP/byte
OI = np.linspace(0.1, 100, 50)  # 从0.1到100 FLOP/byte

# 计算理论性能
# performance = np.minimum(memory_bandwidth * OI, peak_flops)
performance = np.minimum(memory_bandwidth * real_OI, peak_flops)

# 创建图形
plt.figure(figsize=(10, 6))
# plt.loglog(OI, performance, 'b-', linewidth=2, label='Roofline')
plt.plot(bsz, performance, 'b-', linewidth=2, label='Roofline')

# 添加特征点标注
colors = ['purple']
labels = ['Critical Point']
for idx, (x, y) in enumerate(points):
    plt.scatter(x, y, s=80, marker='X', 
                edgecolors=colors[idx], 
                facecolors='none',
                linewidths=1.5,
                label=labels[idx])

# 添加标注线
plt.axhline(peak_flops, color='r', linestyle='--', linewidth=1, label='Peak FLOPS')
plt.axvline(critical_OI, color='g', linestyle='--', linewidth=1, label='Critical OI')

# 设置坐标轴标签
plt.xlabel('batch size (bsz)', fontsize=12)
plt.ylabel('Performance (FLOP/s)', fontsize=12)
plt.title('Roofline Model', fontsize=14)

# 使用si-prefix格式化坐标轴
def si_formatter(value, _):
    return f"{si_format(value, precision=1)}FLOP/s"

plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(si_formatter))

# 对x轴进行特殊处理（FLOP/byte单位）
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(
    lambda x, _: f"{si_format(x, precision=1)} "
))

# 添加图例和网格
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)

# 显示图形
plt.tight_layout()
plt.show()