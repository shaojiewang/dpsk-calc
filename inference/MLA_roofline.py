import numpy as np
import matplotlib.pyplot as plt
from si_prefix import si_format

# 设置硬件参数
peak_flops = 232e12       # 1 TFLOP/s
memory_bandwidth = 5300e9  # 200 GB/s

# 设置模型参数
hidden_dim = 7168
h_q = 1536
h_c = 512
n_h = 128
h_d = 128
h_d_r = 64
tp = 8

def mem_acc_size_per_gemm(b, m, n, k, ele_size):
    return ele_size * (b * m * n + b * m * k + b * n * k)

def ops_per_gemm(b, m, n, k):
    return b * m * n * k * 2

def mem_acc_size_per_attn(b, nh_q, nh_kv, s_q, s_kv, hd_qk, hd_v, ele_size):
    return b * (nh_q * s_q * hd_qk * 2 + nh_kv * s_kv * hd_qk + nh_kv * s_kv * hd_v) * ele_size

def ops_per_attn(b, nh_q, s_q, s_kv, hd_qk, hd_v):
    return b * nh_q * (s_q * s_kv * hd_qk + s_kv * s_kv * hd_v) * 2

# def compute_intensity(b, hidden_dim, h_q, h_c, n_h, h_d, h_dr, tp, ):


# memory_access_per_batch = 

# 计算临界运算强度
critical_OI = peak_flops / memory_bandwidth  # FLOP/byte

points = [
    (2, 400e9),    # 内存受限区示例点
    (8, 1e12),     # 计算受限区示例点
    (critical_OI, peak_flops)  # 临界点
]

# 生成运算强度范围（对数坐标）
# OI = np.logspace(-1, 2, 500)  # 从0.1到100 FLOP/byte
OI = np.linspace(0.1, 100, 50)  # 从0.1到100 FLOP/byte

# 计算理论性能
performance = np.minimum(memory_bandwidth * OI, peak_flops)

# 创建图形
plt.figure(figsize=(10, 6))
# plt.loglog(OI, performance, 'b-', linewidth=2, label='Roofline')
plt.plot(OI, performance, 'b-', linewidth=2, label='Roofline')

# 添加特征点标注
colors = ['red', 'blue', 'purple']
labels = ['Memory-Bound Example', 'Compute-Bound Example', 'Critical Point']
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
plt.xlabel('Operational Intensity (FLOP/byte)', fontsize=12)
plt.ylabel('Performance (FLOP/s)', fontsize=12)
plt.title('Roofline Model', fontsize=14)

# 使用si-prefix格式化坐标轴
def si_formatter(value, _):
    return f"{si_format(value, precision=1)}FLOP/s"

plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(si_formatter))

# 对x轴进行特殊处理（FLOP/byte单位）
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(
    lambda x, _: f"{si_format(x, precision=1)} FLOP/byte"
))

# 添加图例和网格
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)

# 显示图形
plt.tight_layout()
plt.show()