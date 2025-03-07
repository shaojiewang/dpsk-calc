import matplotlib.pyplot as plt
from si_prefix import si_format
from prettytable import PrettyTable


def mem_acc_size_per_gemm(b, m, n, k, ele_size):
    return ele_size * (b * m * n * 0 + b * m * k + b * n * k)

def mem_acc_size_per_grouped_gemm(b_a, b_b, m, n, k, ele_size):
    return ele_size * (b_a * m * n * 0 + b_a * m * k + b_b * n * k)

def ops_per_gemm(b, m, n, k):
    return b * m * n * k * 2

def mem_acc_size_per_attn(b, nh_q, nh_kv, s_q, s_kv, hd_qk, hd_v, ele_size):
    return b * (nh_q * s_q * hd_qk * 2 + nh_kv * s_kv * hd_qk + nh_kv * s_kv * hd_v) * ele_size

def ops_per_attn(b, nh_q, s_q, s_kv, hd_qk, hd_v):
    return b * nh_q * (s_q * s_kv * hd_qk + s_kv * s_kv * hd_v) * 2

def plot_roofline(bsz, perfs, passing_points, hws):
    # 创建图形
    plt.figure(figsize=(10, 6))
    # plt.loglog(OI, performance, 'b-', linewidth=2, label='Roofline')+

    colors = ['green', 'red', 'blue', 'black']

    for idx, perf in enumerate(perfs):
        plt.plot(bsz, perf, '-', color=colors[idx], linewidth=2, label=f'Roofline of {hws[idx].hw_name}')

    # 添加特征点标注
    
    labels = []
    for hw in hws:
        labels.append(f'Critical Point {hw.hw_name}')
    for idx, (x, y) in enumerate(passing_points):
        plt.scatter(x, y, s=80, marker='X', 
                    edgecolors=colors[idx], 
                    facecolors='none',
                    linewidths=1.5,
                    label=labels[idx])

    # 添加标注线
    for hw in hws:
        plt.axhline(hw.peak_flops, color='r', linestyle='--', linewidth=1, label=f'Peak FLOPS of {hw.hw_name}')
        # plt.axvline(hw.critical_OI, color='g', linestyle='--', linewidth=1, label=f'Critical OI of {hw.hw_name}')

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
    plt.legend(loc="lower right")
    plt.grid(True, which="both", ls="--", alpha=0.5)

    # 显示图形
    plt.tight_layout()
    plt.show()

def draw_table(passing_points, hws):
    table = PrettyTable()
    table.field_names = ["ai chip name", "compute bound batch size"]

    # 添加数据
    for idx, points in enumerate(passing_points):
        table.add_row([hws[idx].hw_name, points[0]])

    table.title = "roofline"  # 添加标题

    # 输出
    print(table)

from dataclasses import dataclass

@dataclass
class HW_DATA(object):
    hw_name: str
    peak_flops: float
    memory_bandwidth: float
    nvl_bandwidth: float
    ib_bandwidth: float

    def __post_init__(self):
        self.critical_OI = self.peak_flops / self.memory_bandwidth

hws = [
    HW_DATA("MI308x", 464e12, 5300e9, 896e9, 50e9), 
    HW_DATA("H100-SGM", 1900e12, 3300e9, 900e9, 50e9),
    HW_DATA("H200", 1900e12, 4800e9, 900e9, 50e9),
    ]

