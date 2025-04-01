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
    return b * (nh_q * s_q * hd_qk * 2 + nh_kv * s_kv * hd_qk + 0 * nh_kv * s_kv * hd_v) * ele_size

def ops_per_attn(b, nh_q, s_q, s_kv, hd_qk, hd_v):
    return b * nh_q * (s_q * s_kv * hd_qk + s_q * s_kv * hd_v) * 2

def comm_time_ms(bs, h, ep, topk, ib_bw, ele_size):
    return ele_size * topk * bs * h / ep / ib_bw

def shared_expert_time_ms(bs, h, dp, e, num_shared, tflops, hbm_bw):
    comp_ms = (ops_per_gemm(1, bs / dp, e * 2 * num_shared, h) + ops_per_gemm(num_shared, bs / dp, h * num_shared, e)) / tflops
    mem_ms = (mem_acc_size_per_gemm(1, bs / dp, e * 2 * num_shared, h, 1) + mem_acc_size_per_gemm(num_shared, bs / dp, h * num_shared, e, 1)) / hbm_bw
    return max(comp_ms, mem_ms)

def mla0_time_ms(bs, h, h_q, h_c, h_d, h_dr, nh, tflops, hbm_bw):
    comp_ms = (ops_per_gemm(1, bs, h_q + h_c + h_dr, h) + 
               ops_per_gemm(1, bs, nh * (h_d + h_dr), h_q) + 
               ops_per_gemm(nh, bs, h_c, h_d)) / tflops
    mem_ms = (mem_acc_size_per_gemm(1, bs, h_q + h_c + h_dr, h, 1) + 
              mem_acc_size_per_gemm(1, bs, nh * (h_d + h_dr), h_q, 1) + 
              mem_acc_size_per_gemm(nh, bs, h_c, h_d, 1)) / hbm_bw
    return max(comp_ms, mem_ms)

def mla1_time_ms(bs, context_len, nh, h_d, h_dr, h_c, h, tflops, hbm_bw):
    attn_comp_ms = ops_per_attn(bs, nh, 1, context_len, h_c + h_dr, h_c) / tflops
    #print(f"attn_comp_ms={attn_comp_ms}")
    attn_mem_ms = mem_acc_size_per_attn(bs, nh, 1, 1, context_len, h_c + h_dr, h_c, 2) / hbm_bw
    #print(f"attn_mem_ms={attn_mem_ms}")
    absorption_gemm_comp_ms = ops_per_gemm(nh, bs, h_c, h_d) / tflops
    absorption_gemm_mem_ms = mem_acc_size_per_gemm(nh, bs, h_c, h_d, 1) / hbm_bw
    o_proj_gemm_comp_ms = ops_per_gemm(1, bs, nh * h_d, h) / tflops
    o_proj_gemm_mem_ms = mem_acc_size_per_gemm(1, bs, nh * h_d, h, 1) / hbm_bw
    return max(attn_comp_ms, attn_mem_ms) + max(absorption_gemm_comp_ms, absorption_gemm_mem_ms) + max(o_proj_gemm_comp_ms, o_proj_gemm_mem_ms)

def moe_time_ms(bs, h, e, topk, ep, num_experts, tflops, hbm_bw):
    up_gemm_comp_ms = ops_per_gemm(1, topk * bs // ep, e * 2, h) / tflops
    up_gemm_mem_ms = mem_acc_size_per_grouped_gemm(topk, num_experts // ep, bs // ep, e * 2, h, 1) / hbm_bw
    #print(f"up_gemm_comp_ms={up_gemm_comp_ms}, up_gemm_mem_ms={up_gemm_mem_ms}")
    down_gemm_comp_ms = ops_per_gemm(1, topk * bs // ep, h, e) / tflops
    down_gemm_mem_ms = mem_acc_size_per_grouped_gemm(topk, num_experts // ep, bs // ep, h, e, 1) / hbm_bw
    return max(up_gemm_comp_ms, up_gemm_mem_ms) + max(down_gemm_comp_ms, down_gemm_mem_ms)

def plot_roofline(bsz, perfs, passing_points, hws):
    # 创建图形
    plt.figure(figsize=(10, 6))
    # plt.loglog(OI, performance, 'b-', linewidth=2, label='Roofline')+

    colors = ['green', 'red', 'blue', 'black', 'gray']

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
    plt.xlabel('tokens', fontsize=12)
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
    hbm_size: float
    peak_flops: float
    memory_bandwidth: float
    nvl_bandwidth: float
    nvl_num: int
    ib_bandwidth: float
    node_price: float
    electricity_price: float
    #styles: dict

    def __post_init__(self):
        self.critical_OI = self.peak_flops / self.memory_bandwidth
        self.total_price_per_hour = (self.node_price + self.electricity_price) / 4 / 8760 / 8
        print(f"{self.hw_name}: {self.total_price_per_hour}")

hws = [
    HW_DATA("GB200", 192 * 1024 ** 3, 5000e12, 8000e9, 900e9, 8, 900e9, 1, 1), 
    HW_DATA("MI308x", 192 * 1024 ** 3, 464e12, 5300e9, 448e9, 8, 50e9, 1, 1), 
    HW_DATA("4090-48GB", 48 * 1024 ** 3, 660e12, 1000e9, 32e9, 8, 32e9, 1, 1), 
    HW_DATA("5090-96GB", 96 * 1024 ** 3, 840e12, 1800e9, 64e9, 8, 50e9, 1, 1), 
    HW_DATA("H20", 96 * 1024 ** 3, 296e12, 4000e9, 450e9, 8, 50e9, 1, 1), 
    HW_DATA("H20-141GB", 141 * 1024 ** 3, 296e12, 4000e9, 450e9, 8, 50e9, 1, 1), 
    HW_DATA("H800-SGM", 80 * 1024 ** 3, 1900e12, 3300e9, 200e9, 8, 50e9, 1, 1),
    HW_DATA("H200", 141 * 1024 ** 3, 1900e12, 4800e9, 450e9, 8, 50e9, 1, 1),
    HW_DATA("910B", 64 * 1024 ** 3, 750e12, 1600e9, 196e9, 8, 50e9, 1, 1),
    HW_DATA("910C", 128 * 1024 ** 3, 1500e12, 3200e9, 392e9, 384, 50e9, 1, 1),
    ]

