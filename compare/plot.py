import matplotlib.pyplot as plt
import numpy as np

# 数据
dates = ['2025.02', '2025.04', '2025.08', '2025.10', '2025.11']
context_length = [8, 32, 128, 256, 256]  # 单位：k
mfu = [2.4, 20, 26, 32, 35]  # 单位：%

# 创建图形和第一个坐标轴
fig, ax1 = plt.subplots(figsize=(14, 9))

# 设置专业商务配色方案
bar_color = "#007BFF"  # 更鲜艳的商务蓝
line_color = '#2ECC71'  # 商务绿色
text_color = '#34495E'  # 深蓝灰
grid_color = '#BDC3C7'  # 灰色
label_color = '#FFFFFF'  # 统一的白色标签颜色

# 设置背景色
fig.patch.set_facecolor('#F8F9FA')
ax1.set_facecolor('#FFFFFF')

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 绘制柱状图（上下文长度）- 使用渐变色
bars = ax1.bar(dates, context_length, color=bar_color, alpha=0.9, width=0.6, 
               edgecolor='white', linewidth=2, label='最长上下文 (k)',
               zorder=2)  # zorder确保柱状图在网格线之上

# 添加渐变效果（让柱状图顶部稍微深一些）
for bar in bars:
    bar.set_zorder(3)
    # 添加渐变
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    gradient = np.vstack((gradient, gradient))

# 设置坐标轴标签和样式
ax1.set_xlabel('时间', fontsize=16, color=text_color, fontweight='bold')
ax1.set_ylabel('最长上下文长度 (k)', fontsize=16, color=bar_color, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=bar_color, labelsize=14)
ax1.tick_params(axis='x', colors=text_color, labelsize=14)

# 设置网格线
ax1.grid(True, axis='y', alpha=0.4, color=grid_color, linestyle='--', linewidth=0.8, zorder=1)

# 自动调整柱状图标签位置
for i, (bar, value) in enumerate(zip(bars, context_length)):
    height = bar.get_height()
    # 所有标签都放在柱状图内部，使用统一的白色
    y_pos = height - (height * 0)  # 放在柱状图80%高度的位置
    text_color_bar = label_color
    fontweight = 'bold'
    # 添加背景框提高可读性
    bbox_props = dict(boxstyle='round,pad=0.3', facecolor=bar_color, edgecolor='white', alpha=0.9)
    
    ax1.text(bar.get_x() + bar.get_width()/2., y_pos,
             f'{value}k', ha='center', va='center', fontsize=16, 
             color=text_color_bar, fontweight='bold',
             bbox=bbox_props)

# 创建第二个坐标轴（用于MFU折线图）
ax2 = ax1.twinx()
line = ax2.plot(dates, mfu, color=line_color, marker='o', linewidth=6.0, 
                markersize=10, label='MFU (%)', zorder=4,
                markerfacecolor='white', markeredgecolor=line_color, markeredgewidth=2)

# 设置第二个坐标轴样式
ax2.set_ylabel('MFU (%)', fontsize=16, color=line_color, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=line_color, labelsize=14)

# 设置MFU的y轴范围，为标签留出空间
ax2.set_ylim(0, max(mfu) * 1.15)

# 自动调整折线图标签位置
for i, (date, value) in enumerate(zip(dates, mfu)):
    # 如果是最后一个点，调整位置避免重叠
    if i == len(mfu) - 1:
        # 使用注释框并添加箭头
        ax2.annotate(f'{value}%', 
                    xy=(date, value), 
                    xytext=(15, 10),  # 向右15，向上10
                    textcoords='offset points',
                    fontsize=18, color=line_color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                             edgecolor=line_color, alpha=0.9),
                    arrowprops=dict(arrowstyle='->', color=line_color, 
                                   linewidth=1.5, alpha=0.7))
    else:
        # 其他点正常显示
        ax2.text(date, value + 1.5, f'{value}%', 
                ha='center', va='bottom', fontsize=18, color=line_color,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                         edgecolor=line_color, alpha=0.8))

# 设置标题
plt.title('上下文长度与MFU随时间变化趋势', fontsize=18, pad=20, 
          color=text_color, fontweight='bold')

# 合并图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=16,
           frameon=True, fancybox=True, shadow=True, facecolor='white', 
           edgecolor=grid_color)

# 调整布局
plt.tight_layout()

# 添加边框
for spine in ax1.spines.values():
    spine.set_edgecolor(grid_color)
    spine.set_linewidth(1.5)

for spine in ax2.spines.values():
    spine.set_edgecolor(grid_color)
    spine.set_linewidth(1.5)

# 保存图片
output_path = 'context_length_mfu_trend_professional.png'
fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"专业版图表已保存为 '{output_path}'")

# 显示图形
plt.show()