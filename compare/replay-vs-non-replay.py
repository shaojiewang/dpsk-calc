import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo

def extract_metrics_from_file(file_path):
    """从文件中提取三个目标指标"""
    rollout_probs_diff_max = []
    rollout_probs_diff_mean = []
    rollout_probs_diff_std = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 使用正则表达式提取三个指标的值
            max_match = re.search(r'training/rollout_probs_diff_max:([0-9.eE+-]+)', line)
            mean_match = re.search(r'training/rollout_probs_diff_mean:([0-9.eE+-]+)', line)
            std_match = re.search(r'training/rollout_probs_diff_std:([0-9.eE+-]+)', line)
            
            if max_match and mean_match and std_match:
                rollout_probs_diff_max.append(float(max_match.group(1)))
                rollout_probs_diff_mean.append(float(mean_match.group(1)))
                rollout_probs_diff_std.append(float(std_match.group(1)))
    
    return {
        'rollout_probs_diff_max': rollout_probs_diff_max,
        'rollout_probs_diff_mean': rollout_probs_diff_mean,
        'rollout_probs_diff_std': rollout_probs_diff_std
    }

def create_comparison_plot(file1_data, file2_data, file1_name="文件1", file2_name="文件2"):
    """创建两个文件三个指标的对比图表"""
    
    # 创建子图，3行1列
    fig = make_subplots(
        rows=3, 
        cols=1,
        subplot_titles=(
            'Rollout Probs Diff Max 对比',
            'Rollout Probs Diff Mean 对比', 
            'Rollout Probs Diff Std 对比'
        ),
        vertical_spacing=0.1
    )
    
    # 获取步数（x轴）
    steps = list(range(1, len(file1_data['rollout_probs_diff_max']) + 1))
    
    # 添加 Rollout Probs Diff Max 数据
    fig.add_trace(
        go.Scatter(
            x=steps, 
            y=file1_data['rollout_probs_diff_max'],
            mode='lines+markers',
            name=f'{file1_name} - Max',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=steps, 
            y=file2_data['rollout_probs_diff_max'],
            mode='lines+markers', 
            name=f'{file2_name} - Max',
            line=dict(color='red')
        ),
        row=1, col=1
    )
    
    # 添加 Rollout Probs Diff Mean 数据
    fig.add_trace(
        go.Scatter(
            x=steps, 
            y=file1_data['rollout_probs_diff_mean'],
            mode='lines+markers',
            name=f'{file1_name} - Mean',
            line=dict(color='blue'),
            showlegend=False
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=steps, 
            y=file2_data['rollout_probs_diff_mean'],
            mode='lines+markers',
            name=f'{file2_name} - Mean', 
            line=dict(color='red'),
            showlegend=False
        ),
        row=2, col=1
    )
    
    # 添加 Rollout Probs Diff Std 数据
    fig.add_trace(
        go.Scatter(
            x=steps, 
            y=file1_data['rollout_probs_diff_std'],
            mode='lines+markers',
            name=f'{file1_name} - Std',
            line=dict(color='blue'),
            showlegend=False
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=steps, 
            y=file2_data['rollout_probs_diff_std'],
            mode='lines+markers',
            name=f'{file2_name} - Std',
            line=dict(color='red'),
            showlegend=False
        ),
        row=3, col=1
    )
    
    # 更新布局
    fig.update_layout(
        title='Training Rollout Probs Diff 指标对比',
        height=900,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    # 更新y轴标题
    fig.update_yaxes(title_text="Max Value", row=1, col=1)
    fig.update_yaxes(title_text="Mean Value", row=2, col=1)
    fig.update_yaxes(title_text="Std Value", row=3, col=1)
    fig.update_xaxes(title_text="Step", row=3, col=1)
    
    return fig

def main():
    # 文件路径 - 请修改为你的实际文件路径
    file1_path = "qwen3-30b-4-layer-replay-cp1.log"  # 替换为第一个文件路径
    file2_path = "qwen3-30b-4-layer-non-replay.log"  # 替换为第二个文件路径
    
    # 从两个文件中提取数据
    print("正在从文件中提取数据...")
    file1_data = extract_metrics_from_file(file1_path)
    file2_data = extract_metrics_from_file(file2_path)
    
    print(f"文件1提取到 {len(file1_data['rollout_probs_diff_max'])} 个数据点")
    print(f"文件2提取到 {len(file2_data['rollout_probs_diff_max'])} 个数据点")
    
    # 创建对比图表
    print("正在生成对比图表...")
    fig = create_comparison_plot(
        file1_data, 
        file2_data, 
        file1_name="routing replay",  # 可以修改这些名称
        file2_name="non routing replay"  # 可以修改这些名称
    )
    
    # 保存为HTML文件
    output_file = "rollout_probs_comparison.html"
    pyo.plot(fig, filename=output_file, auto_open=True)
    print(f"图表已保存为: {output_file}")
    
    # 打印一些统计信息
    print("\n统计信息:")
    for metric in ['rollout_probs_diff_max', 'rollout_probs_diff_mean', 'rollout_probs_diff_std']:
        print(f"\n{metric}:")
        print(f"  文件1 - 最后值: {file1_data[metric][-1]:.6f}, 平均值: {sum(file1_data[metric])/len(file1_data[metric]):.6f}")
        print(f"  文件2 - 最后值: {file2_data[metric][-1]:.6f}, 平均值: {sum(file2_data[metric])/len(file2_data[metric]):.6f}")

if __name__ == "__main__":
    main()