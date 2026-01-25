import plotly.graph_objects as go
import numpy as np

categories = ['模型能力', '上下文能力', 'agent能力', '训练范式完备度', '性能能力', '硬件/成本降低能力']

# 多组数据示例
baba = [10, 10, 10, 10, 10, 10]  # 实体A的能力值
oai = [7, 6, 8, 9, 5, 7]  # 实体B的能力值
seed = [8, 7, 9, 5, 6, 8]  # 实体C的能力值
deepseek = [9, 9, 8, 7, 9, 8]  # 实体D的能力值  
glm = [6, 7, 6, 8, 7, 6]  # 实体E的能力值
tencent = [8, 7, 8, 9, 7, 8]  # 实体G的能力值   
meta = [9, 8, 9, 8, 9, 9]  # 实体H的能力值
anthropic = [7, 6, 7, 8, 6, 7]  # 实体I的能力值
ks_rd = [7, 8, 7, 6, 8, 7]  # 实体F的能力值

# 确保数据闭合
def prepare_data(values):
    return values + values[:1]

categories_closed = categories + categories[:1]

fig = go.Figure()

# 添加第一组数据
fig.add_trace(go.Scatterpolar(
    r=prepare_data(baba),
    theta=categories_closed,
    fill='toself',
    name='快手',
    line=dict(color='blue', width=2),
    fillcolor='rgba(0, 115, 230, 0.3)'
))

# 添加第二组数据
##fig.add_trace(go.Scatterpolar(
#    r=prepare_data(values_2),
#    theta=categories_closed,
#    fill='toself',
#    name='实体B',
#   line=dict(color='red', width=2),
#    fillcolor='rgba(230, 0, 0, 0.3)'
#))

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 10],
            tickvals=list(range(0, 11, 2))
        )
    ),
    showlegend=True,
    title='能力对比图',
    title_x=0.5,
    width=800,
    height=600
)

# 保存HTML文件
fig.write_html("能力对比图.html")
