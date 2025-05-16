import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap  # 添加此导入
import matplotlib

line_colors = {}
def style():
    from matplotlib import font_manager
    global line_colors
    
    line_colors['red'] = "#e3716e"
    
    line_colors["light_grey"] = "#afb0b2"
    line_colors["grey"] = "#656565"
    
    line_colors["green"] = "#c0db82"
    line_colors["yellow_green"] = "#54beaa"
    
    line_colors["pink"] = "#efc0d2"
    
    line_colors["light_purple"] = "#eee5f8"
    line_colors["purple"] = "#af8fd0"
    
    line_colors["blue"] = "#6d8bc3"
    line_colors["cyan"] = "#2983b1"
    
    line_colors["yellow"] = "#f9d580"
    line_colors["orange"] = "#eca680"
    
    line_colors["gradual_yellow"] = "#EABE5D"
    line_colors["gradual_purple"] = "#3F314F"


    font_path = '/data/ph16/fonts/cambria.ttc'  # 你的路径
    font_prop = font_manager.FontProperties(fname=font_path)
    font_manager.fontManager.addfont(font_path)
    matplotlib.rcParams['font.family'] = font_prop.get_name()

    
    plt.rcParams['font.size'] = 28
    plt.rcParams['lines.linewidth'] = 1.5
    
    plt.rcParams['axes.titlesize'] = 28
    plt.rcParams['axes.labelsize'] = 28
    plt.rcParams['xtick.labelsize'] = 24
    plt.rcParams['ytick.labelsize'] = 24
    plt.rcParams['legend.fontsize'] = 10
    
    




def plot_heatmaps_in_row(datasets, titles, output_path="figs/setting_and_model.png", gap=0):
    """
    绘制N个热图子图，并排放置在一行中，使用从黄色到紫色的渐变色
    所有子图使用相同的颜色映射范围，确保相同数值在不同图中显示相同颜色
    
    参数:
    - datasets: 数据集列表，每个元素是一个numpy数组
    - titles: 每个子图的标题列表
    - output_path: 输出图像的保存路径
    - gap: 前三个和后三个子图之间的间距大小（相对值）
    """
    n = len(datasets)
    
    # 根据子图数量动态调整图表宽度，每个子图宽度为5英寸
    fig_width = min(24, max(12, 5 * n))  # 限制最小12，最大24英寸
    
    # 使用GridSpec创建布局
    fig = plt.figure(figsize=(fig_width, 4))
    from matplotlib.gridspec import GridSpec
    
    gs = GridSpec(1, n)
    axes = [fig.add_subplot(gs[0, i]) for i in range(n)]
    axes = np.array(axes)

    # 创建从黄色到紫色的渐变颜色映射
    yellow_to_purple = LinearSegmentedColormap.from_list(
        "yellow_to_purple", 
        [line_colors["light_purple"], line_colors["purple"]], 
        N=256
    )
    
    # 找出所有数据集的全局最小值和最大值
    global_min = min(np.min(data) for data in datasets)
    global_max = max(np.max(data) for data in datasets)
    
    # 绘制每个子图
    for i, (data, title, ax) in enumerate(zip(datasets, titles, axes)):
        # 绘制热图，使用自定义的渐变色彩映射，并设置全局颜色范围
        cax = ax.matshow(data, cmap=yellow_to_purple, aspect="auto", vmin=global_min, vmax=global_max)
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_yticks(np.arange(data.shape[0]))
        
        # 设置标签
        ax.set_xticklabels(['w/o ToM', 'w ToM'])
        ax.set_yticklabels(['Base', 'ToMAP'])
        
        # 将x轴移到底部
        ax.xaxis.set_ticks_position('bottom')
        ax.xaxis.set_label_position('bottom')
        
        # 添加数值标签
        for (j, k), val in np.ndenumerate(data):
            text_color = 'black'
            ax.text(k, j, f'{val:.2f}', ha='center', va='center', color=text_color)
        
        # 设置标题
        ax.set_title(title)
    
    # 添加全局颜色条（可选）
    # fig.colorbar(cax, ax=axes.tolist(), orientation='horizontal', pad=0.12, aspect=40)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
# 示例用法
if __name__ == "__main__":
    style()
    data1 = np.array([[12.75, 13.25],
                      [17.60, 23.01]])
    
    data2 = np.array([[4.97, 1.90],
                      [7.44, 8.82]])
    
    data3 = np.array([[5.66, 5.27],
                      [11.28, 15.82]])
    
    data4 = np.array([[7.09, 5.86],
                      [13.08, 12.10]])
    
    data5 = np.array([[6.90, 6.02],
                      [16.95, 18.75]])
    
    data6 = np.array([[4.09, 4.17],
                      [11.16, 10.89]])
    # 定义多个数据集和标题
    titles = ["CMV/Qwen", "CMV/LLaMa", "Anthropic/Qwen", "Anthropic/Llama", "args.me/Qwen", "args.me/Llama"]
    
    datasets = [data1, data2, data3, data4, data5, data6]
    # 调用函数绘制热图
    plot_heatmaps_in_row(datasets[:6], titles[:6])
    
    # 如果只想使用前两个数据集（原始需求）
    # plot_heatmaps_in_row(datasets[:2], titles[:2])
