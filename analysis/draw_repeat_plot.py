import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  # 添加numpy库


line_colors = {}
def style():
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    import matplotlib
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

    
    plt.rcParams['font.size'] = 16
    plt.rcParams['lines.linewidth'] = 1.5
    
    plt.rcParams['axes.titlesize'] = 24
    plt.rcParams['axes.labelsize'] = 24
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18
    plt.rcParams['legend.fontsize'] = 18
    

def time_weighted_ema(x, y, alpha=0.3):
    """
    计算时间加权的指数移动平均
    x: 时间点（如步骤）
    y: 要平滑的值
    alpha: 平滑系数，较小的值会使曲线更平滑
    """
    smoothed = np.zeros_like(y)
    smoothed[0] = y[0]
    
    for i in range(1, len(y)):
        # 计算时间差
        time_diff = x[i] - x[i-1]
        # 动态调整alpha基于时间差
        alpha_t = 1 - (1 - alpha) ** time_diff
        smoothed[i] = alpha_t * y[i] + (1 - alpha_t) * smoothed[i-1]
    
    return smoothed

if __name__ == "__main__":
    style()

    # 读取CSV文件
    df = pd.read_csv('/home/ph16/TinyZero-dev/figs/rep_penalty.csv')  # 替换为你的CSV文件路径

    # 平滑参数
    use_smoothing = True  # 设置为True启用平滑
    alpha = 0.3  # 平滑系数

    # 获取数据
    steps = df["Step"]
    tomap_data = df["debate/Qwen2.5-3B-Instruct-v10-ToMAP - itemized_rewards/repetition_penalty"]
    ToMAP_lite_data = df["debate/Qwen2.5-3B-Instruct-v10-ToMAP_lite - itemized_rewards/repetition_penalty"]
    base_data = df["debate/Qwen2.5-3B-Instruct-v10-base - itemized_rewards/repetition_penalty"]
    
    # 绘图
    plt.figure(figsize=(12, 4))

    if use_smoothing:
        # 先绘制原始数据作为背景，使用较高透明度
        plt.plot(steps, tomap_data, color=line_colors["red"], linewidth=3, alpha=0.4)
        plt.plot(steps, ToMAP_lite_data, color=line_colors["blue"], linewidth=3, alpha=0.4)
        plt.plot(steps, base_data, color=line_colors["green"], linewidth=3, alpha=0.4)
        
        # 应用平滑算法
        tomap_data_smoothed = time_weighted_ema(steps, tomap_data, alpha)
        ToMAP_lite_data_smoothed = time_weighted_ema(steps, ToMAP_lite_data, alpha)
        base_data_smoothed = time_weighted_ema(steps, base_data, alpha)
        
        # 然后绘制平滑后的数据作为前景
        plt.plot(steps, tomap_data_smoothed, label="ToMAP", color=line_colors["red"], linewidth=4)
        plt.plot(steps, ToMAP_lite_data_smoothed, label="ToMAP (w/o att)", color=line_colors["blue"], linewidth=4)
        plt.plot(steps, base_data_smoothed, label="RL", color=line_colors["green"], linewidth=4)
        
        plt.xticks(np.arange(0, 201, 50))
    else:
        assert False
        plt.plot(steps, tomap_data, label="ToMAP", color=line_colors["red"], linewidth=2)
        plt.plot(steps, ToMAP_lite_data, label="ToMAP-lite", color=line_colors["blue"], linewidth=2)
        plt.plot(steps, base_data, label="Base", color=line_colors["green"], linewidth=2)

    # 添加图例、标题和标签
    plt.xlabel("Step")
    plt.ylabel("Repetition Penalty")
    plt.legend()
    plt.xlim(-2, 202)
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=2)

    # 显示图形
    plt.tight_layout()
    output_filename = "figs/repetition_penalty.png"
    plt.savefig(output_filename, dpi=300)
    plt.savefig(output_filename.replace('.png', '.pdf'), bbox_inches='tight')
