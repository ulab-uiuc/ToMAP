import json
import os
import matplotlib.pyplot as plt
import numpy as np

def visualize_strategy_comparison(data_file_1="figs/strategy_data.json", 
                                   data_file_2="figs/strategy_data_tomap.json", 
                                   output_image="figs/strategy_comparison.png"):
    """
    Compare strategy usage percentages between two data files and plot a grouped bar chart.
    """
    # Shorter names for display
    strategy_name_map = {
        "Evidential Appeals": "Evidence",
        "Authority Appeals": "Authority",
        "Emotional Appeals": "Emotional",
        "Social Appeals": "Social",
        "Common Ground Appeals": "CommonGround",
        "Gradual Concession": "Concession",
        "Framing Effects": "Framing",
        "Rhetoric": "Rhetoric",
        "Preemptive Rebuttal": "Preemptive"
    }

    # Helper function to load and process data
    def load_strategy_percentages(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        all_data = data["strategies"]["all"]
        return {strategy: pct for strategy, _, pct in all_data}

    # Load both datasets
    data1 = load_strategy_percentages(data_file_1)
    data2 = load_strategy_percentages(data_file_2)

    # Get complete list of strategies
    all_strategies = sorted(set(strategy_name_map.keys()).union(data1.keys(), data2.keys()))

    # Extract percentages or 0 if missing
    values1 = [data1.get(strategy, 0) for strategy in all_strategies]
    values2 = [data2.get(strategy, 0) for strategy in all_strategies]

    # Short display names
    labels = [strategy_name_map.get(s, s) for s in all_strategies]

    # Plot
    x = np.arange(len(all_strategies))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 4))
    bars1 = ax.bar(x - width/2, values1, width, label='Base Model', color=line_colors["light_purple"])
    bars2 = ax.bar(x + width/2, values2, width, label='ToMAP', color=line_colors["purple"])

    # Add labels and title
    ax.set_ylabel('Strategy Usage (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25)
    ax.legend()
    ax.set_ylim(0, 80)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)

    # Add value labels
    def add_labels(bars, offset=3):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, offset),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=11)


    plt.tight_layout()
    os.makedirs(os.path.dirname(output_image), exist_ok=True)
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    plt.savefig(output_image.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Comparison visualization saved to {output_image}")
    return output_image

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
    plt.rcParams['font.family'] = font_prop.get_name()

    
    plt.rcParams['font.size'] = 24
    plt.rcParams['lines.linewidth'] = 1.5
    
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['legend.fontsize'] = 14


if __name__ == "__main__":
    # Example usage
    style()
    visualize_strategy_comparison()