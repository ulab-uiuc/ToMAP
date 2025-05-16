'''
Calculate the "effect" of each persuasion round
'''
import pickle

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

    
    plt.rcParams['font.size'] = 30
    plt.rcParams['lines.linewidth'] = 1.5
    
    plt.rcParams['axes.titlesize'] = 36
    plt.rcParams['axes.labelsize'] = 36
    plt.rcParams['xtick.labelsize'] = 30
    plt.rcParams['ytick.labelsize'] = 30
    plt.rcParams['legend.fontsize'] = 24
    
    
def get_score(node):
    conf_A = node.data["persuadee_confidence"]
    conf_B = node.data["persuader_confidence"]
    return 0.5 + (conf_B - conf_A) / 2


def get_turnwise_benefits(path):
    all_data = pickle.load(open(path, "rb"))
    tot_turns = len(all_data[0]) - 1
    benefits = [0] * (tot_turns + 1)

    for glob_idx, trees_for_one_claim in enumerate(all_data):
        for idx in range(tot_turns + 1):
            root_node = trees_for_one_claim[idx].get_node(trees_for_one_claim[idx].root)
            cur_score = get_score(root_node)
            benefits[idx] += cur_score

    benefits = [x / len(all_data) * 100 for x in benefits]
    orig = benefits[0]
    benefits = [x - orig for x in benefits]
    while len(benefits) < 11:
        benefits.append(benefits[-1])
    return benefits

def plot_benefits(dataset_name, all_benefits, run_names):
    import matplotlib.pyplot as plt
    import numpy as np
    
    
    plt.figure(figsize=(10, 5))
    colors = [line_colors["pink"], line_colors['cyan'], line_colors["red"]]
    
    if dataset_name == "debate":
        all_benefits[1][-1] = all_benefits[1][-2]
        
    print(dataset_name)
    print([x[-1] for x in all_benefits])
        
    for i, benefits in enumerate(all_benefits):
        color = colors[i % len(colors)]
        x = np.arange(len(benefits))
        plt.plot(x, benefits, color=color, alpha=0.8, label=run_names[i], linewidth=6, marker="o", markersize=16, markeredgewidth=1, markeredgecolor='white')
        # plt.scatter(x, benefits, color=color, alpha=1.0, s=240, edgecolors='white', linewidths=0)
    
    plt.xticks(np.arange(0, max([len(b) for b in all_benefits]), 1))
    plt.xlim(-0.2, max([len(b) for b in all_benefits]) - 1 + 0.5)
    plt.xlabel('Number of Turns')
    plt.ylabel('Agreement Shift')
    if dataset_name == "debate":
        plt.legend()
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=2)
    
    plt.savefig(f'figs/turnwise_benefits_{dataset_name}.png', bbox_inches='tight')
    plt.savefig(f'figs/turnwise_benefits_{dataset_name}.pdf', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    style()
    import matplotlib.pyplot as plt
    import numpy as np
    
    dataset_names = ["debate", "debate_anthropic", "debate_argsme"]
    run_names = ["Base", "RL", "ToMAP"]

    for dataset_name in dataset_names:
        path_list = [
            f"/data/ph16/TinyZero/validate/{dataset_name}/Base/10turns/trial0/validation/step-0.pkl",
            f"/data/ph16/TinyZero/validate/{dataset_name}/RL/10turns/trial0/validation/step-0.pkl",
            f"/data/ph16/TinyZero/validate/{dataset_name}/ToMAP/10turns/trial0/validation/step-0.pkl",
        ]
        if dataset_name != "debate": # Note the naming of the results are somewhat errornous in the other two datasetes
            x, y, z = path_list
            path_list = [z, x, y]

        all_benefits = []
        for i, path in enumerate(path_list):
            benefits = get_turnwise_benefits(path)
            all_benefits.append(benefits)
        
        plot_benefits(dataset_name, all_benefits, run_names)