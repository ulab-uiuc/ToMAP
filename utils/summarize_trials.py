import json
import sys
import os

target_dir = sys.argv[1]
output_path = os.path.join(target_dir, "summary.json")

info = {}
trial_index = 0

while True:
    trial_dir = os.path.join(target_dir, f"trial{trial_index}")
    
    if not os.path.exists(trial_dir):
        break
        
    path = os.path.join(trial_dir, "validation", "step-0.json")
    
    with open(path, "r") as f:
        cur_info = json.load(f)["score"]
        
    if info == {}:
        info = cur_info
    else:
        for k in info.keys():
            if k in cur_info:
                info[k] += cur_info[k]
            else:
                info[k] = cur_info[k]
    
    trial_index += 1


if info:
    for k in info.keys():
        info[k] /= trial_index
    with open(output_path, "w") as f:
        json.dump(info, f, indent=4)
    print(f"Saved to {output_path}")
else:
    print("No valid data found.")