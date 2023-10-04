import yaml
import os
import copy
from tqdm import tqdm

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return os.path.normpath(path)

template = {
    "general": {
        "seed": 42,
        "space": "extended"
        },
    "optimizations": {
        "smbo": {
            "budget_kind": "time",
            "budget": 1800
        }},
    "diversifications": {
        "mmr": {
            "num_results": 3,
            "method": "mmr",
            "lambda": 0.7,
            "criterion": "features_set",
            "metric": "jaccard"
        },
        "exhaustive": {
            "num_results": 3,
            "method": "mmr",
            "lambda": 0.7,
            "criterion": "features_set",
            "metric": "jaccard"
        }},
    "runs": ["smbo_mmr"]
}

if __name__ == "__main__":
    input_path = make_dir(os.path.join("/", "home", "scenarios"))
    with tqdm(total=20) as pbar:
        for dataset in range(20):
            for metric in ["sil-tsne", "sil-pca"]:
                task_template = copy.deepcopy(template)
                task_template["general"]["dataset"] = f"syn{dataset}"
                task_template["optimizations"]["smbo"]["metric"] = metric
                with open(os.path.join(input_path, f"syn{dataset}_{metric}.yaml"), "w") as f:
                    yaml.dump(task_template, f)
            pbar.update()