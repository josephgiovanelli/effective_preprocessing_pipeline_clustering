import yaml
import os
import copy
from tqdm import tqdm


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return os.path.normpath(path)


template = {
    "general": {"seed": 42, "space": "extended"},
    "optimizations": {
        "smbo": {"budget_kind": "time", "budget": 1800, "metric": "sil-tsne"}
    },
    "diversifications": {
        "mmr": {
            "num_results": 3,
            "method": "mmr",
            "lambda": 0.5,
            "criterion": "features_set",
            "metric": "jaccard",
        },
        "exhaustive": {
            "num_results": 3,
            "method": "mmr",
            "lambda": 0.5,
            "criterion": "features_set",
            "metric": "jaccard",
        },
    },
    "runs": ["smbo_mmr"],
}

if __name__ == "__main__":
    input_path = make_dir(os.path.join("/", "home", "scenarios"))
    with tqdm(total=20) as pbar:
        for dataset in range(20):
            for div_lambda in [0.5, 0.7]:
                task_template = copy.deepcopy(template)
                task_template["general"]["dataset"] = f"syn{dataset}"
                for div in ["exhaustive", "mmr"]:
                    task_template["diversifications"][div]["lambda"] = div_lambda
                with open(
                    os.path.join(
                        input_path,
                        f"""syn{dataset}_{str(div_lambda).split(".")[1]}.yaml""",
                    ),
                    "w",
                ) as f:
                    yaml.dump(task_template, f)
            pbar.update()
