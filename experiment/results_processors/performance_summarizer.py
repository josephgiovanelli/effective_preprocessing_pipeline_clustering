import os

import numpy as np
import pandas as pd

import traceback


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return os.path.normpath(path)


def main():

    input_path = os.path.join("/", "home", "results", "diversification", "smbo", "mmr")

    approaches = ["clustering"]
    get_metric = lambda approach: "ami" if approach == "clustering" else "jaccard"
    params = [5]
    datasets = [f"syn{i}" for i in range(20)]
    opt_metrics = [
        "optimization_internal_metric_value",
        "optimization_external_metric_value",
    ]

    results = {}
    for approach in approaches:
        for param in params:
            results[f"{approach}_{param}"] = pd.DataFrame()
            for dataset in datasets:
                try:
                    results[f"{approach}_{param}"] = pd.concat(
                        [
                            results[f"{approach}_{param}"],
                            pd.read_csv(
                                os.path.join(
                                    input_path,
                                    dataset,
                                    "3",
                                    f"{approach}_0-{param}_{get_metric(approach)}.csv",
                                )
                            ),
                        ]
                    )
                except Exception as e:
                    print(
                        f"""MyException: {e}
                        {traceback.print_exc()}"""
                    )

            output_path = make_dir(
                os.path.join(input_path, "summary", f"{approach}_{param}")
            )
            results[f"{approach}_{param}"]["optimization_internal_metric_value"] *= -1
            for metric in opt_metrics:
                results[f"{approach}_{param}"][metric] = round(
                    1 - results[f"{approach}_{param}"][metric], 2
                )

            results[f"{approach}_{param}"].to_csv(os.path.join(output_path, "raw.csv"))
            grouped = results[f"{approach}_{param}"].groupby(by=["dataset"])
            for key, agg in {"max": grouped.max(), "avg": grouped.mean()}.items():
                support = (
                    agg.index.to_series()
                    .str.rsplit("n")
                    .str[-1]
                    .astype(int)
                    .sort_values()
                )
                agg = agg.reindex(index=support.index)
                agg = agg[
                    [
                        "optimization_internal_metric_value",
                        "optimization_external_metric_value",
                    ]
                ]
                agg.to_csv(os.path.join(output_path, f"{key}.csv"))


if __name__ == "__main__":
    main()
