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
    params = [1800, 3600, 5400, 7200]
    datasets = [
            "avila",
            "diabetes",
            "isolet",
            "postures",
            "blood",
            "ecoli",
            "parkinsons",
            "seeds",
            "breast",
            "iris",
            "pendigits",
            "statlog",
            "synthetic",
            "wine",
            "thyroid",
            "vehicle",
        ] + [ f"syn{idx}" for idx in range(20)]
    opt_metrics = [
        "optimization_internal_metric_value",
        "optimization_external_metric_value",
    ]

    results = pd.DataFrame()
    for approach in approaches:
        for param in params:
            for dataset in datasets:
                try:
                    results = pd.concat([
                        results,
                        pd.concat([
                            pd.DataFrame({
                                "approach": [approach]*3,
                                "time": [param]*3,
                            }),
                            pd.read_csv(
                                os.path.join(
                                    input_path,
                                    dataset,
                                    "3",
                                    f"{approach}_0-5_{get_metric(approach)}_{param}.csv",
                                )
                            )
                        ], axis=1)
                    ])
                except Exception as e:
                    print(e)

    output_path = make_dir(
        os.path.join("/", "home", "results", "diversification", "summary")
    )
    results["optimization_internal_metric_value"] *= -1
    for metric in opt_metrics:
        results[metric] = round(
            1 - results[metric], 2
        )
    results = results.reset_index()
    print(results)
    results.to_csv(os.path.join(output_path, "raw.csv"))
    grouped = results.groupby(by=["dataset", "approach", "time"]).max()
    grouped.to_csv(os.path.join(output_path, "max_raw.csv"))
    grouped[opt_metrics].to_csv(os.path.join(output_path, "max.csv"))

    for time in params:
        new_grouped = results[results["time"] == time].groupby(by=["dataset", "approach", "time"]).max().reset_index()
        new_grouped.to_csv(os.path.join(output_path, f"max_{time}_raw.csv"))
        new_grouped[["dataset"] + opt_metrics].to_csv(os.path.join(output_path, f"max_{time}.csv"))

    grouped_filtered = results.groupby(by=["dataset"]).max()
    grouped_filtered.to_csv(os.path.join(output_path, "max_final_raw.csv"))
    grouped_filtered[opt_metrics].to_csv(os.path.join(output_path, "max_final.csv"))


    # for key, agg in {"max": grouped.max(), "avg": grouped.mean()}.items():
    #     support = (
    #         agg.index.to_series()
    #         .str.rsplit("n")
    #         .str[-1]
    #         .astype(int)
    #         .sort_values()
    #     )
    #     agg = agg.reindex(index=support.index)
    #     agg = agg[
    #         [
    #             "optimization_internal_metric_value",
    #             "optimization_external_metric_value",
    #         ]
    #     ]
    #     agg.to_csv(os.path.join(output_path, f"{key}.csv"))


if __name__ == "__main__":
    main()
