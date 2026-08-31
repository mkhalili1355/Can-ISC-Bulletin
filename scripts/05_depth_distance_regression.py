"""Depth against nearest-station distance for the ISC free-depth solutions.

Reads the origin table written by scripts/03_depth_flags.py and fits reported
depth to the base-10 logarithm of nearest-station distance, first on its own
and then with year of occurrence added, so that the gradient quoted in the
manuscript can be checked directly.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), ".."))

from src import VERSION, stats


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True,
                        help="directory holding parsed_isf_origins_full.csv")
    parser.add_argument("--output", default=None, help="output directory")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    return parser.parse_args()


def load_config(path):
    if path is None:
        here = os.path.dirname(os.path.abspath(sys.argv[0]))
        path = os.path.join(here, "..", "config.yaml")
    with open(path) as handle:
        return yaml.safe_load(handle)


def free_depth_subset(frame, last_year, km_per_degree):
    subset = frame[(frame["agency"] == "ISC")
                   & (frame["year"] <= last_year)
                   & (frame["depth_flag"].astype(str).str.strip() == "free")].copy()
    subset = subset[subset["depth_km"].notna() & subset["mdist_deg"].notna()]
    subset["mdist_km"] = subset["mdist_deg"] * km_per_degree
    return subset[subset["mdist_km"] > 0.0]


def report(fit, names):
    return [
        {
            "term": name,
            "coefficient": float(fit["coefficients"][i]),
            "standard_error": float(fit["standard_errors"][i]),
            "p_value": float(fit["p_values"][i]),
            "ci_low": float(fit["ci_low"][i]),
            "ci_high": float(fit["ci_high"][i]),
        }
        for i, name in enumerate(names)
    ]


def main():
    args = parse_args()
    cfg = load_config(args.config)
    out_dir = args.output or args.input

    frame = pd.read_csv(os.path.join(args.input, "parsed_isf_origins_full.csv"))
    subset = free_depth_subset(frame,
                               cfg["study_period"]["primary_last_year"],
                               cfg["physics"]["km_per_degree"])

    depth = subset["depth_km"].to_numpy(float)
    distance = subset["mdist_km"].to_numpy(float)
    log_distance = np.log10(distance)
    year = subset["year"].to_numpy(float)

    rho, p_value = stats.spearmanr(distance, depth)
    plain = stats.ols(depth, [log_distance])
    with_year = stats.ols(depth, [log_distance, year - year.mean()])

    result = {
        "analysis_code_version": VERSION,
        "n_events": int(subset.shape[0]),
        "median_depth_km": float(np.median(depth)),
        "median_nearest_station_km": float(np.median(distance)),
        "spearman": {"rho": float(rho), "p_value": float(p_value)},
        "depth_on_log_distance": report(plain, ["intercept", "log10_distance"]),
        "depth_on_log_distance_and_year": report(
            with_year, ["intercept", "log10_distance", "year_centered"]),
    }

    os.makedirs(out_dir, exist_ok=True)
    target = os.path.join(out_dir, "depth_distance_regression.json")
    with open(target, "w") as handle:
        json.dump(result, handle, indent=2)

    slope = result["depth_on_log_distance"][1]
    print("free-depth ISC solutions : %d" % result["n_events"])
    print("Spearman rho             : %+.3f  (p = %.4f)" % (rho, p_value))
    print("depth per decade of distance : %.1f km, 95%% CI %.1f to %.1f km, p = %.4f"
          % (slope["coefficient"], slope["ci_low"], slope["ci_high"], slope["p_value"]))
    adjusted = result["depth_on_log_distance_and_year"][1]
    print("with year included           : %.1f km, 95%% CI %.1f to %.1f km, p = %.4f"
          % (adjusted["coefficient"], adjusted["ci_low"], adjusted["ci_high"],
             adjusted["p_value"]))
    print("Saved %s" % target)


if __name__ == "__main__":
    main()
