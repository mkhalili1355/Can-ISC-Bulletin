#!/usr/bin/env python
"""
Step 1 of the analysis pipeline. Read the ISC arrival exports, compute every quantity the paper
reports, and write them to results.json together with a provenance manifest.

Usage
-----
    python scripts/01_run_analysis.py --input DATA_DIR --output OUT_DIR

DATA_DIR must contain the ISC arrival exports (files whose names contain
"arr_" and end in .txt). Nothing is written to DATA_DIR; the inputs are
opened read-only.
"""

import argparse
import json
import os
import platform
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), ".."))

import numpy as np
import pandas as pd
import yaml

from src import VERSION, analysis, isc_io


def parse_args():
    here = os.path.dirname(os.path.abspath(sys.argv[0]))
    p = argparse.ArgumentParser(description="Run the Fars Arc depth-resolution analysis and write results.json.",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True,
                   help="directory holding the ISC arrival exports")
    p.add_argument("--output", default=os.path.join(here, "..", "output"),
                   help="directory for results.json and the derived tables")
    p.add_argument("--config", default=os.path.join(here, "..", "config.yaml"))
    return p.parse_args()


def environment_record():
    """Recorded so that a reader can reproduce the exact numerical result."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.system(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "analysis_code_version": VERSION,
    }


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    with open(args.config, "r") as fh:
        cfg = yaml.safe_load(fh)
    np.random.seed(cfg["reproducibility"]["random_seed"])

    print("reading ISC arrival exports from %s ..." % args.input)
    arrivals, manifest = isc_io.load_catalog(args.input)
    print("  %d arrival rows, %d events, %d input files"
          % (len(arrivals), arrivals["EVENTID"].nunique(), len(manifest)))
    empty = manifest[manifest["status"] != "ok"]
    for _, row in empty.iterrows():
        print("  WARNING: %s is empty and contributes nothing" % row["file"])

    events_all = analysis.build_event_table(arrivals, cfg)

    # The era split happens HERE, once, before any statistic is computed.
    # Everything below labelled "primary" sees only the reviewed period; the
    # recent, unreviewed years are analyzed separately and never pooled in.
    events, events_recent = analysis.split_eras(events_all, cfg)
    last_primary = cfg["study_period"]["primary_last_year"]
    print("  primary period (reviewed, <= %d): %d events"
          % (last_primary, len(events)))
    print("  recent period  (unreviewed, > %d): %d events"
          % (last_primary, len(events_recent)))

    # The gap-versus-cut-off analysis works from the arrival table, so it needs
    # the same restriction applied to arrivals rather than to events.
    arrivals_primary = arrivals[arrivals["EVENTID"].isin(set(events["EVENTID"]))]
    print("computing ...")

    results = {
        "configuration": cfg,
        "environment": environment_record(),
        "provenance": manifest.to_dict(orient="records"),
        "dataset": {
            "n_arrival_rows": int(len(arrivals_primary)),
            "n_events": int(len(events)),
            "n_distinct_stations": int(arrivals_primary["STA"].nunique()),
            "first_year": int(events["year"].min()),
            "last_year": int(events["year"].max()),
            "events_per_year": {str(int(k)): int(v) for k, v in
                                events["year"].value_counts().sort_index().items()},
            "magnitude": analysis.describe(events["mag"]),
            "n_arrival_rows_all_periods": int(len(arrivals)),
            "n_events_all_periods": int(len(events_all)),
        },
        "depth_quantization": analysis.depth_quantization(events, cfg),
        "agency_confounding": analysis.agency_confounding(events, cfg),
        "gap_versus_distance_cutoff": analysis.gap_versus_distance_cutoff(arrivals_primary, cfg),
        "network_capability": analysis.network_capability(events, cfg),
        "depth_distance_tradeoff": analysis.depth_distance_tradeoff(events, cfg),
        "takeoff_angle_geometry": analysis.takeoff_angle_geometry(cfg),
        "target_area": analysis.target_area(events, cfg),
        "recent_era": analysis.recent_era(events_recent, cfg),
        "era_comparison": analysis.era_comparison(events, events_recent, cfg),
    }

    out_json = os.path.join(args.output, "results.json")
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2, sort_keys=False)
    print("wrote %s" % out_json)

    # The per-event table is the supplementary data file. It contains only
    # quantities already public in the ISC Bulletin plus derived geometry.
    cols = ["EVENTID", "date", "time", "lat", "lon", "depth_km", "mag",
            "mag_type", "agency", "n_stations_all", "n_stations_local",
            "nearest_station_km", "gap_local_deg", "gap_reported_deg"]
    # Both periods are released, with an explicit label, so that a reader can
    # reproduce either the primary analysis or the recent-era section without
    # having to guess where the boundary was drawn.
    derived = events_all[cols].copy()
    derived["period"] = np.where(events_all["year"] <= last_primary,
                                 "primary", "recent")
    derived.to_csv(os.path.join(args.output, "events_derived.csv"),
                   index=False, float_format="%.4f")
    manifest.to_csv(os.path.join(args.output, "input_manifest.csv"), index=False)

    # Station table for the location map (Figure 1). One row per distinct
    # station code actually used in the primary period, with the coordinates
    # as reported in the Bulletin and the number of primary-period arrivals
    # contributed. Written here, in step 1, so that step 2 continues to read
    # only the outputs of step 1 and never re-opens the raw exports.
    sta = (arrivals_primary
           .dropna(subset=["LAT", "LON"])
           .groupby("STA")
           .agg(lat=("LAT", "median"), lon=("LON", "median"),
                n_arrivals=("STA", "size"))
           .reset_index()
           .sort_values("n_arrivals", ascending=False))
    sta.to_csv(os.path.join(args.output, "stations_derived.csv"),
               index=False, float_format="%.4f")
    area = cfg["study_area"]
    inside = sta[(sta["lat"].between(area["lat_min"], area["lat_max"])) &
                 (sta["lon"].between(area["lon_min"], area["lon_max"]))]
    results["network_capability"]["n_stations_inside_study_area"] = int(len(inside))
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2, sort_keys=False)
    print("wrote events_derived.csv, stations_derived.csv and input_manifest.csv")
    print("\nHeadline numbers")
    q = results["depth_quantization"]
    a = results["agency_confounding"]
    n = results["network_capability"]
    print("  integer-valued depths      : %.1f %%" % q["percent_integer_valued"])
    print("  depth exactly 10.0 km      : %.1f %%" % q["operational_values"]["10.0"]["percent"])
    print("  cumulative median %d->%d : %.2f -> %.2f km"
          % (results["dataset"]["first_year"], results["dataset"]["last_year"],
             a["cumulative_median_first_year_km"], a["cumulative_median_last_year_km"]))
    print("  well-constrained events    : %d of %d (%.1f %%)"
          % (n["n_well_constrained"], n["n_events"], n["percent_well_constrained"]))


main()
