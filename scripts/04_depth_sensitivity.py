#!/usr/bin/env python
"""
Formal depth sensitivity of the recording geometry for the four-parameter
hypocenter problem. Reproduces Table S2 of the electronic supplement and the
depth-sensitivity statements in the Discussion.

The estimate is a first-order linearized sensitivity for a straight-ray,
single-layer model with independent, phase-independent reading errors. It is
not a relocated uncertainty; model error, correlated picks and phase-dependent
weights are not represented.

The reference configuration is not idealized. It is the station geometry the
Bulletin reports for each event of the primary period, so the quoted value is
a median over events rather than a property of a chosen ring. Each proposed
design is evaluated by adding its stations to that recorded geometry, event by
event.

Two properties bound the interpretation and are checked by self_check().
sigma_z is invariant under a uniform rotation of all station azimuths, so the
sense in which the reported azimuth column is measured does not affect the
result, and sigma_z scales linearly with the assumed reading uncertainty.

Usage
-----
    python scripts/05_depth_sensitivity.py --input DATA_DIR --output OUT_DIR
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), ".."))

import numpy as np
import pandas as pd
import yaml

from src import VERSION, isc_io


def parse_args():
    here = os.path.dirname(os.path.abspath(sys.argv[0]))
    p = argparse.ArgumentParser(
        description="Formal depth sensitivity of the recorded and proposed station geometries.")
    p.add_argument("--input", required=True,
                   help="directory holding the ISC arrival exports")
    p.add_argument("--output", default=os.path.join(here, "..", "output"))
    p.add_argument("--config", default=os.path.join(here, "..", "config.yaml"))
    return p.parse_args()


def design_matrix(dist_km, azimuth_deg, depth_km, velocity):
    """Partial derivatives of arrival time with respect to (x, y, z, t0)."""
    r = np.asarray(dist_km, dtype=float)
    a = np.deg2rad(np.asarray(azimuth_deg, dtype=float))
    slant = np.hypot(r, float(depth_km))
    slant = np.where(slant < 1.0e-6, 1.0e-6, slant)
    g = np.empty((r.size, 4), dtype=float)
    g[:, 0] = -(r * np.sin(a)) / (velocity * slant)
    g[:, 1] = -(r * np.cos(a)) / (velocity * slant)
    g[:, 2] = float(depth_km) / (velocity * slant)
    g[:, 3] = 1.0
    return g


def sigma_z(dist_km, azimuth_deg, depth_km, velocity, reading_uncertainty,
            n_parameters=4, max_condition=1.0e12):
    """
    Standard deviation of the depth parameter implied by a station geometry.

    Returns nan when the geometry cannot support the requested number of
    parameters, or when the normal matrix is too ill conditioned to invert
    meaningfully.
    """
    g = design_matrix(dist_km, azimuth_deg, depth_km, velocity)
    column = 2
    if n_parameters == 2:
        g = g[:, 2:]
        column = 0
    if g.shape[0] < g.shape[1]:
        return float("nan")
    normal = g.T.dot(g)
    if not np.all(np.isfinite(normal)):
        return float("nan")
    if np.linalg.cond(normal) > max_condition:
        return float("nan")
    variance = reading_uncertainty ** 2 * np.linalg.inv(normal)[column, column]
    if not np.isfinite(variance) or variance <= 0.0:
        return float("nan")
    return float(np.sqrt(variance))


def recorded_geometry(arrivals, cfg):
    """
    One distance and azimuth per station per event, primary period only.

    Repeated phases at one station carry the same geometry and are collapsed,
    so a station counts once however many phases it reported.
    """
    period = cfg["study_period"]
    km_per_deg = cfg["physics"]["km_per_degree"]
    keep = arrivals[(arrivals["year"] >= period["primary_first_year"]) &
                    (arrivals["year"] <= period["primary_last_year"])]
    keep = keep.dropna(subset=["DIST", "BAZ"])
    keep = keep.drop_duplicates(subset=["EVENTID", "STA"])
    return pd.DataFrame({
        "EVENTID": keep["EVENTID"].values,
        "dist_km": keep["DIST"].values.astype(float) * km_per_deg,
        "azimuth_deg": keep["BAZ"].values.astype(float),
    })


def group_by_event(geometry):
    """Distance and azimuth arrays for each event, in a stable event order."""
    geometry = geometry.sort_values("EVENTID", kind="mergesort")
    event_ids = geometry["EVENTID"].values
    distances = geometry["dist_km"].values
    azimuths = geometry["azimuth_deg"].values
    starts = np.flatnonzero(np.r_[True, event_ids[1:] != event_ids[:-1]])
    stops = np.r_[starts[1:], event_ids.size]
    return [(distances[i:j], azimuths[i:j]) for i, j in zip(starts, stops)]


def design_stations(design):
    """Distances and azimuths a design adds, as declared in the configuration."""
    added = design.get("added_stations") or []
    if not added:
        return np.zeros(0), np.zeros(0)
    array = np.asarray(added, dtype=float)
    return array[:, 0], array[:, 1]


def evaluate(events, design, depths, velocity, reading_uncertainty,
             max_condition, n_parameters=4):
    """Median sigma_z over events, for one design and each source depth."""
    add_r, add_a = design_stations(design)
    out = {}
    for depth in depths:
        values = np.empty(len(events), dtype=float)
        for k, (r, a) in enumerate(events):
            if add_r.size:
                r = np.concatenate([r, add_r])
                a = np.concatenate([a, add_a])
            values[k] = sigma_z(r, a, depth, velocity, reading_uncertainty,
                                n_parameters=n_parameters,
                                max_condition=max_condition)
        finite = values[np.isfinite(values)]
        out["%g" % depth] = {
            "n_events": int(finite.size),
            "n_events_undetermined": int(values.size - finite.size),
            "median_km": float(np.median(finite)) if finite.size else None,
            "q1_km": float(np.percentile(finite, 25)) if finite.size else None,
            "q3_km": float(np.percentile(finite, 75)) if finite.size else None,
        }
    return out


def self_check(velocity, reading_uncertainty, max_condition):
    """Verify the two invariances the interpretation relies on."""
    r = np.array([40.0, 75.0, 120.0, 210.0, 300.0])
    a = np.array([12.0, 96.0, 171.0, 250.0, 318.0])
    base = sigma_z(r, a, 11.0, velocity, reading_uncertainty,
                   max_condition=max_condition)
    rotated = sigma_z(r, (a + 180.0) % 360.0, 11.0, velocity,
                      reading_uncertainty, max_condition=max_condition)
    doubled = sigma_z(r, a, 11.0, velocity, 2.0 * reading_uncertainty,
                      max_condition=max_condition)
    return {
        "azimuth_rotation_invariant": bool(abs(base - rotated) < 1.0e-9),
        "azimuth_rotation_difference_km": float(abs(base - rotated)),
        "linear_in_reading_uncertainty": bool(abs(doubled - 2.0 * base) < 1.0e-9),
    }


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    with open(args.config, "r") as fh:
        cfg = yaml.safe_load(fh)

    sens = cfg["sensitivity"]
    velocity = cfg["physics"]["vp_crust_km_s"]
    depths = sens["source_depths_km"]
    sigma_t = sens["reading_uncertainty_s"]
    max_condition = float(sens["max_condition_number"])
    reference_depth = cfg["physics"]["detachment_depth_km"]

    arrivals, _ = isc_io.load_catalog(args.input)
    events = group_by_event(recorded_geometry(arrivals, cfg))
    stations_per_event = np.array([r.size for r, _ in events])

    designs = []
    for design in sens["designs"]:
        add_r, _ = design_stations(design)
        designs.append({
            "name": design["name"],
            "stations_added": int(add_r.size),
            "added_stations": design.get("added_stations") or [],
            "sigma_z": evaluate(events, design, depths, velocity, sigma_t,
                                max_condition),
        })

    recorded = sens["designs"][0]
    two_parameter = evaluate(events, recorded, depths, velocity, sigma_t,
                             max_condition, n_parameters=2)

    scaling = {}
    for value in sens["reading_uncertainty_scaling_s"]:
        row = evaluate(events, recorded, [reference_depth], velocity, value,
                       max_condition)
        scaling["%g" % value] = row["%g" % reference_depth]

    inflation = {}
    for depth in depths:
        key = "%g" % depth
        four = designs[0]["sigma_z"][key]["median_km"]
        two = two_parameter[key]["median_km"]
        if four and two:
            inflation[key] = float(100.0 * (four / two - 1.0))

    results = {
        "analysis_code_version": VERSION,
        "method": ("first-order linearized sensitivity, straight rays, "
                   "uniform half-space, independent reading errors"),
        "velocity_km_s": velocity,
        "reading_uncertainty_s": sigma_t,
        "source_depths_km": depths,
        "n_events": len(events),
        "stations_per_event": {
            "median": float(np.median(stations_per_event)),
            "min": int(stations_per_event.min()),
            "max": int(stations_per_event.max()),
        },
        "designs": designs,
        "two_parameter_recorded_geometry": two_parameter,
        "four_parameter_inflation_percent": inflation,
        "reading_uncertainty_scaling": scaling,
        "self_check": self_check(velocity, sigma_t, max_condition),
    }

    with open(os.path.join(args.output, "depth_sensitivity.json"), "w") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)

    rows = []
    for design in designs:
        row = {"Network design": design["name"]}
        for depth in depths:
            value = design["sigma_z"]["%g" % depth]["median_km"]
            row["z = %g km" % depth] = None if value is None else round(value, 1)
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(args.output, "table_s2_depth_sensitivity.csv"),
                 index=False)

    print(table.to_string(index=False))
    print()
    print("events used                %d" % len(events))
    print("stations per event, median %g" % np.median(stations_per_event))
    for depth in depths:
        key = "%g" % depth
        print("two-parameter z=%-4s median %.1f km (four-parameter inflation %.0f%%)"
              % (key, two_parameter[key]["median_km"], inflation.get(key, float("nan"))))
    for value in sorted(scaling, key=float):
        print("reading uncertainty %-5s s median %.1f km"
              % (value, scaling[value]["median_km"]))
    print("self check %s" % results["self_check"])


if __name__ == "__main__":
    main()
