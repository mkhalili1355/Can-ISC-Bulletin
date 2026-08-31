"""
Analysis routines for the Fars Arc depth-resolution study.

Every quantity reported in the manuscript is computed here and written to
results.json by scripts/01_run_analysis.py. The figure script reads that file
and recomputes nothing.

Subsetting precedes every statistic. The azimuthal gap at a given distance
cutoff is computed only from stations inside that cutoff, and the gap reported
by the Bulletin over all stations is computed separately.
"""

from typing import Dict, List

import numpy as np
import pandas as pd

from .geometry import azimuthal_gap_deg, critical_angle_deg, straight_ray_takeoff_deg


def _json_float(value):
    """Cast to a JSON-serialisable float, mapping NaN to None."""
    if value is None:
        return None
    value = float(value)
    return None if np.isnan(value) else value


def describe(values) -> Dict:
    """Five-number summary and mean of a numeric series, ignoring NaN."""
    series = pd.Series(values).dropna()
    if series.empty:
        return {"n": 0}
    return {
        "n": int(series.size),
        "min": _json_float(series.min()),
        "p25": _json_float(series.quantile(0.25)),
        "median": _json_float(series.median()),
        "p75": _json_float(series.quantile(0.75)),
        "max": _json_float(series.max()),
        "mean": _json_float(series.mean()),
    }


def build_event_table(arrivals: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """
    Collapse the arrival table to one row per event and attach the recording
    geometry available for that event.

    The ISC assigns one prime hypocenter per event, so AUTHOR is the agency
    whose solution the Bulletin adopted.
    """
    km_per_deg = cfg["physics"]["km_per_degree"]
    arrivals = arrivals.copy()
    arrivals["dist_km"] = arrivals["DIST"] * km_per_deg

    events = (arrivals
              .drop_duplicates("EVENTID")
              .loc[:, ["EVENTID", "TYPE", "AUTHOR", "DATE_1", "TIME_1",
                       "LAT_1", "LON_1", "DEPTH", "MAG", "TYPE_1", "year"]]
              .rename(columns={"LAT_1": "lat", "LON_1": "lon",
                               "DEPTH": "depth_km", "MAG": "mag",
                               "TYPE_1": "mag_type", "DATE_1": "date",
                               "TIME_1": "time", "AUTHOR": "agency"})
              .reset_index(drop=True))

    nearest = arrivals.groupby("EVENTID")["dist_km"].min().rename("nearest_station_km")
    n_all = arrivals.groupby("EVENTID")["STA"].nunique().rename("n_stations_all")

    local_km = cfg["thresholds"]["local_network_km"]
    local = arrivals[arrivals["dist_km"] <= local_km]
    n_local = local.groupby("EVENTID")["STA"].nunique().rename("n_stations_local")
    gap_local = (local.groupby("EVENTID")["BAZ"]
                 .apply(azimuthal_gap_deg).rename("gap_local_deg"))
    gap_all = (arrivals.groupby("EVENTID")["BAZ"]
               .apply(azimuthal_gap_deg).rename("gap_reported_deg"))

    geometry = pd.concat([nearest, n_all, n_local, gap_local, gap_all], axis=1)
    events = events.merge(geometry, left_on="EVENTID", right_index=True, how="left")
    events["n_stations_local"] = events["n_stations_local"].fillna(0).astype(int)
    events["gap_local_deg"] = events["gap_local_deg"].fillna(360.0)
    return events


def depth_quantization(events: pd.DataFrame, cfg: Dict) -> Dict:
    """Integer quantization and operational values in the catalog depth column."""
    tolerance = cfg["quantization"]["integer_tolerance_km"]
    depth = events["depth_km"].dropna()
    is_integer = np.abs(depth - np.round(depth)) <= tolerance

    counts = {}
    for value in cfg["quantization"]["report_values_km"]:
        n = int((depth == value).sum())
        counts["%.1f" % value] = {"n": n, "percent": 100.0 * n / depth.size}

    most_frequent = depth.value_counts().head(10) / depth.size * 100.0
    return {
        "n_events_with_depth": int(depth.size),
        "percent_integer_valued": 100.0 * float(is_integer.mean()),
        "operational_values": counts,
        "ten_most_frequent_values": {("%.1f" % k): _json_float(v)
                                     for k, v in most_frequent.items()},
        "distribution": describe(depth),
    }


def agency_confounding(events: pd.DataFrame, cfg: Dict) -> Dict:
    """
    Decompose the pooled depth trend by reporting agency.

    For each year the pooled median over all events reported up to that year is
    compared with the per-agency medians for the same years. A pooled series
    that moves while the per-agency series do not indicates a composition
    effect rather than a change in the depths themselves.
    """
    min_n = cfg["agencies"]["min_events_to_report"]
    detachment = cfg["physics"]["detachment_depth_km"]
    with_depth = events.dropna(subset=["depth_km"])
    sizes = with_depth.groupby("agency").size()
    reported = sorted(sizes[sizes >= min_n].index.tolist())

    years = sorted(events["year"].unique().tolist())
    per_year = []
    for year in years:
        in_year = with_depth[with_depth["year"] == year]
        cumulative = with_depth[with_depth["year"] <= year]["depth_km"]
        row = {
            "year": int(year),
            "n_events": int((events["year"] == year).sum()),
            "pooled_median_depth_km": _json_float(in_year["depth_km"].median()),
            "pooled_percent_shallower_than_detachment":
                100.0 * float((in_year["depth_km"] < detachment).mean()),
            "cumulative_median_depth_km": _json_float(cumulative.median()),
            "cumulative_percent_shallower_than_detachment":
                100.0 * float((cumulative < detachment).mean()),
        }
        for agency in reported:
            subset = in_year[in_year["agency"] == agency]["depth_km"]
            row["share_%s_percent" % agency] = (
                100.0 * len(subset) / len(in_year) if len(in_year) else None)
            row["median_%s_km" % agency] = (_json_float(subset.median())
                                            if len(subset) else None)
        per_year.append(row)

    per_agency = {}
    for agency in reported:
        subset = with_depth[with_depth["agency"] == agency]["depth_km"]
        fractional = np.abs(subset - np.round(subset))
        per_agency[agency] = {
            "n": int(subset.size),
            "median_km": _json_float(subset.median()),
            "percent_integer_valued": 100.0 * float(
                (fractional <= cfg["quantization"]["integer_tolerance_km"]).mean()),
            "percent_equal_to_10km": 100.0 * float((subset == 10.0).mean()),
        }

    first, last = per_year[0], per_year[-1]
    return {
        "agencies_reported_individually": reported,
        "per_year": per_year,
        "per_agency": per_agency,
        "annual_median_first_year_km": first["pooled_median_depth_km"],
        "annual_median_last_year_km": last["pooled_median_depth_km"],
        "annual_percent_shallow_first_year":
            first["pooled_percent_shallower_than_detachment"],
        "annual_percent_shallow_last_year":
            last["pooled_percent_shallower_than_detachment"],
        "cumulative_median_first_year_km": first["cumulative_median_depth_km"],
        "cumulative_median_last_year_km": last["cumulative_median_depth_km"],
        "cumulative_percent_shallow_first_year":
            first["cumulative_percent_shallower_than_detachment"],
        "cumulative_percent_shallow_last_year":
            last["cumulative_percent_shallower_than_detachment"],
    }


def gap_versus_distance_cutoff(arrivals: pd.DataFrame, cfg: Dict) -> Dict:
    """
    Recompute the azimuthal gap using only stations within a series of
    epicentral-distance cutoffs. A cutoff of None uses every station and
    reproduces the gap the Bulletin reports.
    """
    km_per_deg = cfg["physics"]["km_per_degree"]
    dist_km = arrivals["DIST"] * km_per_deg
    rows: List[Dict] = []
    for cutoff in cfg["thresholds"]["gap_distance_cutoffs_km"]:
        subset = arrivals if cutoff is None else arrivals[dist_km <= cutoff]
        gaps = subset.groupby("EVENTID")["BAZ"].apply(azimuthal_gap_deg)
        rows.append({
            "cutoff_km": cutoff,
            "n_events_with_any_station": int(gaps.size),
            "median_gap_deg": _json_float(gaps.median()),
            "p25_gap_deg": _json_float(gaps.quantile(0.25)),
            "p75_gap_deg": _json_float(gaps.quantile(0.75)),
            "percent_gap_above_180": 100.0 * float((gaps > 180.0).mean()),
        })
    return {"rows": rows}


def network_capability(events: pd.DataFrame, cfg: Dict) -> Dict:
    """Fraction of events whose recording geometry can constrain depth."""
    detachment = cfg["physics"]["detachment_depth_km"]
    max_gap = cfg["thresholds"]["max_gap_deg"]
    min_stations = cfg["thresholds"]["min_stations"]
    local_km = cfg["thresholds"]["local_network_km"]
    nearest = events["nearest_station_km"]

    with_depth = events.dropna(subset=["depth_km", "nearest_station_km"])
    closer_than_depth = int(
        (with_depth["nearest_station_km"] < with_depth["depth_km"]).sum())

    well_constrained = ((events["n_stations_local"] >= min_stations)
                        & (events["gap_local_deg"] < max_gap))
    return {
        "nearest_station_km": describe(nearest),
        "percent_with_station_closer_than_detachment":
            100.0 * float((nearest < detachment).mean()),
        "n_with_station_closer_than_own_depth": closer_than_depth,
        "percent_with_station_closer_than_own_depth":
            100.0 * closer_than_depth / len(with_depth),
        "best_local_gap_deg": _json_float(events["gap_local_deg"].min()),
        "n_well_constrained": int(well_constrained.sum()),
        "n_events": int(len(events)),
        "percent_well_constrained": 100.0 * float(well_constrained.mean()),
        "criterion": {"min_stations_within_%dkm" % local_km: min_stations,
                      "max_local_gap_deg": max_gap},
    }


def depth_distance_tradeoff(events: pd.DataFrame, cfg: Dict) -> Dict:
    """
    Catalog depth as a function of nearest-station distance, reported per
    agency. Pooling agencies here would reintroduce the confounding the study
    is testing for.
    """
    bins = cfg["thresholds"]["nearest_station_bins_km"]
    detachment = cfg["physics"]["detachment_depth_km"]
    out: Dict[str, List[Dict]] = {}
    for agency, subset in events.dropna(subset=["depth_km"]).groupby("agency"):
        if len(subset) < cfg["agencies"]["min_events_to_report"]:
            continue
        rows = []
        for lower, upper in zip(bins[:-1], bins[1:]):
            band = subset[(subset["nearest_station_km"] >= lower) &
                          (subset["nearest_station_km"] < upper)]["depth_km"]
            rows.append({
                "nearest_station_from_km": lower,
                "nearest_station_to_km": upper,
                "n": int(band.size),
                "median_depth_km": _json_float(band.median()),
                "percent_shallower_than_detachment":
                    100.0 * float((band < detachment).mean()) if band.size else None,
            })
        out[agency] = rows
    return out


def takeoff_angle_geometry(cfg: Dict) -> Dict:
    """
    Predicted take-off angles used to interpret the emergence angles reported
    by the local network and to state the network-design requirement.
    """
    physics = cfg["physics"]
    depth = physics["detachment_depth_km"]
    distances = [10, 20, 40, 60, 100, 150, 200, 300, 400]
    direct = [{"distance_km": d,
               "takeoff_from_vertical_deg":
                   float(straight_ray_takeoff_deg(d, depth))}
              for d in distances]
    return {
        "source_depth_km": depth,
        "pn_critical_angle_deg": critical_angle_deg(physics["vp_crust_km_s"],
                                                    physics["vp_mantle_km_s"]),
        "direct_ray_takeoff": direct,
        "method": ("Direct-ray values assume a uniform half-space and are a "
                   "first-order bound. The Pn value is the Snell critical "
                   "angle for the configured crust and mantle velocities."),
    }


def split_eras(events: pd.DataFrame, cfg: Dict):
    """
    Split the event table into the reviewed primary period and the later,
    unreviewed period.

    This is the only place the split is made. It returns two disjoint frames
    and asserts that they exhaust the input, so a later change to the
    configuration cannot silently drop or double-count events.
    """
    last_primary = cfg["study_period"]["primary_last_year"]
    primary = events[events["year"] <= last_primary]
    recent = events[events["year"] > last_primary]
    if len(primary) + len(recent) != len(events):
        raise AssertionError("era split lost or duplicated events")
    if not set(primary["EVENTID"]).isdisjoint(set(recent["EVENTID"])):
        raise AssertionError("era split produced overlapping event sets")
    return primary.copy(), recent.copy()


def _agency_depth_style(frame: pd.DataFrame, cfg: Dict) -> Dict:
    """
    Summarise how one agency populates its depth column.

    A determined depth is rarely an exact integer and rarely repeats the same
    value; an assigned depth concentrates on one value.
    """
    tolerance = cfg["quantization"]["integer_tolerance_km"]
    depth = frame["depth_km"].dropna()
    if depth.empty:
        return {"n": 0}
    is_integer = np.abs(depth - np.round(depth)) <= tolerance
    modal_value = depth.value_counts().idxmax()
    return {
        "n": int(depth.size),
        "median_km": _json_float(depth.median()),
        "percent_integer_valued": 100.0 * float(is_integer.mean()),
        "percent_exactly_10km": 100.0 * float((depth == 10.0).mean()),
        "percent_exactly_0km": 100.0 * float((depth == 0.0).mean()),
        "modal_value_km": _json_float(modal_value),
        "percent_at_modal_value": 100.0 * float((depth == modal_value).mean()),
        "n_distinct_values": int(depth.nunique()),
    }


def recent_era(events: pd.DataFrame, cfg: Dict) -> Dict:
    """
    Composition and depth style of the years after the reviewed period.

    `events` must be the recent frame returned by split_eras(). Passing the
    full table would pool the two periods.
    """
    period = cfg["study_period"]
    min_n = cfg["agencies"]["min_events_to_report_recent"]
    with_depth = events.dropna(subset=["depth_km"])

    sizes = with_depth.groupby("agency").size().sort_values(ascending=False)
    reported = [agency for agency in sizes.index if sizes[agency] >= min_n]
    by_agency = {agency: _agency_depth_style(
                     with_depth[with_depth["agency"] == agency], cfg)
                 for agency in reported}

    years = sorted(events["year"].unique().tolist())
    composition = []
    for year in years:
        in_year = events[events["year"] == year]
        counts = in_year["agency"].value_counts()
        total = int(counts.sum())
        composition.append({
            "year": int(year),
            "n_events": total,
            "is_partial_year": int(year) == period["partial_final_year"],
            "shares_percent": {agency: 100.0 * int(n) / total
                               for agency, n in counts.items()},
            "leading_agency": str(counts.idxmax()),
            "leading_agency_percent": 100.0 * int(counts.max()) / total,
        })

    invariant = {agency: style for agency, style in by_agency.items()
                 if style.get("n_distinct_values") == 1}

    return {
        "years": [int(year) for year in years],
        "n_events": int(len(events)),
        "n_events_with_depth": int(len(with_depth)),
        "partial_final_year": period["partial_final_year"],
        "isc_review_cutoff": period["isc_review_cutoff"],
        "min_events_to_report": min_n,
        "agencies_reported": reported,
        "by_agency": by_agency,
        "composition_by_year": composition,
        "agencies_with_invariant_depth": {
            agency: {"value_km": style["modal_value_km"], "n": style["n"]}
            for agency, style in invariant.items()},
    }


def era_comparison(primary: pd.DataFrame, recent: pd.DataFrame,
                   cfg: Dict) -> Dict:
    """
    Compare the depth style of each agency between the two periods.

    Only agencies present in both periods are compared; the rest are listed as
    entering or leaving. Each side is computed from its own frame.
    """
    min_n = cfg["agencies"]["min_events_to_report_recent"]
    p = primary.dropna(subset=["depth_km"])
    r = recent.dropna(subset=["depth_km"])
    p_agencies = set(p["agency"].value_counts().pipe(lambda s: s[s >= min_n]).index)
    r_agencies = set(r["agency"].value_counts().pipe(lambda s: s[s >= min_n]).index)

    both = sorted(p_agencies & r_agencies)
    compared = {}
    for agency in both:
        before = _agency_depth_style(p[p["agency"] == agency], cfg)
        after = _agency_depth_style(r[r["agency"] == agency], cfg)
        compared[agency] = {
            "primary": before,
            "recent": after,
            "share_of_primary_percent": 100.0 * before["n"] / len(p),
            "share_of_recent_percent": 100.0 * after["n"] / len(r),
            "median_shift_km": (None if before["n"] == 0 or after["n"] == 0
                                else _json_float(after["median_km"]
                                                 - before["median_km"])),
        }
    return {
        "agencies_in_both_periods": both,
        "agencies_only_in_primary": sorted(p_agencies - r_agencies),
        "agencies_only_in_recent": sorted(r_agencies - p_agencies),
        "by_agency": compared,
    }


def target_area(events: pd.DataFrame, cfg: Dict) -> Dict:
    """The same diagnostics restricted to the target sub-box."""
    box = cfg["study_area"]["target"]
    inside = events[(events["lat"].between(box["lat_min"], box["lat_max"])) &
                    (events["lon"].between(box["lon_min"], box["lon_max"]))]
    depth = inside["depth_km"].dropna()
    edges = np.arange(0, 42, 2)
    histogram, _ = np.histogram(depth, bins=edges)
    return {
        "box": box,
        "n_events": int(len(inside)),
        "n_by_year": {str(int(k)): int(v) for k, v in
                      inside["year"].value_counts().sort_index().items()},
        "depth_histogram_2km_bins": [
            {"from_km": float(edges[i]), "to_km": float(edges[i + 1]),
             "n": int(histogram[i])} for i in range(len(histogram))],
        "nearest_station_km": describe(inside["nearest_station_km"]),
        "gap_local_deg": describe(inside["gap_local_deg"]),
    }
