"""
Tests for the analysis layer, including the subsetting guards.

A statistic computed from a restricted station set differs from the one the
Bulletin reports, and the restriction must be applied before the statistic is
taken. The gap tests build an event whose near stations are clustered on one
side and whose far stations close the circle, and require the near-field gap to
remain large; pooling the two sets fails the test.

The era tests exist because pooling the reviewed and unreviewed periods would
not raise an error. The numbers would simply differ, so the guard has to be
explicit.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import pytest
except ImportError:
    import harness as pytest

from src import analysis

CFG = {
    "physics": {"km_per_degree": 111.195, "detachment_depth_km": 11.0,
                "vp_crust_km_s": 6.0, "vp_mantle_km_s": 8.0,
                "earth_radius_km": 6371.0},
    "thresholds": {"gap_distance_cutoffs_km": [100, None],
                   "local_network_km": 200, "max_gap_deg": 180,
                   "min_stations": 8,
                   "nearest_station_bins_km": [0, 50, 100, 200, 100000]},
    "quantization": {"integer_tolerance_km": 1e-9,
                     "report_values_km": [10.0, 24.0]},
    "agencies": {"min_events_to_report": 1, "min_events_to_report_recent": 1},
    "study_area": {"target": {"name": "t", "lat_min": 0, "lat_max": 90,
                              "lon_min": 0, "lon_max": 90}},
    "study_period": {"primary_first_year": 2010, "primary_last_year": 2022,
                     "isc_review_cutoff": "2024-08-01",
                     "partial_final_year": 2026},
}


def _arrivals(records):
    """Build an arrival frame from (event, station, dist_km, baz, depth,
    agency, date) tuples."""
    rows = []
    for event_id, station, dist_km, baz, depth, agency, date in records:
        rows.append({"EVENTID": event_id, "STA": station,
                     "DIST": dist_km / CFG["physics"]["km_per_degree"],
                     "BAZ": baz, "DEPTH": depth, "AUTHOR": agency,
                     "DATE_1": date, "TIME_1": "00:00:00", "TYPE": "ke",
                     "LAT_1": 28.5, "LON_1": 51.5, "MAG": 3.0,
                     "TYPE_1": "ML", "year": int(date[:4])})
    return pd.DataFrame(rows)


def test_gap_at_a_cutoff_uses_only_stations_inside_it():
    # Near stations, all within 100 km, span 0 to 60 degrees, leaving a gap of
    # 300 degrees. Distant stations close the circle, so the unrestricted gap
    # is 90 degrees. The restricted value must stay at 300.
    records = [("E1", "N1", 20.0, 0.0, 10.0, "ISC", "2015-01-01"),
               ("E1", "N2", 40.0, 30.0, 10.0, "ISC", "2015-01-01"),
               ("E1", "N3", 60.0, 60.0, 10.0, "ISC", "2015-01-01"),
               ("E1", "F1", 500.0, 150.0, 10.0, "ISC", "2015-01-01"),
               ("E1", "F2", 700.0, 240.0, 10.0, "ISC", "2015-01-01"),
               ("E1", "F3", 900.0, 300.0, 10.0, "ISC", "2015-01-01")]
    rows = analysis.gap_versus_distance_cutoff(_arrivals(records), CFG)["rows"]
    by_cutoff = {row["cutoff_km"]: row for row in rows}
    assert by_cutoff[100]["median_gap_deg"] == pytest.approx(300.0)
    assert by_cutoff[None]["median_gap_deg"] == pytest.approx(90.0)
    assert by_cutoff[100]["median_gap_deg"] > by_cutoff[None]["median_gap_deg"]


def test_event_table_has_one_row_per_event():
    records = [("E1", "A", 10.0, 0.0, 10.0, "ISC", "2015-01-01"),
               ("E1", "B", 20.0, 90.0, 10.0, "ISC", "2015-01-01"),
               ("E2", "A", 30.0, 45.0, 20.0, "TEH", "2016-01-01")]
    events = analysis.build_event_table(_arrivals(records), CFG)
    assert len(events) == 2
    assert set(events["EVENTID"]) == {"E1", "E2"}


def test_nearest_station_is_the_minimum_over_all_reported_stations():
    records = [("E1", "A", 250.0, 0.0, 10.0, "ISC", "2015-01-01"),
               ("E1", "B", 17.0, 90.0, 10.0, "ISC", "2015-01-01"),
               ("E1", "C", 90.0, 180.0, 10.0, "ISC", "2015-01-01")]
    events = analysis.build_event_table(_arrivals(records), CFG)
    assert events.iloc[0]["nearest_station_km"] == pytest.approx(17.0, abs=1e-6)


def test_local_station_count_excludes_stations_beyond_the_cutoff():
    records = [("E1", "A", 50.0, 0.0, 10.0, "ISC", "2015-01-01"),
               ("E1", "B", 199.0, 90.0, 10.0, "ISC", "2015-01-01"),
               ("E1", "C", 201.0, 180.0, 10.0, "ISC", "2015-01-01")]
    events = analysis.build_event_table(_arrivals(records), CFG)
    assert events.iloc[0]["n_stations_local"] == 2
    assert events.iloc[0]["n_stations_all"] == 3


def test_quantization_counts_are_exact():
    depths = [10.0] * 3 + [24.0] + [12.4]
    records = [("E%d" % i, "A", 50.0, 0.0, depth, "ISC", "2015-01-01")
               for i, depth in enumerate(depths)]
    events = analysis.build_event_table(_arrivals(records), CFG)
    quantization = analysis.depth_quantization(events, CFG)
    assert quantization["n_events_with_depth"] == 5
    assert quantization["percent_integer_valued"] == pytest.approx(80.0)
    assert quantization["operational_values"]["10.0"]["n"] == 3
    assert quantization["operational_values"]["10.0"]["percent"] == pytest.approx(60.0)


def test_agency_confounding_reproduces_a_constructed_composition_effect():
    # Two agencies whose own medians never change, 20 km and 10 km, but whose
    # shares flip between the two years. The pooled cumulative median moves
    # even though neither agency median does.
    records = [("A%d" % i, "S", 50.0, 0.0, 20.0, "AAA", "2015-01-01")
               for i in range(9)]
    records.append(("B0", "S", 50.0, 0.0, 10.0, "BBB", "2015-01-01"))
    records += [("C%d" % i, "S", 50.0, 0.0, 10.0, "BBB", "2016-01-01")
                for i in range(30)]
    events = analysis.build_event_table(_arrivals(records), CFG)
    out = analysis.agency_confounding(events, CFG)

    assert out["cumulative_median_first_year_km"] == pytest.approx(20.0)
    assert out["cumulative_median_last_year_km"] == pytest.approx(10.0)
    for agency, expected in (("AAA", 20.0), ("BBB", 10.0)):
        assert out["per_agency"][agency]["median_km"] == pytest.approx(expected)
    per_year = {row["year"]: row for row in out["per_year"]}
    assert per_year[2015]["median_BBB_km"] == pytest.approx(10.0)
    assert per_year[2016]["median_BBB_km"] == pytest.approx(10.0)


def test_well_constrained_requires_both_conditions():
    # Eight stations clustered on one side do not satisfy the gap condition.
    records = [("E1", "S%d" % i, 50.0, float(i * 5), 10.0, "ISC", "2015-01-01")
               for i in range(8)]
    events = analysis.build_event_table(_arrivals(records), CFG)
    assert analysis.network_capability(events, CFG)["n_well_constrained"] == 0

    # The same eight stations spread around the circle do.
    records = [("E1", "S%d" % i, 50.0, float(i * 45), 10.0, "ISC", "2015-01-01")
               for i in range(8)]
    events = analysis.build_event_table(_arrivals(records), CFG)
    assert analysis.network_capability(events, CFG)["n_well_constrained"] == 1


def test_results_are_deterministic():
    records = [("E%d" % i, "S", 50.0 + i, float(i * 37 % 360), 10.0 + (i % 7),
                "ISC" if i % 2 else "TEH", "201%d-01-01" % (i % 8))
               for i in range(200)]
    first = analysis.build_event_table(_arrivals(records), CFG)
    second = analysis.build_event_table(_arrivals(records), CFG)
    assert (analysis.depth_quantization(first, CFG)
            == analysis.depth_quantization(second, CFG))
    assert (analysis.network_capability(first, CFG)
            == analysis.network_capability(second, CFG))


def test_takeoff_geometry_matches_the_snell_prediction():
    out = analysis.takeoff_angle_geometry(CFG)
    assert out["pn_critical_angle_deg"] == pytest.approx(48.59, abs=0.01)
    angles = [row["takeoff_from_vertical_deg"] for row in out["direct_ray_takeoff"]]
    assert angles == sorted(angles)
    assert angles[-1] < 90.0


def test_era_split_is_exhaustive_and_disjoint():
    records = [("E1", "S1", 30.0, 0.0, 10.0, "ISC", "2015-01-01"),
               ("E2", "S1", 30.0, 0.0, 10.0, "TEH", "2022-06-01"),
               ("E3", "S1", 30.0, 0.0, 0.0, "IDC", "2025-03-01"),
               ("E4", "S1", 30.0, 0.0, 0.0, "IDC", "2026-02-01")]
    events = analysis.build_event_table(_arrivals(records), CFG)
    primary, recent = analysis.split_eras(events, CFG)
    assert len(primary) + len(recent) == len(events)
    assert set(primary["EVENTID"]).isdisjoint(set(recent["EVENTID"]))
    assert set(primary["EVENTID"]) == {"E1", "E2"}
    assert set(recent["EVENTID"]) == {"E3", "E4"}


def test_era_split_happens_before_statistics():
    # The primary period contains only 20 km depths and the recent period only
    # 0 km depths, so pooling across the boundary would return 10 km.
    records = [("P%d" % i, "S1", 30.0, 0.0, 20.0, "ISC", "2015-01-01")
               for i in range(10)]
    records += [("R%d" % i, "S1", 30.0, 0.0, 0.0, "IDC", "2025-01-01")
                for i in range(10)]
    events = analysis.build_event_table(_arrivals(records), CFG)
    primary, recent = analysis.split_eras(events, CFG)

    primary_median = analysis.depth_quantization(primary, CFG)["distribution"]["median"]
    recent_median = analysis.depth_quantization(recent, CFG)["distribution"]["median"]
    assert primary_median == pytest.approx(20.0)
    assert recent_median == pytest.approx(0.0)
    assert primary_median != pytest.approx(10.0)


def test_adding_recent_events_does_not_change_primary_statistics():
    # Extending the catalog with later years must change the recent-era section
    # and nothing else.
    base = [("P%d" % i, "S1", 30.0, 0.0, 12.0 + i, "ISC", "2015-01-01")
            for i in range(8)]
    extra = [("R%d" % i, "S1", 30.0, 0.0, 0.0, "IDC", "2025-01-01")
             for i in range(50)]

    before, _ = analysis.split_eras(
        analysis.build_event_table(_arrivals(base), CFG), CFG)
    after, _ = analysis.split_eras(
        analysis.build_event_table(_arrivals(base + extra), CFG), CFG)

    assert (analysis.depth_quantization(before, CFG)
            == analysis.depth_quantization(after, CFG))
    assert (analysis.network_capability(before, CFG)
            == analysis.network_capability(after, CFG))


def test_recent_era_detects_an_invariant_depth_column():
    records = [("E%d" % i, "S1", 30.0 + i, 0.0, 0.0, "IDC", "2025-01-01")
               for i in range(25)]
    records += [("V%d" % i, "S1", 30.0 + i, 0.0, 8.0 + i, "TEH", "2025-01-01")
                for i in range(25)]
    events = analysis.build_event_table(_arrivals(records), CFG)
    _, recent = analysis.split_eras(events, CFG)
    out = analysis.recent_era(recent, CFG)

    assert "IDC" in out["agencies_with_invariant_depth"]
    assert (out["agencies_with_invariant_depth"]["IDC"]["value_km"]
            == pytest.approx(0.0))
    assert "TEH" not in out["agencies_with_invariant_depth"]
    assert out["by_agency"]["IDC"]["n_distinct_values"] == 1
    assert out["by_agency"]["TEH"]["n_distinct_values"] > 1


def test_era_comparison_never_pools_the_two_periods():
    # One agency reports 20 km throughout the primary period and 4 km
    # throughout the recent one, so a concatenation anywhere would shift both
    # medians away from their exact values.
    records = [("P%d" % i, "S1", 30.0, 0.0, 20.0, "TEH", "2015-01-01")
               for i in range(25)]
    records += [("R%d" % i, "S1", 30.0, 0.0, 4.0, "TEH", "2025-01-01")
                for i in range(25)]
    events = analysis.build_event_table(_arrivals(records), CFG)
    primary, recent = analysis.split_eras(events, CFG)
    teh = analysis.era_comparison(primary, recent, CFG)["by_agency"]["TEH"]

    assert teh["primary"]["median_km"] == pytest.approx(20.0)
    assert teh["recent"]["median_km"] == pytest.approx(4.0)
    assert teh["median_shift_km"] == pytest.approx(-16.0)
    assert teh["primary"]["n"] == 25
    assert teh["recent"]["n"] == 25
