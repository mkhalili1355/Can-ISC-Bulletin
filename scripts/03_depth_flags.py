"""
Depth-determination analysis of ISC Bulletin ISF 2.1 exports.

Reads the ISF 2.1 files obtained from the ISC Bulletin web service and
quantifies, per reporting agency:

  - the explicit depth-determination flags ('f' fixed depth, 'd' depth-phase
    constraint, absent = free), which the Bulletin's Arrival CSV export does
    not carry;
  - the reported formal depth uncertainties, including the 999 km sentinel;
  - the integer-quantization signature of the depth column;
  - the response of reported depth to nearest-station distance, separated
    by depth flag;
  - local recording geometry (station count and azimuthal gap within
    distance cut-offs) recomputed from the phase annotations, and the
    fraction of events meeting a set of well-constrained network criteria.

Column offsets follow the ISF 2.1 fixed-width layout and were verified
against the exports analyzed here: latitude [36:44], longitude [45:54],
depth field [70:78] with the flag character at column 76, depth
uncertainty [77:83], nearest-station distance [97:104].

No input line is discarded silently; every rejection is counted and the
counts are written to the output JSON. Outputs are deterministic given the
input files: a diagnostics JSON and a CSV of all parsed prime hypocenters.

Usage:
    python 03_depth_flags.py --input <dir with 20XX-20YY.txt> --output <dir>
"""

import argparse
import collections
import csv
import glob
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir))

from src.stats import spearmanr

KM_PER_DEG = 111.19

# Each criterion: (minimum stations, radius in km, maximum local gap in degrees)
WELL_CONSTRAINED_CRITERIA = [
    (4, 50, 180),
    (4, 100, 180),
    (6, 100, 180),
    (6, 150, 120),
    (8, 200, 180),
    (10, 200, 150),
]

ORIGIN_RE = re.compile(r"^(\d{4})/(\d{2})/(\d{2}) ")
DEPTH_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)([fd]?)\s*$")
SENTINEL_DEPTH_ERR_KM = 999.0


def parse_isf_export(data_dir):
    """Parse origin lines and phase annotations from all ISF files in data_dir.

    Returns (origins, rejections, n_phase_lines). Each origin carries the
    depth flag, formal uncertainty and defining-phase fields, and collects
    the (epicentral distance, back-azimuth) of each contributing station
    from the phase annotations that follow its origin line.
    """
    files = sorted(glob.glob(os.path.join(data_dir, "2*.txt")) +
                   glob.glob(os.path.join(data_dir, "*.isf")))
    origins = []
    rejections = collections.Counter()
    n_phase_lines = 0
    current = None

    for filepath in files:
        fname = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                line = raw.rstrip("\n")

                match = ORIGIN_RE.match(line)
                if match:
                    try:
                        year = int(match.group(1))
                        lat = float(line[36:44].strip())
                        lon = float(line[45:54].strip())
                        depth_match = DEPTH_RE.match(line[70:78])
                        if depth_match:
                            depth = float(depth_match.group(1))
                            flag_char = depth_match.group(2)
                        else:
                            depth, flag_char = None, ""
                        err_text = line[77:83].strip()
                        depth_err = float(err_text) if err_text else None
                        mdist_text = line[97:104].strip()
                        mdist = float(mdist_text) if mdist_text else None
                        tokens = line.split()
                        agency, orig_id = tokens[-2], tokens[-1]
                    except (ValueError, IndexError) as exc:
                        key = "origin line rejected (%s)" % exc.__class__.__name__
                        rejections[key] += 1
                        current = None
                        continue
                    record = {
                        "orig_id": orig_id,
                        "year": year,
                        "month": int(match.group(2)),
                        "day": int(match.group(3)),
                        "lat": lat,
                        "lon": lon,
                        "depth_km": depth,
                        "depth_flag": flag_char if flag_char in ("f", "d") else "free",
                        "depth_err_km": depth_err,
                        "ndef": line[83:88].strip(),
                        "nsta": line[88:92].strip(),
                        "gap_reported": line[92:97].strip(),
                        "mdist_deg": mdist,
                        "agency": agency,
                        "source_file": fname,
                        "station_azimuths": [],
                        "station_codes": set(),
                    }
                    origins.append(record)
                    current = record
                    continue

                if line.startswith(("Event ", "Magnitude", "Date", "DATA_TYPE",
                                    "Sta ", "ISC Bulletin")):
                    continue

                # Phase annotation line: first token is the station code,
                # second and third are epicentral distance and back-azimuth.
                tokens = line.split()
                if current is None or len(tokens) < 3:
                    continue
                try:
                    dist_deg = float(tokens[1])
                    azimuth = float(tokens[2])
                except ValueError:
                    continue
                n_phase_lines += 1
                if tokens[0] not in current["station_codes"]:
                    current["station_codes"].add(tokens[0])
                    current["station_azimuths"].append((dist_deg, azimuth))

    return origins, rejections, n_phase_lines


def azimuthal_gap(azimuths):
    """Largest angular interval between sorted back-azimuths, duplicates collapsed."""
    unique = sorted({round(a, 2) for a in azimuths})
    if len(unique) < 2:
        return 360.0
    intervals = [unique[i + 1] - unique[i] for i in range(len(unique) - 1)]
    intervals.append(360.0 - unique[-1] + unique[0])
    return max(intervals)


def local_geometry(record, radius_km):
    """Stations and azimuthal gap within radius; subset selected before the gap."""
    cutoff_deg = radius_km / KM_PER_DEG
    azimuths = [a for dist, a in record["station_azimuths"] if dist <= cutoff_deg]
    return len(azimuths), azimuthal_gap(azimuths)


def flag_summary(origins, agencies=("ISC", "TEH")):
    """Depth-determination flags and reported uncertainties, per agency."""
    summary = {}
    for agency in agencies:
        subset = [o for o in origins if o["agency"] == agency]
        if not subset:
            continue
        counts = collections.Counter(o["depth_flag"] for o in subset)
        n = len(subset)
        errors = [o["depth_err_km"] for o in subset if o["depth_err_km"] is not None]
        sentinels = [e for e in errors if e >= SENTINEL_DEPTH_ERR_KM]
        summary[agency] = {
            "n": n,
            "fixed_n": counts.get("f", 0),
            "fixed_percent": round(100.0 * counts.get("f", 0) / n, 1),
            "depth_phase_n": counts.get("d", 0),
            "depth_phase_percent": round(100.0 * counts.get("d", 0) / n, 1),
            "free_n": counts.get("free", 0),
            "free_percent": round(100.0 * counts.get("free", 0) / n, 1),
            "depth_err_reported_n": len(errors),
            "depth_err_median_km": _median(errors),
            "depth_err_p90_km": _percentile(errors, 90),
            "depth_err_max_km": max(errors) if errors else None,
            "depth_err_sentinel_n": len(sentinels),
            "depth_err_median_excl_sentinel_km": _median([e for e in errors
                                                          if e < SENTINEL_DEPTH_ERR_KM]),
        }
    return summary


def annual_flag_stability(origins, agency="ISC"):
    """Fraction of fixed-depth solutions per year, for one agency."""
    by_year = collections.defaultdict(list)
    for o in origins:
        if o["agency"] == agency:
            by_year[o["year"]].append(o)
    return {year: {
        "n": len(subset),
        "fixed_percent": round(100.0 * sum(o["depth_flag"] == "f" for o in subset) / len(subset), 1),
    } for year, subset in sorted(by_year.items())}


def quantization_summary(origins, agencies=("ISC", "TEH")):
    """Integer-depth and modal-value signatures of the depth column, per agency."""
    summary = {}
    for agency in agencies:
        depths = [o["depth_km"] for o in origins
                  if o["agency"] == agency and o["depth_km"] is not None]
        if not depths:
            continue
        n_int = sum(1 for d in depths if abs(d - round(d)) < 1e-9)
        n_ten = sum(1 for d in depths if abs(d - 10.0) < 1e-9)
        summary[agency] = {
            "n": len(depths),
            "median_depth_km": _median(depths),
            "integer_percent": round(100.0 * n_int / len(depths), 1),
            "exactly_10km_percent": round(100.0 * n_ten / len(depths), 1),
        }
    return summary


def annual_medians(origins):
    """Yearly median depth per agency and each agency's share of prime hypocenters."""
    by_key = collections.defaultdict(list)
    for o in origins:
        if o["agency"] in ("ISC", "TEH") and o["depth_km"] is not None:
            by_key[(o["agency"], o["year"])].append(o["depth_km"])
    years = sorted({year for _, year in by_key})
    rows = []
    for year in years:
        isc = by_key.get(("ISC", year), [])
        teh = by_key.get(("TEH", year), [])
        total = len(isc) + len(teh)
        rows.append({
            "year": year,
            "ISC_n": len(isc),
            "ISC_median_km": _median(isc),
            "TEH_n": len(teh),
            "TEH_median_km": _median(teh),
            "TEH_share_percent": round(100.0 * len(teh) / total, 1) if total else None,
        })
    return rows


def depth_distance_response(origins, agency="ISC"):
    """Spearman rank correlation of depth and nearest-station distance, by flag.

    Fixed-depth solutions carry no depth information, so the correlation is
    computed separately for free and fixed subsets.
    """
    response = {}
    for label, flag in (("free", "free"), ("fixed", "f")):
        subset = [o for o in origins
                  if o["agency"] == agency and o["depth_km"] is not None
                  and o["depth_flag"] == flag and o["mdist_deg"]]
        if len(subset) < 5:
            response[label] = {"n": len(subset)}
            continue
        rho, p = spearmanr([o["mdist_deg"] for o in subset],
                           [o["depth_km"] for o in subset])
        response[label] = {
            "n": len(subset),
            "spearman_rho": round(float(rho), 3),
            "p_value": float("%.3g" % p),
            "median_depth_km": _median([o["depth_km"] for o in subset]),
            "median_mdist_km": round(KM_PER_DEG * _median([o["mdist_deg"] for o in subset]), 1),
        }
    return response


def sensitivity_matrix(origins, criteria=None):
    """Events meeting each well-constrained criterion, with the subset selected
    before the gap is computed (the comparison across radii requires it)."""
    criteria = criteria or WELL_CONSTRAINED_CRITERIA
    total = len(origins)
    rows = []
    for n_min, radius_km, max_gap in criteria:
        count = 0
        for o in origins:
            n_stations, gap = local_geometry(o, radius_km)
            if n_stations >= n_min and gap < max_gap:
                count += 1
        rows.append({
            "min_stations": n_min,
            "radius_km": radius_km,
            "max_local_gap_deg": max_gap,
            "n_events": count,
            "percent_of_events": round(100.0 * count / total, 1) if total else None,
        })
    return rows


def write_origin_csv(origins, path):
    """Write all parsed origins with their local geometry to CSV."""
    fields = ["orig_id", "year", "month", "day", "lat", "lon", "depth_km",
              "depth_flag", "depth_err_km", "ndef", "nsta", "gap_reported",
              "mdist_deg", "agency", "n_stations_any",
              "n_sta_100km", "gap_100km_deg", "n_sta_200km", "gap_200km_deg",
              "source_file"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for o in origins:
            n100, gap100 = local_geometry(o, 100)
            n200, gap200 = local_geometry(o, 200)
            writer.writerow([
                o["orig_id"], o["year"], o["month"], o["day"], o["lat"], o["lon"],
                o["depth_km"], o["depth_flag"], o["depth_err_km"], o["ndef"],
                o["nsta"], o["gap_reported"], o["mdist_deg"], o["agency"],
                len(o["station_codes"]),
                n100, gap100, n200, gap200, o["source_file"],
            ])


def _median(values):
    values = sorted(values)
    n = len(values)
    if n == 0:
        return None
    if n % 2:
        return round(values[n // 2], 2)
    return round(0.5 * (values[n // 2 - 1] + values[n // 2]), 2)


def _percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q / 100.0 * (len(ordered) - 1)))))
    return round(ordered[index], 2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Depth-determination flag analysis of ISC ISF 2.1 exports")
    parser.add_argument("--input", default=os.path.dirname(os.path.abspath(sys.argv[0])),
                        help="directory holding the 20XX-20YY.txt ISF files (default: this script's directory)")
    parser.add_argument("--output", default=None,
                        help="output directory (default: <input>/flag_results)")
    parser.add_argument("--end-year", type=int, default=2023,
                        help="last year of the reviewed primary period")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = args.output or os.path.join(args.input, "flag_results")
    os.makedirs(out_dir, exist_ok=True)

    print("Parsing ISF 2.1 exports from %s" % args.input)
    origins, rejections, n_phase_lines = parse_isf_export(args.input)
    print("  origins parsed : %d" % len(origins))
    print("  phase lines    : %d" % n_phase_lines)
    print("  rejections     : %d %s" % (sum(rejections.values()), dict(rejections)))

    primary = [o for o in origins if o["year"] <= args.end_year]
    recent = [o for o in origins if o["year"] > args.end_year]
    print("  primary period (<= %d): %d origins" % (args.end_year, len(primary)))

    report = {
        "config": {
            "input_dir": os.path.abspath(args.input),
            "end_year_primary": args.end_year,
            "well_constrained_criteria": WELL_CONSTRAINED_CRITERIA,
            "sentinel_depth_err_km": SENTINEL_DEPTH_ERR_KM,
        },
        "provenance": {
            "origins_total": len(origins),
            "phase_lines_total": n_phase_lines,
            "rejected_lines": dict(rejections),
        },
        "primary_period": {
            "origins": len(primary),
            "flag_summary": flag_summary(primary),
            "isc_fixed_by_year": annual_flag_stability(primary, "ISC"),
            "quantization": quantization_summary(primary),
            "annual_medians_and_shares": annual_medians(primary),
            "depth_distance_response_isc": depth_distance_response(primary, "ISC"),
        },
        "recent_unreviewed": {
            "origins": len(recent),
            "flag_summary": flag_summary(recent),
        },
        "sensitivity_primary": sensitivity_matrix(primary),
        "sensitivity_all": sensitivity_matrix(origins),
    }

    json_path = os.path.join(out_dir, "depth_flag_diagnostics.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    csv_path = os.path.join(out_dir, "parsed_isf_origins_full.csv")
    write_origin_csv(origins, csv_path)
    print("Saved %s" % json_path)
    print("Saved %s" % csv_path)

    for agency, stats in report["primary_period"]["flag_summary"].items():
        print("%s: n=%d  fixed=%.1f%%  depth-phase=%.1f%%  free=%.1f%%  "
              "depth err: n=%d median=%s km p90=%s km max=%s km (sentinel n=%d)"
              % (agency, stats["n"], stats["fixed_percent"],
                 stats["depth_phase_percent"], stats["free_percent"],
                 stats["depth_err_reported_n"], stats["depth_err_median_km"],
                 stats["depth_err_p90_km"], stats["depth_err_max_km"],
                 stats["depth_err_sentinel_n"]))
    for agency, stats in report["primary_period"]["quantization"].items():
        print("%s quantization: median=%s km  integer=%.1f%%  ==10km=%.1f%%"
              % (agency, stats["median_depth_km"], stats["integer_percent"],
                 stats["exactly_10km_percent"]))
    for label, stats in report["primary_period"]["depth_distance_response_isc"].items():
        print("ISC %s-depth response: %s" % (label, stats))
    print("Well-constrained sensitivity (primary period):")
    for row in report["sensitivity_primary"]:
        print("  >= %d stations within %d km, gap < %d deg : %d events (%s%%)"
              % (row["min_stations"], row["radius_km"], row["max_local_gap_deg"],
                 row["n_events"], row["percent_of_events"]))


main()
