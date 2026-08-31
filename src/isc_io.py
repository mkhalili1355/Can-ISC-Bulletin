"""
Reader for ISC Bulletin ARRIVAL:ASSOCIATED CSV text files.

Two properties of the export require explicit handling, both verified against
the files used in this study.

First, the header declares 28 comma-separated columns while most data rows
carry 27 fields, because the station network code and the station latitude are
emitted as a single field. A parser that trusts the header shifts every column
after STA by one position and reads the station longitude as DEPTH.

Second, the field count is not constant within one file: rows with a blank
network code retain the separating comma.

The reader repairs the merged field from the field count, revalidates the
origin-date column afterwards, and rejects any row that fails the check. No row
is discarded silently; the count of rejected rows is returned for reporting.
"""

import hashlib
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

HEADER_TOKEN = "EVENTID"
# Index of the field that carries the merged network code and station latitude
NET_LAT_FIELD_INDEX = 4

_DATA_ROW = re.compile(r"^\s*\d+\s*,")
_NET_LAT = re.compile(r"^(\S*?)\s*(-?\d+\.\d+)$")
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

NUMERIC_COLUMNS = (
    "LAT", "LON", "ELEV", "DIST", "BAZ", "RES",
    "AMPLITUDE", "PER", "LAT_1", "LON_1", "DEPTH", "MAG",
)


def sha256(path: str) -> str:
    """SHA-256 digest of a file, recorded in the provenance manifest."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dedupe(names: List[str]) -> List[str]:
    """Suffix repeated header names; ISC reuses DATE, TIME, LAT, LON, AUTHOR
    and TYPE for the arrival and the origin blocks."""
    seen: Dict[str, int] = {}
    out: List[str] = []
    for name in names:
        if name in seen:
            seen[name] += 1
            out.append("%s_%d" % (name, seen[name]))
        else:
            seen[name] = 0
            out.append(name)
    return out


def read_arrival_file(path: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Parse one ISC arrival export. Returns (dataframe, diagnostics)."""
    with open(path, "r", errors="replace") as handle:
        lines = handle.read().replace("\r", "").split("\n")

    header_indices = [i for i, line in enumerate(lines)
                      if line.strip().startswith(HEADER_TOKEN)]
    if not header_indices:
        raise ValueError("no ISC arrival header found in %s" % path)
    header_index = header_indices[0]

    columns = _dedupe([name.strip()
                       for name in lines[header_index].split(",")])
    n_expected = len(columns)

    rows: List[List[str]] = []
    n_repaired = 0
    n_rejected = 0
    for line in lines[header_index + 1:]:
        if not _DATA_ROW.match(line):
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == n_expected - 1:
            match = _NET_LAT.match(fields[NET_LAT_FIELD_INDEX])
            if match is None:
                n_rejected += 1
                continue
            fields = (fields[:NET_LAT_FIELD_INDEX] +
                      [match.group(1), match.group(2)] +
                      fields[NET_LAT_FIELD_INDEX + 1:])
            n_repaired += 1
        if len(fields) != n_expected:
            n_rejected += 1
            continue
        rows.append(fields)

    table = pd.DataFrame(rows, columns=columns).replace("", np.nan).infer_objects(copy=False)
    for column in NUMERIC_COLUMNS:
        if column in table.columns:
            table[column] = pd.to_numeric(table[column], errors="coerce")

    diagnostics = {
        "rows_kept": len(table),
        "rows_repaired": n_repaired,
        "rows_rejected": n_rejected,
    }
    return table, diagnostics


def assert_alignment(table: pd.DataFrame) -> None:
    """Raise if the column shift described in the module docstring reappears."""
    misaligned = ~table["DATE_1"].astype(str).str.match(_ISO_DATE)
    if bool(misaligned.any()):
        raise AssertionError(
            "column alignment check failed for %d rows: the origin-date column "
            "does not contain dates, so the network and latitude fields were "
            "not split correctly" % int(misaligned.sum()))


def load_catalog(input_dir: str,
                 pattern: str = "arr_") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read every ISC arrival export in `input_dir` whose name contains `pattern`.

    Returns the arrival table, one row per reported phase, and a manifest with
    one row per input file recording its size, checksum and row diagnostics.
    """
    names = sorted(name for name in os.listdir(input_dir)
                   if pattern in name and name.lower().endswith(".txt"))
    if not names:
        raise FileNotFoundError(
            "no files matching '%s*.txt' in %s" % (pattern, input_dir))

    frames, records = [], []
    for name in names:
        path = os.path.join(input_dir, name)
        size = os.path.getsize(path)
        record = {"file": name, "bytes": size, "sha256": sha256(path)}
        if size == 0:
            record.update({"rows_kept": 0, "rows_repaired": 0,
                           "rows_rejected": 0, "status": "empty file"})
            records.append(record)
            continue
        table, diagnostics = read_arrival_file(path)
        table["source_file"] = name
        frames.append(table)
        record.update(diagnostics)
        record["status"] = "ok"
        records.append(record)

    if not frames:
        raise ValueError("every input file in %s is empty" % input_dir)

    arrivals = pd.concat(frames, ignore_index=True)
    assert_alignment(arrivals)
    arrivals["year"] = arrivals["DATE_1"].str[:4].astype(int)
    return arrivals, pd.DataFrame(records)
