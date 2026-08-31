"""
Tests for the ISC reader, built on a synthetic file that reproduces both field
counts seen in the real exports.

test_the_net_lat_field_split_is_repaired constructs a row in which the network
code and station latitude share one field, then checks that DEPTH afterwards
holds the depth and not the station longitude. Without the repair this test
fails, which is the silent error it guards against.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import pytest
except ImportError:
    import harness as pytest

from src import isc_io

HEADER = ("EVENTID  ,TYPE,REPORTER ,STA  ,NET,LAT     ,LON      ,ELEV   ,CHN,"
          "DIST  ,BAZ  ,ISCPHASE,REPPHASE,DATE      ,TIME       ,RES  ,TDEF,"
          "AMPLITUDE,PER  ,AUTHOR   ,DATE      ,TIME       ,LAT     ,LON      ,"
          "DEPTH,AUTHOR   ,TYPE  ,MAG ")

# 27 fields: NET and LAT share field 4, the case that breaks a naive parser
ROW_MERGED = (" 17499151,ke  ,CSEM     ,GHIR ,IR  28.2855,  52.9867, 1200.0,???,"
              "  2.29,127.4,Pn      ,Pn      ,2010-01-09,08:28:09.60,  1.4,True,"
              "    244.0, 0.70,ISC      ,2010-01-09,08:27:31.59, 29.6976,  50.9225,"
              " 22.0,CSEM     ,ML    , 3.5")

# 28 fields: the network code is blank, so the separating comma is present
ROW_SPLIT = (" 17499151,ke  ,NEIC     ,UOSS ,II ,  24.9453,  56.2042,  284.4,???,"
             "  4.10, 90.0,Pn      ,P       ,2010-01-09,08:28:40.00,  0.5,True,"
             "         ,     ,ISC      ,2010-01-09,08:27:31.59, 29.6976,  50.9225,"
             " 22.0,CSEM     ,ML    , 3.5")


def _write(tmpdir, name, body_rows):
    path = os.path.join(tmpdir, name)
    with open(path, "w") as handle:
        handle.write("International Seismological Centre\nISC: Arrivals\n\n")
        handle.write("DATA_TYPE ARRIVAL:ASSOCIATED CSV\n")
        handle.write("-----EVENT----|---ARRIVAL DATA---|---ORIGIN DATA---\n")
        handle.write(HEADER + "\n")
        for row in body_rows:
            handle.write(row + "\n")
        handle.write("\n")
        handle.write("STOP\n")
    return path


def test_the_net_lat_field_split_is_repaired():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, "arr_test.txt", [ROW_MERGED])
        table, diagnostics = isc_io.read_arrival_file(path)
        assert diagnostics["rows_kept"] == 1
        assert diagnostics["rows_repaired"] == 1
        row = table.iloc[0]
        assert row["NET"] == "IR"
        assert row["LAT"] == pytest.approx(28.2855)
        assert row["LON"] == pytest.approx(52.9867)
        assert row["DEPTH"] == pytest.approx(22.0)
        assert row["DATE_1"] == "2010-01-09"


def test_rows_that_already_have_all_fields_are_left_alone():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, "arr_test.txt", [ROW_SPLIT])
        table, diagnostics = isc_io.read_arrival_file(path)
        assert diagnostics["rows_repaired"] == 0
        assert table.iloc[0]["NET"] == "II"
        assert table.iloc[0]["LAT"] == pytest.approx(24.9453)
        assert table.iloc[0]["DEPTH"] == pytest.approx(22.0)


def test_both_field_counts_can_coexist_in_one_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, "arr_test.txt", [ROW_MERGED, ROW_SPLIT, ROW_MERGED])
        table, diagnostics = isc_io.read_arrival_file(path)
        assert diagnostics["rows_kept"] == 3
        assert diagnostics["rows_repaired"] == 2
        assert list(table["DEPTH"]) == [pytest.approx(22.0)] * 3


def test_blank_lines_inside_the_table_do_not_truncate_it():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, "arr_test.txt", [ROW_MERGED, "", ROW_MERGED])
        table, _ = isc_io.read_arrival_file(path)
        assert len(table) == 2


def test_alignment_assertion_catches_a_shifted_table():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, "arr_test.txt", [ROW_MERGED])
        table, _ = isc_io.read_arrival_file(path)
        isc_io.assert_alignment(table)
        table.loc[0, "DATE_1"] = "52.9867"
        with pytest.raises(AssertionError):
            isc_io.assert_alignment(table)


def test_empty_input_files_are_recorded_not_silently_ignored():
    with tempfile.TemporaryDirectory() as tmp:
        _write(tmp, "arr_2010.txt", [ROW_MERGED])
        open(os.path.join(tmp, "arr_2023.txt"), "w").close()
        arrivals, manifest = isc_io.load_catalog(tmp)
        assert len(arrivals) == 1
        assert len(manifest) == 2
        empty = manifest[manifest["file"] == "arr_2023.txt"].iloc[0]
        assert empty["status"] == "empty file"
        assert empty["rows_kept"] == 0


def test_the_manifest_records_a_checksum_for_every_input():
    with tempfile.TemporaryDirectory() as tmp:
        _write(tmp, "arr_2010.txt", [ROW_MERGED])
        _, manifest = isc_io.load_catalog(tmp)
        digest = manifest.iloc[0]["sha256"]
        assert isinstance(digest, str) and len(digest) == 64


def test_a_directory_with_no_matching_file_raises():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(FileNotFoundError):
            isc_io.load_catalog(tmp)


def test_a_directory_of_only_empty_files_raises():
    with tempfile.TemporaryDirectory() as tmp:
        open(os.path.join(tmp, "arr_2010.txt"), "w").close()
        with pytest.raises(ValueError):
            isc_io.load_catalog(tmp)
