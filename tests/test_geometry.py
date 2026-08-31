"""Known-answer tests for the geometry functions."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import pytest
except ImportError:
    import harness as pytest

from src.geometry import (azimuth_deg, azimuthal_gap_deg, critical_angle_deg,
                          great_circle_km, straight_ray_takeoff_deg)


def test_distance_zero():
    assert great_circle_km(28.0, 51.0, 28.0, 51.0) == pytest.approx(0.0)


def test_distance_one_degree_of_latitude():
    # One degree of latitude on a sphere of radius 6371 km is 111.195 km.
    assert great_circle_km(28.0, 51.0, 29.0, 51.0) == pytest.approx(111.195,
                                                                    abs=0.01)


def test_distance_is_symmetric():
    forward = great_circle_km(28.36, 51.26, 29.64, 52.52)
    reverse = great_circle_km(29.64, 52.52, 28.36, 51.26)
    assert forward == pytest.approx(reverse)


def test_azimuth_cardinal_directions():
    assert azimuth_deg(28.0, 51.0, 29.0, 51.0) == pytest.approx(0.0, abs=1e-6)
    assert azimuth_deg(28.0, 51.0, 28.0, 52.0) == pytest.approx(90.0, abs=0.5)
    assert azimuth_deg(29.0, 51.0, 28.0, 51.0) == pytest.approx(180.0, abs=1e-6)


def test_gap_of_four_stations_at_the_compass_points():
    assert azimuthal_gap_deg([0, 90, 180, 270]) == pytest.approx(90.0)


def test_gap_handles_the_wrap_around():
    # Stations at 10 and 350 degrees: the interval through north spans 20
    # degrees and the other spans 340. The gap is the larger of the two.
    assert azimuthal_gap_deg([10.0, 350.0]) == pytest.approx(340.0)


def test_gap_of_a_single_station_is_360():
    assert azimuthal_gap_deg([42.0]) == 360.0
    assert azimuthal_gap_deg([]) == 360.0


def test_duplicate_azimuths_do_not_improve_the_gap():
    assert azimuthal_gap_deg([0, 0, 0, 180]) == pytest.approx(180.0)


def test_critical_angle_for_a_standard_crust():
    # arcsin(6.0 / 8.0) = 48.59 degrees
    assert critical_angle_deg(6.0, 8.0) == pytest.approx(48.59, abs=0.01)


def test_critical_angle_rejects_impossible_velocities():
    with pytest.raises(ValueError):
        critical_angle_deg(8.0, 6.0)


def test_takeoff_saturates_towards_horizontal():
    near = straight_ray_takeoff_deg(20.0, 12.0)
    far = straight_ray_takeoff_deg(400.0, 12.0)
    assert near < far < 90.0
    assert far > 88.0


def test_takeoff_rejects_zero_depth():
    with pytest.raises(ValueError):
        straight_ray_takeoff_deg(100.0, 0.0)
