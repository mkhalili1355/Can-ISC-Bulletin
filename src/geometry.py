"""
Spherical geometry and network-geometry metrics.

All functions are pure: same input, same output, no global state and no I/O.
They are covered by known-answer tests in tests/test_geometry.py.
"""

import numpy as np


def great_circle_km(lat1, lon1, lat2, lon2, radius_km=6371.0):
    """Great-circle distance in km from the haversine formula. Vectorised."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = phi2 - phi1
    delta_lambda = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = (np.sin(delta_phi / 2.0) ** 2 +
         np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2)
    return 2.0 * radius_km * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def azimuth_deg(lat1, lon1, lat2, lon2):
    """Initial bearing from point 1 to point 2, degrees clockwise from north."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_lambda = np.radians(np.asarray(lon2) - np.asarray(lon1))
    y = np.sin(delta_lambda) * np.cos(phi2)
    x = (np.cos(phi1) * np.sin(phi2) -
         np.sin(phi1) * np.cos(phi2) * np.cos(delta_lambda))
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


def azimuthal_gap_deg(azimuths, decimals=1):
    """
    Largest angular gap in degrees between consecutive station azimuths.

    Duplicate azimuths are collapsed first, since two stations on the same
    bearing constrain the solution no better than one. Fewer than two distinct
    azimuths returns 360, the value for a source that is not surrounded.
    """
    sorted_azimuths = np.sort(np.unique(
        np.round(np.asarray(azimuths, dtype=float), decimals)))
    if sorted_azimuths.size < 2:
        return 360.0
    wrap = 360.0 - (sorted_azimuths[-1] - sorted_azimuths[0])
    return float(max(np.max(np.diff(sorted_azimuths)), wrap))


def critical_angle_deg(v_upper, v_lower):
    """
    Snell critical angle in degrees from the vertical for a head wave at an
    interface between v_upper and v_lower. Used to predict the Pn take-off
    angle for comparison with the emergence angles reported by the network.
    """
    if not v_lower > v_upper > 0:
        raise ValueError("require 0 < v_upper < v_lower")
    return float(np.degrees(np.arcsin(v_upper / v_lower)))


def straight_ray_takeoff_deg(distance_km, depth_km):
    """
    Take-off angle from the downward vertical for a straight ray in a uniform
    half-space.

    This is a first-order bound rather than a substitute for ray tracing. It is
    used only to show how rapidly the take-off angle approaches the horizontal
    as epicentral distance grows.
    """
    depth = np.asarray(depth_km, dtype=float)
    if np.any(depth <= 0):
        raise ValueError("depth must be positive")
    return np.degrees(np.arctan2(np.asarray(distance_km, dtype=float), depth))
