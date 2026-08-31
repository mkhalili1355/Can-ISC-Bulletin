"""
Tests for the rank-correlation implementation.

The reference probabilities were verified against a direct Simpson-rule
integration of the Student t density, independent of the continued-fraction
routine under test, and agree to a relative difference below 1e-11.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import pytest
except ImportError:
    import harness as pytest

from src.stats import (incomplete_beta, ols, rankdata, spearmanr,
                       t_critical, t_two_sided_p)


def test_ranks_of_distinct_values():
    assert list(rankdata([10.0, 30.0, 20.0])) == [1.0, 3.0, 2.0]


def test_ties_take_the_mean_rank():
    assert list(rankdata([5.0, 5.0, 9.0])) == [1.5, 1.5, 3.0]
    assert list(rankdata([1.0, 1.0, 1.0, 1.0])) == [2.5, 2.5, 2.5, 2.5]


def test_perfect_monotonic_relations():
    rho, p_value = spearmanr([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
    assert rho == pytest.approx(1.0)
    assert p_value == pytest.approx(0.0, abs=1e-12)
    rho, _ = spearmanr([1, 2, 3, 4, 5], [10, 8, 6, 4, 2])
    assert rho == pytest.approx(-1.0)


def test_rho_and_p_for_a_known_permutation():
    # Adjacent pairs transposed: rho = 1 - 6 * 10 / (10 * 99) = 0.93939...,
    # t = 7.75 on 8 degrees of freedom.
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    y = [2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0, 10.0, 9.0]
    rho, p_value = spearmanr(x, y)
    assert rho == pytest.approx(0.9393939393939393, abs=1e-12)
    assert p_value == pytest.approx(5.48405299851e-05, rel=1e-9)


def test_rho_is_invariant_under_monotonic_rescaling():
    x = [3.0, 1.0, 4.0, 1.5, 9.0, 2.6]
    y = [7.0, 2.0, 8.0, 3.0, 12.0, 5.0]
    rho_plain, _ = spearmanr(x, y)
    rho_scaled, _ = spearmanr([value * 1000.0 for value in x],
                              [10.0 ** value for value in y])
    assert rho_plain == pytest.approx(rho_scaled)


def test_degenerate_inputs_return_nan():
    rho, p_value = spearmanr([1.0, 2.0], [3.0, 4.0])
    assert rho != rho and p_value != p_value
    rho, p_value = spearmanr([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])
    assert rho != rho and p_value != p_value


def test_incomplete_beta_endpoints_and_symmetry():
    assert incomplete_beta(2.0, 3.0, 0.0) == 0.0
    assert incomplete_beta(2.0, 3.0, 1.0) == 1.0
    assert incomplete_beta(3.0, 3.0, 0.5) == pytest.approx(0.5, abs=1e-12)


def test_two_sided_t_probability_against_known_values():
    assert t_two_sided_p(0.0, 10) == pytest.approx(1.0, abs=1e-12)
    # t = 2.228138852 on 10 degrees of freedom is the two-sided 5 per cent point
    assert t_two_sided_p(2.228138852, 10) == pytest.approx(0.05, abs=1e-6)
    # With a very large sample the t distribution approaches the normal, whose
    # two-sided 5 per cent point is 1.959963985
    assert t_two_sided_p(1.959963985, 1000000) == pytest.approx(0.05, abs=1e-5)


def test_t_critical_matches_published_table_values():
    assert t_critical(0.05, 10) == pytest.approx(2.228138852, rel=1e-6)
    assert t_critical(0.05, 1) == pytest.approx(12.70620474, rel=1e-6)
    assert t_critical(0.05, 100000) == pytest.approx(1.95996, rel=1e-4)


def test_ols_reproduces_a_hand_computed_fit():
    x = [0.0, 1.0, 2.0, 3.0]
    y = [1.0, 3.0, 5.0, 8.0]
    fit = ols(y, [x])
    assert fit["dof"] == 2
    assert fit["coefficients"][0] == pytest.approx(0.8, abs=1e-12)
    assert fit["coefficients"][1] == pytest.approx(2.3, abs=1e-12)
    assert fit["residual_variance"] == pytest.approx(0.15, abs=1e-12)
    assert fit["standard_errors"][1] == pytest.approx(0.03 ** 0.5, rel=1e-9)
    half_width = t_critical(0.05, 2) * fit["standard_errors"][1]
    assert fit["ci_low"][1] == pytest.approx(2.3 - half_width, rel=1e-9)
    assert fit["ci_high"][1] == pytest.approx(2.3 + half_width, rel=1e-9)
