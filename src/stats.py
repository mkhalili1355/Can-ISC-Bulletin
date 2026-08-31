"""
Rank correlation and the supporting special functions.

Implemented directly so that the repository depends only on NumPy, pandas,
Matplotlib and PyYAML. The reference values in tests/test_stats.py were checked
independently against a numerical integration of the Student t density.
"""

import math

import numpy as np

_TINY = 1.0e-300
_EPS = 3.0e-14
_MAX_ITER = 300


def rankdata(values):
    """Ranks starting at 1, with tied values receiving their mean rank."""
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=float)
    ranks[order] = np.arange(1, array.size + 1, dtype=float)

    sorted_values = array[order]
    start = 0
    for index in range(1, array.size + 1):
        if index == array.size or sorted_values[index] != sorted_values[start]:
            if index - start > 1:
                ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index
    return ranks


def _betacf(a, b, x):
    """Continued fraction for the incomplete beta function, Lentz's method."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _TINY:
        d = _TINY
    d = 1.0 / d
    h = d
    for m in range(1, _MAX_ITER + 1):
        m2 = 2 * m
        numerator = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + numerator * d
        if abs(d) < _TINY:
            d = _TINY
        c = 1.0 + numerator / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        h *= d * c

        numerator = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + numerator * d
        if abs(d) < _TINY:
            d = _TINY
        c = 1.0 + numerator / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            return h
    return h


def incomplete_beta(a, b, x):
    """Regularised incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) +
                     a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t, df):
    """Two-sided tail probability of Student's t with df degrees of freedom."""
    df = float(df)
    if df <= 0:
        return float("nan")
    t = abs(float(t))
    return incomplete_beta(0.5 * df, 0.5, df / (df + t * t))


def spearmanr(x, y):
    """
    Spearman rank correlation and its two-sided p-value.

    The p-value uses the t approximation t = rho * sqrt((n - 2) / (1 - rho^2))
    on n - 2 degrees of freedom. Returns (nan, nan) when fewer than three pairs
    are supplied or either input has no variation in rank.
    """
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.size != b.size:
        raise ValueError("x and y must have the same length")
    n = a.size
    if n < 3:
        return float("nan"), float("nan")

    rank_a = rankdata(a)
    rank_b = rankdata(b)
    sd_a = rank_a.std()
    sd_b = rank_b.std()
    if sd_a == 0.0 or sd_b == 0.0:
        return float("nan"), float("nan")

    rho = float(np.mean((rank_a - rank_a.mean()) * (rank_b - rank_b.mean()))
                / (sd_a * sd_b))
    rho = max(-1.0, min(1.0, rho))
    if abs(rho) >= 1.0:
        return rho, 0.0

    df = n - 2
    t = rho * math.sqrt(df / (1.0 - rho * rho))
    return rho, t_two_sided_p(t, df)

def t_critical(p_two_sided, dof, upper=200.0, iterations=200):
    """Return the two-sided critical value of the Student t distribution.

    Obtained by bisecting t_two_sided_p, which is monotonically decreasing in
    the test statistic, so no additional special function is required.
    """
    if not 0.0 < p_two_sided < 1.0:
        raise ValueError("p_two_sided must lie in (0, 1)")
    if dof <= 0:
        raise ValueError("dof must be positive")
    low, high = 0.0, float(upper)
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        if t_two_sided_p(middle, dof) > p_two_sided:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def ols(response, predictors, confidence=0.95):
    """Least-squares fit returning coefficients, standard errors and intervals.

    An intercept is added automatically. Predictors are supplied as a sequence
    of equal-length columns; the returned arrays are ordered intercept first.
    """
    y = np.asarray(response, dtype=float)
    columns = [np.ones(y.size)] + [np.asarray(c, dtype=float) for c in predictors]
    design = np.column_stack(columns)
    dof = y.size - design.shape[1]
    if dof <= 0:
        raise ValueError("not enough observations for the requested model")

    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coefficients
    variance = float(residual @ residual) / dof
    covariance = variance * np.linalg.inv(design.T @ design)
    standard_errors = np.sqrt(np.diag(covariance))

    critical = t_critical(1.0 - confidence, dof)
    p_values = np.array([t_two_sided_p(c / s, dof) if s > 0.0 else 0.0
                         for c, s in zip(coefficients, standard_errors)])
    return {
        "coefficients": coefficients,
        "standard_errors": standard_errors,
        "p_values": p_values,
        "ci_low": coefficients - critical * standard_errors,
        "ci_high": coefficients + critical * standard_errors,
        "dof": dof,
        "residual_variance": variance,
    }
