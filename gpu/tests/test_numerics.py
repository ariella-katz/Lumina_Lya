#!/usr/bin/env python3
"""Tiny independent SciPy checks for the native numerical probe.

Run ``python gpu/tests/test_numerics.py gpu/build/numerics_probe``. This test
uses no lightcone data or GPU. Full-Voigt approximation error is reported
separately from agreement with the selected first/second-order model.
"""

import math
import subprocess
import sys

import numpy as np
from scipy.integrate import quad
from scipy.special import dawsn, wofz


SQRT_PI = math.sqrt(math.pi)


def profile(x, a, second=True):
    """Independent scalar expansion; avoid cancellation in distant wings."""
    if abs(x) >= 1.0e4:
        # The omitted relative term is < 1.4e-23 at this threshold.
        inverse2 = 1.0 / (x * x)
        return a / SQRT_PI * inverse2 * (1.0 + inverse2 * (1.5 + 3.75 * inverse2))
    return math.exp(-x * x) * (1.0 + (a * a * (1.0 - 2.0 * x * x) if second else 0.0)) + (
        2.0 * a / SQRT_PI * (2.0 * x * dawsn(x) - 1.0)
    )


def comb_reference(lo, hi, a, second):
    # Map integration to [0,1] so quadrature tolerances remain useful when
    # the interval is narrow or its opacity is many orders below one.
    width = hi - lo
    scale = max(profile(lo, a, second), profile(hi, a, second), 1.0e-300)
    points = [(x - lo) / width for x in (-8, -4, -2, 0, 2, 4, 8) if lo < x < hi]
    value = quad(
        lambda t: profile(lo + width * t, a, second) / scale,
        0.0, 1.0, points=points, epsabs=2.0e-12, epsrel=2.0e-12,
    )[0]
    return width * scale * value


def band_reference(lo, hi, drift, a, second):
    # The independent double integral evaluates H directly; it does not use
    # either antiderivative or the supplied hypergeometric approximation.
    width = hi - lo
    mid = lo + 0.5 * width - 0.5 * drift
    scale = max(profile(mid, a, second), 1.0e-300)

    def inner(t):
        x = lo + width * t
        points = [(x - y) / drift for y in (-8, -4, -2, 0, 2, 4, 8) if x - drift < y < x]
        return quad(
            lambda s: profile(x - drift * s, a, second) / scale,
            0.0, 1.0, points=points, epsabs=2.0e-11, epsrel=2.0e-11,
        )[0]

    points = [(x - lo) / width for x in (-8, -4, -2, 0, 2, 4, 8, drift) if lo < x < hi]
    return width * drift * scale * quad(
        inner, 0.0, 1.0, points=points, epsabs=2.0e-10, epsrel=2.0e-10,
    )[0]


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: test_numerics.py PATH_TO_NUMERICS_PROBE")
    cases = []

    def add(command, expected, *, absolute=2.0e-12, relative=2.0e-10):
        cases.append((command, expected, absolute, relative))

    for x in (0.0, 1.0e-8, 0.7, 2.14, 4.0, 5.15, 8.0, 20.0, 1.0e5):
        add(f"dawson {x:.17g}", dawsn(x), absolute=5.0e-15, relative=5.0e-14)
        add(f"dawson {-x:.17g}", -dawsn(x), absolute=5.0e-15, relative=5.0e-14)
    for boundary in (2.14, 5.15, 20.0):
        for x in (np.nextafter(boundary, -np.inf), boundary, np.nextafter(boundary, np.inf)):
            expected = 2.0 / SQRT_PI * quad(dawsn, 0.0, x, epsabs=2.0e-13, epsrel=2.0e-13)[0]
            add(f"hypergeo {x:.17g}", expected, absolute=2.0e-13, relative=2.0e-13)

    for second in (False, True):
        for a in (0.0, 4.7e-4, 0.05):
            for lo, hi in ((-2.0, 3.0), (1.2, 1.200001), (5.0, 7.0), (8.0, 20.0)):
                add(f"integral {lo} {hi} {a} {int(second)}", comb_reference(lo, hi, a, second))
            # Short-bin, short-drift and core-crossing branches.
            for lo, hi, drift in (
                (-2.0, 3.0, 0.7), (2.14, 5.15, 1.0), (5.15, 7.15, 0.8), (9.0, 9.0799, 1.0),
            ):
                add(f"band {lo} {hi} {drift} {a} {int(second)}", band_reference(lo, hi, drift, a, second))

    for lo, hi, drift in (
        (1.0e5, 1.0e5 + 1.0, 1.0e-3),
        (1.0e5, 1.0e5 + 1.0e-5, 1.0),
        (-1.0e5 - 1.0, -1.0e5, 1.0e-4),
        (1.0e5, 1.6e5, 5.0e4),
    ):
        a = 4.7e-3
        expected = band_reference(lo, hi, drift, a, True)
        add(f"band {lo:.17g} {hi:.17g} {drift:.17g} {a} 1", expected,
            absolute=1.0e-30, relative=1.0e-8)

    # A small dx/x is insufficient for a Gaussian tail: its relative
    # variation depends on x*dx. These values must retain relative accuracy
    # even though their magnitudes are far below an ordinary absolute floor.
    for lo, hi, drift in ((9.0, 9.0799, 1.0), (16.0, 16.149, 1.0), (26.0, 26.249, 1.0)):
        expected = band_reference(lo, hi, drift, 0.0, True)
        add(f"band {lo:.17g} {hi:.17g} {drift:.17g} 0 1", expected,
            absolute=0.0, relative=1.0e-8)

    add("delta 0 1 3", 0.0, absolute=0.0, relative=0.0)
    add("delta 1 1 3", 3.0 * SQRT_PI)
    add("delta 1 0 3", 0.0, absolute=0.0, relative=0.0)
    add("band_delta -1 1 1 100000000", math.log(2.0), absolute=1.0e-15, relative=1.0e-15)
    add("band_delta 0 1 1 100000000", 1.0e8 * SQRT_PI)
    add("band_delta -1 1 1 1e-12", -math.log1p(0.5 * math.expm1(-SQRT_PI * 1.0e-12)),
        absolute=1.0e-25, relative=1.0e-13)

    completed = subprocess.run(
        [sys.argv[1]], input="\n".join(case[0] for case in cases) + "\n",
        text=True, capture_output=True, check=True,
    )
    results = [float(value) for value in completed.stdout.split()]
    if len(results) != len(cases):
        raise AssertionError(f"Probe returned {len(results)} results for {len(cases)} cases")
    for result, (command, expected, absolute, relative) in zip(results, cases):
        if not math.isfinite(result) or abs(result - expected) > absolute + relative * abs(expected):
            raise AssertionError(f"{command}: got {result:.17g}, expected {expected:.17g}")
    print(f"Passed {len(cases)} independent numerical checks.")

    # These diagnostics are model errors, not CPU/CUDA discrepancies, and
    # are sampled measurements rather than universal approximation bounds.
    xs = np.linspace(-10.0, 10.0, 1001)
    for a in (0.00047, 0.0047, 0.047, 0.1):
        approximation = np.array([profile(x, a, True) for x in xs])
        full = wofz(xs + 1j * a).real
        error = float(np.max(np.abs(approximation - full) / full))
        print(f"Full-Voigt diagnostic: a={a:g}, sampled max relative second-order error={error:.6g}")


if __name__ == "__main__":
    main()
