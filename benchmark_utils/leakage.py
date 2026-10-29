"""Leakage detection for forecasting adapters.

A forecasting adapter receives the *full* series ``x[i]`` of shape
``(T_i, C)`` plus per-series cutoff indexes, and is contractually bound
to use only ``x[i][:cutoff]`` as history (see
:mod:`benchmark_utils.adapters.base`). A model that peeks at
``x[i][cutoff:]`` — the future, which contains the evaluation target —
has *leakage*, and any score it earns is invalid.

This module probes for leakage *behaviourally*, without inspecting model
internals: it perturbs the future segment of the target series and
checks that the forecasts are unchanged. Future *covariates* are
legitimately known ahead of time, so they are NOT perturbed — only the
future target values in ``x`` are.

Soundness
---------
For a series with cutoffs ``C_i``, we perturb only indices
``>= max(C_i)``. The legitimate history for every cutoff ``c`` in
``C_i`` is ``x[:c]``, a subset of ``x[:max(C_i)]``, which is left
untouched. Therefore any change in the forecast is caused solely by the
model reading future target values — i.e. leakage. There are no false
positives.

Cost
----
The probe predicts on a reduced input: the first ``max_series`` series
that have a perturbable future, each with only its last cutoff
``max(C_i)``. The perturbation starts at ``max(C_i)``, so the last window
is the only one guaranteed to see it; earlier windows would only re-run
the model. The soundness argument above is unchanged, since it holds
per cutoff.

Coverage caveat: leakage that only reads the gap ``[c, max(C_i))`` of an
*earlier* cutoff is not exercised, nor is leakage specific to series
beyond the first ``max_series`` probed ones. Leakage is a property of the
adapter's code rather than of a particular series, so a handful of
series is enough to catch it in practice; pass ``max_series=None`` to
probe every series.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from benchmark_utils.covariates import Covariates
from benchmark_utils.inputs import ForecastInput
from benchmark_utils.outputs import ForecastOutput


# Probe size used by default. Leakage is a property of the adapter's code,
# so a few series catch it; 8 single-window series keep the probe to 16
# forecast windows (one small batch for any model) instead of re-predicting
# the whole test set twice.
DEFAULT_MAX_SERIES = 8


@dataclass(frozen=True)
class LeakageReport:
    """Result of a behavioural leakage probe.

    Attributes
    ----------
    leaked : bool
        True if perturbing the future target changed any forecast.
    max_abs_diff : float
        Largest absolute change in any quantile across all tested series.
        ``0.0`` when nothing was tested.
    n_series_tested : int
        Series that had a perturbable future segment.
    n_series_skipped : int
        Series encountered while selecting the probe that have no future
        to perturb (``max(cutoff) >= T``) or no cutoffs — leakage cannot
        be exercised for these.
    offending_series : list of int
        Indexes (into the original ``x``) of series whose forecast
        changed under perturbation.
    """

    leaked: bool
    max_abs_diff: float
    n_series_tested: int
    n_series_skipped: int
    offending_series: List[int] = field(default_factory=list)


def _perturb_future(
    series: np.ndarray, cut: int, rng: np.random.Generator
) -> np.ndarray:
    """Return a copy of ``series`` with indices ``>= cut`` replaced.

    The replacement is a high-magnitude random signal that is independent
    of the original values, so a leaky model is caught whether it *reads*,
    *adds*, or *scales by* the future — there is no perturbation a
    non-degenerate model could be invariant to by coincidence.
    """
    out = np.array(series, dtype=np.float64, copy=True)
    future_shape = out[cut:].shape
    # Scale relative to the history so the perturbation dominates any
    # plausible signal the model might be reading, then shift far off.
    hist = out[:cut]
    scale = 10.0 * (float(np.nanstd(hist)) + 1.0) if hist.size else 100.0
    out[cut:] = rng.standard_normal(future_shape) * scale + 1e4
    return out


def _select_series(covariates: Covariates, idx: List[int]) -> Covariates:
    """Keep only series ``idx`` of every present covariate field.

    Covariate arrays are aligned with the *full* series time axis (one
    entry per series), so selecting series needs no time slicing — unlike
    :meth:`Covariates.slice`, which cuts out a single window.
    """

    def pick(f):
        return None if f is None else [f[i] for i in idx]

    return Covariates(
        static_covars=pick(covariates.static_covars),
        hist_covars=pick(covariates.hist_covars),
        future_covars=pick(covariates.future_covars),
    )


def detect_forecast_leakage(
    adapter,
    forecast_input: ForecastInput,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    seed: int = 0,
    max_series: Optional[int] = DEFAULT_MAX_SERIES,
) -> LeakageReport:
    """Probe a fitted forecasting adapter for target leakage.

    Builds a reduced probe input — the first ``max_series`` series with a
    perturbable future, each restricted to its last cutoff — and runs
    ``adapter.predict`` twice on it: once as is and once with the future
    target values (beyond that cutoff) perturbed. A leakage-free model
    produces identical forecasts; any difference beyond ``rtol``/``atol``
    is flagged.

    Parameters
    ----------
    adapter : BaseTSFMAdapter
        A fitted forecasting adapter (``predict(ForecastInput) ->
        ForecastOutput``).
    forecast_input : ForecastInput
        The input to probe. Only the target ``x`` is perturbed;
        ``covariates`` (including ``future_covars``) are subset to the
        probed series but otherwise unchanged, as future covariates are
        legitimately known.
    rtol, atol : float
        Tolerances forwarded to :func:`numpy.allclose`.
    seed : int
        Seed for the perturbation RNG (deterministic).
    max_series : int or None
        Number of series to probe, taken in order among those with a
        perturbable future. Defaults to :data:`DEFAULT_MAX_SERIES`, which
        keeps the probe at ``2 * max_series`` forecast windows regardless
        of the test-set size. ``None`` probes every series.

    Returns
    -------
    LeakageReport
    """
    rng = np.random.default_rng(seed)

    # Pick the probed series and perturb their future beyond the last
    # cutoff. Series with no cutoff or no future are skipped and do not
    # count towards ``max_series``.
    tested, skipped = [], 0
    probe_x, perturbed_x, probe_cutoffs = [], [], []
    for i, (series, cutoffs) in enumerate(
        zip(forecast_input.x, forecast_input.cutoff_indexes)
    ):
        if max_series is not None and len(tested) >= max_series:
            break
        series = np.asarray(series)
        if len(cutoffs) == 0:
            skipped += 1
            continue
        cut = int(max(cutoffs))
        if cut >= series.shape[0]:
            skipped += 1
            continue
        tested.append(i)
        probe_x.append(series)
        perturbed_x.append(_perturb_future(series, cut, rng))
        # Only the last window sees the perturbation; earlier windows of
        # the same series would add model calls but no coverage.
        probe_cutoffs.append([cut])

    if not tested:
        return LeakageReport(
            leaked=False,
            max_abs_diff=0.0,
            n_series_tested=0,
            n_series_skipped=skipped,
        )

    # Both calls see the same reduced batch (series, cutoffs, covariates),
    # so the only difference between them is the perturbed future.
    covariates = _select_series(forecast_input.covariates, tested)
    baseline = adapter.predict(
        ForecastInput(x=probe_x, cutoff_indexes=probe_cutoffs, covariates=covariates)
    )
    perturbed = adapter.predict(
        ForecastInput(
            x=perturbed_x, cutoff_indexes=probe_cutoffs, covariates=covariates
        )
    )
    _validate_output(baseline)
    _validate_output(perturbed)

    offending: List[int] = []
    max_abs_diff = 0.0
    for k, i in enumerate(tested):
        a = np.asarray(baseline.quantiles[k], dtype=np.float64)
        b = np.asarray(perturbed.quantiles[k], dtype=np.float64)
        if a.shape != b.shape:
            # Output shape itself depends on the future — unambiguous leakage.
            offending.append(i)
            max_abs_diff = float("inf")
            continue
        diff = np.abs(a - b)
        finite = diff[np.isfinite(diff)]
        if finite.size:
            max_abs_diff = max(max_abs_diff, float(finite.max()))
        if not np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
            offending.append(i)

    return LeakageReport(
        leaked=bool(offending),
        max_abs_diff=max_abs_diff,
        n_series_tested=len(tested),
        n_series_skipped=skipped,
        offending_series=offending,
    )


def _validate_output(output) -> None:
    if not isinstance(output, ForecastOutput):
        raise TypeError(
            "leakage probe expects predict() to return a ForecastOutput, "
            f"got {type(output).__name__}"
        )
