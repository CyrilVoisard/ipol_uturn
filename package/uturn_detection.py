"""U-turn segmentation from the yaw angular velocity of a lower-back IMU.

The core of the method is the turning model of Salarian et al. (2009): the heading of the
trunk during a go / U-turn / back walking test is fitted by a smooth step

    M(i) = a * E((i - b) / c) + C * i,                       Equation (model)

    E(u) = -1/2 if u <= -1/2,  sin(pi * u) / 2 if |u| < 1/2,  1/2 if u >= 1/2,   (step)

and the U-turn is the interval [b - c/2, b + c/2]. Before the fit, the heading is corrected for
the gyroscope drift (Step 1), centred so that the turn appears as a sign change (Step 2), and
restricted to a window around the turn, re-centred and oriented (Step 3), so that the fit
(Step 4) is not disturbed by the drift, by the direction of the turn or by movements far from
the turn (e.g. rotations before or after the test).

Reference: A. Salarian, C. Zampieri, F. B. Horak, P. Carlson-Kuhta, J. G. Nutt and K. Aminian,
"Analyzing 180 degrees turns using an inertial system reveals early signs of progress in
Parkinson's disease", IEEE EMBC 2009, pp. 224-227, doi:10.1109/IEMBS.2009.5333970.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from . import config

# Fixed constants of the drift estimation (Section 3.2 of the article). They are not parameters
# of the method: the window lengths are floor(N/4) + k * floor(N/20), k = 0, ..., 5 (from a
# quarter to a half of the recording), the linearity score is (L/N)^2 / (variance + EPSILON).
_N_LENGTHS = 6
_EPSILON = 1e-8      # regularisation of the linearity score (deg^2)

# Constant of the original implementation, used with legacy=True only (Section 5).
_LEGACY_DIRECTION_FRAC = 0.30


class UTurnNotFound(ValueError):
    """Raised when the centred heading never changes sign (no turn in the recording)."""


@dataclass
class UTurnResult:
    """Output of the U-turn segmentation. Indices are sample numbers (time = index / freq)."""
    start: int                    # first sample of the U-turn, round(b - c / 2)
    end: int                      # last sample of the U-turn, round(b + c / 2)
    freq: float                   # acquisition frequency (Hz)
    amplitude: float              # |a|, heading change during the turn (deg)
    direction: int                # +1: counter-clockwise about +X, -1: clockwise
    orientation: int              # sign applied to the heading so that the fitted step increases
    r2: float                     # coefficient of determination of the fit on the window
    n_crossings: int              # number of sign changes of the centred heading
    n_half_turns: int             # protocol indicator: half-turns counted with hysteresis (expected 1)
    peak_yaw_rate: float          # max |angular velocity| between start and end (deg/s)
    drift: float                  # gyroscope drift removed in Step 1 (deg/s)
    params: tuple                 # fitted (a, b, c, C), in the oriented heading, samples
    p0: tuple                     # initial guess (a0, b0, c0, C0)
    middle: int                   # selected sign change (initial guess of b)
    crossings: np.ndarray         # all sign changes of the centred heading
    drift_window: tuple           # (start, end) of the window used for the drift estimate
    fit_window: tuple             # (start, end) of the window used for the model fit
    legacy: bool                  # True if the rules of the original implementation were used
    theta_raw: np.ndarray = field(repr=False)     # integrated heading (deg)
    theta_c: np.ndarray = field(repr=False)       # drift-corrected, centred on the recording (deg)
    theta: np.ndarray = field(repr=False)         # centred on the fit window and oriented (deg)
    model: np.ndarray = field(repr=False)         # fitted model on the fit window (deg)
    model0: np.ndarray = field(repr=False)        # initial model on the fit window (deg)

    @property
    def duration(self):
        """Duration of the U-turn (s)."""
        return (self.end - self.start) / self.freq

    @property
    def fit_inside_window(self):
        """True if both boundaries lie strictly inside the fit window."""
        return self.fit_window[0] < self.start < self.end < self.fit_window[1] - 1


# ---------------------------------------------------------------------------
# Elementary functions
# ---------------------------------------------------------------------------

def integrate(gyr, freq):
    """Heading (deg) obtained by cumulative summation of the angular velocity (deg/s).

    theta_i = (1 / F) * sum_{k=1}^{i} omega_k, so that theta_0 = 0.
    """
    theta = np.cumsum(np.asarray(gyr, dtype=float)) / freq
    return theta - theta[0]


def step_function(u):
    """Smooth step E(u) of Equation (step), from -1/2 to 1/2 with a half-sine transition."""
    u = np.asarray(u, dtype=float)
    return np.where(u <= -0.5, -0.5, np.where(u >= 0.5, 0.5, 0.5 * np.sin(np.pi * u)))


def turning_model(x, a, b, c, C):
    """Turning model M of Equation (model), evaluated at the sample indices x."""
    x = np.asarray(x, dtype=float)
    return a * step_function((x - b) / c) + C * x


def window_lengths(n):
    """Candidate window lengths floor(n/4) + k * floor(n/20), k = 0, ..., 5 (Section 3.2)."""
    return [n // 4 + k * max(1, n // 20) for k in range(_N_LENGTHS)]


def sliding_regression(y, length):
    """Least-squares line fit of y on every window of the given length (Algorithm 2).

    For each start s, the line is fitted on (i, y_i), i = s, ..., s + length - 1. The sums over
    each window are obtained from cumulative sums, so that all windows are processed in O(n).

    Returns
    -------
    slope, residual variance (mean squared residual), r^2 -- one value per start s
    """
    y = np.asarray(y, dtype=float)
    y = y - np.mean(y)                        # reduces round-off, does not change the fits
    n = len(y)
    idx = np.arange(n, dtype=float)
    cy = np.concatenate(([0.0], np.cumsum(y)))
    cyy = np.concatenate(([0.0], np.cumsum(y * y)))
    ciy = np.concatenate(([0.0], np.cumsum(idx * y)))
    s = np.arange(n - length + 1)
    e = s + length
    sum_y = cy[e] - cy[s]
    sum_yy = cyy[e] - cyy[s]
    sum_iy = ciy[e] - ciy[s]
    mean_i = s + (length - 1) / 2.0
    sxx = length * (length ** 2 - 1) / 12.0            # sum of (i - mean_i)^2
    sxy = sum_iy - mean_i * sum_y                      # sum of (i - mean_i)(y - mean_y)
    syy = np.maximum(sum_yy - sum_y ** 2 / length, 0.0)
    slope = sxy / sxx
    sse = np.maximum(syy - sxy ** 2 / sxx, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(syy > 0, sxy / np.sqrt(sxx * syy), 0.0)
    r2 = np.clip(r, -1.0, 1.0) ** 2
    return slope, sse / length, r2


def best_linear_window(y, score="linearity", start_max=np.inf, end_min=-np.inf):
    """Window of y of maximal score, among the lengths of window_lengths(len(y)) (Algorithm 2).

    score = "linearity" (Equation (score)): (L / n)^2 / (residual variance + EPSILON)
    score = "legacy" (original rule of Step 3, legacy mode only): r^2 * (L / n) / slope, for the
            windows such that start < start_max and start + L > end_min

    Lengths and starts are scanned in increasing order; the first maximum is kept.

    Returns
    -------
    (start, end) of the best window (end excluded), or None if no window is admissible
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    best_score, best = -np.inf, None
    for length in window_lengths(n):
        if length < 2 or length > n:
            continue
        slope, var_res, r2 = sliding_regression(y, length)
        starts = np.arange(n - length + 1)
        if score == "linearity":
            values = (length / n) ** 2 / (var_res + _EPSILON)
        elif score == "legacy":
            with np.errstate(divide="ignore", invalid="ignore"):
                values = r2 * (length / n) / slope
            values = np.where((starts < start_max) & (starts + length > end_min), values, -np.inf)
        else:
            raise ValueError("Unknown score: {}".format(score))
        values = np.where(np.isfinite(values), values, -np.inf)
        k = int(np.argmax(values))
        if values[k] > best_score:
            best_score, best = values[k], (k, k + length)
    return best


def line_fit(y, start, end):
    """Slope and intercept of the least-squares line of y_i on i over [start, end)."""
    x = np.arange(start, end, dtype=float)
    slope, intercept = np.polyfit(x, np.asarray(y[start:end], dtype=float), 1)
    return slope, intercept


def centre(theta, start=0, end=None):
    """Subtract the midpoint of the Q_LOW and Q_HIGH quantiles of theta[start:end]."""
    ref = theta[start:end]
    q_low, q_high = np.quantile(ref, config.Q_LOW), np.quantile(ref, config.Q_HIGH)
    return theta - (q_low + q_high) / 2.0


def sign_changes(theta_c, legacy):
    """Indices i such that the centred heading changes sign between samples i and i + 1."""
    if legacy:
        return np.where(np.sign(theta_c[1:] * theta_c[:-1]) == -1)[0]
    positive = theta_c > 0
    return np.where(positive[1:] != positive[:-1])[0]


def count_half_turns(theta, hysteresis):
    """Number of 180 deg rotations of the heading, counted with hysteresis (protocol indicator).

    The heading theta is centred so that the two levels of the U-turn are close to -90 and +90 deg.
    With y = (theta + 90) / 180, these levels are y = 0 and y = 1, and any further half-turn, in
    either direction, leads to another integer level. The state q is set to the nearest integer
    the first time y comes within 1/2 - h of an integer (h = hysteresis / 180); then q moves to
    q + 1 when y > q + 1/2 + h and to q - 1 when y < q - 1/2 - h, and each move counts one
    half-turn. Oscillations and rotations smaller than 90 + hysteresis deg are not counted.
    """
    y = (np.asarray(theta, dtype=float) + 90.0) / 180.0
    h = hysteresis / 180.0
    near = np.where(np.abs(y - np.round(y)) <= 0.5 - h)[0]
    if len(near) == 0:
        return 0
    q, count = int(np.round(y[near[0]])), 0
    for value in y[near[0]:]:
        while value > q + 0.5 + h:
            q, count = q + 1, count + 1
        while value < q - 0.5 - h:
            q, count = q - 1, count + 1
    return count


def r_squared(y, y_hat):
    """Coefficient of determination 1 - SS_res / SS_tot."""
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def detect_uturn(gyr, freq, legacy=None):
    """Detect the boundaries of the single U-turn of a go / U-turn / back walking test.

    Arguments:
        gyr {array} -- angular velocity about the X axis of the lower-back sensor (deg/s),
            pre-processed (low-pass filtered), one value per sample
        freq {float} -- acquisition frequency (Hz)
        legacy {bool} -- use the rules of the original implementation (default: config.LEGACY)

    Returns
    -------
    UTurnResult
    """
    legacy = config.LEGACY if legacy is None else bool(legacy)
    gyr = np.asarray(gyr, dtype=float)
    n = len(gyr)
    if n < 20:
        raise ValueError("The recording is too short ({} samples).".format(n))
    t = np.arange(n)

    # Step 1. Heading and drift correction ------------------------------------------------
    theta_raw = integrate(gyr, freq)
    drift_window = best_linear_window(theta_raw)
    beta, alpha = line_fit(theta_raw, *drift_window)
    theta = theta_raw - (beta * t + alpha)

    # Step 2. Centring and turn candidates (legacy: orientation at this stage) --------------
    theta_c = centre(theta)
    if legacy and np.mean(theta_c[:int(_LEGACY_DIRECTION_FRAC * n)]) > 0:
        theta_c = -theta_c
    crossings = sign_changes(theta_c, legacy)
    if len(crossings) == 0:
        raise UTurnNotFound("The centred heading never changes sign: no U-turn was found.")
    middle = int(crossings[np.argmin(np.abs(crossings - n / 2))])

    # Step 3. Fitting window ----------------------------------------------------------------
    n_back = n - middle
    if legacy:
        go = best_linear_window(theta_c[:middle], score="legacy", end_min=middle / 2.0)
        back = best_linear_window(theta_c[middle:], score="legacy", start_max=middle / 2.0)
        w_start = go[0] if go is not None else 0
        w_end = middle + back[1] if back is not None else n
    else:
        w_start = int(np.ceil((1.0 - config.SIDE_FRAC) * middle))
        w_end = middle + int(np.floor(config.SIDE_FRAC * n_back))
    fit_window = (int(w_start), int(w_end))

    # Step 3 (end). Centring on the fitting window and orientation -------------------------
    theta_w = centre(theta, *fit_window)
    if legacy:
        direction = -1 if np.mean(theta_w[:int(_LEGACY_DIRECTION_FRAC * n)]) > 0 else 1
    else:
        before = np.mean(theta_w[fit_window[0]:middle])
        after = np.mean(theta_w[middle:fit_window[1]])
        direction = 1 if after >= before else -1
    theta_w = direction * theta_w

    # Step 4. Non-linear least-squares fit of the turning model -----------------------------
    x = t[fit_window[0]:fit_window[1]]
    y = theta_w[fit_window[0]:fit_window[1]]
    p0 = (config.A0_DEG, float(middle), float(int(config.C0_S * freq)), 0.0)
    c_max = config.C_MAX_DPS / freq
    fit = least_squares(lambda p: turning_model(x, *p) - y, p0,
                        bounds=([-np.inf, -np.inf, -np.inf, -c_max], [np.inf, np.inf, np.inf, c_max]))
    a, b, c, C = fit.x
    if c < 0:                     # (a, b, c) and (-a, b, -c) describe the same model
        a, c = -a, -c
    start, end = int(round(b - c / 2)), int(round(b + c / 2))
    model = turning_model(x, a, b, c, C)

    lo, hi = max(start, 0), min(end, n - 1)
    peak = float(np.max(np.abs(gyr[lo:hi + 1]))) if hi >= lo else np.nan

    return UTurnResult(start=start, end=end, freq=freq, amplitude=float(abs(a)),
                       direction=int(direction * np.sign(a)) if a != 0 else int(direction),
                       orientation=int(direction),
                       r2=float(r_squared(y, model)), n_crossings=int(len(crossings)),
                       n_half_turns=count_half_turns(theta_w, config.HYSTERESIS_DEG),
                       peak_yaw_rate=peak, drift=float(beta * freq), params=(a, b, c, C), p0=p0,
                       middle=middle, crossings=crossings,
                       drift_window=(int(drift_window[0]), int(drift_window[1])),
                       fit_window=fit_window, legacy=legacy, theta_raw=theta_raw,
                       theta_c=theta_c, theta=theta_w, model=model, model0=turning_model(x, *p0))
