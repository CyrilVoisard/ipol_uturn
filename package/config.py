"""Numerical parameters of the U-turn segmentation algorithm.

Every constant used by the algorithm is defined here, with the equation of the IPOL article
in which it appears. Sample-based quantities are derived from these values and from the
acquisition frequency at run time, so that the algorithm behaves identically at any frequency.
"""

# ---------------------------------------------------------------------------
# Version of the window-selection rules
# ---------------------------------------------------------------------------

#: If True, reproduce exactly the implementation used in the clinical studies published before
#: this article ("legacy" rules, Section 5 of the article). If False (default), use the rules
#: described in the article:
#:   - the fitting window is the half of each side of the turn that is adjacent to the turn
#:     (SIDE_FRAC), instead of the result of a search with the score r^2 * L / (n * slope),
#:     which depends on the sign of the slope and selected about the same window;
#:   - the turn direction is decided by comparing the mean heading before and after the turn,
#:     inside the fitting window (instead of the mean of the first 30% of the recording);
#:   - sign changes are detected with a test that does not miss exact zeros.
LEGACY = False

# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------

#: Butterworth low-pass filter applied to the angular velocity: order and cut-off (Hz).
FILTER_ORDER = 8
FILTER_FC = 14.0

# ---------------------------------------------------------------------------
# Step 1: drift correction, Equation (drift)
# ---------------------------------------------------------------------------

#: Bounds of the window length, as fractions of the recording length N.
DRIFT_MIN_FRAC = 0.25
DRIFT_MAX_FRAC = 0.50

#: Increment of the window length, as a fraction of N.
WINDOW_STEP_FRAC = 0.05

#: Exponent of the length weight (L / N)^p in the linearity score, Equation (score).
DRIFT_WEIGHT_POWER = 2

#: Constant added to the residual variance to avoid a division by zero (deg^2).
EPSILON = 1e-8

# ---------------------------------------------------------------------------
# Step 2: centring, direction and turn candidates
# ---------------------------------------------------------------------------

#: Quantiles defining the two heading levels; the heading is centred on their midpoint.
Q_LOW = 0.15
Q_HIGH = 0.85

#: Legacy rule only: fraction of the recording used to decide the turn direction.
LEGACY_DIRECTION_FRAC = 0.30

#: Hysteresis of the protocol indicator (deg). The centred heading has two levels close to -90
#: and +90 deg; a half-turn is counted when the heading moves from one level k*180 - 90 to the
#: next one by more than 90 + H deg (Equation (halfturns)). H = 45 deg ignores the trunk
#: oscillations during walking and rotations smaller than 135 deg.
HYSTERESIS_DEG = 45.0

# ---------------------------------------------------------------------------
# Step 3: fitting window
# ---------------------------------------------------------------------------

#: Fraction of each side of the turn kept in the fitting window, Equation (window): the window
#: starts at (1 - SIDE_FRAC) * m and ends at m + SIDE_FRAC * (N - m), where m is the selected
#: sign change of the heading.
SIDE_FRAC = 0.5

#: Legacy rules only: bounds of the window length searched on each side of the turn, as
#: fractions of the length of that side.
LEGACY_SIDE_MIN_FRAC = 0.25
LEGACY_SIDE_MAX_FRAC = 0.50

# ---------------------------------------------------------------------------
# Step 4: turning model, Equations (model) and (step)
# ---------------------------------------------------------------------------

#: Initial amplitude of the turn (deg).
A0_DEG = 180.0

#: Initial duration of the turn (s).
C0_S = 3.0

#: Bound on the residual drift slope C of the model (deg/s). After the drift correction of
#: Step 1 the remaining slope is small; the original implementation bounded it at 0.01 deg per
#: sample, i.e. 1 deg/s at 100 Hz.
C_MAX_DPS = 1.0

# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------

#: Unit of the angular velocity in the input file: "rad/s" (XSens default export) or "deg/s".
GYR_UNIT = "rad/s"


def as_dict():
    """Return every parameter as a dictionary, to be logged alongside a run."""
    return {k: v for k, v in globals().items()
            if k.isupper() and not k.startswith("_")}
