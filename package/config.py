"""Parameters of the U-turn segmentation algorithm (Table 1 of the IPOL article).

The constants of the drift estimation (window lengths, weight and regularisation of the
linearity score, Section 3.2 of the article) are fixed and defined in uturn_detection.py.
"""

#: Butterworth low-pass filter of the angular velocity: order and cut-off frequency (Hz).
FILTER_ORDER = 8
FILTER_FC = 14.0

#: Quantiles defining the two heading levels; the heading is centred on their midpoint,
#: Equation (5).
Q_LOW = 0.15
Q_HIGH = 0.85

#: Fraction kappa of each side of the turn kept in the fitting window, Equation (7).
SIDE_FRAC = 0.6

#: Initial amplitude a0 (deg) and duration c0 (s) of the turning model, Equation (12).
A0_DEG = 180.0
C0_S = 3.0

#: Bound C_max on the residual drift of the turning model (deg/s), Equation (11).
C_MAX_DPS = 1.0

#: Hysteresis H of the U-turn indicator (deg), Equation (15).
HYSTERESIS_DEG = 45.0

#: Unit of the angular velocity in the input file: "rad/s" (default) or "deg/s".
GYR_UNIT = "rad/s"


def as_dict():
    """Return every parameter as a dictionary, to be logged alongside a run."""
    return {k: v for k, v in globals().items()
            if k.isupper() and not k.startswith("_")}
