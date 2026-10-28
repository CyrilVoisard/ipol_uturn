"""Parameters of the U-turn segmentation algorithm (Table 1 of the IPOL article).

The constants of the drift estimation (window lengths, weight and regularisation of the
linearity score, Section 3.2 of the article) are fixed and defined in uturn_detection.py.
"""

#: If True, use the rules of the original implementation, as in the clinical studies published
#: before the IPOL article (Section 5 of the article). If False (default), use the rules
#: described in the article.
LEGACY = False

#: Butterworth low-pass filter of the angular velocity: order and cut-off frequency (Hz).
FILTER_ORDER = 8
FILTER_FC = 14.0

#: Quantiles defining the two heading levels; the heading is centred on their midpoint,
#: Equation (centring).
Q_LOW = 0.15
Q_HIGH = 0.85

#: Fraction kappa of each side of the turn kept in the fitting window, Equation (window).
SIDE_FRAC = 0.5

#: Initial amplitude a0 (deg) and duration c0 (s) of the turning model, Equation (init).
A0_DEG = 180.0
C0_S = 3.0

#: Bound C_max on the residual drift of the turning model (deg/s), Equation (bounds).
C_MAX_DPS = 1.0

#: Hysteresis H of the half-turn indicator (deg), Equation (halfturns).
HYSTERESIS_DEG = 45.0

#: Unit of the angular velocity in the input file: "rad/s" (XSens default) or "deg/s".
GYR_UNIT = "rad/s"


def as_dict():
    """Return every parameter as a dictionary, to be logged alongside a run."""
    return {k: v for k, v in globals().items()
            if k.isupper() and not k.startswith("_")}
