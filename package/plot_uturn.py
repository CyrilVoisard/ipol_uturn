"""Figures of the U-turn segmentation: result figure and construction figure."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches


COLOR_SIGNAL = "#1f77b4"
COLOR_MODEL = "#d62728"
COLOR_INIT = "#7f7f7f"
COLOR_TURN = "#ff7f0e"
COLOR_WINDOW = "#2ca02c"

plt.rcParams.update({"font.size": 13, "axes.labelsize": 14, "axes.titlesize": 15,
                     "legend.fontsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 12})


def _phase_bar(ax, res, n):
    """Draw the phase bar: straight (go), U-turn, straight (back)."""
    f = res.freq
    spans = [(0, res.start, "straight (go)", 0.15), (res.start, res.end, "U-turn", 0.45),
             (res.end, n, "straight (back)", 0.15)]
    for a, b, label, alpha in spans:
        if b > a:
            ax.add_patch(patches.Rectangle((a / f, 0), (b - a) / f, 1, color="k", alpha=alpha, lw=0))
            ax.text((a + b) / (2 * f), 0.5, label, ha="center", va="center", fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel("Phases")


def plot_result(res, gyr, output, filename="uturn.png"):
    """Result figure: trajectory with fitted model, phase bar and angular velocity.

    Arguments:
        res {UTurnResult} -- output of uturn_detection.detect_uturn
        gyr {array} -- pre-processed angular velocity (deg/s)
        output {str} -- folder in which the figure is saved
    """
    n, f = len(gyr), res.freq
    time = np.arange(n) / f
    trajectory = res.orientation * res.theta            # trajectory in the original direction
    x = np.arange(*res.fit_window) / f
    model = res.orientation * res.model

    fig, ax = plt.subplots(3, 1, figsize=(14, 8.5), sharex=True,
                           gridspec_kw={"height_ratios": [6, 1, 4]})
    ax[0].plot(time, trajectory, color=COLOR_SIGNAL, lw=1.5, label="angular position (drift-corrected)")
    ax[0].plot(x, model, color=COLOR_MODEL, lw=2, ls="--", label="fitted turning model")
    ax[1].sharex(ax[0])
    ax[2].plot(time, gyr, color=COLOR_SIGNAL, lw=1)
    for a in (ax[0], ax[2]):
        a.axvspan(res.start / f, res.end / f, color=COLOR_TURN, alpha=0.25, lw=0)
        a.axvline(res.start / f, color="k", ls="--", lw=1.5)
        a.axvline(res.end / f, color="k", ls="--", lw=1.5)
        a.grid(alpha=0.4)
    ax[0].plot([], [], color="k", ls="--", label="U-turn boundaries")
    ax[0].set_ylabel("Angular position (deg)")
    ax[0].legend(loc="best")
    turn = "left" if res.direction > 0 else "right"
    ax[0].set_title("U-turn detection ({} turn)".format(turn), fontweight="bold")
    _phase_bar(ax[1], res, n)
    ax[2].set_ylabel("Angular velocity\n(deg/s)")
    ax[2].set_xlabel("Time (s)")
    ax[2].set_xlim(0, n / f)
    fig.tight_layout()
    path = os.path.join(output, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_construction(res, output, filename="uturn_construction.png"):
    """Construction figure: drift correction, turn candidates and fitting window, model fit."""
    n, f = len(res.theta_raw), res.freq
    i = np.arange(n)
    time = i / f
    a0, a1 = res.fit_window
    x = np.arange(a0, a1) / f

    fig, ax = plt.subplots(3, 1, figsize=(14, 11), sharex=True)

    # A. integrated angular position and estimated drift
    slope = res.drift / f
    d0, d1 = res.drift_window
    raw = res.theta_raw
    intercept = np.mean(raw[d0:d1]) - slope * np.mean(np.arange(d0, d1))
    ax[0].plot(time, raw, color=COLOR_SIGNAL, lw=1.5, label="integrated angular position")
    ax[0].plot(time, slope * i + intercept, color=COLOR_INIT, lw=2, ls="--",
               label="estimated drift ({:+.2f} deg/s)".format(res.drift))
    ax[0].axvspan(d0 / f, d1 / f, color=COLOR_INIT, alpha=0.2, lw=0, label="drift window")
    ax[0].set_ylabel("Angular position (deg)")
    ax[0].set_title("Step 1 - Drift correction", loc="left", fontweight="bold")

    # B. centred angular position, sign changes and fitting window
    ax[1].plot(time, res.theta_c, color=COLOR_SIGNAL, lw=1.5, label="drift-corrected, centred trajectory")
    ax[1].axhline(0, color="k", lw=1)
    ax[1].plot(res.crossings / f, np.zeros(len(res.crossings)), "o", ms=7, mfc="none",
               color=COLOR_MODEL, label="sign changes (n = {})".format(res.n_crossings))
    ax[1].plot(res.middle / f, 0, "o", ms=8, color=COLOR_MODEL, label="selected sign change")
    ax[1].axvspan(a0 / f, a1 / f, color=COLOR_WINDOW, alpha=0.15, lw=0, label="fitting window")
    ax[1].set_ylabel("Angular position (deg)")
    ax[1].set_title("Steps 2-3 - centering, turn candidates and fitting window", loc="left", fontweight="bold")

    # C. model fit
    ax[2].plot(x, res.theta[a0:a1], color=COLOR_SIGNAL, lw=1.5, label="trajectory on the window")
    ax[2].plot(x, res.model0, color=COLOR_INIT, lw=2, ls=":", label="initial model")
    ax[2].plot(x, res.model, color=COLOR_MODEL, lw=2, ls="--",
               label="fitted model (R$^2$ = {:.4f})".format(res.r2))
    ax[2].axvspan(res.start / f, res.end / f, color=COLOR_TURN, alpha=0.25, lw=0, label="U-turn")
    ax[2].set_ylabel("Angular position (deg)")
    ax[2].set_xlabel("Time (s)")
    ax[2].set_title("Step 4 - fit of the turning model", loc="left", fontweight="bold")

    for a in ax:
        a.grid(alpha=0.4)
        a.legend(loc="best", fontsize=11)
    ax[2].set_xlim(0, n / f)
    fig.tight_layout()
    path = os.path.join(output, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
