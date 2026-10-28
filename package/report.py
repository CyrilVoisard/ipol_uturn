"""Text and JSON reports of the U-turn segmentation."""

import json
import os
from datetime import datetime

from . import config


def result_dict(res, n_samples):
    """Collect the outputs of the segmentation in a JSON-serialisable dictionary."""
    f = res.freq
    return {
        "DetectionDate": str(datetime.now()),
        "Freq": f,
        "TrialDuration_s": round(n_samples / f, 3),
        "UturnBoundaries": [int(res.start), int(res.end)],
        "UturnBoundaries_s": [round(res.start / f, 3), round(res.end / f, 3)],
        "UturnDuration_s": round(res.duration, 3),
        "UturnAmplitude_deg": round(res.amplitude, 1),
        "UturnDirection": "left" if res.direction > 0 else "right",
        "PeakAngularVelocity_dps": round(res.peak_yaw_rate, 1),
        "Indicators": {
            "R2": round(res.r2, 4),
            "UTurns": int(res.n_half_turns),
            "SignChanges": int(res.n_crossings),
            "BoundariesInsideWindow": bool(res.fit_inside_window),
            "Drift_dps": round(res.drift, 3),
        },
        "FitWindow": [int(res.fit_window[0]), int(res.fit_window[1])],
        "ModelParameters": {"a_deg": res.params[0], "b_sample": res.params[1],
                            "c_samples": res.params[2], "C_deg_per_sample": res.params[3]},
        "Parameters": config.as_dict(),
    }


def write_reports(res, n_samples, output):
    """Write uturn.json (download) and uturn.txt (displayed by the demo)."""
    d = result_dict(res, n_samples)
    with open(os.path.join(output, "uturn.json"), "w") as f:
        json.dump(d, f, indent=2)

    warnings = []
    if d["Indicators"]["UTurns"] == 0:
        warnings.append("No complete U-turn: the heading never changes by more than {:.0f} deg. "
                        "The boundaries are not meaningful.".format(90 + config.HYSTERESIS_DEG))
    elif d["Indicators"]["UTurns"] > 1:
        warnings.append("Protocol: {} U-turns detected instead of 1 (e.g. the subject turned again "
                        "before or after the test). The segmented U-turn is the one closest to the "
                        "middle of the recording.".format(d["Indicators"]["UTurns"]))
    if not d["Indicators"]["BoundariesInsideWindow"]:
        warnings.append("Segmentation: a boundary lies on the edge of the fitting window; check "
                        "the figure.")
    lines = [
        "U-turn segmentation",
        "-------------------------------------------------",
        "{:<34}{:>8.2f} s".format("Trial duration", d["TrialDuration_s"]),
        "{:<34}{:>8.2f} s  (sample {})".format("U-turn start", d["UturnBoundaries_s"][0], res.start),
        "{:<34}{:>8.2f} s  (sample {})".format("U-turn end", d["UturnBoundaries_s"][1], res.end),
        "{:<34}{:>8.2f} s".format("U-turn duration", d["UturnDuration_s"]),
        "{:<34}{:>8.1f} deg ({} turn)".format("U-turn amplitude", d["UturnAmplitude_deg"],
                                              d["UturnDirection"]),
        "{:<34}{:>8.1f} deg/s".format("Peak angular velocity", d["PeakAngularVelocity_dps"]),
        "",
        "Indicators",
        "-------------------------------------------------",
        "{:<34}{:>8.4f}".format("Fit quality R2", d["Indicators"]["R2"]),
        "{:<34}{:>8d}".format("U-turns (protocol, expected 1)", d["Indicators"]["UTurns"]),
        "{:<34}{:>8.2f} deg/s".format("Gyroscope drift removed", d["Indicators"]["Drift_dps"]),
    ]
    if warnings:
        lines += ["", "Warnings"] + ["- " + w for w in warnings]
    with open(os.path.join(output, "uturn.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return d
