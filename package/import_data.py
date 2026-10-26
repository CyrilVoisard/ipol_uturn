"""Loading and pre-processing of the lower-back inertial signal."""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from . import config


def load_xsens(filename):
    """Load an XSens text export and resample it on a regular grid of packet counters.

    The header lines written by MT Manager are skipped: the table starts at the line beginning
    with "PacketCounter". Missing packets are linearly interpolated so that sample i of the
    output corresponds to time i / F. A wrap-around of the 16-bit packet counter is unwrapped.

    Arguments:
        filename {str} -- path of the text file (tab, space or comma separated)

    Returns
    -------
    pandas.DataFrame
        one row per sample; column "PacketCounter" holds 0, 1, ..., N - 1
    """
    with open(filename, "r") as file_id:
        for i, line in enumerate(file_id):
            if line.startswith("PacketCounter"):
                break
        else:
            raise ValueError("The column 'PacketCounter' was not found in {}.".format(filename))

    signal = pd.read_csv(filename, delimiter=r"[,\s]+", skiprows=i, header=0, engine="python")
    signal = signal.dropna(axis=1, how="all")
    if "Gyr_X" not in signal.columns:
        raise ValueError("The column 'Gyr_X' was not found in {}.".format(filename))

    # unwrap the 16-bit packet counter, then express it from 0
    counter = signal["PacketCounter"].to_numpy(dtype=float)
    jumps = np.diff(counter) < -32768
    counter = counter + 65536 * np.concatenate(([0], np.cumsum(jumps)))
    counter = counter - counter[0]
    if np.any(np.diff(counter) <= 0):
        raise ValueError("The packet counter is not increasing in {}.".format(filename))

    grid = np.arange(0, int(counter[-1]) + 1)
    data = {"PacketCounter": grid}
    for column in signal.columns:
        if column == "PacketCounter":
            continue
        values = pd.to_numeric(signal[column], errors="coerce").to_numpy(dtype=float)
        if np.all(np.isnan(values)):
            continue
        data[column] = np.interp(grid, counter, values)

    return pd.DataFrame(data=data)


def low_pass_filter(sig, order, fc, freq):
    """Zero-phase Butterworth low-pass filter.

    Arguments:
        sig {array} -- signal
        order {int} -- order of the filter
        fc {float} -- cut-off frequency (Hz)
        freq {float} -- acquisition frequency (Hz)

    Returns
    -------
    ndarray
        filtered signal (unchanged if fc is above the Nyquist frequency)
    """
    f_nyq = freq / 2.0
    if fc >= f_nyq:
        return np.asarray(sig, dtype=float)
    b, a = butter(N=order, Wn=fc / f_nyq, btype="low", analog=False)
    return filtfilt(b, a, np.asarray(sig, dtype=float))


def import_lower_back(path, freq, gyr_unit=None):
    """Load the lower-back recording and return the filtered yaw angular velocity in deg/s.

    Arguments:
        path {str} -- path of the XSens text file of the lower-back sensor
        freq {float} -- acquisition frequency (Hz)
        gyr_unit {str} -- "rad/s" or "deg/s" (default: config.GYR_UNIT)

    Returns
    -------
    ndarray
        angular velocity about the sensor X axis (deg/s), one value per sample
    """
    gyr_unit = config.GYR_UNIT if gyr_unit is None else gyr_unit
    data = load_xsens(path)
    gyr = data["Gyr_X"].to_numpy(dtype=float)
    if len(gyr) < max(3 * (config.FILTER_ORDER + 1) * 3, 2 * freq):
        raise ValueError("The recording is too short ({:.2f} s).".format(len(gyr) / freq))
    if gyr_unit == "rad/s":
        gyr = np.rad2deg(gyr)
    elif gyr_unit != "deg/s":
        raise ValueError("Unknown angular velocity unit: {}".format(gyr_unit))
    return low_pass_filter(gyr, config.FILTER_ORDER, config.FILTER_FC, freq)
