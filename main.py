#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""IPOL demo: U-turn segmentation of a go / U-turn / back walking test with a lower-back IMU.

Usage:
    python main.py -i0 trial_lb.txt -freq 100 [-unit rad/s] [-legacy 0]

Outputs (in the current directory):
    uturn.png               heading, fitted model, phases and angular velocity
    uturn_construction.png  intermediate steps of the algorithm
    uturn.txt               results and indicators
    uturn.json              results, indicators and parameters (download)
"""

import argparse
import os
import sys

from package import config, import_data, plot_uturn, report, uturn_detection

# folder of the source code, and folder in which the demo is executed
ROOT = os.path.dirname(os.path.realpath(__file__))
data_WD = os.getcwd()


def main():
    parser = argparse.ArgumentParser(description="U-turn segmentation from a lower-back IMU.")
    parser.add_argument("-i0", metavar="data_lb", required=True,
                        help="Text file of the lower-back sensor (columns PacketCounter and Gyr_X).")
    parser.add_argument("-freq", metavar="freq", type=float, required=True,
                        help="Acquisition frequency (Hz).")
    parser.add_argument("-unit", metavar="unit", default=config.GYR_UNIT, choices=["rad/s", "deg/s"],
                        help="Unit of the angular velocity in the file (default: rad/s).")
    parser.add_argument("-legacy", metavar="legacy", type=int, default=int(config.LEGACY),
                        help="1 to use the rules of the original implementation (default: 0).")
    args = parser.parse_args()

    try:
        gyr = import_data.import_lower_back(os.path.join(data_WD, args.i0), args.freq, args.unit)
        res = uturn_detection.detect_uturn(gyr, args.freq, legacy=bool(args.legacy))
    except (ValueError, OSError) as error:
        with open(os.path.join(data_WD, "demo_failure.txt"), "w") as f:
            f.write("The U-turn could not be segmented: {}\n".format(error))
        return 0

    plot_uturn.plot_result(res, gyr, output=data_WD)
    plot_uturn.plot_construction(res, output=data_WD)
    report.write_reports(res, len(gyr), output=data_WD)
    return 0


if __name__ == "__main__":
    sys.exit(main())
