#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import sys

from package import import_data, seg_detection, quality

# if you need to access a file next to the source code, use the variable ROOT
ROOT = .path.dirname(os.path.realpath(__file__))

# Save the current CWD
data_WD = os.getcwd()

# Change the CWD to ROOT
os.chdir(ROOT)
            

def print_seg_detection(uturn_lim, n, freq):
    """Dump the phase segmentation computed from the trial. 

    Parameters
    ----------
        uturn_lim {dict} -- dictionnary with uturn boundaries 
    """

    display_dict = {'Start_title': "Trial start",
                    'Start': "{Start}".format(**seg_lim_dict),
                    'Start_sec': "{}".format(round(seg_lim_dict['Start']/100, 2)),
                    'U-Turn start_title': "U-turn start",
                    'U-Turn start': "{U-Turn start}".format(**seg_lim_dict),
                    'U-Turn start_sec': "{}".format(round(seg_lim_dict['U-Turn start']/100, 2)),
                    'U-Turn end_title': "U-turn end",
                    'U-Turn end': "{U-Turn end}".format(**seg_lim_dict),
                    'U-Turn end_sec': "{}".format(round(seg_lim_dict['U-Turn end']/100, 2)),
                    'End_title': "Trial end",
                    'End': "{End}".format(**seg_lim_dict), 
                    'End_sec': "{}".format(round(seg_lim_dict['End']/100, 2))}
        
    info_msg = """
    {Start_title:<15}| {Start:<10}| {Start_sec:<10}
    {U-Turn start_title:<15}| {U-Turn start:<10}| {U-Turn start_sec:<10}
    {U-Turn end_title:<15}| {U-Turn end:<10}| {U-Turn end_sec:<10}
    {End_title:<15}| {End:<10}| {End_sec:<10}
    """

    # Dump information
    .chdir(data_WD) # Get back to the normal WD

    with open("seg_lim.txt", "wt") as f:
        print(info_msg.format(**display_dict), file=f)
        

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Return a semiogram for a given trial.')
    parser.add_argument('-i0', metavar='data_lb', help='Time series for the lower back sensor.')
    
    parser.add_argument('-freq', metavar='freq',
                        help='Acquistion frequency.')
    parser.add_argument('-n', metavar='n',
                        help='Estimated number of uturns.')
    args = parser.parse_args()

    freq = int(args.freq)
    n = int(args.n)
    
    # load data
    data_lb = import_data.import_XSens(.path.join(data_WD, args.i0))

    # uturn boundaries detection and figure
    uturn_lim = detection.uturn_detection(data_lb, n, freq, output=data_WD)

    # print phases and figure
    # print_seg_detection(uturn_lim, n, freq)

    # print("ok charge")
    sys.exit(0)
