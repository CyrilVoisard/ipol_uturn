#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import sys
import numpy as np

from package import import_data, detection, quality

# if you need to access a file next to the source code, use the variable ROOT
ROOT = os.path.dirname(os.path.realpath(__file__))

# Save the current CWD
data_WD = os.getcwd()

# Change the CWD to ROOT
os.chdir(ROOT)
            

def print_seg_detection(uturn_lim, freq):
    """Dump the phase segmentation computed from the trial. 

    Parameters
    ----------
        uturn_lim {dict} -- dictionnary with uturn boundaries 
    """

    uturn_dict = {'U-Turn start': int(uturn_lim[0][0]),
                  'U-Turn end': int(uturn_lim[0][1])}

    display_dict = {'U-Turn start_title': "U-turn start",
                    'U-Turn start': "{U-Turn start}".format(**uturn_dict),
                    'U-Turn start_sec': "{}".format(round(uturn_dict['U-Turn start']/freq, 2)),
                    'U-Turn end_title': "U-turn end",
                    'U-Turn end': "{U-Turn end}".format(**uturn_dict),
                    'U-Turn end_sec': "{}".format(round(uturn_dict['U-Turn end']/freq, 2))}
        
    info_msg = """
    {U-Turn start_title:<15}| {U-Turn start:<10}| {U-Turn start_sec:<10}
    {U-Turn end_title:<15}| {U-Turn end:<10}| {U-Turn end_sec:<10}
    """

    # Dump information
    os.chdir(data_WD) # Get back to the normal WD

    with open("uturn_lim.txt", "wt") as f:
        print(info_msg.format(**display_dict), file=f)


def print_uturn(uturn_lim, data_lb, n, freq):
    """Dump the phase segmentation computed from the trial. 

    Parameters
    ----------
        uturn_lim {dict} -- dictionnary with uturn boundaries 
    """
    times = []
    for i in range(len(uturn_lim)):
        times.append(uturn_lim[0][1]-uturn_lim[0][0])

    uturn_dict = {'Duration': len(data_lb)/freq, 
                  'N_wanted': n, 
                  'N_found': len(uturn_lim),
                  'Percentage' : round(100*np.sum(times)/len(data_lb), 2), 
                  'Mean U-turn duration': round(np.mean(times)/freq, 2), 
                  'U-turn variation' : round(100*np.std(times), 2)}

    display_dict = {'Title_1': "Search results:",
                    'Duration': "Total duration (s): {Duration}".format(**uturn_dict),
                    'Wanted': "Wanted: {N_wanted}".format(**uturn_dict),
                    'Found': "Found: {N_found}".format(**uturn_dict),
                    'Title_2': "Statistics:",
                    'Percentage': "U-turn percentage (%): {Percentage}".format(**uturn_dict),
                    'Mean': "Mean U-turn duration (s): {Mean U-turn duration}".format(**uturn_dict),
                    'Variation': "U-turn variation (%): {U-turn variation}".format(**uturn_dict),
                    }
            
    info_msg = """
    {Title_1:^30}|{Title_2:^30}
    ------------------------------+------------------------------
    {Duration:<30}| {Percentage:<30}
    {Wanted:<30}| {Mean:<30}
    {Found:<30}| {Variation:<30}
    """

    # Dump information
    os.chdir(data_WD) # Get back to the normal WD

    with open("uturn_lim.txt", "wt") as f:
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
    data_lb = import_data.import_XSens(os.path.join(data_WD, args.i0), freq=100)

    # uturn boundaries detection and figure
    uturn_lim = detection.uturn_detection(data_lb, n, freq, output=data_WD)

    # print phases and figure
    detection.plot_uturn_detection(uturn_lim, data_lb, freq, output=data_WD)
    if (n == 1) & (len(uturn_lim) == 1):
        print_seg_detection(uturn_lim, freq)
    else:
        print("ok charge")
        print_uturn(uturn_lim, data_lb, n, freq)

    # print("ok charge")
    sys.exit(0)
