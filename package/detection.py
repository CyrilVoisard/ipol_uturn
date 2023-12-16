import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

def plot_uturn_detection(uturn_lim, data_lb, regression, freq, output):
    # Graphic signals
    t_full, angle_x_full = signals_for_seg(data_lb)

    # Regression coefficient
    [a_go, b_go, mid_index, a_u, b_u, a_back, b_back] = regression

    # Raw signal and figure initialisation
    plt.rcParams["figure.figsize"] = (20, 10)
    fig, ax = plt.subplots()
    ax.plot(t_full, angle_x_full, linewidth=3, label='angular position')
    ax.grid()
    ax.set_yticks([0, 90, 180])
    ax.yaxis.set_tick_params(labelsize=12)
    ax.set_ylabel('Angular position (°)', fontsize=15)
    ax.set_xlabel('Time (s)', fontsize=15)
    ax.xaxis.set_tick_params(labelsize=12)
    fig.suptitle('Mediolateral angular position', fontsize = 16, fontweight='bold')

    # Phases segmentation delimitation 
    ax.vlines(uturn_lim[1] / freq, -50, 230, 'red', '-', linewidth=2, label="$u_{go}$ and $u_{back}$")
    ax.vlines(uturn_lim[2] / freq, -50, 230, 'red', '-', linewidth=2)
    # Marqueurs du début et de la fin en noir
    ax.vlines(uturn_lim[0] / freq, -50, 230, 'k', '-', linewidth=2, label="$start$ and $end$")
    ax.vlines(uturn_lim[3] / freq, -50, 230, 'k', '-', linewidth=2)
    fig.legend(fontsize=15)

    # save fig 1 without construction lines
    path_out = os.path.join(output, "phases_seg.svg")
    plt.savefig(path_out, dpi=80,
                    transparent=True, bbox_inches="tight")

    # construction lines  
    ax.vlines(t_full.iloc[mid_index], -50, 230, 'orange', '--', linewidth = 2)
    x = np.linspace(t_full.iloc[0], t_full.iloc[-1], len(t_full))
    y = a_go*x + b_go
    ax.plot(x, y, 'orange', linewidth = 2, label = "affine schematization")
    y = a_back*x + b_back
    ax.plot(x, y, 'orange', linewidth = 2)
    x = np.linspace((-50-b_u)/a_u, (230-b_u)/a_u, len(t_full[mid_index-125:mid_index+125]))
    y = a_u*x + b_u
    ax.plot(x, y, 'orange', linewidth = 2)
    
    # save the fig 2 with construction lines
    path_out = os.path.join(output, "phases_seg_construction.svg")
    plt.savefig(path_out, dpi=80,
                    transparent=True, bbox_inches="tight")


def uturn_detection(data_lb, n, freq, output):
    return None
    

def seg_detection(data_lb, steps_lim, freq):

    # boundaries
    start = int(np.min(steps_lim["TO"]))
    end = int(np.max(steps_lim["HS"]))
    
    # useful signals
    t_full, angle_x_full = signals_for_seg(data_lb)

    # qualité : respect du protocole avec une seule rotation pouvant correspondre à un demi tour détectée au niveau du tour.
    count = -1
    for i in range(len(angle_x_full)-1):
        if (angle_x_full[i] - 90)*(angle_x_full[i+1] - 90) < 0:
            count = count + 1
    q = max(0, 100-50*count)

    # middle argument
    mid_index = find_nearest(angle_x_full, 90)
    l_search = min(abs(end - mid_index), abs(start - mid_index))
    start = int(mid_index - l_search)
    end = int(mid_index + l_search)

    t = data_lb["PacketCounter"][start:end]
    angle_x = angle_x_full[start:end]

    L = len(t)
    # affine approximation of go phase
    a_go, b_go, r_go, p_value_go, std_err_go = linregress(t[0:2*L//5], angle_x[0:2*L//5])
    # affine approximation of back phase
    a_back, b_back, r_back, p_value_back, std_err_back = linregress(t[3*L//5:], angle_x[3*L//5:])
    # affine approximation of U-turn phase
    a_u, b_u, r_u, p_value_u, std_err_u = linregress(t_full[mid_index - freq:mid_index + freq],
                                                     angle_x_full[mid_index - freq:mid_index + freq])

    # intersection points
    x_inter_go = (b_go - b_u) / (a_u - a_go)
    x_inter_back = (b_back - b_u) / (a_u - a_back)
    approx_uturn_lim = [start, freq * x_inter_go, freq * x_inter_back, end]

    # seg = [0, start_uturn, end_uturn, len(data_lb["Gyr_X"])]
    seg = [0, int(100*x_inter_go), int(100*x_inter_back), len(data_lb["Gyr_X"])]

    return seg, [a_go, b_go, mid_index, a_u, b_u, a_back, b_back], q


def signals_for_seg(data_lb):

    gyr_x = data_lb['Gyr_X']
    angle_x_full = np.cumsum(gyr_x)
    a = np.median(angle_x_full[0:len(angle_x_full) // 2])  # Tout début du signal
    z = np.median(angle_x_full[len(angle_x_full) // 2:len(angle_x_full)])  # Fin du signal

    angle_x_full = np.sign(z) * (angle_x_full - a) * 180 / abs(z)
    t_full = data_lb["PacketCounter"]

    return t_full, angle_x_full


def find_nearest(array, value):
    i = 0
    while (value - array[i]) * value > 0:  # Tant qu'on est du même côté, c'est à dire qu'ils ont le même signe
        i += 1
    if abs(value - array[i]) < abs(value - array[i-1]):
        return i
    else:
        return i-1
