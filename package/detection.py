import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import patches
from scipy.stats import linregress
import rdp

def plot_uturn_detection(uturns, data_lb, freq, output):

    # data
    t = data_lb["PacketCounter"]
    angle = np.cumsum(data_lb["Gyr_X"]) - np.mean(np.cumsum(data_lb["Gyr_X"]))

    # fig initialization
    fig, ax = plt.subplots(1, figsize=(20, 7), sharex=True)
    ax.plot(t, angle)
    ax.set_ylabel('Angular position (°)', fontsize=15)
    ax.set_title("U-Turn detection", fontsize=15, weight='bold')
    ax.set_xlabel('Time (s)', fontsize=15)

    # min and max
    mi = np.min(angle)
    ma = np.max(angle)

    ax.add_patch(patches.Rectangle((0, mi),  # (x,y)
                                  uturns[0][0]/freq,  # width
                                  ma - mi,  # height
                                  alpha=0.1,
                                  facecolor='green', linestyle='dotted'))

    ax.add_patch(patches.Rectangle((uturns[-1][1]/freq, mi),  # (x,y)
                                  len(data_lb)/freq - uturns[-1][1]/freq,  # width
                                  ma - mi,  # height
                                  alpha=0.1,
                                  facecolor='green', linestyle='dotted'))

    for i in range(len(uturns)):
        ax.add_patch(patches.Rectangle((uturns[i][0]/freq, mi),  # (x,y)
                                  uturns[i][1]/freq - uturns[i][0]/freq,  # width
                                  ma - mi,  # height
                                  alpha=0.1,
                                  facecolor='red', linestyle='dotted'))

    for i in range(len(uturns)-1):
        ax.add_patch(patches.Rectangle((uturns[i][1]/freq, mi),  # (x,y)
                                  uturns[i+1][0]/freq - uturns[i][1]/freq,  # width
                                  ma - mi,  # height
                                  alpha=0.1,
                                  facecolor='green', linestyle='dotted'))

    # legend
    red_patch = mpatches.Patch(color='red', alpha=0.1, label='uturn phase')
    green_patch = mpatches.Patch(color='green', alpha=0.1, label='straight phase')
    red_circle = mpatches.Patch(color='red', label='Toe Off')

    ax.legend(handles=[red_patch, green_patch], loc="upper right")

    # save fig
    path_out = os.path.join(output, "uturn.svg")
    plt.savefig(path_out, dpi=80, transparent=True, bbox_inches="tight")


def uturn_detection(data_lb, n, freq, output):

    # window size for data smoothing
    w = window_estimation(data_lb)

    # data smoothing
    t = data_lb["PacketCounter"]
    angle = np.cumsum(data_lb["Gyr_X"]) - np.mean(np.cumsum(data_lb["Gyr_X"]))
    smooth_angle = angle.rolling(w, center=True).mean()
    smooth_angle.fillna(method = "pad", inplace=True)
    smooth_angle.fillna(method = "bfill", inplace=True)

    # progressive figure construction
    fig, ax = plt.subplots(2, figsize=(20, int(min(15, 10*n))), sharex=True)
    ax[0].plot(t, angle)
    ax[0].set_ylabel('Angular position (°)', fontsize=15)
    ax[0].set_title("U-Turn detection", fontsize=15, weight='bold')
    ax[1].plot(t, angle, label = "raw signal")
    ax[1].plot(t, smooth_angle, label = "merged signal")
    ax[1].set_ylabel('Angular position (°)', fontsize=15)
    ax[1].set_title("Linear interpolation", fontsize=15, weight='bold')
    ax[1].set_xlabel('Time (s)', fontsize=15)

    # 
    if n == 0 : 
        goal = 100
    else:
        goal = 2*(2 + 2*n)  # looking for a number of events corresponding to: start + end + 2*number of U-turns

    # Ramer-Douglas-Peucker Algorithm points selection
    # points = rdp_select(t, smooth_angle, goal)
    points = find_epsilon_inf(t, smooth_angle, goal)
    ax[1].scatter(points[:, 0], points[:, 1], c="blue", label="RDP selection")

    # preselection 
    diff = abs(np.diff(points[:, 1]))
    thres = 0.4*max(diff)
    diff_start = abs(np.diff(points[:, 1], append = points[-1, 1]))
    diff_end = abs(np.diff(points[:, 1], prepend = points[0, 1]))
    start_points = points[diff_start>thres]
    end_points = points[diff_end>thres]
    ax[1].scatter(start_points[:, 0], start_points[:, 1], c="green", label = "start points estimation")
    ax[1].scatter(end_points[:, 0], end_points[:, 1], c="red", label = "end points estimation")

    end_times = np.insert(end_points[:, 0], 0, 0, axis=0)
    end_times = end_times*100
    end_times = end_times.astype(int)
    start_times = np.insert(start_points[:, 0], len(end_points[:, 0]), 0.01*len(angle), axis=0)
    start_times = start_times*100
    start_times = start_times.astype(int)

    print("times", start_times, end_times)

    coef = 1/5

    uturns = []
    for i in range(len(start_times)-1):
        # before u-turn
        start_go = int(end_times[i] + coef*(start_times[i]-end_times[i]))
        end_go = int(start_times[i] - coef*(start_times[i]-end_times[i]))
        a_go, b_go, r_go, p_value_go, std_err_go = linregress(t[start_go:end_go], smooth_angle[start_go:end_go])    
        x = np.linspace((start_go - 2*coef*(start_times[i]-end_times[i]))/100, (end_go + 2*coef*(start_times[i]-end_times[i]))/100)
        y = a_go*x + b_go
        ax[1].plot(x, y, 'grey', linewidth = 2)
        
        # after u-turn
        start_back = int(end_times[i+1] + coef*(start_times[i+1]-end_times[i+1]))
        end_back = int(start_times[i+1] - coef*(start_times[i+1]-end_times[i+1]))
        print("cherche", i, end_times[i+1], start_back, end_back, start_times[i+1])
        a_back, b_back, r_back, p_value_back, std_err_back = linregress(t[start_back:end_back], smooth_angle[start_back:end_back])    
        x = np.linspace((start_back - 2*coef*(start_times[i+1]-end_times[i+1]))/100, (end_back + 2*coef*(start_times[i+1]-end_times[i+1]))/100)
        y = a_back*x + b_back
        ax[1].plot(x, y, 'grey', linewidth = 2)
        
        # u-turn phase
        #a = smooth_angle[int(start_times[i]-50)]
        #z = smooth_angle[int(end_times[i+1]+50)]
        #mid=(a+z)/2
        #print("bornes", a, mid, z)
        #mid_index = find_nearest(smooth_angle[start_times[i]:end_times[i+1]], mid)
        #print("mid_index", mid_index)
        #mid_index = start_times[i] + mid_index
        #print("mid_index", mid_index)
        #start_u = int(mid_index - (1 - coef)*min(end_times[i+1] - mid_index, mid_index - start_times[i]))
        #end_u = int(mid_index + (1 - coef)*min(end_times[i+1] - mid_index, mid_index - start_times[i]))
        start_u = int(start_times[i] + coef*(end_times[i+1]-start_times[i]))
        end_u = int(end_times[i+1] - coef*(end_times[i+1]-start_times[i]))
        #print("start endu", start_u, end_u)
        a_u, b_u, r_u, p_value_u, std_err_u = linregress(t[start_u:end_u], smooth_angle[start_u:end_u])    
    
        x_inter_go = (b_go - b_u)/(a_u - a_go)
        x_inter_back = (b_back - b_u)/(a_u - a_back)

        x = np.linspace(x_inter_go-0.1, x_inter_back+0.1)
        y = a_u*x + b_u
        ax[1].plot(x, y, 'grey', linewidth = 2)

        ax[0].scatter(x_inter_go, angle[int(freq*x_inter_go)], c="green")
        ax[0].scatter(x_inter_back, angle[int(freq*x_inter_back)], c="red")

        uturns.append([int(freq*x_inter_go), int(freq*x_inter_back)])

    # save fig
    path_out = os.path.join(output, "uturn_construction.svg")
    plt.savefig(path_out, dpi=80, transparent=True, bbox_inches="tight")
    
    return uturns

def rdp_select(t, angle, goal):
    array = np.zeros((len(angle), 2))
    array[:, 0] = t
    array[:, 1] = angle

    epsilon_sup = find_epsilon_sup(array, goal)
    epsilon_inf = 0
    select_sup, select_inf, n_sup, n_inf, epsilon_sup, epsilon_inf = find_epsilon_bornes(array, goal, epsilon_inf, epsilon_sup)

    return select_sup


def find_epsilon_inf(t, angle, but):
    array = np.zeros((len(angle), 2))
    array[:, 0] = t
    array[:, 1] = angle
    
    epsilon_sup = 10
    rdp_sup = rdp.rdp(array, epsilon = epsilon_sup, algo="rec")
    found_sup = len(rdp_sup)
    while found_sup < but :
        print("no", found_sup, epsilon_sup, rdp_sup)
        epsilon_sup = epsilon_sup/2
        rdp_sup = rdp.rdp(array, epsilon = epsilon_sup, algo="rec")
        found_sup = len(rdp_sup)
    print(found_sup, epsilon_sup, rdp_sup)
    return rdp_sup


# On optimise la valeur de epsilon par récursivité
def find_epsilon_bornes(array, but, epsilon_inf, epsilon_sup): 
    print(epsilon_inf, epsilon_sup)
    rdp_sup = rdp.rdp(array, epsilon = epsilon_sup, algo="rec")
    found_sup = len(rdp_sup)
    rdp_inf = rdp.rdp(array, epsilon = epsilon_inf, algo="rec")
    found_inf = len(rdp_inf)
    if (but <= found_inf < 2*but) & (but <= found_sup < 2*but):
        return rdp_sup, rdp_inf, found_sup, found_inf, epsilon_sup, epsilon_inf
    else:
        print(but, found_inf, found_sup, 2*but, (but <= found_inf <= 2*but), (but <= found_sup <= 2*but))
        epsilon_mid = (epsilon_sup + epsilon_inf)/2
        rdp_mid = rdp.rdp(array, epsilon = epsilon_mid, algo="rec")
        found_mid = len(rdp_mid)
        if found_mid >= 1.5*but >= found_sup:
            print("Entre mid et sup", found_mid, 1.5*but, found_sup)
            return find_epsilon_bornes(array, but, epsilon_mid, epsilon_sup)
        else:
            print("Entre inf et mid", found_inf, 1.5*but, found_mid)
            return find_epsilon_bornes(array, but, epsilon_inf, epsilon_mid)
            

# On recherche une borne sup de epsilon
def find_epsilon_sup(array, but):
    epsilon_sup = 1
    rdp_sup = rdp.rdp(array, epsilon = epsilon_sup, algo="rec")
    found_sup = len(rdp_sup)
    while found_sup > but :
        epsilon_sup = epsilon_sup + 1
        rdp_sup = rdp.rdp(array, epsilon = epsilon_sup, algo="rec")
        found_sup = len(rdp_sup)
    print(found_sup, epsilon_sup, rdp_sup)
    return epsilon_sup


def window_estimation(data_lb):
    acf_x = autocorr(data_lb["FreeAcc_X"])
    acf_y = autocorr(data_lb["FreeAcc_Y"])
    # acf_tot = acf_x+acf_y
    acf_tot = [0 for i in range(50)]
    for i in range(50, len(acf_x)):
        if acf_y[i] < 0:
            acf_tot.append(0)
        else:
            acf_tot.append(acf_x[i]*(1 - 50/np.sqrt(i))) 
    
    i = indexes(acf_tot[50:400], thres=0.99, thres_abs=False)
    
    return 50 + i[0]


def autocorr(f):
    N = len(f)
    fvi = np.fft.fft(f, n=2 * N)
    acf = np.real(np.fft.ifft(fvi * np.conjugate(fvi))[:N])
    d = N - np.arange(N)
    acf = acf / d # non biased indicator
    acf = acf / acf[0]
    return acf


def indexes(y, thres=0.3, min_dist=1, thres_abs=False):
    """Peak detection routine.

    Finds the numeric index of the peaks in *y* by taking its first order difference. By using
    *thres* and *min_dist* parameters, it is possible to reduce the number of
    detected peaks. *y* must be signed.

    Parameters
    ----------
    y : ndarray (signed)
        1D amplitude data to search for peaks.
    thres : float between [0., 1.]
        Normalized threshold. Only the peaks with amplitude higher than the
        threshold will be detected.
    min_dist : int
        Minimum distance between each detected peak. The peak with the highest
        amplitude is preferred to satisfy this constraint.
    thres_abs: boolean
        If True, the thres value will be interpreted as an absolute value, instead of
        a normalized threshold.

    Returns
    -------
    ndarray
        Array containing the numeric indexes of the peaks that were detected.
        When using with Pandas DataFrames, iloc should be used to access the values at the returned positions.
    """
    if isinstance(y, np.ndarray) and np.issubdtype(y.dtype, np.unsignedinteger):
        raise ValueError("y must be signed")

    if not thres_abs:
        thres = thres * (np.max(y) - np.min(y)) + np.min(y)

    min_dist = int(min_dist)

    # compute first order difference
    dy = np.diff(y)

    # propagate left and right values successively to fill all plateau pixels (0-value)
    zeros, = np.where(dy == 0)

    # check if the signal is totally flat
    if len(zeros) == len(y) - 1:
        return np.array([])

    if len(zeros):
        # compute first order difference of zero indexes
        zeros_diff = np.diff(zeros)
        # check when zeros are not chained together
        zeros_diff_not_one, = np.add(np.where(zeros_diff != 1), 1)
        # make an array of the chained zero indexes
        zero_plateaus = np.split(zeros, zeros_diff_not_one)

        # fix if leftmost value in dy is zero
        if zero_plateaus[0][0] == 0:
            dy[zero_plateaus[0]] = dy[zero_plateaus[0][-1] + 1]
            zero_plateaus.pop(0)

        # fix if rightmost value of dy is zero
        if len(zero_plateaus) and zero_plateaus[-1][-1] == len(dy) - 1:
            dy[zero_plateaus[-1]] = dy[zero_plateaus[-1][0] - 1]
            zero_plateaus.pop(-1)

        # for each chain of zero indexes
        for plateau in zero_plateaus:
            median = np.median(plateau)
            # set leftmost values to leftmost non zero values
            dy[plateau[plateau < median]] = dy[plateau[0] - 1]
            # set rightmost and middle values to rightmost non zero values
            dy[plateau[plateau >= median]] = dy[plateau[-1] + 1]

    # find the peaks by using the first order difference
    peaks = np.where(
        (np.hstack([dy, 0.0]) < 0.0)
        & (np.hstack([0.0, dy]) > 0.0)
        & (np.greater(y, thres))
    )[0]

    # handle multiple peaks, respecting the minimum distance
    if peaks.size > 1 and min_dist > 1:
        highest = peaks[np.argsort(y[peaks])][::-1]
        rem = np.ones(y.size, dtype=bool)
        rem[peaks] = False

        for peak in highest:
            if not rem[peak]:
                sl = slice(max(0, peak - min_dist), peak + min_dist + 1)
                rem[sl] = True
                rem[peak] = False

        peaks = np.arange(y.size)[~rem]

    return peaks



    
    
def plot_uturn_detection_old(uturn_lim, data_lb, regression, freq, output):
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


def signals_for_seg(data_lb):

    gyr_x = data_lb['Gyr_X']
    angle_x_full = np.cumsum(gyr_x)
    a = np.median(angle_x_full[0:len(angle_x_full) // 2])  # Tout début du signal
    z = np.median(angle_x_full[len(angle_x_full) // 2:len(angle_x_full)])  # Fin du signal

    angle_x_full = np.sign(z) * (angle_x_full - a) * 180 / abs(z)
    t_full = data_lb["PacketCounter"]

    return t_full, angle_x_full


def find_nearest(array, value):
    array=np.array(array)
    print("array", array)
    i = 0
    print(array[i])
    while (value - array[i]) * (value - array[0]) > 0:  # Tant qu'on est du même côté, c'est à dire qu'ils ont le même signe
        print(array[i])
        i += 1
    if abs(value - array[i]) < abs(value - array[i-1]):
        return i
    else:
        return i-1
