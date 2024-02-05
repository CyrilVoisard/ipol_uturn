import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import patches
from scipy.stats import linregress
from scipy import interpolate
import rdp


def plot_uturn_detection(uturns, data_lb, freq, output):
    """Plot the uturn detection as a .png figure.

    Arguments:
        uturns {array} -- numpy array with uturn boundaries time samples
        data_lb {dataframe} -- pandas dataframe with pre-processed lower back sensor time series
        freq {int} -- acquisition frequency
        output {str} -- folder path where to save final plot

    Returns
    -------
    None
        Plot the figure.
    """
    # data
    t = data_lb["PacketCounter"]
    angle = np.cumsum(data_lb["Gyr_X"]) - np.mean(np.cumsum(data_lb["Gyr_X"]))

    # fig initialization
    fig, ax = plt.subplots(1, figsize=(20, 7), sharex=True)
    ax.plot(t, angle)
    ax.set_ylabel('AngularAngular position (x100 rad)', fontsize=15)
    ax.set_title("U-Turn detection", fontsize=15, weight='bold')
    ax.set_xlabel('Time (s)', fontsize=15)

    # min and max
    mi = np.min(angle)
    ma = np.max(angle)

    # plot uturns
    ax.add_patch(patches.Rectangle((0, mi),  # (x,y)
                                   uturns[0][0] / freq,  # width
                                   ma - mi,  # height
                                   alpha=0.1,
                                   facecolor='green', linestyle='dotted'))

    ax.add_patch(patches.Rectangle((uturns[-1][1] / freq, mi),  # (x,y)
                                   len(data_lb) / freq - uturns[-1][1] / freq,  # width
                                   ma - mi,  # height
                                   alpha=0.1,
                                   facecolor='green', linestyle='dotted'))

    for i in range(len(uturns)):
        ax.add_patch(patches.Rectangle((uturns[i][0] / freq, mi),  # (x,y)
                                       uturns[i][1] / freq - uturns[i][0] / freq,  # width
                                       ma - mi,  # height
                                       alpha=0.1,
                                       facecolor='red', linestyle='dotted'))

    for i in range(len(uturns) - 1):
        ax.add_patch(patches.Rectangle((uturns[i][1] / freq, mi),  # (x,y)
                                       uturns[i + 1][0] / freq - uturns[i][1] / freq,  # width
                                       ma - mi,  # height
                                       alpha=0.1,
                                       facecolor='green', linestyle='dotted'))

    # legend
    red_patch = mpatches.Patch(color='red', alpha=0.1, label='uturn phase')
    green_patch = mpatches.Patch(color='green', alpha=0.1, label='straight phase')

    ax.legend(handles=[red_patch, green_patch], loc="upper right")

    # save fig
    path_out = os.path.join(output, "uturn.svg")
    plt.savefig(path_out, dpi=80, bbox_inches="tight")


def uturn_detection(data_lb, n, freq, output):
    """Detect uturns boundaries in a trial from the lower back sensor.

    Arguments:
        data_lb {dataframe} -- pandas dataframe with pre-processed lower back sensor time series
        n {int} -- number of attempted uturns. put 0 if no idea.
        freq {int} -- acquisition frequency
        output {str} -- folder path where to save construction plot

    Returns
    -------
    uturns, uturn_val, q_tot
        uturns {array} -- numpy array with uturn boundaries time samples
        uturn_val {array} -- numpy array with uturn amplitude values
        q_tot {array} -- numpy array with quality indicators [q1, q2, q3]
    """

    # step 1.1. window size for data smoothing
    w = window_estimation(data_lb, freq)

    # step 1.2. data smoothing
    t, angle, smooth_angle, fig, ax = signals_for_seg(data_lb, w)

    # step 2.1. Ramer-Douglas-Peucker Algorithm points selection
    points = points_selection(t, smooth_angle, n)
    ax[1].scatter(points[:, 0], points[:, 1], c="blue", label="RDP selection")

    # step 2.2. time intervals selection
    start_times, end_times, uturn_val, ax[1] = time_intervals_selection(points, freq, ax[1])
    start_times = np.insert(start_times, len(start_times), len(angle), axis=0)  # add last time sample

    # step 2.3. time intervals validation
    start_times, end_times, valide = time_intervals_validation(start_times, end_times, uturn_val, smooth_angle, w)

    # step 3. affine approximation for U-turns identification
    uturns = []
    for i in range(len(start_times) - 1):
        if valide[i]:
            # step 3.1. affine approximation
            affine_coeff, ax[1] = affine_approximation(t, smooth_angle, start_times, end_times, i, w, freq, ax[1])

            # step 3.2. detection of u-turn boundaries
            uturn, ax = uturn_boundaries(affine_coeff, angle, freq, ax)

            if len(uturn) == 2:
                uturns.append(uturn)

    # save fig construction
    path_out = os.path.join(output, "uturn_construction.svg")
    fig.savefig(path_out, dpi=80, bbox_inches="tight")

    # quality evaluation
    q_tot = quality_evaluation(uturns, uturn_val, n)

    return uturns, uturn_val, q_tot


def quality_evaluation(uturns, uturn_val, n):
    """Detect uturns boundaries in a trial from the lower back sensor.

    Arguments:
        uturns {array} -- numpy array with uturn boundaries time samples
        uturn_val {array} -- numpy array with uturn amplitude values
        n {int} -- number of attempted uturns. put 0 if no idea.

    Returns
    -------
    q_tot
        q_tot {array} -- numpy array with quality indicators [q1, q2, q3]
    """

    q_tot = [float('nan'), float('nan'), float('nan')]
    if n != 0:
        q_tot[0] = round(100 * (min(n, len(uturns)) / max(n, len(uturns))))
    if n > 1:
        q_tot[1] = round(100 * np.std(abs(uturn_val)) / np.mean(abs(uturn_val)))
    amp_ref = 300
    q_tot[2] = round(100 * abs(amp_ref - np.mean(abs(uturn_val))) / amp_ref)
    print("test", q_tot, np.mean(uturn_val), abs(amp_ref - np.mean(uturn_val)) / amp_ref)
    print(test)

    return q_tot


def uturn_boundaries(affine_coeff, angle, freq, ax):
    """Detect one uturn boundaries in a trial from the affine approximation coefficients.

    Arguments:
        affine_coeff {list} -- coefficients a and b for affine approximation 1, 2 and 3 : y = a*x+b
        angle {time series} -- time series corresponding to estimated angular position of the sensor in the axial plane
        freq {int} -- acquisition frequency
        ax {matplotlib ax} -- ax from the construction plot

    Returns
    -------
    uturn, ax
        uturns {array} -- numpy array with the uturn boundaries
        ax {matplotlib ax} -- modified ax from the construction plot
    """

    # intersection points
    [a_1, b_1, a_2, b_2, a_3, b_3] = affine_coeff
    x_inter_go = (b_1 - b_2) / (a_2 - a_1)
    x_inter_back = (b_3 - b_2) / (a_2 - a_3)

    # progressive figure construction
    if not (np.isnan(x_inter_go) | np.isnan(x_inter_back)):
        x = np.linspace(x_inter_go - 0.1, x_inter_back + 0.1)
        y = a_2 * x + b_2
        ax[1].plot(x, y, 'grey', linewidth=2)

        ax[0].scatter(x_inter_go, angle[int(freq * x_inter_go)], c="green")
        ax[0].scatter(x_inter_back, angle[int(freq * x_inter_back)], c="red")

        uturn = [int(freq * x_inter_go), int(freq * x_inter_back)]

        return uturn, ax
    else:
        return []


def affine_approximation(t, smooth_angle, start_times, end_times, i, w, freq, ax):
    """Compute affine approximation for the phases during a given uturn.
    Phase 1 and 3 are supposed to be straight phases, phase 2 is the uturn phase.

    Arguments:
        t {time series} -- time series corresponding to data_lb["PacketCounter"]
        smooth angle {time series} -- time series corresponding to estimated angular position of the sensor in the
            axial plane smoothed with a windowing process of size w
        start_times {list} -- list containing in this order all the start time samples for uturn intervals and the
            length of the smooth angle time series
        end_times {list} -- list containing in this order the zero time sample and all the end time samples for
            uturn intervals
        i {int} -- index of the considered uturn
        w {int} -- mean stride time estimation
        freq {int} -- acquisition frequency
        ax {matplotlib ax} -- ax from the construction plot

    Returns
    -------
    affine_coeff, ax
        affine_coeff {list} -- coefficients a and b for affine approximation 1, 2 and 3 : y = a*x+b
        ax {matplotlib ax} -- modified ax from the construction plot
    """

    # phase 1: straight phase before u-turn
    start_1 = max(end_times[i],
                  min(int(end_times[i] + 0.5 * (start_times[i] - end_times[i])), int(start_times[i] - w * 10)))
    end_1 = int(start_times[i])
    a_1, b_1, r_1, p_value_1, std_err_1 = linregress(t[start_1:end_1], smooth_angle[start_1:end_1])
    # progressive figure construction
    x = np.linspace(start_1 / freq, end_1 / freq)
    y = a_1 * x + b_1
    ax.plot(x, y, 'grey', linewidth=2)

    # phase 3: straight phase after u-turn
    start_3 = int(end_times[i + 1])
    end_3 = min(start_times[i + 1], max(int(start_times[i + 1] - 0.5 * (start_times[i + 1] - end_times[i + 1])),
                                        int(end_times[i + 1] + w * 10)))
    a_3, b_3, r_3, p_value_3, std_err_3 = linregress(t[start_3:end_3], smooth_angle[start_3:end_3])
    # progressive figure construction
    x = np.linspace(start_3 / freq, end_3 / freq)
    y = a_3 * x + b_3
    ax.plot(x, y, 'grey', linewidth=2)

    # phase 2: u-turn phase
    start_2 = int(start_times[i])
    end_2 = int(end_times[i + 1])
    # detail phase with interpolation to add some points
    x_2 = np.array(t[start_2:end_2])
    y_2 = np.array(smooth_angle[start_2:end_2])
    f = interpolate.interp1d(y_2, x_2, fill_value='extrapolate')
    new_y_2 = np.arange(np.min(y_2), np.max(y_2), (np.max(y_2) - np.min(y_2)) / (10 * len(y_2)))
    new_x_2 = f(new_y_2)
    a_2, b_2, r_2, p_value_2, std_err_2 = linregress(new_x_2, new_y_2)

    affine_coeff = [a_1, b_1, a_2, b_2, a_3, b_3]

    return affine_coeff, ax


def time_intervals_validation(start_times, end_times, uturn_val, smooth_angle, w):
    """Validate or modify time intervals containing uturns.

    Arguments:
        start_times {list} -- list containing in this order all the start time samples for uturn intervals and the
            length of the smooth angle time series
        end_times {list} -- list containing in this order the zero time sample and all the end time samples for
            uturn intervals
        uturn_val {array} -- numpy array with uturn amplitude values
        smooth angle {time series} -- time series corresponding to estimated angular position of the sensor in the
            axial plane smoothed with a windowing process of size w
        w {int} -- mean stride time estimation

    Returns
    -------
    start_times, end_times, valide
        start_times {list} -- modified list containing in this order all the validated start time samples for uturn
            intervals and the length of the smooth angle time series
        end_times {list} -- modified list containing in this order the zero time sample and all the validated end time samples
            for uturn intervals
        valide {list} -- list of boolean to indicated if the intervals start_times and end_times contain uturns
    """

    valide = np.ones(len(start_times))
    valide[-1] = 0
    print("old times", start_times, end_times, valide)

    # case of two intervals too close together
    # print("test", np.min(start_times[valide > 0] - end_times[valide > 0] + 2 * w),
    # start_times[valide > 0] - end_times[valide > 0])
    while np.min(start_times[valide > 0] - end_times[valide > 0] - 2 * w) < 0:
        if len(start_times[valide == 1]) == 1:
            break
        for j in range(0, len(start_times) - 1):
            if start_times[j] < (end_times[j] + 2 * w):  # if 2 intervals are too close together
                if j == 0:  # special case of the first interval, close to the trial boundaries
                    valide[0] = 0  # keep the boundaries but invalidate the uturn interval
                else:
                    slope1 = (smooth_angle[end_times[j]] - smooth_angle[start_times[j - 1]]) / (
                            end_times[j] - start_times[j - 1])
                    slope2 = (smooth_angle[end_times[j + 1]] - smooth_angle[start_times[j]]) / (
                            end_times[j + 1] - start_times[j])
                    # print("slope", slope1, slope2)
                    if min(abs(slope1 / slope2), abs(slope2 / slope1)) > 0.7:  # slopes are similar in absolute values
                        print("Arranging...", j, uturn_val[j - 1], uturn_val[j])
                        if slope1 * slope2 > 0:  # slopes are same direction, they are the same interval : we mix them
                            start_times = np.delete(start_times, j)
                            end_times = np.delete(end_times, j)
                            valide = np.delete(valide, j)
                            break
                        else:  # slopes are in different directions : we invalidate them but keep the boundaries
                            valide[j - 1] = 0
                            valide[j] = 0
                    else:  # slopes are not similar in absolute values : we keep the one with the higher slope
                        if abs(slope1) > abs(slope2):
                            start_times = np.delete(start_times, j)
                            end_times = np.delete(end_times, j + 1)
                            valide = np.delete(valide, j)
                            break
                        else:
                            start_times = np.delete(start_times, j - 1)
                            end_times = np.delete(end_times, j)
                            valide = np.delete(valide, j - 1)
                            break

    # case of the last detection too close to the end
    k = 1
    while (start_times[len(start_times) - k] - (end_times[len(start_times) - k] + 2 * w)) < 0:
        if np.sum(valide) < 2:  # only one uturn
            break
        valide[len(start_times) - k - 1] = 0
        k = k + 1

    print("new times", start_times, end_times, valide)

    return start_times, end_times, valide


def time_intervals_selection(points, freq, ax):
    """Select time intervals potentially containing uturns.

    Arguments:
        points {array} -- numpy array with remaining points of the time series
        freq {int} -- acquisition frequency
        ax {matplotlib ax} -- ax from the construction plot

    Returns
    -------
    start_times, end_times, uturn_val, ax
        start_times {list} -- modified list containing in this order all the validated start time samples for uturn
            intervals and the length of the smooth angle time series
        end_times {list} -- modified list containing in this order the zero time sample and all the validated end time samples
            for uturn intervals
        uturn_val {array} -- numpy array with uturn amplitude values
        ax {matplotlib ax} -- modified ax from the construction plot
    """

    diff = abs(np.diff(points[:, 1]))
    thres = 0.4 * max(diff)
    diff_start = np.diff(points[:, 1], append=points[-1, 1])
    diff_end = np.diff(points[:, 1], prepend=points[0, 1])
    start_points = points[abs(diff_start) > thres]
    end_points = points[abs(diff_end) > thres]
    uturn_val = diff_end[abs(diff_end) > thres] / freq

    # progressive figure construction
    ax.scatter(start_points[:, 0], start_points[:, 1], c="green", label="start points estimation")
    ax.scatter(end_points[:, 0], end_points[:, 1], c="red", label="end points estimation")

    # time sample conservation
    end_times = np.insert(end_points[:, 0], 0, 0, axis=0)
    end_times = end_times * freq
    end_times = end_times.astype(int)
    start_times = start_points[:, 0]
    start_times = start_times * freq
    start_times = start_times.astype(int)

    return start_times, end_times, uturn_val, ax


def points_selection(t, smooth_angle, n):
    """Reduce the number of points of the time series with the Ramer-Douglas-Peucker Algorithm.

    Arguments:
        t {time series} -- time series corresponding to data_lb["PacketCounter"]
        smooth angle {time series} -- time series corresponding to estimated angular position of the sensor in the
            axial plane smoothed with a windowing process of size w
        n {int} -- number of attempted uturns. 0 if no idea.
    Returns
    -------
    rdp_array
        Array with resulting points
    """

    # looking for a number of events corresponding to: start + end + 2*number of U-turns
    if n == 0:
        goal = 100
    else:
        goal = 3 + 2 * (n + 1)

        # construct an array for the RDP algorithm package
    array = np.zeros((len(smooth_angle), 2))
    array[:, 0] = t
    array[:, 1] = smooth_angle

    # RDP algorithm with an even lower epsilon value until the number of points is superior to the attempted number
    epsilon = 10
    rdp_array = rdp.rdp(array, epsilon=epsilon, algo="rec")
    found = len(rdp_array)
    while found < goal:
        # print("no", found, goal, epsilon, rdp_array)
        epsilon = epsilon / 2
        rdp_array = rdp.rdp(array, epsilon=epsilon, algo="rec")
        found = len(rdp_array)
    # print("no", found, goal, epsilon, rdp_array)

    return rdp_array


def signals_for_seg(data_lb, w):
    """Estimate mean stride duration.

    Arguments:
        data_lb {dataframe} -- pandas dataframe with pre-processed lower back sensor time series
        w {int} -- window size for rectangular windowing

    Returns
    -------
    t, angle, smooth_angle, fig, ax
        t {time series} -- time series corresponding to data_lb["PacketCounter"]
        angle {time series} -- time series corresponding to estimated angular position of the sensor in the axial plane
        smooth angle {time series} -- time series corresponding to estimated angular position of the sensor in the
            axial plane smoothed with a windowing process of size w
        fig, ax {matplotlib figure} -- initialized construction figure
    """

    t = data_lb["PacketCounter"]
    angle = np.cumsum(data_lb["Gyr_X"]) - np.mean(np.cumsum(data_lb["Gyr_X"]))
    smooth_angle = angle.rolling(w, center=True).mean()
    smooth_angle.fillna(method="pad", inplace=True)
    smooth_angle.fillna(method="bfill", inplace=True)

    # progressive figure construction
    plt.close("all")
    fig, ax = plt.subplots(2, figsize=(20, 10), sharex=True)
    ax[0].plot(t, angle)
    ax[0].set_ylabel('Angular position (x100 rad)', fontsize=15)
    ax[0].set_title("U-Turn detection", fontsize=15, weight='bold')
    ax[1].plot(t, angle, label="raw signal")
    ax[1].plot(t, smooth_angle, label="merged signal")
    ax[1].set_ylabel('Angular position (x100 rad)', fontsize=15)
    ax[1].set_title("Linear interpolation", fontsize=15, weight='bold')
    ax[1].set_xlabel('Time (s)', fontsize=15)

    return t, angle, smooth_angle, fig, ax


def window_estimation(data_lb, freq):
    """Estimate mean stride duration.

    Arguments:
        data_lb {dataframe} -- pandas dataframe with pre-processed lower back sensor time series
        freq {int} -- acquisition frequency

    Returns
    -------
    int
        w {int} -- window size for rectangular windowing
    """

    # start : 0.75s ; end 4s ; roll : 0.05s
    start = int(round(0.75 * freq))
    end = int(round(4 * freq))
    roll = int(round(0.5 * freq))

    # filtered autocorrelation
    acf_x = filt_autocorr(data_lb["FreeAcc_X"], freq, roll=roll)
    acf_y = filt_autocorr(data_lb["FreeAcc_Y"], freq, roll=roll)

    # indicator with a mixed autocorrelation
    acf_tot = [0 for i in range(start)]
    for i in range(start, len(acf_x)):
        if acf_y[i] < 0:
            acf_tot.append(0)
        else:
            acf_tot.append(acf_x[i])

    # first peak-detection
    i = indexes(acf_tot[start:end], thres=0.99, thres_abs=False)

    # if not detected
    if len(i) == 0:
        # extend the search in amplitude
        i = indexes(acf_tot[start:end], thres=0.75, thres_abs=False)
        if len(i) == 0:
            # extend the search in time
            i = indexes(acf_tot[start:int(2 * end)], thres=0.75, thres_abs=False)

    return start + i[0]


def filt_autocorr(x, freq, roll=49):
    """Autocorrelation non-biased estimator.

    Arguments
    ----------
    x : ndarray
        1D amplitude data to compute autocorrelation.
    freq : int
        Acquisition frequency
    roll : int
        Rolling value for cosine windowing

    Returns
    -------
    acf_ : ndarray
        Filtered array containing non-biased estimator for autocorrelation
    """

    # start : 0.75s ; end 4s
    start = int(round(0.75 * freq))
    end = int(round(4 * freq))

    acf_x = autocorr(x)[0:1000]
    acf_x = pd.DataFrame(acf_x)
    acf_x_mean = acf_x.rolling(roll, center=True, win_type='cosine').mean()
    acf_x_mean = acf_x_mean.fillna(0)
    acf_x = acf_x_mean.to_numpy().transpose()[0]
    acf_x = acf_x - max(0, np.mean(acf_x[start:end]))

    return acf_x


def autocorr(x):
    """Autocorrelation non-biased estimator.

    Arguments
    ----------
    x : ndarray
        1D amplitude data to compute autocorrelation.

    Returns
    -------
    acf : ndarray
        Array containing non-biased estimator for autocorrelation
    """

    N = len(x)
    fvi = np.fft.fft(x, n=2 * N)
    acf = np.real(np.fft.ifft(fvi * np.conjugate(fvi))[:N])
    d = N - np.arange(N)
    acf = acf / d  # non biased indicator
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
            # set rightmost and middle values to rightmost non-zero values
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
