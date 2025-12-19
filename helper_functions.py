import numpy as np
import xarray as xr
import pandas as pd
import pywt
import xrscipy.signal.extra as dsp_extra
from typing import List, Dict
from collections import defaultdict
import os
import math

import datetime
import calendar

from scipy.signal import find_peaks
from scipy.signal import fftconvolve
from scipy.ndimage import convolve1d

from eofs.xarray import Eof
from scipy.signal import butter, filtfilt
from scipy.signal import firwin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
#######################################################################################################

def compute_6h_climatology_hourofyear(da, window_days=90):
    """
    Compute a smoothed 6-hourly climatology using a rolling window in 'hourofyear'.
    Each year has 4 six-hourly slots per day, leading to a range from 0 to ~1460.

    Parameters
    ----------
    da : xarray.DataArray
        6-hourly data with a time dimension.
    window_days : int, optional
        Number of days for the rolling-window smoothing. Default is 90.
        The actual rolling window in 'hourofyear' units will be 4 * window_days,
        because there are 4 six-hourly slots per day.

    Returns
    -------
    da_clim_smooth : xarray.DataArray
        6-hourly climatology grouped by 'hourofyear', smoothed with a rolling window.
    """
    if 'time' not in da.dims:
        raise ValueError("Input DataArray must have a 'time' dimension.")

    # Compute hourofyear in six-hourly units (ranges from 0 to ~1460 for non-leap years)
    hourofyear_vals = ((da.time.dt.dayofyear - 1) * 4 + (da.time.dt.hour // 6)).values

    da_6h = da.assign_coords(hourofyear=("time", hourofyear_vals))

    # Compute raw 6-hourly climatology
    clim_raw = da_6h.groupby("hourofyear").mean(dim="time")

    # Define rolling window size (~90 days in six-hourly slots)
    window_6h = window_days * 4  # 90 * 4 = 360 six-hourly slots

    # Apply rolling mean with cyclic extension
    clim_smooth = (
        xr.concat([clim_raw, clim_raw, clim_raw], dim="hourofyear")
            .rolling(hourofyear=window_6h, center=True, min_periods=1)
            .mean()
            .isel(hourofyear=slice(len(clim_raw), 2 * len(clim_raw)))  # Keep only the original range
    )

    return clim_smooth
#######################################################################################################
def compute_normalization(anomalies, window_days=15):
    """
    Compute the running standard deviation for anomalies using a rolling window in 'hourofyear'.
    This standard deviation can be used for normalization purposes.

    Parameters:
        anomalies (xr.DataArray): Anomalies data with dimensions (time, lat, lon).
        window_days (int): Half-window size in days (default is 15 days for a ±15-day window).

    Returns:
        std_running (xr.DataArray): Running standard deviation computed over a 30-day window.
    """
    if 'time' not in anomalies.dims:
        raise ValueError("Input DataArray must have a 'time' dimension.")

    # Define 'hourofyear' in six-hourly units (0 to ~1460 for non-leap years)
    hourofyear_vals = ((anomalies.time.dt.dayofyear - 1) * 4 + (anomalies.time.dt.hour // 6)).values
    anomalies = anomalies.assign_coords(
        hourofyear=("time", hourofyear_vals)
    )

    # Define rolling window size in six-hourly slots
    rolling_window = (2 * window_days) * 4 + 1  # For window_days=15, rolling_window=121

    # Compute raw standard deviation grouped by 'hourofyear'
    # This calculates the standard deviation for each 'hourofyear' slot across all years
    clim_std_raw = anomalies.groupby("hourofyear").std(dim="time")

    # Apply rolling window with cyclic extension to smooth the standard deviation
    # Concatenate the climatological std three times to handle the cyclic nature
    clim_std_extended = xr.concat([clim_std_raw, clim_std_raw, clim_std_raw], dim="hourofyear")

    # Apply rolling standard deviation over the extended 'hourofyear' dimension
    clim_std_smooth_extended = clim_std_extended.rolling(
        hourofyear=rolling_window, center=True, min_periods=1
    ).mean()

    # Slice to retain only the original range of 'hourofyear' slots
    clim_std_smooth = clim_std_smooth_extended.isel(
        hourofyear=slice(len(clim_std_raw), 2 * len(clim_std_raw))
    )

    return clim_std_smooth
#######################################################################################################
def lanczos_filter_10_day_lowpass(
    dataarray: xr.DataArray,
    timesteps_per_day: int,
    num_passes: int = 1,
    filter_width_days: int = 10,
    filter_cutoff_days: float = 10,
    sigma: float = 1.0
) -> xr.DataArray:
    """
    Apply a 10-day low-pass Lanczos filter with a filter width of 10 days
    and sigma=1.0 to an xarray DataArray along the 'time' dimension.

    Parameters
    ----------
    dataarray : xr.DataArray
        The input data, with a 'time' dimension.
    timesteps_per_day : int
        Number of timesteps per day in the data (e.g., 4 for 6-hourly data).
    num_passes : int, optional
        Number of times the filter is applied sequentially. Defaults to 1.

    Returns
    -------
    xr.DataArray
        The filtered data, same shape as input.
    """
    
    # ---------------------------#
    # 1) Validate inputs
    # ---------------------------#
    if num_passes < 1:
        raise ValueError("num_passes must be at least 1")
    if 'time' not in dataarray.dims:
        raise ValueError("Input DataArray must have a 'time' dimension")
    
    # ---------------------------#
    # 2) Set filter parameters
    # ---------------------------#
    filter_number = 0  # Low-pass filter as per Table 1
    filter_width_days = filter_width_days   
    filter_half_order = (filter_width_days/2) * timesteps_per_day  # Half-width in timesteps
    # filter_half_order = (filter_width_days) * timesteps_per_day + 1 # Half-width in timesteps
    sigma = sigma  # Smoothing factor
    
    # ---------------------------#
    # 3) Determine filter type and cutoff frequencies
    # ---------------------------#
    # For a low-pass filter, we retain timescales t > 10 days
    ftype = 'lowpass'
    day_min = None
    day_max = filter_cutoff_days  # Retain periods longer than 10 days
    
    # ---------------------------#
    # 4) Convert days to "timesteps" → freq = 1/(N timesteps)
    # ---------------------------#
    def day2freq(days):
        # 1 day → timesteps_per_day samples, so N timesteps = days * timesteps_per_day
        # freq in cycles per sample is 1 / (N timesteps).
        return 1.0 / (days * timesteps_per_day)
    
    freq_min = day2freq(day_min) if day_min is not None else None
    freq_max = day2freq(day_max) if day_max is not None else None
    
    # ---------------------------#
    # 5) Build the Lanczos kernel
    # ---------------------------#
    # Define the range of n for the kernel
    N = filter_half_order 
    n = np.arange(-N, N + 1)
    
    def lowpass_kernel(freq_cut):
        # "Ideal" lowpass part
        h_i = 2.0 * freq_cut * np.sinc(2.0 * freq_cut * n)
        # Lanczos window
        w_i = np.sinc((sigma * n) / N)
        h = h_i * w_i
        return h
    
    if ftype == 'lowpass':
        # Single lowpass
        kernel = lowpass_kernel(freq_max)  # day_max → freq_max → periods > day_max
        kernel /= np.sum(kernel)  # Normalize to ensure unity gain
    elif ftype == 'highpass':
        # Highpass = delta(n) - lowpass
        lp = lowpass_kernel(freq_max)
        lp /= np.sum(lp)
        delta = np.zeros_like(lp)
        delta[N] = 1.0  # Dirac delta in the center
        kernel = delta - lp
    else:
        raise ValueError("Unsupported filter type.")
    
    # ---------------------------#
    # 6) Convolve in time
    # ---------------------------#
    filtered = dataarray.copy()
    kernel_da = xr.DataArray(kernel, dims=["kernel"])
    
    for _ in range(num_passes):
        filtered = xr.apply_ufunc(
            convolve1d,
            filtered,
            kernel_da,
            input_core_dims=[['time'], ['kernel']],
            output_core_dims=[['time']],
            kwargs={'mode': 'reflect'},
            vectorize=True,
            dask='parallelized',
            output_dtypes=[filtered.dtype]
        )
    
    # ---------------------------#
    # 7) Return
    # ---------------------------#
    return filtered.transpose('time', 'lat','lon')
#######################################################################################################
def compute_iwr(z500_anomaly, cluster_means):
    """
    Compute the Michel & Rivière (2011)-style Iwr for all clusters and time steps.
    Iwr(t, cluster) = [ dot( z'(t)*cosφ, cluster_mean ) ] / 
                      sqrt[ dot( cluster_mean*cosφ, cluster_mean ) ],

    where dot(...) sums over (lat,lon).  This ensures that Iwr=1 means 
    "the anomaly field matches the mean cluster amplitude" in that cluster.

    Parameters
    ----------
    z500_anomaly : xarray.DataArray
        Instantaneous Z500 anomaly of shape (time, lat, lon).
    cluster_means : xarray.DataArray
        Cluster-mean anomaly patterns of shape (cluster, lat, lon).

    Returns
    -------
    iwr : xarray.DataArray
        Iwr values of shape (time, cluster). 
    """

    # 1) Set up cos(lat) weighting
    coslat = np.cos(np.deg2rad(z500_anomaly.lat))
    coslat_2d = coslat.broadcast_like(z500_anomaly.isel(time=0))  # shape (lat, lon)

    raw_proj = xr.dot(
        z500_anomaly, #* coslat_2d,
        cluster_means,
        dims=["lat", "lon"]
    )  # shape: (time, cluster)

    iwr = raw_proj 

    # Return with cluster dimension labeled
    return iwr.assign_coords(cluster=cluster_means.cluster)
#######################################################################################################  
def identify_weather_regimes_grams(
    IWR,
    threshold=1.0,
    min_days=5.0,
    max_time_diff_days=100.0
):

    regime_labels = ["GL", "ScTr", "EuBL", "AR", "ScBL", "ZO", "AT"]

    # 1) Basic Setup & Checks
    IWR = IWR.sort_values('time').reset_index(drop=True)
    times = IWR['time'].values
    n_steps = len(IWR)

    if not {'time', *regime_labels}.issubset(IWR.columns):
        raise ValueError(f"IWR must have columns 'time' plus all regime labels: {regime_labels}")

    steps_min_lifecycle = int(math.ceil(min_days * 4))
    steps_max_diff = int(math.ceil(max_time_diff_days * 4))

    all_life_cycles = []

    # 2) Find Local Maxima & Preliminary Life Cycles
    def find_prelim_lifecycles(label, iwr_arr):
        prelim = []
        for i in range(1, n_steps - 1):
            if iwr_arr[i] >= threshold:
                if iwr_arr[i] >= iwr_arr[i-1] and iwr_arr[i] >= iwr_arr[i+1]:
                    m = i
                    o = m
                    while o > 0 and iwr_arr[o-1] >= threshold:
                        o -= 1
                    d = m
                    while d < (n_steps - 1) and iwr_arr[d+1] >= threshold:
                        d += 1
                    prelim.append({
                        'regime': label,
                        'onset_idx': o,
                        'max_idx': m,
                        'decay_idx': d
                    })
        return prelim

    for label in regime_labels:
        arr = IWR[label].values
        pcycles = find_prelim_lifecycles(label, arr)

        valid = []
        for c in pcycles:
            dur = c['decay_idx'] - c['onset_idx'] + 1
            if dur >= steps_min_lifecycle:
                valid.append(c)

        valid.sort(key=lambda x: (x['onset_idx'], x['max_idx']))
        merged = []
        i = 0
        while i < len(valid):
            current = valid[i]
            i += 1
            if i < len(valid):
                nxt = valid[i]
                same_onset = (current['onset_idx'] == nxt['onset_idx'])
                same_decay = (current['decay_idx'] == nxt['decay_idx'])
                if same_onset or same_decay:
                    arr_iwr = arr
                    m1, m2 = current['max_idx'], nxt['max_idx']
                    lo, hi = min(m1, m2), max(m1, m2)
                    mean_val = arr_iwr[lo:hi+1].mean()
                    diff_in_steps = abs(m2 - m1)
                    if mean_val >= threshold and diff_in_steps <= steps_max_diff:
                        new_onset = min(current['onset_idx'], nxt['onset_idx'])
                        new_decay = max(current['decay_idx'], nxt['decay_idx'])
                        seg = arr_iwr[new_onset:new_decay+1]
                        rel_idx = np.argmax(seg)
                        new_max = new_onset + rel_idx
                        merged.append({
                            'regime': label,
                            'onset_idx': new_onset,
                            'max_idx': new_max,
                            'decay_idx': new_decay
                        })
                        i += 1
                        continue
                    else:
                        merged.append(current)
                else:
                    merged.append(current)
            else:
                merged.append(current)

        all_life_cycles.extend(merged)

    # 3) Strong & Meaningful Filter
    final_life_cycles = []
    for c in all_life_cycles:
        label = c['regime']
        o = c['onset_idx']
        d = c['decay_idx']
        arr_r = IWR[label].values

        found_winning_step = False
        for t in range(o, d+1):
            val_r = arr_r[t]
            better_than_all = True
            for other_label in regime_labels:
                if other_label == label:
                    continue
                if val_r <= IWR[other_label][t]:
                    better_than_all = False
                    break
            if better_than_all:
                found_winning_step = True
                break

        if found_winning_step:
            final_life_cycles.append(c)

    # 4) Active mask
    active = {label: np.zeros(n_steps, dtype=bool) for label in regime_labels}
    for c in final_life_cycles:
        label = c['regime']
        o = c['onset_idx']
        d = c['decay_idx']
        active[label][o:(d+1)] = True

    # 5) Label selection
    labels_per_timestep = []
    for t in range(n_steps):
        act_labels = [label for label in regime_labels if active[label][t]]
        if not act_labels:
            labels_per_timestep.append(-1)
        elif len(act_labels) == 1:
            labels_per_timestep.append(act_labels[0])
        else:
            values = {label: IWR[label][t] for label in act_labels}
            max_val = max(values.values())
            candidates = [label for label, val in values.items() if val == max_val]
            chosen = sorted(candidates)[0]
            labels_per_timestep.append(chosen)

    return pd.DataFrame({
        "time": IWR["time"],
        "WR": ["No" if r == -1 else r for r in labels_per_timestep]
    })
#######################################################################################################
def check_temporal_std_exists(file_path):
    """Check if the temporal standard deviation NetCDF file exists."""
    return os.path.exists(file_path)
#######################################################################################################
def compute_cluster_means(z500_anomaly, cluster_labels, num_clusters):
    """
    Compute cluster mean spatial patterns for each weather regime,
    averaging over time without latitudinal weighting.
    
    Parameters
    ----------
    z500_anomaly   : xarray.DataArray 
                     Dimensions typically [time, lat, lon].
    cluster_labels : np.ndarray 
                     1D array (same length as 'time') with cluster IDs.
    num_clusters   : int 
                     Number of clusters (weather regimes).
                     
    Returns
    -------
    cluster_means : xarray.DataArray
                    Dimensions [cluster, lat, lon].
    """
    cluster_means = []

    for cluster in range(num_clusters):
        # Get indices for this cluster
        cluster_indices = np.where(cluster_labels == cluster)[0]
        
        # Subset anomalies for the cluster
        cluster_data = z500_anomaly.isel(time=cluster_indices)
        
        # Compute unweighted mean over time
        cluster_mean = cluster_data.mean(dim="time")  
        
        cluster_means.append(cluster_mean)

    return xr.concat(cluster_means, dim="cluster").assign_coords(cluster=np.arange(num_clusters))