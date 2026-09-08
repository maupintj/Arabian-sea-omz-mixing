import os, gzip
from scipy.interpolate import interp1d, interp2d
from scipy.optimize import fsolve, fmin
from scipy import signal
import pandas as pd
import numpy as np
import gsw
from datetime import datetime as dt
import os, gc, sys, warnings
from glob import glob
import importlib
from tqdm import tqdm
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import interp1d
from scipy.optimize import fmin
import gsw
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.pyplot as plt
import cmocean.cm as cmo


### GENERAL SCRIPTS
def RunningMedian(x,N):
    grid = np.ones((len(x)+2*N, 1 + 2*N ))*np.nan
    for istep in range(np.shape(grid)[1]):
        grid[istep:len(x)+istep, istep] = x
    return np.nanmedian(grid,axis=1)[N:-N]

def RunningMax(x,N):
    grid = np.ones((len(x)+2*N, 1 + 2*N ))*np.nan
    for istep in range(np.shape(grid)[1]):
        grid[istep:len(x)+istep, istep] = x
    return np.nanmax(grid,axis=1)[N:-N]

def RunningMin(x,N):
    grid = np.ones((len(x)+2*N, 1 + 2*N ))*np.nan
    for istep in range(np.shape(grid)[1]):
        grid[istep:len(x)+istep, istep] = x
    return np.nanmin(grid,axis=1)[N:-N]

def RunningMean(x,N):
    grid = np.ones((len(x)+2*N, 1 + 2*N ))*np.nan
    for istep in range(np.shape(grid)[1]):
        grid[istep:len(x)+istep, istep] = x
    return np.nanmean(grid,axis=1)[N:-N]

def interp(x,y,xi):
    _gg = np.isfinite(x+y)
    return interp1d(x[_gg], y[_gg], bounds_error=False, fill_value=np.nan)(xi)

def rmsd(x):
    return np.sqrt(np.nanmean(x**2))

def plog(msg):
    print(str(dt.now().replace(microsecond=0))+' : '+msg)
    return None
def assign(ds, name, values, append_comment = None, attrs = None):
    if name in ds:
        Attrs = ds[name].attrs
    else:
        Attrs = {}
    ds[name] = ('time', values)
    ds[name].attrs = Attrs
    if attrs is not None:
        for key, val in attrs.items():
            ds[name].attrs[key] = val
    if append_comment is not None:
        if 'comment' not in Attrs.keys():
            ds[name].attrs['comment'] = []
        ds[name].attrs['comment'] = ds[name].attrs['comment'] + ' ' + append_comment
    return ds
def make_nan_gap_like(ds, ds2, last_dive_num, last_profile_index, gap_len=4):
    """
    Create a gap dataset with NaNs for visual separation between dive segments.
    
    Parameters:
        ds (xarray.Dataset): Template dataset (last in previous segment).
        ds2 (xarray.Dataset): Next dataset (used to define gap timing).
        last_dive_num (int or float): Last used dive number.
        last_profile_index (int): Last used profile index.
        gap_len (int): Number of time steps in the gap (default: 4).
        
    Returns:
        xarray.Dataset: Dataset with gap_len time steps of NaN-filled data, with dive/profile indexes and matching attributes.
    """
    import numpy as np
    import xarray as xr

    if gap_len != 4:
        raise ValueError("This version requires gap_len = 4 for expected time logic.")

    nan_data = {}

    # Compute dive_num values (e.g., [251, 251, 252, 252])
    num_dive_increments = int(np.ceil(gap_len / 2))
    dive_nums = np.repeat(np.arange(last_dive_num + 1,
                                     last_dive_num + 1 + num_dive_increments),
                          2)[:gap_len]

    for var in ds.data_vars:
        shape = (gap_len,)
        nan_vals = np.full(shape, np.nan, dtype=ds[var].dtype)

        if var == 'dive_num':
            nan_vals = dive_nums.astype(ds[var].dtype)

        elif var == 'profile_index':
            nan_vals = np.arange(last_profile_index + 1,
                                 last_profile_index + 1 + gap_len,
                                 dtype=ds[var].dtype)

        da = xr.DataArray(nan_vals, dims=('time',))
        da.attrs = ds[var].attrs.copy()
        nan_data[var] = da

    # Time based on surrounding datasets
    last_time = ds.time.values[-1]
    first_time_next = ds2.time.values[0]

    if np.isnat(last_time) or np.isnat(first_time_next):
        raise ValueError("NaT found in input times.")

    gap_times = np.array([
        last_time + np.timedelta64(5, 'm'),
        last_time + np.timedelta64(10, 'm'),
        first_time_next - np.timedelta64(10, 'm'),
        first_time_next - np.timedelta64(5, 'm'),
    ], dtype='datetime64[ns]')

    coords = {
        'time': (('time',), gap_times)
    }

    gap_ds = xr.Dataset(nan_data, coords=coords)
    gap_ds.attrs = ds.attrs.copy()

    return gap_ds
def extract_time(d57_gridded):
    """
    Interpolate missing datetime64[s] (NaT) values in a 1D array.
    Special handling for exactly two consecutive NaTs:
        - first is 5 min after previous valid time
        - second is 5 min before next valid time
    Parameters:
        t_idx (np.ndarray): 1D datetime64[s] array with possible NaT values.
    Returns:
        np.ndarray: datetime64[s] array with interpolated or filled values.
    """
    import numpy as np
    import pandas as pd
    t_idx=d57_gridded.time.values.astype('float64')
    t_idx[t_idx < 0] = np.nan  # Handle negative times
    t_idx=np.nanmean(t_idx, axis=0)  # Take the mean across the depth dimension
    t_idx=t_idx.astype('datetime64[s]')
    t_idx = np.asarray(t_idx)
    ts = pd.Series(t_idx)

    is_nat = ts.isna()
    result = ts.copy()

    # Group consecutive NaNs
    group_ids = (is_nat != is_nat.shift()).cumsum()
    groups = is_nat[is_nat].groupby(group_ids)

    for _, grp in groups:
        idx = grp.index
        first, last = idx[0], idx[-1]

        prev_idx = first - 1
        next_idx = last + 1

        # Assign first NaT
        if prev_idx >= 0 and not pd.isna(ts[prev_idx]):
            result.iloc[first] = ts[prev_idx] + pd.Timedelta(minutes=5)

        # Assign last NaT
        if next_idx < len(ts) and not pd.isna(ts[next_idx]):
            result.iloc[last] = ts[next_idx] - pd.Timedelta(minutes=5)

    # Linear interpolate remaining NaTs (middle ones)
    time_float = result.apply(lambda x: x.timestamp() if pd.notnull(x) else np.nan)
    time_interp = time_float.interpolate(method='linear', limit_direction='both')

    return pd.to_datetime(time_interp, unit='s').values.astype('datetime64[s]')

#########################################################################################################
################################ GENERAL GLIDER TOOLS ###################################################
#########################################################################################################
def make_hotel(ds, ADCP, output_name='Hotel.mat'):
    """
    Interpolates ADCP and CTD data to 1-second resolution and saves to a hotel.mat file.
    Parameters:
        ds (xarray.Dataset): CTD dataset with pressure and other glider data.
        ADCP (xarray.Dataset): Velocity dataset with X, Y, Z and time.
        output_name (str): Name of the output .mat file.
    Returns:
        bool: True if successful.
    """
    from scipy.io import savemat
    def get_unixtime(dt64):
        return dt64.astype('datetime64[s]').astype('float')
    def safe_mean(var):
        return var.mean(dim='gridded_bin') if 'gridded_bin' in var.dims else var
    # 1. Define 1-second resolution time vector from ADCP.time
    tmin = ds.time.values[0]
    tmax = ds.time.values[-1]
    time_1s = np.arange(tmin, tmax, np.timedelta64(1, 's')).astype('datetime64[s]')
    time_1s_float = get_unixtime(time_1s)
    # 2. Preprocess variables (interpolated to 1s)
    interp_ds = xr.Dataset(coords={'time': time_1s})
    # Interpolate ADCP components (mean over bins if needed)
    interp_ds['X'] = safe_mean(ADCP['X']).interp(time=time_1s)
    interp_ds['Y'] = safe_mean(ADCP['Y']).interp(time=time_1s)
    interp_ds['Z'] = safe_mean(ADCP['Z']).interp(time=time_1s)
    # Interpolate CTD variables
    for varname in ['Pressure', 'Pitch', 'Roll', 'glider_salinity', 'glider_temperature', 'Heading']:
        if varname in ADCP:
            interp_ds[varname] = ADCP[varname].interp(time=time_1s)
        else:
            print(f"Warning: '{varname}' not found in ADCP. Skipping.")
    if 'pressure' in ds:
        interp_ds['P_CTD'] = ds['pressure'].interp(time=time_1s)
    else:
        print("Warning: 'pressure' not found in CTD dataset.")
    # 3. Derived variables
    xyz_spd = np.sqrt(interp_ds['X']**2 + interp_ds['Y']**2 + interp_ds['Z']**2)
    aoa = 180 - np.rad2deg(np.arccos(np.clip(interp_ds['X'] / xyz_spd, -1.0, 1.0)))
    # 4. Build output dictionary
    output = {
        'speed': {'time': time_1s_float, 'data': xyz_spd.values.astype('float')},
        'aoa': {'time': time_1s_float, 'data': aoa.values.astype('float')}
    }
    # Standard variables
    var_map = {
        'Pres_ADCP': 'Pressure',
        'Pitch_ADCP': 'Pitch',
        'Roll_ADCP': 'Roll',
        'S_CTD': 'glider_salinity',
        'T_CTD': 'glider_temperature',
        'Heading_ADCP': 'Heading',
        'P_CTD': 'P_CTD'
    }
    for mat_name, ds_var in var_map.items():
        if ds_var in interp_ds:
            output[mat_name] = {
                'time': time_1s_float,
                'data': interp_ds[ds_var].values.astype('float')
            }
    # ADCP components
    for comp in ['X', 'Y', 'Z']:
        output[f'ADCP_{comp}'] = {
            'time': time_1s_float,
            'data': interp_ds[comp].values.astype('float')
        }
    # 5. Save to .mat
    savemat(output_name, output)
    return True

def grid2d(x, y, v, xi=1, yi=1, fn='median'):
    """
    Quick data binning function relying on pandas.
    x,y,v are flat np.arrays of x, y coordinates and values at these points.
    xi and yi are either np.arrays of bins to be binned into, or the spacing used between min and max of x or y respectively.
    fn defines the function applied to all points from v that fall into the same bin.
    """
    if np.size(xi) == 1:
        dx = xi
        x_min = np.nanmin(x)
        x_max = np.nanmax(x)
        # Extend upper bound slightly to include max(x)
        xi = np.arange(x_min, x_max + dx * 1.01, dx)
    else:
        # Extend xi slightly to ensure last bin includes max value
        xi = np.append(xi, xi[-1] + (xi[-1] - xi[-2]))

    if np.size(yi) == 1:
        dy = yi
        y_min = np.nanmin(y)
        y_max = np.nanmax(y)
        yi = np.arange(y_min, y_max + dy * 1.01, dy)
    else:
        # Extend yi similarly
        yi = np.append(yi, yi[-1] + (yi[-1] - yi[-2]))

    raw = pd.DataFrame({'x': x, 'y': y, 'v': v}).dropna()

    grid = np.full([np.size(yi) - 1, np.size(xi) - 1], np.nan)

    raw['xbins'], _ = pd.cut(raw.x, xi, retbins=True, labels=False, right=False)
    raw['ybins'], _ = pd.cut(raw.y, yi, retbins=True, labels=False, right=False)

    _tmp = raw.groupby(['xbins', 'ybins'])['v'].agg(fn)
    grid[
        _tmp.index.get_level_values(1).astype(int),
        _tmp.index.get_level_values(0).astype(int),
    ] = _tmp.values

    XI, YI = np.meshgrid(xi[:-1], yi[:-1], indexing='ij')
    return grid, XI.T, YI.T

def grid_dataset(ds, depth_step=2.0, profile_step=1.0, default_fn='median'):
    """
    Grids an xarray.Dataset onto a regular (depth, profile_index) grid and adds 2D time coordinate.

    Parameters:
        ds (xarray.Dataset): Input dataset with 'profile_index', 'depth', and 'time'.
        depth_step (float): Vertical resolution of the output grid.
        profile_step (float): Horizontal resolution of the output grid (along profile index).
        default_fn (str): Aggregation function for non-nearest variables, e.g., 'median', 'mean'.

    Returns:
        xarray.Dataset: Gridded dataset with variables on (depth, profile_index).
    """
    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    # Variables to grid using nearest (first observed value)
    nearest_vars = {
        'FM1', 'FM2',
        'segment_fast1', 'segment_fast2',
        'segment_slow1', 'segment_slow2',
        'flag1', 'flag2',
        'K_max1', 'K_max2','uid_code'
    }

    # Clean: keep only integer profile_index
    #ds_clean = ds.where(ds.profile_index == np.floor(ds.profile_index), drop=True) SOME TIMES THIS TAKES FOREVER
    #
    # Make a clean boolean mask: "profile_index is (almost) an integer"
    pi = ds['profile_index']  # 1-D over time
    valid = xr.apply_ufunc(
        lambda a: np.isfinite(a) & np.isclose(a, np.rint(a), atol=1e-6),
        pi,
        dask='parallelized',
        vectorize=True,
    )
    # Select along the only dim ('time')
    ds_clean = ds.isel(time=valid)
    #
    #
    # Define bin edges for profile and depth
    prof_min = np.nanmin(ds_clean.profile_index.values)
    prof_max = np.nanmax(ds_clean.profile_index.values)
    profile_bins = np.arange(prof_min, prof_max + profile_step, profile_step)

    depth_max = np.nanmax(ds_clean.depth.values)
    depth_bins = np.arange(0, depth_max + depth_step, depth_step)

    # Extract core coordinates
    pidx = ds_clean.profile_index.values
    d = ds_clean.depth.values
    t = ds_clean.time.values
    t_float = t.astype('datetime64[s]').astype(float)

    # Grid time (POSIX float) and convert to datetime64
    time_grid, _, _ = grid2d(pidx, d, t_float, xi=profile_bins, yi=depth_bins, fn='mean')
    time_matrix = time_grid.astype('datetime64[s]')

    # Set coordinates
    coords = {
        'depth': depth_bins,
        'profile_index': profile_bins,
        'time': (('depth', 'profile_index'), time_matrix)
    }

    # Grid all variables
    data_vars = {}
    skip_vars = {'profile_index', 'depth', 'time'}

    for varname in tqdm(ds_clean.data_vars, desc="Gridding variables"):
        if varname in skip_vars:
            continue

        da = ds_clean[varname]
        x = ds_clean.profile_index.values
        y = ds_clean.depth.values
        v = da.values

        fn = 'nearest' if varname in nearest_vars else default_fn

        try:
            if fn == 'nearest':
                # Nearest: bin and take first value in each bin
                raw_df = pd.DataFrame({'x': x, 'y': y, 'v': v}).dropna()
                raw_df['xbins'] = pd.cut(raw_df['x'], profile_bins, labels=False, right=False)
                raw_df['ybins'] = pd.cut(raw_df['y'], depth_bins, labels=False, right=False)

                # First observation per bin
                grouped = raw_df.groupby(['ybins', 'xbins'], sort=False).first().reset_index()

                # Create grid and fill
                grid = np.full((len(depth_bins), len(profile_bins)), np.nan)
                yb = grouped['ybins'].astype(int)
                xb = grouped['xbins'].astype(int)
                grid[yb, xb] = grouped['v'].values
            else:
                # Default aggregation (e.g., median)
                grid, _, _ = grid2d(x, y, v, xi=profile_bins, yi=depth_bins, fn=fn)

            data_vars[varname] = (('depth', 'profile_index'), grid, da.attrs)

        except Exception as e:
            print(f"Warning: failed to grid {varname} — {e}")
            continue

    return xr.Dataset(data_vars=data_vars, coords=coords)

def compare_UD(data,varname, plot=False, pressure_time_offset=0,funct=np.nanmedian):
    #idx = (data.profile_index.values >= 1) & (data.profile_index.values <= 600)
    #data = data.isel(time=idx)
    raw_seconds = data.time.values.astype('float') / 1e9; raw_seconds = raw_seconds - np.nanmin(raw_seconds)
    pressure = interp(raw_seconds, data['pressure'].values, raw_seconds + pressure_time_offset)

    PD, X, Y = grid2d(data.profile_index.values, pressure, data[varname].values, xi=1, yi=1, fn=np.nanmean)
    
    downcasts = np.remainder(X[0,:],2) == 1
    upcasts = np.remainder(X[0,:],2) == 0
    comparable_profiles = np.nanmin([len(downcasts), len(upcasts)])

    if np.remainder(X[0,0],2) == 1: # start with downcasts
        difference = PD[:,downcasts[:comparable_profiles]] - PD[:,upcasts[:comparable_profiles]] 
    else:
        difference = PD[:,downcasts[:comparable_profiles]] - PD[:,upcasts[:comparable_profiles]] 
        
    if plot:
        plt.figure(figsize=(10,4))
        plt.subplot(142)
        plt.plot(funct(difference, axis=1),Y[:,0], '-b')
        plt.axvline(0)
        plt.gca().invert_yaxis()
        plt.xlim(np.array([-1,1])*0.025)
        plt.xlabel('funct difference in ' + varname)
        plt.ylabel('pressure [dbar]')
        
        plt.subplot(122)
        plt.pcolormesh(difference,cmap=plt.get_cmap('cmo.balance',15))
        plt.colorbar()
        plt.gca().invert_yaxis()
        plt.clim(np.array([-1,1])*0.05)
        plt.title('difference in ' + varname)
        plt.xlabel('dive number')

    return np.sqrt(np.nanmean(difference**2))
def RBRlag(data, model='standard', speed_smoothing=5, lagcoef=None):
    ds = data.copy(deep=True)
    # Define the correction coefficients based on the model
    if model == 'standard':
        alpha_TM = [0.05 , -0.83]
        tau_TM =   [334.21 , 0.03]
        alpha_LB = [0.18 , -1.10]
        tau_LB =   [179 , 0]
        alpha_SB = [0.23 , -0.82]
        tau_SB =   [27.15 , -0.58]
        deltat =  0.6 # 0.9
    if model == 'fast':
        alpha_TM = [1 , 0]
        tau_TM =   [0 , 0]
        alpha_LB = [0.40 , -1.25]
        tau_LB =   [200 , 0]
        alpha_SB = [0.06 , 0]
        tau_SB =   [79.31 , -0.87]
        deltat =  0.06
        
    if lagcoef is not None:
        deltat = lagcoef
        
    # Configure smoothing
    def smooth(x, window=speed_smoothing): # Smoothing function (boxcar filter)
        return np.convolve(x, np.ones(speed_smoothing)/speed_smoothing, mode='same')
    
    # Calculate necessary variables
    raw_seconds = ds.time.values.astype('float') / 1e9; raw_seconds = raw_seconds - np.nanmin(raw_seconds)
    
    # Calculate speed variable
    spd = ds['speed'].values
    spd[spd < 0.01] = 0.01
    spd[spd > 1] = 1
    spd[~np.isfinite(spd)] = 0.05
    spd_cm = spd*100  # Mathieu Dever's coefficients determined for speed in cm.s-1
    
    # Begin processing
    #Fs = np.mean(1/np.gradient(raw_seconds)) # Determine sampling frequency
    Fs = np.median(1/np.gradient(raw_seconds)) # Determine sampling frequency
    
    #print('Performing thermal mass correction... Assuming a constant sampling frequency of '+str(Fs)+' Hz.')
    #print('Sampling frequency standard deviation: '+str(np.nanstd(1/np.gradient(raw_seconds))))
    fn = Fs/2 # Nyquist frequency
    
    # Step 1 : produce clean temperature
    corr_temp = ds.temperature.bfill().values # Remember to backfill temperature as the correction is iterative. 
    
    # Lag correct temperature
    corr_temp = interp(raw_seconds, corr_temp, raw_seconds + deltat)
    
    # Correct temperature probe's thermal lag to get real temperature
    alpha = alpha_TM[0] * smooth(spd_cm)**alpha_TM[1]
    tau = tau_TM[0] * smooth(spd_cm)**tau_TM[1]
    bias_temp = np.full_like(corr_temp,0)
    a = 4*fn*alpha * tau / (1+4*fn*tau) # Lueck and Picklo (1990)
    b = 1-2*a/alpha # Lueck and Picklo (1990)
    for sample in np.arange(1,len(bias_temp)):
        bias_temp[sample] = -b[sample] * bias_temp[sample-1] + a[sample] * (corr_temp[sample] - corr_temp[sample-1])
        
    # Produce final temperatire
    corr_temp = corr_temp + bias_temp    
    
    # Step 2 : estimate temperature in the conductivity cell
    # Estimate effective temperature of the conductivity measurement (long thermal lag)
    alpha = alpha_LB[0] * smooth(spd_cm)**alpha_LB[1]
    tau = tau_LB[0] * smooth(spd_cm)**tau_LB[1]
    bias_long = np.full_like(corr_temp,0)
    a = 4*fn*alpha * tau / (1+4*fn*tau) # Lueck and Picklo (1990)
    b = 1-2*a/alpha # Lueck and Picklo (1990)
    for sample in np.arange(1,len(bias_long)):
        bias_long[sample] = -b[sample] * bias_long[sample-1] + a[sample] * (corr_temp[sample] - corr_temp[sample-1])
        
    
    # Estimate effective temperature of the conductivity measurement (short thermal lag)
    alpha = alpha_SB[0] * smooth(spd_cm)**alpha_SB[1]
    tau = tau_SB[0] * smooth(spd_cm)**tau_SB[1]
    bias_short = np.full_like(corr_temp,0)
    a = 4*fn*alpha * tau / (1+4*fn*tau) # Lueck and Picklo (1990)
    b = 1-2*a/alpha # Lueck and Picklo (1990)
    for sample in np.arange(1,len(bias_short)):
        bias_short[sample] = -b[sample] * bias_short[sample-1] + a[sample] * (corr_temp[sample] - corr_temp[sample-1])
    

    ct_temp = corr_temp - bias_long - bias_short
    corr_sal = gsw.SP_from_C(ds.conductivity.values, ct_temp, ds.pressure.values)
    
    # data['rbr_bias_temp'] = bias_temp
    # data['rbr_bias_short'] = bias_short
    # data['rbr_bias_long'] = bias_long
    # data['temperature'] = corr_temp
    # data['salinity'] = corr_sal
        
    return corr_temp, corr_sal


def correct_RBR_lag(data, model='standard', lagcoef=None, viewonly=False,var2run='dive_num'):
    ds = data.copy(deep=True)
    #var2run='profile_index'  
    ds[var2run] = np.round(ds[var2run])
    df = pd.DataFrame()
    df[var2run] = ds[var2run].values
    df['time'] = ds['time'].values
    
    raw_seconds = ds.time.values.astype('float') / 1e9; raw_seconds = raw_seconds - np.nanmin(raw_seconds)
    vert_spd = np.abs(np.gradient( ds['depth'].values, raw_seconds)) # vertical speed, m.s-1
    angle = np.abs(ds.pitch.values) + 5 # assuming 5 degree angle of attack
    spd = vert_spd / np.sin(np.deg2rad(angle)) # speed through water, m.s-1
    
    df['speed'] = spd
    df['pressure'] = ds['pressure'].values
    df['temperature'] = ds['temperature'].values
    df['conductivity'] = ds['conductivity'].values
    dnum = ds[var2run]
    #dives = np.unique(df[var2run].values)
    dives = np.unique(ds[var2run][~np.isnan(ds[var2run])][ds[var2run][~np.isnan(ds[var2run])] % 1 == 0])
    
    for dive in tqdm(dives):
        idx = df[var2run] == dive
        df.loc[idx, 'temperature'],df.loc[idx, 'salinity'] = RBRlag(df[idx], model=model, lagcoef=lagcoef)
        #print(dive)
    
    if viewonly:
        final = ds.copy()
    
    ds['temperature_uncorrected'] = ds['temperature']
    ds['salinity_uncorrected'] = ds['salinity']

    ds['temperature'] = ('time', df['temperature'].values)
    ds['salinity'] = ('time', df['salinity'].values) 
    
    if not viewonly:
        final = ds
    
    _deep = ds['pressure'] > 650
    
    def smooth(x, window=11): # Smoothing function (boxcar filter)
        return np.convolve(x, np.ones(11)/11, mode='same')
    
    plt.figure(figsize=(12,12))
    dives = np.floor(np.linspace(0,len(np.unique(np.floor(ds.dive_num.values))),18))[1:-1]
    for idx,dn in enumerate(dives):
        plt.subplot(4,4,idx+1)
        idy = (ds.dive_num == dn)
        plt.plot(ds.salinity_uncorrected[idy].values,ds.temperature_uncorrected[idy].values,'-r', alpha=0.2, linewidth=1)
        plt.plot(ds.salinity[idy].values,ds.temperature[idy].values,'-b', alpha=0.2, linewidth=1)
        XLim = plt.xlim()
        YLim = plt.ylim()
        try:
            plt.plot(smooth(ds.salinity_uncorrected[idy].values),smooth(ds.temperature_uncorrected[idy].values),'-r', alpha=0.8, linewidth=1)
            plt.plot(smooth(ds.salinity[idy].values),smooth(ds.temperature[idy].values),'-b', alpha=0.8, linewidth=1)
        except:
            print(f'Not data in dive {dn}')
        plt.xlim(XLim)
        plt.ylim(YLim)
        plt.title(f'Dive {str(dn)}')
        
    
    plt.figure(figsize=(12,12))
    dives = np.floor(np.linspace(0,len(np.unique(np.floor(ds.dive_num.values))),18))[1:-1]
    for idx,dn in enumerate(dives):
        plt.subplot(4,4,idx+1)
        idy = (ds.dive_num == dn) & _deep
        plt.plot(ds.salinity_uncorrected[idy].values,ds.temperature_uncorrected[idy].values,'-r', alpha=0.2, linewidth=1)
        plt.plot(ds.salinity[idy].values,ds.temperature[idy].values,'-b', alpha=0.2, linewidth=1)
        XLim = plt.xlim()
        YLim = plt.ylim()
        try:
            plt.plot(smooth(ds.salinity_uncorrected[idy].values),smooth(ds.temperature_uncorrected[idy].values),'-r', alpha=0.8, linewidth=1)
            plt.plot(smooth(ds.salinity[idy].values),smooth(ds.temperature[idy].values),'-b', alpha=0.8, linewidth=1)
        except:
            print(f'Not data in dive {dn}')
        plt.xlim(XLim)
        plt.ylim(YLim)
        plt.title(f'Dive {str(dn)}')
    
    return final

# Calculate derived variables
def derive_TS_variables(data):
    data = assign(data, 'abs_salinity', gsw.SA_from_SP(data['salinity'].values,data['pressure'].values,data['longitude'].values,data['latitude'].values), attrs = {
                                                         'long_name': 'water absolute salinity',
                                                         'standard_name': 'sea_water_absolute_salinity',
                                                         'units': 'g kg-1',
                                                         'comment': 'Lag corrected',
                                                         'sources': 'salinity pressure longitude latitute',
                                                         'method': 'gsw.SA_from_SP',
                                                         'observation_type': 'calulated',
                                                         'instrument': 'instrument_ctd',
                                                         'valid_max': '40.0',
                                                         'valid_min': '0.0',
                                                         'accuracy': '0.01',
                                                         'precision': '0.01',
                                                         'resolution': '0.001'})
                                                       
    data = assign(data, 'cons_temperature', gsw.CT_from_t(data['abs_salinity'].values,data['temperature'].values,data['pressure'].values), attrs = {
                                                         'long_name': 'water conservative temperature',
                                                         'standard_name': 'sea_water_conservative_temperature',
                                                         'units': 'Celsius',
                                                         'comment': 'Lag corrected',
                                                         'sources': 'abs_salinity temperature pressure',
                                                         'method': 'gsw.CT_from_t',
                                                         'observation_type': 'calulated',
                                                         'instrument': 'instrument_ctd',
                                                         'accuracy': '0.002',
                                                         'precision': '0.001',
                                                         'resolution': '0.0001'})
                                                       
    data = assign(data, 'potential_density', 1000+gsw.sigma0(data['abs_salinity'].values,data['cons_temperature'].values), append_comment = 'Lag corrected.')
                                                       
    data = assign (data, 'sound_speed', gsw.sound_speed(data['abs_salinity'].values,data['cons_temperature'].values,data['pressure'].values), attrs = {
                                                         'long_name': 'water sound speed',
                                                         'standard_name': 'speed_of_sound_in_sea_water',
                                                         'units': 'm s-1',
                                                         'comment': 'Lag corrected',
                                                         'sources': 'abs_salinity cons_temperature pressure',
                                                         'method': 'gsw.sound_speed',
                                                         'observation_type': 'calulated',
                                                         'instrument': 'instrument_ctd'})
    
    
    data = assign (data, 'spiciness', gsw.spiciness0(data['abs_salinity'].values,data['cons_temperature'].values), attrs = {
                                                         'long_name': 'water spiciness',
                                                         'standard_name': 'sea_water_spiciness',
                                                         'units': '',
                                                         'comment': 'Lag corrected, referenced to 0 dbar.',
                                                         'sources': 'abs_salinity cons_temperature',
                                                         'method': 'gsw.spiciness0',
                                                         'observation_type': 'calulated',
                                                         'instrument': 'instrument_ctd'})
    return None
def quick_plot(data,varname):
    #P_bins=np.arange(0,1000,1)
    #x_bins=np.arange(1,np.nanmax(data.profile_index.values),1)
    PD, X, Y = grid2d(data.profile_index.values, data.pressure, data[varname].values, xi=1, yi=1, fn=np.nanmean)
    vmin, vmax = np.nanpercentile(PD, [5, 95])
    plt.figure()
    plt.pcolormesh(X, Y,PD,cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.ylim([1000,0])
    #plt.clim([36.3,38])

def correct_oxygen_global_gain(data, sensor_fixed_salinity=None, surface_threshold=20):
    """
    Applies a global gain correction to oxygen data based on surface saturation mismatch.
    Assumes input oxygen is in mmol/m³ and corrects via saturation scaling using proper solubility.
    Also generates QC plots comparing pre/post saturation and raw/corrected profiles.
    """
    # Prepare variables
    SA = data['abs_salinity']
    CT = data['cons_temperature']
    p = data['depth']
    lon = data['longitude']
    lat = data['latitude']

    # In-situ density [kg/m³]
    rho = gsw.rho(SA, CT, p)

    # Convert observed oxygen from mmol/m³ → µmol/kg
    O2_umolkg = data['oxygen_concentration'] * 1000 / rho
    data['oxygen_umolkg_raw'] = O2_umolkg

    # Compute actual solubility with proper salinity and temperature
    O2sol_actual = gsw.O2sol(SA, CT, p, lon, lat)
    data['oxygen_solubility'] = O2sol_actual

    # Simulate sensor's solubility (with fixed salinity if given)
    SA_sensor = SA * 0 + sensor_fixed_salinity if sensor_fixed_salinity is not None else SA
    O2sol_sensor = gsw.O2sol(SA_sensor, CT, p, lon, lat)

    # Raw sensor-inferred saturation (%)
    sat_percent_raw = O2_umolkg / O2sol_sensor * 100
    data['oxygen_saturation_raw'] = sat_percent_raw

    # Surface mask for scaling
    surface_mask = p < surface_threshold
    mean_surface_sat = sat_percent_raw.where(surface_mask).mean().item()
    gain = 100 / mean_surface_sat
    print(f"Estimated surface saturation = {mean_surface_sat:.2f}% → applying gain factor = {gain:.3f}")

    # Apply gain to saturation, reconstruct corrected O₂
    sat_percent_corrected = sat_percent_raw * gain
    O2_corrected = O2sol_actual * (sat_percent_corrected / 100)
    data['oxygen_corrected'] = O2_corrected
    data['AOU'] = O2sol_actual - O2_corrected

    # --- QC plots ---
    fig, axs = plt.subplots(1, 3, figsize=(15, 6), dpi=100)

    # Left: Histogram of pre-correction saturation
    axs[0].hist(sat_percent_raw.values, np.arange(120), alpha=0.7, label='Full profile')
    axs[0].hist(sat_percent_raw.where(surface_mask).values, np.arange(120), alpha=0.9,
                label=f'Top {surface_threshold}m')
    axs[0].axvline(100, linestyle='--', color='r', label='100%')
    axs[0].set_title('O₂ Saturation (Pre-correction)')
    axs[0].set_xlabel('O₂ saturation (%)')
    axs[0].set_ylabel('Count')
    axs[0].set_ylim(0, 1000000 * 0.2)
    axs[0].legend(loc='upper right')
    axs[0].grid(True)

    # Center: Depth vs O₂ (raw, corrected, solubility)
    axs[1].scatter(data['oxygen_umolkg_raw'], data['depth'], s=0.1, color='k', label='O₂ raw')
    axs[1].scatter(data['oxygen_corrected'], data['depth'], s=0.1, color='c', label='O₂ corrected')
    axs[1].scatter(data['oxygen_solubility'], data['depth'], s=0.1, color='r', label='O₂ solubility')
    axs[1].invert_yaxis()
    axs[1].set_title('O₂: Raw vs Corrected vs Solubility')
    axs[1].set_xlabel('[µmol/kg]')
    axs[1].set_ylabel('Depth [m]')
    axs[1].legend(loc='lower center')
    axs[1].grid(True)

    # Right: Histogram of post-correction saturation
    axs[2].hist(sat_percent_corrected.values, np.arange(120), alpha=0.7, label='Full profile')
    axs[2].hist(sat_percent_corrected.where(surface_mask).values, np.arange(120), alpha=0.9,
                label=f'Top {surface_threshold}m')
    axs[2].axvline(100, linestyle='--', color='r', label='100%')
    axs[2].set_title('O₂ Saturation (Post-correction)')
    axs[2].set_xlabel('O₂ saturation (%)')
    axs[2].set_ylabel('Count')
    axs[2].set_ylim(0, 1000000 * 0.2)
    axs[2].legend(loc='upper right')
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

    return data
    
def betasw_ZHH2009(Tc,S,wavelength=700,theta=117,delta=0.039):
    # Xiaodong Zhang, Lianbo Hu, and Ming-Xia He (2009), Scatteirng by pure
    # seawater: Effect of salinity, Optics Express, Vol. 17, No. 7, 5698-5710 
    #
    # wavelength: backscatter wavelength (nm)
    # Tc: temperauter in degree Celsius
    # S: salinity
    # delta: depolarization ratio, if not provided, default = 0.039 will be used.
    # theta = beam scattering angle in degrees
    # betasw: volume scattering at angles defined by theta. Its size is [x y],
    # where x is the number of angles (x = length(theta)) and y is the number
    # of wavelengths in wavelength (y = length(wavelength))
    # beta90sw: volume scattering at 90 degree. Its size is [1 y]
    # bw: total scattering coefficient. Its size is [1 y]
    # for backscattering coefficients, divide total scattering by 2
    #
    # Xiaodong Zhang, March 10, 2009
    
    def RInw():
        # refractive index of air is from Ciddor (1996,Applied Optics)
        n_air = 1.0 + (5792105.0 / (238.0185 - 1 / (wavelength/1e3)**2) + 167917.0 / (57.362 - 1/(wavelength/1e3)**2)) / 1e8

        # refractive index of seawater is from Quan and Fry (1994, Applied Optics)
        n0 = 1.31405
        n1 = 1.779e-4
        n2 = -1.05e-6
        n3 = 1.6e-8
        n4 = -2.02e-6
        n5 = 15.868
        n6 = 0.01155
        n7 = -0.00423
        n8 = -4382
        n9 = 1.1455e6

        nsw = n0  +  (n1 + n2*Tc + n3*Tc**2)*S  +  n4*Tc**2  +  (n5 + n6*S + n7*Tc)/wavelength  +  n8/wavelength**2  +  n9/wavelength**3 # pure seawater
        nsw = nsw*n_air
        dnswds = (n1 + n2*Tc + n3*Tc**2 + n6/wavelength) * n_air
        return nsw, dnswds
    
    def BetaT():
        # pure water secant bulk Millero (1980, Deep-sea Research)
        kw = 19652.21 + 148.4206*Tc - 2.327105*Tc**2 + 1.360477e-2*Tc**3 - 5.155288e-5*Tc**4
        Btw_cal = 1/kw
        # isothermal compressibility from Kell sound measurement in pure water
        # Btw = (50.88630+0.717582*Tc+0.7819867e-3*Tc**2+31.62214e-6*Tc**3-0.1323594e-6*Tc**4+0.634575e-9*Tc**5)./(1+21.65928e-3*Tc)*1e-6;
        # seawater secant bulk
        a0 = 54.6746 - 0.603459*Tc + 1.09987e-2*Tc**2 - 6.167e-5*Tc**3
        b0 = 7.944e-2 + 1.6483e-2*Tc - 5.3009e-4*Tc**2
        Ks = kw + a0*S + b0*S**1.5

        # calculate seawater isothermal compressibility from the secant bulk
        IsoComp = 1/Ks*1e-5 # unit is pa
        return IsoComp

    def rhou_sw():
        # density of water and seawater,unit is Kg/m^3, from UNESCO,38,1981
        a0 = 8.24493e-1
        a1 = -4.0899e-3
        a2 = 7.6438e-5
        a3 = -8.2467e-7
        a4 = 5.3875e-9
        a5 = -5.72466e-3
        a6 = 1.0227e-4
        a7 = -1.6546e-6
        a8 = 4.8314e-4
        
        b0 = 999.842594
        b1 = 6.793952e-2
        b2 = -9.09529e-3
        b3 = 1.001685e-4
        b4 = -1.120083e-6
        b5 = 6.536332e-9

        # density for pure water 
        density_w = b0+b1*Tc+b2*Tc**2+b3*Tc**3+b4*Tc**4+b5*Tc**5
        # density for pure seawater
        density_sw = density_w +((a0+a1*Tc+a2*Tc**2+a3*Tc**3+a4*Tc**4)*S+(a5+a6*Tc+a7*Tc**2)*S**1.5+a8*S**2)
        return density_sw

    
    def dlnasw_ds():
        # water activity data of seawater is from Millero and Leung (1976,American
        # Journal of Science,276,1035-1077). Table 19 was reproduced using
        # Eqs.(14,22,23,88,107) then were fitted to polynominal equation.
        # dlnawds is partial derivative of natural logarithm of water activity
        # w.r.t.salinity
        # lnaw = (-1.64555e-6-1.34779e-7*Tc+1.85392e-9*Tc**2-1.40702e-11*Tc**3)+......
        #            (-5.58651e-4+2.40452e-7*Tc-3.12165e-9*Tc**2+2.40808e-11*Tc**3)*S+......
        #            (1.79613e-5-9.9422e-8*Tc+2.08919e-9*Tc**2-1.39872e-11*Tc**3)*S**1.5+......
        #            (-2.31065e-6-1.37674e-9*Tc-1.93316e-11*Tc**2)*S**2;

        dlnawds = (-5.58651e-4+2.40452e-7*Tc-3.12165e-9*Tc**2+2.40808e-11*Tc**3) + 1.5*(1.79613e-5-9.9422e-8*Tc+2.08919e-9*Tc**2-1.39872e-11*Tc**3)*S**0.5 + 2*(-2.31065e-6-1.37674e-9*Tc-1.93316e-11*Tc**2)*S
        return dlnawds

    
    # density derivative of refractive index from PMH model
    def PMH(n_wat):
        n_wat2 = n_wat**2
        n_density_derivative = (n_wat2 - 1) * (1+2/3*(n_wat2+2) * (n_wat/3-1/3/n_wat)**2)
        return n_density_derivative
    
    # values of the constants
    Na = 6.0221417930e23   #  Avogadro's constant
    Kbz = 1.3806503e-23    #  Boltzmann constant
    Tk = Tc+273.15         #  Absolute tempearture
    M0 = 18e-3             #  Molecular weigth of water in kg/mol
    
    rad = theta*np.pi/180 # angle in radian as a colum variable

    # nsw: absolute refractive index of seawater
    # dnds: partial derivative of seawater refractive index w.r.t. salinity
    nsw, dnds = RInw()

    # isothermal compressibility is from Lepple & Millero (1971,Deep
    # Sea-Research), pages 10-11
    # The error ~ +/-0.004e-6 bar^-1
    IsoComp = BetaT()

    # density of water and seawater,unit is Kg/m^3, from UNESCO,38,1981
    density_sw = rhou_sw()

    # water activity data of seawater is from Millero and Leung (1976,American
    # Journal of Science,276,1035-1077). Table 19 was reproduced using
    # Eq.(14,22,23,88,107) then were fitted to polynominal equation.
    # dlnawds is partial derivative of natural logarithm of water activity
    # w.r.t.salinity
    dlnawds = dlnasw_ds()

    # density derivative of refractive index from PMH model
    DFRI = PMH(nsw)  ## PMH model

    # volume scattering at 90 degree due to the density fluctuation
    beta_df = np.pi*np.pi/2*((wavelength*1e-9)**(-4))*Kbz*Tk*IsoComp*DFRI**2*(6+6*delta)/(6-7*delta)
    
    # volume scattering at 90 degree due to the concentration fluctuation
    flu_con = S*M0*dnds**2/density_sw/(-dlnawds)/Na
    beta_cf = 2*np.pi*np.pi*((wavelength*1e-9)**(-4))*nsw**2*(flu_con)*(6+6*delta)/(6-7*delta)
    
    # total volume scattering at 90 degree
    beta90sw = beta_df+beta_cf
    bsw=8*np.pi/3*beta90sw*(2+delta)/(1+delta)
    
    betasw = beta90sw * (1+((np.cos(rad))**2)*(1-delta)/(1+delta))

    return betasw,beta90sw,bsw
def calc_particulate_backscatter(data):
    def calculate_bbp(beta_total, beam_angle=117, wavelength=700):
        # Scaled output from SeaExplorer gives us Beta for water and particple.
        # https://oceanobservatories.org/wp-content/uploads/2015/10/1341-00540_Data_Product_SPEC_FLUBSCT_OOI.pdf
        beta_sw = betasw_ZHH2009(data.temperature.values,data.salinity.values, wavelength, beam_angle)
        beta_p = beta_total - beta_sw
        Chi_p = 1.08 # For 117* angle (Sullivan & Twardowski, 2009)
        # Chi_p = 1.17 # For 140* angle (Sullivan & Twardowski, 2009)
        bbp = 2 * np.pi * Chi_p * beta_p # in m-1
        return bbp
    
    bbp,_,_ = calculate_bbp(data['backscatter_scaled'].values, wavelength=700)
    data['paticulate_backscatter'] = ('time', bbp)

    return data


def extract_nearest_timeseries(ds, lonD, latD):
    """
    Extract a time series of nearest grid point data from an xarray.Dataset.
    
    Parameters:
        ds (xarray.Dataset): The input dataset with 'time', 'latitude', 'longitude' coordinates.
        lonD (array-like): Array of longitudes corresponding to each time step.
        latD (array-like): Array of latitudes corresponding to each time step.
        
    Returns:
        xarray.Dataset: A new dataset with dimension 'time' containing data variables
                        at the nearest (longitude, latitude) point for each time.
    """
    # List to store the subset for each time step
    data_list = []

    # Ensure the number of positions matches the time dimension length
    if len(lonD) != ds.dims['time'] or len(latD) != ds.dims['time']:
        raise ValueError("Length of lonD and latD must equal the length of the 'time' dimension in the dataset.")

    # Loop over each time step
    for i, time_value in enumerate(ds.time.values):
        # Select the data at the nearest (lon, lat) for this time step
        subset = ds.sel(
            time=time_value,
            longitude=lonD[i],
            latitude=latD[i],
            method='nearest'
        )
        data_list.append(subset)

    # Concatenate along the time dimension
    ds_subset = xr.concat(data_list, dim="time")
    
    # Optionally, reassign the original time coordinate (if needed)
    ds_subset = ds_subset.assign_coords(time=ds.time)

    return ds_subset
def compute_N2(ds, sal_key='abs_salinity', temp_key='cons_temperature'):
    # Automatically detect vertical and profile dimension names
    dim_y, dim_x = ds[sal_key].dims  # Assume 2D input: (vertical, profile)

    # # Extract vertical coordinate (depth/pressure)
    # P = np.tile(ds[dim_y].values[:, np.newaxis], (1, ds.dims[dim_x]))

    # # Compute N² and mid-pressure
    # N2, Pmid =  gsw.Nsquared(ds[sal_key], ds[temp_key], P)

    # # Interpolate N² to original pressure grid
    # N2_interp = np.full_like(P, np.nan)
    # for i in range(N2.shape[1]):
    #     N2_interp[:, i] = np.interp(P[:, i], Pmid[:, i], N2[:, i])

    # # Assign to dataset
    rho=ds.potential_density
    
    N2_interp=((9.8/rho)*np.gradient(rho,ds.depth.values,axis=0)).values
    ds['N2'] = ((dim_y, dim_x), N2_interp)
    return ds
def compute_Ri(ds, sigma=1.0):
    from scipy.ndimage import gaussian_filter1d
    """
    Compute Ri after vertically smoothing abs_salinity and cons_temperature.

    Parameters:
    -----------
    ds : xarray.Dataset
        Must contain 'abs_salinity' and 'cons_temperature' with shape (depth, profile).
    sigma : float
        Standard deviation (in grid points) for Gaussian smoothing along the vertical (axis=0).

    Returns:
    --------
    Ri : np.ndarray
        Ri computed from smoothed inputs, same shape as input.
    """
    depth_step=ds.depth[2].values-ds.depth[1].values
    # Smooth inputs vertically (axis=0 = depth)
    SA_smooth = gaussian_filter1d(ds.abs_salinity.values, sigma=sigma, axis=0, mode='nearest')
    CT_smooth = gaussian_filter1d(ds.cons_temperature.values, sigma=sigma, axis=0, mode='nearest')
    shear_E = np.gradient(ds.velocity_E_DAC_reference_detided, depth_step, axis=0)
    shear_N = np.gradient(ds.velocity_N_DAC_reference_detided, depth_step, axis=0)
    dim_y, dim_x = ds['abs_salinity'].dims  # Assume 2D input: (vertical, profile)
    # Extract vertical coordinate (depth/pressure)
    P = np.tile(ds[dim_y].values[:, np.newaxis], (1, ds.dims[dim_x]))
    # Compute N² and mid-pressure
    N2, Pmid =  gsw.Nsquared(SA_smooth, CT_smooth, P)
    # Interpolate N² to original pressure grid
    N2_interp = np.full_like(P, np.nan)
    for i in range(N2.shape[1]):
        N2_interp[:, i] = np.interp(P[:, i], Pmid[:, i], N2[:, i])

    # Compute Ri
    Ri = N2_interp /(shear_E ** 2 + shear_N ** 2)

    # Assign to dataset
    ds['Ri'] = ((dim_y, dim_x), Ri)
    return ds

def compute_Tu(ds, sal_key='abs_salinity', temp_key='cons_temperature'):
    # Automatically detect vertical and profile dimension names
    dim_y, dim_x = ds[sal_key].dims  # Assume 2D input: (vertical, profile)

    # Extract pressure/depth array
    P = np.tile(ds[dim_y].values[:, np.newaxis], (1, ds.dims[dim_x]))

    # Compute Turner angle and pressure midpoints
    Tu, Rrho, Pmid = gsw.Turner_Rsubrho(ds[sal_key], ds[temp_key], P, axis=0)

    # Interpolate Tu to original pressure grid
    Tu_interp = np.full_like(P, np.nan)
    Rrho_interp = np.full_like(P, np.nan)
    for i in range(Tu.shape[1]):
        Tu_interp[:, i] = np.interp(P[:, i], Pmid[:, i], Tu[:, i])
        Rrho_interp[:, i] = np.interp(P[:, i], Pmid[:, i], Rrho[:, i])
    # Recover original NaNs from temperature
    temp_vals = ds[temp_key].values
    Tu_interp[np.isnan(temp_vals)] = np.nan
    Rrho_interp[np.isnan(temp_vals)] = np.nan
    # Assign to dataset
    ds['Tu'] = ((dim_y, dim_x), Tu_interp)
    ds['Rrho'] = ((dim_y, dim_x), Rrho_interp)
    return ds
def plot_density_ratio(da, cbar_label=r'$R_\rho$',limit=3,limit_DC=1/3, ax=None, cmap='none'):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import numpy.ma as ma
    #da=da.where(da<limit,np.nan)

    #d57["Rrho_filtered"] = (["depth", "time_profile"], Rrho.where(Rrho<30,np.nan).values)
    # 4 categories -> 5 bounds (we'll FORCE extremes into first/last bin)
    bounds = [-1, limit_DC,1, limit, limit+1]

    # # --- Colors: 4 bins -> 4 colors ---
    # if colc == 0:
    #     colors = [
    #         (213/255, 132/255,  79/255),   # Rrho < 0
    #         (166/255, 216/255, 146/255),   # 0 <= Rrho <= 1
    #         (245/255, 155/255, 193/255),   # 1 < Rrho <= 10
    #         (170/255, 170/255, 170/255),   # Rrho > 10
    #     ]
    # elif colc == 1:
    #     factor2 = 1
    #     colors = ["#9FC5E8",  "#AA1616", "darkorange", "#1a1a1a"]
    # elif colc == 2:
    #     # requires: import cmocean.cm as cmo
    #     colors = [
    #         cmo.deep.resampled(6)(0)[:3],     # Rrho < 0
    #         cmo.matter.resampled(4)(2)[:3],   # 0 <= Rrho <= 1
    #         cmo.deep.resampled(7)(3)[:3],     # 1 < Rrho <= 10
    #         cmo.gray.resampled(4)(0)[:3],     # Rrho > 10
    #     ]

    # cmap = ListedColormap(colors)
    cmap=cmap
    cmap.set_bad(color='none')  # NaNs transparent

    # IMPORTANT: clip=True so nothing becomes "under/over" -> no white outside
    norm = BoundaryNorm(bounds, ncolors=4, clip=True)

    # --- Force extremes into first/last bin (keeps your categorical logic) ---
    R = da.values.astype(float)
    R = np.where(np.isnan(R), np.nan, R)
    R = np.where(R < limit_DC, -0.5, R)     # inside [-1,limit_DC)
    R = np.where(R > limit, limit+0.5, R)    # inside (limit,limit+1]

    R_array = ma.masked_invalid(R)

    y = da[da.dims[0]].values
    x = da[da.dims[1]].values

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    pc = ax.pcolormesh(x, y, R_array, cmap=cmap, norm=norm, shading='auto')

    # --- Colorbar with equal segment sizes ---
    cax = ax.inset_axes([1.01, 0.05, 0.015, 0.9])
    cbar = plt.colorbar(
        pc, cax=cax,
        boundaries=bounds,   # SAME as norm bounds
        spacing='uniform',   # equal space per category
        pad=0.02
    )
    cbar.ax.tick_params(axis='y', which='both', length=0)

    ticks = [(-1+limit_DC)/2, (limit_DC+1)/2, (1+limit)/2, limit+0.5]
    #labels = [r'$R_\rho<0$', r'$0\leq R_\rho\leq 1$', r'$1<R_\rho\leq 10$', r'$R_\rho>10$']
    labels = [r'$R_\rho<1/3$','DC', 'SF', r'$R_\rho>' + str(limit) + '$']
    cbar.set_ticks(ticks)
    cbar.ax.set_yticklabels(labels,fontsize=14,rotation=-30)
    # cbar.set_label(cbar_label)

    return pc, cbar


def plot_turner_angle(da, cbar_label=r'$Tu\ (°)$', ax=None,colc=0):
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import numpy.ma as ma
    """
    Manual Turner angle plot with proportional colorbar segments and NaN transparency.

    Parameters:
    -----------
    da : xarray.DataArray
        Turner angle in degrees with dims (depth, profile) or similar.
    cbar_label : str
        Colorbar label.
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        Axis to plot into.
    """
    # --- 1. Define Turner angle bins and color regime ---
    #bounds = [-90, -60, -45, 45, 60, 90, 100.0]
    bounds=[-90, -45,45,90, 100.0]
    if colc==0:
        colors=[cmo.matter.resampled(4)(2)[:3],cmo.deep.resampled(6)(0)[:3],cmo.deep.resampled(7)(3)[:3],cmo.gray.resampled(4)(0)[:3]]
    elif colc==1:
        colors=[cmo.matter.resampled(4)(2)[:3],cmo.deep.resampled(6)(0)[:3],cmo.deep.resampled(7)(3)[:3],cmo.gray.resampled(4)(0)[:3]]
    elif colc==2:
        #colors=[cmo.matter.resampled(4)(1)[:3],cmo.matter.resampled(4)(2)[:3],cmo.deep.resampled(4)(0)[:3],cmo.deep.resampled(4)(2)[:3],cmo.deep.resampled(4)(1)[:3],cmo.gray.resampled(4)(0)[:3]]
        colors=[cmo.matter.resampled(4)(2)[:3],cmo.deep.resampled(6)(0)[:3],cmo.deep.resampled(7)(3)[:3],cmo.gray.resampled(4)(0)[:3]]
    #ticks = [-75, -52.5, 0, 52.5, 75, 95]
    ticks = [-60, 0, 60, 95]
    #labels = ['CD', 'wCD', 'stable', 'wSF', 'SF', 'Unstable']
    labels = ['CD', 'stable', 'SF', r'$N_2<0$']

    # --- 2. Setup colormap and norm ---
    cmap = ListedColormap(colors)
    cmap.set_bad(color='none')  # NaNs will be transparent
    norm = BoundaryNorm(bounds, len(colors))

    # --- 3. Clean & mask data ---
    Tu_plot = da.where((da >= -90) & (da <= 90), other=120)
    Tu_plot = Tu_plot.where(~np.isnan(da))  # preserve NaNs
    Tu_array = ma.masked_invalid(Tu_plot.values)

    # --- 4. Prepare meshgrid for pcolormesh ---
    y = da[da.dims[0]].values
    x = da[da.dims[1]].values
    # --- 5. Plot ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    pc = ax.pcolormesh(x, y, Tu_array, cmap=cmap, norm=norm, shading='auto')
    #ax.set_title('Turner Angle')
    #ax.set_xlabel(da.dims[1])
    #ax.set_ylabel(da.dims[0])

    # --- 6. Add proportional colorbar ---
    #cbar = plt.colorbar(pc, ax=ax, boundaries=bounds, spacing='proportional')
    cax = ax.inset_axes([1.01, 0.05, 0.015, 0.9])
    cbar = plt.colorbar(pc, cax=cax, boundaries=bounds, spacing='proportional', pad=0.02)
    cbar.ax.tick_params(axis='y', which='both', length=0)

    cbar.set_ticks(ticks)
    cbar.ax.set_yticklabels(labels)
    #cbar.set_label(cbar_label)

    return ax
def grid_in_rho(ds):
    # Variables to interpolate (if present)
    variables_to_interpolate = [
        "pressure","chlorophyll", "backscatter_scaled", "potential_density", "abs_salinity",
        "cons_temperature", "spiciness", "oxygen_corrected", "AOU", "Tu",
        "N2", "e1", "e2", "epsi_QC_gm", "epsi_QC_am",
        "velocity_N_DAC_reference_sb_corrected", "velocity_E_DAC_reference_sb_corrected",
        "velocity_E_DAC_reference_detided", "velocity_N_DAC_reference_detided",
        "nitrate_concentration", "nitrate_molar_concentration"
    ]

    # Automatically determine vertical and profile dimensions
    for dim1 in ds.dims:
        for dim2 in ds.dims:
            if dim1 == dim2:
                continue
            test_shape = (ds.dims[dim1], ds.dims[dim2])
            if ds.potential_density.shape == test_shape:
                vertical_dim, profile_dim = dim1, dim2
                break

    sigma0 = ds["potential_density"]-1000
    dens_array = np.arange(
        np.round(np.nanmin(sigma0.values) * 10) / 10 - 0.1,
        np.round(np.nanmax(sigma0.values) * 10) / 10 + 0.1,
        0.01,
    )

    new_data = {}

    for var in tqdm(variables_to_interpolate, desc="Interpolating variables"):
        if var not in ds:
            continue  # Skip missing variables

        var_data = ds[var]
        interpolated = np.full((len(dens_array), ds.dims[profile_dim]), np.nan)

        for i in range(ds.dims[profile_dim]):
            sigma_profile = sigma0.isel({profile_dim: i}).values
            var_profile = var_data.isel({profile_dim: i}).values

            valid = ~np.isnan(sigma_profile) & ~np.isnan(var_profile)
            sigma_valid = sigma_profile[valid]
            var_valid = var_profile[valid]

            if len(sigma_valid) > 1:
                if var == "Tu":
                    # Nearest interpolation
                    idx = np.abs(sigma_valid[:, None] - dens_array[None, :]).argmin(axis=0)
                    interpolated[:, i] = var_valid[idx]
                else:
                    # Linear interpolation
                    f_interp = interp1d(sigma_valid, var_valid, kind='linear', bounds_error=False, fill_value=np.nan)
                    interpolated[:, i] = f_interp(dens_array)

        new_data[var] = (["density", profile_dim], interpolated)

    dsd_rho = xr.Dataset(
        new_data,
        coords={
            "density": dens_array,
            profile_dim: ds[profile_dim],
        },
    )

    return dsd_rho
def grid_in_P(ds_inrho, ds):
    """
    Regrid all variables from ds_inrho (density grid) onto the vertical grid of d57 (e.g., pressure or depth).
    Assumes:
        - ds_inrho: gridded on (dim1, dim2)
        - d57: target grid on (dim3, dim2)
    Returns:
        - New xarray.Dataset with variables on (dim3, dim2)
    """
    # Infer dimensions
    dim1, dim2 = next(iter(ds_inrho.data_vars.values())).dims
    dim3 = [d for d in ds.dims if d != dim2][0]  # vertical dim from ds

    sigma0 = ds["potential_density"]-1000  # shape (dim3, dim2)
    density_grid = ds_inrho[dim1].values
    new_data = {}

    for var in tqdm(ds_inrho.data_vars, desc="Regridding to pressure grid"):
        var_interp = np.full_like(sigma0.values, np.nan)

        for i in range(ds.dims[dim2]):
            sigma_profile = sigma0.isel({dim2: i}).values
            var_profile = ds_inrho[var].isel({dim2: i}).values

            if np.all(np.isnan(var_profile)) or np.all(np.isnan(sigma_profile)):
                continue

            try:
                f_interp = interp1d(density_grid, var_profile, bounds_error=False, fill_value=np.nan)
                var_interp[:, i] = f_interp(sigma_profile)
            except:
                continue

        new_data[var] = ([dim3, dim2], var_interp)

    ds_inP = xr.Dataset(
        new_data,
        coords={
            dim3: ds[dim3],
            dim2: ds[dim2],
        },
    )

    return ds_inP

def spiciness_(ds, sigma=1.0):
    from scipy.ndimage import gaussian_filter1d
    """
    Compute spiciness0 after vertically smoothing abs_salinity and cons_temperature.

    Parameters:
    -----------
    ds : xarray.Dataset
        Must contain 'abs_salinity' and 'cons_temperature' with shape (depth, profile).
    sigma : float
        Standard deviation (in grid points) for Gaussian smoothing along the vertical (axis=0).

    Returns:
    --------
    spiciness_smooth : np.ndarray
        Spiciness0 computed from smoothed inputs, same shape as input.
    """
    # Smooth inputs vertically (axis=0 = depth)
    SA_smooth = gaussian_filter1d(ds.abs_salinity.values, sigma=sigma, axis=0, mode='nearest')
    CT_smooth = gaussian_filter1d(ds.cons_temperature.values, sigma=sigma, axis=0, mode='nearest')

    # Compute spiciness0 from smoothed SA and CT
    spiciness_smooth = gsw.spiciness0(SA_smooth, CT_smooth)

    return spiciness_smooth
#########################################################################################################
####################################### UNTESTED FUNCTIONS ##############################################
#########################################################################################################

def download_ERA5(years=[2021], months=[1], days=[1], hours=[0], area=[0,0,0,0], variables=None, output_name=None):
    """
    Returns xarray dataset.
    
    years = list of year numbers
    months = list of month numbers
    days = list of days, from 1 to 31
    hours = list of hours from 0 to 23
    area = list of [N,W,S,E]
    variables = default collects heat flux variables, otherwise, specify.
    output_name = file to save to if desired
    """
    
    # Area = [N,W,S,E]
    
    # First provided by Marcel du Plessis
    # Loading ERA5 data directly from the Copernicus Data Service
    # The goal of this notebook is to be able to access and analysis ERA5 data using cloud computing. 
    # This has the obvious advantage of not having to download and store the data on your local computer, 
    # which can quicly add up to terrabytes if you're looking for long term data.

    # I am following an example from https://towardsdatascience.com/read-era5-directly-into-memory-with-python-511a2740bba0

    # Variables on the single levels reanalysis can be found here: 
    # https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview

    # Let's see how we go:
    
    if variables is None:
        variables = [
            '2m_temperature',
            '10m_u_component_of_wind',
            '10m_v_component_of_wind',
            'sea_surface_temperature',
            'skin_temperature',
            'surface_pressure',
            'surface_latent_heat_flux',
            'surface_sensible_heat_flux',
            'surface_net_solar_radiation',
            'surface_net_thermal_radiation',
            'total_precipitation',
            'evaporation']
        
    import cdsapi
    import xarray as xr
    from urllib.request import urlopen # start the client

    import certifi
    import urllib3
    http = urllib3.PoolManager(
        cert_reqs='CERT_REQUIRED',
        ca_certs=certifi.where()
        )

    cds = cdsapi.Client() # dataset you want to read

    dataset = "reanalysis-era5-single-levels" # flag to download data
    download_flag = True # api parameters 
    params = {
              'variable': variables,
              'product_type': 'reanalysis',
              'year': [str(x) for x in years],
              'month': [str(format(x, '02')) for x in months],
              'day': [str(format(x, '02')) for x in days],
              'time': [str(format(x, '02'))+':00' for x in hours],
              'area': area,
              'format': 'netcdf'
             }

    fl = cds.retrieve(dataset, params) # download the file 

    if output_name is not None:
        if output_name[-3:] != '.nc':
            output_name = output_name+'.nc'
            
        fl.download(str(output_name))

    with urlopen(fl.location) as f:
        ds = xr.open_dataset(f.read())
        
    return ds


def trapz_from_mld_plus_m(flux: xr.DataArray,depth: xr.DataArray,mld: xr.DataArray,m: float,*,max_depth: float | None = None,require_both_endpoints: bool = True,fill_all_nan_with_zero: bool = False,):
    """
    Trapezoidal vertical integral of `flux(depth, time_profile)` from z0 = mld + m
    down to `max_depth` (if provided) or the deepest grid depth.
    Assumes depth in meters, increasing downward.
    Returns
    -------
    integrated : DataArray (time_profile,)
    """

    # Ensure depth is a coordinate for flux
    flux = flux.assign_coords(depth=depth)

    # Start depth per profile
    z0 = mld + m  # (time_profile,)

    # Interval thickness and endpoint fluxes
    dz = depth.diff("depth")  # (depth-1,)
    f0 = flux.isel(depth=slice(None, -1))
    f1 = flux.isel(depth=slice(1, None))

    # Depth at interval endpoints
    z_lo = depth.isel(depth=slice(None, -1))
    z_hi = depth.isel(depth=slice(1, None))

    # Broadcast start depth to interval grid
    z0_b = z0.broadcast_like(f0)

    # Upper integration limit
    if max_depth is None:
        zmax = float(depth.max().values)
    else:
        zmax = float(max_depth)

    # Intervals fully inside [z0, zmax]
    z_lo_b = z_lo.broadcast_like(f0)
    z_hi_b = z_hi.broadcast_like(f0)

    inside = (z_lo_b >= z0_b) & (z_hi_b >= z0_b) & (z_hi_b <= zmax)

    # If z0 is NaN, integrate nothing
    inside = inside.where(np.isfinite(z0_b), other=False)

    # Valid flux endpoints (to avoid bridging across NaNs)
    if require_both_endpoints:
        valid_flux = np.isfinite(f0) & np.isfinite(f1)
    else:
        valid_flux = np.isfinite(f0) | np.isfinite(f1)

    w = inside & valid_flux

    # Align dz with interval depth axis
    dz_i = dz.assign_coords(depth=f0.depth)

    # Trapezoid contribution and sum
    contrib = 0.5 * (f0 + f1) * dz_i
    integrated = contrib.where(w).sum("depth", skipna=True)

    # Ensure profiles with no valid intervals become NaN (or 0 if chosen)
    any_interval = w.any("depth")
    integrated = integrated.where(any_interval)

    if fill_all_nan_with_zero:
        integrated = integrated.fillna(0)

    return integrated



