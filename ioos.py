"""IOOS mooring and shore-station helpers for build_climatology.py and update.py.

Mooring and shore-station data come from the IOOS sensor catalog's ERDDAP server, and for the
Washington moorings NANOOS's ERDDAP server (which names depth and the QC flag differently) and the
University of Washington's (its ORCA/NPBY profiling buoys: one cast every few hours). Only surface
sensors are used, to match the buoys (NDBC and CDIP measure ~0.5-1 m deep):
the shallowest sensor within SURFACE_MAX_DEPTH_M of the surface. Values that
failed the catalog's automated QC (qc_agg 4) or fall outside a plausible
range (some records hold huge fill values) are dropped, and so are spikes:
readings more than SPIKE_C from the median of the surrounding SPIKE_WINDOW
(much of the mooring data is never QC-tested, e.g. MBARI M1 has a lone 0 °C
reading in a 17 °C record).
"""
import io
from urllib.parse import quote

import pandas as pd

from ndbc import http

ERDDAP = 'https://erddap.sensors.ioos.us/erddap'
NANOOS_ERDDAP = 'https://erddap.nanoos.org/erddap'
# datasets read from NANOOS's server: depth is positive down ("depth", not "z") and the QC flag
# is <var>_qc_aggregate (QARTOD flags as on the IOOS server: 4 is a failed test)
NANOOS_DATASETS = {'chaba_hydro', 'bellinghambay_surfhydro'}
UW_ERDDAP = 'https://nwem.apl.washington.edu/erddap'
# UW's profiling buoys in Puget Sound and Hood Canal (what NANOOS's orca_hydro_* datasets point to):
# gridded casts (cast_start_time x depth, 0.25 m bins), whose shallowest good bin is ~0.25-3 m
UW_PROFILERS = {'orca1_L3_depthgridded_025', 'orca2_L3_depthgridded_025', 'orca3_L3_depthgridded_025',
                'npby1_L3_depthgridded_025', 'npby2_L3_depthgridded_025'}
SURFACE_MAX_DEPTH_M = 3
VALID_C = (-2, 40)
SPIKE_C = 2.0
SPIKE_WINDOW = '6h'


def _query(dataset, constraints, extra='', var='sea_water_temperature'):
    """CSV rows from ERDDAP tabledap, or an empty frame when there are none (ERDDAP answers 404).
    `var` is the dataset's water-temperature variable (e.g. sea_water_temperature_ctd at
    Stearns Wharf); it's returned as the column sea_water_temperature."""
    if dataset in NANOOS_DATASETS:
        url = (f'{NANOOS_ERDDAP}/tabledap/{dataset}.csv?time,depth,{var}'
               f'&depth>=0&depth<={SURFACE_MAX_DEPTH_M}'
               f'&{var}>={VALID_C[0]}&{var}<={VALID_C[1]}'
               f'&{var}_qc_aggregate!=4{constraints}{extra.replace(",z", "")}')
    else:
        url = (f'{ERDDAP}/tabledap/{dataset}.csv?time,z,{var}'
               f'&z>={-SURFACE_MAX_DEPTH_M}&z<=0'
               f'&{var}>={VALID_C[0]}&{var}<={VALID_C[1]}'
               f'&{var}_qc_agg!=4{constraints}{extra}')
    r = http.get(quote(url, safe=':/?&=,!<>'), timeout=300)
    if r.status_code == 404:
        return pd.DataFrame(columns=['time', 'z', 'sea_water_temperature'])
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), skiprows=[1]).rename(columns={var: 'sea_water_temperature'})
    if 'depth' in df:
        df = df.assign(z=-df.pop('depth'))  # as z: height, negative below the surface
    return df


def _profiles(dataset, start):
    """The top SURFACE_MAX_DEPTH_M of a UW profiling buoy's casts since `start` (an ERDDAP time, e.g.
    2026-10-01T00:00:00Z; None for the whole record), as rows of time, z and sea_water_temperature."""
    casts = f'[({start}):1:(last)]' if start else '[0:1:last]'
    dims = f'{casts}[(0):1:({SURFACE_MAX_DEPTH_M})]'
    url = (f'{UW_ERDDAP}/griddap/{dataset}.csv?sea_water_temperature{dims},'
           f'sea_water_temperature_qc_aggregate{dims}')
    r = http.get(quote(url, safe=':/?&=,()'), timeout=600)
    if r.status_code == 404:  # nothing since `start`
        return pd.DataFrame(columns=['time', 'z', 'sea_water_temperature'])
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), skiprows=[1])
    t = df['sea_water_temperature']
    df = df[t.between(*VALID_C) & (df['sea_water_temperature_qc_aggregate'] != 4)]
    return pd.DataFrame({'time': df['cast_start_time'], 'z': -df['depth'], 'sea_water_temperature': df['sea_water_temperature']})


def surface_series(df):
    """Water temperature (°C) from the shallowest reading at each time (within the top
    SURFACE_MAX_DEPTH_M), indexed by UTC time. Profiling sensors such as Cal Poly Pier's
    report 1 m bins whose shallowest level shifts with the tide."""
    if df.empty:
        return pd.Series(dtype=float, name='wtmp_c', index=pd.DatetimeIndex([], tz='UTC'))
    df = df.sort_values('z').groupby('time').tail(1)
    s = pd.Series(df['sea_water_temperature'].values, index=pd.to_datetime(df['time'], utc=True), name='wtmp_c')
    return s[~s.index.duplicated()].sort_index()


def despike(s):
    """Drop readings more than SPIKE_C from the centered SPIKE_WINDOW running median."""
    if s.empty:
        return s
    median = s.rolling(SPIKE_WINDOW, center=True, min_periods=1).median()
    return s[(s - median).abs() <= SPIKE_C]


def recent(dataset, days, var='sea_water_temperature'):
    """The past `days` of surface water temperature, despiked."""
    if dataset in UW_PROFILERS:
        start = (pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=days)).strftime('%Y-%m-%dT%H:%M:%SZ')
        return despike(surface_series(_profiles(dataset, start)))
    return despike(surface_series(_query(dataset, f'&time>=now-{days}days', var=var)))


def daily_history(dataset, var='sea_water_temperature'):
    """Daily-mean surface water temperature over the whole record (averaged by ERDDAP), and the
    sensor's height (z, negative below the surface)."""
    if dataset in UW_PROFILERS:  # daily means of each cast's shallowest good reading; its usual depth
        df = _profiles(dataset, None)
        df = df.sort_values('z').groupby('time').tail(1)
        s = surface_series(df)
        return s.resample('D').mean().dropna(), (None if df.empty else float(df['z'].median()))
    df = _query(dataset, '', '&orderByMean("time/1day,z")', var=var)
    return surface_series(df), (None if df.empty else float(df['z'].max()))
