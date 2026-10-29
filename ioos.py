"""IOOS mooring helpers for build_climatology.py and update.py.

Mooring data come from the IOOS sensor catalog's ERDDAP server. Only surface
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
SURFACE_MAX_DEPTH_M = 3
VALID_C = (-2, 40)
SPIKE_C = 2.0
SPIKE_WINDOW = '6h'


def _query(dataset, constraints, extra=''):
    """CSV rows from ERDDAP tabledap, or an empty frame when there are none (ERDDAP answers 404)."""
    url = (f'{ERDDAP}/tabledap/{dataset}.csv?time,z,sea_water_temperature'
           f'&z>={-SURFACE_MAX_DEPTH_M}&z<=0'
           f'&sea_water_temperature>={VALID_C[0]}&sea_water_temperature<={VALID_C[1]}'
           f'&sea_water_temperature_qc_agg!=4{constraints}{extra}')
    r = http.get(quote(url, safe=':/?&=,!<>'), timeout=300)
    if r.status_code == 404:
        return pd.DataFrame(columns=['time', 'z', 'sea_water_temperature'])
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text), skiprows=[1])


def surface_series(df):
    """Water temperature (°C) at the shallowest sensor, indexed by UTC time."""
    if df.empty:
        return pd.Series(dtype=float, name='wtmp_c', index=pd.DatetimeIndex([], tz='UTC'))
    df = df[df['z'] == df['z'].max()]
    s = pd.Series(df['sea_water_temperature'].values, index=pd.to_datetime(df['time'], utc=True), name='wtmp_c')
    return s[~s.index.duplicated()].sort_index()


def despike(s):
    """Drop readings more than SPIKE_C from the centered SPIKE_WINDOW running median."""
    if s.empty:
        return s
    median = s.rolling(SPIKE_WINDOW, center=True, min_periods=1).median()
    return s[(s - median).abs() <= SPIKE_C]


def recent(dataset, days):
    """The past `days` of surface water temperature, despiked."""
    return despike(surface_series(_query(dataset, f'&time>=now-{days}days')))


def daily_history(dataset):
    """Daily-mean surface water temperature over the whole record (averaged by ERDDAP)."""
    df = _query(dataset, '', '&orderByMean("time/1day,z")')
    return surface_series(df), (None if df.empty else float(df['z'].max()))
