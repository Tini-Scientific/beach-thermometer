"""Shared NDBC helpers for build_climatology.py and update.py."""
import io

import pandas as pd
import requests

NDBC_URL = 'https://www.ndbc.noaa.gov'

http = requests.Session()
http.headers['User-Agent'] = 'TiniScientific-beachthermometer/0.1 (beachthermometer.com)'


def read_stdmet(text):
    """Time (UTC) and water temperature (°C) from an NDBC stdmet text file.
    Handles every era of the format: 2-digit years before 1999, no minute
    column before 2005, '#'-prefixed header and units lines from 2007, and
    missing values as MM, 99.0 or 999.0."""
    lines = text.splitlines()
    if not lines:
        return pd.Series(dtype=float, name='wtmp_c', index=pd.DatetimeIndex([], tz='UTC'))
    header = lines[0].lstrip('#').split()
    body = '\n'.join(l for l in lines[1:] if not l.startswith('#'))
    df = pd.read_csv(io.StringIO(body), sep=r'\s+', header=None, names=header, na_values=['MM'])
    year = df[header[0]].astype(int)
    year = year.where(year >= 100, year + 1900)
    minute = df['mm'] if 'mm' in df else 0
    time_utc = pd.to_datetime(dict(year=year, month=df['MM'], day=df['DD'], hour=df['hh'], minute=minute), utc=True)
    wtmp = df['WTMP'].where(df['WTMP'] < 90)
    return pd.Series(wtmp.values, index=time_utc, name='wtmp_c').dropna().sort_index()
