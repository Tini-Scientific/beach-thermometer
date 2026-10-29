"""One-time setup: the station list and each buoy's climatology.

Writes data/stations.csv (names and positions from NDBC's active-station list)
and data/climatology.csv: for every buoy and day of the year, the mean and the
10th and 90th percentiles of daily-mean water temperature over CLIM_YEARS
(1991-2020, the WMO standard base period), pooling all days within
±CLIM_HALF_WINDOW_DAYS (an 11-day window, as in Hobday et al. 2016's marine
heatwave definition). NDBC's yearly archives are cached in data/historical/
(not committed). Re-run only to add stations or change the method.
"""
import gzip
import os
import re
import time

import numpy as np
import pandas as pd

from ndbc import NDBC_URL, http, read_stdmet

# California NDBC buoys: station ID -> short name for the popups
STATIONS = {'46027': 'St Georges', '46022': 'Eel River', '46014': 'Point Arena', '46013': 'Bodega Bay',
            '46026': 'San Francisco', '46012': 'Half Moon Bay', '46042': 'Monterey Bay',
            '46028': 'Cape San Martin', '46011': 'Santa Maria', '46054': 'West Santa Barbara',
            '46053': 'East Santa Barbara', '46025': 'Santa Monica Basin', '46069': 'South Santa Rosa Island'}
CLIM_YEARS = range(1991, 2021)
CLIM_HALF_WINDOW_DAYS = 5
MIN_HOURS_PER_DAY = 12
MIN_DAYS_PER_DOY = 50  # a day of year needs this many daily means in its window

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
HIST = os.path.join(DATA, 'historical')


def station_table():
    xml = http.get(f'{NDBC_URL}/activestations.xml', timeout=60).text
    rows = []
    for sid, name in STATIONS.items():
        tag = re.search(rf'<station id="{sid}"[^>]*>', xml, flags=re.I)
        attr = dict(re.findall(r'(\w+)="([^"]*)"', tag.group(0))) if tag else {}
        rows.append({'station': sid, 'name': name, 'lat': float(attr.get('lat', 'nan')),
                     'lon': float(attr.get('lon', 'nan')), 'ndbc_name': attr.get('name', '')})
    return pd.DataFrame(rows)


def daily_history(sid):
    parts = []
    for year in CLIM_YEARS:
        path = os.path.join(HIST, f'{sid}h{year}.txt.gz')
        if not os.path.exists(path):
            r = http.get(f'{NDBC_URL}/data/historical/stdmet/{sid}h{year}.txt.gz', timeout=60)
            if r.status_code != 200:
                continue
            with open(path, 'wb') as f:
                f.write(r.content)
            time.sleep(0.2)  # be gentle with NDBC
        with gzip.open(path, 'rt') as f:
            parts.append(read_stdmet(f.read()))
    w = pd.concat(parts)
    w = w[~w.index.duplicated()].resample('h').mean()
    n = w.resample('D').count()
    return w.resample('D').mean()[n >= MIN_HOURS_PER_DAY].dropna()


def climatology(sid, daily):
    doy = daily.index.dayofyear.values
    rows = []
    for d in range(1, 367):
        near = np.abs((doy - d + 183) % 366 - 183) <= CLIM_HALF_WINDOW_DAYS
        if near.sum() >= MIN_DAYS_PER_DOY:
            v = daily.values[near]
            rows.append({'station': sid, 'doy': d, 'mean_c': v.mean(), 'p10_c': np.percentile(v, 10),
                         'p90_c': np.percentile(v, 90), 'n_days': int(near.sum()),
                         'n_years': daily.index[near].year.nunique()})
    return rows


if __name__ == '__main__':
    os.makedirs(HIST, exist_ok=True)
    stations = station_table()
    stations.to_csv(os.path.join(DATA, 'stations.csv'), index=False)
    rows = []
    for sid in STATIONS:
        rows += climatology(sid, daily_history(sid))
        print(f'{sid}: done')
    clim = pd.DataFrame(rows)
    clim.round(3).to_csv(os.path.join(DATA, 'climatology.csv'), index=False)
    print(f'Wrote data/stations.csv ({len(stations)} stations) and data/climatology.csv ({len(clim)} rows)')
