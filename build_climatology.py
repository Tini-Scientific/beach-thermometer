"""One-time setup: the station list and each buoy's climatology.

Writes data/stations.csv (names, types and positions; positions from NDBC's
active-station list) and data/climatology.csv: for every buoy and day of the
year, the mean and the 10th and 90th percentiles of daily-mean water
temperature, pooling all days within ±CLIM_HALF_WINDOW_DAYS (an 11-day window,
as in Hobday et al. 2016's marine heatwave definition), then smoothed with a
SMOOTH_DAYS (31-day) running mean that wraps around the year end, also as in
Hobday et al., so short records don't give a jumpy climatology.

Stations: California's NDBC offshore buoys and CDIP wave buoys that reported
water temperature in the past year (the same buoys as BoB_261028's
CA_WaterTemp_SideBySide.png).

Climatology period, per buoy: the WMO standard BASE_YEARS (1991-2020) when the
buoy has at least MIN_BASE_YEARS years of data in it; otherwise every year
available through LAST_YEAR, when that's at least MIN_YEARS years; otherwise
none (the site then shows the temperature without a climatology). NDBC's
archive of CDIP buoys starts in 2004, so the older CDIP buoys use 2004-2020
and the newer ones their whole record.

Water temperature comes from NDBC's yearly standard-meteorological archives
(CDIP buoys are archived there under their NDBC numbers), cached in
data/historical/ (not committed). Re-run only to add stations or change the
method.
"""
import gzip
import os
import re
import time

import numpy as np
import pandas as pd

from ndbc import NDBC_URL, http, read_stdmet

# NDBC offshore buoys: NDBC ID -> short name
NDBC_STATIONS = {'46027': 'St Georges', '46022': 'Eel River', '46014': 'Point Arena', '46013': 'Bodega Bay',
                 '46026': 'San Francisco', '46012': 'Half Moon Bay', '46042': 'Monterey Bay',
                 '46028': 'Cape San Martin', '46011': 'Santa Maria', '46054': 'West Santa Barbara',
                 '46053': 'East Santa Barbara', '46025': 'Santa Monica Basin', '46069': 'South Santa Rosa Island'}
# CDIP wave buoys: NDBC ID -> (CDIP station number, short name)
CDIP_STATIONS = {'46244': ('168', 'Humboldt Bay North Spit'), '46213': ('094', 'Cape Mendocino'),
                 '46214': ('029', 'Point Reyes'), '46237': ('142', 'San Francisco Bar'),
                 '46240': ('158', 'Cabrillo Point'), '46239': ('157', 'Point Sur'),
                 '46215': ('076', 'Diablo Canyon'), '46259': ('222', 'Santa Lucia Escarpment'),
                 '46218': ('071', 'Harvest'), '46268': ('103', 'Topanga'), '46221': ('028', 'Santa Monica Bay'),
                 '46251': ('203', 'Santa Cruz Basin'), '46256': ('215', 'Long Beach Channel'),
                 '46222': ('092', 'San Pedro'), '46253': ('213', 'San Pedro South'),
                 '46277': ('271', 'Green Beach'), '46275': ('264', 'Red Beach'),
                 '46219': ('067', 'San Nicolas Island'), '46224': ('045', 'Oceanside Offshore'),
                 '46274': ('262', 'Leucadia'), '46266': ('153', 'Del Mar'), '46225': ('100', 'Torrey Pines Outer'),
                 '46254': ('201', 'Scripps Nearshore'), '46258': ('220', 'Mission Bay West'),
                 '46235': ('155', 'Imperial Beach'), '46232': ('191', 'Point Loma South')}

# positions for buoys missing from NDBC's active-station list (last known, from the IOOS catalog)
FALLBACK_POSITIONS = {'46240': (36.626, -121.907)}

BASE_YEARS = (1991, 2020)
MIN_BASE_YEARS = 15
LAST_YEAR = pd.Timestamp.now().year - 1  # last complete year in NDBC's yearly archive
MIN_YEARS = 8
MIN_DAYS_PER_YEAR = 60  # a year counts toward the record with this many daily means
CLIM_HALF_WINDOW_DAYS = 5
MIN_HOURS_PER_DAY = 12
MIN_DAYS_PER_DOY = 40  # a day of year needs this many daily means in its window
SMOOTH_DAYS = 31

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
HIST = os.path.join(DATA, 'historical')


def station_table():
    xml = http.get(f'{NDBC_URL}/activestations.xml', timeout=60).text
    defs = [(sid, name, 'ndbc', '') for sid, name in NDBC_STATIONS.items()]
    defs += [(sid, name, 'cdip', cdip) for sid, (cdip, name) in CDIP_STATIONS.items()]
    rows = []
    for sid, name, kind, cdip in defs:
        tag = re.search(rf'<station id="{sid}"[^>]*>', xml, flags=re.I)
        attr = dict(re.findall(r'(\w+)="([^"]*)"', tag.group(0))) if tag else {}
        lat, lon = FALLBACK_POSITIONS.get(sid, (float(attr.get('lat', 'nan')), float(attr.get('lon', 'nan'))))
        rows.append({'station': sid, 'name': name, 'type': kind, 'cdip': cdip, 'lat': lat, 'lon': lon,
                     'ndbc_name': attr.get('name', '')})
    return pd.DataFrame(rows)


def daily_history(sid, years):
    parts = []
    for year in years:
        path = os.path.join(HIST, f'{sid}h{year}.txt.gz')
        missing = path + '.missing'  # remembers years NDBC doesn't have
        if os.path.exists(missing):
            continue
        if not os.path.exists(path):
            r = http.get(f'{NDBC_URL}/data/historical/stdmet/{sid}h{year}.txt.gz', timeout=60)
            if r.status_code != 200:
                open(missing, 'w').close()
                continue
            with open(path, 'wb') as f:
                f.write(r.content)
            time.sleep(0.2)  # be gentle with NDBC
        with gzip.open(path, 'rt') as f:
            parts.append(read_stdmet(f.read()))
    if not parts:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz='UTC'))
    w = pd.concat(parts)
    w = w[~w.index.duplicated()].resample('h').mean()
    n = w.resample('D').count()
    return w.resample('D').mean()[n >= MIN_HOURS_PER_DAY].dropna()


def years_with_data(daily):
    counts = daily.groupby(daily.index.year).size()
    return sorted(counts[counts >= MIN_DAYS_PER_YEAR].index)


def choose_period(sid):
    """Daily means over the climatology period chosen for this buoy, and the period."""
    daily = daily_history(sid, range(BASE_YEARS[0], BASE_YEARS[1] + 1))
    years = years_with_data(daily)
    if len(years) >= MIN_BASE_YEARS:
        return daily, (years[0], years[-1])  # the years actually covered, e.g. 2004-2020 for older CDIP buoys
    daily = daily_history(sid, range(BASE_YEARS[0], LAST_YEAR + 1))
    years = years_with_data(daily)
    if len(years) >= MIN_YEARS:
        return daily[daily.index.year.isin(years)], (years[0], years[-1])
    return None, None


def smooth(values):
    """SMOOTH_DAYS running mean over days 1-366, wrapping around the year end;
    days with no climatology are skipped (a window needs half its days)."""
    half = SMOOTH_DAYS // 2
    padded = pd.Series(np.concatenate([values[-half:], values, values[:half]]))
    return padded.rolling(SMOOTH_DAYS, center=True, min_periods=half + 1).mean().values[half:-half]


def climatology(sid, daily):
    doy = daily.index.dayofyear.values
    rows = []
    for d in range(1, 367):
        near = np.abs((doy - d + 183) % 366 - 183) <= CLIM_HALF_WINDOW_DAYS
        v = daily.values[near]
        ok = near.sum() >= MIN_DAYS_PER_DOY
        rows.append({'station': sid, 'doy': d, 'mean_c': v.mean() if ok else np.nan,
                     'p10_c': np.percentile(v, 10) if ok else np.nan, 'p90_c': np.percentile(v, 90) if ok else np.nan,
                     'n_days': int(near.sum()), 'n_years': daily.index[near].year.nunique()})
    clim = pd.DataFrame(rows)
    for col in ['mean_c', 'p10_c', 'p90_c']:
        clim[col] = smooth(clim[col].values)
    return clim.dropna(subset=['mean_c']).to_dict('records')


if __name__ == '__main__':
    os.makedirs(HIST, exist_ok=True)
    stations = station_table()
    rows, periods = [], {}
    for sid in stations['station']:
        daily, period = choose_period(sid)
        periods[sid] = period
        if daily is not None:
            rows += climatology(sid, daily)
        print(f'{sid}: {f"{period[0]}-{period[1]}" if period else "no climatology (record too short)"}')
    stations['clim_start'] = [periods[s][0] if periods[s] else None for s in stations['station']]
    stations['clim_end'] = [periods[s][1] if periods[s] else None for s in stations['station']]
    stations.to_csv(os.path.join(DATA, 'stations.csv'), index=False)
    clim = pd.DataFrame(rows)
    clim.round(3).to_csv(os.path.join(DATA, 'climatology.csv'), index=False)
    print(f'Wrote data/stations.csv ({len(stations)} stations, {sum(p is not None for p in periods.values())} '
          f'with a climatology) and data/climatology.csv ({len(clim)} rows)')
