"""Hourly update for beachthermometer.com (run by .github/workflows/update.yml).

1. Fetch each buoy's NDBC realtime2 file (the past 45 days, updated about
   hourly) and merge its water temperatures, as hourly means, into the archive:
   data/archive/{station}/{YYYY-MM}.csv. Only the current month's files change
   from hour to hour, so the committed history stays small.
2. Write the site data (not committed; published straight to GitHub Pages):
   - site/data/latest.json: per buoy, the latest water temperature and time,
     the anomaly of the past 24 h from the climatology, and a status.
   - site/data/stations/{id}.json: the past RECENT_DAYS as hourly means plus
     the climatology (mean, 10th and 90th percentile by day of year).
   - site/data/stations/{id}.csv: the same hourly data for download.
   Status follows Hobday et al. (2016): a suspected marine heatwave when the
   daily mean has been above the climatological 90th percentile for at least
   MHW_MIN_DAYS days in a row up to today (a suspected cold spell for below
   the 10th), otherwise warmer / cooler than usual when the 24 h anomaly is
   beyond ±NEAR_NORMAL_C, else near normal. Buoys whose record is too short
   for a climatology (see build_climatology.py) get status 'noclim'.
3. Exit with an error (so GitHub emails the repository owner) if no buoy has
   reported within STALE_HOURS.
"""
import json
import os
import sys

import pandas as pd

from ndbc import NDBC_URL, http, read_stdmet

RECENT_DAYS = 45
MHW_MIN_DAYS = 5
NEAR_NORMAL_C = 0.25
STALE_HOURS = 6

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
ARCHIVE = os.path.join(DATA, 'archive')
SITE_DATA = os.path.join(HERE, 'site', 'data')


def clean(values):
    """Rounded floats for JSON, with NaN (no climatology that day) as null."""
    return [None if pd.isna(v) else round(float(v), 2) for v in values]


def streak(flags):
    """Number of True values at the end of a boolean sequence."""
    n = 0
    for f in flags[::-1]:
        if not f:
            break
        n += 1
    return n


def merge_into_archive(sid, hourly):
    """Merge hourly means into the station's monthly archive files."""
    folder = os.path.join(ARCHIVE, sid)
    os.makedirs(folder, exist_ok=True)
    for month, part in hourly.groupby(hourly.index.strftime('%Y-%m')):
        path = os.path.join(folder, f'{month}.csv')
        if os.path.exists(path):
            old = pd.read_csv(path, index_col=0)['wtmp_c']
            old.index = pd.to_datetime(old.index, utc=True)
            part = pd.concat([old, part])
            part = part[~part.index.duplicated(keep='last')].sort_index()
        part.round(3).rename_axis('time_utc').to_frame('wtmp_c').to_csv(path)


def read_archive(sid, since):
    folder = os.path.join(ARCHIVE, sid)
    parts = []
    for name in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
        if name.endswith('.csv') and name[:7] >= since.strftime('%Y-%m'):
            s = pd.read_csv(os.path.join(folder, name), index_col=0)['wtmp_c']
            s.index = pd.to_datetime(s.index, utc=True)
            parts.append(s)
    if not parts:
        return pd.Series(dtype=float, name='wtmp_c', index=pd.DatetimeIndex([], tz='UTC'))
    s = pd.concat(parts).sort_index()
    return s[s.index >= since]


def main():
    now = pd.Timestamp.now(tz='UTC')
    since = now - pd.Timedelta(days=RECENT_DAYS)
    stations = pd.read_csv(os.path.join(DATA, 'stations.csv'), dtype={'station': str, 'cdip': str}).set_index('station')
    stations[['name', 'ndbc_name', 'cdip']] = stations[['name', 'ndbc_name', 'cdip']].fillna('')  # blanks, not NaN
    clim = pd.read_csv(os.path.join(DATA, 'climatology.csv'), dtype={'station': str})
    os.makedirs(os.path.join(SITE_DATA, 'stations'), exist_ok=True)

    entries = []
    for sid, info in stations.iterrows():
        r = http.get(f'{NDBC_URL}/data/realtime2/{sid}.txt', timeout=60)
        raw = read_stdmet(r.text) if r.status_code == 200 else read_stdmet('')
        if len(raw):
            merge_into_archive(sid, raw.resample('h').mean().dropna())
        hourly = read_archive(sid, since)
        cs = clim[clim['station'] == sid].set_index('doy').reindex(range(1, 367))
        has_clim = pd.notna(info['clim_start'])
        entry = {'id': sid, 'name': info['name'], 'type': info['type'], 'ndbc_name': info['ndbc_name'],
                 'lat': round(float(info['lat']), 4), 'lon': round(float(info['lon']), 4),
                 'clim_years': [int(info['clim_start']), int(info['clim_end'])] if has_clim else None}
        if info['type'] == 'cdip':
            entry['cdip'] = info['cdip']
        recent_raw = raw[raw.index >= since]
        if len(recent_raw) and not has_clim:
            entry.update({'last_time_ms': int(recent_raw.index[-1].timestamp() * 1000),
                          'last_c': round(float(recent_raw.iloc[-1]), 2), 'status': 'noclim'})
        elif len(recent_raw):
            t_last = recent_raw.index[-1]
            last24 = recent_raw[recent_raw.index > t_last - pd.Timedelta(hours=24)].mean()
            anom = last24 - cs.loc[t_last.dayofyear, 'mean_c']
            daily = hourly.resample('D').mean().dropna()
            c_day = cs.loc[daily.index.dayofyear]
            hot = streak(list(daily.values > c_day['p90_c'].values))
            cold = streak(list(daily.values < c_day['p10_c'].values))
            if hot >= MHW_MIN_DAYS:
                status = 'heatwave'
            elif cold >= MHW_MIN_DAYS:
                status = 'coldspell'
            elif anom > NEAR_NORMAL_C:
                status = 'warmer'
            elif anom < -NEAR_NORMAL_C:
                status = 'cooler'
            else:
                status = 'normal'
            entry.update({'last_time_ms': int(t_last.timestamp() * 1000), 'last_c': round(float(recent_raw.iloc[-1]), 2),
                          'anom_c': round(float(anom), 2), 'status': status, 'hot_days': hot, 'cold_days': cold,
                          # a streak as long as the record may have started earlier
                          'streak_is_whole_record': bool(max(hot, cold) == len(daily))})
        entries.append(entry)
        with open(os.path.join(SITE_DATA, 'stations', f'{sid}.json'), 'w') as f:
            json.dump({'id': sid, 't': [int(t.timestamp() * 1000) for t in hourly.index],
                       'c': [round(float(v), 2) for v in hourly.values],
                       'clim_mean': clean(cs['mean_c']), 'clim_p10': clean(cs['p10_c']),
                       'clim_p90': clean(cs['p90_c'])}, f, separators=(',', ':'), allow_nan=False)
        hourly.rename('water_temp_c').round(2).rename_axis('time_utc').to_frame().to_csv(
            os.path.join(SITE_DATA, 'stations', f'{sid}.csv'))

    with open(os.path.join(SITE_DATA, 'latest.json'), 'w') as f:
        json.dump({'generated_ms': int(now.timestamp() * 1000), 'recent_days': RECENT_DAYS,
                   'stations': entries}, f, indent=1, allow_nan=False)  # NaN isn't valid JSON for browsers
    counts = {s: sum(e.get('status') == s for e in entries)
              for s in ['heatwave', 'warmer', 'normal', 'cooler', 'coldspell', 'noclim']}
    print(f"{now:%Y-%m-%d %H:%M} UTC: {sum('status' in e for e in entries)}/{len(entries)} buoys reporting; {counts}")

    newest_h = min(((now.timestamp() * 1000 - e['last_time_ms']) / 3.6e6 for e in entries if 'last_time_ms' in e),
                   default=float('inf'))
    if newest_h > STALE_HOURS:
        sys.exit(f'No buoy has reported in the past {STALE_HOURS} h (newest: {newest_h:.1f} h ago)')


if __name__ == '__main__':
    main()
