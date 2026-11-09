"""Hourly update for beachthermometer.com (run by .github/workflows/update.yml).

1. Fetch each buoy's NDBC realtime2 file (the past 45 days, updated about hourly),
   each tide station's readings from the CO-OPS API (the past 45 days, without the
   ones CO-OPS's automatic checks flag; the archive's copy of that window is replaced,
   so readings flagged later drop out), or each mooring's or shore station's surface sensor from
   the IOOS ERDDAP server (ioos.py), and merge its water temperatures, as hourly means, into the archive:
   data/archive/{station}/{YYYY-MM}.csv. Only the current month's files change
   from hour to hour, so the committed history stays small.
2. Write the site data (not committed; published straight to GitHub Pages):
   - site/data/latest.json: per buoy, the latest water temperature and time,
     the anomaly of the past 24 h from the climatology, and a status.
   - site/data/stations/{id}.json: the past RECENT_DAYS as hourly means plus
     the climatology (mean, 10th and 90th percentile by day of year).
   - site/data/stations/{id}.csv: the same hourly data for download.
   The station's daily means (data/daily/{id}.csv, committed: one small file a station, kept for good;
   backfill_daily.py filled the year before the archive began) get the complete days of the hourly window,
   and the past YEAR_DAYS of them go in its JSON for the chart's Year view.
   Status follows Hobday et al. (2016): a suspected marine heatwave when the
   daily mean has been above the climatological 90th percentile for at least
   MHW_MIN_DAYS days in a row up to today (a suspected cold spell for below
   the 10th), otherwise warmer / cooler than usual when the 24 h anomaly is
   beyond ±NEAR_NORMAL_C, else near normal. Buoys whose record is too short
   for a climatology (see build_climatology.py) get status 'noclim', and
   buoys whose latest reading is more than STALE_DAYS old get 'stale', and stations
   listed in SUSPECT (readings in question) get 'suspect'.
3. Once a day, add an entry to the daily marine heatwave summary feed
   (site/feed.xml; see feed.py).
4. Fetch sea level at the tide stations (site/data/sealevel*; see sealevel.py).
5. Exit with an error (so GitHub emails the repository owner) if no buoy has
   reported within STALE_HOURS.
"""
import json
import math
import os
import sys

import pandas as pd

import ioos
from feed import update_feed
from sealevel import update_sealevel, water_temperature
from ndbc import NDBC_URL, http, read_stdmet

RECENT_DAYS = 45
MHW_MIN_DAYS = 5
NEAR_NORMAL_C = 0.25
STALE_HOURS = 6
STALE_DAYS = 14
YEAR_DAYS = 365         # the cards' Year view: daily means (data/daily/)
MIN_HOURS_PER_DAY = 12  # a day's mean needs this many hourly means (as for the climatology)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
ARCHIVE = os.path.join(DATA, 'archive')
DAILY = os.path.join(DATA, 'daily')  # {station}.csv: daily means for the Year view (backfill_daily.py started them)
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


# stations whose recent readings look wrong (a lasting jump that nearby stations don't show): from the
# date given (UTC) they get status 'suspect', with the note on their card, instead of a heatwave or
# cold-spell status. Remove an entry once the data are corrected or the readings look right again.
SUSPECT = {
    # 2026-06-07: rose ~2.5 °C over one afternoon after a day without temperature readings and has stayed
    # ~2.5 °C above every year since 2005, with unchanged daily swings (so not a shallower sensor);
    # Port Angeles, New Dungeness and Point Wells show no such jump
    'PTWW1': ('2026-06-07', 'Readings jumped about 4.5 °F (2.5 °C) on June 7, 2026, and have stayed that much '
                            'higher, while nearby stations show no such jump. Until this is resolved, no heatwave '
                            'or cold-spell status is given here.'),
}


def merge_into_archive(sid, hourly, replace=False):
    """Merge hourly means into the station's monthly archive files. With `replace`, the archived hours
    between the first and last of `hourly` are replaced rather than merged, so readings since dropped
    (e.g. ones CO-OPS has flagged since) leave the archive too."""
    folder = os.path.join(ARCHIVE, sid)
    os.makedirs(folder, exist_ok=True)
    for month, part in hourly.groupby(hourly.index.strftime('%Y-%m')):
        path = os.path.join(folder, f'{month}.csv')
        if os.path.exists(path):
            old = pd.read_csv(path, index_col=0)['wtmp_c']
            old.index = pd.to_datetime(old.index, utc=True)
            if replace:
                old = old[(old.index < hourly.index[0]) | (old.index > hourly.index[-1])]
            part = pd.concat([old, part])
            part = part[~part.index.duplicated(keep='last')].sort_index()
        part.round(3).rename_axis('time_utc').to_frame('wtmp_c').to_csv(path)


def daily_means(hourly):
    """Daily (UTC) means of hourly means, for days with at least half the station's usual readings: MIN_HOURS_PER_DAY
    hourly means for a station that reports hourly, fewer for one that reports less often (the Hood Canal
    profiling buoys report every 6 hours, so 2 of their 4 a day)."""
    if hourly.empty:
        return hourly
    step_h = max(1.0, hourly.index.to_series().diff().median() / pd.Timedelta(hours=1)) if len(hourly) > 1 else 1.0
    need = min(MIN_HOURS_PER_DAY, max(2, math.ceil(MIN_HOURS_PER_DAY / step_h)))
    n = hourly.resample('D').count()
    return hourly.resample('D').mean()[n >= need].dropna()


def archive_daily(sid, now):
    """Daily means of the station's whole hourly archive, complete days only (before today)."""
    d = daily_means(read_archive(sid, pd.Timestamp('2000-01-01', tz='UTC')))
    return d[d.index < now.floor('D')]


def read_daily(sid):
    path = os.path.join(DAILY, f'{sid}.csv')
    if not os.path.exists(path):
        return pd.Series(dtype=float, name='wtmp_c', index=pd.DatetimeIndex([], tz='UTC'))
    s = pd.read_csv(path, index_col=0)['wtmp_c'].astype(float)  # float even when the file has no days yet
    s.index = pd.to_datetime(s.index, utc=True)
    return s


def write_daily(sid, daily):
    os.makedirs(DAILY, exist_ok=True)
    out = daily.astype(float).round(2).rename('wtmp_c')
    out.index = out.index.strftime('%Y-%m-%d')
    out.rename_axis('date_utc').to_frame().to_csv(os.path.join(DAILY, f'{sid}.csv'))


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


def coops_temperature(coops, now, days):
    """A tide station's water temperature (°C) for the past `days` from CO-OPS, 6-minute readings
    (in 31-day requests) without the ones its automatic checks flag; empty when unavailable."""
    parts, end = [], now
    while end > now - pd.Timedelta(days=days):
        begin = max(end - pd.Timedelta(days=30), now - pd.Timedelta(days=days))
        parts.append(water_temperature(coops, begin, end))
        end = begin
    s = pd.concat(parts).sort_index()
    return s[~s.index.duplicated()] if len(s) else read_stdmet('')


def main():
    now = pd.Timestamp.now(tz='UTC')
    since = now - pd.Timedelta(days=RECENT_DAYS)
    stations = pd.read_csv(os.path.join(DATA, 'stations.csv'), dtype={'station': str, 'cdip': str, 'coops': str}).set_index('station')
    text_cols = ['name', 'long_name', 'cdip', 'coops', 'operator', 'erddap_var']
    stations[text_cols] = stations[text_cols].fillna('')  # blanks, not NaN
    clim = pd.read_csv(os.path.join(DATA, 'climatology.csv'), dtype={'station': str})
    os.makedirs(os.path.join(SITE_DATA, 'stations'), exist_ok=True)

    entries = []
    for sid, info in stations.iterrows():
        if info['type'] in ('ioos', 'shore'):
            raw = ioos.recent(sid, RECENT_DAYS, info['erddap_var'])
        elif info['type'] == 'tide':
            raw = coops_temperature(info['coops'], now, RECENT_DAYS)
        else:
            r = http.get(f'{NDBC_URL}/data/realtime2/{sid}.txt', timeout=60)
            raw = read_stdmet(r.text) if r.status_code == 200 else read_stdmet('')
        if len(raw):
            merge_into_archive(sid, raw.resample('h').mean().dropna(), replace=info['type'] == 'tide')
        hourly = read_archive(sid, since)
        # the Year view's daily means: the complete days in the hourly window replace what the file held for them
        recent_days = daily_means(hourly)
        recent_days = recent_days[(recent_days.index >= since.ceil('D')) & (recent_days.index < now.floor('D'))]
        year = pd.concat([read_daily(sid), recent_days])
        year = year[~year.index.duplicated(keep='last')].sort_index()
        write_daily(sid, year)
        year_start = now.floor('D') - pd.Timedelta(days=YEAR_DAYS)
        year = year.reindex(pd.date_range(year_start, now.floor('D') - pd.Timedelta(days=1), freq='D'))
        cs = clim[clim['station'] == sid].set_index('doy').reindex(range(1, 367))
        has_clim = pd.notna(info['clim_start'])
        entry = {'id': sid, 'name': info['name'], 'type': info['type'], 'long_name': info['long_name'],
                 'lat': round(float(info['lat']), 4), 'lon': round(float(info['lon']), 4),
                 'clim_years': [int(info['clim_start']), int(info['clim_end'])] if has_clim else None}
        if info['type'] == 'cdip':
            entry['cdip'] = info['cdip']
        if info['type'] == 'tide':
            entry['coops'] = info['coops']
        if info['type'] == 'ioos':
            entry.update({'operator': info['operator'], 'depth_m': float(info['depth_m'])})
        if info['type'] == 'shore':
            entry['operator'] = info['operator']
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
        if sid in SUSPECT and 'last_time_ms' in entry and entry['last_time_ms'] >= pd.Timestamp(SUSPECT[sid][0], tz='UTC').timestamp() * 1000:
            # readings in question: the temperature is shown, but no status (and no heatwave count)
            entry = {k: v for k, v in entry.items() if k not in ('anom_c', 'hot_days', 'cold_days', 'streak_is_whole_record')}
            entry.update({'status': 'suspect', 'note': SUSPECT[sid][1]})
        if 'last_time_ms' in entry and now.timestamp() * 1000 - entry['last_time_ms'] > STALE_DAYS * 864e5:
            # too old to describe the water now: shown gray, and kept out of the heatwave count
            entry = {k: v for k, v in entry.items() if k not in ('anom_c', 'hot_days', 'cold_days', 'streak_is_whole_record')}
            entry['status'] = 'stale'
        entries.append(entry)
        with open(os.path.join(SITE_DATA, 'stations', f'{sid}.json'), 'w') as f:
            json.dump({'id': sid, 't': [int(t.timestamp() * 1000) for t in hourly.index],
                       'daily_t0': int(year_start.timestamp() * 1000), 'daily': clean(year.values),
                       'c': [round(float(v), 2) for v in hourly.values],
                       'clim_mean': clean(cs['mean_c']), 'clim_p10': clean(cs['p10_c']),
                       'clim_p90': clean(cs['p90_c'])}, f, separators=(',', ':'), allow_nan=False)
        hourly.rename('water_temp_c').round(2).rename_axis('time_utc').to_frame().to_csv(
            os.path.join(SITE_DATA, 'stations', f'{sid}.csv'))

    with open(os.path.join(SITE_DATA, 'latest.json'), 'w') as f:
        json.dump({'generated_ms': int(now.timestamp() * 1000), 'recent_days': RECENT_DAYS, 'stale_days': STALE_DAYS,
                   'stations': entries}, f, indent=1, allow_nan=False)  # NaN isn't valid JSON for browsers
    counts = {s: sum(e.get('status') == s for e in entries)
              for s in ['heatwave', 'warmer', 'normal', 'cooler', 'coldspell', 'noclim', 'suspect', 'stale']}
    print(f"{now:%Y-%m-%d %H:%M} UTC: {sum('status' in e for e in entries)}/{len(entries)} buoys reporting; {counts}")
    update_feed(entries, now)
    try:
        update_sealevel(now)
    except Exception as e:  # CO-OPS trouble shouldn't stop the temperature update; the map keeps working without it
        print(f'Sea level update failed: {e!r}')

    newest_h = min(((now.timestamp() * 1000 - e['last_time_ms']) / 3.6e6 for e in entries if 'last_time_ms' in e),
                   default=float('inf'))
    if newest_h > STALE_HOURS:
        sys.exit(f'No buoy has reported in the past {STALE_HOURS} h (newest: {newest_h:.1f} h ago)')


if __name__ == '__main__':
    main()
