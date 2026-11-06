"""One-time backfill of data/daily/{station}.csv: each station's daily-mean water temperature over the
past year, for the Year view of its chart. update.py keeps these files current from the hourly archive
(data/archive/), so this only needs re-running to add a station; it fills days missing from a file and
never replaces days already there.

Days are UTC, and a day counts with at least MIN_HOURS_PER_DAY hourly means (as for the climatology).
Sources, by station type:
- NDBC and CDIP buoys: NDBC's yearly standard-meteorological file for last year, then this year's
  monthly files (data/stdmet/{Mon}/: finished months as {id}{month}{year}.txt.gz, the newest as {id}.txt);
- tide stations: CO-OPS hourly water temperature, without flagged readings (sealevel.water_temperature);
- moorings and shore stations: the daily means build_climatology.py caches from ERDDAP (data/historical/);
and for every station, the days the hourly archive already covers.
Run: python backfill_daily.py
"""
import gzip
import os
import re
import time

import pandas as pd

import build_climatology as bc
from ndbc import NDBC_URL, http, read_stdmet
from sealevel import water_temperature
from update import DAILY, MIN_HOURS_PER_DAY, archive_daily, read_daily, write_daily

YEAR_DAYS = 366
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def to_daily(hourly):
    """Daily means of hourly readings, for days with at least MIN_HOURS_PER_DAY of them."""
    if hourly.empty:
        return hourly
    h = hourly.resample('h').mean()
    n = h.resample('D').count()
    return h.resample('D').mean()[n >= MIN_HOURS_PER_DAY].dropna()


def ndbc_months(year, listings):
    """This year's monthly NDBC files: {station id (lower case): [urls]} from each month's folder listing."""
    files = {}
    for mon in MONTHS:
        if mon not in listings:
            r = http.get(f'{NDBC_URL}/data/stdmet/{mon}/', timeout=60)
            listings[mon] = r.text if r.status_code == 200 else ''
        for name in re.findall(r'href="([0-9a-z]+(?:\d{1,2}' + str(year) + r')?\.txt(?:\.gz)?)"', listings[mon]):
            sid = re.match(r'^([0-9a-z]{5})', name).group(1)
            files.setdefault(sid, []).append(f'{NDBC_URL}/data/stdmet/{mon}/{name}')
    return files


def ndbc_daily(sid, start, month_files):
    parts = [bc.daily_history(sid, range(start.year, pd.Timestamp.now().year))]  # yearly files (cached)
    for url in month_files.get(sid.lower(), []):
        r = http.get(url, timeout=60)
        if r.status_code != 200:
            continue
        text = gzip.decompress(r.content).decode() if url.endswith('.gz') else r.text
        parts.append(to_daily(read_stdmet(text)))
        time.sleep(0.2)  # be gentle with NDBC
    parts = [p for p in parts if len(p)]
    return pd.concat(parts) if parts else pd.Series(dtype=float)


def tide_daily(coops, start, now):
    mid = start + (now - start) / 2  # CO-OPS serves at most a year of hourly readings per request
    return to_daily(pd.concat([water_temperature(coops, start, mid, 'h'), water_temperature(coops, mid, now, 'h')]))


if __name__ == '__main__':
    os.makedirs(DAILY, exist_ok=True)
    os.makedirs(bc.HIST, exist_ok=True)
    now = pd.Timestamp.now(tz='UTC').floor('h')
    start = (now - pd.Timedelta(days=YEAR_DAYS)).floor('D')
    stations = pd.read_csv(os.path.join(bc.DATA, 'stations.csv'), dtype={'station': str, 'coops': str})
    month_files = ndbc_months(now.year, {})
    for _, info in stations.iterrows():
        sid, kind = info['station'], info['type']
        if kind == 'tide':
            daily = tide_daily(info['coops'], start.tz_localize(None), now.tz_localize(None))
        elif kind in ('ioos', 'shore'):
            daily, _ = bc.ioos_history(sid)
        else:
            daily = ndbc_daily(sid, start, month_files)
        if len(daily):
            daily.index = pd.to_datetime(daily.index, utc=True).floor('D')
            daily = daily[~daily.index.duplicated(keep='last')]
        # priority: the backfill < what the file already holds < days the hourly archive covers
        merged = pd.concat([daily, read_daily(sid), archive_daily(sid, now)])
        merged = merged[~merged.index.duplicated(keep='last')].sort_index()
        merged = merged[(merged.index >= start) & (merged.index < now.floor('D'))]
        write_daily(sid, merged)
        print(f'{sid}: {len(merged)} days' + (f' ({merged.index[0]:%Y-%m-%d} to {merged.index[-1]:%Y-%m-%d})' if len(merged) else ''))
