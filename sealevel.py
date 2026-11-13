"""Sea level at the NOAA tide stations, for update.py (or run on its own: python sealevel.py).

For each tide station in data/stations.csv (its CO-OPS id), and the gauges in LEVEL_ONLY, from the CO-OPS API:
- the measured water level and the predicted tide (6-minute data, metres above MLLW),
  thinned to every STEP_MIN minutes: the past RECENT_DAYS and, for the tide, FORECAST_HOURS ahead.
  Recent measured readings are preliminary: NOAA reviews them weeks to months later and may revise
  them (the verified ones are marked; the time of the last verified reading is recorded). Readings its
  automatic checks flag for a sudden jump or a value past the sensor's limits are dropped;
- the excess sea level: measured minus predicted, averaged over the past hour (so a seiche or a
  passing wake doesn't swing it);
- the National Weather Service's minor / moderate / major coastal flood levels (the ones its flood
  advisories use), converted to metres above MLLW; many stations have only some of them, or none.
  Also the station's mean higher high water (MHHW, a typical high tide), above MLLW, for context.
- NOAA's next predicted high tide (its published high/low tide predictions, as in tide tables).
  Flood status is the highest level the water held for at least two readings in a row (12 minutes)
  in the past hour, so a single odd reading can't raise it. A station whose latest reading is more
  than STALE_HOURS old gets no excess or flood status, rather than keeping an old one.

Writes (not committed; published to GitHub Pages):
- site/data/sealevel.json: per station, the latest water level, excess and flood status.
- site/data/sealevel/{id}.json: the measured and predicted series for the station's chart.
- site/data/sealevel/{id}.csv: the same at full 6-minute resolution, for download.
Nothing is archived: CO-OPS keeps the full record.
"""
import json
import os

import pandas as pd

from ndbc import http

API = 'https://api.tidesandcurrents.noaa.gov/api/prod/datagetter'
MDAPI = 'https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations'
RECENT_DAYS = 30        # CO-OPS serves at most 31 days of 6-minute data per request
FORECAST_HOURS = 48
STEP_MIN = 30           # the 6-minute readings fall on :00 and :30, which both series share
STALE_HOURS = 3
LEVELS = ['minor', 'moderate', 'major']
# Tide gauges not in data/stations.csv (no water temperature: none measured, a sensor switched off, or fresh
# river water), shown only in the site's Tides & flooding view. Every active CO-OPS water level station in
# Washington, Oregon and California not already listed (as of November 2026), except the Port of West
# Sacramento (inland, at the end of the ship channel). By CO-OPS id, which is also their id on the site: name,
# NOAA's station name for the card, state (the Columbia River is the border, which the site's latitude rule
# would get wrong), latitude, longitude.
LEVEL_ONLY = {
    '9449424': ('Cherry Point', 'Cherry Point, WA', 'WA', 48.863, -122.759),
    '9449880': ('Friday Harbor', 'Friday Harbor, WA', 'WA', 48.545, -123.013),
    '9447130': ('Seattle', 'Seattle, WA', 'WA', 47.603, -122.339),
    '9445958': ('Bremerton', 'Bremerton, WA', 'WA', 47.562, -122.623),
    '9440581': ('Cape Disappointment', 'Cape Disappointment, WA', 'WA', 46.281, -124.046),
    '9440569': ('Skamokawa', 'Skamokawa, WA', 'WA', 46.27, -123.457),
    '9439040': ('Astoria (Tongue Point)', 'Astoria, OR', 'OR', 46.207, -123.768),
    '9439099': ('Wauna', 'Wauna, OR', 'OR', 46.16, -123.405),
    '9440422': ('Longview', 'Longview, WA', 'WA', 46.106, -122.954),
    '9440357': ('Kalama', 'TEMCO Kalama Terminal, WA', 'WA', 45.987, -122.837),
    '9439201': ('St. Helens', 'St Helens, OR', 'OR', 45.865, -122.797),
    '9440083': ('Vancouver', 'Vancouver, WA', 'WA', 45.631, -122.696),
    '9437540': ('Garibaldi (Tillamook Bay)', 'Garibaldi, OR', 'OR', 45.555, -123.919),
    '9432780': ('Charleston (Coos Bay)', 'Charleston, OR', 'OR', 43.345, -124.322),
    '9411340': ('Santa Barbara', 'Santa Barbara, CA', 'CA', 34.405, -119.692),
    '9410840': ('Santa Monica', 'Santa Monica, CA', 'CA', 34.008, -118.5),
    '9410678': ('Long Beach (Fire Boat Pier)', 'Long Beach Fire Boat Pier, CA', 'CA', 33.748, -118.216),
}

HERE = os.path.dirname(os.path.abspath(__file__))
SITE_DATA = os.path.join(HERE, 'site', 'data')


def timed(values):
    """A Series from {UTC time: value}, with a time index even when empty (CO-OPS sometimes replies with
    no readings), so time comparisons on it still work."""
    return pd.Series(values, dtype=float, index=pd.DatetimeIndex(list(values), tz='UTC') if not values else None)


def series(params, verified=None):
    """A CO-OPS product as a Series of metres above MLLW, indexed by UTC time; empty when unavailable.
    With a list as `verified`, appends the times of verified readings (quality 'v') to it."""
    r = http.get(API, params={'datum': 'MLLW', 'units': 'metric', 'time_zone': 'gmt', 'format': 'json',
                              'application': 'beach-thermometer', **params}, timeout=60)
    body = r.json() if r.status_code == 200 else {}
    rows = body.get('data') or body.get('predictions') or []  # an error comes back as {'error': ...}
    # water level flags 'O,F,R,L': R (rate of change) or L (beyond the sensor's limits) mark a suspect reading
    ok = lambda x: x.get('v') not in (None, '') and x.get('f', '0,0,0,0').split(',')[2:] == ['0', '0']
    s = timed({pd.Timestamp(x['t'], tz='UTC'): float(x['v']) for x in rows if ok(x)})
    if verified is not None:
        verified += [pd.Timestamp(x['t'], tz='UTC') for x in rows if x.get('q') == 'v']
    return s.sort_index()


def water_temperature(coops, begin, end, interval=None):
    """A tide station's water temperature (°C) from CO-OPS, indexed by UTC time; empty when unavailable.
    `begin` and `end` are Timestamps; 6-minute readings (at most 31 days per request), or hourly ones with
    interval='h' (at most a year). Readings CO-OPS's automatic checks flag (flags 'X,N,R': past the
    sensor's maximum or minimum, or a sudden jump) are dropped."""
    fmt = '%Y%m%d %H:%M'
    params = {'product': 'water_temperature', 'station': coops, 'begin_date': f'{begin:{fmt}}',
              'end_date': f'{end:{fmt}}', 'units': 'metric', 'time_zone': 'gmt', 'format': 'json',
              'application': 'beach-thermometer'}
    if interval:
        params['interval'] = interval
    try:
        r = http.get(API, params=params, timeout=120)
        rows = (r.json().get('data') or []) if r.status_code == 200 else []
    except Exception:
        rows = []
    ok = lambda x: x.get('v') not in (None, '') and '1' not in x.get('f', '0,0,0').split(',')
    s = pd.Series({pd.Timestamp(x['t'], tz='UTC'): float(x['v']) for x in rows if ok(x)}, dtype=float, name='wtmp_c')
    if s.empty:
        s.index = pd.DatetimeIndex([], tz='UTC')
    return s.sort_index()


def sustained_high(s):
    """The highest level held by two consecutive 6-minute readings (the lower of each pair); NaN if no pair."""
    pairs = pd.concat([s, s.shift(1)], axis=1).min(axis=1)
    consecutive = s.index.to_series().diff() <= pd.Timedelta(minutes=6)
    return pairs[consecutive].max()


def flood_levels(coops):
    """The station's NWS flood levels in metres above MLLW (those it has; {} if none or unavailable), and its
    MHHW above MLLW (None if unavailable)."""
    try:
        fl = http.get(f'{MDAPI}/{coops}/floodlevels.json', params={'units': 'metric'}, timeout=30).json()
        datums = {d['name']: d['value'] for d in
                  http.get(f'{MDAPI}/{coops}/datums.json', params={'units': 'metric'}, timeout=30).json()['datums']}
        mllw = datums['MLLW']
    except Exception:
        return {}, None
    levels = {k: round(fl[f'nws_{k}'] - mllw, 3) for k in LEVELS if fl.get(f'nws_{k}') is not None}
    return levels, (round(datums['MHHW'] - mllw, 3) if datums.get('MHHW') is not None else None)


def high_tides(coops, now):
    """NOAA's predicted high tides (its high/low predictions, type 'H') over the two days after `now`:
    a Series of metres above MLLW by UTC time; empty when unavailable."""
    fmt = '%Y%m%d %H:%M'
    r = http.get(API, params={'product': 'predictions', 'interval': 'hilo', 'station': coops, 'datum': 'MLLW',
                              'units': 'metric', 'time_zone': 'gmt', 'format': 'json', 'application': 'beach-thermometer',
                              'begin_date': f'{now:{fmt}}',
                              'end_date': f'{now + pd.Timedelta(days=2):{fmt}}'}, timeout=60)
    rows = (r.json().get('predictions') or []) if r.status_code == 200 else []
    return timed({pd.Timestamp(x['t'], tz='UTC'): float(x['v']) for x in rows if x.get('type') == 'H'}).sort_index()


def thin(s):
    return s[(s.index.minute % STEP_MIN == 0)]


def update_sealevel(now):
    stations = pd.read_csv(os.path.join(HERE, 'data', 'stations.csv'), dtype=str)
    os.makedirs(os.path.join(SITE_DATA, 'sealevel'), exist_ok=True)
    fmt = '%Y%m%d %H:%M'
    begin = now - pd.Timedelta(days=RECENT_DAYS)
    entries = []
    tides = [(info['station'], info['coops'], {}) for _, info in stations[stations['type'] == 'tide'].iterrows()]
    # the page has no other record of the water-level-only gauges, so their entries carry their names and positions
    tides += [(coops, coops, {'level_only': True, 'name': name, 'long_name': f'{coops} - {noaa}', 'state': state,
                              'lat': lat, 'lon': lon})
              for coops, (name, noaa, state, lat, lon) in LEVEL_ONLY.items()]
    for sid, coops, about in tides:
        verified = []
        obs = series({'product': 'water_level', 'station': coops, 'begin_date': f'{begin:{fmt}}', 'end_date': f'{now:{fmt}}'},
                     verified)
        pred = series({'product': 'predictions', 'interval': '6', 'station': coops, 'begin_date': f'{begin:{fmt}}',
                       'end_date': f'{now + pd.Timedelta(hours=FORECAST_HOURS):{fmt}}'})
        levels, mhhw = flood_levels(coops)
        entry = {'id': sid, 'coops': coops, **about, 'flood_levels_m': levels, 'mhhw_m': mhhw,
                 # readings after this are preliminary; None when none in the window are verified yet
                 'verified_until_ms': int(max(verified).timestamp() * 1000) if verified else None}
        if len(obs):
            entry['time_ms'] = int(obs.index[-1].timestamp() * 1000)
        if len(obs) and len(pred) and now - obs.index[-1] <= pd.Timedelta(hours=STALE_HOURS):
            last_hour = obs[obs.index > obs.index[-1] - pd.Timedelta(hours=1)]
            excess = (last_hour - pred.reindex(last_hour.index)).dropna()
            high = sustained_high(last_hour)
            hi = high_tides(coops, now)
            ahead = hi[hi.index > now].index
            if len(ahead):
                entry.update({'next_high_m': round(float(hi[ahead[0]]), 3), 'next_high_ms': int(ahead[0].timestamp() * 1000)})
            entry.update({'level_m': round(float(obs.iloc[-1]), 3),
                          'excess_m': round(float(excess.mean()), 3) if len(excess) else None,
                          'flood': next((k for k in LEVELS[::-1] if k in levels and high >= levels[k]), None)})
        entries.append(entry)
        pd.DataFrame({'water_level_m_mllw': obs, 'predicted_tide_m_mllw': pred,
                      'measured_minus_predicted_m': obs - pred}).round(3).rename_axis('time_utc').to_csv(
            os.path.join(SITE_DATA, 'sealevel', f'{sid}.csv'))
        o, p = thin(obs), thin(pred)
        with open(os.path.join(SITE_DATA, 'sealevel', f'{sid}.json'), 'w') as f:
            json.dump({'id': sid, 'obs_t': [int(t.timestamp() * 1000) for t in o.index], 'obs': [round(v, 3) for v in o],
                       'pred_t': [int(t.timestamp() * 1000) for t in p.index], 'pred': [round(v, 3) for v in p]},
                      f, separators=(',', ':'))
    with open(os.path.join(SITE_DATA, 'sealevel.json'), 'w') as f:
        json.dump({'generated_ms': int(now.timestamp() * 1000), 'recent_days': RECENT_DAYS, 'stale_hours': STALE_HOURS,
                   'stations': entries}, f, indent=1)
    flooding = [e['id'] for e in entries if e.get('flood')]
    print(f"Sea level: {sum('level_m' in e for e in entries)}/{len(entries)} tide stations reporting; flooding: {flooding or 'none'}")


if __name__ == '__main__':
    update_sealevel(pd.Timestamp.now(tz='UTC'))
