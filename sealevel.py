"""Sea level at the NOAA tide stations, for update.py (or run on its own: python sealevel.py).

For each tide station in data/stations.csv (its CO-OPS id), from the CO-OPS API:
- the measured water level and the predicted tide (6-minute data, metres above MLLW),
  thinned to every STEP_MIN minutes: the past RECENT_DAYS and, for the tide, FORECAST_HOURS ahead.
  Measured readings are preliminary (not yet reviewed by NOAA); those its automatic checks flag for
  a sudden jump or a value past the sensor's limits are dropped;
- the excess sea level: measured minus predicted, averaged over the past hour (so a seiche or a
  passing wake doesn't swing it);
- the National Weather Service's minor / moderate / major coastal flood levels (the ones its flood
  advisories use), converted to metres above MLLW; many stations have only some of them, or none.
  Flood status is the highest level the water held for at least two readings in a row (12 minutes)
  in the past hour, so a single odd reading can't raise it. A station whose latest reading is more
  than STALE_HOURS old gets no excess or flood status, rather than keeping an old one.

Writes (not committed; published to GitHub Pages):
- site/data/sealevel.json: per station, the latest water level, excess and flood status.
- site/data/sealevel/{id}.json: the measured and predicted series for the station's chart.
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

HERE = os.path.dirname(os.path.abspath(__file__))
SITE_DATA = os.path.join(HERE, 'site', 'data')


def series(params):
    """A CO-OPS product as a Series of metres above MLLW, indexed by UTC time; empty when unavailable."""
    r = http.get(API, params={'datum': 'MLLW', 'units': 'metric', 'time_zone': 'gmt', 'format': 'json',
                              'application': 'beach-thermometer', **params}, timeout=60)
    body = r.json() if r.status_code == 200 else {}
    rows = body.get('data') or body.get('predictions') or []  # an error comes back as {'error': ...}
    # water level flags 'O,F,R,L': R (rate of change) or L (beyond the sensor's limits) mark a suspect reading
    ok = lambda x: x.get('v') not in (None, '') and x.get('f', '0,0,0,0').split(',')[2:] == ['0', '0']
    s = pd.Series({pd.Timestamp(x['t'], tz='UTC'): float(x['v']) for x in rows if ok(x)}, dtype=float)
    return s.sort_index()


def sustained_high(s):
    """The highest level held by two consecutive 6-minute readings (the lower of each pair); NaN if no pair."""
    pairs = pd.concat([s, s.shift(1)], axis=1).min(axis=1)
    consecutive = s.index.to_series().diff() <= pd.Timedelta(minutes=6)
    return pairs[consecutive].max()


def flood_levels(coops):
    """The station's NWS flood levels in metres above MLLW (those it has); {} if none or unavailable."""
    try:
        fl = http.get(f'{MDAPI}/{coops}/floodlevels.json', params={'units': 'metric'}, timeout=30).json()
        datums = http.get(f'{MDAPI}/{coops}/datums.json', params={'units': 'metric'}, timeout=30).json()['datums']
        mllw = next(d['value'] for d in datums if d['name'] == 'MLLW')
    except Exception:
        return {}
    return {k: round(fl[f'nws_{k}'] - mllw, 3) for k in LEVELS if fl.get(f'nws_{k}') is not None}


def thin(s):
    return s[(s.index.minute % STEP_MIN == 0)]


def update_sealevel(now):
    stations = pd.read_csv(os.path.join(HERE, 'data', 'stations.csv'), dtype=str)
    os.makedirs(os.path.join(SITE_DATA, 'sealevel'), exist_ok=True)
    fmt = '%Y%m%d %H:%M'
    begin = now - pd.Timedelta(days=RECENT_DAYS)
    entries = []
    for _, info in stations[stations['type'] == 'tide'].iterrows():
        sid, coops = info['station'], info['coops']
        obs = series({'product': 'water_level', 'station': coops, 'begin_date': f'{begin:{fmt}}', 'end_date': f'{now:{fmt}}'})
        pred = series({'product': 'predictions', 'interval': '6', 'station': coops, 'begin_date': f'{begin:{fmt}}',
                       'end_date': f'{now + pd.Timedelta(hours=FORECAST_HOURS):{fmt}}'})
        levels = flood_levels(coops)
        entry = {'id': sid, 'coops': coops, 'flood_levels_m': levels}
        if len(obs):
            entry['time_ms'] = int(obs.index[-1].timestamp() * 1000)
        if len(obs) and len(pred) and now - obs.index[-1] <= pd.Timedelta(hours=STALE_HOURS):
            last_hour = obs[obs.index > obs.index[-1] - pd.Timedelta(hours=1)]
            excess = (last_hour - pred.reindex(last_hour.index)).dropna()
            high = sustained_high(last_hour)
            entry.update({'level_m': round(float(obs.iloc[-1]), 3),
                          'excess_m': round(float(excess.mean()), 3) if len(excess) else None,
                          'flood': next((k for k in LEVELS[::-1] if k in levels and high >= levels[k]), None)})
        entries.append(entry)
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
