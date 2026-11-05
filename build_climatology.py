"""One-time setup: the station list and each buoy's climatology.

Writes data/stations.csv (names, types and positions; positions from NDBC's
active-station list) and data/climatology.csv: for every buoy and day of the
year, the mean and the 10th and 90th percentiles of daily-mean water
temperature, pooling all days within ±CLIM_HALF_WINDOW_DAYS (an 11-day window,
as in Hobday et al. 2016's marine heatwave definition), then smoothed with a
SMOOTH_DAYS (31-day) running mean that wraps around the year end, also as in
Hobday et al., so short records don't give a jumpy climatology.

Stations: California's NDBC offshore buoys, CDIP wave buoys, IOOS moorings,
NOAA CO-OPS tide stations and IOOS shore stations that reported water temperature in the past year (the same stations as
BoB_261028's CA_WaterTemp_SideBySide.png). Moorings are included only if they
have a surface sensor (top 3 m, see ioos.py), to match the buoys; that leaves
out CCE Mooring 4 (21 m) and the Orange County mini-mooring (20 m). The
Imperial Beach Central (IB-C) mini-mooring is left out too: its surface sensor
stopped on 2026-09-24 after a week of erratic readings (daily means 22-33 °C).

Climatology period, per buoy: the WMO standard BASE_YEARS (1991-2020) when the
buoy has at least MIN_BASE_YEARS years of data in it; otherwise every year
available through LAST_YEAR, when that's at least MIN_YEARS years; otherwise
none (the site then shows the temperature without a climatology). NDBC's
archive of CDIP buoys starts in 2004, so the older CDIP buoys use 2004-2020
and the newer ones their whole record. Stations in PERIOD_OVERRIDES (tide
stations only), whose record has a lasting shift, use the years given there,
and the BAD_YEARS of a station (clearly wrong readings) are left out.

Buoy water temperature comes from NDBC's yearly standard-meteorological
archives (CDIP buoys are archived there under their NDBC IDs); tide-station
water temperature from the CO-OPS API's hourly readings from TIDE_FIRST_YEAR,
without the ones its automatic checks flag (as on the live site; NDBC's copy
carries no flags);
mooring and shore-station water temperature from the IOOS ERDDAP server as
daily means over the whole record. All are cached in data/historical/ (not committed). Re-run only to add stations or change the
method.
"""
import gzip
import os
import re
import time

import numpy as np
import pandas as pd

import ioos
from ndbc import NDBC_URL, http, read_stdmet
from sealevel import water_temperature

# NDBC offshore buoys: NDBC ID -> short name. Includes moorings that report through NDBC (OOI's
# Westport and Newport Shelf, OSU/NANOOS CB-06). 46087 is named for Cape Flattery and 46015 keeps
# Port Orford, so the tide stations there are Neah Bay and Port Orford Harbor.
NDBC_STATIONS = {'46087': 'Cape Flattery', '46088': 'New Dungeness', '46041': 'Cape Elizabeth',
                 '46099': 'OOI Westport Shelf', '46029': 'Columbia River Bar', '46089': 'Tillamook',
                 '46050': 'Stonewall Bank', '46097': 'OOI Newport Shelf', '46128': 'Coos Bay (CB-06)',
                 '46015': 'Port Orford',
                 '46027': 'St Georges', '46022': 'Eel River', '46014': 'Point Arena', '46013': 'Bodega Bay',
                 '46026': 'San Francisco', '46012': 'Half Moon Bay', '46042': 'Monterey Bay',
                 '46028': 'Cape San Martin', '46011': 'Santa Maria', '46054': 'West Santa Barbara',
                 '46053': 'East Santa Barbara', '46025': 'Santa Monica Basin', '46069': 'South Santa Rosa Island'}
# CDIP wave buoys: NDBC ID -> (CDIP station number, short name)
CDIP_STATIONS = {'46267': ('248', 'Angeles Point'), '46211': ('036', 'Grays Harbor'),
                 '46243': ('162', 'Clatsop Spit'), '46248': ('179', 'Astoria Canyon'),
                 '46278': ('270', 'Tillamook Bay South Jetty'), '46229': ('139', 'Umpqua Offshore'),
                 '46244': ('168', 'Humboldt Bay North Spit'), '46213': ('094', 'Cape Mendocino'),
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

# NOAA CO-OPS tide stations (water temperature sensor on the tide gauge, ~1-2 m below
# low water): NDBC ID -> (CO-OPS station number, short name, lat, lon). Names differ from
# CO-OPS's where they'd repeat a buoy's (San Francisco, Point Reyes, North Spit).
TIDE_STATIONS = {'NEAW1': ('9443090', 'Neah Bay', 48.371, -124.602),
                 'PTAW1': ('9444090', 'Port Angeles', 48.125, -123.44),
                 'PTWW1': ('9444900', 'Port Townsend', 48.111, -122.76),
                 'LAPW1': ('9442396', 'La Push', 47.913, -124.636),
                 'TCNW1': ('9446484', 'Tacoma', 47.267, -122.413),
                 'WPTW1': ('9441102', 'Westport', 46.904, -124.105),
                 'TOKW1': ('9440910', 'Toke Point (Willapa Bay)', 46.708, -123.967),
                 'SBEO3': ('9435380', 'South Beach (Newport)', 44.625, -124.045),
                 'PORO3': ('9431647', 'Port Orford Harbor', 42.739, -124.498),
                 'CECC1': ('9419750', 'Crescent City Harbor', 41.746, -124.184),
                 'HBYC1': ('9418767', 'Humboldt Bay (North Spit)', 40.766, -124.217),
                 'ANVC1': ('9416841', 'Arena Cove', 38.915, -123.711),
                 'PCOC1': ('9415144', 'Port Chicago', 38.056, -122.04),
                 'MZXC1': ('9415102', 'Martinez-Amorco Pier', 38.035, -122.125),
                 'PRYC1': ('9415020', 'Point Reyes (Drakes Bay)', 37.994, -122.974),
                 'RCMC1': ('9414863', 'Richmond', 37.923, -122.41),
                 'FTPC1': ('9414290', 'San Francisco (Golden Gate)', 37.806, -122.466),
                 'AAMC1': ('9414750', 'Alameda', 37.772, -122.3),
                 'RTYC1': ('9414523', 'Redwood City', 37.507, -122.21),
                 'MEYC1': ('9413450', 'Monterey Harbor', 36.605, -121.888),
                 'PSLC1': ('9412110', 'Port San Luis', 35.169, -120.754),
                 'OHBC1': ('9410660', 'Los Angeles Harbor', 33.72, -118.273),
                 'LJAC1': ('9410230', 'La Jolla (Scripps Pier)', 32.867, -117.257),
                 'SDBC1': ('9410170', 'San Diego Bay', 32.714, -117.174)}

# positions for buoys missing from NDBC's active-station list (last known, from the IOOS catalog)
FALLBACK_POSITIONS = {'46240': (36.626, -121.907)}

# IOOS moorings with a surface sensor: ERDDAP dataset ID -> (short name, operator, lat, lon)
IOOS_STATIONS = {
    # Washington: NANOOS's server (Cha'Ba, Bellingham Bay) and UW's profiling buoys (see ioos.py).
    # Dabob Bay (orca4) is left out: no casts since May 2025.
    'bellinghambay_surfhydro': ("Bellingham Bay (Se'lhaem)", 'University of Washington / NANOOS', 48.7237, -122.5765),
    'chaba_hydro': ("Cha'Ba (off La Push)", 'University of Washington / NANOOS', 47.9662, -124.95),
    'orca3_L3_depthgridded_025': ('Hansville (Hood Canal)', 'University of Washington / NANOOS', 47.9073, -122.627),
    'npby1_L3_depthgridded_025': ('Point Wells', 'University of Washington / NANOOS', 47.7612, -122.3972),
    'orca2_L3_depthgridded_025': ('Hoodsport (Hood Canal)', 'University of Washington / NANOOS', 47.422, -123.113),
    'orca1_L3_depthgridded_025': ('Twanoh (Hood Canal)', 'University of Washington / NANOOS', 47.375, -123.0083),
    'npby2_L3_depthgridded_025': ('Carr Inlet', 'University of Washington / NANOOS', 47.28, -122.73),
    'bodega-marine-laboratory-bml-': ('Bodega Marine Lab Buoy', 'UC Davis Bodega Marine Laboratory', 38.3118, -123.083),
    'oa2-mbari-buoy': ('MBARI OA2', 'MBARI', 37.1109, -122.3),
    'org_mbari_m1': ('MBARI M1', 'MBARI', 36.7511, -122.0292),
    'oa1-mbari-buoy-1': ('MBARI OA1', 'MBARI', 36.623, -121.9005),
    'mini-mooring-deployment-off-sant': ('Santa Monica Mini-Mooring', 'Scripps', 33.9592, -118.4641),
    'mini-mooring-deployment-off-del-': ('Del Mar Mini-Mooring', 'Scripps', 32.9497, -117.2829),
    'del-mar-mooring-1': ('Del Mar Mooring', 'Scripps', 32.93, -117.32),
    'imperial-beach-mini-mooring-ib-n': ('Imperial Beach North', 'Scripps', 32.6154, -117.1404),
    'imperial-beach-mini-mooring-ib-w': ('Imperial Beach West', 'Scripps', 32.5816, -117.163),
    'imperial-beach-mini-mooring-ib-s': ('Imperial Beach South', 'Scripps', 32.5396, -117.1512),
}

# IOOS shore stations (piers, wharves, seawater intakes; the catalog lists every sensor
# at the surface): ERDDAP dataset ID -> (short name, operator, lat, lon).
# Left out: Hog Island Oyster Hatchery, which reads ~9 °C warmer than Tomales Bay next door
# (hatchery water, not the bay) and stopped reporting on 2026-09-15.
SHORE_STATIONS = {
    'klamath-yurok': ('Klamath (Yurok)', 'UC Davis Bodega Marine Laboratory', 41.5397, -124.1043),
    'edu_humboldt_tdp': ('Trinidad Head', 'Cal Poly Humboldt / CeNCOOS', 41.055, -124.147),
    'edu_humboldt_humboldt': ('Humboldt Bay Pier', 'Cal Poly Humboldt', 40.7775, -124.1965),
    'humboldt-south-bay': ('Humboldt South Bay', 'Cal Poly Humboldt', 40.7234, -124.2234),
    'noyo-bay': ('Noyo Bay', 'UC Davis Bodega Marine Laboratory', 39.429, -123.8136),
    'timber-cove': ('Timber Cove', 'UC Davis Bodega Marine Laboratory', 38.528, -123.2733),
    'fort-ross': ('Fort Ross', 'UC Davis Bodega Marine Laboratory', 38.5097, -123.2442),
    'bodega-bay-bml_wts': ('Bodega Marine Lab Intake', 'UC Davis Bodega Marine Laboratory', 38.3165, -123.0709),
    'tomales-bay-1': ('Tomales Bay', 'UC Davis Bodega Marine Laboratory', 38.1897, -122.9287),
    'inverness': ('Inverness', 'UC Davis Bodega Marine Laboratory', 38.118, -122.867),
    'tiburon-water-tibc1': ('Romberg Tiburon Pier', 'SFSU Romberg Tiburon Center / CeNCOOS', 37.8915, -122.4467),
    'exploratorium-weather': ('Exploratorium Pier', 'Exploratorium', 37.8025, -122.3967),
    'edu_ucsc_scwharf1': ('Santa Cruz Wharf', 'UC Santa Cruz', 36.9603, -122.0203),
    'mlml_mlml_sea': ('Moss Landing Marine Labs Intake', 'Moss Landing Marine Laboratories', 36.8025, -121.7915),
    'monterey-bay-aquarium-seawate': ('Monterey Bay Aquarium Intake', 'Monterey Bay Aquarium', 36.6186, -121.9015),
    'edu_calpoly_marine_morro': ('Morro Bay', 'Cal Poly San Luis Obispo', 35.3708, -120.8589),
    'morro-bay-bs1': ('Morro Bay (Back Bay)', 'Cal Poly San Luis Obispo', 35.3338, -120.8473),
    'san-luis-bay-cal-poly-pier-shore': ('Cal Poly Pier', 'Cal Poly San Luis Obispo', 35.1701, -120.7406),
    'stearns-wharf-automated-shore-st-1': ('Stearns Wharf', 'SCCOOS', 34.4107, -119.6874),
    'santa-monica-pier-automated-shor': ('Santa Monica Pier', 'SCCOOS', 34.0086, -118.4986),
    'newport-pier-automated-shore-sta': ('Newport Beach Pier', 'SCCOOS', 33.6073, -117.9289),
}
# ERDDAP water-temperature variable where it isn't sea_water_temperature
# (the CTD; Newport's sea_water_temperature only starts in 2026)
ERDDAP_VARS = {'stearns-wharf-automated-shore-st-1': 'sea_water_temperature_ctd',
               'newport-pier-automated-shore-sta': 'sea_water_temperature_ctd'}

BASE_YEARS = (1991, 2020)
MIN_BASE_YEARS = 15
LAST_YEAR = pd.Timestamp.now().year - 1  # last complete year (NDBC's yearly archive)
MIN_YEARS = 8
MIN_DAYS_PER_YEAR = 60  # a year counts toward the record with this many daily means
CLIM_HALF_WINDOW_DAYS = 5
MIN_HOURS_PER_DAY = 12
MIN_DAYS_PER_DOY = 40  # a day of year needs this many daily means in its window
SMOOTH_DAYS = 31
# CO-OPS's water temperature records go back to the 1990s at most stations, but before 2005 they hold
# long stretches its flags don't catch (e.g. Los Angeles Harbor ~4 °C cold in 1999-2000, South Beach
# ~5.6 °C cold in 2004, Humboldt Bay ~5.9 °C warm in 2000, a 2 °C step at Point Reyes 1999-2006), so
# tide-station climatologies start in 2005, as NDBC's archive of these stations does
TIDE_FIRST_YEAR = 2005
# stations whose record has a lasting shift, so the general rule would give a misleading normal: years used
# instead. Tacoma ran 1-1.5 °C colder in 2008-2012 than in any other stretch (colder than every other
# gauge on the coast) and has stayed ~0.6 °C above its 2005-2020 mean every year since 2015, so a
# 2005-2020 normal flagged ordinary recent autumns as suspected marine heatwaves
PERIOD_OVERRIDES = {'TCNW1': (2015, 2025)}
# years left out of a station's climatology: stretches of clearly wrong readings that the operators' flags
# don't catch, found by comparing each station's monthly anomalies with the median of its neighbors' (within
# 150 km) and then looking at the readings themselves
BAD_YEARS = {
    'CECC1': (2005,),        # Crescent City: readings climb to 21-41 °C from September
    'RCMC1': (2007,),        # Richmond: 21-25 °C in January-June (San Francisco Bay is ~10-17 °C then)
    'PRYC1': (2005, 2006),   # Point Reyes: ~3 °C below its neighbors (the end of a 1999-2006 cold step)
    '46232': (2006, 2007),   # Point Loma South: 1-3 °C below Torrey Pines Outer (46225, ~30 km) every quarter
}

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
HIST = os.path.join(DATA, 'historical')


def station_table():
    xml = http.get(f'{NDBC_URL}/activestations.xml', timeout=60).text
    defs = [(sid, name, 'ndbc', '', '') for sid, name in NDBC_STATIONS.items()]
    defs += [(sid, name, 'cdip', cdip, '') for sid, (cdip, name) in CDIP_STATIONS.items()]
    defs += [(sid, name, 'tide', '', coops) for sid, (coops, name, _, _) in TIDE_STATIONS.items()]
    fallback = {**FALLBACK_POSITIONS, **{sid: (lat, lon) for sid, (_, _, lat, lon) in TIDE_STATIONS.items()}}
    rows = []
    for sid, name, kind, cdip, coops in defs:
        tag = re.search(rf'<station id="{sid}"[^>]*>', xml, flags=re.I)
        attr = dict(re.findall(r'(\w+)="([^"]*)"', tag.group(0))) if tag else {}
        lat, lon = (float(attr['lat']), float(attr['lon'])) if 'lat' in attr and sid not in FALLBACK_POSITIONS \
            else fallback[sid]
        rows.append({'station': sid, 'name': name, 'type': kind, 'cdip': cdip, 'coops': coops, 'lat': lat, 'lon': lon,
                     'long_name': attr.get('name', ''), 'operator': ''})
    for kind, table in [('ioos', IOOS_STATIONS), ('shore', SHORE_STATIONS)]:
        for ds, (name, operator, lat, lon) in table.items():
            rows.append({'station': ds, 'name': name, 'type': kind, 'cdip': '', 'coops': '', 'lat': lat, 'lon': lon,
                         'long_name': '', 'operator': operator,
                         'erddap_var': ERDDAP_VARS.get(ds, 'sea_water_temperature')})
    return pd.DataFrame(rows)


def daily_history(sid, years):
    parts = []
    for year in years:
        path = os.path.join(HIST, f'{sid.lower()}h{year}.txt.gz')  # NDBC file names are lower case
        missing = path + '.missing'  # remembers years NDBC doesn't have
        if os.path.exists(missing):
            continue
        if not os.path.exists(path):
            r = http.get(f'{NDBC_URL}/data/historical/stdmet/{sid.lower()}h{year}.txt.gz', timeout=60)
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


def coops_history(coops, years):
    """A tide station's daily-mean water temperature from CO-OPS's hourly readings, without the ones its
    automatic checks flag; each year cached in data/historical/ (an empty file for a year with none)."""
    parts = []
    for year in years:
        path = os.path.join(HIST, f'coops_{coops}_{year}.csv')
        if not os.path.exists(path):
            s = water_temperature(coops, pd.Timestamp(f'{year}-01-01'), pd.Timestamp(f'{year}-12-31 23:00'), 'h')
            s.rename_axis('time_utc').to_csv(path)
            time.sleep(0.2)  # be gentle with CO-OPS
        df = pd.read_csv(path, index_col=0)
        if len(df):
            parts.append(pd.Series(df['wtmp_c'].values, index=pd.to_datetime(df.index, utc=True)))
    if not parts:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz='UTC'))
    w = pd.concat(parts)
    w = w[~w.index.duplicated()].resample('h').mean()
    n = w.resample('D').count()
    return w.resample('D').mean()[n >= MIN_HOURS_PER_DAY].dropna()


def years_with_data(daily):
    counts = daily.groupby(daily.index.year).size()
    return sorted(counts[counts >= MIN_DAYS_PER_YEAR].index)


def ioos_history(dataset):
    """A mooring's or shore station's daily-mean surface temperature and sensor depth,
    cached in data/historical/."""
    path = os.path.join(HIST, f'ioos_{dataset}.csv')
    if not os.path.exists(path):
        daily, z = ioos.daily_history(dataset, ERDDAP_VARS.get(dataset, 'sea_water_temperature'))
        daily.rename_axis('time_utc').to_frame().assign(z=z).to_csv(path)
    df = pd.read_csv(path, index_col=0)
    daily = pd.Series(df['wtmp_c'].values, index=pd.to_datetime(df.index, utc=True))
    return daily, (float(df['z'].iloc[0]) if len(df) else None)


def pick_period(daily_base, get_all):
    """Daily means over the climatology period, and the period: BASE_YEARS when they
    hold MIN_BASE_YEARS years of data, else every year through LAST_YEAR if at
    least MIN_YEARS, else (None, None)."""
    years = years_with_data(daily_base)
    if len(years) >= MIN_BASE_YEARS:
        return daily_base, (years[0], years[-1])  # the years actually covered, e.g. 2004-2020 for older CDIP buoys
    daily = get_all()
    daily = daily[daily.index.year <= LAST_YEAR]
    years = years_with_data(daily)
    if len(years) >= MIN_YEARS:
        return daily[daily.index.year.isin(years)], (years[0], years[-1])
    return None, None


def choose_period(sid, kind):
    if sid in PERIOD_OVERRIDES:
        first, last = PERIOD_OVERRIDES[sid]
        daily = without_bad_years(sid, coops_history(TIDE_STATIONS[sid][0], range(first, last + 1)))
        years = years_with_data(daily)
        return daily[daily.index.year.isin(years)], (years[0], years[-1])
    if kind == 'tide':
        coops = TIDE_STATIONS[sid][0]
        get = lambda last: coops_history(coops, range(TIDE_FIRST_YEAR, last + 1))
    elif kind in ('ioos', 'shore'):
        daily, _ = ioos_history(sid)
        get = lambda last: daily[(daily.index.year >= BASE_YEARS[0]) & (daily.index.year <= last)]
    else:
        get = lambda last: daily_history(sid, range(BASE_YEARS[0], last + 1))
    return pick_period(without_bad_years(sid, get(BASE_YEARS[1])), lambda: without_bad_years(sid, get(LAST_YEAR)))


def without_bad_years(sid, daily):
    """The daily means without the station's BAD_YEARS."""
    return daily[~daily.index.year.isin(BAD_YEARS.get(sid, ()))]


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
    for sid, kind in zip(stations['station'], stations['type']):
        daily, period = choose_period(sid, kind)
        periods[sid] = period
        if daily is not None:
            rows += climatology(sid, daily)
        print(f'{sid}: {f"{period[0]}-{period[1]}" if period else "no climatology (record too short)"}')
    stations['clim_start'] = [periods[s][0] if periods[s] else None for s in stations['station']]
    stations['clim_end'] = [periods[s][1] if periods[s] else None for s in stations['station']]
    # sensor depth for moorings (m below the surface); the buoys measure at ~0.5-1 m
    stations['depth_m'] = [abs(ioos_history(s)[1]) if k == 'ioos' and ioos_history(s)[1] is not None else None
                           for s, k in zip(stations['station'], stations['type'])]
    stations.to_csv(os.path.join(DATA, 'stations.csv'), index=False)
    clim = pd.DataFrame(rows)
    clim.round(3).to_csv(os.path.join(DATA, 'climatology.csv'), index=False)
    print(f'Wrote data/stations.csv ({len(stations)} stations, {sum(p is not None for p in periods.values())} '
          f'with a climatology) and data/climatology.csv ({len(clim)} rows)')
