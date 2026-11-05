"""Daily marine heatwave summary as an Atom feed, for update.py.

Once a day, on the first hourly run at or after POST_HOUR Pacific time, the
current suspected marine heatwaves are compared with the previous day's entry
and a new entry is written: which stations started or ended a heatwave, and
which are continuing. An entry is posted every day, changes or not.

- data/feed/entries.json: the newest MAX_ENTRIES entries, each with the
  heatwave stations it saw (committed by the workflow, so the next day has
  something to compare with).
- site/feed.xml: the Atom feed, rewritten from those entries on every run
  (not committed; published to GitHub Pages).
"""
import json
import os
from html import escape
from xml.sax.saxutils import escape as xml_escape

import pandas as pd

# switch to https://beachthermometer.com/ with the custom domain
SITE_URL = 'https://tini-scientific.github.io/beach-thermometer/'
TINI_URL = 'https://www.tiniscientific.com'
TZ = 'America/Los_Angeles'
POST_HOUR = 7
MAX_ENTRIES = 30
# coastal regions for the continuing list, by latitude: the southern edge of each (north to south)
# (Washington includes Puget Sound and the Strait; the Columbia River mouth, ~46.25 N, is the state line)
REGIONS = [('Washington', 46.25), ('Oregon', 42.0), ('Northern California', 38.2), ('San Francisco Bay Area', 37.2),
           ('Central California', 34.6), ('Southern California', -90)]
OTHER_LINKS = [
    ('NOAA California Current Marine Heatwave Tracker',
     'https://www.integratedecosystemassessment.noaa.gov/regions/california-current/california-current-marine-heatwave-tracker-blobtracker'),
    ('NOAA marine heatwave forecasts (Physical Sciences Laboratory)', 'https://psl.noaa.gov/marine-heatwaves/'),
    ('NOAA Coral Reef Watch: Marine Heatwave Watch', 'https://coralreefwatch.noaa.gov/product/marine_heatwave/'),
    ('CeNCOOS marine heatwave', 'https://www.cencoos.org/marine-heatwave/'),
    ('CalOOS ocean sensor dashboard (SCCOOS and CeNCOOS)',
     'https://data.caloos.org/#dashboards/sensors/6f6e75db-710c-491a-87d6-af4a4d065537/by-station?selected=110889'),
    ('NANOOS Visualization System (Washington and Oregon ocean data)', 'https://nvs.nanoos.org/'),
]
TYPE_ONE = {'ndbc': 'Offshore buoy', 'cdip': 'Wave buoy', 'ioos': 'Mooring', 'tide': 'Tide station', 'shore': 'Shore station'}

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, 'data', 'feed', 'entries.json')
FEED = os.path.join(HERE, 'site', 'feed.xml')


def anomaly(c):
    """5.1 °F (2.8 °C) above normal, for an anomaly in °C (words, not a sign, so it reads well aloud)."""
    return f"{abs(c) * 9 / 5:.1f} °F ({abs(c):.1f} °C) {'above' if c >= 0 else 'below'} normal"


def days(e):
    """Length of the heatwave so far; 'at least' when it runs back to the start of the record."""
    n = e.get('hot_days', 0)
    return f"{'at least ' if e.get('streak_is_whole_record') else ''}{n} day{'s' * (n != 1)}"


def link(e):
    return f'<a href="{SITE_URL}?station={escape(e["id"])}">{escape(e["name"])}</a>'


def region(e):
    return next(name for name, south in REGIONS if e['lat'] >= south)


def make_entry(entries, previous, now):
    """Today's entry: heatwaves started, ended and continuing since `previous` (None on the first day)."""
    by_id = {e['id']: e for e in entries}
    hot = sorted((e for e in entries if e.get('status') == 'heatwave'), key=lambda e: -e['lat'])  # north to south
    hot_ids = {e['id'] for e in hot}
    before = set(previous['heatwave_ids']) if previous else hot_ids
    started = [e for e in hot if e['id'] not in before]
    ongoing = [e for e in hot if e['id'] in before]
    no_data = lambda e: e.get('status') in ('stale', None)
    # a station that stops reporting mid-heatwave is neither ended nor continuing: it's held over, so when it
    # reports again it counts as continuing (still hot) or ended, not as a new start
    held = {i for i in before - hot_ids if i in by_id and no_data(by_id[i])}
    ended = sorted((by_id[i] for i in before - hot_ids - held if i in by_id), key=lambda e: -e['lat'])
    stale = sorted((e for e in entries if no_data(e)), key=lambda e: e['name'])

    local = now.tz_convert(TZ)
    changes = ', '.join(f'{len(x)} {word}' for x, word in [(started, 'new'), (ended, 'ended')] if x) or 'no changes'
    # the product name goes in the sender name (set in Mailchimp); the date stays, last, so daily emails don't thread together
    title = f"Suspected marine heatwaves: {changes}, {len(hot)} total ({local:%b} {local.day})"

    parts = ['<p><strong>Experimental:</strong> a product from Tini Scientific that flags suspected marine heatwaves with its own '
             'method, which differs from NOAA\'s, so it won\'t always agree with NOAA\'s products (linked below).</p>']
    if previous is None:
        parts.append('<p>First daily summary: every station currently in a suspected marine heatwave is listed as continuing.</p>')
    if started:
        parts.append('<h2>Suspected marine heatwaves started</h2><ul>' + ''.join(
            f'<li>{link(e)} ({TYPE_ONE[e["type"]]}): {days(e)}, {anomaly(e["anom_c"])} for the date</li>'
            for e in started) + '</ul>')
    if ended:
        parts.append('<h2>Suspected marine heatwaves ended</h2><ul>' + ''.join(
            f'<li>{link(e)}</li>' for e in ended) + '</ul>')
    parts.append(f'<h2>Suspected marine heatwaves continuing ({len(ongoing)})</h2>' + ('' if ongoing else '<p>None.</p>'))
    for name, _ in REGIONS:
        here = [e for e in ongoing if region(e) == name]
        if here:
            parts.append(f'<h3>{name} ({len(here)})</h3><ul>' + ''.join(f'<li>{link(e)}: {days(e)}</li>' for e in here) + '</ul>')
    if stale:
        parts.append(f'<h2>No recent data ({len(stale)})</h2><ul>' + ''.join(
            f'<li>{escape(e["name"])}' + (' (in a suspected marine heatwave when it stopped reporting)' if e['id'] in held else '') + '</li>'
            for e in stale) + '</ul>')
    parts.append(f'<p>A suspected marine heatwave is at least 5 days in a row with the daily mean above the '
                 f'station\'s 90th percentile for the date. These flags are automated, not official warnings. '
                 f'<a href="{SITE_URL}">Open the map</a>.</p>')
    parts.append('<p>Marine heatwave products from NOAA and the West Coast\'s ocean observing systems:</p><ul>'
                 + ''.join(f'<li><a href="{url}">{escape(label)}</a></li>' for label, url in OTHER_LINKS) + '</ul>')
    # a quiet credit, like the map's "Dashboard by": small purple-text logo (the site's white-text one would vanish
    # on a white email) and a line of small text, which also carries the link when an email app blocks images
    parts.append(f'<div style="margin-top:28px;padding-top:14px;border-top:1px solid #e5e7eb;font-size:13px;color:#6b7280">'
                 f'<a href="{TINI_URL}"><img src="{SITE_URL}assets/logo-email.png" alt="Tini Scientific" width="105" height="24" '
                 f'style="display:block;border:0;margin-bottom:6px"></a>'
                 f'Beach Thermometer is an experimental project by Tini Scientific: '
                 f'<a href="{TINI_URL}" style="color:#6b7280">tiniscientific.com</a></div>')
    return {'date': f'{local:%Y-%m-%d}', 'updated': now.strftime('%Y-%m-%dT%H:%M:%SZ'), 'title': title,
            'html': ''.join(parts), 'heatwave_ids': sorted(hot_ids | held)}


def write_atom(feed_entries):
    updated = feed_entries[0]['updated'] if feed_entries else pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%dT%H:%M:%SZ')
    items = ''.join(f"""
  <entry>
    <id>tag:tiniscientific.com,2026:beach-thermometer/{e['date']}</id>
    <title>{xml_escape(e['title'])}</title>
    <updated>{e['updated']}</updated>
    <link rel="alternate" type="text/html" href="{SITE_URL}"/>
    <content type="html">{xml_escape(e['html'])}</content>
  </entry>""" for e in feed_entries)
    with open(FEED, 'w') as f:
        f.write(f"""<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <id>tag:tiniscientific.com,2026:beach-thermometer</id>
  <title>Beach Thermometer: daily marine heatwave summary</title>
  <subtitle>Suspected marine heatwaves at West Coast buoys, moorings, tide gauges and shore stations: which started, ended or are continuing.</subtitle>
  <link rel="self" type="application/atom+xml" href="{SITE_URL}feed.xml"/>
  <link rel="alternate" type="text/html" href="{SITE_URL}"/>
  <author><name>Tini Scientific</name><email>waves@tiniscientific.com</email></author>
  <updated>{updated}</updated>{items}
</feed>
""")


def update_feed(entries, now):
    """Add today's entry when it's due, then rewrite the Atom feed."""
    feed_entries = []
    if os.path.exists(STATE):
        with open(STATE) as f:
            feed_entries = json.load(f)
    local = now.tz_convert(TZ)
    due = local.hour >= POST_HOUR and not (feed_entries and feed_entries[0]['date'] == f'{local:%Y-%m-%d}')
    if due:
        feed_entries = [make_entry(entries, feed_entries[0] if feed_entries else None, now)] + feed_entries[:MAX_ENTRIES - 1]
        print(f"Feed: {feed_entries[0]['title']}")
    if due or not os.path.exists(STATE):  # written from the first run, so the workflow always has it to commit
        os.makedirs(os.path.dirname(STATE), exist_ok=True)
        with open(STATE, 'w') as f:
            json.dump(feed_entries, f, indent=1)
    write_atom(feed_entries)
