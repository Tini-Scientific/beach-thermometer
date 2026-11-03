# Beach Thermometer: notes for Claude Code

An experimental Tini Scientific site: hourly water temperature (and water level at the NOAA tide
stations) along California's coast, flagging suspected marine heatwaves and gauges reaching NWS flood
levels. Static site on GitHub Pages, rebuilt hourly by `.github/workflows/update.yml`. See README.md
for the file layout and site/methods.html for the methods as published.

Live: https://tini-scientific.github.io/beach-thermometer/ (switching back to beachthermometer.com
around 2026-11-30: then update the og: URLs in site/index.html and SITE_URL in feed.py).

## Rules
- The repository is public: anything committed here, including this file, is visible to anyone.
- Don't contact data providers or other organizations. Don't get around websites' bot protection.
- Don't use the NOAA logo (or other agencies' logos) without permission. Name sources in text instead.
- Wording: flags are "suspected" (marine heatwave, cold spell) and "flood level reached" at a gauge,
  never "flooding" or a "warning". The site is experimental, not an official product; official
  products and alerts are linked on the methods page.
- When a method, threshold or data source changes, update site/methods.html (and the accessibility
  statement in site/accessibility.html if it affects accessibility), and keep "last updated" dates current.

## Accessibility (a grant commitment: WCAG 2.0 AA, Section 508)
- Keyboard access, screen-reader names and announcements, the station table as the map's text
  equivalent, a written summary for every chart, 200% and 400% zoom without sideways scrolling,
  reduced-motion support, and nothing shown by color alone.
- Test changes to site/ with Playwright and axe-core (axe should report no violations) on desktop,
  an iPhone-sized window, and 200%/400% zoom (e.g. 640x450 at scale 2, 320x225 at scale 4).
- Professional accessibility audit planned for Q1 2027. Contact for the public: waves@tiniscientific.com.

## Units and style
- Temperatures in °F and °C (°F by default), lengths in ft and m; both are rendered, CSS shows one
  (body.imperial). Signed values use a true minus sign (−).
- Montserrat (with its italic) and the Tini colors defined at the top of site/index.html.

## Running locally
- Serve the site: `cd site && python3 -m http.server 8766`, then http://localhost:8766.
- site/data/ and site/feed.xml are gitignored and only exist locally after running the scripts.
  `python sealevel.py` alone refreshes the tide-station data. Running `update.py` locally also
  rewrites data/archive/ (which the hourly job commits), so avoid committing local archive changes.
- Python with pandas: /opt/anaconda3/bin/python3. JavaScript syntax check: pull the <script>
  blocks out of index.html into a file and run `node --check` on it.

## Daily email
- feed.py posts one entry a day (~7:20 am Pacific) to site/feed.xml; data/feed/entries.json keeps
  the history the next day compares with (committed by the workflow).
- Subscribers get it by email through Buttondown's RSS-to-email (set up by the user; set to draft
  mode while testing).
