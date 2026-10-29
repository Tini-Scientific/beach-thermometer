# Beach Thermometer

Real-time surface water temperature at California's NDBC offshore buoys (diamonds),
CDIP wave buoys (circles), IOOS moorings (pentagons), NOAA CO-OPS tide stations (squares) and IOOS shore stations
(downward triangles), compared with each station's own climatology (1991–2020 where the
record allows; see `build_climatology.py`), at [beachthermometer.com](https://beachthermometer.com).
A Tini Scientific project.

Every hour a GitHub Actions job fetches the latest buoy data from NOAA's National
Data Buoy Center, flags suspected marine heatwaves (Hobday et al. 2016: at least
5 days in a row above the climatological 90th percentile), and republishes the
site on GitHub Pages.

## What's here

| Path | What it is |
|---|---|
| `site/index.html` | The map (Leaflet + Chart.js); reads `site/data/` in the browser |
| `site/assets/` | Logo |
| `update.py` | Hourly job: fetch NDBC, update the archive, write `site/data/` |
| `build_climatology.py` | One-time setup: `data/stations.csv` and `data/climatology.csv` |
| `ndbc.py` | Shared NDBC file reader |
| `ioos.py` | Mooring and shore-station data from the IOOS ERDDAP server (surface sensors only, despiked) |
| `data/archive/{station}/{YYYY-MM}.csv` | Hourly water temperature, kept going forward (committed by the job) |
| `.github/workflows/update.yml` | The hourly schedule |

`site/data/` is written fresh on every run and published straight to Pages, so it
isn't committed (that would add hundreds of MB of history a year).

## Setting up the GitHub site (once)

1. Create an empty **public** repository on GitHub named `beachthermometer`
   (no README, no .gitignore).
2. Push this folder to it (see below).
3. In the repository: **Settings → Pages → Build and deployment → Source:
   GitHub Actions**.
4. **Actions** tab → *Update and deploy* → **Run workflow**. After a minute or
   two the site is live at `https://<username>.github.io/beachthermometer/`.
   From then on it updates at 20 past every hour.

### Custom domain (beachthermometer.com)

1. **Settings → Pages → Custom domain**: enter `beachthermometer.com` and save.
2. At the domain registrar, add DNS records:
   - four `A` records for `@`: `185.199.108.153`, `185.199.109.153`,
     `185.199.110.153`, `185.199.111.153`
   - a `CNAME` record for `www` → `<username>.github.io`
3. In your GitHub account (not the repo): **Settings → Pages → Add a domain**
   to verify `beachthermometer.com`, so no one else can claim it.
4. Once the DNS check passes (minutes to a few hours), tick **Enforce HTTPS**.

## Running it locally

```bash
pip install -r requirements.txt
python update.py                      # fetch data, write site/data/
cd site && python -m http.server 8765 # then open http://localhost:8765
```

Open a buoy directly with `?station=46026`. (Opening `index.html` as a file won't
work: browsers block a file page from loading its data.)

## Maintenance

- **If updates stop**, `update.py` fails when no buoy has reported for 6 hours,
  and GitHub emails the repository owner about the failed run.
- GitHub pauses scheduled workflows in repositories with no activity for 60 days.
  The job's hourly archive commits should count as activity; if it is ever paused,
  re-enable it from the Actions tab.
- **Adding a station**: add it to `STATIONS` in `build_climatology.py`, run
  `python build_climatology.py` (downloads its 1991–2020 history), and commit
  `data/stations.csv` and `data/climatology.csv`.

## Data

Buoy observations: NOAA National Data Buoy Center (realtime2 and historical
standard meteorological files), including the Coastal Data Information Program (CDIP,
Scripps Institution of Oceanography) wave buoys and NOAA CO-OPS tide stations it distributes; IOOS sensor catalog
(erddap.sensors.ioos.us) for the MBARI, Scripps and UC Davis Bodega Marine Laboratory
moorings and the CeNCOOS, SCCOOS, Bodega Marine Laboratory, Cal Poly, UC Santa Cruz, Moss
Landing, Monterey Bay Aquarium, SFSU Tiburon and Exploratorium shore stations. Basemaps: Esri.
