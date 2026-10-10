# cams_eac4_algeria_00_12_desc_2015_2024.py
# CAMS Global Reanalysis (EAC4)
# Algeria-only box, two steps/day (00:00, 12:00 UTC), monthly files,
# descending from 2024-12 to 2015-01. Output: NetCDF.

from pathlib import Path
import time
import calendar
import cdsapi  # ADS/CDS API client

# ====== Config ======
START_YEAR, START_MONTH = 2024, 12   # inclusive (EAC4 ends at 2024-12)
END_YEAR, END_MONTH     = 2015,  1   # inclusive

# Algeria bounding box (N, W, S, E) — slightly buffered
AREA = [38, -9, 18, 12]

# Two analysis hours per day
TIMES = ["00:00", "12:00"]

# Output
OUTDIR = Path("./datasets/CAMS")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---- Variables helpful for air-pollution prediction ----
# Surface pollutants (mass mixing ratio at surface in EAC4)
POLLUTANTS = [
    "ozone",
    "nitrogen_dioxide",
    "nitric_oxide",
    "carbon_monoxide",
    "sulphur_dioxide",
    "particulate_matter_2.5um",
    "particulate_matter_10um",
]

# Meteorology & drivers (single/surface level in EAC4)
MET_VARS = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "2m_dewpoint_temperature",
    "boundary_layer_height",
    "surface_pressure",
    "total_cloud_cover",
]

VARIABLES = POLLUTANTS + MET_VARS

# If you hit "cost limits exceeded", set True to fetch one variable per request
SPLIT_BY_VARIABLE = False

DATASET = "cams-global-reanalysis-eac4"
FORMAT = "netcdf"  # or "grib" if you prefer GRIB

# ====== Helpers ======
def months_desc(start_y: int, start_m: int, end_y: int, end_m: int):
    y, m = start_y, start_m
    while (y, m) >= (end_y, end_m):
        yield y, m
        if m == 1:
            y, m = y - 1, 12
        else:
            m -= 1

def last_day(year: int, month: int) -> str:
    return f"{calendar.monthrange(year, month)[1]:02d}"

def backoff_retrieve(c, dataset, request, target, retries=6, base_sleep=20):
    for attempt in range(1, retries + 1):
        try:
            c.retrieve(dataset, request, target)
            return
        except Exception as e:
            # remove tiny/partial file so next attempt can overwrite
            try:
                p = Path(target)
                if p.exists() and p.stat().st_size < 4096:
                    p.unlink()
            except Exception:
                pass
            if attempt == retries:
                raise
            wait = base_sleep * attempt
            print(f"[{dataset}] Attempt {attempt} failed: {e}\nRetrying in {wait}s…")
            time.sleep(wait)

def main():
    # Tip: if you also use CDS for ERA5, you can keep using ~/.cdsapirc;
    # ADS and CDS share the same client URL/key mechanism. (ADS hosts EAC4.)
    c = cdsapi.Client()

    for year, month in months_desc(START_YEAR, START_MONTH, END_YEAR, END_MONTH):
        mstr = f"{month:02d}"
        date_range = f"{year}-{mstr}-01/{year}-{mstr}-{last_day(year, month)}"

        if SPLIT_BY_VARIABLE:
            # ultra-safe mode: one variable per file
            for var in VARIABLES:
                target = OUTDIR / f"eac4_alg_{var}_{year}_{mstr}_00_12.nc"
                if target.exists():
                    print(f"✓ Exists: {target}")
                    continue
                print(f"→ {year}-{mstr} | {var} @ 00/12 UTC")
                req = {
                    "date": [date_range],
                    "time": TIMES,
                    "variable": [var],
                    "area": AREA,                 # [N, W, S, E]
                    "data_format": FORMAT,        # ADS uses 'data_format'
                }
                backoff_retrieve(c, DATASET, req, str(target))
        else:
            # batch variables into one monthly file
            target = OUTDIR / f"eac4_alg_{year}_{mstr}_00_12.nc"
            if target.exists():
                print(f"✓ Exists: {target}")
            else:
                print(f"→ {year}-{mstr} | {len(VARIABLES)} vars @ 00/12 UTC")
                req = {
                    "date": [date_range],
                    "time": TIMES,
                    "variable": VARIABLES,
                    "area": AREA,                 # [N, W, S, E]
                    "data_format": FORMAT,
                }
                backoff_retrieve(c, DATASET, req, str(target))

    print("All done.")

if __name__ == "__main__":
    main()
