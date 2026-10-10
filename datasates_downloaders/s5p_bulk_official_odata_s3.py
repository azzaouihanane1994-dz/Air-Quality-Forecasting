#!/usr/bin/env python3
# Bulk Sentinel-5P L2 downloader (official CDSE logic)
# OData (contains(Name,...)) -> temporary S3 credentials -> S3 download
# After download: trim to Algeria bbox and save as <filename>.dz.nc
# Resume-safe: per-file and per-product ".done" markers
# Limit to at most one product per day (first that intersects Algeria)
# pip install boto3 requests tqdm xarray netCDF4 numpy cftime
# pip install shapely pyproj rtree

import os
import time
import json
import boto3
import requests
import numpy as np
import xarray as xr
from tqdm import tqdm
from datetime import date, timedelta
from typing import Tuple

# =========================
# CREDENTIALS & CONFIG
# =========================
CDSE_USER = os.environ.get("CDSE_USER", "")
CDSE_PASS = os.environ.get("CDSE_PASS", "")

OUT_DIR    = "./datasets/s5p"
DATE_START = date(2025, 1, 1)
DATE_END   = date.today()
STREAM     = "OFFL"  # OFFL or NRTI

# Algeria bbox (W, S, E, N) in degrees
BBOX: Tuple[float, float, float, float] = (-9.0, 18.0, 12.0, 38.0)

# Keep only up to this many products per day (per variable).
# Set to 1 to pick the first granule that actually intersects Algeria.
MAX_PRODUCTS_PER_DAY = 1

# Delete big original .nc after writing trimmed .dz.nc?
DELETE_ORIGINAL_AFTER_TRIM = True

# Product label -> S5P Name token (used in filenames)
PRODUCT_CODES = {
    "NO2":    "NO2____",
    "CO":     "CO_____",
    "SO2":    "SO2____",
    "HCHO":   "HCHO___",
    "AER_AI": "AER_AI_",  # AAI
    "CLOUD":  "CLOUD__",
}

# CDSE endpoints (official)
CONFIG = {
    "auth_server_url": "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
    "odata_base_url":  "https://catalogue.dataspace.copernicus.eu/odata/v1/Products",
    "s3_endpoint_url": "https://eodata.dataspace.copernicus.eu",
    "tmp_keys_api":    "https://s3-keys-manager.cloudferro.com/api/user/credentials",
    "tmp_keys_delete": "https://s3-keys-manager.cloudferro.com/api/user/credentials/access_id/{access_id}",
}

# Politeness throttle for OData calls
ODATA_SLEEP_SECONDS = 0.2

# Refresh temp S3 creds after this many minutes (safety margin)
TEMP_CREDS_REFRESH_MIN = 55

# =========================
# OFFICIAL-FLOW HELPERS
# =========================
def get_access_token(username: str, password: str) -> str:
    data = {
        "client_id": "cdse-public",
        "grant_type": "password",
        "username": username,
        "password": password,
    }
    r = requests.post(CONFIG["auth_server_url"], data=data, verify=True, allow_redirects=False)
    if r.status_code != 200:
        raise SystemExit(f"Failed to retrieve access token (HTTP {r.status_code}): {r.text[:300]}")
    return r.json()["access_token"]

def odata_query_by_name_contains(name_substring: str, headers: dict, top: int = 200) -> list:
    url = f"{CONFIG['odata_base_url']}?$filter=contains(Name,'{name_substring}')&$top={top}"
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        raise SystemExit(f"OData error (HTTP {r.status_code}): {r.text[:300]}")
    return r.json().get("value", [])

def make_temp_s3_credentials(headers: dict) -> dict:
    r = requests.post(CONFIG["tmp_keys_api"], headers=headers)
    if r.status_code != 200:
        raise SystemExit(f"Failed to create temporary S3 credentials (HTTP {r.status_code}): {r.text[:300]}")
    creds = r.json()
    print(f"Temp S3 credentials created. access_id={creds['access_id']}")
    return creds

def delete_temp_s3_credentials(headers: dict, access_id: str):
    url = CONFIG["tmp_keys_delete"].format(access_id=access_id)
    r = requests.delete(url, headers=headers)
    if r.status_code == 204:
        print("Temp S3 credentials deleted.")
    else:
        print(f"Failed to delete temp S3 credentials (HTTP {r.status_code}).")

def daterange(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)

# =========================
# IO / RESUME HELPERS
# =========================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def list_nc_keys(s3_resource, bucket: str, prefix: str):
    """Yield only .nc object keys under prefix."""
    bucket_obj = s3_resource.Bucket(bucket)
    for obj in bucket_obj.objects.filter(Prefix=prefix):
        if obj.key.lower().endswith(".nc"):
            yield obj.key

def skip_if_done_marker(path: str) -> bool:
    return os.path.exists(path)

def write_done_marker(path: str):
    with open(path, "w") as f:
        f.write("ok")

def download_nc_file(s3_client, bucket: str, key: str, local_path: str) -> tuple[bool, int]:
    """
    Download a single .nc file with resume behavior.
    Returns (downloaded_bool, size_bytes_written_or_existing)
    """
    head = s3_client.head_object(Bucket=bucket, Key=key)
    size = head["ContentLength"]

    # Resume: if local exists and size matches, skip
    if os.path.exists(local_path) and os.path.getsize(local_path) == size:
        print(f"    [skip exists] {os.path.basename(local_path)}")
        return (False, size)

    disp_name = os.path.basename(local_path)
    with tqdm(
        total=size, unit="B", unit_scale=True, desc=disp_name, ncols=90,
        bar_format="{desc:.60}|{bar:25}| {percentage:3.0f}% {n_fmt}/{total_fmt}B"
    ) as pbar:
        def cb(bytes_transferred):
            pbar.update(bytes_transferred)
        s3_client.download_file(bucket, key, local_path, Callback=cb)

    return (True, size)

# =========================
# TRIMMER (Algeria bbox)
# =========================
def _normalize_lon(lon_arr):
    lon = np.array(lon_arr)
    return ((lon + 180.0) % 360.0) - 180.0

def _open_s5p_for_latlon(nc_path: str):
    """Open a group that has lat/lon; return (ds, lat_name, lon_name, group)."""
    tried = []
    for grp in ["PRODUCT", None]:   # most S5P L2 place lat/lon in PRODUCT
        try:
            ds = xr.open_dataset(nc_path, group=grp, engine="netcdf4", mask_and_scale=False)
        except Exception as e:
            tried.append(f"{grp or 'root'}: {e}")
            continue
        lower = {k.lower(): k for k in ds.variables}
        lat = next((lower[k] for k in ["latitude","lat"] if k in lower), None)
        lon = next((lower[k] for k in ["longitude","lon"] if k in lower), None)
        if lat and lon:
            return ds, lat, lon, (grp or "root")
        ds.close()
        tried.append(f"{grp or 'root'}: no lat/lon")
    print("    [trim] Could not find lat/lon. Tried ->", "; ".join(tried))
    return None, None, None, None

def trim_nc_to_bbox(in_nc: str, out_nc: str, bbox: Tuple[float,float,float,float],
                    delete_original_if_empty: bool = True) -> bool:
    """
    Trim any S5P L2 file (2-D or 3-D lat/lon) to bbox (W,S,E,N).
    Writes compressed NetCDF to out_nc. Returns True if a file was written.
    If no Algeria pixels are present, optionally delete the original file.
    """
    W, S, E, N = bbox
    ds, lat_name, lon_name, grp = _open_s5p_for_latlon(in_nc)
    if ds is None:
        return False

    lat = ds[lat_name].values
    lon = _normalize_lon(ds[lon_name].values)

    # Reduce to 2-D if we have a leading time dim of length 1
    if lat.ndim == 3 and lat.shape[0] == 1:
        lat2 = lat[0, :, :]
        lon2 = lon[0, :, :]
        time_dim = ds[lat_name].dims[0]          # 'time'
        row_dim  = ds[lat_name].dims[1]          # 'scanline'
        col_dim  = ds[lat_name].dims[2]          # 'ground_pixel'
        time_indexer = slice(None)               # keep all time (usually 1)
    elif lat.ndim == 2:
        lat2 = lat
        lon2 = lon
        row_dim, col_dim = ds[lat_name].dims
        time_dim = None
        time_indexer = None
    else:
        print(f"    [trim] Unexpected lat/lon shape: {lat.shape}")
        ds.close()
        return False

    inside = (lat2 >= S) & (lat2 <= N) & (lon2 >= W) & (lon2 <= E)
    if not inside.any():
        print("    [trim] No pixels over Algeria.")
        ds.close()
        if delete_original_if_empty:
            try:
                os.remove(in_nc)
                print("    [clean] removed original (outside DZ):", os.path.basename(in_nc))
            except Exception as e:
                print("    [clean] could not remove original:", e)
        return False

    rows = np.where(inside.any(axis=1))[0]
    cols = np.where(inside.any(axis=0))[0]
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    print(f"    [trim] Using group={grp}, lat='{lat_name}', lon='{lon_name}', "
          f"window rows {r0}-{r1}, cols {c0}-{c1}")

    # Build indexers for isel (keep full time if present)
    indexers = {row_dim: slice(r0, r1+1), col_dim: slice(c0, c1+1)}
    if time_dim is not None:
        indexers[time_dim] = time_indexer

    ds_win = ds.isel(indexers)

    # Mask outside-bbox values within the window (for float vars sharing both row/col dims)
    inside_win = inside[r0:r1+1, c0:c1+1]
    for v in list(ds_win.data_vars):
        vv = ds_win[v]
        if row_dim in vv.dims and col_dim in vv.dims and np.issubdtype(vv.dtype, np.floating):
            ds_win[v] = xr.where(inside_win, vv, np.nan)

    # Compression
    encoding = {
        v: dict(zlib=True, complevel=4, shuffle=True)
        for v in ds_win.data_vars
        if np.issubdtype(ds_win[v].dtype, np.floating)
    }

    os.makedirs(os.path.dirname(out_nc), exist_ok=True)
    try:
        ds_win.to_netcdf(out_nc, mode="w", format="NETCDF4", encoding=encoding)
        print(f"    [trim] Wrote {os.path.basename(out_nc)}")
        return True
    except Exception as e:
        print(f"    [trim] Write failed: {e}")
        return False
    finally:
        ds_win.close()
        ds.close()

# =========================
# MAIN
# =========================
def main():
    print("Authenticating to CDSE…")
    token = get_access_token(CDSE_USER, CDSE_PASS)
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

    print("Requesting temporary S3 credentials…")
    creds = make_temp_s3_credentials(headers)
    creds_start = time.time()

    s3_resource = boto3.resource(
        "s3",
        endpoint_url=CONFIG["s3_endpoint_url"],
        aws_access_key_id=creds["access_id"],
        aws_secret_access_key=creds["secret"],
    )
    s3_client = s3_resource.meta.client

    os.makedirs(OUT_DIR, exist_ok=True)

    for prod_label, code in PRODUCT_CODES.items():
        print(f"\n==============================")
        print(f"PRODUCT: {prod_label} ({code}), STREAM={STREAM}")
        print(f"==============================")

        for d in daterange(DATE_START, DATE_END):
            ymd = d.strftime("%Y%m%d")
            day_marker = os.path.join(OUT_DIR, prod_label, ymd[:4], ymd[4:6], f".day_{ymd}.done")
            if MAX_PRODUCTS_PER_DAY == 1 and skip_if_done_marker(day_marker):
                # already processed a DZ-intersecting product this day for this variable
                continue

            # Refresh temp creds if old
            if (time.time() - creds_start) / 60.0 > TEMP_CREDS_REFRESH_MIN:
                print("\nRefreshing temporary S3 credentials…")
                try:
                    delete_temp_s3_credentials(headers, creds["access_id"])
                except Exception:
                    pass
                creds = make_temp_s3_credentials(headers)
                s3_resource = boto3.resource(
                    "s3",
                    endpoint_url=CONFIG["s3_endpoint_url"],
                    aws_access_key_id=creds["access_id"],
                    aws_secret_access_key=creds["secret"],
                )
                s3_client = s3_resource.meta.client
                creds_start = time.time()

            name_sub = f"S5P_{STREAM}_L2__{code}{ymd}"
            print(f"\n[{prod_label}] {ymd} → OData contains(Name,'{name_sub}')")
            recs = odata_query_by_name_contains(name_sub, headers, top=200)
            time.sleep(ODATA_SLEEP_SECONDS)

            if not recs:
                print("  No products found for this day.")
                continue

            # Sort records by their Name (creation time is at the end) for a stable order
            recs = sorted(recs, key=lambda r: r.get("Name", ""))

            print(f"  Found {len(recs)} product(s). Will keep up to {MAX_PRODUCTS_PER_DAY} that hit DZ.")

            kept_today = 0
            for rec in recs:
                if MAX_PRODUCTS_PER_DAY == 1 and kept_today >= 1:
                    break

                name = rec.get("Name")
                s3_path = rec.get("S3Path")
                if not name or not s3_path:
                    print("  ! Missing Name/S3Path, skipping.")
                    continue

                # s3_path like: /eodata/Sentinel-5P/TROPOMI/L2__NO2___/YYYY/MM/DD/S5P_...SAFE/...
                bucket, base_prefix = s3_path.lstrip("/").split("/", 1)

                # Save per product/date under ./datasets/s5p/<PRODUCT>/<YYYY>/<MM>/
                out_dir = os.path.join(OUT_DIR, prod_label, ymd[:4], ymd[4:6])
                ensure_dir(out_dir)

                # Per-product done marker directory
                product_marker_dir = os.path.join(out_dir, f"{name}.files")
                ensure_dir(product_marker_dir)
                product_done_path = os.path.join(product_marker_dir, ".done")
                if skip_if_done_marker(product_done_path):
                    print(f"  [skip] {name} (product .done exists)")
                    kept_today = 1 if MAX_PRODUCTS_PER_DAY == 1 else kept_today
                    continue

                print(f"  Downloading: {name}")
                print(f"    S3:  s3://{bucket}/{base_prefix}")
                nc_keys = list(list_nc_keys(s3_resource, bucket, base_prefix))
                print(f"    .nc files to process: {len(nc_keys)}")

                wrote_any_dz = False
                for key in nc_keys:
                    fname = os.path.basename(key)
                    local_nc = os.path.join(out_dir, fname)
                    local_trim = local_nc.replace(".nc", ".dz.nc")

                    # If trimmed already exists, skip everything for this file
                    if os.path.exists(local_trim):
                        print(f"    [skip trimmed] {os.path.basename(local_trim)}")
                        wrote_any_dz = True
                        continue

                    # 1) Download original .nc (resume-safe)
                    try:
                        did_dl, _ = download_nc_file(s3_client, bucket, key, local_nc)
                        if did_dl:
                            print(f"    [ok] downloaded {fname}")
                    except Exception as e:
                        print(f"    ! Download failed: {fname} ({e})")
                        continue

                    # 2) Trim to Algeria bbox and write .dz.nc
                    try:
                        wrote = trim_nc_to_bbox(local_nc, local_trim, BBOX,
                                                delete_original_if_empty=DELETE_ORIGINAL_AFTER_TRIM)
                        wrote_any_dz = wrote_any_dz or wrote
                    except Exception as e:
                        print(f"    ! Trim failed for {fname}: {e}")
                        wrote = False

                    # 3) Optionally delete original if trimmed written
                    if wrote and DELETE_ORIGINAL_AFTER_TRIM and os.path.exists(local_nc):
                        try:
                            os.remove(local_nc)
                            print(f"    [clean] removed original {fname}")
                        except Exception as e:
                            print(f"    ! Could not remove original {fname}: {e}")

                # Mark product done if all its .nc have corresponding .dz.nc (or were skipped)
                nc_done = True
                for key in nc_keys:
                    fname = os.path.basename(key)
                    if not os.path.exists(os.path.join(out_dir, fname.replace(".nc", ".dz.nc"))):
                        nc_done = False
                        break

                if nc_done:
                    write_done_marker(product_done_path)
                    print(f"    ✔ Completed {name}")
                else:
                    print(f"    ⚠ Incomplete {name} (some .dz.nc missing)")

                # If we produced at least one Algeria-trimmed file, count this day as covered
                if wrote_any_dz and MAX_PRODUCTS_PER_DAY == 1:
                    kept_today = 1
                    # Day-level marker so we skip other products on re-runs
                    write_done_marker(day_marker)
                    print(f"  ✓ Day {ymd} covered for {prod_label} (first DZ-intersecting product).")
                    break  # stop after the first DZ-hit product for this day

            # if none of the products hit DZ, no day marker is written; next run will keep trying

    # Cleanup temp creds
    try:
        delete_temp_s3_credentials(headers, creds["access_id"])
    except Exception:
        pass

    print("\nAll done. Output:", OUT_DIR)

if __name__ == "__main__":
    main()
