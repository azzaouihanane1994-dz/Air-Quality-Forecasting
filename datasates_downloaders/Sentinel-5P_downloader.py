# pip install pystac-client requests tqdm pandas python-dateutil
# Sentinel-5P (S5P) Level-2 downloader for Copernicus Data Space STAC
# Saves to ./datasets/s5p/<PRODUCT>/<YYYY>/<MM>/file.nc + manifest.csv
# Products: NO2, CO, SO2, HCHO, AER_AI (AAI), CLOUD
# Requirements: pip install pystac-client requests tqdm python-dateutil pandas

# Sentinel-5P (S5P) Level-2 downloader for Copernicus Data Space (CDSE)
# Saves to ./datasets/s5p/<PRODUCT>/<YYYY>/<MM>/file.nc + manifest.csv
# Products: NO2, CO, SO2, HCHO, AER_AI (AAI), CLOUD

# Terminal
# setx CDSE_USER "khaldi.yacine@ens-ouargla.dz"
# setx CDSE_PASS "7&UvrTyvv8kZVxp"
# setx CDSE_S3_KEY "632F4HDCSLD1PYKWXAWE"
# setx CDSE_S3_SECRET "1G657Q6Nc37RqJtX1ouukTQMoVadfnGkklFmvosr"

from pystac_client import os
import Client
import boto3, botocore
import requests, os, csv, sys
from datetime import datetime, timedelta, timezone
from dateutil import rrule
from urllib.parse import urlparse
from tqdm import tqdm

# =======================
# CONFIG
# =======================
STAC_URL   = "https://stac.dataspace.copernicus.eu/v1/"
BBOX       = [-9, 18, 12, 38]               # Algeria [W, S, E, N]
DATE_START = "2018-01-01"
DATE_END   = datetime.now(timezone.utc).date().isoformat()

OUT_DIR    = "./datasets/s5p"
PREFER_NRTI = False                         # False -> prefer OFFL (recommended), True -> prefer NRTI
CHUNK      = 1 << 20                        # only used for rare HTTPS fallback

# Products to fetch (match by substring in collection id)
PRODUCT_TOKENS = {
    "NO2"   : ["no2"],                                  # tropospheric NO2
    "CO"    : ["co"],                                   # carbon monoxide
    "SO2"   : ["so2"],                                  # sulfur dioxide
    "HCHO"  : ["hcho"],                                 # formaldehyde
    "AER_AI": ["aer_ai","aai","aerosol_index","uvai"],  # UV aerosol index
    "CLOUD" : ["cloud"],                                # cloud product
}

# ---- Hardcoded credentials (as requested) ----
CDSE_USER = os.environ.get("CDSE_USER", "")
CDSE_PASS = os.environ.get("CDSE_PASS", "")
CDSE_S3_KEY = os.environ.get("CDSE_S3_KEY", "")
CDSE_S3_SECRET = os.environ.get("CDSE_S3_SECRET", "")

# Print every found item’s EO product name (filename)
PRINT_NAMES = True

# =======================
# Helpers
# =======================
def open_stac(url: str) -> Client:
    try:
        return Client.open(url)
    except Exception:
        return Client.open(url, ignore_conformance=True)

def month_windows(start_iso: str, end_iso: str):
    s0 = datetime.fromisoformat(start_iso[:10])
    e0 = datetime.fromisoformat(end_iso[:10])
    s = s0.replace(day=1); e = e0.replace(day=1)
    for dt in rrule.rrule(rrule.MONTHLY, dtstart=s, until=e):
        m_start = max(s0, dt)
        nxt = (dt.replace(day=28) + timedelta(days=4)).replace(day=1)
        m_end = min(e0, nxt - timedelta(days=1))
        yield m_start.date().isoformat(), m_end.date().isoformat(), dt

def pick_collections(client: Client, prefer_nrti: bool):
    ids = [c.id for c in client.get_collections()]
    lowers = [cid.lower() for cid in ids]

    def rank(cid_lower: str):
        is_nrti = ("nrti" in cid_lower) or ("nrt" in cid_lower)
        is_offl = ("offl" in cid_lower)
        if prefer_nrti:
            return (0 if is_nrti else 1, cid_lower)
        return (0 if is_offl else (1 if is_nrti else 2), cid_lower)

    prod2coll = {}
    for prod, toks in PRODUCT_TOKENS.items():
        cands = []
        for cid in lowers:
            if "sentinel-5p" in cid and "l2" in cid:
                if any(tok in cid for tok in [t.lower() for t in toks]):
                    cands.append(cid)
        prod2coll[prod] = sorted(cands, key=rank)[0] if cands else None
    return prod2coll

def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

def make_s3_client():
    if not CDSE_S3_KEY or not CDSE_S3_SECRET:
        sys.exit("❌ Missing S3 keys. Edit CDSE_S3_KEY / CDSE_S3_SECRET at top of script.")
    return boto3.client(
        "s3",
        endpoint_url="https://eodata.dataspace.copernicus.eu",
        aws_access_key_id=CDSE_S3_KEY,
        aws_secret_access_key=CDSE_S3_SECRET,
        config=botocore.config.Config(signature_version="s3v4"),
    )

def parse_s3(href: str):
    if not href or not href.startswith("s3://eodata/"):
        return None, None
    bucket = "eodata"
    key = href.replace("s3://eodata/", "")
    return bucket, key

# =======================
# Main
# =======================
def main():
    client = open_stac(STAC_URL)
    ensure_dir(OUT_DIR)

    prod2coll = pick_collections(client, PREFER_NRTI)
    for k, v in prod2coll.items():
        print(f"[{'OK' if v else 'WARN'}] {k} -> {v}")

    if not prod2coll.get("AER_AI"):
        avail = [c.id for c in client.get_collections() if "sentinel-5p" in c.id.lower() and "l2" in c.id.lower()]
        print("\n(No AER_AI match) Available S5P L2 collections:\n", avail)

    sess = requests.Session()  # https fallback (rare)
    sess.auth = (CDSE_USER, CDSE_PASS)
    sess.headers.update({"User-Agent": "s5p-downloader/4.1"})

    s3c = make_s3_client()

    manifest = os.path.join(OUT_DIR, "manifest.csv")
    if not os.path.exists(manifest):
        with open(manifest, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["product","collection_id","datetime","href","local_path","bytes","status"])

    total_new = 0
    for m_start, m_end, dt in month_windows(DATE_START, DATE_END):
        for prod, coll in prod2coll.items():
            if not coll:
                continue
            print(f"\nSearching {prod} ({coll}) {m_start}..{m_end}")
            search = client.search(collections=[coll], bbox=BBOX, datetime=f"{m_start}/{m_end}")
            items = list(search.items())
            if not items:
                print("  No items.")
                continue

            subdir = os.path.join(OUT_DIR, prod, dt.strftime("%Y"), dt.strftime("%m"))
            ensure_dir(subdir)

            # Print all names up-front if requested
            if PRINT_NAMES:
                print("  Found items:")
                for it in items:
                    asset = it.assets.get("data") or it.assets.get("product") or next(iter(it.assets.values()))
                    name = os.path.basename(urlparse(asset.href).path)
                    print(f"   - {name}")

            for it in tqdm(items, desc=f"{prod} {dt.strftime('%Y-%m')}"):
                asset = it.assets.get("data") or it.assets.get("product") or next(iter(it.assets.values()))
                href = asset.href
                fname = os.path.basename(urlparse(href).path)
                local = os.path.join(subdir, fname)

                # show per-file action
                bucket, key = parse_s3(href)
                via = "S3" if bucket else "HTTPS"

                if os.path.exists(local):
                    # already present
                    with open(manifest, "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow([prod, coll, it.datetime.strftime("%Y-%m-%dT%H:%M:%SZ"), href, local, os.path.getsize(local), "skip_exists"])
                    continue

                status = "ok"
                try:
                    if bucket and key:
                        ensure_dir(os.path.dirname(local))
                        s3c.download_file(bucket, key, local)
                    else:
                        with sess.get(href, stream=True) as r:
                            r.raise_for_status()
                            with open(local, "wb") as f:
                                for chunk in r.iter_content(chunk_size=CHUNK):
                                    if chunk: f.write(chunk)
                    total_new += 1
                except botocore.exceptions.ClientError as e:
                    print(f"  ! S3 error ({via}): {e}")
                    status = f"s3_error:{e.response.get('Error',{}).get('Code','')}"
                except requests.HTTPError as e:
                    print(f"  ! HTTP error ({via}): {e}")
                    status = f"http_error:{e.response.status_code}"
                except Exception as e:
                    print(f"  ! Download failed ({via}): {e}")
                    status = f"error:{type(e).__name__}"

                size = os.path.getsize(local) if os.path.exists(local) else 0
                with open(manifest, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([prod, coll, it.datetime.strftime("%Y-%m-%dT%H:%M:%SZ"), href, local, size, status])

    print(f"\n✅ Done. Saved to {OUT_DIR}. Newly downloaded files: {total_new}")

if __name__ == "__main__":
    main()