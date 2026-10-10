#!/usr/bin/env python3
# CDSE OData → temporary S3 credentials → S3 download (no CLI args, creds hardcoded)

import os
import json
import time
import requests
import boto3
from tqdm import tqdm

# =========================
# CONFIG (edit these)
# =========================
# Your CDSE portal login (for OAuth token & temp S3 keys)
CDSE_USER = os.environ.get("CDSE_USER", "")
CDSE_PASS = os.environ.get("CDSE_PASS", "")

# Output root
OUT_DIR = "./datasets/s5p"

# What to download: exact name or a unique substring of the EO product "Name"
# Example (exact): "S5P_OFFL_L2__NO2____20240101T110759_20240101T124929_32221_03_020600_20240103T033227.nc"
# Example (partial, first match will be chosen): "S5P_OFFL_L2__NO2____20240101"
EO_PRODUCT_NAME_QUERY = "S5P_OFFL_L2__AER_AI_20180630"

# If your query returns multiple products, take the first automatically
TAKE_FIRST_MATCH = True  # True = auto-pick first; False = raise if multiple

# CDSE endpoints (per official docs)
CONFIG = {
    "auth_server_url": "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
    "odata_base_url": "https://catalogue.dataspace.copernicus.eu/odata/v1/Products",
    "s3_endpoint_url": "https://eodata.dataspace.copernicus.eu",
    "tmp_keys_api":   "https://s3-keys-manager.cloudferro.com/api/user/credentials",
    "tmp_keys_delete":"https://s3-keys-manager.cloudferro.com/api/user/credentials/access_id/{access_id}",
}

# =========================
# Helpers
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

def odata_query_by_name(name_query: str, headers: dict) -> list:
    """Try exact Name first, then contains(Name, ...) as fallback."""
    base = CONFIG["odata_base_url"]

    # Exact
    url_eq = f"{base}?$filter=Name eq '{name_query}'&$top=50"
    r = requests.get(url_eq, headers=headers)
    if r.status_code != 200:
        raise SystemExit(f"OData error (HTTP {r.status_code}): {r.text[:300]}")
    vals = r.json().get("value", [])
    if vals:
        return vals

    # Contains
    url_like = f"{base}?$filter=contains(Name,'{name_query}')&$top=50"
    r = requests.get(url_like, headers=headers)
    if r.status_code != 200:
        raise SystemExit(f"OData error (HTTP {r.status_code}): {r.text[:300]}")
    return r.json().get("value", [])

def choose_one(records: list, take_first: bool) -> dict:
    if not records:
        raise SystemExit("No matching products found for the given Name query.")
    if len(records) == 1 or take_first:
        return records[0]
    # If not taking first automatically, show a few and exit so user can refine query
    print(f"Multiple matches ({len(records)}). First 10:")
    for i, rec in enumerate(records[:10], start=1):
        print(f"  [{i}] {rec.get('Name','<no Name>')}")
    raise SystemExit("Refine EO_PRODUCT_NAME_QUERY to make it unique, or set TAKE_FIRST_MATCH=True.")

def make_temp_s3_credentials(headers: dict) -> dict:
    r = requests.post(CONFIG["tmp_keys_api"], headers=headers)
    if r.status_code != 200:
        raise SystemExit(f"Failed to create temporary S3 credentials (HTTP {r.status_code}): {r.text[:300]}")
    creds = r.json()
    print("Temporary S3 credentials created.")
    return creds

def delete_temp_s3_credentials(headers: dict, access_id: str):
    url = CONFIG["tmp_keys_delete"].format(access_id=access_id)
    r = requests.delete(url, headers=headers)
    if r.status_code == 204:
        print("Temporary S3 credentials deleted.")
    else:
        print(f"Failed to delete temporary S3 credentials (HTTP {r.status_code}).")

def format_filename(filename: str, length: int = 40) -> str:
    return (filename[: length - 3] + "...") if len(filename) > length else filename.ljust(length)

def download_file_s3(s3_client, bucket: str, key: str, local_path: str, failed: list):
    try:
        head = s3_client.head_object(Bucket=bucket, Key=key)
        size = head["ContentLength"]

        # Resume-safe: skip if already present at same size
        if os.path.exists(local_path) and os.path.getsize(local_path) == size:
            return

        disp = format_filename(os.path.basename(local_path))
        with tqdm(
            total=size, unit="B", unit_scale=True, desc=disp, ncols=80,
            bar_format="{desc:.40}|{bar:20}| {percentage:3.0f}% {n_fmt}/{total_fmt}B"
        ) as pbar:
            def callback(bytes_transferred):
                pbar.update(bytes_transferred)

            s3_client.download_file(bucket, key, local_path, Callback=callback)
    except Exception as e:
        print(f"Failed to download {key}: {e}")
        failed.append(key)

def traverse_and_download_s3(s3_resource, bucket: str, base_prefix: str, out_dir: str, failed: list):
    bucket_obj = s3_resource.Bucket(bucket)
    objs = bucket_obj.objects.filter(Prefix=base_prefix)
    for obj in objs:
        s3_key = obj.key
        rel = os.path.relpath(s3_key, base_prefix)
        local_file = os.path.join(out_dir, rel)
        os.makedirs(os.path.dirname(local_file), exist_ok=True)
        download_file_s3(s3_resource.meta.client, bucket, s3_key, local_file, failed)

# =========================
# MAIN
# =========================
def main():
    # 1) OAuth token
    token = get_access_token(CDSE_USER, CDSE_PASS)
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

    # 2) Find product(s) by Name (exact → fallback contains)
    records = odata_query_by_name(EO_PRODUCT_NAME_QUERY, headers)
    chosen = choose_one(records, TAKE_FIRST_MATCH)
    name = chosen.get("Name")
    s3_path = chosen.get("S3Path")
    if not s3_path:
        raise SystemExit("Chosen product has no S3Path in OData response.")

    # 3) Temp S3 creds
    creds = make_temp_s3_credentials(headers)
    time.sleep(3)  # small delay to ensure keys are active

    # 4) S3 setup
    bucket_name, base_prefix = s3_path.lstrip("/").split("/", 1)
    out_root = os.path.join(OUT_DIR, name)
    os.makedirs(out_root, exist_ok=True)

    s3_resource = boto3.resource(
        "s3",
        endpoint_url=CONFIG["s3_endpoint_url"],
        aws_access_key_id=creds["access_id"],
        aws_secret_access_key=creds["secret"],
    )

    # 5) Download all files under the product prefix
    print(f"Downloading product:\n  Name: {name}\n  S3:   s3://{bucket_name}/{base_prefix}\n  Out:  {out_root}")
    failed = []
    traverse_and_download_s3(s3_resource, bucket_name, base_prefix, out_root, failed)

    # 6) Report + cleanup temp key
    if not failed:
        print("✅ Product download complete.")
    else:
        print("⚠️ Product download incomplete; failed files:")
        for k in failed:
            print(" -", k)

    delete_temp_s3_credentials(headers, creds["access_id"])

if __name__ == "__main__":
    main()
