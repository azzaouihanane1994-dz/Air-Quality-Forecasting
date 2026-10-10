#!/usr/bin/env python3
# pip install xarray netCDF4 numpy

import xarray as xr
import numpy as np
from pathlib import Path

# >>> point this to your file
NC = Path("./datasates_downloaders/S5P_OFFL_L2__NO2____20180629T114634_20180629T132804_03676_01_010002_20180705T134204.nc")

def show_dataset_info(ds, title):
    print("\n" + "="*80)
    print(title)
    print("="*80)
    # Dimensions
    print("\nDimensions:")
    for k, v in ds.dims.items():
        print(f"  - {k}: {int(v)}")

    # Variables (first 20)
    print("\nVariables (first 20):")
    names = list(ds.variables)
    for k in names[:20]:
        v = ds[k]
        print(f"  - {k}: shape {tuple(v.shape)}, dtype {str(v.dtype)}, dims {v.dims}")

    # Global attributes (a few common ones)
    print("\nGlobal attributes (subset):")
    for a in ["title", "institution", "processor_version", "processing_mode",
              "time_coverage_start", "time_coverage_end", "orbit"]:
        if a in ds.attrs:
            print(f"  - {a}: {ds.attrs[a]}")

def try_key(ds, keys):
    # return first variable name that exists (case-insensitive) from keys
    lower = {k.lower(): k for k in ds.variables}
    for cand in keys:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None

def main():
    print(f"Opening: {NC.name}")

    # 1) ROOT
    ds_root = xr.open_dataset(NC, engine="netcdf4", mask_and_scale=False)
    show_dataset_info(ds_root, "ROOT GROUP")

    # 2) PRODUCT group (where geolocation + data usually live)
    try:
        ds_prod = xr.open_dataset(NC, group="PRODUCT", engine="netcdf4", mask_and_scale=False)
        show_dataset_info(ds_prod, "PRODUCT GROUP")
    except Exception as e:
        print("\n(No PRODUCT group found or cannot open it:", e, ")")
        ds_prod = None

    # 3) Try to extract key fields from PRODUCT (if available)
    if ds_prod is not None:
        print("\nKey fields (PRODUCT):")
        lat_name = try_key(ds_prod, ["latitude", "lat"])
        lon_name = try_key(ds_prod, ["longitude", "lon"])
        qa_name  = try_key(ds_prod, ["qa_value", "qa_value_no2", "qa_value_cloud"])
        no2_name = try_key(ds_prod, [
            "nitrogendioxide_tropospheric_column",
            "nitrogendioxide_total_column",
            "no2_column_number_density"
        ])

        print(f"  latitude var : {lat_name}")
        print(f"  longitude var: {lon_name}")
        print(f"  qa var       : {qa_name}")
        print(f"  NO2 var      : {no2_name}")

        # Show simple stats for each if present (using a small sample to be quick)
        def qstats(arr, name):
            a = np.asarray(arr).ravel()
            a = a[np.isfinite(a)]
            if a.size:
                q = np.quantile(a, [0, 0.25, 0.5, 0.75, 1.0])
                print(f"    {name} stats -> min {q[0]:.6g}, Q1 {q[1]:.6g}, med {q[2]:.6g}, Q3 {q[3]:.6g}, max {q[4]:.6g}, n={a.size}")
            else:
                print(f"    {name} stats -> all NaN")

        if lat_name and lon_name:
            print("\n  Geolocation sample:")
            qstats(ds_prod[lat_name].isel({ds_prod[lat_name].dims[0]: slice(0, 50)}), "latitude")
            qstats(ds_prod[lon_name].isel({ds_prod[lon_name].dims[0]: slice(0, 50)}), "longitude")

        if qa_name:
            print("\n  QA sample:")
            qstats(ds_prod[qa_name].isel({ds_prod[qa_name].dims[0]: slice(0, 50)}), qa_name)

        if no2_name:
            print("\n  NO2 column sample:")
            qstats(ds_prod[no2_name].isel({ds_prod[no2_name].dims[0]: slice(0, 50)}), no2_name)

        ds_prod.close()

    ds_root.close()
    print("\nDone.")

if __name__ == "__main__":
    main()
