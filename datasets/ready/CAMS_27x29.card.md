# Merged Daily Dataset on EAC4 Grid (27×29)

## Files
- **4D**: `datasets\CAMS\merged_27x29_4d.npy`  — shape `(days=3653, features=36, y=27, x=29)`
- **Flat**: `datasets\CAMS\merged_27x29_flat.npy`  — shape `(days*27*29, features=36)`
- **Meta**: `datasets\CAMS\merged_27x29_meta.npz`

## Time & Grid
- Days: `3653` (2015-01-01 → 2024-12-31)
- Grid: `27 × 29` (EAC4 Algeria subgrid)

## Features
- EAC4 features: `27`
- Emissions kept (after dedupe): `9`
- **Merged total**: `36`

## Notes
- Emissions → EAC4 grid via bilinear interpolation (not strictly conservative).
- If emissions were monthly, daily expansion uses equal weights within each month.
- Duplicated variables by base-name were removed from emissions (EAC4 has priority).
- Approx raw 4D size (float32): ~0.38 GB.
