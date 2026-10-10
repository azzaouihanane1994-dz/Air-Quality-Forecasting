# save as check_redundant_features.py and run as-is
from pathlib import Path
import hashlib, json
import numpy as np
import csv

ROOT = Path("./datasets/CAMS")

EMIS_X    = ROOT / "CAMS_emission_inventories_ALG_v6_2_2015_2024_X.npy"
EMIS_META = ROOT / "CAMS_emission_inventories_ALG_v6_2_2015_2024_meta.npz"

EAC4_X    = ROOT / "EAC4_ALG_X.npy"
EAC4_META = ROOT / "EAC4_ALG_meta.npz"

REPORT_CSV = ROOT / "redundancy_report.csv"

# sample settings (keeps it fast & RAM-safe)
RNG_SEED = 123
N_ROWS_SAMPLE = 20000   # adjust up/down depending on speed/RAM
CORR_THRESHOLD = 0.999  # near-duplicate if |r| >= this on the sample

def load_meta(p):
    m = np.load(p, allow_pickle=True)
    names = m["feature_names"].astype(str)
    units = m["feature_units"].astype(str) if "feature_units" in m else np.array([""]*names.size, dtype=object)
    longs = m["feature_long_names"].astype(str) if "feature_long_names" in m else names
    # shape_info may be json string inside an object array
    shape = json.loads(str(m["shape_info"]))
    return names, units, longs, shape

def row_indices(shape, frac=1.0, n_rows_sample=N_ROWS_SAMPLE, seed=RNG_SEED):
    n_rows = shape["n_instances"]
    if n_rows_sample >= n_rows:
        return np.arange(n_rows, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, size=n_rows_sample, replace=False).astype(np.int64))

def md5_of_column(sample_vals: np.ndarray) -> str:
    # NaNs normalized to a byte pattern by replacing NaN with a sentinel
    v = sample_vals.copy()
    nan_mask = ~np.isfinite(v)
    if nan_mask.any():
        v[nan_mask] = np.float32(1.2345e30)
    return hashlib.md5(v.tobytes()).hexdigest()

def corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan
    x = a[mask].astype(np.float64)
    y = b[mask].astype(np.float64)
    vx = x - x.mean()
    vy = y - y.mean()
    denom = np.sqrt((vx**2).sum() * (vy**2).sum())
    return float((vx*vy).sum()/denom) if denom > 0 else np.nan

def normalize_name(n: str) -> str:
    # strip prefixes like 'emis:' or 'eac4:', lower-case, drop level suffixes like '@isobaricInhPa=850'
    n = n.lower().strip()
    if ":" in n: n = n.split(":", 1)[1]
    if "@isobaricinhpa=" in n: n = n.split("@isobaricinhpa=", 1)[0]
    if "@level=" in n: n = n.split("@level=", 1)[0]
    if "@step=" in n: n = n.split("@step=", 1)[0]
    return n

def main():
    # --- load metas
    emis_names, emis_units, emis_longs, emis_shape = load_meta(EMIS_META)
    eac4_names, eac4_units, eac4_longs, eac4_shape = load_meta(EAC4_META)

    # --- open memmaps
    ne_r = emis_shape["n_instances"]; ne_c = emis_shape["n_features"]
    ea_r = eac4_shape["n_instances"]; ea_c = eac4_shape["n_features"]
    X_em = np.memmap(EMIS_X, mode="r", dtype=np.float32, shape=(ne_r, ne_c))
    X_ea = np.memmap(EAC4_X, mode="r", dtype=np.float32, shape=(ea_r, ea_c))

    # --- build sample indices (same count; not necessarily same rows/time alignment)
    idx_em = row_indices(emis_shape)
    idx_ea = row_indices(eac4_shape)

    # --- quick name/unit similarity map
    emis_base = [normalize_name(n) for n in emis_names]
    eac4_base = [normalize_name(n) for n in eac4_names]
    name_overlap = set(emis_base).intersection(set(eac4_base))

    # --- compute hashes for exact-duplicate detection on the sample
    print("[info] hashing sampled columns…")
    emis_hash = [md5_of_column(X_em[idx_em, j].astype(np.float32)) for j in range(ne_c)]
    eac4_hash = [md5_of_column(X_ea[idx_ea, j].astype(np.float32)) for j in range(ea_c)]

    # inverted hash maps
    emis_hash_to_cols = {}
    for j, h in enumerate(emis_hash):
        emis_hash_to_cols.setdefault(h, []).append(j)
    eac4_hash_to_cols = {}
    for j, h in enumerate(eac4_hash):
        eac4_hash_to_cols.setdefault(h, []).append(j)

    # exact duplicates across sets (on sample)
    exact_dups_cross = []
    for h, cols in emis_hash_to_cols.items():
        if h in eac4_hash_to_cols:
            for j in cols:
                for k in eac4_hash_to_cols[h]:
                    exact_dups_cross.append(("cross_exact", j, k))

    # optionally check exact duplicates within each set too
    exact_dups_within_emis = [(a,b) for h, cols in emis_hash_to_cols.items() for i,a in enumerate(cols) for b in cols[i+1:]]
    exact_dups_within_eac4 = [(a,b) for h, cols in eac4_hash_to_cols.items() for i,a in enumerate(cols) for b in cols[i+1:]]

    # --- near-duplicate via high correlation (cross only, limited to name-overlap to keep work bounded)
    print("[info] computing correlations for potential overlaps…")
    base_to_emis_idx = {}
    for j, nb in enumerate(emis_base):
        base_to_emis_idx.setdefault(nb, []).append(j)
    base_to_eac4_idx = {}
    for j, nb in enumerate(eac4_base):
        base_to_eac4_idx.setdefault(nb, []).append(j)

    near_dups_cross = []
    for nb in sorted(name_overlap):
        for j in base_to_emis_idx[nb]:
            v_em = X_em[idx_em, j]
            for k in base_to_eac4_idx[nb]:
                v_ea = X_ea[idx_ea, k]
                r = corr(v_em, v_ea)
                if np.isfinite(r) and abs(r) >= CORR_THRESHOLD:
                    near_dups_cross.append(("cross_corr", j, k, r))

    # --- write report
    with REPORT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["type","emis_col","emis_name","emis_units","eac4_col","eac4_name","eac4_units","corr_or_hash"])
        # exact cross
        for _, j, k in exact_dups_cross:
            w.writerow(["cross_exact", j, emis_names[j], emis_units[j], k, eac4_names[k], eac4_units[k], "hash_match"])
        # near duplicates by correlation
        for _, j, k, r in near_dups_cross:
            w.writerow(["cross_corr", j, emis_names[j], emis_units[j], k, eac4_names[k], eac4_units[k], f"{r:.6f}"])
        # within sets (optional rows at end)
        for (a,b) in exact_dups_within_emis:
            w.writerow(["emis_within_exact", a, emis_names[a], emis_units[a], b, emis_names[b], emis_units[b], "hash_match"])
        for (a,b) in exact_dups_within_eac4:
            w.writerow(["eac4_within_exact", a, "", "", b, "", "", "hash_match"])

    # --- console summary
    print("✅ Report written:", REPORT_CSV)
    print(f"   cross exact matches: {len(exact_dups_cross)}")
    print(f"   cross near-duplicates (|r|≥{CORR_THRESHOLD} on sample): {len(near_dups_cross)}")
    print(f"   within emissions exact duplicates: {len(exact_dups_within_emis)}")
    print(f"   within EAC4 exact duplicates: {len(exact_dups_within_eac4)}")

if __name__ == "__main__":
    main()
