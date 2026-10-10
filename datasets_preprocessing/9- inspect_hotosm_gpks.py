#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_humdata_gpkgs.py

Scans ./datasets/humdata for GeoPackages and writes markdown "cards" describing:
- File-level info (path, size)
- Layer list, feature counts, CRS, bounds
- Schema overview (column -> dtype), with small value-count samples for common fields
- A short "semantics" hint (roads/buildings/places) based on known columns

Outputs (created if missing):
  ./datasets/humdata/cards/
    - index.card.md
    - <file>.card.md  (one per GeoPackage)

Dependencies:
  pip install geopandas fiona pyogrio shapely
  (pyogrio is optional but makes metadata queries much faster)
"""

from pathlib import Path
import os
import sys
import textwrap
import json
from typing import Dict, Any, List, Tuple, Optional

# Prefer pyogrio for fast metadata access if available
_HAS_PYOGRIO = True
try:
    from pyogrio import read_info, list_layers, read_dataframe
except Exception:
    _HAS_PYOGRIO = False

import geopandas as gpd
import fiona
import pandas as pd
import numpy as np

# ------------------------- CONFIG -------------------------
ROOT = Path("./datasets/humdata")
OUT_DIR = ROOT / "cards"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Limit heavy operations
HEAD_ROWS = 10             # read this many rows for sample schema/value preview
TOP_VALUE_COUNTS = 8       # show up to N most common values for key columns
KEY_COL_GUESSES = [
    # roads/transport
    "highway", "surface", "lanes", "bridge", "tunnel", "oneway",
    # buildings/landuse
    "building", "amenity", "landuse",
    # places/population
    "name", "place", "population", "is_in", "admin_level",
    # admin/codes
    "iso3", "iso2", "insee", "wikidata",
]
# ----------------------------------------------------------


def human_size(bytes_: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    s = float(bytes_)
    for u in units:
        if s < 1024:
            return f"{s:.2f} {u}"
        s /= 1024
    return f"{s:.2f} PB"


def safe_list_layers(path: Path) -> list[str]:
    """
    Return a plain Python list of layer names for a GeoPackage, using pyogrio if present.
    Handles pyogrio returning a tuple, a DataFrame, or an array-like.
    """
    # Try pyogrio first
    if _HAS_PYOGRIO:
        try:
            info = list_layers(path)
            # pyogrio may return (names, ...) or a DataFrame with "name" column, or just an array-like
            if isinstance(info, tuple) and len(info) > 0:
                names = info[0]
            elif hasattr(info, "columns"):          # DataFrame-like
                names = info["name"].tolist()
            else:
                names = info

            # Flatten to a plain list[str]
            if isinstance(names, (list, tuple)):
                return [str(x) for x in names]
            try:
                import numpy as np
                return [str(x) for x in np.array(names).ravel().tolist()]
            except Exception:
                return [str(names)]
        except Exception as e:
            print(f"[warn] pyogrio.list_layers failed for {path.name}: {e}")

    # Fallback: Fiona
    try:
        return [str(x) for x in fiona.listlayers(path)]
    except Exception as e:
        print(f"[warn] Fiona listlayers failed for {path.name}: {e}")
        return []


def layer_info_fast(path: Path, layer: str) -> Dict[str, Any]:
    """
    Try to get cheap metadata (feature count, CRS, geometry type, bounds) without loading all features.
    Falls back to reading a small head with GeoPandas if needed.
    """
    info: Dict[str, Any] = {}
    # pyogrio fast path
    if _HAS_PYOGRIO:
        try:
            meta = read_info(path, layer=layer)
            info["feature_count"] = int(meta["features"])
            info["crs"] = str(meta.get("crs", "")) or "unknown"
            info["geometry_type"] = str(meta.get("geometry_type", ""))
            # bounds (if present)
            if meta.get("bbox") is not None:
                minx, miny, maxx, maxy = meta["bbox"]
                info["bounds"] = (float(minx), float(miny), float(maxx), float(maxy))
            return info
        except Exception as e:
            print(f"[warn] pyogrio read_info failed for {path.name}:{layer}: {e}")

    # fiona fallback (driver-level)
    try:
        with fiona.open(path, layer=layer) as src:
            info["feature_count"] = len(src)  # Fiona provides __len__ efficiently for many drivers
            info["crs"] = str(src.crs) if src.crs else "unknown"
            info["geometry_type"] = src.schema.get("geometry", "")
            try:
                b = src.bounds
                info["bounds"] = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
            except Exception:
                pass
            return info
    except Exception as e:
        print(f"[warn] Fiona fallback failed for {path.name}:{layer}: {e}")

    # last resort: read small head with GeoPandas
    try:
        gdf = gpd.read_file(path, layer=layer, rows=slice(0, min(HEAD_ROWS, 50)))
        info["feature_count"] = None
        info["crs"] = str(gdf.crs) if gdf.crs else "unknown"
        info["geometry_type"] = gdf.geom_type.unique().tolist() if "geometry" in gdf else ""
        try:
            info["bounds"] = tuple(map(float, gdf.total_bounds))
        except Exception:
            pass
        return info
    except Exception as e:
        print(f"[error] Could not read layer at all for {path.name}:{layer}: {e}")
        return {"error": str(e)}


def sample_schema_and_values(path: Path, layer: str) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    """
    Return (schema_df, value_counts) from a small sample (HEAD_ROWS).
    schema_df: DataFrame with columns: name, dtype, example_non_null
    value_counts: dict(col -> Series) for a subset of relevant columns
    """
    try:
        if _HAS_PYOGRIO:
            gdf = read_dataframe(path, layer=layer, max_features=HEAD_ROWS)
        else:
            gdf = gpd.read_file(path, layer=layer, rows=slice(0, HEAD_ROWS))
    except Exception as e:
        print(f"[warn] sample read failed for {path.name}:{layer}: {e}")
        return pd.DataFrame(columns=["name", "dtype", "example_non_null"]), {}

    # Build schema table
    cols = []
    for c in gdf.columns:
        if c == "geometry":
            continue
        ser = gdf[c]
        example = None
        non_null = ser[ser.notna()]
        if not non_null.empty:
            example = non_null.iloc[0]
            if isinstance(example, str) and len(example) > 80:
                example = example[:77] + "..."
        cols.append({"name": c, "dtype": str(ser.dtype), "example_non_null": example})
    schema_df = pd.DataFrame(cols).sort_values("name").reset_index(drop=True)

    # Value counts for a few interesting columns
    vcounts: Dict[str, pd.Series] = {}
    lower_names = {c.lower(): c for c in gdf.columns if c != "geometry"}
    for k in KEY_COL_GUESSES:
        if k in lower_names:
            col = lower_names[k]
            try:
                vc = gdf[col].astype("string").str.slice(0, 64).value_counts(dropna=True).head(TOP_VALUE_COUNTS)
                vcounts[col] = vc
            except Exception:
                pass

    return schema_df, vcounts


def semantics_hint(layer_name: str, column_names: List[str]) -> str:
    """
    Heuristic label for quick human scan.
    """
    low = layer_name.lower()
    cols = [c.lower() for c in column_names]
    if "road" in low or "transport" in low or "highway" in cols:
        return "Road network / transport"
    if "build" in low or "building" in cols:
        return "Buildings / built-up"
    if "place" in low or "populated" in low or "population" in cols:
        return "Places / settlements (points)"
    if "admin" in low or "boundary" in cols:
        return "Administrative / boundaries"
    return "General OSM layer"


def write_layer_card(md_path: Path, file_path: Path, layer: str, meta: Dict[str, Any],
                     schema_df: pd.DataFrame, vcounts: Dict[str, pd.Series]) -> None:
    md = []
    md.append(f"# {file_path.name} — Layer: `{layer}`")
    md.append("")
    if "error" in meta:
        md.append(f"**Error reading layer**: {meta['error']}")
        md_path.write_text("\n".join(md), encoding="utf-8")
        return

    size = human_size(file_path.stat().st_size)
    md.append(f"- **File**: `{file_path}`  •  **Size**: {size}")
    md.append(f"- **Feature count**: {meta.get('feature_count', 'unknown')}")
    md.append(f"- **CRS**: `{meta.get('crs', 'unknown')}`")
    md.append(f"- **Geometry**: `{meta.get('geometry_type', '')}`")
    if "bounds" in meta and meta["bounds"]:
        xmin, ymin, xmax, ymax = meta["bounds"]
        md.append(f"- **Bounds**: xmin={xmin:.4f}, ymin={ymin:.4f}, xmax={xmax:.4f}, ymax={ymax:.4f}")
    md.append(f"- **Semantics**: {semantics_hint(layer, schema_df['name'].tolist())}")
    md.append("")
    md.append("## Schema (sampled)")
    if schema_df.empty:
        md.append("_No attributes (or failed to read sample)._")
    else:
        md.append("| name | dtype | example |")
        md.append("|---|---|---|")
        for _, r in schema_df.iterrows():
            ex = str(r["example_non_null"]) if pd.notna(r["example_non_null"]) else ""
            ex = ex.replace("\n", " ")
            md.append(f"| `{r['name']}` | `{r['dtype']}` | {ex} |")

    if vcounts:
        md.append("")
        md.append("## Value samples")
        for col, vc in vcounts.items():
            md.append(f"**{col}**")
            md.append("")
            md.append("```")
            md.append(str(vc))
            md.append("```")

    md_path.write_text("\n".join(md), encoding="utf-8")


def write_file_card(gpkg_path: Path, layer_cards: List[Tuple[str, Path]]) -> Path:
    md = []
    md.append(f"# {gpkg_path.name}")
    md.append("")
    md.append(f"- **Path**: `{gpkg_path}`  •  **Size**: {human_size(gpkg_path.stat().st_size)}")
    md.append(f"- **Layers**: {len(layer_cards)}")
    md.append("")
    if layer_cards:
        md.append("## Layers")
        for layer, card_path in layer_cards:
            md.append(f"- `{layer}` → see: `{card_path.name}`")
    out = OUT_DIR / f"{gpkg_path.stem}.card.md"
    md.append("")
    out.write_text("\n".join(md), encoding="utf-8")
    return out


def write_index_card(file_cards: List[Tuple[Path, Path]]) -> None:
    md = []
    md.append("# Humdata GeoPackages — Index")
    md.append("")
    if not file_cards:
        md.append("_No GeoPackages found in ./datasets/humdata/_")
    else:
        md.append("## Files")
        for gpkg, fcard in file_cards:
            md.append(f"- `{gpkg.name}` → `{fcard.name}`")
    (OUT_DIR / "index.card.md").write_text("\n".join(md), encoding="utf-8")


def main():
    # Find GeoPackages (handle both .gpkg and odd double extensions like .gpkg.gpkg)
    files = sorted([p for p in ROOT.glob("**/*") if p.is_file() and (p.suffix.lower() == ".gpkg" or p.name.lower().endswith(".gpkg.gpkg"))])
    if not files:
        print(f"[info] No GeoPackage files found under {ROOT.resolve()}")
        write_index_card([])
        return

    file_cards: List[Tuple[Path, Path]] = []
    for gpkg in files:
        print(f"\n[scan] {gpkg.name}")
        layers = safe_list_layers(gpkg)
        if layers is None or len(layers) == 0:
            print(f"  [warn] no layers in {gpkg.name}")
            continue


        per_layer_cards: List[Tuple[str, Path]] = []
        for lyr in layers:
            print(f"  [layer] {lyr} …")
            meta = layer_info_fast(gpkg, lyr)
            schema_df, vcounts = sample_schema_and_values(gpkg, lyr)

            out_layer_card = OUT_DIR / f"{gpkg.stem}__{lyr.replace(':','_').replace('/','-')}.card.md"
            write_layer_card(out_layer_card, gpkg, lyr, meta, schema_df, vcounts)
            per_layer_cards.append((lyr, out_layer_card))
            print(f"    ↳ wrote {out_layer_card.name}")

        file_card_path = write_file_card(gpkg, per_layer_cards)
        file_cards.append((gpkg, file_card_path))
        print(f"  [file-card] {file_card_path.name}")

    write_index_card(file_cards)
    print(f"\n✅ Done. Cards written to: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
