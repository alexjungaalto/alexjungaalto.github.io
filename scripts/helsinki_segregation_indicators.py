"""
helsinki_tenure_map.py
======================
Three-panel choropleth of socio-spatial indicators across Helsinki
sub-areas (osa-alue).

Indicators:
  (A) ARA share              -- subsidised rental share of dwellings.
                                Source: Aluesarjat alu_askan_005r.
  (B) Foreign-language share -- residents whose mother tongue is
                                "Muu kieli" (i.e. not Finnish/Swedish
                                /Sami) divided by total population.
                                Source: Aluesarjat alu_vaerak_004p.
  (C) Low-income share       -- household share in the lowest income
                                quintile of the Helsinki region (HS).
                                Source: Aluesarjat alu_astul_011y.

Note on toimeentulotuki (social-assistance recipients):
  This figure is not published on Aluesarjat. It lives in Kela /
  THL Sotkanet at municipal level only; an area-level breakdown
  would require microdata access. Left out here, but mentioned for
  honesty.

Requirements:
    pip install pandas geopandas matplotlib requests numpy
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import requests


# =====================================================================
# Constants
# =====================================================================

PX_BASE = "https://stat.hel.fi/api/v1/fi/Aluesarjat"
WFS_URL = ("https://kartta.hel.fi/ws/geoserver/avoindata/wfs"
           "?service=WFS&version=1.0.0&request=GetFeature"
           "&typeName=avoindata:Piirijako_osaalue"
           "&outputFormat=json&srsName=EPSG:4326")
TARGET_YEAR = "2023"           # latest year with all 3 tables populated
MIN_DWELLINGS = 500            # filter to residential areas only
GEO_CACHE = Path("helsinki_osaalue.geojson")
OUT_PNG = Path("helsinki_segregation_indicators.png")


# =====================================================================
# Generic PxWeb fetch helper
# =====================================================================

def fetch_pxweb(table_path: str, query_body: dict) -> pd.DataFrame:
    """POST a PxWeb JSON query and unfold the JSON-stat2 response."""
    url = f"{PX_BASE}/{table_path}"
    r = requests.post(url, json=query_body, timeout=60)
    r.raise_for_status()
    js = r.json()

    dim_ids = js["id"]
    dimension = js["dimension"]
    dim_codes: dict[str, list[tuple[str, str]]] = {}
    for d in dim_ids:
        cat = dimension[d]["category"]
        index = cat["index"]
        labels = cat["label"]
        if isinstance(index, dict):
            codes = sorted(index, key=lambda k: index[k])
        else:
            codes = list(index)
        dim_codes[d] = [(c, labels.get(c, c)) for c in codes]

    rows = []
    for i, combo in enumerate(product(*[dim_codes[d] for d in dim_ids])):
        row = {}
        for d, (code, label) in zip(dim_ids, combo):
            row[d] = label
            row[f"{d}_code"] = code
        v = js["value"][i]
        row["value"] = None if v in (None, -1) else v
        rows.append(row)
    return pd.DataFrame(rows)


# =====================================================================
# Per-indicator fetchers, each returning a Series keyed by area code
# =====================================================================

def fetch_total_dwellings() -> pd.Series:
    """Total dwellings per Helsinki osa-alue (residential mass)."""
    q = {
        "query": [
            {"code": "Osa-alue",
             "selection": {"filter": "all", "values": ["*"]}},
            {"code": "Huoneistotyyppi",
             "selection": {"filter": "item", "values": ["ALL"]}},
            {"code": "Hallintaperuste",
             "selection": {"filter": "item", "values": ["ALL"]}},
            {"code": "Vuosi",
             "selection": {"filter": "item", "values": [TARGET_YEAR]}},
        ],
        "response": {"format": "json-stat2"},
    }
    df = fetch_pxweb("asu/askan/alu_askan_005r.px", q)
    df = df[df["Osa-alue_code"].str.startswith("091")]
    return df.set_index("Osa-alue_code")["value"].rename("dwellings")


def fetch_ara_share() -> pd.Series:
    """ARA share = (ARA dwellings) / (all dwellings) per osa-alue."""
    q = {
        "query": [
            {"code": "Osa-alue",
             "selection": {"filter": "all", "values": ["*"]}},
            {"code": "Huoneistotyyppi",
             "selection": {"filter": "item", "values": ["ALL"]}},
            {"code": "Hallintaperuste",
             "selection": {"filter": "item",
                           "values": ["ALL", "3_4"]}},
            {"code": "Vuosi",
             "selection": {"filter": "item", "values": [TARGET_YEAR]}},
        ],
        "response": {"format": "json-stat2"},
    }
    df = fetch_pxweb("asu/askan/alu_askan_005r.px", q)
    df = df[df["Osa-alue_code"].str.startswith("091")]
    wide = (df.pivot_table(index="Osa-alue_code",
                           columns="Hallintaperuste_code",
                           values="value", aggfunc="first")
              .rename(columns={"ALL": "total", "3_4": "ara"}))
    return (wide["ara"] / wide["total"]).rename("ara_frac")


def fetch_foreign_lang_share() -> pd.Series:
    """Share of residents whose mother tongue is 'Muu kieli'."""
    q = {
        "query": [
            {"code": "Alue",
             "selection": {"filter": "all", "values": ["*"]}},
            # Aeidinkieli: "ALL" = total, "3" = Muu kieli (other lang)
            {"code": "Äidinkieli",
             "selection": {"filter": "item", "values": ["ALL", "3"]}},
            {"code": "Ikä",
             "selection": {"filter": "item", "values": ["ALL"]}},
            {"code": "Vuosi",
             "selection": {"filter": "item", "values": [TARGET_YEAR]}},
        ],
        "response": {"format": "json-stat2"},
    }
    df = fetch_pxweb("vrm/vaerak/alu_vaerak_004p.px", q)
    df = df[df["Alue_code"].str.startswith("091")]
    wide = (df.pivot_table(index="Alue_code",
                           columns="Äidinkieli_code",
                           values="value", aggfunc="first")
              .rename(columns={"ALL": "total", "3": "muu"}))
    return (wide["muu"] / wide["total"]).rename("foreign_lang_frac")


def fetch_low_income_share() -> pd.Series:
    """
    Share of household-dwelling-units in the lowest income quintile
    of the Helsinki region (HS = Helsingin seutu).
    The table already exposes 'askun_osuus' (% within area), so we
    just pick the bottom quintile row and rescale to a fraction.
    """
    q = {
        "query": [
            {"code": "Alue",
             "selection": {"filter": "all", "values": ["*"]}},
            {"code": "Kvintiilien määrittelyperuste",
             "selection": {"filter": "item", "values": ["HS"]}},
            {"code": "Tuloluokka",
             "selection": {"filter": "item", "values": ["1"]}},
            {"code": "Vuosi",
             "selection": {"filter": "item", "values": [TARGET_YEAR]}},
            {"code": "Tiedot",
             "selection": {"filter": "item",
                           "values": ["askun_osuus"]}},
        ],
        "response": {"format": "json-stat2"},
    }
    df = fetch_pxweb("tul/astul/alu_astul_011y.px", q)
    df = df[df["Alue_code"].str.startswith("091")]
    s = df.set_index("Alue_code")["value"] / 100.0
    return s.rename("low_income_frac")


# =====================================================================
# Geometry helper
# =====================================================================

def fetch_geometry() -> gpd.GeoDataFrame:
    if not GEO_CACHE.exists():
        print("Downloading polygons...")
        GEO_CACHE.write_bytes(
            requests.get(WFS_URL, timeout=120).content)
    return gpd.read_file(GEO_CACHE)


# =====================================================================
# Plot orchestration
# =====================================================================

INDICATORS = [
    ("ARA (subsidised rental) share", "ara_frac",
     fetch_ara_share,    "magma"),
    ("Foreign-language speaker share", "foreign_lang_frac",
     fetch_foreign_lang_share, "viridis"),
    ("Low-income household share (bottom HS quintile)",
     "low_income_frac",
     fetch_low_income_share, "cividis"),
]


def main() -> None:
    # Pull all three indicators and the polygons.
    gdf = fetch_geometry()
    print(f"{len(gdf)} polygons in WFS layer")

    series = {}
    for title, col, fetcher, _cmap in INDICATORS:
        print(f"  fetching {title} ...")
        series[col] = fetcher()

    # Residential-area filter: drop industrial / harbour / park
    # polygons with < MIN_DWELLINGS housing units. These otherwise
    # produce spurious extremes (e.g. Kylaesaari at 76% bottom-
    # quintile because only a handful of households live there).
    dwellings = fetch_total_dwellings()
    residential = dwellings[dwellings >= MIN_DWELLINGS].index

    # Merge everything on the WFS area code.
    df = pd.concat(series.values(), axis=1)
    df = df.loc[df.index.intersection(residential)]
    merged = gdf.merge(df, left_on="kokotunnus",
                       right_index=True, how="inner")
    # Keep only areas with all three indicators present so the
    # spatial-autocorrelation calculation is on the same support.
    merged = merged.dropna(subset=list(df.columns))
    print(f"{len(merged)} residential sub-areas "
          f"(>={MIN_DWELLINGS} dwellings) with all three indicators")

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    for ax, (title, col, _fetcher, cmap) in zip(axes, INDICATORS):
        merged.plot(column=col, ax=ax, cmap=cmap,
                    legend=True, edgecolor="white", linewidth=0.2,
                    legend_kwds={"shrink": 0.5})
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal")

    fig.suptitle(
        f"Helsinki socio-spatial indicators by osa-alue ({TARGET_YEAR})",
        fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
