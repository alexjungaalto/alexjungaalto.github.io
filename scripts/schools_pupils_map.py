"""Fetch HRI per-school pupil counts, join them to school locations, and
plot Helsinki districts containing at least one school with a known
pupil count (markers sized by pupil count).

Inputs:
  helsinki_schools.csv  (from fetch_schools.py)
Outputs:
  helsinki_schools_with_pupils.csv
  helsinki_schools_with_pupils.png
"""

import io
import math
import re
import unicodedata

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import requests
from shapely.geometry import Point

XLSX_URL = (
    "https://www.hel.fi/hel2/tietokeskus/data/helsinki/koulutus/"
    "Helsingin_koulujen_oppilasmaarat_2012-.xlsx"
)
SHEETS = ["Suomenkieliset pkt", "Ruotsinkieliset pkt"]
NAME_COL = 1
DISTRICT_COL = 0
LATEST_TOTAL_COL = 15  # 2021-09-20 grand total

DISTRICTS_WFS = (
    "https://kartta.hel.fi/ws/geoserver/avoindata/wfs"
    "?service=WFS&version=2.0.0&request=GetFeature"
    "&typeName=avoindata:Kaupunginosajako"
    "&outputFormat=application/json&srsName=EPSG:4326"
)
SCHOOLS_CSV = "helsinki_schools.csv"
OUT_CSV = "helsinki_schools_with_pupils.csv"
OUT_PNG = "helsinki_schools_with_pupils.png"


def normalize(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    for w in ("peruskoulu", "ala asteen koulu", "ylakoulu", "alakoulu",
              "koulu", "skola", "school", "comprehensive"):
        s = s.replace(w, "")
    return re.sub(r"\s+", " ", s).strip()


def fetch_pupils() -> pd.DataFrame:
    r = requests.get(XLSX_URL, timeout=60)
    r.raise_for_status()
    buf = io.BytesIO(r.content)
    frames = []
    for sheet in SHEETS:
        df = pd.read_excel(buf, sheet_name=sheet, header=None)
        rows = df.iloc[7:, [DISTRICT_COL, NAME_COL, LATEST_TOTAL_COL]].copy()
        rows.columns = ["district_code", "school_name", "pupils_2021"]
        rows = rows.dropna(subset=["school_name"])
        rows["pupils_2021"] = pd.to_numeric(rows["pupils_2021"], errors="coerce")
        rows["language"] = "fi" if sheet.startswith("Suomen") else "sv"
        frames.append(rows)
    return pd.concat(frames, ignore_index=True)


def merge_pupils() -> pd.DataFrame:
    pupils = fetch_pupils()
    pupils["key"] = pupils["school_name"].map(normalize)
    print(f"Pupil rows: {len(pupils)}")

    schools = pd.read_csv(SCHOOLS_CSV)
    schools["key"] = schools["name"].map(normalize)

    merged = schools.merge(
        pupils[["key", "district_code", "pupils_2021", "school_name"]],
        on="key", how="left", suffixes=("", "_hri"),
    )
    matched = merged["pupils_2021"].notna().sum()
    print(f"Matched {matched}/{len(schools)} schools to pupil counts")

    unmatched = merged[merged["pupils_2021"].isna()]["name"].tolist()
    if unmatched:
        print("Unmatched schools (first 15):")
        for n in unmatched[:15]:
            print(" ", n)

    merged = merged.drop(columns=["key"])
    merged.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")
    return merged


def plot(merged: pd.DataFrame) -> None:
    schools = merged[merged["pupils_2021"].notna()].copy()
    print(f"Schools with known pupil counts: {len(schools)}")

    schools_gdf = gpd.GeoDataFrame(
        schools,
        geometry=[Point(xy) for xy in zip(schools["lon"], schools["lat"])],
        crs="EPSG:4326",
    )
    districts = gpd.read_file(DISTRICTS_WFS).to_crs(epsg=4326)

    joined = gpd.sjoin(schools_gdf, districts, how="left", predicate="within")
    keep_idx = joined["index_right"].dropna().unique().astype(int)
    districts_with_schools = districts.iloc[keep_idx]
    print(f"Districts retained: {len(districts_with_schools)}/{len(districts)}")

    fig, ax = plt.subplots(figsize=(11, 11))
    districts_with_schools.plot(
        ax=ax, edgecolor="black", facecolor="lightsteelblue", linewidth=0.6,
    )

    sizes = schools["pupils_2021"].clip(lower=20) * 0.15
    ax.scatter(
        schools["lon"], schools["lat"],
        s=sizes, color="crimson", edgecolor="black",
        linewidth=0.4, alpha=0.75, zorder=3,
    )

    for n in (100, 300, 600, 900):
        ax.scatter([], [], s=n * 0.15, color="crimson", edgecolor="black",
                   linewidth=0.4, alpha=0.75, label=f"{n} pupils")
    ax.legend(scatterpoints=1, frameon=False, labelspacing=1.4,
              loc="lower right", title="School size")

    ax.set_title(
        f"Helsinki districts with comprehensive schools "
    )
    ax.set_aspect(1 / abs(math.cos(math.radians(60.17))))
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    plt.show()


def main():
    merged = merge_pupils()
    plot(merged)


if __name__ == "__main__":
    main()
