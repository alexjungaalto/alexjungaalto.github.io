"""
helsinki_pupils_per_school.py
=============================
Two stacked choropleths comparing two indicators across Helsinki
sub-areas (osa-alue):

  (top)    Pupils per primary school (peruskoulu) per osa-alue
           -- a proxy for educational infrastructure density.
  (bottom) Share of residents whose mother tongue is neither
           Finnish, Swedish, nor Sami ("foreign-language speakers")
           -- a standard demographic indicator in Finnish urban
           statistics.

Putting the two on top of each other makes it easy to eyeball
spatial overlap between school sizes and population composition.

Why pupils-per-school and not pupils-per-classroom?
---------------------------------------------------
Classroom-level (group / "ryhmae") counts at school granularity are
NOT exposed via the public Vipunen API. Only annual pupil counts per
school are openly available there. So the proxy is:

    pupils per school in osa-alue =
        sum of pupils across schools in that osa-alue
        / number of schools in that osa-alue

A LOWER number means schools are smaller / there are more of them
relative to the pupil population in that area -- typically a sign
of better spatial accessibility (more walkable schools).

Data sources (all open, no auth)
--------------------------------
1) Statistics Finland WFS for school locations
   https://geo.stat.fi/geoserver/oppilaitokset/wfs
   Layer "oppilaitokset:oppilaitokset" lists every Finnish
   educational institution with coordinates. Filter to
   oltyp == "11" -> Peruskoulu / comprehensive school.

2) Vipunen API for school-level pupil counts
   https://api.vipunen.fi/api/resources/
   Resource "perusopetus_oppilaat_lukuvuosi_oppilaitos" returns
   one row per (school_year, school) with the pupil count
   (oppilaatLukuvuosiLkm). We use the most recent school year.

3) Helsinki sub-area polygons (cached locally as
   helsinki_osaalue.geojson by helsinki_segregation_indicators.py).
   The same script also exposes a fetcher for the foreign-language
   share, which we re-import here.

Requirements
------------
    pip install pandas geopandas matplotlib requests
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import requests

# Reuse the Aluesarjat fetcher for the foreign-language indicator.
import helsinki_segregation_indicators as hsi


# =====================================================================
# Endpoints / paths / constants
# =====================================================================

# Statistics Finland's GeoServer; one feature per educational
# institution in Finland with EPSG:4326 coordinates.
WFS_OPP = ("https://geo.stat.fi/geoserver/oppilaitokset/wfs"
           "?service=WFS&version=2.0.0&request=GetFeature"
           "&typeNames=oppilaitokset:oppilaitokset"
           "&outputFormat=application/json&srsName=EPSG:4326")

# Vipunen REST data endpoint for the chosen resource. Returns JSON
# with one row per (school_year, school).
VIPUNEN_URL = ("https://api.vipunen.fi/api/resources/"
               "perusopetus_oppilaat_lukuvuosi_oppilaitos/data")

# Cached Helsinki osa-alue polygons (downloaded by the segregation
# script on first run). We do not re-download here -- if missing,
# we exit and ask the user to run that script first.
GEO_OSA = Path("helsinki_osaalue.geojson")

OUT_PNG = Path("helsinki_pupils_per_school.png")

# Bounding box around Helsinki proper, used to trim the nationwide
# WFS response before the more expensive spatial join. Order:
# (min_lon, min_lat, max_lon, max_lat).
HEL_BBOX = (24.78, 60.10, 25.27, 60.30)


# =====================================================================
# Step 1 -- fetch primary-school locations
# =====================================================================

def fetch_schools() -> gpd.GeoDataFrame:
    """
    Pull the nationwide school layer, keep only peruskoulu (oltyp=11),
    then trim to a Helsinki bounding box for performance.
    """
    # geopandas can read a WFS GetFeature response directly via fiona.
    g = gpd.read_file(WFS_OPP)
    # oltyp = institution type code; "11" = Peruskoulu / comprehensive.
    g = g[g["oltyp"] == "11"].copy()
    # .cx is geopandas' coordinate-based slicer (rough bbox filter).
    # The exact assignment to osa-alue happens later via spatial join.
    minx, miny, maxx, maxy = HEL_BBOX
    g = g.cx[minx:maxx, miny:maxy].copy()
    return g


# =====================================================================
# Step 2 -- fetch latest pupil counts per school from Vipunen
# =====================================================================

def fetch_pupils() -> pd.DataFrame:
    """
    Fetch the entire 'pupils per school per year' resource and reduce
    to the most recent school year present in the response.

    Returned columns:
        tunn   -- 5-digit school code (matches WFS attribute 'tunn')
        pupils -- yearly pupil count
    """
    r = requests.get(VIPUNEN_URL, timeout=120)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    # 'lukuvuosi' looks like "2025/2026"; lexicographic max == latest.
    latest = df["lukuvuosi"].max()
    df = df[df["lukuvuosi"] == latest]
    print(f"Vipunen latest school year: {latest}, "
          f"{len(df)} schools nationwide")
    return (df[["kooditOppilaitos", "oppilaatLukuvuosiLkm"]]
            .rename(columns={"kooditOppilaitos": "tunn",
                             "oppilaatLukuvuosiLkm": "pupils"}))


# =====================================================================
# Step 3 -- load Helsinki osa-alue polygons (cached)
# =====================================================================

def fetch_areas() -> gpd.GeoDataFrame:
    if not GEO_OSA.exists():
        raise SystemExit(
            f"{GEO_OSA} not found -- run "
            "helsinki_segregation_indicators.py once to cache it.")
    return gpd.read_file(GEO_OSA)


# =====================================================================
# Step 4 -- join, aggregate, plot
# =====================================================================

def main() -> None:
    # ---- 4a: schools (point) joined with their pupil counts -------
    schools = fetch_schools()
    pupils = fetch_pupils()
    schools = schools.merge(pupils, on="tunn", how="left")
    print(f"{len(schools)} primary schools in the Helsinki bbox, "
          f"{schools['pupils'].notna().sum()} matched to Vipunen")

    # ---- 4b: spatial join schools -> osa-alue (polygon) -----------
    areas = fetch_areas()
    # 'within' assigns each school point to the polygon containing it.
    # 'how=left' keeps schools even if they fall outside any polygon
    # (we drop them on the next line).
    joined = gpd.sjoin(schools, areas[["kokotunnus", "geometry"]],
                       predicate="within", how="left")

    # ---- 4c: aggregate per osa-alue -------------------------------
    agg = (joined.dropna(subset=["kokotunnus"])
                 .groupby("kokotunnus")
                 .agg(schools=("tunn", "count"),
                      pupils=("pupils", "sum"))
                 .reset_index())
    agg["pupils_per_school"] = agg["pupils"] / agg["schools"]

    # Bring the aggregate back to the polygon GeoDataFrame.
    merged = areas.merge(agg, on="kokotunnus", how="inner")
    # Drop osa-alueet that contain no schools (would be NaN anyway).
    merged = merged[merged["schools"] >= 1]
    print(f"{len(merged)} osa-alueet contain at least one school")

    # ---- 4d: add the second indicator (foreign-language share) ----
    # Re-uses the Aluesarjat fetcher from the segregation script so
    # we do not duplicate the PxWeb call here.
    foreign = hsi.fetch_foreign_lang_share()
    merged = merged.merge(foreign, left_on="kokotunnus",
                          right_index=True, how="left")

    # Sequential id 1..N (sorted alphabetically by area name) -- used
    # both for the in-map annotation and for the side legend.
    merged = merged.sort_values("nimi_fi").reset_index(drop=True)
    merged["aid"] = merged.index + 1

    # ---- 4e: figure layout ----------------------------------------
    # Two stacked map panels, with horizontal padding on left and
    # right to leave room for the area-name legend columns.
    fig, axes = plt.subplots(
        2, 1, figsize=(14, 11),
        gridspec_kw={"hspace": 0.05,    # near-zero gap between maps
                     "left": 0.18,      # leave room for left legend
                     "right": 0.82})    # leave room for right legend

    # Top panel: pupils per primary school.
    # 'magma_r' is a perceptually uniform reversed colour map -- dark
    # = small schools, bright = large schools.
    merged.plot(column="pupils_per_school", ax=axes[0], cmap="magma_r",
                legend=True, edgecolor="white", linewidth=0.3,
                legend_kwds={"label": "Pupils per school",
                             "shrink": 0.6})
    axes[0].set_title("Pupils per primary school, Helsinki sub-area")

    # Bottom panel: foreign-language speaker share.
    # 'missing_kwds' draws osa-alueet without an Aluesarjat value
    # (typically tiny non-residential polygons) in light grey.
    merged.plot(column="foreign_lang_frac", ax=axes[1], cmap="viridis",
                legend=True, edgecolor="white", linewidth=0.3,
                missing_kwds={"color": "lightgrey", "label": "no data"},
                legend_kwds={"label": "Foreign-language speaker share",
                             "shrink": 0.6})
    axes[1].set_title(
        "Share of foreign-language speakers, Helsinki sub-area")

    # Common axes treatment: lock aspect ratio so polygons aren't
    # squished, and hide ticks/labels for a cleaner look.
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_axis_off()
        # Annotate every polygon with its sequential id at a point
        # guaranteed to lie INSIDE the polygon (representative_point()
        # is more robust than centroid for L- or U-shaped areas).
        for _, row in merged.iterrows():
            c = row.geometry.representative_point()
            ax.annotate(str(row["aid"]), (c.x, c.y),
                        ha="center", va="center", fontsize=6,
                        color="black")

    # ---- 4f: side legends mapping id -> area name -----------------
    # Build "  3 = Etu-Toeloe"-style strings and split them into two
    # equally tall columns placed in the figure's left and right
    # margins (figure-fraction coordinates).
    items = [f"{row.aid:>3} = {row.nimi_fi}"
             for _, row in merged.iterrows()]
    half = (len(items) + 1) // 2
    fig.text(0.005, 0.5, "\n".join(items[:half]),
             ha="left", va="center",
             fontsize=10, family="monospace")
    fig.text(0.80, 0.5, "\n".join(items[half:]),
             ha="left", va="center",
             fontsize=10, family="monospace")

    # bbox_inches='tight' trims away whitespace produced by the
    # figure-margin legend texts.
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT_PNG}")

    # Sanity-check / quick read: which osa-alueet have the largest
    # average school size? (Often single-school polygons.)
    print("\nTop 5 osa-alueet by pupils per school:")
    print(merged[["kokotunnus", "schools", "pupils",
                  "pupils_per_school"]]
          .sort_values("pupils_per_school", ascending=False)
          .head(5).to_string(index=False))


if __name__ == "__main__":
    main()
