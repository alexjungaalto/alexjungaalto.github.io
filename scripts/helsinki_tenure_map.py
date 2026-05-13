"""
helsinki_tenure_map.py
======================
Choropleth showing the share of right-of-occupancy housing
(Asumisoikeusasunto) per Helsinki sub-area (osa-alue) on the
lat/lon plane.

============================================================
THE BIG PICTURE
============================================================

Goal:
    For every small statistical area in Helsinki, find out
    what fraction of its dwellings are "right-of-occupancy"
    apartments, and colour-code that fraction on a map.

We need TWO things from the open web:

  (A) NUMBERS   how many dwellings of each tenure type live
                in each sub-area.
                -> Aluesarjat PxWeb statistical database.

  (B) SHAPES    where each sub-area is geographically
                (polygons in latitude/longitude).
                -> Helsinki Open Data WFS server.

Then we JOIN (A) and (B) on a common identifier (a 10-digit
area code that both datasets happen to use), compute the
fraction, and plot the result with geopandas + matplotlib.

============================================================
ABOUT THE TENURE CATEGORY
============================================================

"Right-of-occupancy" (asumisoikeusasunto, often abbreviated
"ASO") is a Finnish hybrid between renting and owning:
the resident pays a one-off deposit -- roughly 15% of the
construction cost of the apartment -- plus a monthly fee.
The building itself stays owned by a non-profit cooperative.
The resident cannot resell the apartment for profit, but
they also cannot be evicted at the owner's whim.

ASO is heavily concentrated in newer, planned districts
(it was politically promoted in certain construction waves),
so the geographic distribution is genuinely interesting and
not flat across the city.

============================================================
HOW THE TWO DATA SOURCES TALK TO US
============================================================

1) Aluesarjat is a PxWeb server. PxWeb is a statistical-cube
   protocol used by many Nordic statistics offices. Instead
   of returning the whole multi-dimensional table, you POST
   a JSON 'query' describing which slices you want, and it
   returns just those cells in JSON-stat2 format.

2) Helsinki Open Data WFS is a Web Feature Service. WFS is a
   standard GIS protocol; here we use the simplest variant
   (HTTP GET with query parameters) and ask for the layer as
   GeoJSON in WGS84 lat/lon. WGS84 is the same coordinate
   system Google Maps uses, so the plot axes will be
   familiar-looking longitudes and latitudes.

Requirements:
    pip install pandas geopandas matplotlib requests
"""

from __future__ import annotations

# itertools.product is used below to walk the Cartesian product
# of dimension categories in the same order PxWeb laid them out.
from itertools import product
from pathlib import Path

import geopandas as gpd          # spatial DataFrames (= pandas + geometry)
import matplotlib.pyplot as plt
import pandas as pd
import requests


# =====================================================================
# Endpoints and constants
# =====================================================================

# PxWeb tables are queried with HTTP POST + a JSON body that says
# which dimensions you want to keep and which categories you want
# to slice on. The URL itself identifies the table.
#
#   alu_askan_005r.px =
#     "Paeaekaupunkiseudun asunnot osa-alueen, hallintaperusteen ja
#      huoneistotyypin mukaan 2002-"
#   (Dwellings in the capital region by sub-area, tenure and
#    dwelling type, 2002-)
PX_URL = ("https://stat.hel.fi/api/v1/fi/Aluesarjat/asu/askan/"
          "alu_askan_005r.px")

# Helsinki Open Data WFS server.
#   typeName         -- which layer we want (osa-alue polygons)
#   outputFormat=json -- give us GeoJSON, not GML
#   srsName=EPSG:4326 -- give us lat/lon coordinates (WGS84)
#   version=1.0.0    -- 2.0.0 sometimes rejects the bbox-less request
WFS_URL = ("https://kartta.hel.fi/ws/geoserver/avoindata/wfs"
           "?service=WFS&version=1.0.0&request=GetFeature"
           "&typeName=avoindata:Piirijako_osaalue"
           "&outputFormat=json&srsName=EPSG:4326")

# Aluesarjat refreshes this table once a year. 2024 is the most
# recent fully populated release at the time of writing. If you
# bump this, double-check the table still has data for that year
# (older years exist back to 2002).
TARGET_YEAR = "2024"

# We cache the GeoJSON on disk after the first download so we do
# not hammer the Helsinki WFS on every run. Delete the file to
# force a fresh download.
GEO_CACHE = Path("helsinki_osaalue.geojson")
OUT_PNG = Path("helsinki_tenure_map.png")


# =====================================================================
# Step 1 -- pull the tenure cube from Aluesarjat
# =====================================================================

def fetch_tenure() -> pd.DataFrame:
    """
    Fetch a small slice of the alu_askan_005r table and return it
    as a long-form DataFrame.

    The table has four dimensions:
        Osa-alue         -- statistical area (hundreds of values)
        Huoneistotyyppi  -- dwelling type (1 room, 2 rooms, ..., total)
        Hallintaperuste  -- tenure category (own, rent, ASO, ...)
        Vuosi            -- year (2002..2024)
    plus one "Tiedot" measure variable (= dwelling count).

    Asking for the entire cube would be wasteful. We only need:
        every area (denominator + numerator come from there),
        dwelling type collapsed to its total (we do not care
          about how many 1-room vs 2-room flats there are),
        two tenure rows: the grand total ("ALL")
                        and right-of-occupancy ("6"),
        one year.
    """
    q = {
        "query": [
            # filter="all" + values=["*"] means "every category".
            {"code": "Osa-alue",
             "selection": {"filter": "all", "values": ["*"]}},

            # filter="item" + values=[...] means "exactly these codes".
            # Collapse dwelling type to the grand total to avoid
            # multiplying the response by ~10 (one row per size).
            {"code": "Huoneistotyyppi",
             "selection": {"filter": "item", "values": ["ALL"]}},

            # We pull two tenure rows:
            #   "ALL" -- all dwellings (denominator for the fraction)
            #   "6"   -- "Asumisoikeusasunto" / right-of-occupancy
            # The other tenure codes (1, 2, 3_4, 5, 7) would let us
            # map any other category; not needed for this map.
            {"code": "Hallintaperuste",
             "selection": {"filter": "item", "values": ["ALL", "6"]}},

            {"code": "Vuosi",
             "selection": {"filter": "item", "values": [TARGET_YEAR]}},
        ],
        # JSON-stat2 is the modern, self-describing JSON format
        # used by Statistics Sweden / Finland / Norway. It encodes
        # the cube as one flat 'value' array; we unfold it below.
        "response": {"format": "json-stat2"},
    }
    r = requests.post(PX_URL, json=q, timeout=60)
    r.raise_for_status()
    js = r.json()

    # ---- JSON-stat2 unrolling ---------------------------------------
    #
    # The response looks like:
    #   {
    #     "id":     ["Osa-alue", "Huoneistotyyppi",
    #                "Hallintaperuste", "Vuosi"],
    #     "size":   [148, 1, 2, 1],
    #     "value":  [12.0, 3.0, 15.0, 4.0, ...],  # flat array
    #     "dimension": {
    #       "Osa-alue":        {"category": {"index": {...},
    #                                        "label": {...}}},
    #       "Hallintaperuste": {"category": {"index": {...},
    #                                        "label": {...}}},
    #       ...
    #     }
    #   }
    #
    # The "value" array is the cube flattened in row-major order
    # using the dimension order in "id". So we need to walk the
    # Cartesian product of dimension categories in the same order
    # to assign each value back to its (area, dwelling-type,
    # tenure, year) coordinates.

    dim_ids = js["id"]                  # dimension names in order
    dimension = js["dimension"]         # metadata per dimension

    # For each dimension, build an ordered list of (code, label).
    # 'index' tells us which slot each code occupies; 'label' gives
    # the human-readable name (e.g. "6" -> "Asumisoikeusasunto").
    dim_codes: dict[str, list[tuple[str, str]]] = {}
    for d in dim_ids:
        cat = dimension[d]["category"]
        index = cat["index"]
        labels = cat["label"]
        if isinstance(index, dict):
            # 'index' may be {code: position}; sort by position so
            # our walk matches the layout of 'value'.
            codes = sorted(index, key=lambda k: index[k])
        else:
            # 'index' may also be a plain list of codes already in
            # the right order; just use it as-is.
            codes = list(index)
        dim_codes[d] = [(c, labels.get(c, c)) for c in codes]

    # Now enumerate all (code-tuple, value) combinations in the
    # same order as 'value', and collect them as DataFrame rows.
    rows = []
    for i, combo in enumerate(product(*[dim_codes[d] for d in dim_ids])):
        row = {}
        for d, (code, label) in zip(dim_ids, combo):
            row[d] = label                # human-readable label
            row[f"{d}_code"] = code       # original code for joins
        v = js["value"][i]
        # PxWeb encodes suppressed/protected cells as -1 in many
        # tables; JSON-stat2 may also surface them as None. Treat
        # both as missing.
        row["value"] = None if v in (None, -1) else v
        rows.append(row)

    return pd.DataFrame(rows)


# =====================================================================
# Step 2 -- pull the polygon layer from Helsinki Open Data
# =====================================================================

def fetch_geometry() -> gpd.GeoDataFrame:
    """
    Download the Helsinki osa-alue polygon layer, or load it from a
    local cache if we have already pulled it in a previous run.

    The returned GeoDataFrame contains, among others:
        kokotunnus  -- 10-digit area code (matches Aluesarjat 'Osa-alue')
        nimi_fi     -- Finnish area name
        geometry    -- shapely Polygon/MultiPolygon in WGS84 lat/lon
    """
    if not GEO_CACHE.exists():
        # First run: ~1.5 MB GeoJSON download.
        print(f"Downloading polygons from {WFS_URL[:60]}... ")
        r = requests.get(WFS_URL, timeout=120)
        r.raise_for_status()
        # Cache the raw response bytes verbatim so subsequent runs
        # are instant and offline-capable.
        GEO_CACHE.write_bytes(r.content)
    # geopandas reads GeoJSON via fiona/pyogrio under the hood.
    return gpd.read_file(GEO_CACHE)


# =====================================================================
# Step 3 -- join, compute fractions, plot
# =====================================================================

def main() -> None:
    # ---- 3a: pull the tenure numbers and keep only Helsinki ---------
    df = fetch_tenure()

    # Aluesarjat covers the whole capital region (PKS = Helsinki +
    # Espoo + Vantaa + Kauniainen). Helsinki sub-areas all have
    # codes starting with "091" -- the official municipality code
    # for Helsinki in Statistics Finland's classification. Filter
    # everything else out so the map shows only Helsinki.
    df = df[df["Osa-alue_code"].str.startswith("091")]

    # ---- 3b: pivot from long to wide so each row is one area --------
    #
    # The 'value' column currently holds either total dwellings or
    # ASO dwellings, distinguished by the 'Hallintaperuste_code'
    # column. Pivot so each area gets one row with two columns:
    #   "total" (was tenure code "ALL")
    #   "aso"   (was tenure code "6")
    wide = (df.pivot_table(index="Osa-alue_code",
                           columns="Hallintaperuste_code",
                           values="value", aggfunc="first")
              .rename(columns={"ALL": "total", "6": "aso"}))

    # The fraction we want to visualise:
    #   aso_frac = ASO dwellings / all dwellings, per sub-area.
    # Division by NaN/zero yields NaN, which we drop below.
    wide["aso_frac"] = wide["aso"] / wide["total"]

    # ---- 3c: bring in the polygons and join on the area code --------
    gdf = fetch_geometry()
    print(f"{len(gdf)} polygons in WFS layer")

    # The Aluesarjat "Osa-alue" code matches exactly the WFS
    # attribute "kokotunnus" (= the full 10-character area code,
    # for example "0911101010"). This is the cleanest possible
    # join key: no string normalisation, no fuzzy matching.
    merged = gdf.merge(wide.reset_index(),
                       left_on="kokotunnus",
                       right_on="Osa-alue_code",
                       how="inner")

    # The WFS layer contains a handful of polygons that do not
    # appear in the housing table -- typically uninhabited islands,
    # parks, or industrial-only blocks. Drop them; they would just
    # render as empty grey blobs.
    merged = merged.dropna(subset=["aso_frac"])
    print(f"Plotting {len(merged)} sub-areas with Aluesarjat data")

    # ---- 3d: render the choropleth ----------------------------------
    #
    # We requested lat/lon (EPSG:4326) from the WFS, so the axes
    # are already longitudes (x) and latitudes (y). The visual
    # distortion at Helsinki's latitude (~60 N) is small enough
    # for an illustrative map; for a publication-quality figure
    # one would reproject to a Finnish national CRS (e.g. ETRS-
    # TM35FIN, EPSG:3067) before plotting.
    fig, ax = plt.subplots(figsize=(9, 9))
    merged.plot(
        column="aso_frac",       # value to colour by
        ax=ax,
        cmap="viridis",          # perceptually uniform colour map
        legend=True,
        edgecolor="white",       # thin white boundaries between areas
        linewidth=0.3,
        legend_kwds={"label": "Asumisoikeusasunto-osuus",
                     "shrink": 0.6},
    )
    ax.set_title(
        f"Right-of-occupancy share by Helsinki sub-area ({TARGET_YEAR})")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    # set_aspect('equal') keeps lon and lat units the same on screen.
    # At Helsinki's latitude one degree of longitude is significantly
    # shorter on the ground than one degree of latitude, so the city
    # appears horizontally compressed. For a quick exploration map
    # this is fine; reproject to EPSG:3067 for a true-shape version.
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
