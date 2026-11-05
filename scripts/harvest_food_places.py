"""Harvest grocery, convenience and liquor stores across SF, Alameda and Contra Costa
counties with the Google Places API (New), using an adaptive quadtree search grid.

For our city planning project this script does NOT NEED TO BE RUN AGAIN. Harvesting was
already performed with a personal API key. 

Run with:  python harvest_food_places.py
Set the key first:  export GOOGLE_API_KEY="your key"
"""
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import shapely
from pyproj import Transformer
from shapely.geometry import Point

API_KEY = os.environ.get("GOOGLE_API_KEY", "key")
TRACTS_PATH = "selected_bay_tracts"

BASE_DIR = Path("Bay-Area-Food-Access")
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_DIR = BASE_DIR / "data"
CHECKPOINT_PATH = OUTPUT_DIR / "places_checkpoint.pkl"

PLACES_URL = "https://places.googleapis.com/v1/places:searchNearby"
MAX_RESULTS = 20
TARGET_TYPES = ["grocery_store", "convenience_store", "liquor_store"]

# rating, userRatingCount and priceLevel move each request to the Enterprise tier
# (about $35 vs $32 per 1,000, and 1,000 vs 5,000 free requests per month)
INCLUDE_RATINGS = True

FIELDS = ["places.id", "places.displayName", "places.location", "places.formattedAddress",
          "places.shortFormattedAddress", "places.businessStatus", "places.types", "places.primaryType"]
if INCLUDE_RATINGS:
    FIELDS += ["places.rating", "places.userRatingCount", "places.priceLevel"]
FIELD_MASK = ",".join(FIELDS)

ROOT_CELL_M = 8000
MIN_CELL_M = 500
MAX_TRACTS_PER_CELL = 6

BUDGET_USD = 250
RATE_PER_1000 = 35 if INCLUDE_RATINGS else 32
FREE_REQUESTS = 1000 if INCLUDE_RATINGS else 5000

# set to the resume index printed after a budget stop
START_AT = 0


class BudgetExceeded(Exception):
    pass


def est_cost(n_requests):
    return max(n_requests - FREE_REQUESTS, 0) * RATE_PER_1000 / 1000


if API_KEY == "key":
    raise SystemExit("Set the GOOGLE_API_KEY environment variable before running.")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Tracts
tracts = pd.read_csv(TRACTS_PATH, dtype={"geoid": str})
tract_gdf = gpd.GeoDataFrame(tracts, geometry=gpd.GeoSeries.from_wkt(tracts["geometry"]), crs="EPSG:4326")
in_counties = tract_gdf["geoid"].str[:5].isin(["06075", "06013", "06001"])
bay = tract_gdf[in_counties & (tract_gdf["geoid"] != "06075980401")].reset_index(drop=True)
print(f"Loaded {len(bay)} tracts")

# Adaptive grid: split any cell holding too many tracts (tracts hold roughly equal
# populations, so tract count per cell works as a density proxy)
# this deals with the fact that google allows 20 results max per page. 
# we anticipate more than 20 stores in one tract!
bay_meters = bay.to_crs(epsg=26910)
to_ll = Transformer.from_crs(26910, 4326, always_xy=True)

land = shapely.union_all(bay_meters.geometry.values)
cent = bay_meters.geometry.centroid
cx, cy = cent.x.values, cent.y.values
minx, miny, maxx, maxy = bay_meters.total_bounds


def land_in_cell(x0, y0, s):
    return shapely.clip_by_rect(land, x0, y0, x0 + s, y0 + s)


def cell_circle(x0, y0, s):
    """Smallest circle covering the land in the cell, as (lat, lon, radius_m), or None if no land."""
    piece = land_in_cell(x0, y0, s)
    if piece.is_empty:
        return None
    center = shapely.minimum_bounding_circle(piece).centroid
    radius = max(shapely.minimum_bounding_radius(piece) * 1.02, 100.0)
    lon, lat = to_ll.transform(center.x, center.y)
    return lat, lon, min(radius, 50000.0)


def children(x0, y0, s):
    h = s / 2
    return [(x0, y0, h), (x0 + h, y0, h), (x0, y0 + h, h), (x0 + h, y0 + h, h)]


def plan_cells():
    leaves = []

    def rec(x0, y0, s):
        if land_in_cell(x0, y0, s).is_empty:
            return
        n_tracts = ((cx >= x0) & (cx < x0 + s) & (cy >= y0) & (cy < y0 + s)).sum()
        if n_tracts > MAX_TRACTS_PER_CELL and s / 2 >= MIN_CELL_M:
            for c in children(x0, y0, s):
                rec(*c)
        else:
            leaves.append((x0, y0, s))

    for i in range(int(np.ceil((maxx - minx) / ROOT_CELL_M))):
        for j in range(int(np.ceil((maxy - miny) / ROOT_CELL_M))):
            rec(minx + i * ROOT_CELL_M, miny + j * ROOT_CELL_M, ROOT_CELL_M)
    return leaves


cells_plan = plan_cells()

plan_rows = []
for (x0, y0, s) in cells_plan:
    lat, lon, r = cell_circle(x0, y0, s)
    plan_rows.append({"x0": x0, "y0": y0, "size_m": s, "lat": lat, "lon": lon, "radius_m": r})
plan = pd.DataFrame(plan_rows)
plan_gdf = gpd.GeoDataFrame(plan, geometry=gpd.points_from_xy(plan.lon, plan.lat), crs="EPSG:4326")

print(f"Adaptive grid: {len(plan)} search circles")
print(plan.size_m.value_counts().sort_index().to_string())

plan_gdf.to_parquet(OUTPUT_DIR / "adaptive_grid.parquet")
ax = plan_gdf.plot(column="size_m", categorical=True, legend=True, markersize=6, figsize=(8, 8))
ax.set_axis_off()
plt.savefig(OUTPUT_DIR / "adaptive_grid.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved adaptive grid to {OUTPUT_DIR}")

# Harvest
all_places = {}
n_requests = 0
unresolved = []

HEADERS = {"Content-Type": "application/json",
           "X-Goog-Api-Key": API_KEY,
           "X-Goog-FieldMask": FIELD_MASK}


def nearby(lat, lon, radius_m, place_type, retries=5):
    """One Nearby Search (New) request. Returns up to 20 places."""
    global n_requests
    if est_cost(n_requests + 1) > BUDGET_USD:
        raise BudgetExceeded(f"Next request would push the estimate past ${BUDGET_USD} "
                             f"({n_requests:,} requests made so far)")
    body = {
        "includedTypes": [place_type],
        "maxResultCount": MAX_RESULTS,
        "locationRestriction": {"circle": {"center": {"latitude": lat, "longitude": lon},
                                           "radius": float(radius_m)}},
    }
    for attempt in range(retries):
        resp = requests.post(PLACES_URL, headers=HEADERS, data=json.dumps(body), timeout=30)
        if resp.status_code == 200:
            n_requests += 1
            return resp.json().get("places", [])
        if resp.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** attempt)
            continue
        raise RuntimeError(f"Places API error {resp.status_code}: {resp.text[:500]}")
    raise RuntimeError("Too many retries")


def absorb(place, place_type):
    """harvests useful information about a place."""
    pid = place.get("id")
    if not pid:
        return
    if pid not in all_places:
        loc = place.get("location", {})
        all_places[pid] = {
            "place_id": pid,
            "name": place.get("displayName", {}).get("text"),
            "lat": loc.get("latitude"),
            "lon": loc.get("longitude"),
            "rating": place.get("rating"),
            "user_ratings_total": place.get("userRatingCount"),
            "vicinity": place.get("shortFormattedAddress"),
            "address": place.get("formattedAddress"),
            "price_level": place.get("priceLevel"),
            "business_status": place.get("businessStatus"),
            "primary_search_type": place_type,
            "primary_type": place.get("primaryType"),
            "types": ", ".join(place.get("types", [])),
            "found_by": set(),
        }
    all_places[pid]["found_by"].add(place_type)


def harvest_cell(x0, y0, s, types):
    """Query each type for this cell. A type that hits the 20 result cap is re-run on the four sub-cells."""
    circ = cell_circle(x0, y0, s)
    if circ is None:
        return
    lat, lon, r = circ
    saturated = []
    for t in types:
        places = nearby(lat, lon, r, t)
        for p in places:
            absorb(p, t)
        if len(places) >= MAX_RESULTS:
            saturated.append(t)
    if saturated:
        if s / 2 >= MIN_CELL_M:
            for c in children(x0, y0, s):
                harvest_cell(*c, saturated)
        else:
            unresolved.append((lat, lon, saturated))


def save_checkpoint():
    pd.to_pickle({"places": all_places, "n_requests": n_requests}, CHECKPOINT_PATH)


if START_AT > 0 and CHECKPOINT_PATH.exists():
    saved = pd.read_pickle(CHECKPOINT_PATH)
    all_places, n_requests = saved["places"], saved["n_requests"]
    print(f"Resumed from checkpoint: {len(all_places):,} stores, {n_requests:,} requests so far")

print(f"\nStarting harvest of {len(cells_plan)} cells (stops automatically at ${BUDGET_USD})")
idx = START_AT
try:
    for idx in range(START_AT, len(cells_plan)):
        x0, y0, s = cells_plan[idx]
        harvest_cell(x0, y0, s, TARGET_TYPES)
        done = idx + 1
        if done % 25 == 0 or done == len(cells_plan):
            print(f"Cell {done}/{len(cells_plan)} | requests: {n_requests:,} | "
                  f"est. cost: ${est_cost(n_requests):,.2f} | unique stores: {len(all_places):,}", flush=True)
            save_checkpoint()
except BudgetExceeded as e:
    print(f"\nSTOPPED: {e}")
    print(f"Resume next time with START_AT = {idx} (this cell will be redone; duplicates are ignored).")
finally:
    save_checkpoint()

print(f"\nTotal requests: {n_requests:,} (est. cost ${est_cost(n_requests):,.2f})")
print(f"Cells still saturated at {MIN_CELL_M} m (possible missing stores): {len(unresolved)}")

# Convert and save
rows = []
for p in all_places.values():
    p = dict(p)
    p["found_by"] = ", ".join(sorted(p["found_by"]))
    p["geometry"] = Point(p["lon"], p["lat"])
    rows.append(p)
gdf_food_places = gpd.GeoDataFrame(rows, crs="EPSG:4326")

gdf_food_places = gdf_food_places[gdf_food_places["business_status"] != "CLOSED_PERMANENTLY"]
gdf_food_places = gpd.sjoin(gdf_food_places, bay[["geometry"]], how="inner", predicate="within")
gdf_food_places = gdf_food_places.drop(columns=["index_right"], errors="ignore")

gdf_food_places.to_parquet(DATA_DIR / "bay_food_places.parquet")
gdf_food_places.to_csv(DATA_DIR / "bay_food_places.csv", index=False)
print(f"\nDone. Saved {len(gdf_food_places):,} unique active stores to {DATA_DIR}")
