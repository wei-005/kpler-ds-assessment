# import polars as pl
# import math, numpy as np
# from .config import INTERIM_DIR

# WAYPOINT_RX = "(?i)light|anchorage|canal|suez|panama|offshore"

# def build_origin_next_transitions(train_samples: pl.DataFrame) -> pl.DataFrame:
#     # Starting point->Next port frequency
#     return (train_samples
#         .filter(pl.col("destination").is_not_null() & pl.col("next_call_name").is_not_null())
#         .group_by(["destination","next_call_name"])
#         .len()
#         .rename({"len":"cnt"})
#     )

# def global_mf_next(transitions: pl.DataFrame) -> str:
#     g = (transitions.group_by("next_call_name")
#          .agg(pl.col("cnt").sum().alias("cnt"))
#          .sort("cnt", descending=True))
#     return g["next_call_name"][0]

# def build_pc_coords(pc_clean: pl.DataFrame) -> pl.DataFrame:
#     return (pc_clean
#             .select(["destination","destination_latitude","destination_longitude"])
#             .rename({"destination_latitude":"lat","destination_longitude":"lon"})
#             .drop_nulls()
#             .unique())

# def _haversine_km(lat1, lon1, lat2, lon2):
#     if None in (lat1, lon1, lat2, lon2):
#         return float("nan")
#     R=6371.0
#     import math
#     la1, lo1, la2, lo2 = [math.radians(x) for x in [lat1, lon1, lat2, lon2]]
#     dlat = la2-la1; dlon=lo2-lo1
#     a = math.sin(dlat/2)**2 + math.cos(la1)*math.cos(la2)*math.sin(dlon/2)**2
#     c = 2*math.asin(math.sqrt(a))
#     return R*c

# def precompute_geo_neighbors(pc_coords: pl.DataFrame, M:int=10) -> dict:
#     # Calculate the geographically nearest M ports for each port
#     pdf = pc_coords.to_pandas()
#     by_port = {}
#     for i, r in pdf.iterrows():
#         dists = []
#         for j, r2 in pdf.iterrows():
#             if r2["destination"] == r["destination"]: 
#                 continue
#             d = _haversine_km(r["lat"], r["lon"], r2["lat"], r2["lon"])
#             dists.append((r2["destination"], d))
#         dists.sort(key=lambda x: (float("inf") if (x[1] is None or math.isnan(x[1])) else x[1]))
#         by_port[r["destination"]] = [n for n,_ in dists[:M]]
#     return by_port

# def build_candidates_for_split(split_samples: pl.DataFrame,
#                                trans: pl.DataFrame,
#                                pc_coords: pl.DataFrame,
#                                add_true_label: bool,
#                                N:int=10, M:int=10, global_top1:str=None) -> pl.DataFrame:
#     """
#     #Candidates = historical Top-N ∪ geo M ∪ GlobalTop1; include truth for Train/Val if needed
#     """
#     # Historical Top-N (by starting point)
#     topn = (trans.sort(["destination","cnt"], descending=[False,True])
#                  .group_by("destination")
#                  .agg(pl.col("next_call_name").head(N).alias("topN")))

#     # Precompute geo neighbors
#     geo_map = precompute_geo_neighbors(pc_coords, M=M)

#     # Expand the candidate for each sample
#     rows = []
#     import pandas as pd
#     ss = split_samples.select(["sample_port_call_id","destination","next_call_name"]).to_pandas()
#     for _, r in ss.iterrows():
#         origin = r["destination"]
#         truth = r["next_call_name"]
#         top_hist = []
#         sub = topn.filter(pl.col("destination")==origin)
#         if sub.height:
#             top_hist = sub["topN"][0]
#         geo_list = geo_map.get(origin, [])

#         cand = list(dict.fromkeys([*top_hist, *geo_list, global_top1] if global_top1 else [*top_hist, *geo_list]))
#         if add_true_label and (truth is not None) and (truth not in cand):
#             cand.append(truth)

#         for c in cand:
#             rows.append((r["sample_port_call_id"], origin, truth, c, 1 if c==truth else 0))

#     cands = pl.DataFrame(rows, orient="row", schema=[
#         "sample_port_call_id","origin","label","candidate","y"
#     ])

#     # Append coordinates for subsequent features (lat/lon)
#     coords = pc_coords.rename({"destination":"port","lat":"lat","lon":"lon"})
#     cands = (cands
#              .join(coords.rename({"port":"origin","lat":"lat_o","lon":"lon_o"}), on="origin", how="left")
#              .join(coords.rename({"port":"candidate","lat":"lat_c","lon":"lon_c"}), on="candidate", how="left"))
#     return cands

import polars as pl
import math, numpy as np
import pandas as pd
from .config import INTERIM_DIR

WAYPOINT_RX = "(?i)light|anchorage|canal|suez|panama|offshore"


# ---------- 1. Transition statistics ----------
def build_origin_next_transitions(train_samples: pl.DataFrame) -> pl.DataFrame:
    """Build frequency table of origin -> next port"""
    return (
        train_samples
        .filter(pl.col("destination").is_not_null() & pl.col("next_call_name").is_not_null())
        .group_by(["destination","next_call_name"])
        .len()
        .rename({"len":"cnt"})
    )


def global_mf_next(transitions: pl.DataFrame) -> str:
    """Return the single most frequent next port globally."""
    g = (
        transitions.group_by("next_call_name")
        .agg(pl.col("cnt").sum().alias("cnt"))
        .sort("cnt", descending=True)
    )
    return g["next_call_name"][0]


# ---------- 2. Port coordinate helpers ----------
def build_pc_coords(pc_clean: pl.DataFrame) -> pl.DataFrame:
    """Extract unique (destination, lat, lon) pairs."""
    return (
        pc_clean
        .select(["destination","destination_latitude","destination_longitude"])
        .rename({"destination_latitude":"lat","destination_longitude":"lon"})
        .drop_nulls()
        .unique()
    )


def _haversine_km(lat1, lon1, lat2, lon2):
    if None in (lat1, lon1, lat2, lon2):
        return float("nan")
    R = 6371.0
    la1, lo1, la2, lo2 = [math.radians(float(x)) for x in [lat1, lon1, lat2, lon2]]
    dlat = la2 - la1
    dlon = lo2 - lo1
    a = math.sin(dlat/2)**2 + math.cos(la1)*math.cos(la2)*math.sin(dlon/2)**2
    c = 2*math.asin(math.sqrt(a))
    return R * c


def precompute_geo_neighbors(pc_coords: pl.DataFrame, M:int=10) -> dict:
    """Compute the M nearest geographic neighbors for each port."""
    pdf = pc_coords.to_pandas()
    by_port = {}
    for i, r in pdf.iterrows():
        dists = []
        for j, r2 in pdf.iterrows():
            if r2["destination"] == r["destination"]:
                continue
            d = _haversine_km(r["lat"], r["lon"], r2["lat"], r2["lon"])
            dists.append((r2["destination"], d))
        dists.sort(key=lambda x: (float("inf") if (x[1] is None or math.isnan(x[1])) else x[1]))
        by_port[r["destination"]] = [n for n, _ in dists[:M]]
    return by_port


# ---------- 3. Candidate expansion ----------
def build_candidates_for_split(
    split_samples: pl.DataFrame,
    trans: pl.DataFrame,
    pc_coords: pl.DataFrame,
    add_true_label: bool,
    N: int = 10,
    M: int = 10,
    global_top1: str = None
) -> pl.DataFrame:
    """
    Construct candidate ports for each sample:
    candidates = historical Top-N ∪ geographic M ∪ global_top1
    """
    # --- Top-N historical transitions ---
    topn = (
        trans.sort(["destination","cnt"], descending=[False,True])
        .group_by("destination")
        .agg(pl.col("next_call_name").head(N).alias("topN"))
    )

    # --- Geo neighbors ---
    geo_map = precompute_geo_neighbors(pc_coords, M=M)

    # --- Expand candidates per sample ---
    rows = []
    ss = split_samples.select(["sample_port_call_id","destination","next_call_name"]).to_pandas()
    for _, r in ss.iterrows():
        origin = r["destination"]
        truth = r["next_call_name"]
        top_hist = []
        sub = topn.filter(pl.col("destination") == origin)
        if sub.height:
            top_hist = sub["topN"][0]

        geo_list = geo_map.get(origin, [])
        cand = list(dict.fromkeys(
            [*top_hist, *geo_list, global_top1] if global_top1 else [*top_hist, *geo_list]
        ))

        if add_true_label and (truth is not None) and (truth not in cand):
            cand.append(truth)

        for c in cand:
            rows.append((r["sample_port_call_id"], origin, truth, c, 1 if c == truth else 0))

    # --- Build DataFrame ---
    cands = pl.DataFrame(
        rows,
        orient="row",
        schema=["sample_port_call_id","origin","label","candidate","y"]
    )

    # --- Attach coordinates for origin/candidate ---
    coords = pc_coords.rename({"destination":"port","lat":"lat","lon":"lon"})
    cands = (
        cands
        .join(coords.rename({"port":"origin_port","lat":"lat_o","lon":"lon_o"}),
              left_on="origin", right_on="origin_port", how="left")
        .join(coords.rename({"port":"candidate_port","lat":"lat_c","lon":"lon_c"}),
              left_on="candidate", right_on="candidate_port", how="left")
    )

    # --- Final clean-up: drop redundant port ID columns & duplicates ---
    keep = [c for c in cands.columns if not c.endswith("_port")]
    cands = cands.select(keep)
    # Remove duplicate columns (rare edge case)
    seen = {}
    unique_cols = []
    for c in cands.columns:
        if c not in seen:
            seen[c] = 1
            unique_cols.append(c)
    if len(unique_cols) != len(cands.columns):
        cands = cands.select(unique_cols)

    return cands