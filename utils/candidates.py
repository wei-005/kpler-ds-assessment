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

def global_top_list(transitions: pl.DataFrame, K:int=5) -> list[str]:
    """Return the global Top-K most frequent next ports."""
    g = (
        transitions.group_by("next_call_name")
        .agg(pl.col("cnt").sum().alias("cnt"))
        .sort("cnt", descending=True)
        .select("next_call_name")
        .head(K)
    )
    return g["next_call_name"].to_list()


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

def build_conditional_transitions(train_samples: pl.DataFrame, condition_cols: list[str], top_k:int=10) -> pl.DataFrame:
    """
    Build conditional Top-K transitions keyed by origin + condition columns.
    Returns a frame with columns: ['destination', *condition_cols, 'topN_cond'] (list of ports).
    """
    cond_cols = [c for c in condition_cols if c in train_samples.columns]
    if not cond_cols:
        return pl.DataFrame({"destination": [], "topN_cond": []})
    grp_cols = ["destination", *cond_cols, "next_call_name"]
    trans = (
        train_samples
        .filter(pl.col("destination").is_not_null() & pl.col("next_call_name").is_not_null())
        .group_by(grp_cols)
        .len()
        .rename({"len":"cnt"})
        .sort(["destination", *cond_cols, "cnt"], descending=[False, *([False]*len(cond_cols)), True])
    )
    topk = (
        trans
        .group_by(["destination", *cond_cols])
        .agg(pl.col("next_call_name").head(top_k).alias("topN_cond"))
    )
    return topk


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
    global_top1: str = None,
    global_top_list: list[str] | None = None,
    trans_cond: pl.DataFrame | None = None,
    condition_cols: list[str] | None = None,
    M_stages: list[int] | None = None,
    max_cands_per_sample: int | None = 60,
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
    max_M = max(M_stages) if M_stages else M
    geo_map = precompute_geo_neighbors(pc_coords, M=max_M)

    # --- Build quick-access maps ---
    # Global Top-N by origin
    topn_map = {}
    if topn.height:
        for r in topn.iter_rows(named=True):
            topn_map[r["destination"]] = r["topN"]

    # Conditional Top-N map keyed by (origin, *conditions)
    cond_map = None
    cond_cols = condition_cols or []
    if trans_cond is not None and len(cond_cols) > 0 and trans_cond.height:
        cond_map = {}
        for r in trans_cond.iter_rows(named=True):
            key = tuple([r.get("destination")] + [r.get(c) for c in cond_cols])
            cond_map[key] = r.get("topN_cond", [])

    # --- Expand candidates per sample ---
    rows = []
    ss = split_samples.select(["sample_port_call_id","destination","next_call_name"]).to_pandas()
    for _, r in ss.iterrows():
        origin = r["destination"]
        truth = r["next_call_name"]
        # Global Top-N list
        top_hist = topn_map.get(origin, [])

        # Conditional Top-N list (if available)
        cond_list = []
        if cond_map is not None:
            key = tuple([origin] + [r.get(c, None) for c in cond_cols])
            cond_list = cond_map.get(key, [])

        # Geo neighbors (apply staged slicing if provided)
        full_geo = geo_map.get(origin, [])
        if M_stages:
            geo_list = []
            for m in M_stages:
                geo_list = list(dict.fromkeys([*geo_list, *full_geo[:m]]))
        else:
            geo_list = full_geo[:M]

        base = [*top_hist, *cond_list, *geo_list]
        if global_top_list:
            base.extend(global_top_list)
        if global_top1 and (not global_top_list or global_top1 not in global_top_list):
            base.append(global_top1)
        cand = list(dict.fromkeys(base))
        # limit per-sample candidate count to control memory (if configured)
        if (max_cands_per_sample is not None) and (len(cand) > max_cands_per_sample):
            cand = cand[:max_cands_per_sample]

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
