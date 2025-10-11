
import polars as pl
import math, numpy as np
from .config import INTERIM_DIR

WAYPOINT_RX = "(?i)light|anchorage|canal|suez|panama|offshore"

def build_origin_next_transitions(train_samples: pl.DataFrame) -> pl.DataFrame:
    # 起点->下一港 频次
    return (train_samples
        .filter(pl.col("destination").is_not_null() & pl.col("next_call_name").is_not_null())
        .group_by(["destination","next_call_name"])
        .len()
        .rename({"len":"cnt"})
    )

def global_mf_next(transitions: pl.DataFrame) -> str:
    g = (transitions.group_by("next_call_name")
         .agg(pl.col("cnt").sum().alias("cnt"))
         .sort("cnt", descending=True))
    return g["next_call_name"][0]

def build_pc_coords(pc_clean: pl.DataFrame) -> pl.DataFrame:
    return (pc_clean
            .select(["destination","destination_latitude","destination_longitude"])
            .rename({"destination_latitude":"lat","destination_longitude":"lon"})
            .drop_nulls()
            .unique())

def _haversine_km(lat1, lon1, lat2, lon2):
    if None in (lat1, lon1, lat2, lon2):
        return float("nan")
    R=6371.0
    import math
    la1, lo1, la2, lo2 = [math.radians(x) for x in [lat1, lon1, lat2, lon2]]
    dlat = la2-la1; dlon=lo2-lo1
    a = math.sin(dlat/2)**2 + math.cos(la1)*math.cos(la2)*math.sin(dlon/2)**2
    c = 2*math.asin(math.sqrt(a))
    return R*c

def precompute_geo_neighbors(pc_coords: pl.DataFrame, M:int=10) -> dict:
    # 为每个港口计算地理最近 M 个港口
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
        by_port[r["destination"]] = [n for n,_ in dists[:M]]
    return by_port

def build_candidates_for_split(split_samples: pl.DataFrame,
                               trans: pl.DataFrame,
                               pc_coords: pl.DataFrame,
                               add_true_label: bool,
                               N:int=10, M:int=10, global_top1:str=None) -> pl.DataFrame:
    """
    候选 = 历史 Top-N（按频次） ∪ 地理最近 M ∪ GlobalTop1；Train/Val 可强行包含真值
    #Candidates = historical Top-N ∪ geo M ∪ GlobalTop1; include truth for Train/Val if needed
    """
    # 历史 Top-N（按起点）
    topn = (trans.sort(["destination","cnt"], descending=[False,True])
                 .group_by("destination")
                 .agg(pl.col("next_call_name").head(N).alias("topN")))

    # 预计算 geo 近邻
    geo_map = precompute_geo_neighbors(pc_coords, M=M)

    # 展开每条样本的候选
    rows = []
    import pandas as pd
    ss = split_samples.select(["sample_port_call_id","destination","next_call_name"]).to_pandas()
    for _, r in ss.iterrows():
        origin = r["destination"]
        truth = r["next_call_name"]
        top_hist = []
        sub = topn.filter(pl.col("destination")==origin)
        if sub.height:
            top_hist = sub["topN"][0]
        geo_list = geo_map.get(origin, [])

        cand = list(dict.fromkeys([*top_hist, *geo_list, global_top1] if global_top1 else [*top_hist, *geo_list]))
        if add_true_label and (truth is not None) and (truth not in cand):
            cand.append(truth)

        for c in cand:
            rows.append((r["sample_port_call_id"], origin, truth, c, 1 if c==truth else 0))

    cands = pl.DataFrame(rows, schema=[
        "sample_port_call_id","origin","label","candidate","y"
    ])

    # 追加坐标便于后续特征（lat/lon）
    coords = pc_coords.rename({"destination":"port","lat":"lat","lon":"lon"})
    cands = (cands
             .join(coords.rename({"port":"origin","lat":"lat_o","lon":"lon_o"}), on="origin", how="left")
             .join(coords.rename({"port":"candidate","lat":"lat_c","lon":"lon_c"}), on="candidate", how="left"))
    return cands
