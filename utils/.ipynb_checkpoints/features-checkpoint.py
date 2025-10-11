
import polars as pl
import numpy as np
import math

WAYPOINT_RX = "(?i)light|anchorage|canal|suez|panama|offshore"

def region_id(lat, lon):
    if lat is None or lon is None: return "r_unk"
    try:
        ilat = int((lat + 90)//30)
        ilon = int((lon + 180)//60)
        return f"r{ilat}_{ilon}"
    except:
        return "r_unk"

def _haversine_km(lat1, lon1, lat2, lon2):
    if None in (lat1, lon1, lat2, lon2):
        return float("nan")
    la1, lo1, la2, lo2 = [math.radians(x) for x in [lat1, lon1, lat2, lon2]]
    dlat = la2 - la1; dlon = lo2 - lo1
    a = math.sin(dlat/2)**2 + math.cos(la1)*math.cos(la2)*math.sin(dlon/2)**2
    c = 2*math.asin(math.sqrt(a))
    return 6371.0*c

def build_ports_attr(pc_coords: pl.DataFrame) -> pl.DataFrame:
    return (pc_coords
            .with_columns([
                pl.struct(["lat","lon"]).map_elements(lambda s: region_id(s["lat"], s["lon"])).alias("region")
            ]))

def compute_port_degree(train_trans: pl.DataFrame) -> pl.DataFrame:
    deg_in  = train_trans.group_by("next_call_name").agg(pl.col("cnt").sum().alias("in_cnt")).rename({"next_call_name":"destination"})
    deg_out = train_trans.group_by("destination").agg(pl.col("cnt").sum().alias("out_cnt"))
    port_degree = deg_out.join(deg_in, on="destination", how="full").fill_null(0)
    return port_degree

# def attach_port_side(cands: pl.DataFrame, ports_attr: pl.DataFrame, port_degree: pl.DataFrame) -> pl.DataFrame:
#     df = cands.join(ports_attr.rename({"destination":"origin"}), on="origin", how="left", suffix="_o")
#     df = df.join(ports_attr.rename({"destination":"candidate"}), on="candidate", how="left", suffix="_c")
#     df = df.join(port_degree.rename({"destination":"candidate"}), on="candidate", how="left")
#     # 距离、是否同区域
#     pdf = df.select(["sample_port_call_id","origin","candidate","lat_o","lon_o","lat_c","lon_c","region_o","region_c"]).to_pandas()
#     def hv(r):
#         return _haversine_km(r["lat_o"], r["lon_o"], r["lat_c"], r["lon_c"])
#     pdf["dist_km"] = pdf.apply(hv, axis=1)
#     pdf["is_same_region"] = (pdf["region_o"] == pdf["region_c"]).astype(int)
#     df = df.join(pl.from_pandas(pdf[["sample_port_call_id","dist_km","is_same_region"]]), on="sample_port_call_id", how="left")
#     return df

def attach_port_side(cands: pl.DataFrame, ports_attr: pl.DataFrame, port_degree: pl.DataFrame) -> pl.DataFrame:
    # 1️⃣ 先合并起点（origin）信息
    df = cands.join(
        ports_attr.rename({
            "destination": "origin",
            "lat": "lat_o",
            "lon": "lon_o",
            "region": "region_o"
        }),
        on="origin", how="left"
    )

    # 2️⃣ 再合并候选（candidate）信息
    df = df.join(
        ports_attr.rename({
            "destination": "candidate",
            "lat": "lat_c",
            "lon": "lon_c",
            "region": "region_c"
        }),
        on="candidate", how="left"
    )

    # 3️⃣ 合并港口网络度量
    df = df.join(
        port_degree.rename({"destination": "candidate"}),
        on="candidate", how="left"
    )

    # 4️⃣ 计算距离 & 区域特征
    pdf = df.select([
        "sample_port_call_id", "origin", "candidate",
        "lat_o", "lon_o", "lat_c", "lon_c", "region_o", "region_c"
    ]).to_pandas()

    def hv(r):
        return _haversine_km(r["lat_o"], r["lon_o"], r["lat_c"], r["lon_c"])

    pdf["dist_km"] = pdf.apply(hv, axis=1)
    pdf["is_same_region"] = (pdf["region_o"] == pdf["region_c"]).astype(int)

    df = df.join(pl.from_pandas(pdf[["sample_port_call_id", "dist_km", "is_same_region"]]),
                 on="sample_port_call_id", how="left")

    return df


def dwt_bucketize(x):
    if x is None or (isinstance(x, float) and math.isnan(x)): return "dwt_unk"
    try:
        x = float(x)
    except:
        return "dwt_unk"
    if x < 10_000:   return "<10k"
    if x < 50_000:   return "10–50k"
    if x < 120_000:  return "50–120k"
    if x < 200_000:  return "120–200k"
    return "200k+"

def build_sample_side(samples: pl.DataFrame, pc_clean: pl.DataFrame, vessels: pl.DataFrame) -> pl.DataFrame:
    # 连接船舶静态属性
    vcols = [c for c in ["id","dead_weight","build_year","vessel_type"] if c in vessels.columns]
    vs = vessels.select(vcols).rename({"id":"vessel_id"})
    s = samples.join(vs, on="vessel_id", how="left")

    # 派生列
    s = s.with_columns([
        pl.col("dead_weight").cast(pl.Float64).map_elements(lambda x: dwt_bucketize(x)).alias("dwt_bucket"),
        pl.when(pl.col("is_load")==True).then(1)
          .when(pl.col("is_discharge")==True).then(0)
          .otherwise(None).alias("is_laden_after_call"),
        (2023 - pl.col("build_year").cast(pl.Int64)).alias("age"),
        pl.col("call_ts").dt.month().alias("month"),
        pl.col("call_ts").dt.weekday().alias("dow"),
    ])

    # 季节性 sin/cos
    s = s.with_columns([
        pl.lit(np.pi).alias("_pi")  # helper
    ]).with_columns([
        (pl.col("_pi")*2*pl.col("month")/12).alias("_mang"),
        (pl.col("_pi")*2*pl.col("dow")/7).alias("_dangg"),
    ]).with_columns([
        pl.col("_mang").map_elements(lambda a: math.sin(a)).alias("month_sin"),
        pl.col("_mang").map_elements(lambda a: math.cos(a)).alias("month_cos"),
        pl.col("_dangg").map_elements(lambda a: math.sin(a)).alias("dow_sin"),
        pl.col("_dangg").map_elements(lambda a: math.cos(a)).alias("dow_cos"),
    ]).drop(["_pi","_mang","_dangg"])

    # 保留下游会用的列（包括 product_family_dom / prev_dist_km / last_leg_knots_est）
    keep = [
        "sample_port_call_id","vessel_id","destination","call_ts",
        "is_load","is_discharge","is_laden_after_call",
        "dead_weight","dwt_bucket","build_year","age","vessel_type",
        "prev_dist_km","last_leg_knots_est","product_family_dom",
        "month","dow","month_sin","month_cos","dow_sin","dow_cos"
    ]
    keep = [c for c in keep if c in s.columns]
    return s.select(keep)

def merge_all_features(cands_df: pl.DataFrame, sample_side: pl.DataFrame, split_df: pl.DataFrame) -> pl.DataFrame:
    df = cands_df.join(split_df.select(["sample_port_call_id","is_crisis_time"]), on="sample_port_call_id", how="left")
    df = df.join(sample_side, on="sample_port_call_id", how="left")
    df = df.with_columns((pl.col("dist_km") * pl.col("is_crisis_time").cast(pl.Int8)).alias("dist_x_crisis"))
    return df
