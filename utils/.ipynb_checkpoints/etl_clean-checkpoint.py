
from .config import DATA_DIR, INTERIM_DIR, PROCESSED_DIR
import polars as pl
import math, numpy as np

ESSENTIAL = ("id","vessel_id","start_utc","end_utc")

# 可调整: cargo_volume 的低阈值分位（仅在列存在时使用）
#Adjustable: lower-quantile threshold of cargo_volume (if column exists)
CARGO_FLOOR_Q = 0.10

def _haversine_km(lat1, lon1, lat2, lon2):
    if None in (lat1, lon1, lat2, lon2):
        return None
    R = 6371.0
    la1, lo1, la2, lo2 = [math.radians(x) for x in [lat1, lon1, lat2, lon2]]
    dlat = la2 - la1; dlon = lo2 - lo1
    a = math.sin(dlat/2)**2 + math.cos(la1)*math.cos(la2)*math.sin(dlon/2)**2
    c = 2*math.asin(math.sqrt(a))
    return R*c

def load_raw():
    pc = pl.read_csv(DATA_DIR / "port_calls.csv", try_parse_dates=True)
    tr = pl.read_csv(DATA_DIR / "trades.csv", try_parse_dates=True)
    vs = pl.read_csv(DATA_DIR / "vessels.csv", try_parse_dates=True)
    return pc, tr, vs

def clean_port_calls(pc: pl.DataFrame) -> pl.DataFrame:
    pc = pc.filter(pl.all_horizontal([pl.col(c).is_not_null() for c in ESSENTIAL]))
    # 去重：同船同 start_utc 保留最后一条（按 end_utc/cargo_volume 作为tie-break）
    #Deduplicate: keep last per (vessel_id,start_utc)
    sort_cols = ["vessel_id","start_utc","end_utc"]
    if "cargo_volume" in pc.columns: sort_cols.append("cargo_volume")
    pc = pc.sort(sort_cols).unique(subset=["vessel_id","start_utc"], keep="last")

    # 停时（小时）+ 极端长停标记 + 负停时剔除
    #Dwell hours + long-stay flag + drop negative durations
    pc = pc.with_columns(((pl.col("end_utc").cast(pl.Int64) - pl.col("start_utc").cast(pl.Int64)) / 3_600_000_000).alias("dur_h"))
    pc = pc.filter(pl.col("dur_h") >= 0)
    p995 = pc.select(pl.col("dur_h").quantile(0.995)).item()
    pc = pc.with_columns([
        (pl.col("dur_h") > p995).alias("is_long_stay_outlier"),
        pl.when(pl.col("dur_h") > p995).then(p995).otherwise(pl.col("dur_h")).alias("dur_h_winsor")
    ])
    return pc

def add_operation_flags(pc: pl.DataFrame) -> pl.DataFrame:
    if "cargo_volume" in pc.columns:
        q10 = pc.filter(pl.col("cargo_volume")>0).select(pl.col("cargo_volume").quantile(CARGO_FLOOR_Q)).item()
        cargo_floor = float(q10) if q10 is not None else 0.0
        pc = pc.with_columns([
            ((pl.col("draught_change") > 0) & (pl.col("cargo_volume") >= cargo_floor)).alias("is_load"),
            ((pl.col("draught_change") < 0) & (pl.col("cargo_volume") >= cargo_floor)).alias("is_discharge")
        ])
    else:
        pc = pc.with_columns([
            (pl.col("draught_change") > 0).alias("is_load"),
            (pl.col("draught_change") < 0).alias("is_discharge")
        ])
    return pc

def ensure_interim() -> pl.DataFrame:
    """
    清洗 port_calls 并落盘到 INTERIM_DIR/port_calls.cleaned.parquet
    #Clean port_calls and save to INTERIM_DIR/port_calls.cleaned.parquet
    """
    out = INTERIM_DIR / "port_calls.cleaned.parquet"
    pc, tr, vs = load_raw()
    pc = clean_port_calls(pc)
    pc = add_operation_flags(pc)
    pc.write_parquet(out)
    return pl.read_parquet(out)

def build_samples_taskA(pc: pl.DataFrame = None, trades: pl.DataFrame = None, vessels: pl.DataFrame = None) -> pl.DataFrame:
    """
    构建任务A样本（严格下一站），并保留下游会用到的列：
    #Build Task‑A samples (strict next-call), keeping downstream-useful columns:

    - is_load / is_discharge
    - prev_dist_km / last_leg_knots_est
    - product_family_dom（基于 trades 的 origin 聚合）
    """
    if pc is None or trades is None or vessels is None:
        pc0, tr0, vs0 = load_raw()
        if pc is None: pc = pc0
        if trades is None: trades = tr0
        if vessels is None: vessels = vs0

    # 确保时间排序，准备 next_call
    pc = pc.sort(["vessel_id","start_utc"]).with_columns([
        pl.col("id").shift(-1).over("vessel_id").alias("next_call_id"),
        pl.col("destination").shift(-1).over("vessel_id").alias("next_call_name"),
        pl.col("start_utc").alias("call_ts")
    ])

    # 计算上一段距离/速度（基于 port_calls 的相邻两点）
    #Compute previous-leg distance/speed (adjacent port_calls per vessel)
    coords = pc.select(["id","vessel_id","destination_latitude","destination_longitude","start_utc"]).rename({
        "destination_latitude":"lat","destination_longitude":"lon"
    }).sort(["vessel_id","start_utc"])

    coords_prev = coords.with_columns([
        pl.col("lat").shift(1).over("vessel_id").alias("lat_prev"),
        pl.col("lon").shift(1).over("vessel_id").alias("lon_prev"),
        pl.col("start_utc").shift(1).over("vessel_id").alias("t_prev")
    ])

    # 转 Pandas 做逐行 haversine + 速度估算（节）
    pdf = coords_prev.select(["id","lat","lon","lat_prev","lon_prev","start_utc","t_prev"]).to_pandas()
    def hv_km_row(r):
        if any(pd is None for pd in [r["lat"],r["lon"],r["lat_prev"],r["lon_prev"]]):
            return float("nan")
        return _haversine_km(r["lat_prev"], r["lon_prev"], r["lat"], r["lon"])
    import pandas as pd
    pdf["prev_dist_km"] = pdf.apply(hv_km_row, axis=1)
    # 小心时间为空
    def hours_row(r):
        if r["t_prev"] is None or r["start_utc"] is None:
            return float("nan")
        return (pd.to_datetime(r["start_utc"]) - pd.to_datetime(r["t_prev"])).total_seconds()/3600.0
    pdf["prev_hours"] = pdf.apply(hours_row, axis=1)
    pdf["last_leg_knots_est"] = (pdf["prev_dist_km"] / (pdf["prev_hours"] + 1e-9)) / 1.852

    prev_df = pl.from_pandas(pdf[["id","prev_dist_km","last_leg_knots_est"]])
    pc = pc.join(prev_df, on="id", how="left")

    # product_family_dom（按 origin 汇总成交量取最大）
    if all(c in trades.columns for c in ["port_call_origin_id","product_family","traded_volume"]):
        tr_pf = (trades
                 .group_by(["port_call_origin_id","product_family"])
                 .agg(pl.col("traded_volume").sum().alias("vol_pf"))
                 .sort(["port_call_origin_id","vol_pf"], descending=[False,True])
                 .group_by("port_call_origin_id")
                 .agg(pl.first("product_family").alias("product_family_dom")))
        pc = pc.join(tr_pf.rename({"port_call_origin_id":"id"}), on="id", how="left")

    keep = [
        "id","vessel_id","destination","destination_latitude","destination_longitude",
        "start_utc","end_utc","call_ts","is_load","is_discharge",
        "next_call_id","next_call_name",
        "prev_dist_km","last_leg_knots_est","product_family_dom"
    ]
    keep = [c for c in keep if c in pc.columns]
    samples = pc.select(keep).rename({"id":"sample_port_call_id"})
    # 仅保留有真值行
    samples = samples.filter(pl.col("next_call_name").is_not_null())

    outp = PROCESSED_DIR / "samples_taskA.parquet"
    samples.write_parquet(outp)
    return samples
