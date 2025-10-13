# utils/features.py
import polars as pl
import numpy as np
import math

WAYPOINT_RX = "(?i)light|anchorage|canal|suez|panama|offshore"

# ---------- helpers ----------
def region_id(lat, lon):
    if lat is None or lon is None:
        return "r_unk"
    try:
        ilat = int((lat + 90)//30)   # 6 bands
        ilon = int((lon + 180)//60)  # 6 bands
        return f"r{ilat}_{ilon}"
    except Exception:
        return "r_unk"

def _haversine_km(lat1, lon1, lat2, lon2):
    if None in (lat1, lon1, lat2, lon2):
        return float("nan")
    la1, lo1, la2, lo2 = [math.radians(float(x)) for x in [lat1, lon1, lat2, lon2]]
    dlat = la2 - la1
    dlon = lo2 - lo1
    a = math.sin(dlat/2)**2 + math.cos(la1)*math.cos(la2)*math.sin(dlon/2)**2
    c = 2*math.asin(math.sqrt(a))
    return 6371.0 * c

def dwt_bucketize(x):
    # robust bucketing
    if x is None:
        return "dwt_unk"
    try:
        x = float(x)
    except Exception:
        return "dwt_unk"
    if x < 10_000:   return "<10k"
    if x < 50_000:   return "10–50k"
    if x < 120_000:  return "50–120k"
    if x < 200_000:  return "120–200k"
    return "200k+"

# ---------- port-side features ----------
def build_ports_attr(pc_coords: pl.DataFrame) -> pl.DataFrame:
    """
    Input: pc_coords with columns ["destination","lat","lon"]
    Output: add "region"
    """
    return (
        pc_coords
        .with_columns([
            pl.struct(["lat","lon"]).map_elements(
                lambda s: region_id(s["lat"], s["lon"])
            ).alias("region")
        ])
    )

def compute_port_degree(train_trans: pl.DataFrame) -> pl.DataFrame:
    """
    train_trans: columns ["destination","next_call_name","cnt"]
    returns per-port in/out degree counts
    """
    deg_in  = (train_trans.group_by("next_call_name")
               .agg(pl.col("cnt").sum().alias("in_cnt"))
               .rename({"next_call_name":"destination"}))
    deg_out = (train_trans.group_by("destination")
               .agg(pl.col("cnt").sum().alias("out_cnt")))
    return deg_out.join(deg_in, on="destination", how="full").fill_null(0)

def attach_port_side(cands: pl.DataFrame,
                     ports_attr: pl.DataFrame,
                     port_degree: pl.DataFrame,
                     compute_distance: bool = True) -> pl.DataFrame:
    """
    Join port attributes for origin/candidate without creating duplicate 'origin'/'candidate' columns.
    NOTE: use left_on/right_on instead of renaming 'destination' to 'origin'.
    """
    # origin-side join
    df = cands.join(
        ports_attr.rename({"lat":"lat_o","lon":"lon_o","region":"region_o"}),
        left_on="origin", right_on="destination", how="left"
    )
    # candidate-side join
    df = df.join(
        ports_attr.rename({"lat":"lat_c","lon":"lon_c","region":"region_c"}),
        left_on="candidate", right_on="destination", how="left"
    )
    # degree on candidate
    df = df.join(port_degree, left_on="candidate", right_on="destination", how="left")

    # same-region can be computed cheaply in polars
    df = df.with_columns((pl.col("region_o") == pl.col("region_c")).cast(pl.Int8).alias("is_same_region"))

    if not compute_distance:
        # skip heavy haversine; leave dist_km as null to reduce cost
        return df

    # compute distance via pandas apply (heavier)
    pdf = df.select(["sample_port_call_id", "lat_o", "lon_o", "lat_c", "lon_c"]).to_pandas()
    def hv(r):
        return _haversine_km(r["lat_o"], r["lon_o"], r["lat_c"], r["lon_c"])
    pdf["dist_km"] = pdf.apply(hv, axis=1)
    df = df.join(pl.from_pandas(pdf[["sample_port_call_id","dist_km"]]), on="sample_port_call_id", how="left")
    return df

# ---------- sample-side features ----------
def build_sample_side(samples: pl.DataFrame,
                      pc_clean: pl.DataFrame,
                      vessels: pl.DataFrame) -> pl.DataFrame:
    """
    Prepare sample-side features for each sample_port_call_id:
    - vessel_type / dead_weight / build_year -> age, dwt_bucket
    - laden state: is_load/is_discharge -> is_laden_after_call
    - prev_dist_km / last_leg_knots_est: assumed from samples
    - seasonal month/dow and sin/cos
    - product_family_dom pass-through if present in samples
    """
    s = samples

    # ensure call_ts is datetime
    if s.schema.get("call_ts") == pl.Utf8:
        s = s.with_columns(pl.col("call_ts").str.strptime(pl.Datetime, strict=False))

    # join vessel static fields
    vcols = [c for c in ["id","dead_weight","build_year","vessel_type"] if c in vessels.columns]
    vs = vessels.select(vcols).rename({"id":"vessel_id"})
    s = s.join(vs, on="vessel_id", how="left")

    # derived columns
    s = s.with_columns([
        pl.col("dead_weight").cast(pl.Float64).map_elements(dwt_bucketize).alias("dwt_bucket"),
        pl.when(pl.col("is_load")==True).then(1)
         .when(pl.col("is_discharge")==True).then(0)
         .otherwise(None).alias("is_laden_after_call"),
        (2023 - pl.col("build_year").cast(pl.Int64)).alias("age"),
        pl.col("call_ts").dt.month().alias("month"),
        pl.col("call_ts").dt.weekday().alias("dow"),
    ])

    # vectorized seasonal sin/cos
    s = s.with_columns([
        (2 * np.pi * pl.col("month") / 12).sin().alias("month_sin"),
        (2 * np.pi * pl.col("month") / 12).cos().alias("month_cos"),
        (2 * np.pi * pl.col("dow") / 7).sin().alias("dow_sin"),
        (2 * np.pi * pl.col("dow") / 7).cos().alias("dow_cos"),
    ])

    keep = [
        "sample_port_call_id","vessel_id","destination","call_ts",
        "is_load","is_discharge","is_laden_after_call",
        "dead_weight","dwt_bucket","build_year","age","vessel_type",
        "prev_dist_km","last_leg_knots_est","product_family_dom",
        "month","dow","month_sin","month_cos","dow_sin","dow_cos",
    ]
    keep = [c for c in keep if c in s.columns]
    return s.select(keep)

def merge_all_features(cands_df: pl.DataFrame,
                       sample_side: pl.DataFrame,
                       split_df: pl.DataFrame) -> pl.DataFrame:
    """
    Merge sample-side features and crisis flag into candidates.
    """
    df = cands_df.join(
        split_df.select(["sample_port_call_id","is_crisis_time"]),
        on="sample_port_call_id", how="left"
    )
    df = df.join(sample_side, on="sample_port_call_id", how="left")
    # ensure dist_km exists (could be skipped to reduce cost)
    if "dist_km" not in df.columns:
        df = df.with_columns(pl.lit(0.0).alias("dist_km"))
    df = df.with_columns((pl.col("dist_km") * pl.col("is_crisis_time").cast(pl.Int8)).alias("dist_x_crisis"))
    return df
