import polars as pl
import pandas as pd

#Adjustable: max steps to look ahead for discharge after a load
MAX_LOOKAHEAD = 8

def infer_trades_from_port_calls(pc: pl.DataFrame) -> pl.DataFrame:
    """
    Greedy match load→discharge using is_load/is_discharge to infer trades (prototype).
    output columns: [origin_id, destination_id, vessel_id, start_ts, end_ts, inferred_confidence]
    """
    cols = ["id","vessel_id","start_utc","end_utc","destination","destination_latitude","destination_longitude","is_load","is_discharge"]
    need = [c for c in cols if c in pc.columns]
    df = pc.select(need).sort(["vessel_id","start_utc"])
    pdf = df.to_pandas()

    rows = []
    for vid, g in pdf.groupby("vessel_id"):
        g = g.reset_index(drop=True)
        for i, row in g.iterrows():
            if not bool(row.get("is_load", False)): 
                continue
            # Find the first unloading afterwords
            for j in range(i+1, min(i+1+MAX_LOOKAHEAD, len(g))):
                if bool(g.loc[j,"is_discharge"]):
                    rows.append({
                        "port_call_origin_id": int(row["id"]),
                        "port_call_destination_id": int(g.loc[j,"id"]),
                        "vessel_id": int(vid),
                        "start_date_time": row["end_utc"],
                        "end_date_time": g.loc[j,"start_utc"],
                        "inferred_confidence": 0.7  # Simplified: Fixed confidence level, which can be further refined by volume/duration/distance
                    })
                    break
    return pl.from_pandas(pd.DataFrame(rows))

def calibrate_with_official(inferred: pl.DataFrame, official: pl.DataFrame) -> pl.DataFrame:
    """
    Align with official trades and report hits (by origin_id and by (origin,dst) pair).
    """
    key_cols = ["port_call_origin_id","port_call_destination_id"]
    inf = inferred.select(key_cols).unique()
    off = official.select(key_cols).unique()
    hit_pair = inf.join(off, on=key_cols, how="inner").height

    # Only hit by origin
    hit_origin = inferred.select(["port_call_origin_id"]).unique().join(
        official.select(["port_call_origin_id"]).unique(),
        on="port_call_origin_id", how="inner"
    ).height

    return pl.DataFrame({
        "metric":["pair_exact","origin_only"],
        "value":[hit_pair, hit_origin]
    })
