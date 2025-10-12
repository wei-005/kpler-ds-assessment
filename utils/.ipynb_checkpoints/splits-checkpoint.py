
import polars as pl
from datetime import datetime
from .config import PROCESSED_DIR

# 可调整: 时间切分边界（闭区间）
#Adjustable: temporal split boundaries (closed intervals)
TRAIN_START = pl.datetime(2023,1,1)
TRAIN_END   = pl.datetime(2023,9,30,23,59,59)
VAL_START   = pl.datetime(2023,10,1)
VAL_END     = pl.datetime(2023,11,30,23,59,59)
TEST_START  = pl.datetime(2023,12,1)
TEST_END    = pl.datetime(2023,12,31,23,59,59)

CRISIS_START = pl.datetime(2023,10,20)

def _ensure_dt(df: pl.DataFrame, col: str = "call_ts") -> pl.DataFrame:
    if df.schema.get(col) == pl.Utf8:
        df = df.with_columns(pl.col(col).str.strptime(pl.Datetime, strict=False))
    return df

def temporal_split(samples: pl.DataFrame, ts_col: str="call_ts"):
    s = _ensure_dt(samples, ts_col)
    train = s.filter((pl.col(ts_col) >= TRAIN_START) & (pl.col(ts_col) <= TRAIN_END))
    val   = s.filter((pl.col(ts_col) >= VAL_START)   & (pl.col(ts_col) <= VAL_END))
    test  = s.filter((pl.col(ts_col) >= TEST_START)  & (pl.col(ts_col) <= TEST_END))
    return train, val, test

def add_crisis_flag(df: pl.DataFrame, ts_col: str="call_ts") -> pl.DataFrame:
    df = _ensure_dt(df, ts_col)
    return df.with_columns((pl.col(ts_col) >= CRISIS_START).alias("is_crisis_time"))
