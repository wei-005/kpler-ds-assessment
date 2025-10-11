
from pathlib import Path
import os

# 可调整: 默认数据目录（raw/interim/processed），可在 Notebook 顶部用 set_* 动态覆盖
#Adjustable: default data dirs (raw/interim/processed). Override via set_* at the top of notebooks.
PROJ_ROOT = Path.cwd()
if (PROJ_ROOT.name.lower() == "notebooks" and (PROJ_ROOT.parent/"utils").exists()):
    PROJ_ROOT = PROJ_ROOT.parent

DATA_DIR = Path(os.getenv("KPLER_DATA_DIR", "")) if os.getenv("KPLER_DATA_DIR") else (PROJ_ROOT / "data" / "raw")
INTERIM_DIR = Path(os.getenv("KPLER_INTERIM_DIR", "")) if os.getenv("KPLER_INTERIM_DIR") else (PROJ_ROOT / "data" / "interim")
PROCESSED_DIR = Path(os.getenv("KPLER_PROCESSED_DIR", "")) if os.getenv("KPLER_PROCESSED_DIR") else (PROJ_ROOT / "data" / "processed")

for d in [DATA_DIR, INTERIM_DIR, PROCESSED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def set_data_dir(path_str: str):
    """
    可调整: 运行时切换原始数据目录（需包含 vessels.csv / port_calls.csv / trades.csv）
    #Adjustable: change raw data dir at runtime (must contain vessels.csv / port_calls.csv / trades.csv)
    """
    global DATA_DIR
    DATA_DIR = Path(path_str)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

def set_output_dirs(interim: str = None, processed: str = None):
    """
    可调整: 运行时切换中间产物与输出目录
    #Adjustable: change interim/processed dirs at runtime
    """
    global INTERIM_DIR, PROCESSED_DIR
    if interim:
        INTERIM_DIR = Path(interim); INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    if processed:
        PROCESSED_DIR = Path(processed); PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
