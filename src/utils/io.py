from pathlib import Path
import pandas as pd

from .paths import RAW_DIR


DATE_FORMAT = "%m/%d/%y"
DATE_COLUMN = "CALENDAR_DATE"

RAW_DATE_INFO = "Cafe+-+DateInfo.csv"
RAW_TRANSACTIONS = "Cafe+-+Transaction+-+Store.csv"
RAW_PRODUCT_META = "Cafe+-+Sell+Meta+Data.csv"


def _resolve(path: str | Path) -> Path:
    return Path(path) if isinstance(path, Path) else RAW_DIR / path


def read_date_info(path: str | Path = RAW_DATE_INFO) -> pd.DataFrame:
    df = pd.read_csv(_resolve(path))
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], format=DATE_FORMAT)
    return df


def read_transactions(path: str | Path = RAW_TRANSACTIONS) -> pd.DataFrame:
    df = pd.read_csv(_resolve(path))
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], format=DATE_FORMAT)
    return df


def read_product_meta(path: str | Path = RAW_PRODUCT_META) -> pd.DataFrame:
    return pd.read_csv(_resolve(path))
