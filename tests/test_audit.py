import pandas as pd

from src.eda.audit import format_counts, summarize_dataframe


def test_summarize_dataframe_basic():
    df = pd.DataFrame(
        {
            "CALENDAR_DATE": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "A": [1, None],
            "B": [1, 1],
        }
    )
    audit = summarize_dataframe(df, "test", "CALENDAR_DATE")
    assert audit.rows == 2
    assert audit.columns == 3
    assert audit.duplicate_rows == 0
    assert audit.missing_cells == 1
    assert audit.date_unique == 2


def test_format_counts_basic():
    series = pd.Series(["x", "y", "x"])
    lines = format_counts(series, top_n=2)
    assert lines[0].startswith("- x: 2")
