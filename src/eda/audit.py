from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.utils.io import read_date_info, read_product_meta, read_transactions
from src.utils.paths import REPORTS_DIR, ensure_dir
from src.utils.reporting import write_markdown


DATE_COLUMN = "CALENDAR_DATE"


@dataclass(frozen=True)
class DataframeAudit:
    name: str
    rows: int
    columns: int
    duplicate_rows: int
    missing_cells: int
    date_min: str | None = None
    date_max: str | None = None
    date_unique: int | None = None


def summarize_dataframe(df: pd.DataFrame, name: str, date_column: str | None = None) -> DataframeAudit:
    date_min = date_max = date_unique = None
    if date_column and date_column in df.columns:
        series = df[date_column]
        date_min = str(series.min())
        date_max = str(series.max())
        date_unique = int(series.nunique())
    return DataframeAudit(
        name=name,
        rows=int(len(df)),
        columns=int(df.shape[1]),
        duplicate_rows=int(df.duplicated().sum()),
        missing_cells=int(df.isna().sum().sum()),
        date_min=date_min,
        date_max=date_max,
        date_unique=date_unique,
    )


def format_counts(series: pd.Series, top_n: int = 10) -> list[str]:
    counts = series.value_counts(dropna=False).head(top_n)
    return [f"- {idx}: {int(counts[idx])}" for idx in counts.index]


def build_audit_sections(
    date_info: pd.DataFrame, transactions: pd.DataFrame, product: pd.DataFrame
) -> Iterable[tuple[str, list[str]]]:
    sections: list[tuple[str, list[str]]] = []

    audits = [
        summarize_dataframe(date_info, "Date Info", DATE_COLUMN),
        summarize_dataframe(transactions, "Transactions", DATE_COLUMN),
        summarize_dataframe(product, "Product Meta", None),
    ]
    overview_lines = []
    for audit in audits:
        overview_lines.append(
            f"- {audit.name}: {audit.rows} rows, {audit.columns} columns, "
            f"{audit.duplicate_rows} duplicate rows, {audit.missing_cells} missing cells"
        )
        if audit.date_min:
            overview_lines.append(
                f"  - {audit.name} date range: {audit.date_min} to {audit.date_max} "
                f"({audit.date_unique} unique dates)"
            )
    sections.append(("Dataset Overview", overview_lines))

    if "HOLIDAY" in date_info.columns:
        holiday_missing = (date_info["HOLIDAY"].isna() | date_info["HOLIDAY"].eq("NULL")).sum()
        holiday_lines = [
            f"- Missing/NULL holidays: {int(holiday_missing)}",
            "- Top holiday values:",
            *format_counts(date_info["HOLIDAY"], top_n=12),
        ]
        sections.append(("Holiday Coverage", holiday_lines))

    if "SELL_ID" in transactions.columns:
        sections.append(("Top SELL_ID Counts", format_counts(transactions["SELL_ID"], top_n=10)))

    if "SELL_CATEGORY" in transactions.columns:
        sections.append(
            ("SELL_CATEGORY Counts", format_counts(transactions["SELL_CATEGORY"], top_n=10))
        )

    return sections


def run_audit(output_path: Path | None = None) -> Path:
    date_info = read_date_info()
    transactions = read_transactions()
    product = read_product_meta()

    ensure_dir(REPORTS_DIR)
    output_path = output_path or (REPORTS_DIR / "eda_audit.md")
    sections = build_audit_sections(date_info, transactions, product)
    write_markdown(output_path, "EDA Audit Summary", sections)
    return output_path


if __name__ == "__main__":
    path = run_audit()
    print(f"Wrote audit report to {path}")
