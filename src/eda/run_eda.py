from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.eda.audit import run_audit
from src.eda.collinearity import build_master, run_collinearity
from src.utils.paths import FIGURES_DIR, ensure_dir


DATE_COLUMN = "CALENDAR_DATE"


def generate_plots(master: pd.DataFrame, output_dir: Path) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as exc:
        raise RuntimeError("matplotlib and seaborn are required for plotting") from exc

    ensure_dir(output_dir)
    outputs: list[Path] = []
    sns.set_theme(style="whitegrid")

    def save_plot(fig, name: str) -> None:
        path = output_dir / name
        fig.savefig(path, bbox_inches="tight")
        outputs.append(path)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(master["PRICE"], bins=30, ax=ax)
    ax.set_title("Price Distribution")
    save_plot(fig, "price_distribution.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(master["QUANTITY"], bins=30, ax=ax)
    ax.set_title("Quantity Distribution")
    save_plot(fig, "quantity_distribution.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(master["revenue"], bins=30, ax=ax)
    ax.set_title("Revenue Distribution")
    save_plot(fig, "revenue_distribution.png")
    plt.close(fig)

    daily = (
        master.groupby([DATE_COLUMN, "SELL_ID"], as_index=False)["revenue"]
        .sum()
        .sort_values(DATE_COLUMN)
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    palette = ["#d95f02", "#1b9e77", "#7570b3", "#e7298a"]  # high-contrast manual palette
    sns.lineplot(data=daily, x=DATE_COLUMN, y="revenue", hue="SELL_ID", palette=palette, ax=ax)
    ax.set_title("Daily Revenue by SKU")
    ax.legend(title="SELL_ID", fontsize="small")
    save_plot(fig, "daily_revenue_by_sku.png")
    plt.close(fig)

    holiday_summary = (
        master.assign(is_holiday=master["HOLIDAY"] != "Not Holiday")
        .groupby("is_holiday", as_index=False)["revenue"]
        .mean()
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(data=holiday_summary, x="is_holiday", y="revenue", ax=ax)
    ax.set_title("Avg Revenue: Holiday vs Non-Holiday")
    ax.set_xlabel("Is Holiday")
    save_plot(fig, "holiday_vs_nonholiday_revenue.png")
    plt.close(fig)

    return outputs


def run_all() -> dict[str, list[Path] | Path | dict[str, Path]]:
    report_path = run_audit()
    collinearity_paths = run_collinearity()

    master = build_master()
    master["revenue"] = master["PRICE"] * master["QUANTITY"]
    plot_paths = generate_plots(master, FIGURES_DIR)

    return {"audit": report_path, "collinearity": collinearity_paths, "plots": plot_paths}


if __name__ == "__main__":
    outputs = run_all()
    print(f"Wrote audit report to {outputs['audit']}")
    for name, path in outputs["collinearity"].items():
        print(f"Wrote {name} to {path}")
    for path in outputs["plots"]:
        print(f"Wrote plot to {path}")
