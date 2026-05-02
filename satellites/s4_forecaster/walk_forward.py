import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date, timedelta
from loguru import logger
from dataclasses import dataclass


@dataclass
class WalkForwardSplit:
    split_id:    int
    train_start: str
    train_end:   str
    val_start:   str
    val_end:     str


def generate_splits(
    start_date: str = "2020-06-01",
    end_date:   str = "2025-12-31",
    train_months: int = 18,
    val_months:   int = 3,
    step_months:  int = 3,
    embargo_days: int = 5,
) -> list[WalkForwardSplit]:
    from dateutil.relativedelta import relativedelta

    splits = []
    split_id = 0

    train_start = pd.Timestamp(start_date)
    final_end   = pd.Timestamp(end_date)

    while True:
        train_end = train_start + relativedelta(months=train_months)
        val_start = train_end + timedelta(days=embargo_days)
        val_end   = val_start + relativedelta(months=val_months)

        if val_end > final_end:
            break

        splits.append(WalkForwardSplit(
            split_id    = split_id,
            train_start = train_start.strftime("%Y-%m-%d"),
            train_end   = train_end.strftime("%Y-%m-%d"),
            val_start   = val_start.strftime("%Y-%m-%d"),
            val_end     = val_end.strftime("%Y-%m-%d"),
        ))

        train_start += relativedelta(months=step_months)
        split_id += 1

    logger.info(f"[S4] Generated {len(splits)} walk-forward splits")
    for s in splits:
        logger.debug(
            f"  Split {s.split_id}: "
            f"train [{s.train_start}→{s.train_end}] "
            f"val [{s.val_start}→{s.val_end}]"
        )

    return splits


def load_split_data(
    split: WalkForwardSplit,
    features_df: pd.DataFrame,
    targets_df:  pd.DataFrame,
    regime_map:  dict,
) -> tuple:
    # Merge features with targets
    merged = features_df.merge(targets_df, on=["symbol", "date"], how="inner")

    # Add regime
    merged["date_ts"] = pd.to_datetime(merged["date"])
    merged["regime"]  = merged["date_ts"].map(
        lambda d: regime_map.get(d.strftime("%Y-%m-%d"), -1)
    )
    merged = merged[merged["regime"] != -1]

    # Split by date
    train_mask = (
        (merged["date"] >= split.train_start) &
        (merged["date"] <  split.train_end)
    )
    val_mask = (
        (merged["date"] >= split.val_start) &
        (merged["date"] <  split.val_end)
    )

    train = merged[train_mask].copy()
    val   = merged[val_mask].copy()

    feature_cols = [
        c for c in features_df.columns
        if c not in ["symbol", "date", "close"]
    ]

    X_train = train[feature_cols].values
    y_train = train["target"].values
    X_val   = val[feature_cols].values
    y_val   = val["target"].values

    regime_train = train["regime"].values
    regime_val   = val["regime"].values

    return X_train, y_train, X_val, y_val, regime_train, regime_val, feature_cols


if __name__ == "__main__":
    splits = generate_splits()
    print(f"\nTotal splits: {len(splits)}")
    print(f"\nFirst split:")
    print(f"  Train: {splits[0].train_start} → {splits[0].train_end}")
    print(f"  Val:   {splits[0].val_start} → {splits[0].val_end}")
    print(f"\nLast split:")
    print(f"  Train: {splits[-1].train_start} → {splits[-1].train_end}")
    print(f"  Val:   {splits[-1].val_start} → {splits[-1].val_end}")