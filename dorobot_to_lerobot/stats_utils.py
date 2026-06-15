from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _serialise_stats(values: np.ndarray) -> list[float]:
    return [float(value) for value in values.tolist()]


def _value_to_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes, dict)):
        return None
    if isinstance(value, (bool, int, float, np.integer, np.floating)):
        return np.asarray([value], dtype=np.float64)

    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if array.size == 0:
        return None
    return array


def _column_to_2d(series: pd.Series) -> np.ndarray | None:
    rows: list[np.ndarray] = []
    width: int | None = None

    for value in series:
        row = _value_to_array(value)
        if row is None:
            continue
        if width is None:
            width = int(row.size)
        if row.size != width:
            return None
        rows.append(row)

    if not rows:
        return None

    data = np.vstack(rows)
    if not np.issubdtype(data.dtype, np.number):
        return None
    finite_mask = np.isfinite(data)
    if not finite_mask.any():
        return None
    return data


def compute_episode_stats(
    parquet_path: Path,
    episode_index: int,
    columns: list[str] | None = None,
) -> dict[str, Any]:
    df = pd.read_parquet(parquet_path)
    stats: dict[str, dict[str, Any]] = {}
    selected_columns = [column for column in (columns or list(df.columns)) if column in df.columns]

    for column in selected_columns:
        data = _column_to_2d(df[column])
        if data is None:
            continue

        stats[column] = {
            "min": _serialise_stats(np.nanmin(data, axis=0)),
            "max": _serialise_stats(np.nanmax(data, axis=0)),
            "mean": _serialise_stats(np.nanmean(data, axis=0)),
            "std": _serialise_stats(np.nanstd(data, axis=0)),
        }

    return {
        "episode_index": episode_index,
        "stats": stats,
    }

