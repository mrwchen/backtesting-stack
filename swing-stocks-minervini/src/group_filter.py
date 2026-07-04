"""IBKR industry/category leadership filter.

The filter is intentionally built from the same per-symbol RS raw score used by
the main screen. Group strength is the daily median RS raw score of rankable
members, then groups are percentile-ranked 1..99. Stock leadership is the
stock's percentile rank inside its own IBKR industry/category.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


def _rating(scores: pd.DataFrame) -> pd.DataFrame:
    pct = scores.rank(axis=1, pct=True)
    return np.ceil(pct * 99).clip(1, 99)


def _clean_labels(labels: pd.Series) -> pd.Series:
    cleaned = labels.astype("string").str.strip()
    return cleaned.where(cleaned.ne(""))


def _category_keys(industry: pd.Series, category: pd.Series) -> pd.Series:
    return (industry + "\x1f" + category).where(industry.notna() & category.notna())


def _group_rating_by_symbol(
    rs_raw: pd.DataFrame,
    labels: pd.Series,
    min_symbols: int,
) -> pd.DataFrame:
    out = pd.DataFrame(np.nan, index=rs_raw.index, columns=rs_raw.columns)
    group_scores = pd.DataFrame(index=rs_raw.index)
    grouped = labels.dropna().groupby(labels.dropna(), sort=True).groups

    valid_groups: dict[str, list[str]] = {}
    for label, symbol_index in grouped.items():
        columns = [symbol for symbol in symbol_index if symbol in rs_raw.columns]
        if len(columns) < min_symbols:
            continue
        scores = rs_raw.loc[:, columns]
        valid_count = scores.notna().sum(axis=1)
        group_scores[str(label)] = scores.median(axis=1).where(valid_count >= min_symbols)
        valid_groups[str(label)] = columns

    if group_scores.empty:
        return out

    group_ratings = _rating(group_scores)
    for label, columns in valid_groups.items():
        for symbol in columns:
            out[symbol] = group_ratings[label]
    return out


def _member_rating_by_symbol(
    rs_raw: pd.DataFrame,
    labels: pd.Series,
    min_symbols: int,
) -> pd.DataFrame:
    out = pd.DataFrame(np.nan, index=rs_raw.index, columns=rs_raw.columns)
    grouped = labels.dropna().groupby(labels.dropna(), sort=True).groups

    for _, symbol_index in grouped.items():
        columns = [symbol for symbol in symbol_index if symbol in rs_raw.columns]
        if len(columns) < min_symbols:
            continue
        scores = rs_raw.loc[:, columns]
        valid_count = scores.notna().sum(axis=1)
        ratings = _rating(scores).where(valid_count >= min_symbols)
        out.loc[:, columns] = ratings
    return out


def compute_leadership(
    rs_raw: pd.DataFrame,
    universe: pd.DataFrame,
    cfg: Config,
) -> dict[str, pd.DataFrame]:
    """Return per-symbol daily IBKR group leadership ratings and pass flags."""
    symbols = rs_raw.columns
    meta = universe.drop_duplicates("symbol").set_index("symbol").reindex(symbols)
    industry = _clean_labels(meta["ibkr_industry"])
    category = _clean_labels(meta["ibkr_category"])
    category_key = _category_keys(industry, category)

    industry_rs = _group_rating_by_symbol(
        rs_raw, industry, cfg.ibkr_industry_min_symbols
    )
    category_rs = _group_rating_by_symbol(
        rs_raw, category_key, cfg.ibkr_category_min_symbols
    )
    stock_industry_rs = _member_rating_by_symbol(
        rs_raw, industry, cfg.ibkr_industry_min_symbols
    )
    stock_category_rs = _member_rating_by_symbol(
        rs_raw, category_key, cfg.ibkr_category_min_symbols
    )

    industry_pass = industry_rs >= cfg.ibkr_industry_rs_min
    category_pass = category_rs >= cfg.ibkr_category_rs_min
    stock_industry_pass = stock_industry_rs >= cfg.ibkr_stock_industry_rs_min
    stock_category_pass = stock_category_rs >= cfg.ibkr_stock_category_rs_min
    group_filter_pass = (
        industry_pass
        & category_pass
        & stock_industry_pass
        & stock_category_pass
    )

    if not cfg.ibkr_group_filter_enable:
        group_filter_pass = pd.DataFrame(True, index=rs_raw.index, columns=symbols)

    return {
        "ibkr_industry_rs_rating": industry_rs,
        "ibkr_category_rs_rating": category_rs,
        "stock_industry_rs_rating": stock_industry_rs,
        "stock_category_rs_rating": stock_category_rs,
        "ibkr_industry_pass": industry_pass,
        "ibkr_category_pass": category_pass,
        "stock_industry_pass": stock_industry_pass,
        "stock_category_pass": stock_category_pass,
        "group_filter_pass": group_filter_pass,
    }
