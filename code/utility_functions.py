import logging
from pathlib import Path

import pandas as pd
import numpy as np

from default_assumptions import REG_REG_CLASSES


# ===== DATA FORMATTING / CONVERSION =====

def clean_labels_for_plotting(df, cols):
    df = df.copy()
    for col in cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace("_", " ", regex=False)
            .str.replace("df", "DF", regex=False)
            .str.replace("lng", "LNG", regex=False)
            .str.replace("mgo", "MGO", regex=False)
            .str.replace("lsfo", "fuel oil", regex=False)
            .str.replace("ccs", "CCS", regex=False)
            .str.replace(" ice ", " ", regex=False)
            .str.replace("bio-methanol", "bio methanol/bio-ethanol", regex=False)
        )
    return df

def reduce_size_df(df):
    # used to reduce the size of dataframes when they are too big for the Mem
    # Iterate directly over columns to avoid memory-intensive select_dtypes() copy
    for col in df.columns:
        dtype = df[col].dtype
        if dtype == "float64":
            df[col] = df[col].astype("float32")
        elif dtype == "int64":
            df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def extend_years(df, max_year_needed, year_col='year'):
    """
    # Reusable helper to extend a dataframe to a given year ---
    Extend a dataframe by copying the last year’s rows for all missing years up to max_year_needed.
    """
    df = df.copy()
    max_year_in_df = df[year_col].max()
    if max_year_needed > max_year_in_df:
        last_year_data = df[df[year_col] == max_year_in_df].copy()
        max_year_needed = int(max_year_needed)
        max_year_in_df = int(max_year_in_df)
        for y in range(max_year_in_df + 1, max_year_needed + 1):
            new_row = last_year_data.copy()
            new_row[year_col] = y
            df = pd.concat([df, new_row], ignore_index=True)
    return df

def weighted_percentile(values, weights, percentile):
    """
    Compute a weighted percentile (no NaNs).
    values: 1D array
    weights: 1D array of same length
    percentile: between 0 and 100
    """
    values = np.array(values)
    weights = np.array(weights)

    # Sort the data and weights by the values
    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]

    # Compute cumulative weight fraction
    cdf = np.cumsum(weights) / np.sum(weights)

    # Interpolate to find the percentile value
    return np.interp(percentile / 100, cdf, values)

def prepare_logger(DEBUG: bool, debug_file: str) -> logging.Logger | None:
    """
    Configure and return a logger if DEBUG is True, otherwise return None.
    """
    if not DEBUG:
        return None

    debug_path = Path(debug_file)
    debug_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("DEBUG_LOGGER")
    logger.setLevel(logging.DEBUG)

    # Clear previous handlers so we don't duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    fh = logging.FileHandler(debug_file, mode="w")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

def clean_results_for_plotting(
    score_results: pd.DataFrame,
    history_full: pd.DataFrame,
    op_pivot: pd.DataFrame,
    drop_in_fuels: pd.DataFrame,
    rep_ships: pd.DataFrame,
    designs: list,
    variables_filter: list,  # e.g. ["foresight", "retrofit_scenario"]
    values_filter: list,  # e.g. [3, "Base"]
    imo_type: str,
    size_id: int,
    current_year: int,
):
    """Load and clean outputs of CRM runs and prepare for plotting using a flexible filter system"""

    # ----------------------------------------------------
    # Build general filters
    # ----------------------------------------------------
    filter_dict = dict(zip(variables_filter, values_filter))
    # example result: {"foresight": 3, "retrofit_scenario": "Base"}
    print(f"Filtering results by: {filter_dict}")
    # Build mask
    mask = pd.Series(True, index=history_full.index)
    for col, val in filter_dict.items():
        mask &= history_full[col] == val
        print(val, len(history_full[mask]))

    # ----------------------------------------------------
    # Historical filtering (replaces all hard-coded filters)
    # ----------------------------------------------------
    history_df_copy = history_full[mask].merge(
        rep_ships[
            [
                "imo_type",
                "Size_ID",
                "size_unit",
                "dwt",
                "mcr",
                "bunker_t_lsfo_e",
                "size",
                "n_stroke",
                "generation",
                "scrap_year",
                "ship_id",
            ]
        ],
        on="ship_id",
        how="left",
    )

    filtered_history = history_full[mask]
    assert len(history_df_copy) == len(
        filtered_history
    ), "history_df_copy length mismatch"
    assert len(history_df_copy) > 0, "empty history_df_copy for specific scenario"

    history_df_copy = history_df_copy[
        (history_df_copy.imo_type == imo_type) & (history_df_copy.Size_ID == size_id)
    ]
    check_len = len(history_df_copy)
    assert len(history_df_copy) > 0, "empty history_df_copy for specific ship"

    # Merge score results
    score_mask = pd.Series(True, index=score_results.index)
    for col, val in filter_dict.items():
        if col in score_results.columns:
            score_mask &= score_results[col] == val

    history_df_copy = history_df_copy.merge(
        score_results[score_mask][["ship_id", "capex_premium"]],
        on="ship_id",
        how="left",
    )
    # only add the capex premium in the first year
    mask = history_df_copy[history_df_copy.year != current_year].index
    history_df_copy.loc[mask, "capex_premium"] = 0

    assert check_len == len(history_df_copy)
    assert (
        history_df_copy["capex_premium"].isna().sum() == 0
    ), "NaNs in capex_premium"

    # ----------------------------------------------------
    # Cheapest fuels logic
    # ----------------------------------------------------

    cheapest_fuels = pd.DataFrame()
    for fd in designs:
        allowed_fuels = drop_in_fuels[
            (drop_in_fuels.drop_in == True) & (drop_in_fuels.fuel_design == fd)
        ].fuel
        if len(allowed_fuels) > 0:
            to_add = op_pivot[allowed_fuels].idxmin(axis=1).reset_index()
            to_add["fuel_end"] = fd
            cheapest_fuels = pd.concat((cheapest_fuels, to_add))

    cheapest_fuels = cheapest_fuels.rename(columns={0: "fuel"})

    df = history_df_copy[
        ["scenario_id", "year", "initial_fuel_design", "fuel_end", "engine_type_end"]
    ].merge(
        cheapest_fuels.rename(columns={"engine_type": "engine_type_end"}),
        on=["scenario_id", "year", "fuel_end", "engine_type_end"],
        how="left",
    )
    assert len(df) == len(history_df_copy) * len(REG_REG_CLASSES), f"Unexpected length of df after merge with cheapest fuels: {len(df) , len(history_df_copy) * len(REG_REG_CLASSES)}"

    df = df.merge(
        history_df_copy[
            [
                "scenario_id",
                "year",
                "initial_fuel_design",
                "op_cost",
                "retrofitting_cost",
                "opportunity_cost",
                'op_cost_discounted',
                'retrofitting_cost_discounted',
                'opportunity_cost_discounted',
                "capex_premium",
                "# duplicates",
            ]
        ],
        on=["scenario_id", "year", "initial_fuel_design"],
        how="left",
    )
    assert len(df) == len(history_df_copy) * len(REG_REG_CLASSES), "Unexpected length of df after merge with history_df_copy"

    return df
