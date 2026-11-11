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

# Nature Sustainability figure specs: 88 mm (1 column) / 180 mm (2 columns) width,
# text 5-7 pt sans-serif, editable vector PDFs (fonttype 42).
MM = 1 / 25.4
FIG_1COL = 88 * MM
FIG_2COL = 180 * MM

# Max figure height for captions under 150 words (NS artwork guide):
# 180 mm for 1-column (88 mm wide) figures, 210 mm for 2-column (180 mm wide) figures.
MAX_H_1COL = 180 * MM
MAX_H_2COL = 210 * MM

NATURE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 6,
    "axes.labelsize": 6,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "legend.title_fontsize": 7,
    "axes.linewidth": 0.5,
    "grid.linewidth": 0.3,
    "lines.linewidth": 1,
    "lines.markersize": 3,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "pdf.fonttype": 42,  # embed TrueType so text stays editable in Illustrator
    "ps.fonttype": 42,
    "savefig.dpi": 300,
}


def nature_style():
    """Apply the Nature Sustainability figure style (fonts, sizes, line widths)."""
    plt.close("all")
    sns.set_theme(style="whitegrid", rc=NATURE_RC)


def wrap_ticks(ax, width=10):
    """Wrap long x tick labels onto multiple lines to keep panels narrow."""
    ax.set_xticklabels(
        [textwrap.fill(t.get_text(), width) for t in ax.get_xticklabels()]
    )


def save_figure(fig, name, source_data):
    """Save the figure as a vector PDF and its underlying data as a source CSV."""
    out = Path(FIGURE_PATH)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    source_data.to_csv(out / f"{name}_source_data.csv", index=False)
    print(f"Saved {name}.pdf and {name}_source_data.csv to {out}")

def boxplot_summary(data, group_cols, value_col):
    """Summary statistics matching what sns.boxplot draws: Q1/median/Q3, the
    whisker ends (largest/smallest value within 1.5 x IQR) and outlier count.
    """
    from matplotlib.cbook import boxplot_stats

    rows = []
    for keys, g in data.groupby(group_cols, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = g[value_col].dropna().values
        stats = boxplot_stats(values)[0]
        rows.append(
            {
                **dict(zip(group_cols, keys)),
                "n": len(values),
                "min_within_1_5_iqr": stats["whislo"],
                "q1": stats["q1"],
                "median": stats["med"],
                "q3": stats["q3"],
                "max_within_1_5_iqr": stats["whishi"],
                "n_outliers": len(stats["fliers"]),
            }
        )
    return pd.DataFrame(rows)
