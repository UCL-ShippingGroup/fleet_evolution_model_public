import itertools
import logging
import numpy as np
import pandas as pd
from scipy.stats import qmc, triang
from tqdm import tqdm
from default_assumptions import COLS_SEGMENT_SIZE_ID
from utility_functions import prepare_logger, reduce_size_df
from typing import Dict, Tuple, Any, List
from technology import get_available_retrofits
from fuel import calculate_fuel_consumption, create_fuel_mix_on_the_base, apply_engine_efficiency_correction, calculate_emissions
from policy import calculate_policy_compliance

# Column renaming mapping (from notebook cell 5)
COLUMN_RENAMES = {
    "reward": "reward_scenario",
    "fuel_prices": "fuel_cost_scenario",
    "ru_prices": "ru_scenario",
    "znz": "reward_threshold_gco2_mj",
    "imo_type": "ship_type",
    "fuel_design": "engine",
    "fuel_design": "engine_simple",
    "ets_price": "EU_ETS",
}

def annual_transport_work(df):
    # comute annual tonne-miles of transport work
    needed_cols = ['Avg. distance (nm)','dwt','utilisation']
    for c in needed_cols:
        assert c in df.columns, f"{c} missing from the columns"
        assert not np.any(df[c].isna()), f"NaNs found in {c}: {df[df[c].isna()].sample(3)}"
    return df['Avg. distance (nm)'] * df['dwt'] * df['utilisation']

def rename_columns(df, rename_dict=COLUMN_RENAMES):
    """
    Rename columns in a DataFrame according to a standard mapping.
    Only columns present in the DataFrame will be renamed.
    Uses copy=False to avoid memory-intensive copy operations.
    """
    # Filter dict to only existing columns
    filtered_rename = {k: v for k, v in rename_dict.items() if k in df.columns}
    return df.rename(columns=filtered_rename, copy=False)

def process_operating_cost_batch(
    op_cost_batch,
    drop_in_fuels_in,
    pilot_fuel_in,
    engines_design_mapping_in,
    fuel_engine_mapping_in,
    WIND_TECHNO_LIST,
    WIND_ENERGY_SAVINGS,
    energy_density,
    PILOT_FUELS,
    policy_compliance_params_cols,
    ccs_rate,
    CLEAN=True
):
    """
    Process a batch of scenarios through the full operating cost pipeline.
    
    Takes pre-expansion data (subset of op_cost_batch_in) and applies:
    1. Wind/CCS expansion
    2. Manual amendments (e-fuels filtering, FuelEU removal)
    3. Full operating cost pipeline (fuel costs -> emissions -> fuel mixes -> compliance)
    
    Uses COLUMN_RENAMES defined at module level.
    
    Returns:
        Tuple of:
        - operating_cost_batch: Final operating cost DataFrame for this batch
        - drop_in_fuels: Fully expanded drop-in fuels mapping (wind/CCS + fuel mixes)
    """
    import gc
    
    # ============= WIND/CCS EXPANSION =============
    # Reshape drop-in fuels
    drop_in_fuels_temp = drop_in_fuels_in.melt(
        id_vars="fuel",
        var_name="fuel_design",
        value_name="drop_in",
    )
    
    # Make working copies for expansion
    drop_in_fuels_temp_temp = drop_in_fuels_temp.copy()
    op_cost_batch_in = op_cost_batch.copy()
    op_cost_temp = op_cost_batch_in.copy()
    fuel_engine_mapping = fuel_engine_mapping_in.copy()
    engines_design_mapping = engines_design_mapping_in.copy()
    pilot_fuel_input = pilot_fuel_in.copy()
    
    # Helper function for adding tags
    def add_tag(df, column, tag):
        """Return a copy of df with `column = column + ' + ' + tag`."""
        out = df.copy()
        out[column] = out[column] + "_+_" + tag
        return out
    
    # ============= WIND EXPANSIONS =============
    for wind in WIND_TECHNO_LIST:
        drop_in_fuels_temp_temp = pd.concat(
            [
                drop_in_fuels_temp_temp,
                add_tag(drop_in_fuels_temp, "fuel", wind).assign(
                    fuel_design=lambda x: x["fuel_design"] + "_+_" + wind
                ),
            ],
            ignore_index=True,
        )

        op_cost_temp = pd.concat(
            [op_cost_temp, add_tag(op_cost_batch_in, "fuel", wind).assign(wind=wind)],
            ignore_index=True,
        )

        fuel_engine_mapping = pd.concat(
            [fuel_engine_mapping, add_tag(fuel_engine_mapping_in, "fuel", wind)],
            ignore_index=True,
        )

        engines_design_mapping = pd.concat(
            [
                engines_design_mapping,
                add_tag(engines_design_mapping_in, "fuel_design", wind),
            ],
            ignore_index=True,
        )

        to_add = add_tag(pilot_fuel_in, "fuel", wind)
        to_add["fuel_savings"] = WIND_ENERGY_SAVINGS[WIND_ENERGY_SAVINGS.wind == wind][
            "fuel_savings"
        ].iloc[0]
        pilot_fuel_input = pd.concat(
            [
                pilot_fuel_input,
                to_add,
            ],
            ignore_index=True,
        )

    # Correct the share of pilot fuel by the share of wind
    pilot_fuel_input["fuel_savings"] = pilot_fuel_input["fuel_savings"].fillna(0)
    pilot_fuel_input["share_pilot_mj"] *= 1 - pilot_fuel_input["fuel_savings"]

    # ============= CCS EXPANSIONS (only for conventional / df_ice_lng) =============
    conv_lng_designs = ["conventional", "df_ice_lng"]
    conv_lng_fuels = drop_in_fuels_temp[
        (drop_in_fuels_temp.fuel_design.isin(conv_lng_designs))
        & (drop_in_fuels_temp.drop_in)
    ].fuel.unique()

    ccs_tag = "ccs"

    # drop_in_fuels
    tmp = drop_in_fuels_temp[
        drop_in_fuels_temp.fuel.isin(conv_lng_fuels)
        & drop_in_fuels_temp.fuel_design.isin(conv_lng_designs)
    ]
    drop_in_fuels_temp_temp = pd.concat(
        [
            drop_in_fuels_temp_temp,
            add_tag(tmp, "fuel", ccs_tag).assign(
                fuel_design=lambda x: x["fuel_design"] + "_+_" + ccs_tag
            ),
        ],
        ignore_index=True,
    )

    tmp = op_cost_batch_in[op_cost_batch_in.fuel.isin(conv_lng_fuels)]
    op_cost_temp = pd.concat(
        [op_cost_temp, add_tag(tmp, "fuel", ccs_tag)],
        ignore_index=True,
    )
    op_cost_temp["wind"] = op_cost_temp["wind"].fillna("")

    # fuel_engine_mapping
    tmp = fuel_engine_mapping_in[fuel_engine_mapping_in.fuel.isin(conv_lng_fuels)]
    fuel_engine_mapping = pd.concat(
        [fuel_engine_mapping, add_tag(tmp, "fuel", ccs_tag)],
        ignore_index=True,
    )

    # engines_design_mapping
    tmp = engines_design_mapping_in[
        engines_design_mapping_in.fuel_design.isin(conv_lng_designs)
    ]
    engines_design_mapping = pd.concat(
        [engines_design_mapping, add_tag(tmp, "fuel_design", ccs_tag)],
        ignore_index=True,
    )

    # pilot_fuel_input
    tmp = pilot_fuel_input[pilot_fuel_input.fuel.isin(conv_lng_fuels)]
    pilot_fuel_input = pd.concat(
        [pilot_fuel_input, add_tag(tmp, "fuel", ccs_tag)],
        ignore_index=True,
    )

    fuels_in_need_of_pilot = pilot_fuel_input.fuel.unique()
    drop_in_fuels_base = drop_in_fuels_temp_temp
    avail_fuels_no_mix = drop_in_fuels_base.fuel.unique()
    
    # Rename columns
    op_cost_temp = op_cost_temp.rename(
        columns={
            "fuel_prices_value": "main_fuel_price_usd_tvslfoeq",
            "ru1": "ru1_price",
            "ru2": "ru2_price",
            "su_value": "su_price",
            "reward_value": "reward_usd_tlsfoeq",
            "carbon_intensity_value": "wtw_gco2e_p_mj",
            "wtt_ef_value": "wtt_gco2e_p_mj",
            "fueleu_penalty": "FuelEU_penalty",
        }
    )

    # Now op_cost_temp is the expanded, amended batch ready for processing
    op_cost_batch = op_cost_temp

    if CLEAN:
        del op_cost_batch_in, drop_in_fuels_temp, tmp
        gc.collect()
    
    # ============= Calculate fuel costs =============
    to_input = rename_columns(op_cost_batch, COLUMN_RENAMES)
    to_input["fuel_consumption_mj_year"] = 1_000
    to_input["pilot_fuel_price_usd_tvslfoeq"] = 0
    
    out_cons = calculate_fuel_consumption(
        params=to_input[["fuel_consumption_mj_year", "fuel"]],
        fuel=op_cost_batch["fuel"],
        fuel_result=to_input,
        df_pilot_fuel_input_ref=pilot_fuel_input,
        INCLUDE_PILOT_FUELS=False,
    )
    
    for var in ["total_fuel_cost", "fuel_savings", "main_fuel_consumption_mj_year", "pilot_fuel_consumption_mj_year"]:
        assert len(op_cost_batch) == len(out_cons[var])
        op_cost_batch[var] = out_cons[var]
        assert not np.any(
            op_cost_batch[var].isna()
        ), f"NaNs found in {var}"
    
    # correct the max capture rate for conventional to 30% (from Matrix_CCS VP)
    op_cost_batch["ccs_rate"] = ccs_rate
    mask = op_cost_batch[op_cost_batch.engine_type == "All"].index
    if ccs_rate == 0.5:
        op_cost_batch.loc[mask, "ccs_rate"] = 0.3
    elif ccs_rate == 0.2:
        op_cost_batch.loc[mask, "ccs_rate"] = 0.1

    op_cost_batch["main_emissions_factor_wtt_gco2eq_mj"] = op_cost_batch["wtt_gco2e_p_mj"]
    op_cost_batch["main_emissions_factor_ttw_gco2eq_mj"] = op_cost_batch["ttw_gco2e_p_mj"]
    op_cost_batch["pilot_emissions_factor_wtt_gco2eq_mj"] = 0
    op_cost_batch["pilot_emissions_factor_ttw_gco2eq_mj"] = 0

    # Calculate emissions
    out_emissions = calculate_emissions(
        op_cost_batch,
        # op_cost_batch["wtt_gco2e_p_mj"],
        # op_cost_batch["ttw_gco2e_p_mj"],
        # 0, 0,
        op_cost_batch["ccs_rate"],
    )
    assert len(out_emissions["overall_emissions_factor_wtw_gco2eq_mj"]) == len(out_emissions["overall_emissions_factor_ttw_gco2eq_mj"]) == len(op_cost_batch)
    op_cost_batch["wtw_gco2e_p_mj"] = out_emissions["overall_emissions_factor_wtw_gco2eq_mj"]
    op_cost_batch["ttw_gco2e_p_mj"] = out_emissions["overall_emissions_factor_ttw_gco2eq_mj"]
    assert not np.any(op_cost_batch["wtw_gco2e_p_mj"].isna()), "NaNs found in wtw_gco2e_p_mj"

    op_cost_batch = reduce_size_df(op_cost_batch)
    if CLEAN:
        del to_input, out_cons, out_emissions
        gc.collect()
    
    # ============= Add fuel mixes =============
    op_template = op_cost_batch[op_cost_batch.fuel.isin(avail_fuels_no_mix)].copy()
    drop_in_fuels = drop_in_fuels_base[drop_in_fuels_base.fuel.isin(avail_fuels_no_mix)].copy()
    drop_template = drop_in_fuels.copy()

    for gfi_var, ci_var in [
        ("base_compliance_target", "wtw_gco2e_p_mj"),
        ("FuelEU", "wtw_gco2e_p_mj"),
    ]:
        # create a dataframe of the cheapest fuel mix to operate on the base
        fuel_mix, drop_in_fuels_temp = create_fuel_mix_on_the_base(
            op_template, drop_template, pilot_fuel_input,
            engines_design_mapping, PILOT_FUELS,
            gfi_var, ci_var, False,
        )

        drop_in_fuels = pd.concat(
            (drop_in_fuels_temp, drop_in_fuels), ignore_index=True
        ).drop_duplicates()
        
        to_merge = fuel_mix.rename(columns={
            "total_fuel_cost_mix": "total_fuel_cost",
            "ttw_mix": "ttw_gco2e_p_mj",
            "wtw_mix": "wtw_gco2e_p_mj",
        })[[
            "scenario_id", "year", "reg_reg_class", "engine_type", "fuel",
            "total_fuel_cost", "ttw_gco2e_p_mj", "wtw_gco2e_p_mj",
        ]]
        to_merge["reward_usd_tlsfoeq"] = 0 # assume that the fuel mix wont receive any rewards
        
        # Merge policy parameters
        to_merge = to_merge.merge(
            op_cost_batch[[
                            "ru_prices",
                            "gfi",
                            "Fuel reward bio",
                            "Fuel reward e-fuels",
                            "Fuel reward gas",
                            "Fuel price bio",
                            "Fuel price e-fuels",
                            "Fuel price gas",
                            "Carbon intensity bio",
                            "Carbon intensity e-fuels",
                            "Carbon intensity gas",
                            "SU",
                            "scenario_id",
                            "year",
                            "reg_reg_class",
                            "ru1_price",
                            "ru2_price",
                            "base_compliance_target",
                            "direct_compliance_target",
                            "znz",
                            "FuelEU",
                            "ets_price",
                            "FuelEU_penalty",
                            "reward_delay",
                            "wacc",
                            "e-fuels available after",
                            "stringency_nzf_flag",
                        ]].drop_duplicates(),
            on=["scenario_id", "year", "reg_reg_class"],
            how="left",
        )
        assert len(to_merge) == len(fuel_mix)

        # checks
        assert not np.any(
            to_merge["total_fuel_cost"].isna()
        ), "NaNs found in total_fuel_cost"
        assert not np.any(
                to_merge["stringency_nzf_flag"].isna()
            ), "NaNs found in stringency_nzf_flag"
        n_unique_combinations = (
            op_cost_batch[["scenario_id", "year", "reg_reg_class"]]
            .drop_duplicates()
            .shape[0]
        )
        n_valid_engines = engines_design_mapping.loc[
            (engines_design_mapping.fuel_design != "other")
            &(~(engines_design_mapping.fuel_design.str.contains("nuclear")))
            & (
                engines_design_mapping.engine_type.isin(
                    fuel_engine_mapping.engine_type.unique()
                )
            )
        ].shape[0]
        assert (
            fuel_mix.shape[0] == n_unique_combinations * n_valid_engines
        ), "fuel_mix length does not match expected combinations"
        
        op_cost_batch = pd.concat((op_cost_batch, to_merge), ignore_index=True)

    # Test that the function finds a fuel mix when asked
    test = op_cost_batch[
        (op_cost_batch.fuel == "base_compliance_target mix df_ice_ammonia")
        & (op_cost_batch.year == 2035)
        & (op_cost_batch.total_fuel_cost == np.inf)
        & (op_cost_batch.base_compliance_target < 91)
    ]
    assert len(test) == 0, test.head(3)
    
    op_cost_batch = reduce_size_df(op_cost_batch)
    if CLEAN:
        del to_merge, fuel_mix, op_template, drop_template, test
        gc.collect()
    
    # ============= SU price =============
    # Calculate and add the SU price
    # This is either the marginal abatement cost of bio-MGO or RU2
    op_cost_batch["su_price"] = 0
    ind = op_cost_batch[
        (op_cost_batch.SU == "RU2") &
        (op_cost_batch.year >= 2028) &
        (op_cost_batch.direct_compliance_target < 100)
    ].index
    op_cost_batch.loc[ind, "su_price"] = op_cost_batch.loc[ind, "ru2_price"]
    op_cost_batch["pooling_value"] = op_cost_batch["FuelEU_penalty"]
    
    # Abatement cost calculation
    op_cost_reshaped = (
        op_cost_batch[
            (op_cost_batch.fuel.isin(["bio-mgo", "lsfo"])) &
            (op_cost_batch.engine_type == "All") &
            (op_cost_batch.reg_reg_class == "noneu_to_noneu")
        ]
        .set_index(["scenario_id", "year", "fuel"])[[
            "main_fuel_price_usd_tvslfoeq", "wtw_gco2e_p_mj"
        ]]
        .unstack()
    )
    
    abatement_cost = (
        (op_cost_reshaped[("main_fuel_price_usd_tvslfoeq", "bio-mgo")] -
         op_cost_reshaped[("main_fuel_price_usd_tvslfoeq", "lsfo")]) / 41.20 * 1_000 /
        (op_cost_reshaped[("wtw_gco2e_p_mj", "lsfo")] -
         op_cost_reshaped[("wtw_gco2e_p_mj", "bio-mgo")])
    ).reset_index().rename(columns={0: "abatement_cost_usd_p_tCO2e"})

    check_len = len(op_cost_batch)
    op_cost_batch = op_cost_batch.merge(
        abatement_cost, on=["scenario_id", "year"], how="left"
    )
    assert check_len == len(op_cost_batch)

    # --- SU value: min of abatement cost and ru2_price (for years >= 2028) ---
    mask_su = (
        (op_cost_batch.SU == "abatement_cost") &
        (op_cost_batch.year >= 2028) &
        (op_cost_batch.direct_compliance_target < 100)
    )
    op_cost_batch.loc[mask_su, "su_price"] = np.minimum(
        op_cost_batch.loc[mask_su, "abatement_cost_usd_p_tCO2e"],
        op_cost_batch.loc[mask_su, "ru2_price"],
    )

    # --- Pooling value: min of abatement cost and FuelEU_penalty ---
    mask_pool = op_cost_batch.SU == "abatement_cost"
    op_cost_batch.loc[mask_pool, "pooling_value"] = np.minimum(
        op_cost_batch.loc[mask_pool, "abatement_cost_usd_p_tCO2e"],
        op_cost_batch.loc[mask_pool, "FuelEU_penalty"],
    )

    # in No IMO NZF scenarios, SU price = 0
    mask = op_cost_batch[op_cost_batch.gfi == "No IMO NZF"].index
    op_cost_batch.loc[mask, "su_price"] = 0

    # check that all inputs are not void
    cols_to_check = [
        "wtw_gco2e_p_mj",
        "main_fuel_price_usd_tvslfoeq",
        "reward_usd_tlsfoeq",
        "su_price",
        "ru2_price",
        "ru1_price",
        "direct_compliance_target",
        "base_compliance_target",
    ]
    mask = op_cost_batch[cols_to_check].isna().any(axis=1) & (
        ~op_cost_batch["fuel"].str.contains("mix")
    )
    assert not mask.any(), "Some required columns are NaN for non-mix fuels"
    
    op_cost_batch = reduce_size_df(op_cost_batch)
    if CLEAN:
        del abatement_cost, op_cost_reshaped, mask_su, mask_pool, mask
        gc.collect()
    
    # ============= Pilot fuels =============
    pattern = "|".join(fuels_in_need_of_pilot)
    template = op_cost_batch[
        op_cost_batch["fuel"].str.contains(pattern, case=False, regex=True) &
        ~op_cost_batch["fuel"].str.contains("mix", case=False)
    ]
    
    op_cost_pilot = template.merge(
        pd.DataFrame({"pilot_fuel": PILOT_FUELS}), how="cross"
    )
    op_cost_pilot = op_cost_pilot[
        (op_cost_pilot.pilot_fuel != "e-mgo") |
        (op_cost_pilot.year >= op_cost_pilot['e-fuels available after'])
    ].reset_index(drop=True)

    # Add pilot fuel properties
    to_merge = (
        op_cost_batch[op_cost_batch.fuel.isin(PILOT_FUELS)][[
            "year", "scenario_id", "fuel",
            "ttw_gco2e_p_mj", "wtt_gco2e_p_mj", "main_fuel_price_usd_tvslfoeq",
        ]]
        .rename(columns={
            "fuel": "pilot_fuel",
            "ttw_gco2e_p_mj": "ttw_gco2e_p_mj_pilot",
            "main_fuel_price_usd_tvslfoeq": "pilot_fuel_price_usd_tvslfoeq",
            "wtt_gco2e_p_mj": "wtt_gco2e_p_mj_pilot",
        })
        .drop_duplicates()
    )
    
    idx = ["year", "scenario_id", "pilot_fuel"]
    lk = to_merge.set_index(idx)
    key = pd.MultiIndex.from_frame(op_cost_pilot[idx])
    
    op_cost_pilot["ttw_gco2e_p_mj_pilot"] = key.map(lk["ttw_gco2e_p_mj_pilot"])
    op_cost_pilot["pilot_fuel_price_usd_tvslfoeq"] = key.map(lk["pilot_fuel_price_usd_tvslfoeq"])
    op_cost_pilot["wtt_gco2e_p_mj_pilot"] = key.map(lk["wtt_gco2e_p_mj_pilot"])
    op_cost_pilot["wtw_gco2e_p_mj_pilot"] = (
        op_cost_pilot["ttw_gco2e_p_mj_pilot"] + op_cost_pilot["wtt_gco2e_p_mj_pilot"]
    )
    
    # Calculate fuel consumption with pilot
    to_input = rename_columns(op_cost_pilot, COLUMN_RENAMES)
    to_input["fuel_consumption_mj_year"] = 1_000
    
    out_cons = calculate_fuel_consumption(
        to_input[["fuel_consumption_mj_year", "pilot_fuel"]],
        op_cost_pilot["fuel"],
        to_input,
        pilot_fuel_input,
    )
    for var in ["total_fuel_cost","share_pilot_mj"]:
        assert len(out_cons[var]) == len(op_cost_pilot)
        op_cost_pilot[var] = out_cons[var]
    
    # Correct emissions
    op_cost_pilot["wtw_gco2e_p_mj"] = (
        op_cost_pilot["wtw_gco2e_p_mj"] * (1 - op_cost_pilot["share_pilot_mj"]) +
        op_cost_pilot["wtw_gco2e_p_mj_pilot"] * op_cost_pilot["share_pilot_mj"]
    )
    assert op_cost_pilot["total_fuel_cost"].notna().all()
    assert op_cost_pilot["wtw_gco2e_p_mj"].notna().all()
    
    # Calculate compliance costs for pilot fuels
    ls_hfo_energy_density = float(
        energy_density.loc[energy_density.fuel == "lsfo", "gj_p_tfuel"]
    )
    total_consumption_tlsfo_eq = 1 / ls_hfo_energy_density # t LSFO / GJ
    op_cost_pilot['eu_pooling_price'] = op_cost_pilot['pooling_value']
    to_input = rename_columns(op_cost_pilot, COLUMN_RENAMES)
    
    out = calculate_policy_compliance(
        params=to_input[policy_compliance_params_cols],
        total_consumption_tlsfo_eq=total_consumption_tlsfo_eq,
        overall_emissions_factor_wtw_gco2eq_mj=to_input["wtw_gco2e_p_mj"],
        total_ttw_emissions_tco2eq=to_input["ttw_gco2e_p_mj"] / 1_000,
        attained_gfi_wtw_gco2eq_mj=to_input["wtw_gco2e_p_mj"],
        compliance_mode="fueleu_imo"
    )
    assert out["net_policy_cost"].isna().sum() == 0, "NaNs in net_policy_cost"
    assert (
        len(out["total_compliance_cost_usd"]) 
        == len(out["reward_usd"]) 
        == len(out["eu_compliance_cost_usd"]) 
        == len(out["net_policy_cost"]) 
        == len(op_cost_pilot)
    )
    op_cost_pilot["NZF compliance cost"] = out["total_compliance_cost_usd"]
    op_cost_pilot["fuel reward"] = -out["reward_usd"]
    op_cost_pilot["EU compliance cost"] = out["eu_compliance_cost_usd"]
    op_cost_pilot["net_policy_cost"] = out["net_policy_cost"]
    op_cost_pilot["usd_p_GJ"] = op_cost_pilot[["net_policy_cost", "total_fuel_cost"]].sum(axis=1)
    
    # only keep the cheapest option for pilot fuel available for each year and scenario
    # here, we assume that the pilot fuel cost is determined at the route level with the NZF cost broken by route
    # this is technically not correct, but it allows to remove a degree of complexity in the model
    op_cost_pilot.sort_values(
        ["scenario_id", "year", "fuel", "engine_type", "reg_reg_class", "usd_p_GJ"],
        ascending=True, na_position="last", inplace=True
    )
    op_cost_pilot.drop_duplicates(
        subset=["scenario_id", "year", "fuel", "engine_type", "reg_reg_class"],
        keep="first", inplace=True
    )
    op_cost_pilot.reset_index(drop=True, inplace=True)
    assert (
        len(op_cost_pilot) == op_cost_batch["fuel"].isin(fuels_in_need_of_pilot).sum()
    )
    
    if CLEAN:
        del to_input, out_cons, out, template, to_merge
        gc.collect()
    
    # ============= Non-pilot fuels =============
    op_cost_no_pilot = op_cost_batch[~op_cost_batch.fuel.isin(fuels_in_need_of_pilot)]
    op_cost_no_pilot['eu_pooling_price'] = op_cost_no_pilot['pooling_value']
    to_input = rename_columns(op_cost_no_pilot, COLUMN_RENAMES)[policy_compliance_params_cols]
    
    out = calculate_policy_compliance(
        params=to_input,
        total_consumption_tlsfo_eq=total_consumption_tlsfo_eq,
        overall_emissions_factor_wtw_gco2eq_mj=to_input["wtw_gco2e_p_mj"],
        total_ttw_emissions_tco2eq=to_input["ttw_gco2e_p_mj"] / 1_000,
        # CRM has no attained-GFI concept separate from the raw WTW intensity.
        attained_gfi_wtw_gco2eq_mj=to_input["wtw_gco2e_p_mj"],
        compliance_mode="fueleu_imo",
    )
    assert len(out["total_compliance_cost_usd"]) == len(out["reward_usd"]) == len(out["eu_compliance_cost_usd"]) == len(out["net_policy_cost"]) == len(op_cost_no_pilot)
    op_cost_no_pilot["NZF compliance cost"] = out["total_compliance_cost_usd"]
    op_cost_no_pilot["fuel reward"] = -out["reward_usd"]
    op_cost_no_pilot["EU compliance cost"] = out["eu_compliance_cost_usd"]
    op_cost_no_pilot["net_policy_cost"] = out["net_policy_cost"]
    op_cost_no_pilot["usd_p_GJ"] = op_cost_no_pilot[["net_policy_cost", "total_fuel_cost"]].sum(axis=1)

    mask_non_mix = ~op_cost_no_pilot["fuel"].str.contains("mix", case=False, na=False)
    assert (
        op_cost_no_pilot.loc[mask_non_mix, "total_fuel_cost"].isna().sum() == 0
    ), "NaNs in total_fuel_cost for non-mix fuels"
    
    if CLEAN:
        del to_input, out
        gc.collect()
    
    # Concat the df with and without pilot fuels
    operating_cost_batch = pd.concat(
        (op_cost_pilot, op_cost_no_pilot),
        ignore_index=True,
    )
    operating_cost_batch = reduce_size_df(operating_cost_batch)
    operating_cost_batch["year"] = operating_cost_batch["year"].astype("int32")
    
    if CLEAN:
        del op_cost_pilot, op_cost_no_pilot
        gc.collect()
    
    # Return operating cost batch and fully expanded drop_in_fuels (with mixes)
    # drop_in_fuels is scenario-independent and only needs to be computed once per batch
    return operating_cost_batch, drop_in_fuels, engines_design_mapping

def build_retrofit_lookup(retrofit_cost_df: pd.DataFrame) -> Dict[Tuple[Any, int, str, str], Tuple[float, float]]:
    """
    Build lookup:
      (segment_size_id, year, initial_fuel_design, retrofitted_fuel_design)
        -> (retrofit_cost, opportunity_cost)
    """
    retrofit_lookup = {
        (
            row["segment_size_id"],
            int(row["year"]),
            row["initial_fuel_design"],
            row["retrofitted_fuel_design"],
        ): (float(row["retrofit_cost"]), float(row["opportunity_cost"]))
        for _, row in retrofit_cost_df.iterrows()
    }
    return retrofit_lookup

def _get_defined_levels(scenario_variables: pd.DataFrame, var: str) -> List[Any]:
    """
    Get all defined levels for a variable from `scenario_variables`.
    
    For discrete variables, returns all unique levels (could be 1, 2, 3, or more).
    For continuous variables used in interpolation, the caller should verify
    there are exactly 2 levels (low and high).
    
    Returns:
        List of all non-NaN values for the variable, preserving order.
    """
    defined = [v for v in scenario_variables[var] if pd.notna(v)]
    assert len(defined) > 0, f"'{var}' has no defined level in scenario_variables"
    return defined


def generate_interior_lhs_scenarios(
    n_scenarios_N: int,
    *,
    scenario_variables: pd.DataFrame,
    scenario_mapping: pd.DataFrame,
    reference_tables: Dict[str, pd.DataFrame],
    cont_vars: List[str],
    discrete_vars: List[str],
    distribution: str = "uniform",
    seed: int = None,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Build `n_scenarios_N` interior scenarios via a Sliced Latin Hypercube Design (Qian, 2012).

    `discrete_vars` lists every genuinely categorical `scenario_variables` column to
    consider for interior scenario construction (typically `DISCRETE_VARS`). Each column is
    automatically classified from the data itself (not hardcoded):
      - a *slicing* variable, if it has more than one distinct defined level in
        `scenario_variables` -- the `n_scenarios_N` draws are partitioned into one independently-
        sampled batch per combination of slicing variables' levels (`itertools.product`
        across all of them, so e.g. 2 slicing variables with 2 levels each give up to
        4 slices, or 1 variable with 3 levels gives 3 slices);
      - a *fixed* variable, if it only has a single distinct defined level -- held
        constant at that level on every interior row.
    So a `discrete_vars` column only ever slices if `scenario_variables` genuinely
    defines more than one level for it; adding/removing levels for a column
    automatically changes whether it slices or stays fixed, with no separate list to
    keep in sync.

    Each slicing variable's levels are found independently via `_get_defined_levels`.
    `n_scenarios_N` is split as evenly as possible across the resulting slices
    (`np.array_split`), and each slice draws an independent len(cont_vars)-dimensional
    LHS point (`scipy.stats.qmc.LatinHypercube`), blending each `cont_vars` variable's
    low/high named trajectory pointwise with that draw's weight lambda in [0, 1].

    `distribution` controls the shape of the blend weight lambda within [0, 1] and
    applies identically to every `cont_vars` variable (their marginal uncertainty
    shapes are otherwise treated as independent of one another):
      - "uniform" (default): every point between a variable's low and high level is
        equally likely -- this is the LHS unit-cube sample used as-is.
      - "triangular": a symmetric triangular density peaked at the midpoint (0.5)
        between low and high, concentrating draws toward the center instead of
        spreading them flat. Implemented as a monotonic ppf transform of the raw LHS
        sample, which preserves its per-dimension stratification.
      - "corners": Full factorial design at extreme values only (lambda = 0 or 1 for
        each continuous variable). Returns exactly (# discrete combinations) × 2^(# cont_vars)
        scenarios, ignoring the n_scenarios_N parameter. Useful for validation against
        previous corner-based designs.

    For the trajectory-style `cont_vars` (looked up via `scenario_mapping`'s "file"
    column against `reference_tables`), the blended year-indexed trajectory is
    materialized as new rows in that reference table under a unique synthetic
    scenario label, vectorized across all draws in a slice (no per-draw pd.concat).
    For any `cont_vars` entry absent from `scenario_mapping` (i.e. "e-fuels available
    after" -- a scalar switch-point, not a trajectory), the interpolated value itself
    is stored directly in the returned scenario rows.

    Returns (interior_scenarios, updated_reference_tables): `interior_scenarios` has one
    row per draw (n_scenarios_N rows total, `scenario_id` not yet assigned -- the caller assigns it
    after concatenating with the corner scenarios); `updated_reference_tables` is a copy
    of `reference_tables` with the newly-materialized synthetic rows appended.
    """
    assert distribution in ("uniform", "triangular", "corners"), (
        f"Unknown distribution '{distribution}'; expected 'uniform', 'triangular', or 'corners'"
    )
    assert len(discrete_vars) > 0, "discrete_vars must contain at least 1 variable"
    missing_discrete_vars = [
        v for v in discrete_vars if v not in scenario_variables.columns
    ]
    assert len(missing_discrete_vars) == 0, (
        f"discrete_vars not found in scenario_variables columns: {missing_discrete_vars}. "
        f"Available columns: {scenario_variables.columns.tolist()}"
    )

    # classify each discrete_vars column, from the data, as slicing (>1 distinct
    # defined level) or fixed (exactly 1 distinct defined level)
    levels_by_var = {
        var: _get_defined_levels(scenario_variables, var)
        for var in discrete_vars
    }
    slice_vars = [var for var in discrete_vars if len(levels_by_var[var]) > 1]
    fixed_vars = [var for var in discrete_vars if len(levels_by_var[var]) == 1]
    print(f"slice_vars: {slice_vars}\n, fixed_vars:{fixed_vars}\n, cont_vars: {cont_vars}\n")

    # every possible combination of levels across all slicing variables -> one slice each
    slice_combinations = list(
        itertools.product(*(levels_by_var[var] for var in slice_vars))
    )
    assert len(slice_combinations) > 0, (
        f"No slice combinations found for slice_vars={slice_vars}, "
        f"levels_by_var={levels_by_var}"
    )

    reference_tables = {name: df.copy() for name, df in reference_tables.items()}
    mapping_by_var = scenario_mapping.set_index("variable")["file"].to_dict()

    slice_row_indices = np.array_split(np.arange(n_scenarios_N), len(slice_combinations))
    slice_frames = []
    for combo, idx in zip(slice_combinations, slice_row_indices):
        n_slice = len(idx)
        if n_slice == 0:
            continue
        # lam holds the blend weights that determine where between each cont_vars variable's low and high level a particular scenario sits. The values are in [0, 1]
        if distribution == "corners":
            cont_corners = list(itertools.product([0.0, 1.0], repeat=len(cont_vars)))
            n_slice = len(cont_corners)  # Override n_slice
            # Create new index array matching the corner scenario count
            idx = np.arange(len(slice_frames) * n_slice, (len(slice_frames) + 1) * n_slice)
            lam = np.array(cont_corners)
        else:
            random_generator = np.random.default_rng(seed)
            lam = qmc.LatinHypercube(d=len(cont_vars), seed=random_generator).random(n=n_slice)
            if distribution == "triangular":
                # symmetric triangular density on [0, 1], peak at the midpoint (c=0.5).
                # This monotonic ppf transform preserves the LHS's per-dimension
                # stratification while concentrating draws toward the center of each
                # cont_vars variable's low/high range instead of spreading them flat.
                lam = triang.ppf(lam, c=0.5)

        slice_rows = pd.DataFrame(index=idx)
        for var, level in zip(slice_vars, combo):
            slice_rows[var] = level
        for var in fixed_vars:
            levels = _get_defined_levels(scenario_variables, var)
            slice_rows[var] = levels[0]  # fixed vars have exactly 1 level

        for j, var in enumerate(cont_vars):
            levels = _get_defined_levels(scenario_variables, var)
            assert len(levels) == 2, (
                f"Continuous variable '{var}' must have exactly 2 defined levels "
                f"for interpolation, but has {len(levels)}: {levels}"
            )
            low = levels[0]
            high = levels[-1]  # if only 1 level, low == high
            lam_j = lam[:, j]

            if var not in mapping_by_var:
                # scalar variable (e.g. "e-fuel prices"): interpolate directly
                low_v, high_v = float(low), float(high)
                slice_rows[var] = np.round(low_v + lam_j * (high_v - low_v))
                continue

            if low == high:
                slice_rows[var] = low
                continue

            file_name = mapping_by_var[var]
            df_ref = reference_tables[file_name]
            low_df = (
                df_ref[df_ref["scenario"] == low]
                .drop(columns="scenario")
                .set_index("year")
                .sort_index()
            )
            high_df = (
                df_ref[df_ref["scenario"] == high]
                .drop(columns="scenario")
                .set_index("year")
                .sort_index()
            )
            assert list(low_df.index) == list(high_df.index), (
                f"'{var}': low ('{low}') and high ('{high}') trajectories in "
                f"'{file_name}' don't share the same years"
            )

            # blend, vectorized across (draws) x (years) x (value columns)
            low_arr = low_df.to_numpy()
            delta_arr = high_df.to_numpy() - low_arr
            blended = low_arr[None, :, :] + lam_j[:, None, None] * delta_arr[None, :, :]

            labels = [f"lhs_{i:04d}_{var}" for i in idx]
            blended_df = pd.DataFrame(
                blended.reshape(-1, blended.shape[-1]), columns=low_df.columns
            )
            blended_df["year"] = np.tile(low_df.index.to_numpy(), n_slice)
            blended_df["scenario"] = np.repeat(labels, len(low_df.index))
            reference_tables[file_name] = pd.concat(
                [reference_tables[file_name], blended_df], ignore_index=True
            )
            slice_rows[var] = labels

        slice_frames.append(slice_rows)

    interior_scenarios = pd.concat(slice_frames, ignore_index=True)

    if distribution == "corners":
        expected = len(slice_combinations) * (2 ** len(cont_vars))
    else:
        expected = n_scenarios_N
    assert len(interior_scenarios) == expected, (
        f"Expected {expected} interior scenarios "
        f"({'corners' if distribution == 'corners' else 'LHS'}: "
        f"{len(slice_combinations)} discrete slices × "
        f"{2**len(cont_vars) if distribution == 'corners' else 'split'}), "
        f"got {len(interior_scenarios)}"
    )
    return interior_scenarios, reference_tables


def create_op_cube(
    years_all,
    designs,
    engine_types,
    scenario_ids,
    REG_REG_CLASSES,
    op,
    dropin,
    engines_design_mapping,
    DEBUG,
):
    """
    Returns a 5D matrix op_cube_df containing the operating cost:
    (year, fuel_design, engine_type, reg_reg_class, scenario_id)

    Operating costs include fuel + carbon/compliance cost.
    It finds the minimum cost among the drop-in fuels allowed for each design/engine type.

    Intermediate dataframes returned:
    - op_pivot_df: pivoted operating cost per scenario, year, and fuel, with engine_type in index.
    - op_pivot_fc: same structure but with fuel cost only.
    """

    n_years_all = len(years_all)
    n_scenarios = len(scenario_ids)
    n_designs = len(designs)
    n_engines = len(engine_types)
    n_reg_reg = len(REG_REG_CLASSES)

    # -------------------------------------------------------------------
    # Allowed fuels per fuel_design
    # -------------------------------------------------------------------
    allowed_fuels_design = {
        d: set(
            dropin.loc[
                (dropin["fuel_design"] == d) & (dropin["drop_in"]), "fuel"
            ].unique()
        )
        for d in designs
    }

    # -------------------------------------------------------------------
    # Allowed fuel_design per engine_type (design/engine compatibility)
    # -------------------------------------------------------------------
    allowed_design_engine = {
        e: set(
            engines_design_mapping.loc[
                engines_design_mapping["engine_type"] == e, "fuel_design"
            ].unique()
        )
        for e in engine_types
    }

    # -------------------------------------------------------------------
    # Combined allowed fuels for each (fuel_design, engine_type)
    # -------------------------------------------------------------------
    allowed_fuels = {}

    for d in designs:
        for e in engine_types:
            # check if engine_type supports this fuel_design
            if d in allowed_design_engine.get(e, set()):
                allowed_fuels[(d, e)] = allowed_fuels_design.get(d, set())
            else:
                # engine does not support this design → no fuels allowed
                allowed_fuels[(d, e)] = set()

    # Pivot operating cost including engine_type
    op_pivot_df = op.pivot_table(
        index=["scenario_id", "year", "engine_type", "reg_reg_class"],
        columns="fuel",
        values="usd_p_GJ",
    ).sort_index()
    op_pivot_fc = op.pivot_table(
        index=["scenario_id", "year", "engine_type", "reg_reg_class"],
        columns="fuel",
        values="total_fuel_cost",
    ).sort_index()

    # Initialize op_cube (year, design, engine_type, reg_reg_class, scenario)
    op_cube_df = np.full(
        (n_years_all, n_designs, n_engines, n_reg_reg, n_scenarios),
        np.inf,
        dtype=np.float32,
    )

    # Fill cube
    for y_i, year in enumerate(years_all):
        # Select price data for this year
        price_df = op_pivot_df.loc[
            (slice(None), year, slice(None), slice(None)), :
        ].reset_index(level="year", drop=True)

        for d_i, design in enumerate(designs):
            for e_i, engine in enumerate(engine_types):
                fuels_allowed = allowed_fuels.get((design, engine), set())
                if DEBUG and (y_i == 10):
                    print(f"\n \n design, engine: {design, engine}")
                    print(f"fuels_allowed: {fuels_allowed}")
                if not fuels_allowed:
                    continue
                for r_i, reg_reg_class in enumerate(REG_REG_CLASSES):

                    # Extract cost matrix (scenarios × fuels)
                    try:
                        prices_allowed = price_df.loc[
                            (slice(None), engine, reg_reg_class), list(fuels_allowed)
                        ].values.astype(float)
                        prices_allowed = prices_allowed.astype(np.float32)
                    except KeyError:
                        if DEBUG and (y_i == 10):
                            print(f"\n No allowed prices found for {reg_reg_class}")
                        continue  # skip if some fuels missing

                    # Min cost across allowed fuels for each scenario
                    if DEBUG and (y_i == 10):
                        print(f"\n reg_reg_class: {reg_reg_class}")
                        print(f"prices_allowed: {prices_allowed}")
                    op_cube_df[y_i, d_i, e_i, r_i, :] = np.nanmin(
                        prices_allowed, axis=1
                    )

    return op_cube_df, op_pivot_df, op_pivot_fc


def prepare_inputs(
    operating_cost: pd.DataFrame,
    rep_ships: pd.DataFrame,
    retrofit_cost: pd.DataFrame,
    drop_in_fuels: pd.DataFrame,
    current_year: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Defensive copies and standard type casts.
    """
    operating_cost_df = operating_cost[operating_cost.year>=current_year].copy()
    ships = rep_ships.copy()
    retrofit_cost_df = retrofit_cost.copy()
    drop_in_fuels_df = drop_in_fuels.copy()

    operating_cost_df["year"] = operating_cost_df["year"].astype(int)
    retrofit_cost_df["year"] = retrofit_cost_df["year"].astype(int)
    ships["generation"] = ships["generation"].astype(int)
    ships["scrap_year"] = ships["scrap_year"].astype(int)

    return operating_cost_df, ships, retrofit_cost_df, drop_in_fuels_df


def build_op_cubes(
    modelling_years: List[int],
    fuel_designs: List[str],
    engine_types: np.ndarray,
    scenario_ids: np.ndarray,
    REG_REG_CLASSES: np.ndarray,
    operating_cost_df: pd.DataFrame,
    drop_in_fuels_df: pd.DataFrame,
    engines_design_mapping: pd.DataFrame,
    logger,
    DEBUG,
):
    """
    Wrapper around create_op_cube to build the global operating-cost cube and pivots.
    """
    op_cube_global, op_pivot, op_pivot_baseline = create_op_cube(
        modelling_years,
        fuel_designs,
        engine_types,
        scenario_ids,
        REG_REG_CLASSES,
        operating_cost_df,
        drop_in_fuels_df,
        engines_design_mapping,
        False,
    )

    if logger is not None:
        logger.debug(
            f"\n op_cube_global year 10, scenario 0, engine All and reg_reg 0: "
            f"{op_cube_global[10, :, 0, 0, 0]}"
        )

    return op_cube_global, op_pivot, op_pivot_baseline


def build_ship_level_op_cube(
    ship_instance: pd.Series,
    ships: pd.DataFrame,
    op_cube_global: np.ndarray,
    modelling_years: List[int],
    year_to_index: Dict[int, int],
    fuel_designs: List[str],
    avail_fuel_designs: [str],
    engine_types: np.ndarray,
    REG_REG_CLASSES: np.ndarray,
    design_to_global: Dict[str, int],
    n_fuel_designs: int,
    n_scenarios: int,
    n_reg_reg: int,
    logger: logging.Logger | None = None,
    CHECK: bool = False,
    DEBUG: bool = False,
    df_engine_efficiency_ref: pd.DataFrame = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    List[int],
    int,
    float,
    float,
    float,
    Any,
    pd.DataFrame,
    int,
]:
    """
    For a representative ship (segment + size + generation), build its ship-level
    operating-cost cube:
       op_cube[year_idx, design_idx, scenario_idx]
    and return relevant metadata.
    """
    segment_size_id = ship_instance.segment_size_id
    # ship_instance is representative of the designs available
    ships_subset = ships[ships.segment_size_id == segment_size_id]
    generation_year = int(ship_instance["generation"])
    scrap_year = int(ship_instance["scrap_year"])
    annual_consumption = float(ship_instance["cons_pa"])
    tc_rate = float(ship_instance["tc_usd_per_day"])
    transport_work_thousand_tnm_pa = float(ship_instance.get("transport_work_tnm_pa", 0.0)) / 1_000

    year_indices = [year_to_index[y] for y in modelling_years if generation_year <= y <= scrap_year]
    years_arr = np.array(modelling_years)[year_indices]
    n_years = len(year_indices)
    if n_years == 0:
        raise ValueError(
            f"No operating years for segment {segment_size_id} between "
            f"generation {generation_year} and scrap_year {scrap_year}"
            f"year_to_index: {year_to_index}"
        )

    # Engine type
        # find the indices of the engine type to  retrieve the relevant information of op_cube_global
        # if the fuel_design is in conventional, methanol DF or ammonia DF, then the engine type = All
        # if the fuel_design is df_ice_lng, then it is either the actual engine type; or the default engine type for this ship
    lng_row = ships_subset[ships_subset.initial_fuel_design.str.contains("df_ice_lng", na = False)]
    if len(ships_subset) > 1 and len(lng_row) > 0 :
        # If the ship represents multiple designs, ensure LNG engine consistency across retrofits
        if len(lng_row.engine_type.unique()) > 1:
            raise ValueError(f"Competing LNG engine types for {segment_size_id}: {lng_row.engine_type.unique()} ")

        lng_engine_type = lng_row.iloc[0].engine_type

    elif "df_ice_lng" in ship_instance["initial_fuel_design"]:
        # if we are looking at an individual IMO number and the ship is df_ice_lng, we use the engine installed
        lng_engine_type = ship_instance.engine_type
    elif ship_instance.n_stroke == "2stroke":
        # if we are looking at an individual IMO number the ship is not df_ice_lng, we assume that it retrofits to Diesel Slow speed if 2 stroke, Otto - Medium Speed if 4 stroke
        lng_engine_type = "LNG Diesel 2024+"
    elif ship_instance.n_stroke == "4stroke":
        lng_engine_type = "LNG Otto MSD 2024+"
    else:
        raise ValueError(
            f"Could not determine LNG engine type for representative ship {segment_size_id}"
        )

    if lng_engine_type not in engine_types:
        raise ValueError(f"lng_engine_type '{lng_engine_type}' not found in engine_types")

    lng_engine_index = int(np.where(engine_types == lng_engine_type)[0][0])

    # Build op_cube with engine dimension collapsed to per-design choice
    # Allocate cube only for available designs
    ship_op_cube = np.full((n_years, n_fuel_designs, n_scenarios), np.inf, dtype=np.float32)

    initial_base_design = str(ship_instance["initial_fuel_design"]).split('_+_')[0]
    n_stroke = ship_instance["n_stroke"]

    for design_name in avail_fuel_designs:
        global_i = design_to_global[design_name]
        ship_op_cube[:, global_i, :] = 0.0
        if DEBUG:
            logger.debug(f"\n Building ship_op_cube: global_i: {global_i}, design_name: {design_name}")

        eng_idx = lng_engine_index if design_name.startswith("df_ice_lng") else 0
        for reg_reg_class_index in range(n_reg_reg):
            ship_op_cube[:, global_i, :] += (
                op_cube_global[year_indices, global_i, eng_idx, reg_reg_class_index, :].astype(np.float32)
                * ship_instance[REG_REG_CLASSES[reg_reg_class_index]]
            )

        # Correct consumption for the BTE of this retrofit target design relative to the
        # ship's own initial engine (ratio = 1 for the ship's own initial_fuel_design).
        design_consumption = annual_consumption
        base_design = design_name.split('_+_')[0]
        if df_engine_efficiency_ref is not None and base_design != initial_base_design:
            bte_row = df_engine_efficiency_ref[
                (df_engine_efficiency_ref['fuel_design'] == base_design)
                & (df_engine_efficiency_ref['n_stroke'] == n_stroke)
            ]
            bte_initial_row = df_engine_efficiency_ref[
                (df_engine_efficiency_ref['fuel_design'] == initial_base_design)
                & (df_engine_efficiency_ref['n_stroke'] == n_stroke)
            ]
            assert len(bte_row) == 1, f"Expected 1 BTE row for {base_design}/{n_stroke}, got {len(bte_row)}"
            assert len(bte_initial_row) == 1, f"Expected 1 BTE row for {initial_base_design}/{n_stroke}, got {len(bte_initial_row)}"
            design_consumption = apply_engine_efficiency_correction(
                annual_consumption, bte_initial_row['bte'].iloc[0], bte_row['bte'].iloc[0]
            )
        ship_op_cube[:, global_i, :] *= design_consumption

    # ensure there is at least one feasible design across scenarios
    if not np.isfinite(ship_op_cube).any():
        raise RuntimeError(f"No finite operating costs for ship segment {segment_size_id}")

    if CHECK and ship_op_cube.shape != (n_years, n_fuel_designs, n_scenarios):
        raise ValueError(
            f"ship_op_cube shape {ship_op_cube.shape} unexpected; expected {(n_years, n_fuel_designs, n_scenarios)}"
        )

    if DEBUG:
        logger.debug(f"\n LNG engine_type for segment {segment_size_id}: {lng_engine_type}")
        logger.debug(f"\n ship_op_cube sample for segment {segment_size_id}:\n{ship_op_cube[10:12, :, :2]}")

    return (
        ship_op_cube,
        years_arr,
        year_indices,
        n_years,
        annual_consumption,
        tc_rate,
        transport_work_thousand_tnm_pa,
        segment_size_id,
        ships_subset,
        generation_year,
        lng_engine_type,
    )


def preallocate_history(
    ships: pd.DataFrame,
    modelling_years: List[int],
    scenario_ids: np.ndarray,
    year_to_index: Dict[int, int],
) -> Tuple[Dict[str, np.ndarray], int, List[str], pd.DataFrame]:
    """
    Pre-count required rows and preallocate history_storage arrays.
    Returns:
      - history_storage dict of arrays
      - initial write index (0)
      - cols_segment_size used to identify representative ships
      - rep_ships_df (unique by segment/size/generation)
    """
    n_scenarios = len(scenario_ids)

    rep_ships_df = ships.drop_duplicates(subset=COLS_SEGMENT_SIZE_ID)

    total_history_rows = 0
    for _, rep_ship_counting in rep_ships_df.iterrows():
        segment_size_id_count = rep_ship_counting.segment_size_id
        ships_subset_len_count = len(ships[ships.segment_size_id == segment_size_id_count])
        gen_count = int(rep_ship_counting["generation"])
        scrap_count = int(rep_ship_counting["scrap_year"])
        year_indices_count = [
            year_to_index[y] for y in modelling_years if gen_count <= y <= scrap_count
        ]
        n_years_count = len(year_indices_count)
        total_history_rows += ships_subset_len_count * n_scenarios * max(0, n_years_count)

    total_history_rows += 100  # small buffer

    history_storage = {
        "ship_id": np.empty(total_history_rows, dtype=object),
        "segment_size_id": np.empty(total_history_rows, dtype=object),
        "initial_fuel_design": np.empty(total_history_rows, dtype=object),
        "scenario_id": np.empty(total_history_rows, dtype=object),
        "year": np.empty(total_history_rows, dtype=np.int32),
        "fuel_start_idx": np.empty(total_history_rows, dtype=np.int32),
        "fuel_start": np.empty(total_history_rows, dtype=object),
        "decision": np.empty(total_history_rows, dtype=object),
        "fuel_end_idx": np.empty(total_history_rows, dtype=np.int32),
        "fuel_end": np.empty(total_history_rows, dtype=object),
        "engine_type_end": np.empty(total_history_rows, dtype=object),
        "op_cost": np.full(total_history_rows, np.nan, dtype=np.float32),
        "op_cost_discounted": np.full(total_history_rows, np.nan, dtype=np.float32),
        "retrofitting_cost": np.full(total_history_rows, np.nan, dtype=np.float32),
        "retrofitting_cost_discounted": np.full(total_history_rows, np.nan, dtype=np.float32),
        "opportunity_cost": np.full(total_history_rows, np.nan, dtype=np.float32),
        "opportunity_cost_discounted": np.full(total_history_rows, np.nan, dtype=np.float32),
        "# duplicates": np.full(total_history_rows, np.nan, dtype=np.float32),
    }

    return history_storage, 0, rep_ships_df


def run_dynamic_programming(
    ship_op_cube: np.ndarray,
    avail_fuel_designs: List[str],
    design_to_global: Dict[str, int],
    retrofit_lookup: Dict[Tuple[Any, int, str, str], Tuple[float, float]],
    segment_size_id: Any,
    years_arr: np.ndarray,
    generation_year: int,
    min_age_for_retrofit: int,
    retrofit_delay: int,
    UNCERTAINTY_CLEARING_YEARS: List[int],
    weights: List[float],
    discount_rate: float,
    DEBUG: bool = False,
    logger: logging.Logger | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Backward dynamic programming on a single ship's ship_op_cube.

    At each `year in UNCERTAINTY_CLEARING_YEARS` (policy/market uncertainty review point),
    `min_cost` is pooled across all scenarios (weighted by `weights`, i.e. the
    `# duplicates` scenario weights) independently per fuel design, then broadcast
    back across the scenario axis -- this reflects uncertainty resolving only
    gradually at discrete review points rather than all at once. This pooling is
    independent of retrofit eligibility (`min_age_for_retrofit`): a review year can
    occur before a ship is old enough to retrofit.

    Returns:
      - future_costs: (n_fuel_designs, n_scenarios) PV cost for each design at first year
      - best_action: (n_years, n_fuel_designs, n_scenarios) with 'continue' or 'retrofit_to_X'
    """
    # set up results files
    n_years, n_fuel_designs, n_scenarios = ship_op_cube.shape
    future_costs = np.zeros((n_fuel_designs, n_scenarios), dtype=np.float32)
    best_action = np.full(
        (n_years, n_fuel_designs, n_scenarios), "continue", dtype=object
    )

    # Precompute discount matrix for opportunity cost PV
    DISCOUNT_matrix = np.zeros((n_years, n_years), dtype=np.float32)
    for t in range(n_years):
        DISCOUNT_matrix[t, t:] = 1.0 / (
            (1.0 + discount_rate) ** np.arange(0, n_years - t)
        )
    if DEBUG:
        logger.debug(f"DEBUG in dynamic programming for segment_size_id {segment_size_id}")
        logger.debug(f"Discount matrix for dynamic programming {DISCOUNT_matrix}")
        logger.debug(f"n_years {n_years}")

    # calculate cost of all design_fuels available, starting from the last year
    for y_i in reversed(range(n_years)):
        year = int(years_arr[y_i])

        # operating costs if no retrofits
        current_op_cost = ship_op_cube[y_i, :, :]  # (n_fuel_designs, n_scenarios)
        continue_cost = (current_op_cost + future_costs / (1.0 + discount_rate)).astype(np.float32)
        min_cost = continue_cost.copy()
        action = np.full_like(min_cost, "continue", dtype=object)

        if DEBUG and logger is not None:
            logger.debug(f"Initial cost before any retrofitting in {year}")
            logger.debug(f"\n year {year}")
            logger.debug(f"\n current_op_cost {current_op_cost}, future_costs {future_costs}")

        # run through possible retrofits to check if retrofitting is not better
        # Only allow retrofits after min_age and once the uncertainty resolves, i.e. at a review year
        if (year - generation_year >= min_age_for_retrofit) and (year - retrofit_delay in UNCERTAINTY_CLEARING_YEARS):
            # loop through all target designs (and all initial designs at year y_i
            for target_design in avail_fuel_designs:
                global_td_i = design_to_global[target_design]
                for initial_design in avail_fuel_designs:
                    global_d_i = design_to_global[initial_design]

                    # get the retrofit cost
                    key = (segment_size_id, year, initial_design, target_design)
                    if key not in retrofit_lookup:
                        continue

                    retrofit_cost, opportunity_cost = retrofit_lookup[key]
                    total_opportunity_pv = opportunity_cost * DISCOUNT_matrix[y_i, y_i:].sum()

                    # operating costs if retrofitting
                    op_after = ship_op_cube[y_i, global_td_i, :]
                    exercise_total = (
                        retrofit_cost
                        + total_opportunity_pv
                        + op_after
                        + future_costs[global_td_i, :] / (1.0 + discount_rate)
                    ).astype(np.float32)

                    # only keep the retrofit option for the scenarios where it is cheaper to do so
                    mask = exercise_total < min_cost[global_d_i, :]
                    min_cost[global_d_i, mask] = exercise_total[mask]
                    action[global_d_i, mask] = f"retrofit_to_{target_design}"

                    if DEBUG and logger is not None and year == 2035 and mask.any():
                        logger.debug(
                            f"ini_design: {initial_design}, target design : {target_design}, "
                            f"global_td_i: {global_td_i}"
                        )
                        logger.debug(
                            f"\n retrofit_cost:{retrofit_cost}, opportunity_cost:{opportunity_cost}, total_opportunity_pv:{total_opportunity_pv}"
                            f"\n retrofit_cost + total_opportunity_pv: "
                            f"{retrofit_cost + total_opportunity_pv}"
                        )
                        logger.debug(
                            f"\n op_after + discounted future: "
                            f"{op_after + future_costs[global_td_i, :] / (1.0 + discount_rate)}"
                        )
                        logger.debug(
                            f"\n Investment decision = retrofitting from {initial_design} "
                            f"to {target_design} for certain scenarios"
                        )

        # Pooling step at policy/market uncertainty review years: gated on
        # `year in UNCERTAINTY_CLEARING_YEARS`, not on the retrofit-eligibility check above,
        # but also guarded on `y_i > 0` -- the DP loop runs in reverse, so y_i == 0 is
        # the LAST iteration processed and its min_cost becomes the function's
        # returned `future_costs`. Pooling there would collapse all scenario
        # dispersion in the returned `future_costs` (and therefore in
        # `pv_additional_cost`/`mu`) down to a single repeated value, regardless of
        # calendar alignment between `years_arr[0]` and `UNCERTAINTY_CLEARING_YEARS`. A
        # ship's own first modelled year is always "known" information for that ship,
        # so it must never be pooled away, even if it happens to fall on a review year.
        # min_cost has shape (n_fuel_designs, n_scenarios). We average across the
        # scenario axis (axis=1) SEPARATELY for each fuel design row (weights =
        # `# duplicates`, same convention as aggregate_results) -- i.e. the pooled
        # value for design D only uses design D's own scenario costs, never mixed
        # with other designs -- then broadcast that per-design average back across
        # the scenario axis. Outside review years (or at y_i == 0), min_cost is left
        # untouched.
        if y_i > 0 and year in UNCERTAINTY_CLEARING_YEARS:
            pooled = np.average(min_cost, axis=1, weights=weights).astype(np.float32)
            min_cost = np.repeat(pooled[:, np.newaxis], n_scenarios, axis=1)

        future_costs = min_cost
        best_action[y_i, :, :] = action

    return future_costs, best_action


def reconstruct_path_history(
    ship_id: Any,
    ship_instance: pd.Series,
    segment_size_id: Any,
    years_arr: np.ndarray,
    fuel_designs: List[str],
    design_to_global: Dict[str, int],
    best_action: np.ndarray,
    ship_op_cube: np.ndarray,
    retrofit_lookup: Dict[Tuple[Any, int, str, str], Tuple[float, float]],
    scenario_ids: np.ndarray,
    discount_rate: float,
    current_year: int,
    weights: List[float],
    initial_design_index: int,
    lng_engine_type: str,
    history_storage: Dict[str, np.ndarray],
    history_pointer: int,
    DEBUG: bool,
    logger: logging.Logger | None = None,
) -> int:
    """
    For a given ship, reconstruct cheapest path over time and scenarios, writing
    into history_storage and returning the updated write index history_pointer.
    """
    n_years = len(years_arr)
    _, _, n_scenarios = ship_op_cube.shape

    curr_fuel_idx = initial_design_index * np.ones(n_scenarios, dtype=int)
    retrofit_years_local = {s_i: None for s_i in range(n_scenarios)}
    retrofit_costs_local = {s_i: 0.0 for s_i in range(n_scenarios)}
    opportunity_costs_local = {s_i: 0.0 for s_i in range(n_scenarios)}

    for y_i, year in enumerate(years_arr):
        actions_this_year = best_action[y_i, curr_fuel_idx, np.arange(n_scenarios)]
        next_fuel_idx = np.zeros_like(curr_fuel_idx)
        op_costs = np.zeros_like(curr_fuel_idx, dtype=float)
        decisions = []

        for s_i in range(n_scenarios):
            act = actions_this_year[s_i]
            if isinstance(act, str) and act.startswith("retrofit_to_"):
                target_design = act.replace("retrofit_to_", "")
                next_idx = design_to_global[target_design]
                key = (
                    segment_size_id,
                    int(year),
                    fuel_designs[int(curr_fuel_idx[s_i])],
                    target_design,
                )
                if key in retrofit_lookup:
                    retrofit_cost, opportunity_cost = retrofit_lookup[key]
                    retrofit_years_local[s_i] = int(year)
                    retrofit_costs_local[s_i] = retrofit_cost
                    opportunity_costs_local[s_i] += opportunity_cost
            else:
                next_idx = curr_fuel_idx[s_i]

            next_fuel_idx[s_i] = next_idx
            op_costs[s_i] = ship_op_cube[y_i, next_idx, s_i]
            decisions.append(act)

        for s_i in range(n_scenarios):
            retrofit_cost_value = 0.0
            opportunity_cost_value = 0.0
            if retrofit_years_local[s_i] == year:
                retrofit_cost_value = retrofit_costs_local[s_i]
            if (
                retrofit_years_local[s_i] is not None
                and year >= retrofit_years_local[s_i]
            ):
                opportunity_cost_value = opportunity_costs_local[s_i]

            discount_factor = 1.0 / ((1.0 + discount_rate) ** (int(year) - current_year))

            design_name = fuel_designs[
                int(next_fuel_idx[s_i])
            ]
            engine = lng_engine_type if design_name.startswith("df_ice_lng") else "All"
            history_storage["ship_id"][history_pointer] = ship_id
            history_storage["segment_size_id"][history_pointer] = ship_instance["segment_size_id"]
            history_storage["initial_fuel_design"][history_pointer] = ship_instance["initial_fuel_design"]
            history_storage["scenario_id"][history_pointer] = scenario_ids[s_i]
            history_storage["year"][history_pointer] = int(year)
            history_storage["fuel_start_idx"][history_pointer] = int(curr_fuel_idx[s_i])
            history_storage["fuel_start"][history_pointer] = fuel_designs[int(curr_fuel_idx[s_i])]
            history_storage["decision"][history_pointer] = decisions[s_i]
            history_storage["fuel_end_idx"][history_pointer] = int(next_fuel_idx[s_i])
            history_storage["fuel_end"][history_pointer] = fuel_designs[int(next_fuel_idx[s_i])]
            history_storage["engine_type_end"][history_pointer] = engine
            history_storage["op_cost"][history_pointer] = float(op_costs[s_i])
            history_storage["op_cost_discounted"][history_pointer] = float(
                op_costs[s_i] * discount_factor
            )
            history_storage["retrofitting_cost"][history_pointer] = float(retrofit_cost_value)
            history_storage["retrofitting_cost_discounted"][history_pointer] = float(
                retrofit_cost_value * discount_factor
            )
            history_storage["opportunity_cost"][history_pointer] = float(opportunity_cost_value)
            history_storage["opportunity_cost_discounted"][history_pointer] = float(
                opportunity_cost_value * discount_factor
            )
            history_storage["# duplicates"][history_pointer] = float(weights[s_i])
            history_pointer += 1

        curr_fuel_idx[:] = next_fuel_idx

    return history_pointer


def simulate_cvar_paths(
    n_paths: int,
    ship_op_cube: np.ndarray,
    baseline_costs: np.ndarray,
    years_arr: np.ndarray,
    fuel_designs: List[str],
    design_to_global: Dict[str, int],
    best_action: np.ndarray,
    retrofit_lookup: Dict[Tuple[Any, int, str, str], Tuple[float, float]],
    segment_size_id: Any,
    UNCERTAINTY_CLEARING_YEARS: List[int],
    weights: List[float],
    discount_rate: float,
    current_year: int,
    initial_design_index: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Monte Carlo simulation of `n_paths` realized (design - baseline) discounted total
    cost outcomes, under the "i.i.d. regime redraw at every review point" reading of
    the pooled dynamic program in `run_dynamic_programming` (see paper_revision.md
    Topic 1 interpretation discussion): the active scenario/regime is drawn
    independently (weighted by `weights`, i.e. `# duplicates`) for the ship's first
    modelled year, then redrawn again -- independent of the previous draw -- at every
    subsequent `year in UNCERTAINTY_CLEARING_YEARS` (the redraw happens *before* that review
    year's own cost/decision are processed, since the review year is when the new
    regime is revealed and retrofit eligibility is gated on `year in
    UNCERTAINTY_CLEARING_YEARS`), and held fixed for the years in between. Each path replays
    the already-computed optimal policy `best_action` against `ship_op_cube` /
    `retrofit_lookup` using its own sequence of regime draws (retrofit/opportunity
    costs follow the same accumulation rules as `reconstruct_path_history`: retrofit
    cost is one-off at the retrofit year, opportunity cost recurs every year from the
    retrofit year onward and accumulates across repeated retrofits).

    This recovers the joint distribution of realized outcomes across all remaining
    review windows, which the pooled `future_costs` scalar (mean-only) structurally
    cannot represent -- used for `CVaR95`/`ci90_lower`/`ci90_upper` in
    `aggregate_results`. `mu` (the mean) is unaffected and continues to be computed
    exactly from `future_costs`/`baseline_pv`.

    Returns:
      - pv_additional_cost_paths: (n_paths,) array of simulated realized discounted
        (design - baseline) cost outcomes.
    """
    n_years, n_fuel_designs, n_scenarios = ship_op_cube.shape
    assert baseline_costs.shape == (n_years, n_scenarios), (
        f"baseline_costs shape {baseline_costs.shape} does not match "
        f"(n_years, n_scenarios) = {(n_years, n_scenarios)}"
    )
    probs = np.asarray(weights, dtype=np.float64)
    probs = probs / probs.sum()

    curr_fuel_idx = np.full(n_paths, initial_design_index, dtype=int)
    active_scenario = rng.choice(n_scenarios, size=n_paths, p=probs)

    total_design_dcf = np.zeros(n_paths, dtype=np.float64)
    total_baseline_dcf = np.zeros(n_paths, dtype=np.float64)
    retrofitted = np.zeros(n_paths, dtype=bool)
    opportunity_cost_per_path = np.zeros(n_paths, dtype=np.float64)

    for y_i, year in enumerate(years_arr):
        year = int(year)

        # redraw the active regime for this window BEFORE processing the review
        # year itself (except at y_i == 0, which already uses the initial draw above)
        if y_i > 0 and year in UNCERTAINTY_CLEARING_YEARS:
            active_scenario = rng.choice(n_scenarios, size=n_paths, p=probs)

        discount_factor = 1.0 / (1.0 + discount_rate) ** (year - current_year)

        actions_this_year = best_action[y_i, curr_fuel_idx, active_scenario]
        next_fuel_idx = curr_fuel_idx.copy()

        for act in set(actions_this_year.tolist()):
            if not (isinstance(act, str) and act.startswith("retrofit_to_")):
                continue
            target_design = act[len("retrofit_to_"):]
            target_idx = design_to_global[target_design]
            path_idx = np.where(actions_this_year == act)[0]
            for init_idx in np.unique(curr_fuel_idx[path_idx]):
                sub_mask = path_idx[curr_fuel_idx[path_idx] == init_idx]
                key = (segment_size_id, year, fuel_designs[init_idx], target_design)
                if key not in retrofit_lookup:
                    continue
                retrofit_cost, opportunity_cost = retrofit_lookup[key]
                total_design_dcf[sub_mask] += retrofit_cost * discount_factor
                opportunity_cost_per_path[sub_mask] += opportunity_cost
                retrofitted[sub_mask] = True
            next_fuel_idx[path_idx] = target_idx

        # recurring opportunity cost for already-retrofitted paths, charged every
        # year (including the retrofit year itself) from the retrofit year onward
        total_design_dcf[retrofitted] += opportunity_cost_per_path[retrofitted] * discount_factor

        op_cost = ship_op_cube[y_i, next_fuel_idx, active_scenario]
        total_design_dcf += op_cost * discount_factor
        total_baseline_dcf += baseline_costs[y_i, active_scenario] * discount_factor

        curr_fuel_idx = next_fuel_idx

    return total_design_dcf - total_baseline_dcf


def aggregate_results(
    ship_id: Any,
    ship_instance: pd.Series,
    baseline_pv: np.ndarray,
    baseline_tc_rate_pv: np.ndarray,
    baseline_tnm_pv: np.ndarray,
    future_costs: np.ndarray,
    initial_design_index: int,
    weights: List[float],
    pv_additional_cost_paths: np.ndarray,
    lambda_risk: float = 0.5,   # optional risk aversion coefficient
) -> Dict[str, Any]:
    """
    Build the result record for a given ship.

    `mu` (mean additional PV cost) and `design_cost_mean`/`baseline_cost_mean` are
    computed from `future_costs`/`baseline_pv` across the `n_scenarios` pooled columns
    -- exact for mean-based quantities under the tower-property equivalence established
    in paper_revision.md's Topic 1 interpretation discussion. `CVaR95`/`ci90_lower`/
    `ci90_upper` are instead computed from `pv_additional_cost_paths`, a Monte Carlo
    sample of *realized* outcomes across genuinely-remixing review windows (see
    `simulate_cvar_paths`), since the pooled `future_costs` scalar structurally cannot
    represent path-level dispersion/tail risk.
    """
    design_pv = future_costs[initial_design_index, :]
    pv_additional_cost = design_pv - baseline_pv
    mu = float(np.average(pv_additional_cost, weights=weights))

    var95 = float(np.percentile(pv_additional_cost_paths, 95))
    mask_tail = pv_additional_cost_paths >= var95
    if mask_tail.sum() > 0:
        cvar95 = float(pv_additional_cost_paths[mask_tail].mean())
    else:
        cvar95 = var95  # fallback (should rarely happen)
    ci90_lower = float(np.percentile(pv_additional_cost_paths, 5))
    risk_indicator = mu + lambda_risk * (cvar95 - mu)

    return {
        "ship_id": ship_id,
        "segment_size_id": ship_instance["segment_size_id"],
        "initial_fuel_design": ship_instance["initial_fuel_design"],
        "baseline_cost_mean": float(np.nanmean(baseline_pv)),
        "design_cost_mean": float(np.average(design_pv, weights=weights)),
        "pv_additional_cost": mu,
        "ci90_lower": ci90_lower,
        "ci90_upper": var95,
        "CVaR95": cvar95,
        "risk_indicator": risk_indicator,
        "discounted_tc_rates": float(np.nanmean(baseline_tc_rate_pv)),
        "discounted_tnm": float(np.nanmean(baseline_tnm_pv)),
        "scrap_year": ship_instance['scrap_year'],
        "generation": ship_instance['generation'],
    }


def compute_ship_option_values(
    operating_cost: pd.DataFrame,
    rep_ships: pd.DataFrame,
    retrofit_cost: pd.DataFrame,
    newbuild_premium: pd.DataFrame,
    drop_in_fuels: pd.DataFrame,
    engines_design_mapping: pd.DataFrame,
    DISCOUNT_rate: float = 0.08,
    baseline_fuel: str = "lsfo",
    min_age_for_retrofit: int = 0,
    retrofit_delay: int = 0,
    UNCERTAINTY_CLEARING_YEARS: List[int] = (2030, 2035, 2040, 2045, 2050),
    current_year: int = 2025,
    foresight: int = np.inf,
    risk_aversion: int = 1,
    COMPUTE_HISTORY: bool = True,
    CHECK: bool = False,
    DEBUG: bool = False,
    debug_file: str = "./debug_log.txt",
    n_cvar_paths: int = 2000,
    cvar_seed: int | None = None,
    df_engine_efficiency_ref: pd.DataFrame = None,
):
    """
    Calculates the value of the fuel flexibility of the ships contained in rep_ships, across a wide range of scenarios contained in operating_cost.
    The fuel flexibility comprises the flexibility of the drop-in fuels and the flexibility of the retrofits, compared to a conventional solution.
    Returns:
      - results_df: option values for the ship's initial fuel design
      - history_df_copy: cheapest path for the ship, with discounted costs
    """

    # ================================================================
    # COMPUTE_SHIP_OPTION_VALUES — STRUCTURE OVERVIEW
    # ------------------------------------------------
    # 1. Setup + input preparation
    # 2. Create operating-cost cubes (vectorised)
    # 3. Build retrofit lookup tables
    # 4. Preallocate history arrays (for speed)
    # 5. Loop over representative ships
    #       a. Prepare ship-specific slices (years, designs, engines, ship-level op_cube)
    #       b. Dynamic programming backwards in time
    #       c. Calculate PV relative to baseline
    #       d. Write results and per-year history
    # 6. Trim arrays, finalise results, merge newbuild premium
    # ================================================================

    # === 1. Setup + input preparation ===
    logger = prepare_logger(DEBUG, debug_file)

    # Input prep
    operating_cost_df, ships, retrofit_cost_df, drop_in_fuels_df = prepare_inputs(
        operating_cost, rep_ships, retrofit_cost, drop_in_fuels, current_year
    )

    # Basic sets and mappings
    scenario_ids = operating_cost_df["scenario_id"].unique()
    weights = (
        operating_cost[["scenario_id", "# duplicates"]]
        .drop_duplicates()["# duplicates"]
        .to_list()
    )
    modelling_years = sorted(
        [x for x in operating_cost["year"].unique() if x < current_year + foresight and current_year <= x]
    )
    fuel_designs = drop_in_fuels_df["fuel_design"].unique().tolist()
    engine_types = engines_design_mapping.engine_type.unique()
    REG_REG_CLASSES = operating_cost.reg_reg_class.unique()

    n_fuel_designs = len(fuel_designs)
    n_scenarios = len(scenario_ids)
    n_reg_reg = len(REG_REG_CLASSES)
    year_to_index = {y: i for i, y in enumerate(modelling_years)}
    design_to_global = {d: i for i, d in enumerate(fuel_designs)}

    if logger is not None:
        logger.debug(f"\n fuel_designs: {fuel_designs}")
        logger.debug(f"\n engine_types: {engine_types}")
        logger.debug(f"\n REG_REG_CLASSES: {REG_REG_CLASSES}")
        logger.debug(f"\n modelling_years: {modelling_years}")
        logger.debug(f"\n design_to_global: {design_to_global}")

    if len(weights) != n_scenarios:
        raise ValueError(
            f"Length of weights ({len(weights)}) does not match number of scenarios ({n_scenarios})"
        )

    # === 2. Create operating-cost cubes ===
    # op_cube_global is a 5D matrix with ID (n_years, n_designs, n_engine_types, n_reg_reg, n_scenarios)
    # it minimises the operating costs by choosing, for each design, the drop-in fuel with the lowest operating costs
    op_cube_global, _, op_pivot_baseline = build_op_cubes(
        modelling_years,
        fuel_designs,
        engine_types,
        scenario_ids,
        REG_REG_CLASSES,
        operating_cost_df,
        drop_in_fuels_df,
        engines_design_mapping,
        logger,
        DEBUG,
    )

    # === 3. Build retrofit lookup tables ===
    retrofit_lookup = build_retrofit_lookup(retrofit_cost_df)

    if logger is not None and not rep_ships.empty:
        segment_id = rep_ships.segment_size_id.iloc[0]
        filtered = {
            k: v
            for k, v in retrofit_lookup.items()
            if k[0] == segment_id
            and k[1] == 2035
            and k[2] == "conventional_+_rigid_sail"
        }
        logger.debug(
            f"\nRetrofit options for ship {segment_id}, year 2035, "
            f"initial design 'conventional_+_rigid_sail':\n{filtered}\n"
        )

    # === 4. Preallocate history arrays ===
    # We'll pre-allocate storage for history rows to avoid repeated dict/list allocations.
    # First do a lightweight pass to count the exact number of rows we will need.
    ships = ships.set_index("ship_id")
    results: List[Dict[str, Any]] = []

    history_storage, history_pointer, rep_ships_df = preallocate_history(
        ships, modelling_years, scenario_ids, year_to_index
    )

    n_loops = len(ships) / max(1, n_fuel_designs)
    cvar_rng = np.random.default_rng(cvar_seed)

    # === 5. Loop over representative ships ===
    # we do not loop through every single ship, but every ship segment * size * generation, as the calculations throughout the lifetime are the same
    # we then allocate the results to the different initial_fuel_design of the same ship segment * size * generation
    for _, rep_ship in tqdm(
        rep_ships_df.iterrows(),
        total=n_loops,
        desc="Ships",
        mininterval=5.0,
    ):
        # Available fuel_designs for this rep ship
        avail_fuel_designs = get_available_retrofits(rep_ship, retrofit_cost_df)
        # === 5.a. Prepare ship-specific inputs (years, designs, engines, ship_op_cube) ===
        (
            ship_op_cube,
            years_arr,
            _,
            _,
            cons_pa,
            tc_rate,
            transport_work_thousand_tnm_pa,
            segment_size_id,
            ships_subset,
            generation_year,
            lng_engine_type,
        ) = build_ship_level_op_cube(
            rep_ship,
            ships,
            op_cube_global,
            modelling_years,
            year_to_index,
            fuel_designs,
            avail_fuel_designs,
            engine_types,
            REG_REG_CLASSES,
            design_to_global,
            n_fuel_designs,
            n_scenarios,
            n_reg_reg,
            logger=logger,
            CHECK=CHECK,
            DEBUG = DEBUG,
            df_engine_efficiency_ref=df_engine_efficiency_ref,
        )

        if logger is not None:
            logger.debug("Going through each representative ship now")
            logger.debug(f"\n segment_size_id:{segment_size_id}")
            logger.debug(f"\n ships_subset: {ships_subset.columns, ships_subset[COLS_SEGMENT_SIZE_ID]}")
            logger.debug(f"\n rep_ship:{rep_ship[COLS_SEGMENT_SIZE_ID]}")
            logger.debug(f"\n years_arr:{years_arr}")
            logger.debug(f"\n avail_fuel_designs:{avail_fuel_designs}")

        # === 5.b. Dynamic programming backwards in time ===
        future_costs, best_action = run_dynamic_programming(
            ship_op_cube,
            avail_fuel_designs,
            design_to_global,
            retrofit_lookup,
            segment_size_id,
            years_arr,
            generation_year,
            min_age_for_retrofit,
            retrofit_delay,
            UNCERTAINTY_CLEARING_YEARS,
            weights,
            DISCOUNT_rate,
            DEBUG=DEBUG,
            logger=logger,
        )
        # future_costs are discounted to the max(year of build, current_year). we discount it back to current_year
        future_costs = future_costs / (1 + DISCOUNT_rate) ** (max(generation_year, current_year) - current_year)

        # === 5.c. Calculate present value relative to baseline ===
        # baseline cost = total_fuel_cost only, no compliance cost
        baseline_pv, baseline_tc_rate_pv, baseline_tnm_pv, baseline_costs_undiscounted = compute_baseline_costs(
            op_pivot_baseline,
            years_arr,
            baseline_fuel,
            cons_pa,
            tc_rate,
            transport_work_thousand_tnm_pa,
            DISCOUNT_rate,
            current_year,
        )

        if logger is not None:
            logger.debug(f"\n future_costs {future_costs}")
            logger.debug(f"\n best_action {best_action}")
            logger.debug(f"\n baseline_pv {baseline_pv}")

        # === 5.d. Write results and per-year history ===
        # now we run through the 4 initial designs available to that ship, and take te design PV of each initial design
        # cvar_paths_cache avoids re-running the Monte Carlo simulation for actual ships
        # that share the same initial_fuel_design within this representative ship
        cvar_paths_cache: Dict[int, np.ndarray] = {}
        for ship_id, ship_unique in ships_subset.iterrows():
            initial_design_index = design_to_global[ship_unique["initial_fuel_design"]]
            if initial_design_index not in cvar_paths_cache:
                cvar_paths_cache[initial_design_index] = simulate_cvar_paths(
                    n_paths=n_cvar_paths,
                    ship_op_cube=ship_op_cube,
                    baseline_costs=baseline_costs_undiscounted,
                    years_arr=years_arr,
                    fuel_designs=fuel_designs,
                    design_to_global=design_to_global,
                    best_action=best_action,
                    retrofit_lookup=retrofit_lookup,
                    segment_size_id=segment_size_id,
                    UNCERTAINTY_CLEARING_YEARS=UNCERTAINTY_CLEARING_YEARS,
                    weights=weights,
                    discount_rate=DISCOUNT_rate,
                    current_year=current_year,
                    initial_design_index=initial_design_index,
                    rng=cvar_rng,
                )
            rec = aggregate_results(
                ship_id,
                ship_unique,
                baseline_pv,
                baseline_tc_rate_pv,
                baseline_tnm_pv,
                future_costs,
                initial_design_index,
                weights,
                cvar_paths_cache[initial_design_index],
                risk_aversion,
            )
            results.append(rec)

            if COMPUTE_HISTORY:
                # Build cheapest path history for this representative ship and its actual ships
                history_pointer = reconstruct_path_history(
                    ship_id = ship_id,
                    ship_instance = ship_unique,
                    segment_size_id = segment_size_id,
                    years_arr = years_arr,
                    fuel_designs = fuel_designs,
                    design_to_global = design_to_global,
                    best_action = best_action,
                    ship_op_cube = ship_op_cube,
                    retrofit_lookup = retrofit_lookup,
                    scenario_ids = scenario_ids,
                    discount_rate = DISCOUNT_rate,
                    current_year = current_year,
                    weights = weights,
                    initial_design_index = initial_design_index,
                    lng_engine_type = lng_engine_type,
                    history_storage = history_storage,
                    history_pointer = history_pointer,
                    DEBUG = DEBUG,
                    logger = logger,
                )

            # Optional internal check: recompute PV from history vs dynamic programming (unchanged logic)
            # Check that discounted history matches design_pv
            if CHECK and COMPUTE_HISTORY:
                history_df_tmp = pd.DataFrame(history_storage).copy()
                for s_i, scen in enumerate(scenario_ids):
                    df_s = history_df_tmp[
                        (history_df_tmp["ship_id"] == ship_id)
                        & (history_df_tmp["scenario_id"] == scen)
                    ]
                    total_dcf = (
                        df_s["op_cost_discounted"].sum()
                        + df_s["retrofitting_cost_discounted"].sum()
                        + df_s["opportunity_cost_discounted"].sum()
                    )
                    design_pv = future_costs[initial_design_index, s_i]
                    if not np.isclose(total_dcf, design_pv, rtol=1e3):
                        print(
                            f"WARNING: mismatch for ship {ship_id}, scenario {scen}: "
                            f"DCF={total_dcf:.2f}, design_pv={design_pv:.2f}"
                        )

    # === 6. Finalise results and merge newbuild premium ===

    # Finalise history
    if COMPUTE_HISTORY:
        if history_pointer < len(next(iter(history_storage.values()))):
            for k in history_storage:
                history_storage[k] = history_storage[k][:history_pointer]
        history_df = pd.DataFrame(history_storage)
        for col in ["initial_fuel_design", "fuel_start", "fuel_end", "decision"]:
            history_df[col] = history_df[col].astype("category")
    else:
        history_df = pd.DataFrame()

    results_df = pd.DataFrame(results)

    pre_merge_len = len(results_df)
    results_df = results_df.merge(
        newbuild_premium[["ship_id", "newbuild_premium"]],
        on="ship_id",
        how="left",
    )
    if pre_merge_len != len(results_df):
        raise RuntimeError(
            f"Length of results_df changed after merging newbuild_premium: "
            f"before={pre_merge_len}, after={len(results_df)}"
        )
    # we depreciate the newbuild premium to the age of the ship
    results_df['capex_premium'] = (
        results_df['newbuild_premium']
        *(1
          -(current_year - results_df['generation'])
          /(results_df['scrap_year'] - results_df['generation'])
        )
    )

    for var in ["pv_additional_cost", "ci90_upper", "ci90_lower", "risk_indicator", 'CVaR95']:
        results_df[var] += results_df["capex_premium"]
        results_df[f"{var}_over_tc"] = results_df[var]/results_df["discounted_tc_rates"]
        results_df[f"{var}_over_tnm"] = results_df[var]/results_df["discounted_tnm"]

    return results_df, history_df

def compute_baseline_costs(
    op_pivot_baseline: pd.DataFrame,
    years_arr: np.ndarray,
    baseline_fuel: str,
    annual_consumption: float,
    tc_rate: float,
    transport_work_thousand_tnm_pa: float,
    discount_rate: float,
    current_year: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute baseline PV of operating costs, TC rates, and tonne-nautical miles.

    Also returns `baseline_costs_undiscounted` (n_years, n_scenarios): the per-year,
    per-scenario baseline operating cost before discounting/summing across years.
    Used by `simulate_cvar_paths` to replay a baseline cost consistent with a Monte
    Carlo path's active scenario at each year (the aggregated `baseline_pv` below
    stays tied to a fixed scenario column for the whole horizon, which is what the
    exact mean-based `pv_additional_cost` needs, but not what a genuinely-remixing
    path replay needs).
    """
    discount_factors = 1.0 / ((1.0 + discount_rate) ** (years_arr - current_year))[:, None]

    baseline_prices = (
        op_pivot_baseline.loc[
            (slice(None), years_arr, "All", "noneu_to_noneu"), baseline_fuel
        ]
        .unstack(level=0)
        .values
    )

    # nan_to_num matches the np.nansum below (missing scenario/year combos contribute 0)
    baseline_costs_undiscounted = np.nan_to_num(annual_consumption * baseline_prices, nan=0.0)
    baseline_costs = baseline_costs_undiscounted * discount_factors
    baseline_pv = np.nansum(baseline_costs, axis=0)

    baseline_tc_rate = tc_rate * 365.25 * discount_factors
    baseline_tc_rate_pv = np.nansum(baseline_tc_rate, axis=0)

    baseline_tnm = transport_work_thousand_tnm_pa * discount_factors
    baseline_tnm_pv = np.nansum(baseline_tnm, axis=0)

    return baseline_pv, baseline_tc_rate_pv, baseline_tnm_pv, baseline_costs_undiscounted