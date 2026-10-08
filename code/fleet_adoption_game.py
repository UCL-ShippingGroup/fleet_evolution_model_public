import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import zipfile
import json
import copy
import uuid

from su_market import iterate_su_market, choose_fuel_given_su, build_yearly_fuel_group_caps, update_prev_year_fuel_group_avail
import technology
import utility_functions as util
import load_data as load
from ship_econ import (
    run_vessel_scenario,
)
# from default_assumptions import FUEL_GROUPS_AVAIL

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)

#  ====== GAME SCENARIOS  ======
def set_vessel_actions(imo_type, vessel_prev_engine,
                       retrofit_options, newbuild_options):
    '''for each vessel, populate available actions based on latest engine'''

    # Wind Type for Container vs other ship types
    if imo_type == 'Container':
        wind_drop = 'rigid_sail'
    else:
        wind_drop = 'kite'

    retrofit_options = {
        opt for opt in retrofit_options
        if wind_drop not in opt
    }

    newbuild_options = {
        opt for opt in newbuild_options
        if wind_drop not in opt
    }

    # Retrofit to conventional+ccs/wind only possible with conventional vessel
    if vessel_prev_engine!='conventional':
        retrofit_options.discard('conventional_+_ccs')
        retrofit_options.discard('conventional_+_kite')
        retrofit_options.discard('conventional_+_rigid_sail')

    else:
        pass

    # Set vessel actions
    vessel_action_tree = {
        "keep": {vessel_prev_engine},
        "retrofit": set(retrofit_options), #if retrofit_options else {"ammonia_df", "lng_df"},
        "newbuild": set(newbuild_options)    #, #if newbuild_options else {"conventional", "ammonia_df", "lng_df"},
    }

    combinations = []

    for action, engines in vessel_action_tree.items():
        for engine in engines:
            combinations.append({
                "action": action,
                "engine": engine,
            })

    # make output df
    df_vessel_combos = pd.DataFrame(combinations)

    # make scenario label
    df_vessel_combos['choice_id'] = range(1,len(df_vessel_combos)+1)
    df_vessel_combos['scenario'] = df_vessel_combos['action'] + '_' + df_vessel_combos['engine'] #+ '_' + df_vessel_combos['fuel']
    df_vessel_combos['scenario'] =df_vessel_combos['scenario'].str.rstrip('_')

    return df_vessel_combos, vessel_action_tree


def build_engine_for_peer_influence(engine_series, ccs_series):
    """Build peer-influence engine labels from base engine + CCS only (exclude wind)."""
    base_engine = engine_series.astype(str).str.split('_+_', regex=False).str[0] + "_" + ccs_series.astype(str)
    return base_engine

def compute_scenario_dcf(
        df_vessel_scenarios, start_year, planning_horizon_years,
        user_params, engines_design_mapping, engine_fuels_dict, df_drop_in_fuels_ref, df_energy_density_ref, 
        df_imo_targets_data, df_fueleu_targets_data,
        df_wtt_carbon_intensity_ref, df_ttw_carbon_intensity_ref,
        df_reward_ref, df_tc_rates_ref, df_pilot_fuel_input_ref,
        ru_price_data, initial_su_price_data, fuel_price_data, df_capex_ref, df_retrofit_cost_ref,
        df_opex_conventional_ref, df_opex_nonconv_wages_ref, df_opex_nonconv_insurance_ref,
        df_opex_nonconv_ssf_ref, df_opex_nonconv_isf_ref, df_opex_nonconv_spares_ref,
        df_operational_data_ref,
        df_speed_ref, df_efficiency_ref, df_engine_efficiency_ref,
):

    individual_ships_cols = ["ship_year_id", "ship_name", "archetype"]
    same_ships_cols = ['imo_type', 'size_id', 'year_built', 'action', 'engine', 'prev_vessel_year_built', 'prev_engine']

    missing_cols = [
        c for c in (individual_ships_cols + same_ships_cols)
        if c not in df_vessel_scenarios.columns
    ]
    assert not missing_cols, f"Missing grouping columns in df_vessel_scenarios: {missing_cols}"

    df_unique_scenarios = (
        df_vessel_scenarios
        .drop(columns=individual_ships_cols, errors='ignore')
        .drop_duplicates(subset=same_ships_cols)
        .reset_index(drop=True)
    )
    df_results, df_rrc_results = run_vessel_scenario(
        start_year,
        planning_horizon_years,
        user_params,
        df_unique_scenarios,
        engines_design_mapping,
        engine_fuels_dict,
        df_drop_in_fuels_ref,
        df_energy_density_ref,
        df_imo_targets_data,
        df_fueleu_targets_data,
        df_wtt_carbon_intensity_ref,
        df_ttw_carbon_intensity_ref,
        df_reward_ref,
        df_tc_rates_ref,
        df_pilot_fuel_input_ref,
        ru_price_data,
        initial_su_price_data,
        fuel_price_data,
        df_capex_ref,
        df_retrofit_cost_ref,
        df_opex_conventional_ref,
        df_opex_nonconv_wages_ref,
        df_opex_nonconv_insurance_ref,
        df_opex_nonconv_ssf_ref,
        df_opex_nonconv_isf_ref,
        df_opex_nonconv_spares_ref,
        df_operational_data_ref,
        game_year=start_year, df_speed_ref=df_speed_ref, df_efficiency_ref=df_efficiency_ref,
        df_engine_efficiency_ref=df_engine_efficiency_ref,
    )

    assert len(df_results) > 0, "No DCF rows returned from grouped scenario evaluation"
    assert all(c in df_results.columns for c in same_ships_cols), (
        f"Grouped DCF results missing merge columns: "
        f"{[c for c in same_ships_cols if c not in df_results.columns]}"
    )
    assert not df_results.duplicated(subset=same_ships_cols).any(), (
        "Grouped DCF results are not unique on same_ships_cols"
    )

    expected_len = len(df_vessel_scenarios)
    df_dcf_to_add = df_vessel_scenarios[individual_ships_cols + same_ships_cols].merge(
        df_results,
        on=same_ships_cols,
        how='left',
    )
    assert len(df_dcf_to_add) == expected_len, (
        f"Post-merge DCF length mismatch: expected {expected_len}, got {len(df_dcf_to_add)}"
    )

    if len(df_rrc_results) > 0:
        assert all(c in df_rrc_results.columns for c in same_ships_cols), (
            f"Grouped RRC results missing merge columns: "
            f"{[c for c in same_ships_cols if c not in df_rrc_results.columns]}"
        )
        df_rrc_fuel_to_add = df_vessel_scenarios[individual_ships_cols + same_ships_cols].merge(
            df_rrc_results,
            on=same_ships_cols,
            how='left',
        )
    else:
        df_rrc_fuel_to_add = pd.DataFrame()

    df_dcf_to_add.reset_index(drop=True, inplace=True)
    df_rrc_fuel_to_add.reset_index(drop=True, inplace=True)

    return df_dcf_to_add, df_rrc_fuel_to_add

def update_wind_ccs_flags(vessel_row):
    # Strip whitespace first
    engine_str = str(vessel_row['engine']).strip()

    # Initialize / reset
    vessel_row['wind'] = ""
    vessel_row['ccs'] = ""

    # Check for CCS
    if "_+_ccs" in engine_str:
        vessel_row['ccs'] = "ccs"

    # Check for wind
    if "_+_rigid_sail" in engine_str:
        vessel_row['wind'] = "rigid_sail"
    elif "_+_kite" in engine_str:
        vessel_row['wind'] = "kite"

    return vessel_row

def get_vessel_scenarios(vessel_row, game_year, user_retrofit_options, newbuild_options, df_retrofit_cost_ref):
    '''
    Precompute all combinations of action/engine/fuel for each vessel

    Vessels younger than lifetime can keep, retrofit, scrap
    Vessels at or beyond lifetime can scrap or newbuild
    Retrofits of the same engine as “keep” are removed
    Retrofits are disabled for vessels near the end of life
    Newbuilds reset vessel age and year built
    '''
    # Get available combos for this vessel
    # get available / allowed retrofits
    vessel_lifetime = vessel_row['lifetime']
    vessel_row = update_wind_ccs_flags(vessel_row)
    imo_type = vessel_row['imo_type']

    retrofit_options= technology.get_retrofit_options(df_retrofit_cost_ref, vessel_row, user_retrofit_options)

    # --- HARD CONSTRAINT: cannot remove CCS once installed ---
    prev_engine = vessel_row['prev_engine']
    prev_has_ccs = 'ccs' in prev_engine.lower()

    if prev_has_ccs:
        retrofit_options = [
            r for r in retrofit_options
            if 'ccs' in r.lower()
        ]

    # --- HARD CONSTRAINT: cannot remove wind once installed ---
    prev_wind = vessel_row.get('wind', "")
    if prev_wind not in (None, "", {}):
        retrofit_options = [
            r for r in retrofit_options
            if prev_wind in r
        ]

    df_vessel_combos, vessel_action_tree = set_vessel_actions(imo_type, prev_engine,
                                                              retrofit_options, newbuild_options)

    # Create a DataFrame with one row per combo
    vessel_df = pd.DataFrame([vessel_row])
    vessel_expanded = vessel_df.loc[vessel_df.index.repeat(len(df_vessel_combos))].reset_index(drop=True)
    combo_cols = ['engine', 'action', 'choice_id', 'scenario']
    vessel_expanded = vessel_expanded.drop(columns=[c for c in combo_cols if c in vessel_expanded.columns])

    df_scenarios = pd.concat([vessel_expanded, df_vessel_combos], axis=1)

    # --- Filter scenarios based on vessel lifetime ---
    df_scenarios = df_scenarios[
        ((df_scenarios['vessel_age'] < vessel_lifetime) &
         (df_scenarios['action'].isin(['keep', 'retrofit', 'newbuild', 'scrap']))) |    # allowed actions when age < lifetime
        ((df_scenarios['vessel_age'] >= vessel_lifetime) &
         (df_scenarios['action'].isin(['newbuild'])))     # allowed actions when age > lifetime
    ]

    # Reset age and year built for newbuilds
    df_scenarios.loc[df_scenarios['action'] == 'newbuild', 'vessel_age'] = 0
    df_scenarios.loc[df_scenarios['action'] == 'newbuild', 'year_built'] = game_year

    # --- Fleet growth ships: restrict to newbuild only ---
    # Ships injected for fleet growth have year_built == game_year (vessel_age == 0).
    # They must choose a newbuild technology; keep and retrofit do not apply.
    if vessel_row['year_built'] == game_year:
        df_scenarios = df_scenarios[df_scenarios['action'] == 'newbuild']

    # --- Remove retrofits of the same engine type as "keep" ---
    keep_rows = df_scenarios[df_scenarios['action'] == 'keep']
    if not keep_rows.empty:
        keep_engine = keep_rows['engine'].iloc[0]
        df_scenarios = df_scenarios[
            ~((df_scenarios['action'] == 'retrofit') & (df_scenarios['engine'] == keep_engine))
        ]

    # --- Disable retrofits for vessels near end of life ---
    retrofit_age_limit = vessel_lifetime - 1
    df_scenarios = df_scenarios[
        ~((df_scenarios['vessel_age'] > retrofit_age_limit)
          & (df_scenarios['action'] == 'retrofit'))]

    # --- MAKE CCS AND WIND LABELS ---
    ## CCS flag
    df_scenarios['ccs'] = df_scenarios['engine'].str.contains('ccs', case=False, regex=False).map({True: 'ccs', False: ""})
    ## Wind type (kite / rigid_sail / None)
    df_scenarios['wind'] = (df_scenarios['engine'].str.extract(r'(kite|rigid_sail)', expand=False).fillna(""))  # ← make empty string if no wind tech

    return df_scenarios, vessel_action_tree

def choose_action(group):
    return group.loc[group['utility'].idxmax()]


def classify_as_early_scrap(df, year_step):
    """
    Compute generic early-scrap timing for rows with replacement context.

    Returns a DataFrame with:
    - scrap_early_years
    - is_early_scrap
    """
    required_cols = ['action', 'lifetime', 'prev_vessel_age']
    missing_required = [c for c in required_cols if c not in df.columns]
    assert len(missing_required) == 0, (
        f"Missing required columns for early-scrap classification: {missing_required}. "
        f"Available columns: {df.columns.tolist()}"
    )
    assert year_step > 0, f"year_step must be positive, got {year_step}"

    df = df.copy()
    mask_replacement_candidate = (
        (df['action'] == 'newbuild')
        & (df['prev_vessel_age'] < df['lifetime'])
    )
    df['scrap_early_years'] = np.where(
        mask_replacement_candidate,
        df['lifetime'] - df['prev_vessel_age'],
        np.nan,
    )
    df['is_early_scrap'] = (
        mask_replacement_candidate
        & (df['scrap_early_years'] >= year_step)
    )

    return df

def reclassify_action(df, year_step):
    '''
    update action column in results to differentiate between:
     - newbuild: newbuild at end of lifetime
     - early_scrap_replaced_by_newbuild: early scrapping to replace with a new (supply-side risk)
    - early_scrap_to_reduce_fleet: fleet shrinkage removes ship with >= year_step years of life remaining (demand-side risk materialising in early scrapping)
     - scrap_not_replaced: fleet shrinkage removes ship at/near end of life (demand-side trade reduction in line with fleet depreciation)
     - newbuild to grow fleet: trade expansion --> fleet expansion --> new ships needed (on top of those replaced)
    '''

    df = classify_as_early_scrap(df, year_step)

    mask_ss_early = (
            (df['action'] == 'newbuild') &
            (df['is_early_scrap'])
    )
    df.loc[mask_ss_early, 'action'] = 'early_scrap_replaced_by_newbuild'

    mask_growth = (
        (df['action'] == 'newbuild')
        & (df['modelled_ships_nb'].isna())
    )
    df.loc[mask_growth,'action'] = "newbuild_to_grow_fleet"

    mask_ds_early = (
        (df['action'] == 'scrap_not_replaced')
        & (df['vessel_age'] < df['lifetime'])
        & ((df['lifetime'] - df['vessel_age']) >= year_step)
    )
    df.loc[mask_ds_early, 'action'] = 'early_scrap_to_reduce_fleet'

    return df

# ====== PLAY GAMES ======
def update_prev_columns(df: pd.DataFrame, game_year: int):
    """
    Update 'prev_*' columns in a DataFrame safely if the source columns exist.
    Computes 'prev_vessel_age' if 'prev_vessel_year_built' exists.
    Handles engine and fuel previous state as well.
    """
    mapping = {
        'prev_vessel_capex': 'vessel_capex',
        'prev_capex_annualized': 'capex_annualized',
        'prev_vessel_year_built': 'year_built',
        'prev_engine': 'engine',
    }

    for prev_col, source_col in mapping.items():
        if source_col in df.columns:
            mask = df[df.year_built != game_year].index
            df.loc[mask, prev_col] = df.loc[mask, source_col]
        else:
            print(f"Column {source_col} missing from the dataset")

    # Compute vessel age if year built exists
    if 'prev_vessel_year_built' in df.columns:
        df['prev_vessel_age'] = game_year - df['prev_vessel_year_built']


def create_fleet_growth_ships(df_active, game_year, df_fleet_growth, year_step,
                               df_speed_ref, df_operational_data_ref, *, rng, early_movers_fraction):
    """
    Handle fleet size changes for the current game_year.

    For each (imo_type, size_id) group, looks up the annual growth_rate from
    df_fleet_growth (already filtered to the correct scenario) and compounds
    the individual rates for every year in the current time step:
        compounded_factor = (1 + r_y1) * (1 + r_y2) * ... * (1 + r_yn)
        n_change = round(n_active * (compounded_factor - 1))

    Positive growth_rate -> add n_change new ships, each cloned from the most
    recently built ship of that group.  New ships are forced into a newbuild
    investment decision (prev_vessel_age = np.inf).

    Negative growth_rate -> scrap the abs(n_change) oldest ships of that group
    (lowest year_built).  These are demand-driven retirements, not economic
    decisions, so they bypass the investment / operating decision phases.
    Their stranded_value is computed using the same formula as ship_econ:
        prev_vessel_residual_value_usd = salvage_value_usd
            + (prev_vessel_capex - salvage_value_usd) * (1 - prev_vessel_age / lifetime)
        stranded_value = max(prev_vessel_residual_value_usd - salvage_value_usd, 0)

    Args:
        df_active (pd.DataFrame): Current active fleet with prev_* columns
            already populated by update_prev_columns().
        game_year (int): Current simulation year.
        df_fleet_growth (pd.DataFrame): Growth rates filtered to the correct
            scenario; columns: imo_type, year, growth_rate (annual).
        year_step (int): Number of years per simulation step; used to compound
            the annual growth rate.

    Returns:
        tuple: (df_new_ships, df_demand_scrapped)
            df_new_ships: ships to add to df_active (may be empty DataFrame).
            df_demand_scrapped: ships removed from the fleet with action =
                'scrap_not_replaced' and stranded_value populated (may be empty).
    """
    new_ships_list = []
    demand_scrapped_list = []

    assert df_speed_ref is not None, "df_speed_ref is required"

    for (imo_type, size_id), df_group in df_active.groupby(['imo_type', 'size_id']):
        mask_type = df_fleet_growth['imo_type'] == imo_type
        assert mask_type.any(), f"Not fleet growth inputs for {imo_type}"

        # Compound each individual annual growth rate across all years in this time step
        compounded_factor = 1.0
        for yr in range(game_year - year_step + 1, game_year + 1):
            df_rate_row = df_fleet_growth[mask_type & (df_fleet_growth['year'] == yr)]
            assert len(df_rate_row) > 0, (
                f"Missing growth rate for imo_type={imo_type}, year={yr}"
            )
            compounded_factor *= (1.0 + float(df_rate_row['growth_rate'].iloc[0]))
        n_change = int(round(len(df_group) * (compounded_factor - 1.0)))

        # Energy efficiency threading: fleet-size correction for speed reduction.
        # When the operating speed changes between years, a different number of ships
        # is required to provide the same transport capacity at the new speed.
        # note that a 1% decrease in speed leads to somewhat lesser decrease in distance hence transport work
        # so we correct for the elasticity
        # Correction formula: n_speed = round(n_active * ((1-s_{t-Δt})/(1-s_t) - 1))
        # Positive when s_t > s_{t-Δt} (slower this year → more ships needed).
        prev_year = game_year - year_step
        row_t = df_speed_ref[df_speed_ref['year'] == game_year]
        row_prev = df_speed_ref[df_speed_ref['year'] == prev_year]
        assert len(row_t) > 0 and len(row_prev) > 0, f"empty speed for {game_year} or {prev_year}"
        s_t = float(row_t['speed_reduction'].iloc[0])
        s_t_prev = float(row_prev['speed_reduction'].iloc[0])
        df_elasticity = df_operational_data_ref[
            (df_operational_data_ref.imo_type == imo_type) 
            & (df_operational_data_ref.size_id == size_id)
        ]
        assert len(df_elasticity) > 0, (
            f"Missing elasticity_distance_by_speed row for imo_type={imo_type}, size_id={size_id}"
        )
        elasticity = df_elasticity['elasticity_distance_by_speed'].iloc[0]
        if s_t > 0 or s_t_prev > 0:
            denom = (1.0 - s_t * elasticity)
            assert denom > 0, (
                f"Invalid speed correction denominator for imo_type={imo_type}, size_id={size_id}: "
                f"1 - s_t*elasticity = {denom} (s_t={s_t}, elasticity={elasticity})"
            )
            n_speed_correction = int(round(
                len(df_group) * ((1.0 - s_t_prev*elasticity) / denom - 1.0)
            ))
            n_change += n_speed_correction

        if n_change == 0:
            continue

        if n_change > 0:
            # --- Positive growth: clone youngest ship as newbuild template ---
            template_idx = df_group['year_built'].idxmax()
            df_template = df_group.loc[[template_idx]]

            for i in range(1, n_change + 1):
                df_new = df_template.copy()
                ship_name = (
                    str(imo_type).replace(' ', '-') + '_' +
                    str(size_id) + '_' +
                    str(game_year) + '_10' +
                    f'{i:03d}'
                )
                df_new['ship_name'] = ship_name
                df_new['year_built'] = game_year
                df_new['vessel_age'] = 0
                # Set prev_vessel_year_built so that ship_econ computes prev_vessel_age == lifetime.
                # This gives dcf_adjustment = 0 and stranded_value = 0 (no real previous vessel),
                # and prevents early_scrap_replaced_by_newbuild mislabelling (lifetime < lifetime is False).
                df_new['prev_vessel_year_built'] = game_year - df_new['lifetime']
                df_new['modelled_ships_nb'] = np.nan
                df_new['ship_year_id'] = f'ship_year_{uuid.uuid4()}'
                df_new['ship_id'] = f'ship_{uuid.uuid4()}'
                new_ships_list.append(df_new)

        else:
            # --- Negative growth: scrap the oldest ships (demand-driven) ---
            n_remove = abs(n_change)
            df_to_scrap = df_group.sort_values('year_built').head(n_remove).copy()
            df_to_scrap['action'] = 'scrap_not_replaced'

            # Stranded value: same formula as ship_econ.run_vessel_scenario
            stranded_cols = ['prev_vessel_capex', 'salvage_value_usd', 'prev_vessel_age', 'lifetime']
            missing_stranded_cols = [c for c in stranded_cols if c not in df_to_scrap.columns]
            assert not missing_stranded_cols, (
                f"Missing stranded-value columns for scrap_not_replaced in ({imo_type}, {size_id}) "
                f"game_year={game_year}: {missing_stranded_cols}"
            )
            df_to_scrap['prev_vessel_residual_value_usd'] = (
                df_to_scrap['salvage_value_usd']
                + (df_to_scrap['prev_vessel_capex'] - df_to_scrap['salvage_value_usd'])
                * (1 - df_to_scrap['prev_vessel_age'] / df_to_scrap['lifetime'])
            )
            df_to_scrap['stranded_value'] = np.maximum(
                df_to_scrap['prev_vessel_residual_value_usd'] - df_to_scrap['salvage_value_usd'],
                0.0,
            )

            demand_scrapped_list.append(df_to_scrap)

    df_new_ships = pd.concat(new_ships_list, ignore_index=True) if new_ships_list else pd.DataFrame()
    if not df_new_ships.empty:
        # Random early/mass split; draws from the scenario's own rng (not global NumPy state),
        # so this is reproducible and independent of other concurrently running scenarios.
        load.assign_archetypes(df_new_ships, early_fraction=early_movers_fraction, rng=rng)
    df_demand_scrapped = pd.concat(demand_scrapped_list, ignore_index=True) if demand_scrapped_list else pd.DataFrame()
    return df_new_ships, df_demand_scrapped

def run_nash_equilibrium_iterative(
        df_vessels, start_year, end_year, planning_horizon_years, user_params,
        engines_design_mapping, engine_fuels_dict, df_drop_in_fuels_ref, df_energy_density_ref,
        df_imo_targets_data, df_fueleu_targets_data,
        df_wtt_carbon_intensity_ref, df_ttw_carbon_intensity_ref, df_reward_ref,
        df_tc_rates_ref, df_pilot_fuel_input_ref, ru_price_data, initial_su_price_data,
        su_market_flag, fuel_price_data, df_growth_infra, df_capex_ref, df_retrofit_cost_ref,
        df_opex_conventional_ref, df_opex_nonconv_wages_ref, df_opex_nonconv_insurance_ref,
        df_opex_nonconv_ssf_ref, df_opex_nonconv_isf_ref, df_opex_nonconv_spares_ref,
        df_operational_data_ref,
        ENGINE_PEER_FACTOR, SHAPE_PEER_INFLUENCE,
        critical_mass_engine, slope_engine, max_iterations,
        prev_year_fuel_group_avail,
        df_fleet_growth,
        df_speed_ref, df_efficiency_ref, df_engine_efficiency_ref,
):
    """
    Iterative Nash equilibrium per year with peer influence, respecting vessel lifetime rules:
    - Active vessels choose among keep/retrofit/scrap (if age < lifetime)
    - Vessels at or past lifetime choose scrap or newbuild
    - Iteratively applies peer influence feedback until convergence
    - WITHOUT su calculation for unselected vessels - SU market updates su price for next year.

    """
    print("Running Nash equilibrium per year (iterative with lifetime rules)...")

    # Remove previous debug log if it exists (only once at start of run)
    if bool(user_params.get('DEBUG', False)):
        from pathlib import Path
        outdir = Path(user_params.get('RUN_OUTPUT_DIR', '.'))
        outdir.mkdir(parents=True, exist_ok=True)
        logpath = outdir / 'debug_log.txt'
        if logpath.exists():
            logpath.unlink()

    df_vessels_working = df_vessels.copy()
    length_fleet = len(df_vessels_working)
    game_years_results = []
    df_rrc_fuel = []
    df_dcf = pd.DataFrame()
    df_peer_tracking = []
    prev_year_fuel_group_avail = copy.deepcopy(prev_year_fuel_group_avail)
    # peer-influence scaling factor: DCF of conventional ships at start_year, held constant for
    # all later years. Keyed by ship_name (stable across years, unlike ship_year_id which is
    # regenerated every game_year), set once below when game_year == start_year.
    scaling_factor_peer_influence = None

    # we collect from which year the EU policy is dropped, to avoid running through each reg_reg_class afterwards
    df_no_eu_policy = df_fueleu_targets_data[
        (df_fueleu_targets_data.scenario == user_params['fueleu_scenario'])
        &(df_fueleu_targets_data.FuelEU == 100)
    ]
    if len(df_no_eu_policy) > 0: 
        year_eu_policy_dropped = df_no_eu_policy.year.min()
    else:
        year_eu_policy_dropped = 2100
    user_params['year_eu_policy_dropped'] = year_eu_policy_dropped
    print(f"Scenario {user_params['scenario_number']} (Nash equilibrium) year_eu_policy_dropped: {year_eu_policy_dropped}")

    year_step = int(user_params['year_step'])
    for game_year in range(start_year, end_year + 1, year_step):
        # Each game year has two sequential decision phases:
        #   PHASE 1 – INVESTMENT DECISION: each ship evaluates keep / retrofit / newbuild
        #            scenarios, iterated with peer-influence feedback until Nash convergence.
        #   PHASE 2 – OPERATING DECISION: given the converged design, each ship selects
        #            its fuel mix for the current year, optionally via the SU market.
        print(f"Scenario {user_params['scenario_number']} (Nash equilibrium), year {game_year}")

        df_active = df_vessels_working.copy()
        df_active["ship_year_id"] = [
            f"ship_year_{uuid.uuid4()}"
            for _ in range(len(df_active))
        ]

        # --- update vessel age according to the game year ---
        df_active['vessel_age'] = game_year - df_active['year_built']

        # --- Update previous fields, if columns exist ---
        update_prev_columns(df_active, game_year)

        # --- Fleet growth: inject new ships / remove demand-scrapped ships ---
        demand_scrapped_this_year = pd.DataFrame()
        if game_year > start_year:
            df_growth_new, df_demand_scrapped = create_fleet_growth_ships(
                df_active, game_year, df_fleet_growth, year_step,
                df_speed_ref=df_speed_ref, df_operational_data_ref = df_operational_data_ref,
                rng=user_params['rng'],
                early_movers_fraction=float(user_params['early_movers_fraction']),
            )
            if len(df_growth_new) > 0:
                df_active = pd.concat([df_active, df_growth_new], ignore_index=True)
                length_fleet += len(df_growth_new)
            if len(df_demand_scrapped) > 0:
                demand_scrap_ids = set(df_demand_scrapped['ship_year_id'])
                df_active = df_active[~df_active['ship_year_id'].isin(demand_scrap_ids)].copy()
                length_fleet -= len(df_demand_scrapped)
                demand_scrapped_this_year = df_demand_scrapped

        individual_ships_cols = ["ship_year_id", "ship_name", "archetype"]
        same_ships_cols = ['imo_type', 'size_id', 'prev_vessel_year_built', 'prev_engine']

        missing_cols = [
            c for c in (individual_ships_cols + same_ships_cols)
            if c not in df_active.columns
        ]
        assert not missing_cols, f"Missing grouping columns in df_active: {missing_cols}"

        # ===================================================================
        # PHASE 1 – INVESTMENT DECISION (keep / retrofit / newbuild)
        # Enumerate feasible actions, compute DCF, then iterate with peer-
        # influence feedback until the Nash equilibrium converges (i.e. no
        # ship wants to unilaterally change its choice).
        # ===================================================================

        # 1a. Enumerate feasible investment actions for each unique ship profile
        df_unique_active = (
            df_active
            .drop(columns=individual_ships_cols)
            .drop_duplicates(subset=same_ships_cols)
            .reset_index(drop=True)
        )

        df_scenarios_list = []
        for _, vessel_row in df_unique_active.iterrows():
            df_scenarios, _ = get_vessel_scenarios(
                vessel_row,
                game_year,
                user_params['retrofit_options'],
                user_params['newbuild_options'],
                df_retrofit_cost_ref,
            )
            df_scenarios_list.append(df_scenarios)
            
        df_unique_vessel_scenarios = pd.concat(df_scenarios_list, ignore_index=True)
        assert all(c in df_unique_vessel_scenarios.columns for c in same_ships_cols), (
            f"Grouped scenario rows missing merge columns: "
            f"{[c for c in same_ships_cols if c not in df_unique_vessel_scenarios.columns]}"
        )

        df_vessel_scenarios = df_active[individual_ships_cols + same_ships_cols].merge(
            df_unique_vessel_scenarios,
            on=same_ships_cols,
            how='left',
        )
        assert len(df_vessel_scenarios) >= len(df_active), (
            f"Scenario merge unexpectedly reduced rows: active={len(df_active)}, scenarios={len(df_vessel_scenarios)}"
        )
        required_scenario_cols = ['action', 'engine', 'choice_id', 'scenario']
        for col in required_scenario_cols:
            assert df_vessel_scenarios[col].notna().all(), (
                f"Post-merge scenario column '{col}' contains NaNs. Grouping key mismatch?"
            )

        # we assume that mass market actor will not even consider an option which has not had a previous uptake of at least threshold uptake
        # wind is not subject to this effect
        df_vessels_working['engine_for_peer_influence'] = build_engine_for_peer_influence(
            df_vessels_working['engine'],
            df_vessels_working['ccs'],
        )
        df_vessel_scenarios['engine_for_peer_influence'] = build_engine_for_peer_influence(
            df_vessel_scenarios['engine'],
            df_vessel_scenarios['ccs'],
        )
        df_vessel_scenarios = classify_as_early_scrap(
            df_vessel_scenarios,
            int(user_params['year_step']),
        )

        engine_uptake_prev_year = df_vessels_working.groupby('engine_for_peer_influence').dwt.count()/ df_vessels_working.dwt.count()
        engine_uptake_prev_year = engine_uptake_prev_year.reset_index()
        newbuild_option_for_mass_actors = engine_uptake_prev_year[engine_uptake_prev_year.dwt >= user_params['threshold_uptake_newbuild']].engine_for_peer_influence.unique()
        retrofit_option_for_mass_actors = engine_uptake_prev_year[engine_uptake_prev_year.dwt >= user_params['threshold_uptake_retrofit']].engine_for_peer_influence.unique()

        if game_year > start_year:
            df_vessel_scenarios = df_vessel_scenarios[
                (df_vessel_scenarios.archetype == "early") # early movers can do whatever they want
                | (df_vessel_scenarios.action == "keep") # everyone is always free to keep its design
                | (
                    (df_vessel_scenarios.engine_for_peer_influence.isin(newbuild_option_for_mass_actors)) # for pure vessel replaement, the threshold of uptake is much lower
                    &(df_vessel_scenarios.action.str.contains("newbuild"))
                    &(~df_vessel_scenarios.is_early_scrap)
                )
                | (
                    (df_vessel_scenarios.engine_for_peer_influence.isin(retrofit_option_for_mass_actors)) # high threshold of uptake for retrofit and early scrapping
                    &(df_vessel_scenarios.action == "retrofit")
                )
                | (
                    (df_vessel_scenarios.engine_for_peer_influence.isin(retrofit_option_for_mass_actors)) # high threshold of uptake for retrofit and early scrapping
                    &(df_vessel_scenarios.action == "newbuild")
                    &(df_vessel_scenarios.is_early_scrap)
                )
            ]
            mask_mass_unproven_early_scrap = (
                (df_vessel_scenarios['archetype'] != 'early')
                & (df_vessel_scenarios['action'] == 'newbuild')
                & (df_vessel_scenarios['is_early_scrap'])
                & (~df_vessel_scenarios['engine_for_peer_influence'].isin(retrofit_option_for_mass_actors))
            )
            assert not mask_mass_unproven_early_scrap.any(), (
                "Mass-market ship has early-scrap newbuild scenario with unproven target option. "
                f"rows={int(mask_mass_unproven_early_scrap.sum())}, "
                f"threshold_uptake_retrofit={user_params['threshold_uptake_retrofit']}"
            )
        missing_ship_year_ids = [x for x in df_active.ship_year_id.unique() if x not in df_vessel_scenarios.ship_year_id.unique()]
        assert len(missing_ship_year_ids)==0, f"Missing ships: {df_active[df_active.ship_year_id.isin(missing_ship_year_ids)].sample(3)}"
        df_vessel_scenarios = technology.update_engine_type(df_vessel_scenarios)    # update LNG engine type depending on engine strokes

        # 1b. Evaluate DCF over the planning horizon for every (ship x action) scenario
        df_dcf_to_add, df_rrc_fuel_to_add = compute_scenario_dcf(
            df_vessel_scenarios, game_year, planning_horizon_years,
            user_params, engines_design_mapping, engine_fuels_dict, df_drop_in_fuels_ref, df_energy_density_ref,
            df_imo_targets_data, df_fueleu_targets_data,
            df_wtt_carbon_intensity_ref, df_ttw_carbon_intensity_ref,
            df_reward_ref, df_tc_rates_ref, df_pilot_fuel_input_ref,
            ru_price_data, initial_su_price_data, fuel_price_data, df_capex_ref, df_retrofit_cost_ref,
            df_opex_conventional_ref, df_opex_nonconv_wages_ref, df_opex_nonconv_insurance_ref,
            df_opex_nonconv_ssf_ref, df_opex_nonconv_isf_ref, df_opex_nonconv_spares_ref,
            df_operational_data_ref,
            df_speed_ref=df_speed_ref, df_efficiency_ref=df_efficiency_ref,
            df_engine_efficiency_ref=df_engine_efficiency_ref,
        )
        df_dcf_to_add['game_year'] = game_year
        df_rrc_fuel_to_add['game_year'] = game_year
        df_rrc_fuel.append(df_rrc_fuel_to_add)

        # find the initial scaling factor for peer influence: this is costs = -DCF of conventional
        # ships at start_year only. It is NOT recomputed in later years - the same baseline is
        # merged in every year so the scaling factor stays constant over the run.
        if game_year == start_year:
            df_conventional = df_dcf_to_add[
                (df_dcf_to_add.engine == "conventional")
                # &(df_dcf_to_add.stranded_value < df_dcf_to_add.dcf_planning_horizon * 0.01)
            ]
            youngest_idx = (
                df_conventional
                .groupby(['imo_type', 'size_id'])['vessel_age']
                .idxmin()
            )
            # the scaling factor is the costs of conventional, but we need to ignore potential stranded value if the ship was scrapped early
            # as we are only interested in annualised costs
            scaling_factor_peer_influence = (
                df_conventional.loc[youngest_idx, ['imo_type', 'size_id', 'dcf_planning_horizon']]
                .assign(scaling_factor_peer_influence=lambda d: -d['dcf_planning_horizon'] - df_dcf_to_add['stranded_value'])
                .drop(columns='dcf_planning_horizon')
                .reset_index(drop=True)
            )

        # merge the (constant) baseline into every year's DCF rows, keyed by ship_name since
        # ship_year_id is regenerated every game_year and cannot be used to carry values forward
        check_len = len(df_dcf_to_add)
        df_dcf_to_add = df_dcf_to_add.merge(
            scaling_factor_peer_influence,
            on=['imo_type','size_id'],
            how='left',
        )
        assert check_len == len(df_dcf_to_add), f"Post merging length does not match {check_len , len(df_dcf_to_add)}"
        assert len(df_dcf_to_add[df_dcf_to_add.scaling_factor_peer_influence.isna()]) == 0, f"NA in the min DCF: {df_dcf_to_add[df_dcf_to_add.scaling_factor_peer_influence.isna()].sample(3)}"

        df_dcf_to_add['utility'] = df_dcf_to_add['dcf_planning_horizon']
        df_dcf_to_add['peer_influence'] = 0.0
        df_dcf_to_add['game_year'] = game_year

        # 1c. Seed initial choices, then iterate with peer influence until convergence
        df_dcf_to_add['engine_for_peer_influence'] = build_engine_for_peer_influence(
            df_dcf_to_add['engine'],
            df_dcf_to_add['ccs'],
        )
        if game_year == start_year:
            df_year_results = (df_dcf_to_add.groupby('ship_name').apply(choose_action)
                               .reset_index(drop=True))
            previous_choices = {
                row['ship_name']: (row['choice_id'], row['engine_for_peer_influence'], row['imo_type'])
                for _, row in df_year_results.iterrows()
            }
        else:
            pass    # use choices from last year.

        # 1d. Iterative Nash procedure – peer influence feedback loop
        for iteration in range(max_iterations):
            if (df_dcf_to_add['engine_for_peer_influence'].isna() | (df_dcf_to_add['engine_for_peer_influence'] == "")).any():
                raise AssertionError("Engine type is empty")

            previous_choices_df = pd.DataFrame([
                {
                    'ship_name': ship_name,
                    'chosen_engine': choice[1],      # Engine from (choice_id, engine, imo_type) tuple
                    'chosen_imo_type': choice[2],    # IMO type from the choice tuple
                }
                for ship_name, choice in previous_choices.items()
            ])

            n_ships = len(previous_choices_df)
            total_other_ships = max(n_ships - 1, 1)  # Total other ships (excluding self in adoption rate)

            engine_counts = previous_choices_df['chosen_engine'].value_counts()

            # Counts how many peers chose each engine across all scenarios via column mapping
            df_dcf_to_add['same_engine_total'] = df_dcf_to_add['engine_for_peer_influence'].map(engine_counts).fillna(0).astype(float)

            # Build lookup tables for each ship's current choice (engine and IMO type)
            # These are used for self-exclusion: we don't count a ship's influence on itself
            chosen_engine_by_ship = previous_choices_df.set_index('ship_name')['chosen_engine']

            # Map each row's ship back to its previously chosen engine/type
            # Allows us to exclude self from peer influence calculations
            df_dcf_to_add['self_engine'] = df_dcf_to_add['ship_name'].map(chosen_engine_by_ship)

            # Self-exclusion via element-wise comparison
            # Creates boolean masks (0.0 or 1.0) for whether this row is the same as self choice
            df_dcf_to_add['self_same_engine'] = (df_dcf_to_add['self_engine'] == df_dcf_to_add['engine_for_peer_influence']).astype(float)

            # Subtract self from total counts to get peer peer_influence (no self-influence)
            # This ensures each ship counts adoption among *other* ships, not including itself
            df_dcf_to_add['same_engine_count'] = df_dcf_to_add['same_engine_total'] - df_dcf_to_add['self_same_engine']

            #  Compute adoption rate as proportion of other ships choosing same engine
            # Ranges from 0 (no one else chose this engine) to 1 (all others chose it)
            df_dcf_to_add['adoption_rate_engine'] = df_dcf_to_add['same_engine_count'] / total_other_ships

            # the scale of the peer influence is either model as a linear or logistic function of technology uptake
            # Apply logistic curve to convert adoption rate to peer influence factors
            # Logistic curve creates S-shaped incentive: slow at low adoption, steep near critical mass, flat at high
            assert SHAPE_PEER_INFLUENCE in ['logistics','linear'], f"{SHAPE_PEER_INFLUENCE} not an valid peer influence shape"
            if SHAPE_PEER_INFLUENCE == "logistics":
                df_dcf_to_add['peer_influence_term']= ENGINE_PEER_FACTOR / (1.0 + np.exp(-slope_engine * (df_dcf_to_add['adoption_rate_engine'] - critical_mass_engine)))
            else:
                df_dcf_to_add['peer_influence_term'] = ENGINE_PEER_FACTOR * df_dcf_to_add['adoption_rate_engine']

            # Apply archetype-specific factors to peer influence
            # Early adopters get 0x bonus (no peer influence), mass market 1x
            unexpected_archetypes = df_dcf_to_add.loc[~df_dcf_to_add['archetype'].isin(['early', 'mass']), 'archetype'].unique()
            assert len(unexpected_archetypes) == 0, f"Unexpected archetype values: {unexpected_archetypes}"
            df_dcf_to_add['archetype_factor'] = np.select(
                [df_dcf_to_add['archetype'] == 'early', df_dcf_to_add['archetype'] == 'mass'],
                [0.0, 1]            )
            df_dcf_to_add['peer_influence'] = (
                df_dcf_to_add['peer_influence_term'] 
                * df_dcf_to_add['archetype_factor'] 
                * df_dcf_to_add['scaling_factor_peer_influence'] 
            )

            # Compute the utility as a function of net savings and peer influence
            df_dcf_to_add['utility'] = (
                df_dcf_to_add['dcf_planning_horizon'] 
                + df_dcf_to_add['peer_influence']
            )

            # Choose max utility scenario per ship
            df_year_results_new = (
                df_dcf_to_add.groupby('ship_name')
                .apply(choose_action)
                .reset_index(drop=True)
            )

            # ** Track peer influence diagnostics
            df_peer_track = df_dcf_to_add.copy()
            df_peer_track['iteration'] = iteration
            df_peer_track['game_year'] = game_year

            # mark which design is chosen in this iteration
            df_peer_track['design_chosen'] = False
            chosen_ids_peer = df_year_results_new['dcf_id'].unique()
            mask_iter = df_peer_track['dcf_id'].isin(chosen_ids_peer)
            df_peer_track.loc[mask_iter, 'design_chosen'] = True

            # keep only useful columns for peer influence tracking
            peer_tracking_cols= [
                'game_year',
                'iteration',
                'ship_name',
                'action',
                'engine',
                'engine_for_peer_influence',
                'imo_type',
                'ship_year_id',
                'dcf_id',
                'same_engine_total', 'self_engine', 'self_same_engine', 'same_engine_count', 'adoption_rate_engine',
                'peer_influence_term' ,
                'archetype', 'archetype_factor','scaling_factor_peer_influence',
                'peer_influence', 
                'utility',
                'dcf_planning_horizon',
                'design_chosen',
                # 'chosen_iteraion'
            ]
            df_peer_tracking.append(df_peer_track[peer_tracking_cols])

            # Check convergence
            new_choices = {
                row['ship_name']: (row['choice_id'], row['engine_for_peer_influence'], row['imo_type']) for _, row in df_year_results_new.iterrows()} # row['fuel']
            if new_choices == previous_choices:
                break
            previous_choices = new_choices.copy()

        df_year_active = df_year_results_new.copy()

        # flag which ship designs have been chosen
        df_dcf_to_add['design_chosen'] = False
        df_dcf_to_add['su_chosen'] = False
        df_dcf_to_add['peer_influence_iter_chosen'] = False
        chosen_pairs = set(zip(df_year_active['ship_year_id'], df_year_active['dcf_id']))
        mask_chosen = df_dcf_to_add.apply(
            lambda row: (row['ship_year_id'], row['dcf_id']) in chosen_pairs,
            axis=1,
        )
        df_dcf_to_add.loc[mask_chosen,'design_chosen'] = True
        df_dcf = pd.concat([df_dcf, df_dcf_to_add], ignore_index=True)

        df_year_active = df_year_active.rename(columns={'year': 'planning_start_year'})
        df_year_active['vessel_age'] = game_year - df_year_active['year_built']

        # ===================================================================
        # PHASE 2 – OPERATING DECISION (fuel choice for the current year)
        # The investment decision above fixed each ship's engine/design.
        # Now, given that design, choose the cost-minimising fuel mix for
        # the current game year.  When the SU market is active (regulation
        # target strictly between 0 and 100), fuel choice is determined
        # through the SU market clearing loop (iterate_su_market).
        # Otherwise, fuels are chosen directly via choose_fuel_given_su
        # with su_price = 0 and fuel-group availability constraints.
        # ===================================================================
        tgt = df_year_active['direct_compliance_target'].iloc[0]
        run_su = su_market_flag and 0 < tgt < 100

        if run_su:
            df_year_active, su_price, df_dcf_to_add, df_rrc_fuel_to_add, prev_year_fuel_group_avail = iterate_su_market(
                df_year_active, user_params, game_year, engines_design_mapping, engine_fuels_dict,
                df_drop_in_fuels_ref, df_energy_density_ref,
                df_imo_targets_data, df_fueleu_targets_data,
                df_wtt_carbon_intensity_ref, df_ttw_carbon_intensity_ref,
                df_reward_ref, df_tc_rates_ref, df_pilot_fuel_input_ref,
                ru_price_data, initial_su_price_data, fuel_price_data, df_growth_infra, 
                df_capex_ref, df_retrofit_cost_ref,
                df_opex_conventional_ref, df_opex_nonconv_wages_ref,
                df_opex_nonconv_insurance_ref,
                df_opex_nonconv_ssf_ref, df_opex_nonconv_isf_ref,
                df_opex_nonconv_spares_ref, df_operational_data_ref,
                max_iterations, 
                prev_year_fuel_group_avail,
                df_speed_ref=df_speed_ref, df_efficiency_ref=df_efficiency_ref,
                df_engine_efficiency_ref=df_engine_efficiency_ref,
            )
            df_year_active['su_price'] = su_price

            # add the information recorded, for future plotting / debugging
            df_dcf = pd.concat((df_dcf, df_dcf_to_add), ignore_index = True)
            df_rrc_fuel.append(df_rrc_fuel_to_add)
        else:
            # Enforce fuel availability without SU market
            year_step = int(user_params['year_step'])
            fuel_group_avail = build_yearly_fuel_group_caps(prev_year_fuel_group_avail, df_growth_infra, year_step)
            df_year_active, df_rrc_fuel_to_add = choose_fuel_given_su(
                df_year_active, 0.0, user_params,
                engines_design_mapping, engine_fuels_dict,
                df_drop_in_fuels_ref, df_energy_density_ref,
                df_imo_targets_data, df_fueleu_targets_data,
                df_wtt_carbon_intensity_ref, df_ttw_carbon_intensity_ref,
                df_reward_ref, df_tc_rates_ref, df_pilot_fuel_input_ref,
                ru_price_data, initial_su_price_data, fuel_price_data,
                df_capex_ref, df_retrofit_cost_ref,
                df_opex_conventional_ref, df_opex_nonconv_wages_ref,
                df_opex_nonconv_insurance_ref,
                df_opex_nonconv_ssf_ref, df_opex_nonconv_isf_ref,
                df_opex_nonconv_spares_ref, df_operational_data_ref,
                game_year, fuel_group_avail, max_iterations,
                df_speed_ref=df_speed_ref, df_efficiency_ref=df_efficiency_ref,
                df_engine_efficiency_ref=df_engine_efficiency_ref,
            )
            df_year_active['su_loop_iteration'] = -1
            df_rrc_fuel_to_add['su_loop_iteration'] = -1
            df_rrc_fuel_to_add['game_year'] = game_year
            df_year_active['switched_to_balance_the_su_market'] = False
            df_dcf = pd.concat((df_dcf, df_year_active), ignore_index=True)
            df_rrc_fuel.append(df_rrc_fuel_to_add)
            prev_year_fuel_group_avail = update_prev_year_fuel_group_avail(df_year_active, prev_year_fuel_group_avail, fuel_group_avail)

        # --- Append to results and update working dataframe ---
        df_year_active.loc[:, 'game_year'] = game_year
        df_year_active['vessel_age'] = game_year - df_year_active['year_built']
        game_years_results.append(df_year_active)
        if len(demand_scrapped_this_year) > 0:
            demand_scrapped_this_year['game_year'] = game_year
            game_years_results.append(demand_scrapped_this_year)
        df_vessels_working = df_year_active.copy()    # get latest simulation year results
        assert length_fleet == len(df_vessels_working), f"Unexpected number of vessels for game year {game_year}, scenario {user_params['scenario_number']}: {len(df_vessels_working)}"

    # --- Combine all years ---
    df_game_results = pd.concat(game_years_results, ignore_index=True)
    df_game_results = df_game_results.sort_values(by=['ship_name', 'game_year'])
    df_rrc_fuel = pd.concat(df_rrc_fuel, ignore_index=True)
    df_peer_tracking = pd.concat(df_peer_tracking, ignore_index=True)

    return df_game_results, df_rrc_fuel, df_dcf, df_peer_tracking

def preprocess_reference_data(*, user_params, start_year, end_year, planning_horizon_years,
                              fuel_price_data, df_growth_infra, df_imo_targets_data,
                              df_fueleu_targets_data, ru_price_data, initial_su_price_data,
                              df_wtt_carbon_intensity_ref, df_reward_ref,
                              df_opex_conventional_ref, df_opex_nonconv_wages_ref, df_opex_nonconv_insurance_ref,
                              df_opex_nonconv_ssf_ref, df_opex_nonconv_isf_ref, df_opex_nonconv_spares_ref,
                              df_fleet_growth,
                              df_speed_ref, df_efficiency_ref,
                              ):
    """
    Apply scenario filters and ensure all time series extend far enough
    for model execution.
    """

    max_year_needed = end_year + planning_horizon_years
    required_years = set(range(start_year, end_year + 1))

    def filter_scenario(df, name, scenario_key, check_years):
        scenario_value = user_params[scenario_key]
        filtered = df[df["scenario"] == scenario_value].copy()
        assert len(filtered) > 0, (
            f"{name} is empty after filtering on scenario "
            f"'{scenario_value}' (user_params['{scenario_key}']) — "
            f"scenario not found in Model_Database."
        )
        if check_years:
            missing_years = required_years - set(filtered['year'].unique())
            assert not missing_years, (
                f"{name} (scenario '{scenario_value}') is missing years "
                f"{sorted(missing_years)} needed to cover start_year={start_year} to end_year={end_year}."
            )
        return filtered

    # def build_reward_ref_by_group(df, name):
    #     """Filter df_reward_ref independently per FUEL_GROUPS_AVAIL fuel_group, using each
    #     group's own reward_scenario_<group> user setting, and stack the results into a
    #     long-form table (year, fuel_group, value) for per-group lookups in fuel.py."""
    #     frames = []
    #     for fuel_group in FUEL_GROUPS_AVAIL['fuel_group'].unique():
    #         scenario_key = f'reward_scenario_{fuel_group}'
    #         assert scenario_key in user_params, f"user_params is missing '{scenario_key}'"
    #         filtered = filter_scenario(df, f"{name} (fuel_group '{fuel_group}')", scenario_key, False)
    #         filtered = filtered.drop_duplicates(subset=['year'])
    #         filtered['fuel_group'] = fuel_group
    #         frames.append(filtered)
    #     return pd.concat(frames, ignore_index=True)

    # --- Apply scenario filters ---
    # (dataframe, name, user_params scenario key, whether annual coverage must be checked)
    scenario_filters = [
        (df_imo_targets_data, "df_imo_targets_data", "imo_scenario", True),
        (df_fueleu_targets_data, "df_fueleu_targets_data", "fueleu_scenario", True),
        (ru_price_data, "ru_price_data", "ru_scenario", True),
        (initial_su_price_data, "initial_su_price_data", "initial_su_price_scenario", True),
        (df_wtt_carbon_intensity_ref, "df_wtt_carbon_intensity_ref", "wtt_emissions_factor_scenario", True),
        (df_reward_ref, "df_reward_ref", "reward_scenario", False),
        (df_opex_conventional_ref, "df_opex_conventional_ref", "capex_opex_scenario", False),
        (df_opex_nonconv_wages_ref, "df_opex_nonconv_wages_ref", "capex_opex_scenario", False),
        (df_opex_nonconv_insurance_ref, "df_opex_nonconv_insurance_ref", "capex_opex_scenario", False),
        (df_opex_nonconv_ssf_ref, "df_opex_nonconv_ssf_ref", "capex_opex_scenario", False),
        (df_opex_nonconv_isf_ref, "df_opex_nonconv_isf_ref", "capex_opex_scenario", False),
        (df_opex_nonconv_spares_ref, "df_opex_nonconv_spares_ref", "capex_opex_scenario", False),
        (df_growth_infra, "df_growth_infra", "growth_infra", False),
        (df_fleet_growth, "df_fleet_growth", "fleet_growth", True),
        (df_speed_ref, "df_speed_ref", "speed_reduction", True),
        (df_efficiency_ref, "df_efficiency_ref", "energy_efficiency_impr", True),
    ]
    (df_imo_targets_data, df_fueleu_targets_data, ru_price_data, initial_su_price_data,
     df_wtt_carbon_intensity_ref, df_reward_ref, df_opex_conventional_ref, df_opex_nonconv_wages_ref,
     df_opex_nonconv_insurance_ref, df_opex_nonconv_ssf_ref, df_opex_nonconv_isf_ref,
     df_opex_nonconv_spares_ref, df_growth_infra, df_fleet_growth, df_speed_ref, df_efficiency_ref) = (
        filter_scenario(df, name, key, check_years) for df, name, key, check_years in scenario_filters
    )
    # df_reward_ref = build_reward_ref_by_group(df_reward_ref, "df_reward_ref")

    # Set fuel price data for bio non bio (different scenarios)
    bio_cols = [c for c in fuel_price_data.columns if "bio-" in c]     # Identify bio fuel columns
    df_fuels_prices_sub = filter_scenario(fuel_price_data, "df_fuels_prices_sub", "fuel_price_scenario", True)
    df_biofuels_prices = filter_scenario(fuel_price_data, "df_biofuels_prices", "biofuel_price_scenario", True)

    df_fuels_prices_sub = df_fuels_prices_sub.sort_values("year").reset_index(drop=True)    # Ensure alignment by year
    df_biofuels_prices = df_biofuels_prices.sort_values("year").reset_index(drop=True)

    df_fuels_prices_sub[bio_cols] = df_biofuels_prices[bio_cols]     # Replace bio columns
    df_fuels_prices_sub["bio_scenario"] = user_params["biofuel_price_scenario"]

    # --- Extend time series where needed ---
    fuel_price_data = util.extend_years(df_fuels_prices_sub, max_year_needed)
    ru_price_data = util.extend_years(ru_price_data, max_year_needed)
    initial_su_price_data = util.extend_years(initial_su_price_data, max_year_needed)
    df_imo_targets_data = util.extend_years(df_imo_targets_data, max_year_needed)
    df_fueleu_targets_data = util.extend_years(df_fueleu_targets_data, max_year_needed)
    df_reward_ref = util.extend_years(df_reward_ref, max_year_needed)
    df_wtt_carbon_intensity_ref = util.extend_years(df_wtt_carbon_intensity_ref, max_year_needed)

    return {
        "fuel_price_data": fuel_price_data,
        "df_imo_targets_data": df_imo_targets_data,
        "df_fueleu_targets_data": df_fueleu_targets_data,
        "ru_price_data": ru_price_data,
        "initial_su_price_data": initial_su_price_data,
        "df_wtt_carbon_intensity_ref": df_wtt_carbon_intensity_ref,
        "df_reward_ref": df_reward_ref,
        "df_opex_conventional_ref": df_opex_conventional_ref,
        "df_opex_nonconv_wages_ref": df_opex_nonconv_wages_ref,
        "df_opex_nonconv_insurance_ref": df_opex_nonconv_insurance_ref,
        "df_opex_nonconv_ssf_ref": df_opex_nonconv_ssf_ref,
        "df_opex_nonconv_isf_ref": df_opex_nonconv_isf_ref,
        "df_opex_nonconv_spares_ref": df_opex_nonconv_spares_ref,
        "df_growth_infra": df_growth_infra,
        "df_fleet_growth": df_fleet_growth,
        "df_speed_ref": df_speed_ref,
        "df_efficiency_ref": df_efficiency_ref,
    }

def main(
        df_vessels, start_year, end_year, planning_horizon_years,su_market_flag,
        user_params, engines_design_mapping, engine_fuels_dict, df_drop_in_fuels_ref, df_energy_density_ref,
        df_imo_targets_data, df_fueleu_targets_data, ru_price_data, initial_su_price_data, fuel_price_data,
        df_growth_infra, df_capex_ref, df_retrofit_cost_ref,
        df_opex_conventional_ref, df_opex_nonconv_wages_ref, df_opex_nonconv_insurance_ref,
        df_opex_nonconv_ssf_ref, df_opex_nonconv_isf_ref, df_opex_nonconv_spares_ref,
        df_operational_data_ref, df_wtt_carbon_intensity_ref, df_ttw_carbon_intensity_ref, 
        df_reward_ref, df_tc_rates_ref, df_pilot_fuel_input_ref,
        ENGINE_PEER_FACTOR, critical_mass_engine, slope_engine,
        max_iterations, RUN_OUTPUT_DIR, round_nb,
        df_fleet_growth=None, prev_year_fuel_group_avail=None, *,
        df_speed_ref, df_efficiency_ref, df_engine_efficiency_ref, SHAPE_PEER_INFLUENCE
        ):

    # --- record model start time ---
    model_start_time = datetime.now()  # model start
    # make output directory available for debugging/logging routines
    if RUN_OUTPUT_DIR is not None:
        user_params['RUN_OUTPUT_DIR'] = str(RUN_OUTPUT_DIR)
        # if DEBUG is enabled, create a unique log file for this run
        if bool(user_params.get('DEBUG', False)):
            dbg_name = f"debug_log_{model_start_time.strftime('%Y%m%d_%H%M%S')}.txt"
            user_params['DEBUG_LOG_PATH'] = str(Path(user_params['RUN_OUTPUT_DIR']) / dbg_name)

    # --- Apply scenario filter, fill data beyond 2050 if required ---
    ref_data = preprocess_reference_data(
        user_params=user_params, start_year=start_year, end_year=end_year, planning_horizon_years=planning_horizon_years,
        fuel_price_data=fuel_price_data, df_growth_infra=df_growth_infra, df_imo_targets_data=df_imo_targets_data,
        df_fueleu_targets_data=df_fueleu_targets_data, ru_price_data=ru_price_data,initial_su_price_data = initial_su_price_data,
        df_wtt_carbon_intensity_ref=df_wtt_carbon_intensity_ref, df_reward_ref=df_reward_ref,
        df_opex_conventional_ref=df_opex_conventional_ref,
        df_opex_nonconv_wages_ref=df_opex_nonconv_wages_ref,
        df_opex_nonconv_insurance_ref=df_opex_nonconv_insurance_ref,
        df_opex_nonconv_ssf_ref=df_opex_nonconv_ssf_ref,
        df_opex_nonconv_isf_ref=df_opex_nonconv_isf_ref,
        df_opex_nonconv_spares_ref=df_opex_nonconv_spares_ref, 
        df_fleet_growth=df_fleet_growth,
        df_speed_ref=df_speed_ref, df_efficiency_ref=df_efficiency_ref,
    )

    fuel_price_data = ref_data["fuel_price_data"]
    df_imo_targets_data = ref_data["df_imo_targets_data"]
    df_fueleu_targets_data = ref_data["df_fueleu_targets_data"]
    ru_price_data = ref_data["ru_price_data"]
    initial_su_price_data = ref_data["initial_su_price_data"]
    df_wtt_carbon_intensity_ref = ref_data["df_wtt_carbon_intensity_ref"]
    df_reward_ref = ref_data["df_reward_ref"]
    df_opex_conventional_ref = ref_data["df_opex_conventional_ref"]
    df_opex_nonconv_wages_ref = ref_data["df_opex_nonconv_wages_ref"]
    df_opex_nonconv_insurance_ref = ref_data["df_opex_nonconv_insurance_ref"]
    df_opex_nonconv_ssf_ref = ref_data["df_opex_nonconv_ssf_ref"]
    df_opex_nonconv_isf_ref = ref_data["df_opex_nonconv_isf_ref"]
    df_opex_nonconv_spares_ref = ref_data["df_opex_nonconv_spares_ref"]
    df_growth_infra = ref_data["df_growth_infra"]
    df_fleet_growth = ref_data["df_fleet_growth"]
    df_speed_ref = ref_data["df_speed_ref"]
    df_efficiency_ref = ref_data["df_efficiency_ref"]

    # --- PLAY GAME ---
    # INITIALIZE AND RESET

    # --- Build prev_year_fuel_group_avail from fleet total consumption ---
    # prev_year_fuel_group_avail arrives with share_fuel_cons (from initial_fuel_cons sheet).
    # Scale shares to absolute MJ using the modelled fleet's total consumption.
    assert prev_year_fuel_group_avail is not None and 'share_fuel_cons' in prev_year_fuel_group_avail.columns
    df_fc = (
        df_operational_data_ref[['imo_type', 'size_id', 'tj_per_ship_pa']]
        .drop_duplicates(subset=['imo_type', 'size_id'])
    )
    df_vessels_fc = df_vessels.merge(df_fc, on=['imo_type', 'size_id'], how='left')
    total_fleet_consumption_mj = (df_vessels_fc['tj_per_ship_pa'] * df_vessels_fc['real_ships_nb'] * 1_000_000).sum()
    assert total_fleet_consumption_mj > 0, (
        f"Total fleet consumption is zero/negative: {total_fleet_consumption_mj}"
    )
    prev_year_fuel_group_avail = prev_year_fuel_group_avail.copy()
    prev_year_fuel_group_avail['fuel_consumption_fleet_mj_year'] = (
        prev_year_fuel_group_avail['share_fuel_cons'] * total_fleet_consumption_mj
    )

    df_nash_results = None
    df_dcf_nash = None
    df_rrc_fuel_nash = None
    df_nash_peer_tracking = None

    # Ensure no leftover SU price
    if 'su_price' in user_params:
        user_params['su_price'] = None

    # Run myopic Nash
    nash_user_params = copy.deepcopy(user_params)
    nash_vessels = copy.deepcopy(df_vessels)

    df_nash_results, df_rrc_fuel_nash, df_dcf_nash, df_nash_peer_tracking = run_nash_equilibrium_iterative(
        df_vessels=nash_vessels,
        start_year=start_year,
        end_year=end_year,
        planning_horizon_years=planning_horizon_years,
        user_params=nash_user_params,
        engines_design_mapping=engines_design_mapping,
        engine_fuels_dict=engine_fuels_dict,
        df_drop_in_fuels_ref=df_drop_in_fuels_ref,
        df_energy_density_ref = df_energy_density_ref,
        df_imo_targets_data=df_imo_targets_data,
        df_fueleu_targets_data=df_fueleu_targets_data,
        df_wtt_carbon_intensity_ref=df_wtt_carbon_intensity_ref,
        df_ttw_carbon_intensity_ref=df_ttw_carbon_intensity_ref,
        df_reward_ref=df_reward_ref,
        df_tc_rates_ref=df_tc_rates_ref,
        df_pilot_fuel_input_ref=df_pilot_fuel_input_ref,
        ru_price_data=ru_price_data,
        initial_su_price_data = initial_su_price_data,
        su_market_flag = su_market_flag,
        fuel_price_data=fuel_price_data,
        df_growth_infra = df_growth_infra, 
        df_capex_ref=df_capex_ref,
        df_retrofit_cost_ref=df_retrofit_cost_ref,
        df_opex_conventional_ref=df_opex_conventional_ref,
        df_opex_nonconv_wages_ref=df_opex_nonconv_wages_ref,
        df_opex_nonconv_insurance_ref=df_opex_nonconv_insurance_ref,
        df_opex_nonconv_ssf_ref=df_opex_nonconv_ssf_ref,
        df_opex_nonconv_isf_ref=df_opex_nonconv_isf_ref,
        df_opex_nonconv_spares_ref=df_opex_nonconv_spares_ref,
        df_operational_data_ref=df_operational_data_ref,
        ENGINE_PEER_FACTOR=ENGINE_PEER_FACTOR,
        SHAPE_PEER_INFLUENCE=SHAPE_PEER_INFLUENCE,
        critical_mass_engine=critical_mass_engine,
        slope_engine=slope_engine,
        max_iterations=max_iterations,
        prev_year_fuel_group_avail=copy.deepcopy(prev_year_fuel_group_avail),
        df_fleet_growth=df_fleet_growth,
        df_speed_ref=df_speed_ref, df_efficiency_ref=df_efficiency_ref,
        df_engine_efficiency_ref=df_engine_efficiency_ref,
    )
    df_nash_results = reclassify_action(df_nash_results, int(user_params['year_step']))
    df_nash_results['scenario_name'] = user_params['scenario_name']

    # Record model end time
    model_end_time = datetime.now()  # model end

    # --- SAVE RESULTS ---
    # Folder named by model start date (DDMMYY)
    RUN_OUTPUT_DIR = Path(RUN_OUTPUT_DIR)
    run_round_folder = RUN_OUTPUT_DIR / f"round_{round_nb}"
    run_round_folder.mkdir(parents=True, exist_ok=True)

    # Save CSVs inside a zip file in that folder
    meta_path = run_round_folder / "run_meta.json"
    user_params_path = run_round_folder / "user_params.json"
    zip_name = f"fleet_model_results.zip"
    results_zip_path = run_round_folder / zip_name

    df_results_nash_path = None

    # --- Save CSVs ---
    csv_outputs = {
        "df_results_nash.csv": df_nash_results,
        "df_rcc_fuel_nash.csv": df_rrc_fuel_nash,
        "df_dcf_nash.csv": df_dcf_nash,
        "df_nash_peer_tracking.csv": df_nash_peer_tracking,
    }
    saved_paths = {}

    for filename, df in csv_outputs.items():
        if df is not None:
            path = run_round_folder / filename
            df.to_csv(path, index=False)
            saved_paths[filename] = path

    # --- Save JSONs ---
    meta_data = {
        "model_start_time": model_start_time.isoformat(),
        "model_end_time": model_end_time.isoformat(),
        "num_players": len(df_vessels),
        "duration_seconds": (model_end_time - model_start_time).total_seconds(),
        "outputs": {
            "zip": results_zip_path.name,
            "df_results_nash": saved_paths.get("df_results_nash.csv", None).name if "df_results_nash.csv" in saved_paths else None,
            "df_rrc_fuel_nash": saved_paths.get("df_rcc_fuel_nash.csv", None).name if "df_rcc_fuel_nash.csv" in saved_paths else None,
            "df_dcf_nash": saved_paths.get("df_dcf_nash.csv", None).name if "df_dcf_nash.csv" in saved_paths else None,
            "df_nash_peer_tracking": saved_paths.get("df_nash_peer_tracking.csv", None).name if "df_nash_peer_tracking.csv" in saved_paths else None,
            "meta_json": meta_path.name,
            "user_params_json": user_params_path.name,
        }
    }

    meta_path.write_text(json.dumps(meta_data, indent=2, default=str))
    user_params_path.write_text(json.dumps(user_params, indent=2, default=str))

    # --- Create ZIP including CSVs + JSONs ---
    files_to_zip = list(saved_paths.values()) + [meta_path, user_params_path]

    with zipfile.ZipFile(results_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        for file_path in files_to_zip:
            if file_path.exists():
                zipf.write(file_path, arcname=file_path.name)

    # --- Remove CSVs and JSONs after adding to ZIP ---
    for file_path in files_to_zip:
        if file_path is not None and file_path.exists():
            file_path.unlink()

    return {
        "output_dir": run_round_folder,
        "zip_path": results_zip_path,
        "nash_csv_path": df_results_nash_path,
        "df_dcf_nash": df_dcf_nash,
        "df_nash_results": df_nash_results,
        "df_rrc_fuel_nash": df_rrc_fuel_nash,
        "df_nash_peer_tracking": df_nash_peer_tracking
    }