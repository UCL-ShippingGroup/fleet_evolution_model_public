import warnings
import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from collections import defaultdict
from itertools import combinations

warnings.filterwarnings('ignore', category=FutureWarning, message='Setting an item of incompatible dtype')
warnings.filterwarnings('ignore', category=PerformanceWarning, message='DataFrame is highly fragmented')

from default_assumptions import CCS_ADD_FUEL_CONS, WIND_ENERGY_SAVINGS, PILOT_FUELS, CARBON_CAPTURE_RATE #, FUEL_GROUPS_AVAIL
from policy import calculate_policy_compliance

def is_znz_eligible(reference_wtw_gco2eq_mj, reward_threshold_gco2_mj):
    """Single source of truth for the ZNZ (zero/near-zero) reward-eligibility gate:
    whether a fuel earns the flat per-year reward (build_all_fuel_options)."""
    return reference_wtw_gco2eq_mj < reward_threshold_gco2_mj

def build_all_fuel_options(
    df_params,
    engine_fuels_dict,
    fuel_price_data,
    df_wtt_carbon_intensity_ref,
    df_ttw_carbon_intensity_ref,
    df_reward_ref,
    df_pilot_fuel_input_ref,
    compliance_mode,
    allowed_fuels,
    reward_fuels,
    df_drop_in_fuels_ref=None,
    engines_design_mapping=None,
):
    """fuel selection (single-fuel + fuel-mix) for a batch of scenario rows.

    Pass df_drop_in_fuels_ref and engines_design_mapping to enable fuel-mix candidates.
    When either is None the function falls back to single-fuel only.
    """

    assert isinstance(df_params, pd.DataFrame), f"Expected DataFrame, got {type(df_params)}"
    assert len(df_params) > 0, "select_best_fuel received empty df_params"

    df_base = df_params.copy().reset_index(drop=True)
    assert 'tco_id' in df_base.columns, "select_best_fuel requires tco_id in df_params"
    assert df_base['tco_id'].notna().all(), "select_best_fuel received missing tco_id values"
    assert (df_base['tco_id'].astype(str).str.len() > 0).all(), "select_best_fuel received empty tco_id values"

    # Expand allowed fuels by base engine, then keep only user-allowed fuels.
    df_engine_fuels = pd.DataFrame(
        [(engine, fuel) for engine, fuels in engine_fuels_dict.items() for fuel in fuels],
        columns=['base_engine', 'fuel'],
    )
    df_engine_fuels = df_engine_fuels[df_engine_fuels.fuel.isin(allowed_fuels)]
    check_len = len(df_base)
    df_candidates = df_base.merge(df_engine_fuels, on='base_engine', how='left')
    assert len(df_candidates) >= check_len, "Merge dropped base rows"
    assert len(df_candidates[df_candidates.fuel.isna()]) == 0, f"Some ships do not have any drop-in fuels available: {df_candidates[df_candidates.fuel.isna()][['engine','base_engine']].drop_duplicates()}"

    # Add pilot-fuel combinations.
    # Only fuels listed in df_pilot_fuel_input_ref actually require a pilot (share_pilot_mj > 0).
    # For those, try every PILOT_FUELS entry so the cheapest pilot is selected.
    # For all other fuels the pilot contributes 0 consumption; a single placeholder is enough.
    needs_pilot_fuels = set(df_pilot_fuel_input_ref['fuel'].unique())
    mask_needs_pilot = df_candidates['fuel'].isin(needs_pilot_fuels)

    df_pilot_all = pd.DataFrame({'pilot_fuel': PILOT_FUELS, '__tmp_key': 1})
    df_pilot_one = pd.DataFrame({'pilot_fuel': [PILOT_FUELS[0]], '__tmp_key': 1})

    df_needs = df_candidates[mask_needs_pilot].copy()
    df_noneed = df_candidates[~mask_needs_pilot].copy()

    df_needs['__tmp_key'] = 1
    df_noneed['__tmp_key'] = 1

    df_needs = df_needs.merge(df_pilot_all, on='__tmp_key').drop(columns='__tmp_key')
    df_noneed = df_noneed.merge(df_pilot_one, on='__tmp_key').drop(columns='__tmp_key')

    df_candidates = pd.concat([df_needs, df_noneed], ignore_index=True)

    # Main and pilot fuel prices (long-form merge by year/scenario/fuel).
    id_vars = ['year']
    value_vars = [c for c in fuel_price_data.columns if c not in id_vars]
    df_price_long = fuel_price_data.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name='fuel',
        value_name='fuel_price_usd_tvslfoeq',
    )

    expected_rows = len(df_candidates)
    df_candidates = df_candidates.merge(
        df_price_long.rename(columns={'fuel_price_usd_tvslfoeq': 'main_fuel_price_usd_tvslfoeq'}),
        on=['year', 'fuel'],
        how='left',
    )
    assert len(df_candidates) == expected_rows, (
        f"Row mismatch after main fuel price merge: expected {expected_rows}, got {len(df_candidates)}"
    )

    expected_rows = len(df_candidates)
    df_candidates = df_candidates.merge(
        df_price_long.rename(columns={'fuel': 'pilot_fuel', 'fuel_price_usd_tvslfoeq': 'pilot_fuel_price_usd_tvslfoeq'}),
        on=['year', 'pilot_fuel'],
        how='left',
    )
    assert len(df_candidates) == expected_rows, (
        f"Row mismatch after pilot fuel price merge: expected {expected_rows}, got {len(df_candidates)}"
    )

    # Emission factors (WTT as long-form by year/scenario/fuel; TTW by fuel/engine_type).
    id_vars_wtt = ['year']
    fuel_cols_wtt = [c for c in df_wtt_carbon_intensity_ref.columns if c not in id_vars_wtt]
    df_wtt_long = df_wtt_carbon_intensity_ref.melt(
        id_vars=id_vars_wtt,
        value_vars=fuel_cols_wtt,
        var_name='fuel',
        value_name='wtt_gco2e_p_mj',
    )

    assert len(df_candidates[df_candidates['engine_type'].isna()]) == 0, f"Nan in the engines: {df_candidates[df_candidates['engine_type'].isna()]}"
    df_candidates['engine_type_with_label'] = np.where(
        df_candidates['engine_type'] == 'All',
        'All',
        df_candidates['engine_type'].astype(str) + ' ' + df_candidates['engine_type_label'].astype(str),
    )
    assert len(df_candidates[df_candidates['engine_type_with_label'].isna()]) == 0, f"Nan in the engines: {df_candidates[df_candidates['engine_type_with_label'].isna()]}"

    expected_rows = len(df_candidates)
    df_candidates = df_candidates.merge(
        df_wtt_long.rename(columns={'wtt_gco2e_p_mj': 'main_emissions_factor_wtt_gco2eq_mj'}),
        on=['year', 'fuel'],
        how='left',
    )
    assert len(df_candidates) == expected_rows
    assert len(df_candidates[df_candidates.main_emissions_factor_wtt_gco2eq_mj.isna()]) == 0

    expected_rows = len(df_candidates)
    df_candidates = df_candidates.merge(
        df_wtt_long.rename(columns={'fuel': 'pilot_fuel', 'wtt_gco2e_p_mj': 'pilot_emissions_factor_wtt_gco2eq_mj'}),
        on=['year', 'pilot_fuel'],
        how='left',
    )
    assert len(df_candidates) == expected_rows

    expected_rows = len(df_candidates)
    df_candidates = df_candidates.merge(
        df_ttw_carbon_intensity_ref[['fuel', 'engine_type', 'ttw_gco2e_p_mj']].rename(
            columns={
                'engine_type': 'engine_type_with_label',
                'ttw_gco2e_p_mj': 'main_emissions_factor_ttw_gco2eq_mj',
            }
        ),
        on = ['fuel', 'engine_type_with_label'],
        how='left',
    )
    assert len(df_candidates) == expected_rows
    assert len(df_candidates[df_candidates.main_emissions_factor_ttw_gco2eq_mj.isna()]) == 0, f"Empty ttw: {df_candidates[df_candidates.main_emissions_factor_ttw_gco2eq_mj.isna()]}"

    expected_rows = len(df_candidates)
    df_candidates = df_candidates.merge(
        df_ttw_carbon_intensity_ref[['fuel', 'engine_type', 'ttw_gco2e_p_mj']].rename(
            columns={
                'fuel': 'pilot_fuel',
                'engine_type': 'engine_type_with_label',
                'ttw_gco2e_p_mj': 'pilot_emissions_factor_ttw_gco2eq_mj',
            }
        ),
        on=['pilot_fuel', 'engine_type_with_label'],
        how='left',
    )
    assert len(df_candidates) == expected_rows

    # Pilot share and fuel consumption breakdown.
    df_candidates['wind'] = df_candidates['wind'].fillna('')
    df_candidates['ccs'] = df_candidates['ccs'].fillna('')

    # Call calculate_fuel_consumption with DataFrame path (fuel is a Series, fuel_result is a DataFrame)
    # This function handles merging df_pilot_fuel_input_ref internally for the DataFrame case.
    df_candidates = calculate_fuel_consumption(
        params=df_candidates[['fuel','fuel_consumption_mj_year']],
        fuel=df_candidates['fuel'],  # Series for DataFrame path
        fuel_result=df_candidates,  # DataFrame being updated in-place
        df_pilot_fuel_input_ref=df_pilot_fuel_input_ref,
        INCLUDE_PILOT_FUELS=True,
        CCS_ADD_FUEL_CONS=CCS_ADD_FUEL_CONS,
        WIND_ENERGY_SAVINGS=WIND_ENERGY_SAVINGS,
    )

    emissions = calculate_emissions(
        df_candidates,
        CARBON_CAPTURE_RATE,
    )
    for k, v in emissions.items():
        df_candidates[k] = v

    df_all_fuels = df_candidates
    assert len(df_all_fuels[df_all_fuels['engine_type'].isna()]) == 0, f"Nan in the engine_types: {df_all_fuels[df_all_fuels['engine_type'].isna()]}"

    # Append fuel-mix candidates when reference tables are available.
    if df_drop_in_fuels_ref is not None and engines_design_mapping is not None:
        df_mix = calc_fuel_mix_on_the_base(
                df_all_fuels=df_all_fuels,
                df_drop_in_fuels_ref=df_drop_in_fuels_ref,
                df_pilot_fuel_input_ref=df_pilot_fuel_input_ref,
                engines_design_mapping=engines_design_mapping,
                compliance_mode=compliance_mode,
            )
        if len(df_mix) > 0:
            df_mix = df_mix[df_mix.total_fuel_cost.notna()]
            if len(df_mix) > 0:
                df_all_fuels = pd.concat(
                    [df_all_fuels, df_mix], ignore_index=True, sort=False
                )
    assert len(df_all_fuels[df_all_fuels['engine_type'].isna()]) == 0, f"Nan in the engine_types: {df_all_fuels[df_all_fuels['engine_type'].isna()]}"

    # Compute EU policy compliance once across all candidates (single-fuel + mix).
    reward_by_year = df_reward_ref.drop_duplicates(subset=['year']).set_index('year')['value'].to_dict()
    base_reward = df_all_fuels['year'].map(reward_by_year).fillna(0.0).to_numpy() * 41.2
    df_all_fuels['reward_usd_tlsfoeq'] = np.where(
        df_all_fuels['fuel'].isin(reward_fuels),
        base_reward,
        0.0,
    )
    df_all_fuels['reward_usd'] = (
        df_all_fuels['reward_usd_tlsfoeq']
        / (1 + df_all_fuels['wacc']) ** df_all_fuels['reward_delay']
        * (df_all_fuels['overall_emissions_factor_wtw_gco2eq_mj'] < df_all_fuels['reward_threshold_gco2_mj'])
        * df_all_fuels['total_consumption_tlsfo_eq']
    )

    # in this stage, we only calculate the EU cost, since the NZF cost need to be calculated simulatenously across all reg_reg_class
    policy = calculate_policy_compliance(
        df_all_fuels,
        df_all_fuels['total_consumption_tlsfo_eq'].values,
        df_all_fuels['overall_emissions_factor_wtw_gco2eq_mj'].values,
        df_all_fuels['total_ttw_emissions_tco2eq'].values,
        None,  # attained_gfi_wtw_gco2eq_mj not needed/used in "fueleu" mode
        "fueleu",
    )

    for k, v in policy.items():
        if k in ['FuelEU_penalty','EU_ETS','pooling_value','fueleu_compliance_cost_usd_p_gj','eu_pooling_revenue_usd_p_gj','eu_compliance_cost_usd','eu_ets_usd']:
            df_all_fuels[k] = v
    assert len(df_all_fuels[df_all_fuels['engine_type'].isna()]) == 0, f"Nan in the engine_types: {df_all_fuels[df_all_fuels['engine_type'].isna()]}"

    return df_all_fuels.reset_index(drop=True)

def apply_engine_efficiency_correction(consumption, bte_initial, bte_fd):
    """Scale fuel consumption for a change in engine brake thermal efficiency (BTE).

    Cons_fd = Cons_initial * BTE_initial / BTE_fd — a more efficient engine (higher
    BTE_fd) burns less fuel for the same output. Shared by the Fleet Evolution Model
    and the Climate Risk Metrics tool so the formula only needs to change in one place.
    """
    return consumption * bte_initial / bte_fd

# --- ENERGY EFFICIENCY / SPEED REDUCTION ---
def apply_fuel_efficiency_adjustment(df_params, df_operational_data_ref, game_year, df_speed_ref, df_efficiency_ref):
    """Return an adjusted copy of df_operational_data_ref (which contains both the distance sailed and the fuel consumption) with the combined
    speed-reduction and energy-efficiency factor applied to ``tj_per_ship_pa`` and 'av_dist_nm'.

    Physics: annual energy ∝ v³ (at fixed operating hours), so a fractional
    speed reduction s_t reduces energy by (1−s_t)³.  An independent engine
    efficiency improvement e_t reduces it by a further factor (1−e_t).
    Combined factor: (1−s_t)³ × (1−e_t).

    The function always returns a **new copy** of the input to avoid double-
    counting across repeated SU-market loop calls within the same game year.

    Args:
        df_operational_data_ref (pd.DataFrame): Reference fuel-consumption table
            with at least a ``tj_per_ship_pa`` and 'av_dist_nm' column.
        game_year (int): Current simulation year used to look up the factors.
        df_speed_ref (pd.DataFrame): Speed-reduction table already
            filtered to the active scenario; columns: year, speed_reduction.
            Required and must include the current game_year row.
        df_efficiency_ref (pd.DataFrame): Energy-efficiency table already
            filtered to the active scenario; columns: year, energy_efficiency_impr.
            Required and must include the current game_year row.

    Returns:
        pd.DataFrame: A copy of df_operational_data_ref with ``tj_per_ship_pa``
            scaled by the combined factors, plus the updated df_params.
    """
    assert df_speed_ref is not None, "df_speed_ref is required"
    assert df_efficiency_ref is not None, "df_efficiency_ref is required"

    row = df_speed_ref[df_speed_ref['year'] == game_year]
    assert len(row) > 0, f"Empty speed reduction input for {game_year}"
    s_t = float(row['speed_reduction'].iloc[0])

    row = df_efficiency_ref[df_efficiency_ref['year'] == game_year]
    assert len(row) > 0, f"Empty energy efficiency input for {game_year}"
    e_t = float(row['energy_efficiency_impr'].iloc[0])

    # Only the share of energy at sea is affected by speed reduction
    df_adj = df_operational_data_ref.copy()
    df_adj['tj_per_ship_pa'] =(
        df_adj['tj_per_ship_pa'] 
        * (
            df_adj['share_energy_at_sea'] * (1.0 - s_t) ** 3
            + (1.0 - df_adj['share_energy_at_sea'])
        ) * (1.0 - e_t)
    )
    df_adj[(df_adj.imo_type == "Container")&(df_adj.size_id == 8)]['tj_per_ship_pa']
    # adjust the distance sailed per year by the change in speed
    df_adj['av_dist_nm'] = df_adj['av_dist_nm'] * (1 - df_adj['elasticity_distance_by_speed'] * s_t)

    # Fuel consumption merge
    df_fc = (
        df_adj[['imo_type', 'size_id', 'tj_per_ship_pa','av_dist_nm',]]
        .drop_duplicates(subset=['imo_type', 'size_id'])
        .rename(columns={'tj_per_ship_pa': 'fuel_consumption_tj_year'})
    )
    expected_rows = len(df_params)
    df_params = df_params.merge(df_fc, on=['imo_type', 'size_id'], how='left')
    assert len(df_params) == expected_rows, (
        f"Row count mismatch after fuel consumption merge: expected {expected_rows}, got {len(df_params)}"
    )
    assert df_params['fuel_consumption_tj_year'].notna().all(), (
        "fuel_consumption_tj_year contains NaNs after merge"
    )
    df_params['fuel_consumption_mj_year'] = df_params['fuel_consumption_tj_year'] * 1_000_000
    df_params['transport_work_dwt_nm'] = df_params['av_dist_nm'] * df_params['dwt']

    return df_adj, df_params


def apply_engine_efficiency_adjustment(df_params, df_engine_efficiency_ref):
    """Correct fuel consumption for the BTE of the chosen engine design (df_params['engine'])
    relative to conventional, via apply_engine_efficiency_correction().

    df_operational_data_ref always represents the conventional baseline, so newbuild
    (baseline = conventional) and retrofit/keep (baseline = previous engine) scenarios
    are mathematically equivalent here: the previous engine's BTE cancels out, leaving
    Cons_fd = Cons_conventional * BTE_conventional / BTE_fd.
    """
    df_adj = df_params.copy()
    df_adj['base_engine_design'] = df_adj['engine'].astype(str).str.split('_+_', regex=False).str[0]

    bte_ref = df_engine_efficiency_ref[['fuel_design', 'n_stroke', 'bte']]

    expected_rows = len(df_adj)
    df_adj = df_adj.merge(
        bte_ref.rename(columns={'fuel_design': 'base_engine_design', 'bte': 'bte_fd'}),
        on=['base_engine_design', 'n_stroke'], how='left',
    )
    df_adj = df_adj.merge(
        bte_ref[bte_ref['fuel_design'] == 'conventional'].drop(columns='fuel_design').rename(columns={'bte': 'bte_conventional'}),
        on='n_stroke', how='left',
    )
    assert len(df_adj) == expected_rows, (
        f"Row count mismatch after BTE merge: expected {expected_rows}, got {len(df_adj)}"
    )
    assert df_adj['bte_fd'].notna().all(), (
        f"Missing BTE lookup for: "
        f"{df_adj.loc[df_adj['bte_fd'].isna(), ['base_engine_design', 'n_stroke']].drop_duplicates().to_dict('records')}"
    )
    assert df_adj['bte_conventional'].notna().all(), "Missing conventional BTE baseline for one or more n_stroke values"

    for col in ('fuel_consumption_tj_year', 'fuel_consumption_mj_year'):
        if col in df_adj.columns:
            df_adj[col] = apply_engine_efficiency_correction(df_adj[col], df_adj['bte_conventional'], df_adj['bte_fd'])

    return df_adj.drop(columns=['base_engine_design', 'bte_fd', 'bte_conventional'])

def apply_wind_ccs_adjustments(wtt_factors, ttw_factors, wind_mask, ccs_mask, fuel_savings, ccs_rate):
    """
    Apply wind and CCS reductions to emission factors (WTT and TTW).    
    Wind reduces both WTT and TTW; CCS reduces only TTW.
    
    Args:
      wtt_factors: WTT emission factors (array or scalar)
      ttw_factors: TTW emission factors (array or scalar)
      wind_mask: Boolean mask where wind is active
      ccs_mask: Boolean mask where CCS is active
      fuel_savings: Fractional savings from wind (array matching wind_mask)
      ccs_rate: Fractional reduction rate from CCS (scalar)
    
    Returns:
      Tuple of (adjusted_wtt, adjusted_ttw)
    """
    # Wind reduces both WTT and TTW
    wtt_adj = np.where(wind_mask, wtt_factors * (1 - fuel_savings), wtt_factors)
    ttw_adj = np.where(wind_mask, ttw_factors * (1 - fuel_savings), ttw_factors)
    
    # CCS reduces only TTW
    ttw_adj = np.where(ccs_mask, ttw_adj * (1 - ccs_rate), ttw_adj)
    
    return wtt_adj, ttw_adj


def calculate_emissions(fuel_result, ccs_rate):

    # energy consumptions
    total_mj = fuel_result['main_fuel_consumption_mj_year'] + fuel_result['pilot_fuel_consumption_mj_year']

    # Apply wind/CCS adjustments using shared helper
    wind_mask = fuel_result['wind'] != ""
    ccs_mask = fuel_result['fuel'].str.contains("ccs", na=False)
    savings = fuel_result['fuel_savings'].to_numpy(dtype=float)
    
    main_emissions_factor_wtt_gco2eq_mj, main_emissions_factor_ttw_gco2eq_mj = apply_wind_ccs_adjustments(
        wtt_factors=fuel_result['main_emissions_factor_wtt_gco2eq_mj'],
        ttw_factors=fuel_result['main_emissions_factor_ttw_gco2eq_mj'],
        wind_mask=wind_mask,
        ccs_mask=ccs_mask,
        fuel_savings=savings,
        ccs_rate=ccs_rate,
    )
    
    pilot_emissions_factor_wtt_gco2eq_mj, pilot_emissions_factor_ttw_gco2eq_mj = apply_wind_ccs_adjustments(
        wtt_factors=fuel_result['pilot_emissions_factor_wtt_gco2eq_mj'],
        ttw_factors=fuel_result['pilot_emissions_factor_ttw_gco2eq_mj'],
        wind_mask=wind_mask,
        ccs_mask=ccs_mask,
        fuel_savings=savings,
        ccs_rate=ccs_rate,
    )

    # -----------------------------
    # Compute emissions (units: g -> tonnes by *1e-6)
    # -----------------------------
    # total emissions
    total_wtw_emissions_tco2eq = (
        fuel_result['main_fuel_consumption_mj_year'] * (main_emissions_factor_wtt_gco2eq_mj + main_emissions_factor_ttw_gco2eq_mj)
        + fuel_result['pilot_fuel_consumption_mj_year'] * (pilot_emissions_factor_wtt_gco2eq_mj + pilot_emissions_factor_ttw_gco2eq_mj)
    ) * 1e-6
    total_ttw_emissions_tco2eq = (
        fuel_result['main_fuel_consumption_mj_year'] * main_emissions_factor_ttw_gco2eq_mj
        + fuel_result['pilot_fuel_consumption_mj_year'] * pilot_emissions_factor_ttw_gco2eq_mj
    ) * 1e-6

    # -----------------------------
    # overall emissions factors (gCO2eq/MJ)
    # using the *adjusted* factors
    # -----------------------------
    overall_emissions_factor_wtw_gco2eq_mj = total_wtw_emissions_tco2eq / total_mj * 1e6
    overall_emissions_factor_ttw_gco2eq_mj = total_ttw_emissions_tco2eq / total_mj * 1e6

    return {
        'total_wtw_emissions_tco2eq': total_wtw_emissions_tco2eq,
        'total_ttw_emissions_tco2eq': total_ttw_emissions_tco2eq,
        'overall_emissions_factor_wtw_gco2eq_mj': overall_emissions_factor_wtw_gco2eq_mj,
        'overall_emissions_factor_ttw_gco2eq_mj': overall_emissions_factor_ttw_gco2eq_mj,
    }


def calculate_fuel_consumption(params, fuel, fuel_result, df_pilot_fuel_input_ref,
                               INCLUDE_PILOT_FUELS = True,
                               CCS_ADD_FUEL_CONS=CCS_ADD_FUEL_CONS,
                               WIND_ENERGY_SAVINGS=WIND_ENERGY_SAVINGS,
):
    # ===== FUEL CONSUMPTION CALCULATIONS =====

    # assign the share of pilot fuel
    if INCLUDE_PILOT_FUELS:
        check_len = len(fuel_result)
        fuel_result = fuel_result.merge(df_pilot_fuel_input_ref, on='fuel', how='left')
        assert len(fuel_result) == check_len
        fuel_result['share_pilot_mj'] = fuel_result['share_pilot_mj'].fillna(0)
        share_pilot_fuel = fuel_result['share_pilot_mj']
    else:
        share_pilot_fuel = 0

    # breakdown between pilot and main fuel consumption
    fuel_result['pilot_fuel_consumption_mj_year'] = params['fuel_consumption_mj_year'] * share_pilot_fuel
    fuel_result['main_fuel_consumption_mj_year'] = params['fuel_consumption_mj_year'] - fuel_result['pilot_fuel_consumption_mj_year']

    # ===== CCS & WIND ADJUSTMENTS =====

    # --- 1. CCS increases fuel consumption
    assert len(fuel) == len(fuel_result), f"{len(fuel)}, {len(fuel_result)}"
    mask_ccs = fuel.str.contains("ccs", na=False)

    fuel_result.loc[mask_ccs, "pilot_fuel_consumption_mj_year"] *= (1 + CCS_ADD_FUEL_CONS)
    fuel_result.loc[mask_ccs, "main_fuel_consumption_mj_year"] *= (1 + CCS_ADD_FUEL_CONS)

    # --- 2. Apply wind fuel savings
    wind_map = WIND_ENERGY_SAVINGS.set_index("wind")["fuel_savings"].to_dict()

    # map wind → savings
    # since wind is considered a source of energy, not an energy efficiency, we do not correct the fuel consumption by the fuel savings
    # we will however correct the carbon emissions and fuel costs accordingly
    fuel_result["fuel_savings"] = fuel_result["wind"].map(wind_map)
    fuel_result["fuel_savings"] = fuel_result["fuel_savings"].fillna(0)

    # Calculate VLSFO equivalent consumption
    fuel_result['main_consumption_tlsfo_eq'] = fuel_result['main_fuel_consumption_mj_year'] / 41.2 / 1_000
    fuel_result['pilot_consumption_tlsfo_eq'] = fuel_result['pilot_fuel_consumption_mj_year'] / 41.2 / 1_000
    fuel_result['total_consumption_tlsfo_eq'] = fuel_result['main_consumption_tlsfo_eq'] + fuel_result['pilot_consumption_tlsfo_eq']

    # Fuel cost calculations
    fuel_result['main_fuel_cost'] = (
        fuel_result['main_consumption_tlsfo_eq']
        * fuel_result['main_fuel_price_usd_tvslfoeq']
        * (1 - fuel_result["fuel_savings"])
    )
    fuel_result['pilot_fuel_cost'] = (
        fuel_result['pilot_consumption_tlsfo_eq']
        * fuel_result['pilot_fuel_price_usd_tvslfoeq']
        * (1 - fuel_result["fuel_savings"])
    )

    fuel_result['total_fuel_cost'] = fuel_result['main_fuel_cost'] + fuel_result['pilot_fuel_cost']

    return fuel_result

def calc_fuel_mix_on_the_base(
    df_all_fuels: pd.DataFrame,
    df_drop_in_fuels_ref: pd.DataFrame,
    df_pilot_fuel_input_ref: pd.DataFrame,
    engines_design_mapping: pd.DataFrame,
    compliance_mode: str,
) -> pd.DataFrame:
    """
    Takes the already-computed single-fuel candidate table (df_all_fuels, one row
    per tco_id × fuel after best-pilot dedup) and returns a DataFrame of fuel-mix
    candidates in the same schema so they can be concatenated into df_all_fuels
    before the final min-cost dedup.

    The existing create_fuel_mix_on_the_base function already operates on a batch
    pivot table keyed by (scenario_id, year, engine_type, reg_reg_class); we reuse
    it here mapping tco_id → scenario_id so it handles all rows at once.
    """

    # columns which are needed as an input to the calculation of the fuel mixes
    fuel_mix_input_columns = [
        "scenario_id", "year", "engine_type", "reg_reg_class", "fuel",
        "wtw_gco2e_p_mj", "ttw_gco2e_p_mj", "total_fuel_cost",
    ]
    # List of columns which need to be replaced by the fuel mix - the other can stay the same
    fuel_mix_output_columns = [
        'total_fuel_cost',
        'overall_emissions_factor_wtw_gco2eq_mj',
        'overall_emissions_factor_ttw_gco2eq_mj',
        "fuel",
        "fuel_1",
        "fuel_2",
        "fuel_3",
        "share_fuel_1_final",
        "share_fuel_2_final",
        "share_fuel_3_final",
    ]
    # List of columns which are used to link back the fuel mixes to the tco_id dataframe
    # There is a misalignment btw the CRM and the fleet evolv. : in the CRM, the dropin fuel design (= engine in fleet evolv terminology) 
    # and fuel include both the fuel and the add ons. e.g. df_ice_ammonia_+_rigid_sail/e-ammonia + rigid sail. 
    # In the fleet evolv, engine only includes df_ice_ammonia. So wind and CCS are marked in scenario id (which corresponds literally to the 
    # fuel scenairo in the CRM, here is useless) and the results are merged using both the scenario id, the engine and the engine type 
    # ("= LNG Diesel 2024+")
    id_columns = ['scenario_id', 'base_engine','engine_type_with_label', 'year', 'reg_reg_class']

    pilot_fuels_clean = [f.lower().replace(' ', '_') for f in PILOT_FUELS]
    # Restrict pilot fuels to those that survived the allowed_fuels filter
    available_fuel_names = set(df_all_fuels['fuel'].unique())
    pilot_fuels_clean = [pf for pf in pilot_fuels_clean if pf in available_fuel_names]

    compliance_options = {
        'fueleu': [('FuelEU', 'wtw_gco2e_p_mj')],
        'imo': [('base_compliance_target', 'wtw_gco2e_p_mj')],
        'fueleu_imo': [
            ('FuelEU', 'wtw_gco2e_p_mj'),
            ('base_compliance_target', 'wtw_gco2e_p_mj'),
        ],
        'jointly_imo': [
            ('corrected_base_compliance_target', 'wtw_gco2e_p_mj'),
        ],
    }
    selected_pairs = compliance_options.get(compliance_mode, [])
    if not selected_pairs:
        assert 0 == 1, f"Incompatible compliance mode = {compliance_mode}"
    
    df_all_fuels = df_all_fuels.copy()
    df_all_fuels['scenario_id'] = df_all_fuels['wind'].astype(str) + "_" + df_all_fuels['ccs'].astype(str)
    df_all_fuels['_mix_base_scenario_id'] = df_all_fuels['scenario_id']

    # Apply wind/CCS adjustments to main fuel EFs before passing to fuel mix solver.
    # This ensures the solver uses EFs consistent with the GFI targets (which were computed with wind/CCS already applied).
    wind_mask = df_all_fuels['wind'].astype(str) != ""
    ccs_mask = df_all_fuels['ccs'].astype(str).str.contains('ccs', case=False, na=False)
    fuel_savings_vals = df_all_fuels['fuel_savings'].to_numpy(dtype=float)
    
    main_wtt_adjusted, main_ttw_adjusted = apply_wind_ccs_adjustments(
        wtt_factors=df_all_fuels['main_emissions_factor_wtt_gco2eq_mj'].to_numpy(dtype=float),
        ttw_factors=df_all_fuels['main_emissions_factor_ttw_gco2eq_mj'].to_numpy(dtype=float),
        wind_mask=wind_mask,
        ccs_mask=ccs_mask,
        fuel_savings=fuel_savings_vals,
        ccs_rate=CARBON_CAPTURE_RATE,
    )
    
    # Build op_cost in the format create_fuel_mix_on_the_base expects.
    # We only include the main fuel, as the pilot fuel is re-computed in the fuel mix on the base
    op_cost = df_all_fuels.drop(columns = {'engine_type'}).rename(columns={
        'engine_type_with_label': 'engine_type',
    }).copy()
    # Use adjusted EFs and main-fuel-only cost
    op_cost['wtw_gco2e_p_mj'] = main_wtt_adjusted + main_ttw_adjusted  # WTW = WTT + TTW
    op_cost['ttw_gco2e_p_mj'] = main_ttw_adjusted                        # TTW only
    # Convert main fuel cost to $/mj
    op_cost['total_fuel_cost'] = op_cost['main_fuel_cost'] / op_cost['main_fuel_consumption_mj_year']

    # Melt drop-in fuels — keep all fuel_design rows (not filtered to one engine).
    dropin = df_drop_in_fuels_ref.melt(
        id_vars='fuel',
        var_name='fuel_design',
        value_name='drop_in',
    )
    # only select the engines which are selected in the fleet
    fuel_designs_filtered = df_all_fuels.base_engine.unique()
    dropin = dropin[dropin.fuel_design.isin(fuel_designs_filtered)]
    # only keep fuels that survived the allowed_fuels filter in df_all_fuels
    dropin = dropin[dropin.fuel.isin(df_all_fuels['fuel'].unique())]

    template_drop_columns = [
        col for col in fuel_mix_output_columns if col in df_all_fuels.columns
    ]

    mix_frames = []

    for gfi_var, ci_var in selected_pairs:
        op_cost_current = op_cost.copy()

        # gfi_var column must exist in op_cost (added by calculate_policy_compliance).
        assert gfi_var in op_cost_current.columns, f"[fuel_mix_batch] Missing required column '{gfi_var}' in op_cost"

        # Only look for fuel mixes in the NZF if there is a valid base compliance target (btw 0 and 100)
        # and only if there a legitimate reason to operate on the base (i.e., a form of levy, i.e. direct != base)
        # for the FuelEU one, the EU ETS introduces a de facto levy on emissions, so we always try a fuel mix on the FuelEU
        if op_cost_current.empty:
            print(f"op_cost is empty for gfi_var {gfi_var}")        
        if gfi_var == "base_compliance_target":
            op_cost_current = op_cost_current[
               (op_cost_current['base_compliance_target'] > op_cost_current['direct_compliance_target'])
               &(op_cost_current['base_compliance_target'] < 100)
               &(op_cost_current['base_compliance_target'] > 0)
            ]
            if op_cost_current.empty :
                continue

        # Keep target identity distinct through the solver without changing
        # create_fuel_mix_on_the_base, which treats scenario_id as an opaque key.
        op_cost_current['scenario_id'] = (
            op_cost_current['_mix_base_scenario_id'].astype(str)
            + '__target_'
            + op_cost_current[gfi_var].astype(str)
        )
        
        # Create gfi_var-specific subset with only columns needed for pivot.
        op_cost_unique = (
            op_cost_current[fuel_mix_input_columns + [gfi_var]]
            .drop_duplicates(subset = ['scenario_id','year','engine_type','reg_reg_class','fuel', gfi_var])
        )
        df_fuel_mix_row, _ = create_fuel_mix_on_the_base(
            op_cost_in=op_cost_unique,
            dropin_in=dropin,
            pilot_fuel_input=df_pilot_fuel_input_ref,
            engine_types_mapping=engines_design_mapping,
            pilot_fuels=pilot_fuels_clean,
            gfi_var=gfi_var,
            ci_var=ci_var,
            DEBUG=False,
        )

        # Handle case where no valid fuel mixes are possible (e.g., all drop-ins exhausted).
        if df_fuel_mix_row is None or len(df_fuel_mix_row) == 0 or 'total_fuel_cost_mix' not in df_fuel_mix_row.columns:
            if len(op_cost_unique) > 0:
                print(f"No fuel-mix rows generated for gfi_var={gfi_var}, year={df_all_fuels['year'].drop_duplicates()}. "
                      f"This may occur when all drop-in fuel availability caps are reached.")
            continue

        # Keep only rows where a valid, finite mix cost was found.
        mix_cost_num = pd.to_numeric(df_fuel_mix_row['total_fuel_cost_mix'], errors='coerce')
        valid_mix_mask = mix_cost_num.notna() & np.isfinite(mix_cost_num.to_numpy()) & (mix_cost_num < 10000)
        df_fuel_mix_row = df_fuel_mix_row.loc[valid_mix_mask].copy()
        df_fuel_mix_row['total_fuel_cost_mix'] = mix_cost_num.loc[valid_mix_mask]
        
        if len(df_fuel_mix_row) == 0:
            if len(op_cost_unique) > 0:
                print(f"All fuel-mix costs were invalid/infinite for gfi_var={gfi_var}, year={df_all_fuels['year'].drop_duplicates()}.")
            continue

        # Rename mix-specific columns to match single-fuel schema.
        df_fuel_mix_row = df_fuel_mix_row.rename(columns={
            'total_fuel_cost_mix': 'total_fuel_cost',
            'wtw_mix': 'overall_emissions_factor_wtw_gco2eq_mj',
            'ttw_mix': 'overall_emissions_factor_ttw_gco2eq_mj',
            'engine_type': 'engine_type_with_label'
        })
        df_fuel_mix_row['base_engine'] = df_fuel_mix_row['fuel'].str.replace(f"{gfi_var} mix ", "")

        # merge the fuel mixes back onto the large dataframe
        df_template = (
            df_all_fuels.drop(columns=template_drop_columns)
            .drop_duplicates(subset=['tco_id', gfi_var])
            .copy()
        )
        df_template['scenario_id'] = (
            df_template['_mix_base_scenario_id'].astype(str)
            + '__target_'
            + df_template[gfi_var].astype(str)
        )
        df_fuel_mix = df_template.merge(
            df_fuel_mix_row[id_columns + fuel_mix_output_columns].drop_duplicates(), 
            on = id_columns, 
            how='left'
        )
        assert len(df_fuel_mix) == len(df_template), f"Mismatch in length after merge: {len(df_fuel_mix)}, {len(df_template)}"
        if len(df_fuel_mix[df_fuel_mix.total_fuel_cost.isna()]) > 0:
            f"Empty fuel mixes: {df_fuel_mix[df_fuel_mix.total_fuel_cost.isna()][['engine','year','reg_reg_class']].drop_duplicates()}"

        # Total fuel cost was expressed in $/mj so needs to be converted back to total
        df_fuel_mix['total_fuel_cost'] = df_fuel_mix['total_fuel_cost'] * df_fuel_mix['fuel_consumption_mj_year']

        # the pilot and main fuel consumptions are now dealt with in fuel1, fuel2, fuel3
        df_fuel_mix['pilot_fuel_consumption_mj_year'] = 0
        df_fuel_mix['main_fuel_consumption_mj_year'] = 0

        df_fuel_mix['total_wtw_emissions_tco2eq'] = (
            df_fuel_mix['overall_emissions_factor_wtw_gco2eq_mj']
            * df_fuel_mix['fuel_consumption_mj_year'] * 1e-6
        )
        df_fuel_mix['total_ttw_emissions_tco2eq'] = (
            df_fuel_mix['overall_emissions_factor_ttw_gco2eq_mj']
            * df_fuel_mix['fuel_consumption_mj_year'] * 1e-6
        )

        # Mixes do not receive rewards.
        df_fuel_mix['reward_usd_tlsfoeq'] = 0.0
        df_fuel_mix['reward_usd'] = 0.0
        df_fuel_mix['scenario_id'] = df_fuel_mix['_mix_base_scenario_id']
        df_fuel_mix = df_fuel_mix.drop(columns=['_mix_base_scenario_id'], errors='ignore')

        mix_frames.append(df_fuel_mix)

    if not mix_frames:
        return pd.DataFrame()

    return pd.concat(mix_frames, ignore_index=True, sort=False)


def create_fuel_mix_on_the_base(
    op_cost_in: pd.DataFrame,
    dropin_in: pd.DataFrame,
    pilot_fuel_input: pd.DataFrame,
    engine_types_mapping: pd.DataFrame,
    pilot_fuels: list,
    gfi_var: str,
    ci_var: str,
    DEBUG: bool = False,
):
    """
    This function computes, for each design and each engine type, the fuel mix for the ship to operate on an exogenous GFI (base GFI or FuelEU).
    Each fuel mix is made of only 2 drop-in fuels (+ sometimes a pilot fuel), the carbon intensity weighted average of which equals the base GFI value.
    The fuel mix is chosen as the cheapest mix for each year * scenario, but single fuel are ignored at this stage.
    """

    # defensive copies
    op_cost = op_cost_in.copy()
    dropin = dropin_in.copy()

    # list available fuels per fuel_design
    available_fuels = defaultdict(list)
    for fd, fuel, drop_in_flag in zip(dropin["fuel_design"], dropin["fuel"], dropin["drop_in"]):
        if drop_in_flag:
            available_fuels[fd].append(fuel)
    available_fuels = dict(available_fuels)

    need_pilot_fuel = pilot_fuel_input.fuel.unique()
    if DEBUG:
        print(f"\n need_pilot_fuel: {need_pilot_fuel}")

    # Create the pivot table
    index_vars = ["scenario_id", "year", "engine_type", "reg_reg_class"]
    assert op_cost[index_vars + ["fuel"]].drop_duplicates().shape[0] == op_cost.shape[0], "Duplicate rows in op_cost"
    op_pivot = op_cost.pivot_table(
        index=index_vars,
        columns=["fuel"],
        values=["wtw_gco2e_p_mj", "ttw_gco2e_p_mj", "total_fuel_cost", gfi_var],
    )

    # Flatten MultiIndex columns to single level
    op_pivot.columns = [
        f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col
        for col in op_pivot.columns
    ]
    op_pivot = op_pivot.reset_index()

    # Prepare all fuel pairs per fuel_design and engine_type
    fuel_pairs = {}

    for fd, fuels in available_fuels.items():
        # Select engine types for this fuel_design
        ets = engine_types_mapping.loc[
            engine_types_mapping.fuel_design == fd, "engine_type"
        ].tolist()

        for et in ets:
            combos = set()
            for f1, f2 in combinations(fuels, 2):
                if f1 != f2:
                    # Normal 2-fuel combination if not both in need of a pilot fuel
                    if not (f1 in need_pilot_fuel and f2 in need_pilot_fuel):
                        combos.add((f1, f2, np.nan))
                    else:
                        # If both need pilot → add pilot fuels
                        for pf in pilot_fuels:
                            combos.add((f1, f2, pf))
            fuel_pairs[(fd, et)] = list(combos)

    if DEBUG:
        print(f"fuel_pairs: {fuel_pairs}")
        
    fuel_mix = pd.DataFrame()

    for (fd, et), pairs in fuel_pairs.items():
        if DEBUG:
            print(f"fuel design, engine type, pairs: {fd},{et}, {pairs}")
        # Filter by engine type using MultiIndex
        op_pivot_temp = op_pivot[op_pivot["engine_type"].str.strip() == et.strip()].copy()  
        if len(op_pivot_temp) > 0 and len(pairs) > 0:
            # create a new fuel corresponding to the GFI target and the fuel_design in which it is available, e.g. "Base GFI mix df_ice_lng"
            op_pivot_temp["fuel"] = f"{gfi_var} mix {fd}"

            # if no combination of fuels is found, we need the operating cost of the fuel mix to be too high to be picked up
            op_pivot_temp["total_fuel_cost_mix"] = pd.Series(
                np.full(len(op_pivot_temp), np.inf, dtype=np.float64),
                index=op_pivot_temp.index,
            )
            op_pivot_temp["carbon_intensity_mix"] = np.inf

            for f1, f2, f3 in pairs:
                if DEBUG:
                    print("f1, f2, f3: ")
                    print(f1, f2, f3)
                # Handle pilot vs non-pilot cases
                if pd.isna(f3):
                    op_pivot_temp["share_fuel_3"] = 0
                    f3 = "lsfo"
                else:
                    share_pilot = pilot_fuel_input.loc[
                        pilot_fuel_input.fuel == f1, "share_pilot_mj"
                    ].iloc[0]
                    op_pivot_temp["share_fuel_3"] = share_pilot

                op_pivot_temp["target_ci"] = op_pivot_temp[f"{gfi_var}_{f1}"]

                # Blend share solved so the weighted-average carbon intensity of the
                # blend (main + secondary + pilot) matches the target: s1*ei1 + s2*ei2 + s3*ei3 = target,
                # with s1 + s2 + s3 = 1.
                s3 = op_pivot_temp["share_fuel_3"]
                target = op_pivot_temp["target_ci"]
                ei1 = op_pivot_temp[f"{ci_var}_{f1}"]
                ei2 = op_pivot_temp[f"{ci_var}_{f2}"]
                ei3 = op_pivot_temp[f"{ci_var}_{f3}"]

                solve_denom = ei2 - ei1
                solve_denom = np.where(solve_denom == 0, 1e-12, solve_denom)

                op_pivot_temp["share_fuel_1"] = (
                    (1 - s3) * ei2 + s3 * ei3 - target
                ) / solve_denom
                op_pivot_temp["share_fuel_2"] = (
                    1 - op_pivot_temp["share_fuel_1"] - op_pivot_temp["share_fuel_3"]    # Should this be 100 - share_fuel1 - share_fuel3?
                )

                # Compute temp costs and intensities (share-weighted averages)
                for var in ["total_fuel_cost", "wtw_gco2e_p_mj", "ttw_gco2e_p_mj"]:
                    op_pivot_temp[f"{var}_temp"] = (
                        op_pivot_temp["share_fuel_1"] * op_pivot_temp[f"{var}_{f1}"]
                        + op_pivot_temp["share_fuel_2"] * op_pivot_temp[f"{var}_{f2}"]
                        + op_pivot_temp["share_fuel_3"] * op_pivot_temp[f"{var}_{f3}"]
                    )

                # Mask where new pair is cheaper than all the previous pairs we looked at
                mask = op_pivot_temp[
                    (
                        (
                            op_pivot_temp["total_fuel_cost_temp"]
                            < op_pivot_temp["total_fuel_cost_mix"]
                        )
                        & (op_pivot_temp["share_fuel_1"].between(0, 1))
                        & (op_pivot_temp["share_fuel_2"].between(0, 1))
                    )
                ].index
                if DEBUG:
                    print("Mask where new pair is cheaper than all the previous pairs we looked at: ")
                    print(mask)

                # replace the cheapest pair recorded by the current pair if this pair is found to be cheaper
                op_pivot_temp.loc[mask, "total_fuel_cost_mix"] = op_pivot_temp.loc[
                    mask, "total_fuel_cost_temp"
                ]
                op_pivot_temp.loc[mask, "fuel_1"] = f1
                op_pivot_temp.loc[mask, "fuel_2"] = f2
                op_pivot_temp.loc[mask, "fuel_3"] = f3
                op_pivot_temp.loc[mask, "share_fuel_1_final"] = op_pivot_temp.loc[
                    mask, "share_fuel_1"
                ]
                op_pivot_temp.loc[mask, "share_fuel_2_final"] = op_pivot_temp.loc[
                    mask, "share_fuel_2"
                ]
                op_pivot_temp.loc[mask, "share_fuel_3_final"] = op_pivot_temp.loc[
                    mask, "share_fuel_3"
                ]
                op_pivot_temp.loc[mask, "wtw_mix"] = op_pivot_temp.loc[
                    mask, "wtw_gco2e_p_mj_temp"
                ]
                op_pivot_temp.loc[mask, "ttw_mix"] = op_pivot_temp.loc[
                    mask, "ttw_gco2e_p_mj_temp"
                ]

            fuel_mix = pd.concat(
                (
                    fuel_mix,
                    op_pivot_temp[
                        [
                            "scenario_id",
                            "year",
                            "engine_type",
                            "fuel",
                            "reg_reg_class",
                            "total_fuel_cost_mix",
                            "fuel_1",
                            "fuel_2",
                            "fuel_3",
                            "share_fuel_1_final",
                            "share_fuel_2_final",
                            "share_fuel_3_final",
                            "wtw_mix",
                            "ttw_mix",
                        ]
                    ],
                )
            )

    # add the new fuel designs to the drop-in fuels
    i = dropin.index.max() + 1
    for fd in dropin.fuel_design.unique():
        dropin.loc[i, "fuel"] = f"{gfi_var} mix {fd}"
        dropin.loc[i, "fuel_design"] = fd
        dropin.loc[i, "drop_in"] = True
        i = i + 1

    return fuel_mix, dropin
