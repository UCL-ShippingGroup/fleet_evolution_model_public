from typing import Tuple
import numpy as np
import pandas as pd
import uuid
from default_assumptions import LEARNING_FUELS, DEPRECIATION_TIME, HIGH_WACC_ENGINES
from fuel import build_all_fuel_options, apply_fuel_efficiency_adjustment, apply_engine_efficiency_adjustment, calc_fuel_mix_on_the_base
from policy import get_gfi_targets, get_eu_policy, get_ru_su_prices, calculate_policy_compliance
from utility_functions import extend_years

def aggregate_route_level_quantities(combos, available_regs): 
    # Aggregate route-level quantities into annual vessel-level totals.
    
    cons_mj_cols = [f'fuel_consumption_mj_year_{r}' for r in available_regs if f'fuel_consumption_mj_year_{r}' in combos.columns]
    cons_tlsfo_cols = [f'total_consumption_tlsfo_eq_{r}' for r in available_regs if f'total_consumption_tlsfo_eq_{r}' in combos.columns]
    fuel_cost_cols = [f'total_fuel_cost_{r}' for r in available_regs if f'total_fuel_cost_{r}' in combos.columns]
    eu_cost_cols = [f'eu_cost_usd_{r}' for r in available_regs if f'eu_cost_usd_{r}' in combos.columns]
    reward_cols = [f'reward_usd_{r}' for r in available_regs if f'reward_usd_{r}' in combos.columns]
    wtw_cols = [f'total_wtw_emissions_tco2eq_{r}' for r in available_regs if f'total_wtw_emissions_tco2eq_{r}' in combos.columns]
    ttw_cols = [f'total_ttw_emissions_tco2eq_{r}' for r in available_regs if f'total_ttw_emissions_tco2eq_{r}' in combos.columns]

    combos['fuel_consumption_mj_year'] = combos[cons_mj_cols].sum(axis=1)
    combos['total_consumption_tlsfo_eq'] = combos[cons_tlsfo_cols].sum(axis=1)
    combos['total_fuel_cost'] = combos[fuel_cost_cols].sum(axis=1)
    combos['eu_cost_usd'] = combos[eu_cost_cols].sum(axis=1)
    combos['reward_usd'] = combos[reward_cols].sum(axis=1) if reward_cols else 0.0
    combos['total_wtw_emissions_tco2eq'] = combos[wtw_cols].sum(axis=1)
    combos['total_ttw_emissions_tco2eq'] = combos[ttw_cols].sum(axis=1)

    # Recompute overall WTW/TTW intensities on the combined annual denominator.
    combos['overall_emissions_factor_wtw_gco2eq_mj'] = np.where(
        combos['fuel_consumption_mj_year'] > 0,
        combos['total_wtw_emissions_tco2eq'] * 1_000_000 / combos['fuel_consumption_mj_year'],
        0.0,
    )
    combos['overall_emissions_factor_ttw_gco2eq_mj'] = np.where(
        combos['fuel_consumption_mj_year'] > 0,
        combos['total_ttw_emissions_tco2eq'] * 1_000_000 / combos['fuel_consumption_mj_year'],
        0.0,
    )

    assert len(combos[combos['fuel_consumption_mj_year'] <= 0]) == 0, f"combos['fuel_consumption_mj_year'] <= 0 {combos[combos['fuel_consumption_mj_year'] <= 0].sample(3)}"
    return combos

def prepare_route_candidate_for_combo(df_in: pd.DataFrame, reg_name: str) -> pd.DataFrame:
    required_cols = [
        'tco_id', 'year', 'fuel_id', 'fuel',
        'fuel_consumption_mj_year', 'total_consumption_tlsfo_eq',
        'total_fuel_cost', 'eu_cost_usd', 'reward_usd',
        'total_wtw_emissions_tco2eq', 'total_ttw_emissions_tco2eq',
        'base_compliance_target', 'direct_compliance_target',
        'reward_threshold_gco2_mj',
        'ru2_price', 'ru1_price', 'su_price',
        'wacc', 'reward_delay',
        'FuelEU', 'FuelEU_penalty', 'EU_ETS', 'eu_pooling_price',
    ]
    df = df_in.copy()
    missing_cols = [c for c in required_cols if c not in df.columns]
    assert not missing_cols, (
        f"Missing required columns for {reg_name} in prepare_route_candidate_for_combo: {missing_cols}"
    )
    out = df[required_cols].copy()
    rename_map = {
        'fuel_id': f'{reg_name}_fuel_id',
        'fuel': f'fuel_{reg_name}',
        'fuel_consumption_mj_year': f'fuel_consumption_mj_year_{reg_name}',
        'total_consumption_tlsfo_eq': f'total_consumption_tlsfo_eq_{reg_name}',
        'total_fuel_cost': f'total_fuel_cost_{reg_name}',
        'eu_cost_usd': f'eu_cost_usd_{reg_name}',
        'reward_usd': f'reward_usd_{reg_name}',
        'total_wtw_emissions_tco2eq': f'total_wtw_emissions_tco2eq_{reg_name}',
        'total_ttw_emissions_tco2eq': f'total_ttw_emissions_tco2eq_{reg_name}',
    }
    return out.rename(columns=rename_map)

def create_combos_fuels_across_rcc(df_concat):
    """Create cross-route fuel combinations per tco_id.

    Input is a dict keyed by reg_reg_class with candidate tables as values.
    """

    assert isinstance(df_concat, dict), f"Expected dict, got {type(df_concat)}"
    assert len(df_concat) > 0, "Empty df_concat passed to create_combos_fuels_across_rcc"
    # Keep a deterministic merge order so downstream route columns are stable.
    reg_order = ['eu_to_eu', 'eu_to_noneu', 'noneu_to_noneu']
    available_regs = [r for r in reg_order if r in df_concat]
    assert available_regs, "No known reg_reg_class keys found"

    # Build a base table containing all tco_id values and shared metadata once.
    shared_cols = [
        'tco_id', 'year',
        'base_compliance_target', 'direct_compliance_target',
        'reward_threshold_gco2_mj',
        'ru2_price', 'ru1_price', 'su_price',
        'wacc', 'reward_delay',
        'FuelEU', 'FuelEU_penalty', 'EU_ETS', 'eu_pooling_price',
    ]
    df_shared = pd.concat(
        [df_concat[r][shared_cols] for r in available_regs],
        ignore_index=True,
    )
    combos = df_shared.drop_duplicates(subset=['tco_id'], keep='first').copy()

    # Build the cartesian product across available route classes per tco_id.
    # Missing route rows receive explicit zero placeholders.
    for reg_name in available_regs:
        df_reg = prepare_route_candidate_for_combo(df_concat[reg_name], reg_name)
        route_cols = [
            f'{reg_name}_fuel_id',
            f'fuel_{reg_name}',
            f'fuel_consumption_mj_year_{reg_name}',
            f'total_consumption_tlsfo_eq_{reg_name}',
            f'total_fuel_cost_{reg_name}',
            f'eu_cost_usd_{reg_name}',
            f'reward_usd_{reg_name}',
            f'total_wtw_emissions_tco2eq_{reg_name}',
            f'total_ttw_emissions_tco2eq_{reg_name}',
        ]
        combos = combos.merge(df_reg[['tco_id'] + route_cols], on='tco_id', how='left')

        # Placeholder IDs/labels preserve row coverage for tco_ids that do not use this route.
        combos[f'{reg_name}_fuel_id'] = combos[f'{reg_name}_fuel_id'].fillna(f'placeholder_{reg_name}')
        combos[f'fuel_{reg_name}'] = combos[f'fuel_{reg_name}'].fillna(f'placeholder_{reg_name}')
        numeric_route_cols = [
            f'fuel_consumption_mj_year_{reg_name}',
            f'total_consumption_tlsfo_eq_{reg_name}',
            f'total_fuel_cost_{reg_name}',
            f'eu_cost_usd_{reg_name}',
            f'reward_usd_{reg_name}',
            f'total_wtw_emissions_tco2eq_{reg_name}',
            f'total_ttw_emissions_tco2eq_{reg_name}',
        ]
        combos[numeric_route_cols] = combos[numeric_route_cols].apply(
            pd.to_numeric,
            errors='coerce',
        ).fillna(0.0)

    assert combos is not None and len(combos) > 0, "No combinations built across reg_reg_class"
    
    combos = aggregate_route_level_quantities(combos, available_regs)

    return combos

def run_vessel_scenario(
    start_year, num_years, user_params, df_vessel_params,
    engines_design_mapping, engine_fuels_dict, df_drop_in_fuels_ref, df_energy_density_ref,
    df_imo_targets_data, df_fueleu_targets_data, df_wtt_carbon_intensity_ref, df_ttw_carbon_intensity_ref,
    df_reward_ref, df_tc_rates_ref, df_pilot_fuel_input_ref, ru_price_data, initial_su_price_data,
    fuel_price_data, df_capex_ref, df_retrofit_cost_ref,
    df_opex_conventional_ref, df_opex_nonconv_wages_ref, df_opex_nonconv_insurance_ref,
    df_opex_nonconv_ssf_ref, df_opex_nonconv_isf_ref, df_opex_nonconv_spares_ref,
    df_operational_data_ref,
    game_year=None, *, df_speed_ref, df_efficiency_ref, df_engine_efficiency_ref,
):
    """Evaluate grouped scenarios through a dataframe-first batch API."""
    assert len(df_vessel_params[df_vessel_params['engine_type'].isna()]) == 0, f"Nan in the engine types: {df_vessel_params[df_vessel_params['engine_type'].isna()]}"

    assert isinstance(df_vessel_params, pd.DataFrame), (
        f"Expected df_vessel_params DataFrame, got {type(df_vessel_params)}"
    )
    assert len(df_vessel_params) > 0, "df_vessel_params is empty in run_vessel_scenario"

    required_cols = ['imo_type', 'size_id', 'action', 'engine', 'engine_type', 'year_built', 'prev_vessel_year_built', 'prev_engine']
    missing_cols = [c for c in required_cols if c not in df_vessel_params.columns]
    assert not missing_cols, (
        f"Missing required scenario columns in df_vessel_params: {missing_cols}"
    )

    df_batch = df_vessel_params.copy()

    # Remove stale result columns from previous model iterations, to avoid erroneously carrying on ss values.
    # Keep true scenario inputs (including su_price when injected by SU loop).
    non_input_passthrough_cols = {
        'dcf_id', 'tco_id', 'vessel_age',
        'vessel_capex', 'retrofit_cost_capex', 'vessel_yearly_depreciation',
        'opportunity_cost', 'capex_annualized',
        'annual_opex_excl_fuel_policy', 'annual_survey_cost',
        'annual_crew_cost', 'annual_technician_cost',
        'annual_management_cost', 'annual_insurance_cost', 'annual_spares_cost',
        'fuel', 'pilot_fuel', 'fuel_1', 'fuel_2', 'fuel_3',
        'share_fuel_1_final', 'share_fuel_2_final', 'share_fuel_3_final',
        'total_fuel_cost', 'net_policy_cost', 'total_fuel_policy_cost',
        'fuel_cost_lsfo', 'net_policy_cost_lsfo','su_tco2', 'ru_super_emitter_no_biofuel_tco2', 'ru_tco2',
        'reward_usd', 'reward_usd_tlsfoeq', 'pooling_value','eu_ets_usd', 'eu_compliance_cost_usd',
        'total_wtw_emissions_tco2eq', 'total_ttw_emissions_tco2eq',
        'overall_emissions_factor_wtw_gco2eq_mj', 'overall_emissions_factor_ttw_gco2eq_mj',
        'base_compliance_target', 'direct_compliance_target',
        'FuelEU', 'FuelEU_penalty', 'EU_ETS','total_annual_opex', 'annual_revenue',
        'dcf_planning_horizon', 'capex_annualized_planning_horizon','annual_opex_excl_fuel_policy_planning_horizon',
        'total_fuel_cost_planning_horizon','net_policy_cost_planning_horizon',
        'stranded_value', 'prev_vessel_residual_value_usd',
        'su_loop_iteration', 'design_chosen', 'su_chosen', 'switched_to_balance_the_su_market',
        'prev_vessel_age', 'tc_usd_per_year',
    }
    stale_cols = [c for c in non_input_passthrough_cols if c in df_batch.columns]
    if stale_cols:
        df_batch = df_batch.drop(columns=stale_cols)

    stale_prefixes = ('pv_',)
    stale_suffixes = ('_planning_horizon',)
    stale_pattern_cols = [
        c for c in df_batch.columns
        if c.startswith(stale_prefixes) or c.endswith(stale_suffixes)
    ]
    if stale_pattern_cols:
        df_batch = df_batch.drop(columns=stale_pattern_cols)

    df_batch['dcf_id'] = [f"dcf_{uuid.uuid4()}" for _ in range(len(df_batch))]
    df_batch['wacc'] = user_params['low_wacc']
    mask_high_wacc = df_batch['engine'].isin(HIGH_WACC_ENGINES)
    df_batch.loc[mask_high_wacc, 'wacc'] = user_params['high_wacc']

    # Apply energy-efficiency / speed-reduction adjustment to fuel consumption reference.
    # Returns a new copy so that repeated calls within the same game year (SU market
    # iterations) all see the same adjusted values without double-counting.
    df_operational_data_ref, df_batch = apply_fuel_efficiency_adjustment(
        df_batch, df_operational_data_ref, game_year, df_speed_ref, df_efficiency_ref
    )

    # Correct fuel consumption for the chosen engine's brake-thermal efficiency (BTE).
    df_batch = apply_engine_efficiency_adjustment(df_batch, df_engine_efficiency_ref)

    # Now choose the least cost fuel and calculate voyage costs
    df_rrc_results = run_multi_year_analysis(
        start_year, num_years, user_params, df_batch,
        engines_design_mapping, engine_fuels_dict, df_drop_in_fuels_ref, df_energy_density_ref,
        df_imo_targets_data, df_fueleu_targets_data, df_wtt_carbon_intensity_ref, df_ttw_carbon_intensity_ref,
        df_reward_ref, df_tc_rates_ref, df_pilot_fuel_input_ref, ru_price_data, initial_su_price_data,
        fuel_price_data, df_operational_data_ref,
    )

    # Aggregate over reg_reg_class per scenario-year to compute planning-horizon DCF inputs.
    df_results_per_planning_year = (
        df_rrc_results.groupby(['dcf_id', 'year', 'planning_year'], as_index=False)
        .agg({
            'su_price': 'first', 
            'pooling_value': 'first',
            'direct_compliance_target': 'first',
            'fuel_consumption_mj_year': 'sum', 
            'su_tco2': 'sum',
            'ru_super_emitter_no_biofuel_tco2': 'sum', 
            'reward_usd': 'sum',
            'total_fuel_cost': 'sum',
            'net_policy_cost': 'sum',
        })
    )

    # add back the columns which have not changed by choosing a fuel
    check_len = len(df_results_per_planning_year)
    cols_to_drop = ['su_price', 'fuel_consumption_mj_year']
    for col in cols_to_drop:
        if col in df_batch.columns:
            df_batch = df_batch.drop(columns = col)

    df_results_per_planning_year = df_results_per_planning_year.merge(
        df_batch,
        on = 'dcf_id',
        how = 'left'
    )
    assert check_len == len(df_results_per_planning_year), f"Mismatch in the columns numbers {check_len} vs {len(df_results_per_planning_year)}"
    for col in ['imo_type','engine','engine_type']:
        assert len(df_results_per_planning_year[df_results_per_planning_year[col].isna()]) == 0, f"nans after merge: {df_results_per_planning_year[df_results_per_planning_year[col].isna()]}"

    # Add costs which are constant between years and voyages: capex, non-fuel opex
    df_static_costs = compute_static_cost_inputs(
        df_scenarios=df_batch,
        start_year=start_year,
        df_capex_ref=df_capex_ref,
        df_retrofit_cost_ref=df_retrofit_cost_ref,
        df_opex_conventional_ref=df_opex_conventional_ref,
        df_opex_nonconv_wages_ref=df_opex_nonconv_wages_ref,
        df_opex_nonconv_insurance_ref=df_opex_nonconv_insurance_ref,
        df_opex_nonconv_ssf_ref=df_opex_nonconv_ssf_ref,
        df_opex_nonconv_isf_ref=df_opex_nonconv_isf_ref,
        df_opex_nonconv_spares_ref=df_opex_nonconv_spares_ref,
    )

    expected_rows = len(df_results_per_planning_year)
    df_results_per_planning_year = df_results_per_planning_year.merge(df_static_costs, on='dcf_id', how='left')
    assert len(df_results_per_planning_year) == expected_rows, (
        f"Row mismatch after static-cost merge: expected {expected_rows}, got {len(df_results_per_planning_year)}"
    )
    required_static_cols = [
        'annual_opex_excl_fuel_policy',
        'capex_annualized',
        'opportunity_cost',
        'annual_crew_cost',
        'annual_technician_cost',
        'annual_management_cost',
        'annual_insurance_cost',
        'annual_spares_cost',
        'vessel_capex',
        'retrofit_cost_capex',
        'annual_survey_cost',
        'vessel_age',
        'prev_vessel_age',
        'action',
        'engine',
        'engine_type',
        'su_price',
        'dcf_id',
    ]
    for col in required_static_cols:
        assert df_results_per_planning_year[col].notna().all(), (
            f"{col} contains NaN after static-cost merge"
        )

    years_array = df_results_per_planning_year['planning_year'].values - 1
    discount_factors = 1.0 / ((1.0 + df_results_per_planning_year['wacc'].values) ** years_array)
    pv_input_cols = [
        'annual_opex_excl_fuel_policy', 'capex_annualized', 'total_fuel_cost', 'net_policy_cost', 
        # 'annual_revenue'
    ]
    for column in pv_input_cols:
        df_results_per_planning_year[f'pv_{column}'] = df_results_per_planning_year[column] * discount_factors

    df_results_per_planning_year['dcf_planning_horizon'] = (
        - df_results_per_planning_year['pv_capex_annualized']
        - df_results_per_planning_year['pv_annual_opex_excl_fuel_policy']
        - df_results_per_planning_year['pv_total_fuel_cost']
        - df_results_per_planning_year['pv_net_policy_cost']
    )
    
    df_results = (
        df_results_per_planning_year.sort_values(['dcf_id', 'planning_year'])
        .groupby('dcf_id', as_index=False)
        .agg({
            'imo_type': 'first', 
            'size_id': 'first', 
            'real_ships_nb': 'first', 
            'modelled_ships_nb': 'first',
            'dwt': 'first', 'mcr': 'first', 
            'bunker_t_vlsfo_e': 'first', 
            'n_stroke': 'first',
            'initial_fuel_design': 'first', 
            'tc_usd_per_day': 'first', 
            'wind': 'first', 'ccs': 'first',
            'prev_vessel_year_built': 'first', 
            'segment_size_id': 'first', 
            'lifetime': 'first',
            'usd_per_dwt': 'first', 
            'prev_engine': 'first', 
            'prev_capex_annualized': 'first',
            'scenario': 'first', 
            'choice_id': 'first', 
            'action': 'first', 
            'engine': 'first',
            'engine_type': 'first', 
            'engine_type_label': 'first', 
            'year_built': 'first',
            'opportunity_cost': 'first', 
            'tc_usd_per_year': 'first', 
            'transport_work_dwt_nm':'first',
            'vessel_age': 'first',
            'prev_vessel_age': 'first',
            'prev_vessel_capex': 'first', 
            'vessel_capex': 'first',
            'retrofit_cost_capex': 'first',
            'annual_survey_cost': 'first',
            'salvage_value_usd': 'first', 
            'su_price': 'first', 
            'pooling_value': 'first',
            'direct_compliance_target': 'first',
            # technically this the few next rows are not correct and only for the first year, rather than over planning horizon, 
            # but this is needed for the SU market and plotting
            'su_tco2': 'first', 
            'ru_super_emitter_no_biofuel_tco2': 'first', 
            'reward_usd': 'first',
            'fuel_consumption_mj_year': 'first',
            'annual_crew_cost': 'first',
            'annual_technician_cost': 'first',
            'annual_management_cost': 'first',
            'annual_insurance_cost': 'first',
            'annual_spares_cost': 'first',
            'capex_annualized': 'first', 
            'annual_opex_excl_fuel_policy': 'first',
            'total_fuel_cost': 'first', 
            'net_policy_cost': 'first',
            # those columns are properly calculated
            'dcf_planning_horizon': 'sum', 
            'pv_capex_annualized': 'sum', 
            'pv_annual_opex_excl_fuel_policy': 'sum',
            'pv_total_fuel_cost': 'sum', 
            'pv_net_policy_cost': 'sum',
        })
    )
    df_results = df_results.rename(columns={
        'dcf': 'dcf_planning_horizon',
        'pv_capex_annualized': 'capex_annualized_planning_horizon',
        'pv_annual_opex_excl_fuel_policy': 'annual_opex_excl_fuel_policy_planning_horizon',
        'pv_total_fuel_cost': 'total_fuel_cost_planning_horizon',
        'pv_net_policy_cost': 'net_policy_cost_planning_horizon',
    })

    df_results['prev_vessel_residual_value_usd'] = (
        df_results['salvage_value_usd']
        + (df_results['prev_vessel_capex'] - df_results['salvage_value_usd']) 
        * (1 - (df_results['prev_vessel_age'])/ df_results['lifetime'])
    )

    mask_newbuild = df_results['action'] == 'newbuild'
    dcf_adjustment = np.where(
        mask_newbuild,
        df_results['salvage_value_usd'] - df_results['prev_vessel_residual_value_usd'],
        0.0,
    )
    df_results['dcf_planning_horizon'] = df_results['dcf_planning_horizon'] + dcf_adjustment
    df_results['stranded_value'] = np.where(
        mask_newbuild,
        np.maximum(
            df_results['prev_vessel_residual_value_usd'] - df_results['salvage_value_usd'],
            0.0,
        ),
        0.0,
    )

    assert len(df_results) == len(df_vessel_params), (
        f"Batch DCF rows mismatch: expected {len(df_vessel_params)}, got {len(df_results)}"
    )
    assert len(df_rrc_results) > 0, "No reg_reg_class rows returned in batch mode"

    return df_results, df_rrc_results

def run_multi_year_analysis(
    start_year, num_years, user_params, df_vessel_params,
    engines_design_mapping, engine_fuels_dict, df_drop_in_fuels_ref, df_energy_density_ref,
    df_imo_targets_data, df_fueleu_targets_data, df_wtt_carbon_intensity_ref, df_ttw_carbon_intensity_ref,
    df_reward_ref, df_tc_rates_ref, df_pilot_fuel_input_ref, ru_price_data, initial_su_price_data,
    fuel_price_data, df_operational_data_ref, 
):
    """Run TCO over planning years for a batch of scenarios."""
    assert len(df_vessel_params[df_vessel_params['engine_type'].isna()]) == 0, f"Nan in the engine types: {df_vessel_params[df_vessel_params['engine_type'].isna()]}"

    df_year_results = []
    for year_offset in range(num_years):
        current_year = start_year + year_offset
        df_year = df_vessel_params.copy()
        df_year['planning_year'] = year_offset + 1
        df_year['year'] = current_year
        df_year['tco_id'] = df_year['dcf_id'].astype(str) + f"_py{year_offset + 1}"

        df_year_result_per_rrc = run_tco_analysis(
            current_year, user_params, df_year,
            engines_design_mapping, engine_fuels_dict, df_drop_in_fuels_ref, df_energy_density_ref,
            df_imo_targets_data, df_fueleu_targets_data, df_wtt_carbon_intensity_ref, df_ttw_carbon_intensity_ref,
            df_reward_ref, ru_price_data, initial_su_price_data,
            fuel_price_data, df_operational_data_ref, 
            # df_tc_rates_ref, 
            df_pilot_fuel_input_ref, 
        )
        df_year_results.append(df_year_result_per_rrc)

    df_results_by_rrc = pd.concat(df_year_results, ignore_index=True)
    assert len(df_results_by_rrc) > 0, "No rows returned from run_multi_year_analysis"
    return df_results_by_rrc


def run_tco_analysis(
    current_year, user_params, df_vessel_params,
    engines_design_mapping, engine_fuels_dict, df_drop_in_fuels_ref, df_energy_density_ref,
    df_imo_targets_data, df_fueleu_targets_data, df_wtt_carbon_intensity_ref, df_ttw_carbon_intensity_ref,
    df_reward_ref, ru_price_data, initial_su_price_data,
    fuel_price_data, df_operational_data_ref, 
    # df_tc_rates_ref, 
    df_pilot_fuel_input_ref, 
):
    """TCO analysis without per-row looping."""

    assert isinstance(df_vessel_params, pd.DataFrame), (
        f"Expected df_vessel_params DataFrame, got {type(df_vessel_params)}"
    )
    assert len(df_vessel_params) > 0, "run_tco_analysis received empty df_vessel_params"
    assert len(df_vessel_params[df_vessel_params['engine_type'].isna()]) == 0, f"Nan in the engine types: {df_vessel_params[df_vessel_params['engine_type'].isna()]}"

    df_params = calculate_initial_values(
        df_vessel_params=df_vessel_params,
        current_year=current_year,
        user_params=user_params,
        df_imo_targets_data=df_imo_targets_data,
        df_fueleu_targets_data=df_fueleu_targets_data,
        ru_price_data=ru_price_data,
        initial_su_price_data=initial_su_price_data,
    )
    assert len(df_params[df_params['engine_type'].isna()]) == 0, f"Nan in the engine types: {df_params[df_params['engine_type'].isna()]}"

    df_rrc = calc_per_reg_reg_class(
        df_params=df_params,
        df_operational_data_ref=df_operational_data_ref,
        engines_design_mapping=engines_design_mapping,
        engine_fuels_dict=engine_fuels_dict,
        df_drop_in_fuels_ref=df_drop_in_fuels_ref,
        fuel_price_data=fuel_price_data,
        df_wtt_carbon_intensity_ref=df_wtt_carbon_intensity_ref,
        df_ttw_carbon_intensity_ref=df_ttw_carbon_intensity_ref,
        df_reward_ref=df_reward_ref,
        df_pilot_fuel_input_ref=df_pilot_fuel_input_ref,
        reward_fuels = user_params['reward_fuels'],
        allowed_fuels = user_params['allowed_fuels']
    )

    assert len(df_rrc) > 0, "run_tco_analysis returned empty dataframe"
    return df_rrc

def calc_per_reg_reg_class(
    df_params, df_operational_data_ref, engines_design_mapping, engine_fuels_dict,
    df_drop_in_fuels_ref, 
    fuel_price_data, df_wtt_carbon_intensity_ref, df_ttw_carbon_intensity_ref, df_reward_ref, df_pilot_fuel_input_ref,
    allowed_fuels, reward_fuels,
):
    """
    Simultaneous fuel optimization across all reg_reg_class routes.
    In 4 stpes:
    1. We build a dataframe with all fuels possibles within each reg_reg_class independently.
    That includes "unique" fuels (e.g. lsfo, e-ammonia+pilot), fuel mix on the fuel EU, and fuel mix on the base GFI (carbon intensity = base GFI)
    2. We build all the combos of fuels which across all reg-reg-class meets the base GFI, with the fuel mix adjustement happening in noneu_to_noneu
    3. Select only the cheapest combo
    """

    # Step 0: prepare route shares per ship-year and normalize to a valid partition.
    reg_reg_classes = ['noneu_to_noneu', 'eu_to_noneu', 'eu_to_eu']
    reg_configs = [
        ('eu_to_eu', 'fueleu_imo'),
        ('eu_to_noneu', 'fueleu_imo'),
        ('noneu_to_noneu', 'imo'),
    ]
    consumption_cols = ['fuel_consumption_tj_year', 'fuel_consumption_mj_year']

    df_eu_props = (
        df_operational_data_ref[['imo_type', 'size_id', 'eu_to_eu', 'eu_to_noneu', 'noneu_to_noneu']]
        .drop_duplicates(subset=['imo_type', 'size_id'])
        .copy()
    )

    expected_rows = len(df_params)
    df_work = df_params.merge(df_eu_props, on=['imo_type', 'size_id'], how='left')
    assert len(df_work[df_work['engine_type'].isna()]) == 0, f"Nan in the engine types: {df_work[df_work['engine_type'].isna()]}"
    assert len(df_work) == expected_rows, (
        f"Row count mismatch in reg_prop merge: expected {expected_rows}, got {len(df_work)}"
    )
    for col in reg_reg_classes:
        assert df_work[col].notna().all(), (
            f"NaN values in {col} after reg_prop merge. Missing imo_type/size_id mapping?"
        )

    mask_no_eu = df_work['year'] >= df_work['year_eu_policy_dropped']
    df_work.loc[mask_no_eu, 'noneu_to_noneu'] = 1.0
    df_work.loc[mask_no_eu, 'eu_to_noneu'] = 0.0
    df_work.loc[mask_no_eu, 'eu_to_eu'] = 0.0

    row_sums = df_work[reg_reg_classes].sum(axis=1)
    assert (row_sums > 0).all(), "Reg proportions sum to zero for at least one row"
    for col in reg_reg_classes:
        df_work[col] = df_work[col] / row_sums

    # Step 1: build candidate fuels independently for each route class.
    # The output is a dict of route-level candidate tables keyed by reg_reg_class.
    df_all_fuels_concat = {}
    for reg_reg_class, compliance_mode in reg_configs:
        mask_reg = df_work[reg_reg_class] > 0
        if not mask_reg.any():
            continue

        df_reg = df_work.loc[mask_reg].copy()
        df_reg['reg_reg_class'] = reg_reg_class
        df_reg['reg_reg_prop'] = df_reg[reg_reg_class]
        for col in consumption_cols:
            df_reg[col] = df_reg[col] * df_reg['reg_reg_prop']

        # List all the fuel options for the rrc and calc its costs, excl. NZF cost
        # In practice, that incudes fuel cost, EU cost and reward cost (we assume that fuels receive rewards on GJ basis; this is not coherent with the fuel mix on the base)
        df_all_fuels = build_all_fuel_options(
            df_params=df_reg,
            engine_fuels_dict=engine_fuels_dict,
            fuel_price_data=fuel_price_data,
            df_wtt_carbon_intensity_ref=df_wtt_carbon_intensity_ref,
            df_ttw_carbon_intensity_ref=df_ttw_carbon_intensity_ref,
            df_reward_ref=df_reward_ref,
            df_pilot_fuel_input_ref=df_pilot_fuel_input_ref,
            compliance_mode=compliance_mode,
            df_drop_in_fuels_ref=df_drop_in_fuels_ref,
            engines_design_mapping=engines_design_mapping,
            allowed_fuels=allowed_fuels,
            reward_fuels=reward_fuels,
        )

        # Keep Step-1 policy split explicit: preserve EU-side costs and defer annual policy.
        if 'eu_cost_usd' not in df_all_fuels.columns:
            df_all_fuels['eu_cost_usd'] = 0.0

        df_all_fuels['reg_reg_class'] = reg_reg_class
        df_all_fuels = df_all_fuels.reset_index(drop=True)
        df_all_fuels['fuel_id'] = 'fuel_' + pd.Series(
            [str(uuid.uuid4()) for _ in range(len(df_all_fuels))], index=df_all_fuels.index
        )
        assert len(df_all_fuels[df_all_fuels['engine_type'].isna()]) == 0, f"Nan in the engine types: {df_all_fuels[df_all_fuels['engine_type'].isna()]}"

        df_all_fuels_concat[reg_reg_class] = df_all_fuels

    assert len(df_all_fuels_concat) > 0, "No route candidate tables were built"

    # Now that we have individual fuel options for each rrc, we create all combinations. E.g. with 10 fuels, 10*10*10 combination
    df_all_combinations = create_combos_fuels_across_rcc(df_all_fuels_concat)

    # Step 2: Compute the fuel mix on the base over all rrc. derive corrected IMO target for noneu_to_noneu from EU pair choices.
    # For each EU combo, solve the residual WTW intensity needed on noneu_to_noneu
    # so that the full annual profile can still meet the base target.
    if (
        'eu_to_eu' in df_all_fuels_concat
        and 'eu_to_noneu' in df_all_fuels_concat
        and 'noneu_to_noneu' in df_all_fuels_concat
        and df_all_fuels_concat['noneu_to_noneu']['base_compliance_target'].iloc[0] < 100
        and df_all_fuels_concat['noneu_to_noneu']['base_compliance_target'].iloc[0] > 0
    ):
        # we need to build a fuel mix on the base for all possible combos on the EU waters
        df_eu_combos = create_combos_fuels_across_rcc({
            'eu_to_eu': df_all_fuels_concat['eu_to_eu'],
            'eu_to_noneu': df_all_fuels_concat['eu_to_noneu'],
        })

        # Compute the target GFI that the ship must meet in noneu_to_noneu for the average GFI to be = base GFI
        df_noneu_cons = (
            df_all_fuels_concat['noneu_to_noneu'][['tco_id', 'fuel_consumption_mj_year']]
            .drop_duplicates(subset=['tco_id'])
            .rename(columns={'fuel_consumption_mj_year': 'noneu_to_noneu_consumption_mj_year'})
        )
        check_len = len(df_eu_combos)
        df_eu_combos = df_eu_combos.merge(df_noneu_cons, on='tco_id', how='left')
        assert len(df_eu_combos[df_eu_combos.noneu_to_noneu_consumption_mj_year.isna()]) == 0
        assert len(df_eu_combos) == check_len

        eu_wtw_tco2 = (
            df_eu_combos['total_wtw_emissions_tco2eq_eu_to_eu']
            + df_eu_combos['total_wtw_emissions_tco2eq_eu_to_noneu']
        )
        total_required_wtw_g = (
            df_eu_combos['base_compliance_target']
            * (
                df_eu_combos['fuel_consumption_mj_year_eu_to_eu']
                + df_eu_combos['fuel_consumption_mj_year_eu_to_noneu']
                + df_eu_combos['noneu_to_noneu_consumption_mj_year']
            )
        )
        df_eu_combos['corrected_base_compliance_target'] = np.where(
            df_eu_combos['noneu_to_noneu_consumption_mj_year'] > 0,
            (total_required_wtw_g - eu_wtw_tco2 * 1_000_000)
            / df_eu_combos['noneu_to_noneu_consumption_mj_year'],
            np.nan,
        )
        # we only look for fuel mixes for a subset of combos
        df_eu_combos = df_eu_combos[(np.isfinite(df_eu_combos['corrected_base_compliance_target']))].copy()
        df_eu_combos = df_eu_combos[df_eu_combos['corrected_base_compliance_target']<100].copy()
        df_eu_combos = df_eu_combos[df_eu_combos['fuel_consumption_mj_year_eu_to_eu'] + df_eu_combos['fuel_consumption_mj_year_eu_to_noneu']>0].copy()

        # Deduplicate near-identical targets by keeping the cheapest EU-side combo
        # this is because later on, we use corrected_base_compliance_target for merging, so we need uniqueness
        df_eu_combos['fuel_eu_costs'] = (
            df_eu_combos['total_fuel_cost'] + df_eu_combos['eu_cost_usd']
        )
        df_eu_combos['corrected_target_key'] = df_eu_combos['corrected_base_compliance_target']
        df_eu_combos = (
            df_eu_combos
            .sort_values('fuel_eu_costs')
            .drop_duplicates(subset=['tco_id', 'corrected_target_key'], keep='first')
        )
        df_all_fuels_sub = df_all_fuels_concat['noneu_to_noneu'].merge(
            df_eu_combos[['tco_id', 'corrected_base_compliance_target']],
            on='tco_id',
            how='inner',
        )
        
        # Run targeted noneu_to_noneu mix search only for feasible corrected targets.
        assert len(df_all_fuels_sub) > 0, "df_all_fuels_sub is empty"
        df_mix = calc_fuel_mix_on_the_base(
            df_all_fuels=df_all_fuels_sub,
            df_drop_in_fuels_ref=df_drop_in_fuels_ref,
            df_pilot_fuel_input_ref=df_pilot_fuel_input_ref,
            engines_design_mapping=engines_design_mapping,
            compliance_mode='jointly_imo',
        )
        if len(df_mix) > 0:
            df_mix = df_mix[df_mix.total_fuel_cost.notna()].copy()
            if len(df_mix) > 0:
                df_mix['reg_reg_class'] = 'noneu_to_noneu'
                df_mix['eu_cost_usd'] = 0.0
                df_mix['fuel_id'] = 'fuel_' + pd.Series(
                    [str(uuid.uuid4()) for _ in range(len(df_mix))], index=df_mix.index
                )

                # Append the new fuels to the dictionary with all the fuels
                df_noneu_augmented = pd.concat(
                    [df_all_fuels_concat['noneu_to_noneu'], df_mix],
                    ignore_index=True,
                    sort=False,
                )
                df_all_fuels_concat['noneu_to_noneu'] = df_noneu_augmented

                # We add the new combos created (all meeting the base GFI) to the list of combos
                df_mix_for_combo = prepare_route_candidate_for_combo(df_mix, 'noneu_to_noneu')
                df_mix_for_combo['corrected_base_compliance_target'] = df_mix['corrected_base_compliance_target'].values
                new_combos = df_eu_combos.merge(
                    df_mix_for_combo[[
                        'noneu_to_noneu_fuel_id',
                        'fuel_noneu_to_noneu',
                        'fuel_consumption_mj_year_noneu_to_noneu',
                        'total_consumption_tlsfo_eq_noneu_to_noneu',
                        'total_fuel_cost_noneu_to_noneu',
                        'eu_cost_usd_noneu_to_noneu',
                        'reward_usd_noneu_to_noneu',
                        'total_wtw_emissions_tco2eq_noneu_to_noneu',
                        'total_ttw_emissions_tco2eq_noneu_to_noneu',
                        'tco_id',
                        'corrected_base_compliance_target',
                    ]],
                    on=['tco_id', 'corrected_base_compliance_target'],
                    how='inner',
                )
                new_combos = aggregate_route_level_quantities(
                    new_combos,
                    ['eu_to_eu', 'eu_to_noneu', 'noneu_to_noneu'],
                )

                # Step 2 validation: corrected-target combinations should land
                # on the base GFI when recombined across all routes.
                gfi_tolerance = 1e-3
                gfi_abs_diff = np.abs(
                    new_combos['overall_emissions_factor_wtw_gco2eq_mj']
                    - new_combos['base_compliance_target']
                )
                assert (gfi_abs_diff <= gfi_tolerance).all(), (
                    "Step 2 GFI check failed: combined attained-GFI is not within "
                    f"{gfi_tolerance} of base_compliance_target. "
                    f"{new_combos[new_combos.overall_emissions_factor_wtw_gco2eq_mj<new_combos.base_compliance_target]}"
                )
                df_all_combinations = pd.concat((df_all_combinations, new_combos), ignore_index = True)
            else:
                print(
                    "No fuel-mix rows generated for Step 2 jointly_imo after filtering "
                    f"missing total_fuel_cost, year={df_all_fuels_sub['year'].drop_duplicates().tolist()}."
                )
        else:
            print(
                f"No fuel-mix rows generated for Step 2 jointly_imo, "
                f"year={df_all_fuels_sub['year'].drop_duplicates().tolist()}."
            )

    # Step 3: evaluate annual policy on full combinations and select the cheapest total.
    # compute the NZF costs only - the EU and reward are calculated for each rrc
    policy = calculate_policy_compliance(
        df_all_combinations,
        df_all_combinations['total_consumption_tlsfo_eq'].values,
        df_all_combinations['overall_emissions_factor_wtw_gco2eq_mj'].values,
        # df_all_combinations['total_wtw_emissions_tco2eq'].values,
        df_all_combinations['total_ttw_emissions_tco2eq'].values,
        df_all_combinations['overall_emissions_factor_wtw_gco2eq_mj'].values,
        'imo',
    )

    for k, v in policy.items():
        if k not in [
            'FuelEU_penalty', 'EU_ETS', 'pooling_value',
            'fueleu_compliance_cost_usd_p_gj', 'eu_pooling_revenue_usd_p_gj',
            'eu_compliance_cost_usd', 'eu_ets_usd',
            'reward_usd', 'reward_usd_tlsfoeq',
        ]:
            df_all_combinations[k] = v

    # Select the cheapest combo
    df_all_combinations['net_policy_cost'] = (
        df_all_combinations['total_compliance_cost_usd']
        - df_all_combinations['reward_usd']
        + df_all_combinations['eu_cost_usd']
    )
    df_all_combinations['total_fuel_policy_cost'] = (
        df_all_combinations['total_fuel_cost'] + df_all_combinations['net_policy_cost']
    )

    # Run some checks
    assert len(df_all_combinations) > 0, "df_all_combinations is empty before selection"
    numeric_check_cols = [
        'fuel_consumption_mj_year',
        'net_policy_cost',
        'total_fuel_policy_cost',
    ]
    for c in numeric_check_cols:
        coerced = pd.to_numeric(df_all_combinations[c], errors='coerce')
        invalid_mask = coerced.isna() | ~np.isfinite(coerced.to_numpy())
        assert (~invalid_mask).all(), (
            f"Invalid numeric values in {c} in df_all_combinations. "
            f"failures={df_all_combinations.loc[invalid_mask]}"
        )
        df_all_combinations[c] = coerced

    df_all_combinations_selected = (
        df_all_combinations.sort_values('total_fuel_policy_cost')
        .drop_duplicates(subset=['tco_id'], keep='first')
    )
    assert len(df_all_combinations_selected) > 0, "No feasible rows in df_all_combinations_selected"
    assert df_all_combinations_selected['tco_id'].is_unique, (
        "df_all_combinations_selected should have a single row per tco_id"
    )
    selected_missing_ids = pd.Index(df_work['tco_id'].drop_duplicates()).difference(
        pd.Index(df_all_combinations_selected['tco_id'].drop_duplicates())
    )

    assert selected_missing_ids.empty, (
        "tco_ids missing from df_all_combinations_selected: "
        f"{selected_missing_ids.tolist()}"
    )

    # the NZF costs is reapproportioned to individual rrc
    # this is done on an energy basis
    combo_policy_cols = [
        'su_tco2', 'ru_tco2', 'ru_small_emitter_no_biofuel','ru_super_emitter_no_biofuel_tco2', 
        'su_generated', 'su_revenue_usd_tvslfoeq', 'ru1_cost_usd_tvslfoeq', 
        'ru2_cost_usd_tvslfoeq', 'total_compliance_cost_usd_tvslfoeq', 
        'total_compliance_cost_usd', 'net_policy_cost',
        'fuel_intensity_below_direct_tco2eq_tlsfoeq',
    ]
    assert all(c in df_all_combinations_selected.columns for c in combo_policy_cols)
    combo_policy_rename = {c: f'{c}_combo' for c in combo_policy_cols}
    df_combo_alloc = df_all_combinations_selected[
        ['tco_id', 'fuel_consumption_mj_year'] + combo_policy_cols
    ].rename(columns={
        'fuel_consumption_mj_year': 'fuel_consumption_mj_year_combo',
        **combo_policy_rename,
    })

    # Reconstruct one selected row per route using route-specific fuel_id columns.
    rrc_frames = []
    for reg_reg_class, _ in reg_configs:
        if reg_reg_class not in df_all_fuels_concat:
            continue
        fuel_id_col = f'{reg_reg_class}_fuel_id'
        assert fuel_id_col in df_all_combinations_selected.columns, f"{fuel_id_col} not in f_all_combinations_selected.columns"

        selected_ids = df_all_combinations_selected[['tco_id', fuel_id_col]].rename(
            columns={fuel_id_col: 'fuel_id'}
        )
        df_rrc = df_all_fuels_concat[reg_reg_class].merge(
            selected_ids,
            on=['tco_id', 'fuel_id'],
            how='inner',
        )

        df_rrc = df_rrc.merge(df_combo_alloc, on='tco_id', how='left')
        energy_share = np.where(
            df_rrc['fuel_consumption_mj_year_combo'] > 0,
            df_rrc['fuel_consumption_mj_year'] / df_rrc['fuel_consumption_mj_year_combo'],
            0.0,
        )
        for c in combo_policy_cols:
            combo_c = f'{c}_combo'
            if combo_c in df_rrc.columns:
                df_rrc[c] = df_rrc[combo_c].fillna(0.0) * energy_share
                df_rrc = df_rrc.drop(columns=[combo_c])
            else:
                df_rrc[c] = 0.0
        df_rrc = df_rrc.drop(columns=['fuel_consumption_mj_year_combo'], errors='ignore')

        rrc_frames.append(df_rrc)

    assert len(rrc_frames) > 0, "No reg_reg_class frames produced"

    # Sanity check: all tco_ids from df_work must appear in the reconstructed frames,
    # and the summed fuel consumption across rrc must match the original totals.
    df_result = pd.concat(rrc_frames, ignore_index=True)
    result_cons = df_result.groupby('tco_id')['fuel_consumption_mj_year'].sum().rename('result_mj')
    work_cons = df_work.set_index('tco_id')['fuel_consumption_mj_year'].rename('work_mj')
    missing_ids = work_cons.index.difference(result_cons.index)
    assert missing_ids.empty, f"tco_ids missing from reconstructed frames: {missing_ids.tolist()}"
    cons_check = work_cons.to_frame().join(result_cons, how='left')
    cons_diff = (cons_check['result_mj'] - cons_check['work_mj']).abs()
    cons_tol = 1e-3
    assert (cons_diff <= cons_tol).all(), (
        "Fuel consumption mismatch between df_work and reconstructed rrc frames. "
        f"max_abs_diff={cons_diff.max():.6f}, failures={(cons_diff > cons_tol).sum()}"
    )

    # Policy allocation sanity: route-level allocated values should sum back
    # to the selected combo-level totals per tco_id.
    policy_cols = ['net_policy_cost', 'su_tco2', 'ru_super_emitter_no_biofuel_tco2', 'ru_tco2']
    df_policy_check = (
        df_result.groupby('tco_id')[policy_cols]
        .sum()
        .join(df_all_combinations_selected.set_index('tco_id')[policy_cols], rsuffix='_combo', how='inner')
    )
    for c in policy_cols:
        diff = (df_policy_check[c] - df_policy_check[f'{c}_combo']).abs()
        assert (diff <= cons_tol).all(), (
            f"Policy allocation mismatch for {c}: "
            f"max_abs_diff={diff.max():.6f}, failures={(diff > cons_tol).sum()}"
        )
    assert len(df_result[df_result['engine_type'].isna()]) == 0, f"Nan in the engine types: {df_result[df_result['engine_type'].isna()]}"

    return df_result


# --- CAPEX ---
def process_capex_retrofit_refs(user_params, df_capex_ref, df_retrofit_cost_ref, df_capex_reduction, end_year, planning_horizon_years):
    df_capex_ref = df_capex_ref.copy()
    df_retrofit_cost_ref = df_retrofit_cost_ref.copy()

    # Filter by scenario
    df_retrofit_cost_ref = df_retrofit_cost_ref[df_retrofit_cost_ref['scenario'] == user_params['capex_opex_scenario']]
    df_capex_ref = df_capex_ref.loc[
        (df_capex_ref['scenario'] == user_params['capex_opex_scenario']) &
        (~df_capex_ref['unit'].str.contains('premium', case=False, na=False))
    ]

    # Calculate capex premiums
    df_capex_premiums_ref = calc_capex_premiums(df_capex_ref)

    # Apply capex reductions
    capex_reduction_scenario = user_params['capex_reduction_scenario']
    df_capex_ref, df_retrofit_cost_ref = capex_retrofit_cost_reductions(
        capex_reduction_scenario,
        df_capex_premiums_ref,
        df_capex_ref,
        df_capex_reduction,
        df_retrofit_cost_ref
    )

    # Extend both dfs to cover all years needed
    max_year_needed = end_year + planning_horizon_years
    df_capex_ref = extend_years(df_capex_ref, max_year_needed)
    df_retrofit_cost_ref = extend_years(df_retrofit_cost_ref, max_year_needed)

    return df_capex_ref, df_retrofit_cost_ref

def annualise_capex(capex, wacc, lifetime):
    capital_recovery_factor = wacc *(1+wacc)**lifetime/((1+wacc)**lifetime - 1)
    annualised_capex = capital_recovery_factor * capex
    return annualised_capex


def compute_static_cost_inputs(
    df_scenarios,
    start_year,
    df_capex_ref,
    df_retrofit_cost_ref,
    df_opex_conventional_ref,
    df_opex_nonconv_wages_ref,
    df_opex_nonconv_insurance_ref,
    df_opex_nonconv_ssf_ref,
    df_opex_nonconv_isf_ref,
    df_opex_nonconv_spares_ref,
):
    """static CAPEX/non-fuel OPEX assignment for batch scenarios."""

    assert isinstance(df_scenarios, pd.DataFrame), f"Expected DataFrame, got {type(df_scenarios)}"
    assert len(df_scenarios) > 0, "compute_static_cost_inputs received empty df_scenarios"
    required_cols = [
        'dcf_id', 'imo_type', 'size_id', 'engine', 'prev_engine', 'action',
        'year_built', 'prev_vessel_year_built', 'lifetime', 'wacc',
        'prev_vessel_capex', 'prev_capex_annualized','tc_usd_per_day',
    ]
    missing_cols = [c for c in required_cols if c not in df_scenarios.columns]
    assert not missing_cols, f"Missing required columns in compute_static_cost_inputs: {missing_cols}"

    df_static = df_scenarios[required_cols + ['salvage_value_usd']].copy()
    df_static['base_engine'] = df_static['engine'].astype(str).str.split('_+_', regex=False).str[0]
    df_static['prev_base_engine'] = df_static['prev_engine'].astype(str).str.split('_+_', regex=False).str[0]
    df_static['vessel_age'] = start_year - df_static['year_built']
    df_static['prev_vessel_age'] = start_year - df_static['prev_vessel_year_built']

    # ---- CAPEX inputs ----
    df_static['retrofit_cost_capex'] = 0.0
    df_static['opportunity_cost'] = 0.0
    df_static['vessel_capex'] = df_static['prev_vessel_capex']

    mask_keep = df_static['action'] == 'keep'
    mask_newbuild = df_static['action'] == 'newbuild'
    mask_retrofit = df_static['action'] == 'retrofit'

    if mask_newbuild.any():
        df_capex_year = df_capex_ref[df_capex_ref['year'] == start_year].copy()
        capex_base_cols = {'year', 'imo_type', 'size_id', 'scenario', 'unit', 'min_size', 'max_size'}
        engine_cols = [c for c in df_capex_year.columns if c not in capex_base_cols]

        newbuild_engines = sorted(df_static.loc[mask_newbuild, 'engine'].dropna().unique().tolist())
        missing_engine_cols = [e for e in newbuild_engines if e not in engine_cols]
        assert not missing_engine_cols, (
            f"Missing CAPEX columns for newbuild engines at year {start_year}: {missing_engine_cols}"
        )

        df_capex_long = df_capex_year[['imo_type', 'size_id'] + newbuild_engines].melt(
            id_vars=['imo_type', 'size_id'],
            value_vars=newbuild_engines,
            var_name='engine',
            value_name='vessel_capex_newbuild',
        )

        df_nb = df_static.loc[mask_newbuild, ['dcf_id', 'imo_type', 'size_id', 'engine']].merge(
            df_capex_long,
            on=['imo_type', 'size_id', 'engine'],
            how='left',
        )
        assert df_nb['vessel_capex_newbuild'].notna().all(), (
            f"Missing newbuild CAPEX rows after merge: {df_nb[df_nb['vessel_capex_newbuild'].isna()].sample(5)}"
        )
        df_static.loc[mask_newbuild, 'vessel_capex'] = df_nb['vessel_capex_newbuild'].values

    if mask_retrofit.any():
        df_retrofit_year = df_retrofit_cost_ref[df_retrofit_cost_ref['year'] == start_year][[
            'imo_type', 'size_id', 'initial_fuel_design', 'retrofitted_fuel_design',
            'retrofit_cost', 'opportunity_cost',
        ]].copy()

        df_rt = df_static.loc[mask_retrofit, [
            'dcf_id', 'imo_type', 'size_id', 'prev_base_engine', 'engine', 'prev_vessel_capex',
        ]].merge(
            df_retrofit_year,
            left_on=['imo_type', 'size_id', 'prev_base_engine', 'engine'],
            right_on=['imo_type', 'size_id', 'initial_fuel_design', 'retrofitted_fuel_design'],
            how='left',
        )
        assert df_rt['retrofit_cost'].notna().all(), "Missing retrofit_cost rows in retrofit lookup"
        assert df_rt['opportunity_cost'].notna().all(), "Missing opportunity_cost rows in retrofit lookup"

        df_static.loc[mask_retrofit, 'retrofit_cost_capex'] = df_rt['retrofit_cost'].values
        df_static.loc[mask_retrofit, 'opportunity_cost'] = df_rt['opportunity_cost'].values
        df_static.loc[mask_retrofit, 'vessel_capex'] = (
            df_rt['prev_vessel_capex'].values + df_rt['retrofit_cost'].values
        )

    df_static['capex_annualized'] = 0.0
    if mask_keep.any():
        df_static.loc[mask_keep, 'capex_annualized'] = df_static.loc[mask_keep, 'prev_capex_annualized']
    if mask_newbuild.any():
        df_static.loc[mask_newbuild, 'capex_annualized'] = annualise_capex(
            df_static.loc[mask_newbuild, 'vessel_capex'].values,
            df_static.loc[mask_newbuild, 'wacc'].values,
            DEPRECIATION_TIME,
        )
    if mask_retrofit.any():
        remaining_lifetime = np.maximum(
            df_static.loc[mask_retrofit, 'lifetime'].values - df_static.loc[mask_retrofit, 'vessel_age'].values,
            0.1,
        )
        retrofit_capex_annualized = annualise_capex(
            df_static.loc[mask_retrofit, 'retrofit_cost_capex'].values,
            df_static.loc[mask_retrofit, 'wacc'].values,
            remaining_lifetime,
        )
        df_static.loc[mask_retrofit, 'capex_annualized'] = (
            df_static.loc[mask_retrofit, 'prev_capex_annualized'].values + retrofit_capex_annualized
        )

    # ---- OPEX inputs ----
    opex_required = [
        'crew-wages_usd-p-yr', 'technician-wages_usd-p-yr', 'mgmt-fees_usd-p-yr',
        'insurance_usd-p-yr', 'spares_usd-p-yr',
        'special-survey-fees_usd-p-occurrence', 'intermediate-survey-fees_usd-p-occurrence',
    ]
    mask_conventional = df_static['engine'] == 'conventional'
    mask_nonconv = ~mask_conventional

    df_opex_parts = []
    if mask_conventional.any():
        df_conv_ref = df_opex_conventional_ref.drop(columns=['avg-opex_usd-p-yr', 'scenario'], errors='ignore').copy()
        df_conv = df_static.loc[mask_conventional, ['dcf_id', 'imo_type', 'size_id', 'engine']].merge(
            df_conv_ref,
            on=['imo_type', 'size_id'],
            how='left',
        )
        for col in opex_required:
            assert col in df_conv.columns, f"Missing conventional OPEX column: {col}"
            assert df_conv[col].notna().all(), f"NaN in conventional OPEX column: {col}"
        df_opex_parts.append(df_conv[['dcf_id'] + opex_required])

    if mask_nonconv.any():
        df_nonconv = df_static.loc[mask_nonconv, ['dcf_id', 'imo_type', 'size_id', 'engine']].copy()
        df_wages = df_opex_nonconv_wages_ref.drop(columns='scenario', errors='ignore').copy()
        df_nonconv = df_nonconv.merge(df_wages, on=['imo_type', 'size_id'], how='left')
        for col in ['crew-wages_usd-p-yr', 'technician-wages_usd-p-yr', 'mgmt-fees_usd-p-yr']:
            assert col in df_nonconv.columns, f"Missing non-conventional wages column: {col}"
            assert df_nonconv[col].notna().all(), f"NaN in non-conventional wages column: {col}"

        special_tables = [
            ('insurance_usd-p-yr', df_opex_nonconv_insurance_ref),
            ('special-survey-fees_usd-p-occurrence', df_opex_nonconv_ssf_ref),
            ('intermediate-survey-fees_usd-p-occurrence', df_opex_nonconv_isf_ref),
            ('spares_usd-p-yr', df_opex_nonconv_spares_ref),
        ]
        nonconv_engines = sorted(df_nonconv['engine'].dropna().unique().tolist())
        for out_col, df_ref in special_tables:
            df_ref_clean = df_ref.drop(columns='scenario', errors='ignore').copy()
            id_cols = ['imo_type', 'size_id']
            value_cols = [c for c in df_ref_clean.columns if c not in id_cols and c in nonconv_engines]
            missing_ref_engines = [e for e in nonconv_engines if e not in df_ref_clean.columns]
            assert not missing_ref_engines, (
                f"Missing non-conventional OPEX engine columns for {out_col}: {missing_ref_engines}"
            )
            df_long = df_ref_clean.melt(
                id_vars=id_cols,
                value_vars=value_cols,
                var_name='engine',
                value_name=out_col,
            )
            before_rows = len(df_nonconv)
            df_nonconv = df_nonconv.merge(df_long, on=['imo_type', 'size_id', 'engine'], how='left')
            assert len(df_nonconv) == before_rows, (
                f"Row mismatch after non-conventional OPEX merge for {out_col}: expected {before_rows}, got {len(df_nonconv)}"
            )
            assert df_nonconv[out_col].notna().all(), f"NaN in non-conventional OPEX column: {out_col}"

        df_opex_parts.append(df_nonconv[['dcf_id'] + opex_required])

    df_opex = pd.concat(df_opex_parts, ignore_index=True)
    assert len(df_opex) == len(df_static), (
        f"OPEX rows mismatch: expected {len(df_static)}, got {len(df_opex)}"
    )

    df_static = df_static.merge(df_opex, on='dcf_id', how='left')
    for col in opex_required:
        assert df_static[col].notna().all(), f"NaN in static OPEX column after merge: {col}"

    lifetime = df_static['lifetime'].astype(float).values
    assert (lifetime > 0).all(), "All lifetime values must be positive"
    n_special = np.floor(lifetime / 5.0)
    n_intermediate = np.where(lifetime >= 3.0, np.floor((lifetime - 3.0) / 5.0) + 1.0, 0.0)
    total_survey_cost = (
        n_special * df_static['special-survey-fees_usd-p-occurrence'].values
        + n_intermediate * df_static['intermediate-survey-fees_usd-p-occurrence'].values
    )
    df_static['annual_survey_cost'] = total_survey_cost / lifetime
    df_static['annual_crew_cost'] = df_static['crew-wages_usd-p-yr']
    df_static['annual_technician_cost'] = df_static['technician-wages_usd-p-yr']
    df_static['annual_management_cost'] = df_static['mgmt-fees_usd-p-yr']
    df_static['annual_insurance_cost'] = df_static['insurance_usd-p-yr']
    df_static['annual_spares_cost'] = df_static['spares_usd-p-yr']
    df_static['annual_opex_excl_fuel_policy'] = (
        df_static['annual_crew_cost']
        + df_static['annual_technician_cost']
        + df_static['annual_management_cost']
        + df_static['annual_insurance_cost']
        + df_static['annual_spares_cost']
        + df_static['annual_survey_cost']
    )

    df_static['tc_usd_per_year'] = df_static['tc_usd_per_day'] * 365.25

    out_cols = [
        'dcf_id',
        'vessel_age',
        'prev_vessel_age',
        'capex_annualized',
        'retrofit_cost_capex',
        'vessel_capex',
        'opportunity_cost',
        'annual_survey_cost',
        'annual_crew_cost',
        'annual_technician_cost',
        'annual_management_cost',
        'annual_insurance_cost',
        'annual_spares_cost',
        'annual_opex_excl_fuel_policy',
        'tc_usd_per_year',
    ]
    df_out = df_static[out_cols]
    assert df_out['dcf_id'].is_unique, "dcf_id must be unique in static-cost output"
    return df_out

def calc_capex_premiums(df_capex_ref):
    ref_cols = ['unit', 'imo_type', 'size_id', 'scenario', 'conventional', 'min_size', 'max_size']
    calc_cols = df_capex_ref.columns.drop(ref_cols)

    df_capex_premiums_ref = df_capex_ref.copy()

    # CAPEX -> CAPEX premium
    for col in calc_cols:
        df_capex_premiums_ref[col] = df_capex_premiums_ref[col] - df_capex_premiums_ref['conventional']
        df_capex_premiums_ref['unit'] = 'USD (newbuild premium)'

    return df_capex_premiums_ref

def capex_retrofit_cost_reductions(capex_reduction_scenario, df_capex_premiums_ref, df_capex_ref, df_capex_reduction, df_retrofit_cost_ref):
    id_vars = ['unit', 'imo_type', 'size_id', 'scenario', 'min_size', 'max_size']
    df_capex_reduction = df_capex_reduction[df_capex_reduction.scenario == capex_reduction_scenario]

    # --- Apply projected reductions in CAPEX ---
    df_capex_reduction['reduction_factor'] = df_capex_reduction['reduction_factor_nb'].copy()

    df_capex_premiums_ref_projected = project_capex_with_learning_effects(
        df_capex_premiums_ref.melt(
            id_vars=id_vars,
            var_name='fuel_design',
            value_name='newbuild_premium',
        ),
        df_capex_reduction,
        LEARNING_FUELS,
        ['newbuild_premium']
    )
    # reformatting
    df_capex_premiums_ref_projected = df_capex_premiums_ref_projected.set_index(id_vars + ['year', 'fuel_design']).unstack(
        'fuel_design').reset_index()

    df_capex_premiums_ref_projected.columns = [
        col[1] if isinstance(col, tuple) and col[0] == 'newbuild_premium' else col[0] if isinstance(col, tuple) else col
        for col in df_capex_premiums_ref_projected.columns
    ]
    calc_cols = df_capex_ref.columns.drop(id_vars+['conventional'])
    df_capex_ref_projected = pd.merge(df_capex_ref, df_capex_premiums_ref_projected, on=['imo_type', 'size_id'])
    df_capex_ref_projected['conventional'] = df_capex_ref_projected['conventional_x']
    df_capex_ref_projected['unit'] = 'USD'

    for col in calc_cols:
        df_capex_ref_projected[col] = df_capex_ref_projected['conventional'] + df_capex_ref_projected[col+'_y']

    # Drop columns containing '_x' or '_y'
    df_capex_ref_projected = df_capex_ref_projected.loc[:, ~df_capex_ref_projected.columns.str.contains(r'_x|_y')]

    # --- Apply projected reductions in Retrofit Cost ---
    df_retrofit_reduction = df_capex_reduction.copy()
    df_retrofit_reduction['reduction_factor'] = df_retrofit_reduction['reduction_factor_retrofit']

    df_retrofit_cost_ref_projected = project_capex_with_learning_effects(
        df_retrofit_cost_ref.rename(columns={"retrofitted_fuel_design": "fuel_design"}),
        df_retrofit_reduction,
        LEARNING_FUELS,
        ['retrofit_cost']
    )
    df_retrofit_cost_ref_projected = df_retrofit_cost_ref_projected.rename(
        columns={'fuel_design': 'retrofitted_fuel_design'})

    return df_capex_ref_projected, df_retrofit_cost_ref_projected


# ====== OTHER ======
def assign_first_capex_opex(user_params, df_vessels, df_capex_ref):
    """
    Assign initial CAPEX (and optionally annualized CAPEX) to each ship row
    based on imo_type, size, and engine_type.
    """
    # Prepare df_capex_ref
    df_capex_ref = df_capex_ref[df_capex_ref['year'] == 2025].copy()    # fleet capex initialized using 2025 numbers

    # merge df_vessels with df_capex_ref
    df_vessels = df_vessels.merge(df_capex_ref, on=['imo_type', 'size_id'], how='left')
    engine_col_idx = df_vessels.columns.get_indexer(df_vessels["prev_engine"])
    df_vessels["prev_vessel_capex"] = df_vessels.to_numpy()[np.arange(len(df_vessels)), engine_col_idx]

    # drop extra columns
    cols_to_keep = {"imo_type", "size_id"}
    cols_to_drop = [
        c for c in df_capex_ref.columns
        if c not in cols_to_keep and c in df_vessels.columns
    ]
    df_vessels = df_vessels.drop(columns=cols_to_drop)

    # differentiated the wacc depending on the engine of the ship
    df_vessels['wacc'] = user_params['low_wacc']
    mask = df_vessels[df_vessels.engine.isin(HIGH_WACC_ENGINES)].index
    df_vessels.loc[mask,'wacc'] = user_params['high_wacc']
    df_vessels["prev_capex_annualized"] = annualise_capex(df_vessels["prev_vessel_capex"], df_vessels['wacc'], DEPRECIATION_TIME)

    return df_vessels

def calculate_initial_values(
    df_vessel_params,
    current_year,
    user_params,
    df_imo_targets_data,
    df_fueleu_targets_data,
    # df_operational_data_ref,
    # df_tc_rates_ref,
    ru_price_data,
    initial_su_price_data,
):
    """initial-value construction for batch TCO analysis.

    This function appends parameters as dataframe columns (no per-row series
    construction) and uses merge/map operations for reference lookups.
    """

    assert isinstance(df_vessel_params, pd.DataFrame), (
        f"Expected df_vessel_params DataFrame, got {type(df_vessel_params)}"
    )
    assert len(df_vessel_params) > 0, "calculate_initial_values received empty dataframe"

    df_params = df_vessel_params.copy()
    df_params['year'] = current_year

    # Base engine labels used downstream for compatibility checks.
    df_params['base_engine'] = df_params['engine'].astype(str).str.split('_+_', regex=False).str[0]
    df_params['prev_base_engine'] = df_params['prev_engine'].astype(str).str.split('_+_', regex=False).str[0]
    assert df_params['tc_usd_per_day'].notna().all(), "tc_usd_per_day contains NaNs after merge"

    # Year-level policy values — single call since current_year is a scalar.
    base_target, direct_target, znz = get_gfi_targets(
        current_year,
        user_params['imo_scenario'],
        df_imo_targets_data,
    )
    fueleu, fueleu_penalty, eu_ets = get_eu_policy(
        current_year,
        user_params['fueleu_scenario'],
        df_fueleu_targets_data,
    )
    ru1_price, ru2_price, exogenous_su_price, pooling_price = get_ru_su_prices(
        current_year,
        user_params['ru_scenario'],
        ru_price_data,
        direct_target,
        user_params['initial_su_price_scenario'],
        initial_su_price_data,
        fueleu,
        eu_ets,
    )

    df_params['base_compliance_target'] = base_target
    df_params['direct_compliance_target'] = direct_target
    df_params['znz'] = znz
    df_params['reward_threshold_gco2_mj'] = znz
    df_params['FuelEU'] = fueleu
    df_params['FuelEU_penalty'] = fueleu_penalty
    df_params['EU_ETS'] = eu_ets
    df_params['ru1_price'] = ru1_price
    df_params['ru2_price'] = ru2_price
    df_params['eu_pooling_price'] = pooling_price

    cols_to_add = ['reward_delay', 'low_wacc', 'high_wacc', 'year_eu_policy_dropped']
    for c in cols_to_add:
        df_params[c] = user_params[c]

    # if there is a functioning SU market (direct compliance target > 0), then use a price
    if direct_target > 0:
        df_params['su_price'] = exogenous_su_price
        if 'su_price' in df_vessel_params.columns:
            mask = df_vessel_params[df_vessel_params.su_price>0].index # exclude the first year, where SU price = 0
            df_params.loc[mask, 'su_price'] = df_vessel_params.loc[mask, 'su_price']            
    # otherwise, ships pay for RU2
    else:
        df_params['su_price'] = ru2_price
    return df_params

def project_capex_with_learning_effects(
    capex_df: pd.DataFrame, # a dataframe containing the columns usd_p_kW, usd_p_tfuel, fuel_design
    reduction_factors: pd.DataFrame, # a dataframe containing columns year (e.g. from 2025 to 2050) and reduction_factor
    learning_fuels: list, # a list of fuels which are subject to learning effects
    cols: list, # a list of columns containing costs which should be reduced
):
    years = reduction_factors["year"].sort_values().unique()
    reduction_factors = reduction_factors.set_index("year")

    # Expand the dataframe across years
    capex_df_projected = (
        capex_df
        .assign(key=1)
        .merge(pd.DataFrame({"year": years, "key": 1}), on="key")
        .drop("key", axis=1)
    )
    capex_df_projected[cols] = capex_df_projected[cols].astype(float)

    # For rows with fuel_design in learning_fuels, apply linear decrease
    for col in cols:
        for year in years:
            mask = (
                (capex_df_projected["fuel_design"].isin(learning_fuels))
                & (capex_df_projected["year"] == year)
            )
            capex_df_projected.loc[mask, col] = (
                capex_df_projected.loc[mask, col] * reduction_factors.loc[year,'reduction_factor']
            )
    return capex_df_projected


