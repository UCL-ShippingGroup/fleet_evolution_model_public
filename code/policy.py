import numpy as np
import pandas as pd

def get_gfi_targets(year, imo_scenario, df_imo_targets_data):
    """Get GFI targets for a specific year, scenario"""

    gfi_targets = df_imo_targets_data[
        (df_imo_targets_data['year'] == year) &
        (df_imo_targets_data['scenario'] == imo_scenario)
        ]

    base_gfi = gfi_targets['base_compliance_target'].iloc[0]
    direct_gfi = gfi_targets['direct_compliance_target'].iloc[0]
    znz = gfi_targets['znz'].iloc[0]

    return base_gfi, direct_gfi, znz

def get_eu_policy(year, fueleu_scenario, df_fueleu_targets_data):
    df_fueleu = df_fueleu_targets_data[
        (df_fueleu_targets_data['year'] == year) &
        (df_fueleu_targets_data['scenario'] == fueleu_scenario)
        ]

    FuelEU=df_fueleu['FuelEU'].iloc[0]
    fueleu_penalty = df_fueleu['fueleu_penalty'].iloc[0]
    fuel_eu_ets_price =  df_fueleu['ets_price'].iloc[0]

    return FuelEU, fueleu_penalty, fuel_eu_ets_price

def get_ru_su_prices(year, ru_scenario, ru_price_data, direct_compliance_target, initial_su_price_scenario, initial_su_price_data, fueleu, eu_ets):
    """Get RU, SU prices for a specific year, scenario"""
    if direct_compliance_target < 100:
        ru_prices = ru_price_data[
            (ru_price_data['year'] == year) &
            (ru_price_data['scenario'] == ru_scenario)
            ]

        ru1_price = ru_prices['ru1'].iloc[0]
        ru2_price = ru_prices['ru2'].iloc[0]

        # if there is no SU market (i.e. direct_compliance_target = 0), then assign su_price = ru_price
        su_price = min(initial_su_price_data[
            (initial_su_price_data['year'] == year)
            &(initial_su_price_data['scenario'] == initial_su_price_scenario) 
        ]['value'].iloc[0], ru2_price)
    else:
        ru1_price = 0
        ru2_price = 0
        su_price = 0
    
    if fueleu < 100:
        eu_pooling_price = initial_su_price_data[
            (initial_su_price_data['year'] == year) &
            (initial_su_price_data['scenario'] == initial_su_price_scenario)
        ]['value'].iloc[0] - eu_ets
    else:
        eu_pooling_price = 0

    return ru1_price, ru2_price, su_price, eu_pooling_price

# --- Calculate policy ---
def calculate_imo_compliance_metrics(attained_gfi_wtw_gco2eq_mj,
                                 base_compliance_target, direct_compliance_target,
                                 ):
    """Calculate compliance metrics: Fuel Intensity thresholds against the
    attained-GFI (== overall WTW carbon intensity of the fuel/blend)."""

    # Fuel intensity above base compliance (gCO2eq/MJ)
    fuel_intensity_above_base_gco2eq_mj = np.maximum(0, attained_gfi_wtw_gco2eq_mj - base_compliance_target)
    fuel_intensity_above_base_tco2eq_tlsfoeq = fuel_intensity_above_base_gco2eq_mj*41.2/1000.

    # Fuel intensity between base and direct compliance (gCO2eq/MJ)
    fuel_intensity_between_gco2eq_mj = np.maximum(0, attained_gfi_wtw_gco2eq_mj - direct_compliance_target) - fuel_intensity_above_base_gco2eq_mj
    fuel_intensity_between_tco2eq_tlsfoeq = fuel_intensity_between_gco2eq_mj*41.2/1000.

    # Fuel intensity below direct compliance (gCO2eq/MJ)
    fuel_intensity_below_direct_gco2eq_mj = np.maximum(0, direct_compliance_target - attained_gfi_wtw_gco2eq_mj)
    fuel_intensity_below_direct_tco2eq_tlsfoeq = fuel_intensity_below_direct_gco2eq_mj*41.2/1000.

    # Surplus units
    su_generated = fuel_intensity_below_direct_tco2eq_tlsfoeq

    return {
        'base_compliance_target': base_compliance_target,
        'direct_compliance_target': direct_compliance_target,
        'fuel_intensity_above_base_gco2eq_mj': fuel_intensity_above_base_gco2eq_mj,
        'fuel_intensity_above_base_tco2eq_tlsfoeq': fuel_intensity_above_base_tco2eq_tlsfoeq,
        'fuel_intensity_between_gco2eq_mj': fuel_intensity_between_gco2eq_mj,
        'fuel_intensity_between_tco2eq_tlsfoeq': fuel_intensity_between_tco2eq_tlsfoeq,
        'fuel_intensity_below_direct_gco2eq_mj': fuel_intensity_below_direct_gco2eq_mj,
        'fuel_intensity_below_direct_tco2eq_tlsfoeq': fuel_intensity_below_direct_tco2eq_tlsfoeq,
        'su_generated': su_generated
    }

def calculate_imo_compliance_costs(compliance_metrics, ru2_price, ru1_price, su_price, total_consumption_tlsfo_eq):
    """Calculate compliance costs and revenues following Excel formulas"""

    # Remedial units above base compliance target (cost)
    ru2_cost_usd_tvslfoeq = compliance_metrics['fuel_intensity_above_base_tco2eq_tlsfoeq'] * su_price

    # Remedial units between base and direct compliance target (cost)
    ru1_cost_usd_tvslfoeq = compliance_metrics['fuel_intensity_between_tco2eq_tlsfoeq'] * ru1_price

    # Surplus units below direct compliance target (revenue)
    su_revenue_usd_tvslfoeq = compliance_metrics['su_generated'] * su_price

    # Total compliance cost/revenue
    total_compliance_cost_usd_tvslfoeq = ru2_cost_usd_tvslfoeq + ru1_cost_usd_tvslfoeq - su_revenue_usd_tvslfoeq
    total_compliance_cost_usd = total_compliance_cost_usd_tvslfoeq * total_consumption_tlsfo_eq

    return {
        'ru1_price': ru1_price,
        'ru2_price': ru2_price,
        'su_price': su_price,
        'ru2_cost_usd_tvslfoeq': ru2_cost_usd_tvslfoeq,
        'ru1_cost_usd_tvslfoeq': ru1_cost_usd_tvslfoeq,
        'su_revenue_usd_tvslfoeq': su_revenue_usd_tvslfoeq,
        'total_compliance_cost_usd_tvslfoeq': total_compliance_cost_usd_tvslfoeq,
        'total_compliance_cost_usd': total_compliance_cost_usd
    }

def calculate_eu_policy_cost(reg_reg_class, overall_emissions_factor_wtw_gco2eq_mj, total_ttw_emissions_tco2eq,
                             FuelEU, pooling_value, FuelEU_penalty, EU_ETS,
                             total_consumption_tlsfo_eq):
    """
    EU policy costs and pooling revenues.
    Inputs are pd.Series or arrays of the same length.
    Returns dict of arrays: 'EU pooling revenue' and 'EU compliance cost'.
    """
    n = len(np.atleast_1d(total_ttw_emissions_tco2eq))

    # Initialize arrays with zeros
    fueleu_compliance_cost_usd_p_gj = np.zeros(n)
    eu_pooling_revenue_usd_p_gj = np.zeros(n)
    eu_ets_usd = np.zeros(n)

    # Mask for EU voyages
    mask_eu = np.isin(reg_reg_class, ['eu_to_eu', 'eu_to_noneu'])

    effective_pooling_value = np.minimum(FuelEU_penalty[mask_eu], pooling_value[mask_eu])
    # Fuel EU compliance cost
    fueleu_compliance_cost_usd_p_gj[mask_eu] = (
        np.maximum(overall_emissions_factor_wtw_gco2eq_mj[mask_eu] - FuelEU[mask_eu], 0) / 1000
        * effective_pooling_value
    )

    # EU pooling revenue
    eu_pooling_revenue_usd_p_gj[mask_eu] = -(
        effective_pooling_value 
        * np.maximum(0, FuelEU[mask_eu] - overall_emissions_factor_wtw_gco2eq_mj[mask_eu]) / 1000
    )

    # EU ETS
    eu_ets_usd[mask_eu] = total_ttw_emissions_tco2eq[mask_eu] * EU_ETS[mask_eu]

    # Mask for EU → non-EU voyages (only half of the cost)
    mask_eu_non_eu = reg_reg_class == 'eu_to_noneu'
    fueleu_compliance_cost_usd_p_gj[mask_eu_non_eu] /= 2
    eu_pooling_revenue_usd_p_gj[mask_eu_non_eu] /= 2
    eu_ets_usd[mask_eu_non_eu] /= 2

    # Calculate total costs
    eu_cost_usd = (
        (fueleu_compliance_cost_usd_p_gj + eu_pooling_revenue_usd_p_gj)
        * 41.2
        * total_consumption_tlsfo_eq
        + eu_ets_usd

    )
    return {
        'fueleu_compliance_cost_usd_p_gj': fueleu_compliance_cost_usd_p_gj,
        'eu_pooling_revenue_usd_p_gj': eu_pooling_revenue_usd_p_gj,
        'eu_ets_usd': eu_ets_usd,
        'eu_cost_usd': eu_cost_usd,
    }


def calculate_policy_compliance(params, 
                                total_consumption_tlsfo_eq, 
                                overall_emissions_factor_wtw_gco2eq_mj,
                                total_ttw_emissions_tco2eq,
                                attained_gfi_wtw_gco2eq_mj,
                                compliance_mode):
    """
    Calculate policy compliance costs
    """

    length = len(params)
    # Replace params with a dict of arrays, but keep the name "params"
    params = {k: params[k].values for k in params.columns}
    if 'reward_usd_tlsfoeq' not in params:
        params['reward_usd_tlsfoeq'] = np.zeros(length)

    assert compliance_mode in ["imo","fueleu_imo", "fueleu"], f"{compliance_mode} is not a valid run mode"
    is_imo_mode = (compliance_mode == 'imo') | (compliance_mode == 'fueleu_imo')

    # Calculate compliance metrics against overall_emissions_factor_wtw_gco2eq_mj
    # remains used below for FuelEU/EU-ETS costs and the ZNZ reward threshold check.
    # Skipped for pure 'fueleu' mode, where attained_gfi_wtw_gco2eq_mj is not available/needed.
    if is_imo_mode:
        assert attained_gfi_wtw_gco2eq_mj is not None, "attained_gfi_wtw_gco2eq_mj is required when compliance_mode includes 'imo'"
        compliance_metrics = calculate_imo_compliance_metrics(
            attained_gfi_wtw_gco2eq_mj,
            params['base_compliance_target'], 
            params['direct_compliance_target'],
            )
    else:
        zeros = np.zeros(length)
        compliance_metrics = {
            'base_compliance_target': zeros,
            'direct_compliance_target': zeros,
            'fuel_intensity_above_base_gco2eq_mj': zeros,
            'fuel_intensity_above_base_tco2eq_tlsfoeq': zeros,
            'fuel_intensity_between_gco2eq_mj': zeros,
            'fuel_intensity_between_tco2eq_tlsfoeq': zeros,
            'fuel_intensity_below_direct_gco2eq_mj': zeros,
            'fuel_intensity_below_direct_tco2eq_tlsfoeq': zeros,
            'su_generated': zeros,
        }

    # Get RU for SU/RU feedback loop
    su_tco2 = compliance_metrics['su_generated'] * total_consumption_tlsfo_eq
    ru_super_emitter_no_biofuel_tco2 = compliance_metrics['fuel_intensity_above_base_tco2eq_tlsfoeq'] * total_consumption_tlsfo_eq
    ru_small_emitter_no_biofuel_tco2 = compliance_metrics['fuel_intensity_between_tco2eq_tlsfoeq'] * total_consumption_tlsfo_eq
    ru_tco2 = ru_super_emitter_no_biofuel_tco2 + ru_small_emitter_no_biofuel_tco2

    if is_imo_mode:
        nzf_compliance_costs = calculate_imo_compliance_costs(
            compliance_metrics,
            params['ru2_price'],
            params['ru1_price'],
            params['su_price'],
            total_consumption_tlsfo_eq
        )

        reward_usd_tlsfoeq = (
            params['reward_usd_tlsfoeq']
            / (1 + params["wacc"]) ** params["reward_delay"]
            * (overall_emissions_factor_wtw_gco2eq_mj<params["reward_threshold_gco2_mj"])
        )

        reward_usd = reward_usd_tlsfoeq * total_consumption_tlsfo_eq
    else:
        nzf_compliance_costs = {
            'ru1_price': np.zeros(length),
            'ru2_price': np.zeros(length),
            'su_price': np.zeros(length),
            'ru2_cost_usd_tvslfoeq': np.zeros(length),
            'ru1_cost_usd_tvslfoeq': np.zeros(length),
            'su_revenue_usd_tvslfoeq': np.zeros(length),
            'total_compliance_cost_usd_tvslfoeq': np.zeros(length),
            'total_compliance_cost_usd': np.zeros(length),
        }
        reward_usd_tlsfoeq = np.zeros(length)
        reward_usd = np.zeros(length)

    # Fuel EU and EU ETS costs and revenues
    # note that params must include a column reg_reg_class which contains the regime (eu_to_eu, eu_to_noneu, noneu_to_noneu)
    if (compliance_mode=='fueleu') | (compliance_mode=='fueleu_imo'):

        # if both EU and NZF are in place, pooling value = SU price
        # if only EU is in place, pooling value = FuelEU penalty
        # if EU is not in place, pooling value = 0
        mask_eu = (params['FuelEU'] < 100)
        mask_eu_and_nzf = mask_eu & (params['direct_compliance_target'] < 100)
        mask_eu_only = mask_eu & (params['direct_compliance_target'] >= 100)
        params['pooling_value'] = np.where(
            mask_eu_and_nzf,
            params['su_price'],
            np.where(mask_eu_only, params['eu_pooling_price'], 0)
        )

        # if the EU regulation is in place, assume EU ETS price
        mask = (params['FuelEU'] < 100)
        params['EU_ETS'] = np.where(mask, params['EU_ETS'], 0)

        eu_costs = calculate_eu_policy_cost(
            reg_reg_class=params['reg_reg_class'],    # rep ships
            overall_emissions_factor_wtw_gco2eq_mj=np.atleast_1d(overall_emissions_factor_wtw_gco2eq_mj),
            total_ttw_emissions_tco2eq=np.atleast_1d(total_ttw_emissions_tco2eq),
            FuelEU=params['FuelEU'],    # Input data: FuelEU base GFI.
            pooling_value=params['pooling_value'],    # equivalent to the su price
            FuelEU_penalty=params['FuelEU_penalty'],    # ru prices csv
            EU_ETS=params['EU_ETS'],    # ru prices csv
            total_consumption_tlsfo_eq=np.atleast_1d(total_consumption_tlsfo_eq),
        )
    else:
        params['pooling_value'] = ""
        eu_costs = {
            'fueleu_compliance_cost_usd_p_gj': np.zeros(length),
            'eu_pooling_revenue_usd_p_gj': np.zeros(length),
            'eu_compliance_cost_usd': np.zeros(length),
            'eu_cost_usd': np.zeros(length),
            'eu_ets_usd': np.zeros(length),
        }

    # Prepare results
    results = {
        # Compliance metrics
        'base_compliance_target': compliance_metrics['base_compliance_target'],
        'direct_compliance_target': compliance_metrics['direct_compliance_target'],
        'fuel_intensity_above_base_gco2eq_mj': compliance_metrics['fuel_intensity_above_base_gco2eq_mj'],
        'fuel_intensity_above_base_tco2eq_tlsfoeq': compliance_metrics['fuel_intensity_above_base_tco2eq_tlsfoeq'],
        'fuel_intensity_between_gco2eq_mj': compliance_metrics['fuel_intensity_between_gco2eq_mj'],
        'fuel_intensity_between_tco2eq_tlsfoeq': compliance_metrics['fuel_intensity_between_tco2eq_tlsfoeq'],
        'fuel_intensity_below_direct_gco2eq_mj': compliance_metrics['fuel_intensity_below_direct_gco2eq_mj'],
        'fuel_intensity_below_direct_tco2eq_tlsfoeq': compliance_metrics['fuel_intensity_below_direct_tco2eq_tlsfoeq'],
        'attained_gfi_wtw_gco2eq_mj': attained_gfi_wtw_gco2eq_mj,
        'su_generated': compliance_metrics['su_generated'],

        # SU and RU
        'su_tco2': su_tco2,
        'ru_tco2': ru_tco2,
        'ru_super_emitter_no_biofuel_tco2': ru_super_emitter_no_biofuel_tco2,
        'ru_small_emitter_no_biofuel': ru_small_emitter_no_biofuel_tco2,

        # RU, SU prices
        'ru1_price': nzf_compliance_costs['ru1_price'],
        'ru2_price': nzf_compliance_costs['ru2_price'],
        'su_price': nzf_compliance_costs['su_price'],

        # Compliance costs/revenues
        'ru2_cost_usd_tvslfoeq': nzf_compliance_costs['ru2_cost_usd_tvslfoeq'],
        'ru1_cost_usd_tvslfoeq': nzf_compliance_costs['ru1_cost_usd_tvslfoeq'],
        'su_revenue_usd_tvslfoeq': nzf_compliance_costs['su_revenue_usd_tvslfoeq'],
        'total_compliance_cost_usd_tvslfoeq': nzf_compliance_costs['total_compliance_cost_usd_tvslfoeq'],
        'total_compliance_cost_usd': nzf_compliance_costs['total_compliance_cost_usd'],

        # Additional policy costs
        'reward_usd_tlsfoeq': reward_usd_tlsfoeq,
        'reward_usd': reward_usd,
        'FuelEU': params['FuelEU'],
        'FuelEU_penalty': params['FuelEU_penalty'],
        'EU_ETS': params['EU_ETS'],
        'pooling_value': params['pooling_value'],
        'fueleu_compliance_cost_usd_p_gj': eu_costs['fueleu_compliance_cost_usd_p_gj'],
        'eu_pooling_revenue_usd_p_gj': eu_costs['eu_pooling_revenue_usd_p_gj'],
        'eu_compliance_cost_usd': eu_costs['eu_cost_usd'],
        'eu_ets_usd': eu_costs['eu_ets_usd'],
        'net_policy_cost': nzf_compliance_costs['total_compliance_cost_usd'] - reward_usd + eu_costs['eu_cost_usd'],
        "overall_emissions_factor": overall_emissions_factor_wtw_gco2eq_mj,
    }

    return results