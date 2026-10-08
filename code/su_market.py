import pandas as pd
import numpy as np
from pathlib import Path
from ship_econ import run_vessel_scenario
from default_assumptions import FUEL_GROUPS_AVAIL


def _ship_choice_key(df, ship_col='ship_year_id', choice_col='dcf_id'):
    """Build per-row composite keys to uniquely identify a ship choice."""
    return list(zip(df[ship_col], df[choice_col]))

def _build_cohort_var(df, var):
    assert var in df.columns, f"{var} not in {df.columns}"
    df[f"{var}_cohort"] = df[var] * df["real_ships_nb"]
    return df

def choose_fuel_given_su(df_year_active, su_price, user_params,
                         engines_design_mapping, engine_fuels_dict,
                         df_drop_in_fuels_ref, df_energy_density_ref,
                         df_imo_targets_data, df_fueleu_targets_data,
                         df_wtt_carbon_intensity_ref, df_ttw_carbon_intensity_ref,
                         df_reward_ref, df_tc_rates_ref, df_pilot_fuel_input_ref,
                         ru_price_data, initial_su_price_data, fuel_price_data, df_capex_ref,
                         df_retrofit_cost_ref,
                         df_opex_conventional_ref, df_opex_nonconv_wages_ref,
                         df_opex_nonconv_insurance_ref,
                         df_opex_nonconv_ssf_ref, df_opex_nonconv_isf_ref,
                         df_opex_nonconv_spares_ref,
                         df_operational_data_ref,
                         game_year,
                         fuel_group_avail,
                         max_iterations=1, *,
                         df_speed_ref, df_efficiency_ref, df_engine_efficiency_ref):
    """Run the SU-price-fixed fuel-choice loop under fuel-group availability constraints.

    High-level flow:
    1) Evaluate best fuel choice for all currently unresolved ships.
    2) Allocate limited fuel-group availability by priority rank.
    3) Keep ships that remain within caps; rerun the excluded ships with saturated
       groups removed from allowed fuels.
    4) Repeat until either all ships are assigned or max_iterations is reached.

    Output metadata columns:
    - allocation_iter: iteration index in this fuel-allocation loop.
    - fuel_group_excluded: cumulative excluded fuel groups at that iteration.
    """

    # Step 1: initialize a randomly ordered list of ship_year_id which determines the priority to fuel(s) access
    unique_ship_ids = df_year_active['ship_year_id'].drop_duplicates().to_numpy()
    assert len(unique_ship_ids) > 0, "No ship_year_id found to build fuel-priority order"

    # Priority order is random; draws from the scenario's own rng (not global NumPy state),
    # so this is reproducible and independent of other concurrently running scenarios.
    user_params['rng'].shuffle(unique_ship_ids)

    df_priority = pd.DataFrame({
        'ship_year_id': unique_ship_ids,
        'fuel_priority_rank': np.arange(1, len(unique_ship_ids) + 1, dtype=int),
    })
 
    # Initialise the dataframes
    user_params_work = user_params.copy()
    assert fuel_group_avail is not None, "fuel_group_avail is required for Step 3 availability allocation"
    fuel_group_avail['availability_reached'] = False
    df_unassigned_ships = df_year_active.copy()
    df_selected_ships = pd.DataFrame()
    df_selected_rrc_rows = pd.DataFrame()
    excluded_groups_label = ""

    individual_ships_cols = ["ship_year_id", "ship_name", "archetype"]
    same_ships_cols = [
        'imo_type', 'size_id', 'year_built', 'action', 'engine', 'engine_type','engine_type_label',
        'prev_vessel_year_built', 'prev_engine',
    ]

    missing_cols = [
        c for c in (individual_ships_cols + same_ships_cols)
        if c not in df_year_active.columns
    ]
    assert not missing_cols, f"Missing grouping columns in df_year_active: {missing_cols}"

    for allocation_iter in range(max_iterations):
        # Step 2: for ships which do not have a fuel yet, choose the fuels which minimise the costs among the available fuels and given the SU price
        # In the first round, all fuels are available and all ships are run
        assert len(df_unassigned_ships)>0, f"df_unassigned_ships empty at iteration {allocation_iter}"
        df_unassigned_ships['su_price'] = su_price
        assert len(df_unassigned_ships[df_unassigned_ships['engine_type'].isna()]) == 0, f"Nan in the engine types: {df_unassigned_ships[df_unassigned_ships['engine_type'].isna()]}"
        df_unique_ship_profiles = (
            df_unassigned_ships
            .drop(columns=individual_ships_cols)
            .drop_duplicates(subset=same_ships_cols)
            .reset_index(drop=True)
        )

        df_best_choice_rows, df_compare_unique = run_vessel_scenario(
            start_year=game_year,
            num_years=1,
            user_params=user_params_work,
            df_vessel_params=df_unique_ship_profiles,
            engines_design_mapping=engines_design_mapping,
            engine_fuels_dict=engine_fuels_dict,
            df_drop_in_fuels_ref=df_drop_in_fuels_ref,
            df_energy_density_ref=df_energy_density_ref,
            df_imo_targets_data=df_imo_targets_data,
            df_fueleu_targets_data=df_fueleu_targets_data,
            df_wtt_carbon_intensity_ref=df_wtt_carbon_intensity_ref,
            df_ttw_carbon_intensity_ref=df_ttw_carbon_intensity_ref,
            df_reward_ref=df_reward_ref,
            df_tc_rates_ref=df_tc_rates_ref,
            df_pilot_fuel_input_ref=df_pilot_fuel_input_ref,
            ru_price_data=ru_price_data,
            initial_su_price_data=initial_su_price_data,
            fuel_price_data=fuel_price_data,
            df_capex_ref=df_capex_ref,
            df_retrofit_cost_ref=df_retrofit_cost_ref,
            df_opex_conventional_ref=df_opex_conventional_ref,
            df_opex_nonconv_wages_ref=df_opex_nonconv_wages_ref,
            df_opex_nonconv_insurance_ref=df_opex_nonconv_insurance_ref,
            df_opex_nonconv_ssf_ref=df_opex_nonconv_ssf_ref,
            df_opex_nonconv_isf_ref=df_opex_nonconv_isf_ref,
            df_opex_nonconv_spares_ref=df_opex_nonconv_spares_ref,
            df_operational_data_ref=df_operational_data_ref,
            game_year=game_year, df_speed_ref=df_speed_ref, df_efficiency_ref=df_efficiency_ref,
            df_engine_efficiency_ref=df_engine_efficiency_ref,
        )

        # Validate SU price matches (allow small floating point differences).
        # ignore when the SU market is not actually modelled (i.e. SU = RU2)
        if su_price > 0:
            su_price_diff = (pd.to_numeric(df_best_choice_rows['su_price'], errors='coerce') - su_price).abs()
            if (su_price_diff > 1.0).any():
                bad_rows = df_best_choice_rows.loc[su_price_diff > 1.0, same_ships_cols + ['su_price']]
                raise ValueError(
                    f"SU price mismatch in batch evaluation with SU price in{bad_rows['su_price'].unique()}; expected {su_price}."
                )

            # Merge selected rows back into the currently unresolved ship set.
            assert len(df_best_choice_rows) > 0, "No results returned from run_vessel_scenario, check calculations"
            assert all(c in df_best_choice_rows.columns for c in same_ships_cols), (
                f"Grouped SU results missing merge columns: "
                f"{[c for c in same_ships_cols if c not in df_best_choice_rows.columns]}"
            )
            assert not df_best_choice_rows.duplicated(subset=same_ships_cols).any(), (
                "Grouped SU results are not unique on same_ships_cols"
            )
            assert len(df_best_choice_rows[df_best_choice_rows['dcf_id'].isna()]) == 0, f"Nan in the dcf_id: {df_best_choice_rows[df_best_choice_rows['dcf_id'].isna()]}"

        expected_len = len(df_unassigned_ships)
        df_unassigned_ships = df_unassigned_ships[same_ships_cols + individual_ships_cols].merge(
            df_best_choice_rows,
            on=same_ships_cols,
            how='left',
        )
        assert len(df_unassigned_ships) == expected_len, (
            f"Post-merge SU length mismatch: expected {expected_len}, got {len(df_unassigned_ships)}"
        )
        df_unassigned_ships['game_year'] = game_year

        # Expand grouped per-planning-year/per-RRC rows back to all individual ships.
        assert all(c in df_compare_unique.columns for c in same_ships_cols), (
            f"Grouped SU compare rows missing merge columns: "
            f"{[c for c in same_ships_cols if c not in df_compare_unique.columns]}"
        )
        compare_rows = df_unassigned_ships[individual_ships_cols + same_ships_cols].merge(
            df_compare_unique,
            on=same_ships_cols,
            how='left',
        )
        check_len = len(compare_rows)

        # scale some results to the full fleet, to be able to compare to the fleet-wide limits
        for var in [
            'pilot_fuel_consumption_mj_year','main_fuel_consumption_mj_year','fuel_consumption_mj_year',
            ]:
            compare_rows = _build_cohort_var(compare_rows,var)
        for var in [
            'su_tco2','ru_super_emitter_no_biofuel_tco2'
            ]:
            df_unassigned_ships = _build_cohort_var(df_unassigned_ships,var)

        # Add the priority ranking to per-planning-year/per RCC df
        compare_rows = compare_rows.merge(df_priority, on='ship_year_id', how='left')
        assert len(compare_rows) == check_len, (
            f"Row count changed after adding fuel priority rank: expected {check_len}, got {len(compare_rows)}"
        )
        assert compare_rows['fuel_priority_rank'].notna().all(), (
            f"Missing fuel_priority_rank rows: "
            f"{compare_rows.loc[compare_rows['fuel_priority_rank'].isna(), 'ship_year_id'].unique().tolist()}"
        )
        compare_rows = compare_rows.sort_values('fuel_priority_rank').copy()
        compare_rows['fuel_chosen_available_flag'] = True
        
        # Step 3: allocate fuel-group availability on route rows sorted by ship priority.
        # If any route row of a ship falls beyond the cumulative availability cap,
        # that ship is excluded from first choice and rerun in Step 4.
        df_unassigned_ships['fuel_chosen_available_flag'] = True
        
        # Run through each fuel group
        
        all_fuels_available = True
        for fuel_group in FUEL_GROUPS_AVAIL.fuel_group.unique():
            fuels_in_group = FUEL_GROUPS_AVAIL[FUEL_GROUPS_AVAIL.fuel_group == fuel_group].fuel.unique()

            # compute the fuel consumption for fuel_group in compare_rows
            has_fuel_mix = 'fuel_1' in compare_rows.columns
            group_mask = (compare_rows.fuel.isin(fuels_in_group))
            if has_fuel_mix:
                group_mask = (
                    group_mask
                    |(
                        (compare_rows.fuel_1.isin(fuels_in_group))
                        &(compare_rows.share_fuel_1_final>0)
                    )
                    |(
                        (compare_rows.fuel_2.isin(fuels_in_group))
                        &(compare_rows.share_fuel_2_final>0)
                    )
                    |(
                        (compare_rows.fuel_3.isin(fuels_in_group))
                        &(compare_rows.share_fuel_3_final>0)
                    )
                )
            group_mask = group_mask | (
                (compare_rows.pilot_fuel.isin(fuels_in_group))
                &(compare_rows.pilot_fuel_consumption_mj_year_cohort>0)
            )
            compare_rows_group = compare_rows[group_mask].copy()
            if len(compare_rows_group) == 0:
                continue

            compare_rows_group['demand_fuel_mj_year'] = 0.0
            mask = compare_rows_group.fuel.isin(fuels_in_group)
            compare_rows_group.loc[mask, 'demand_fuel_mj_year'] += compare_rows_group.loc[mask, 'main_fuel_consumption_mj_year_cohort']
            mask = compare_rows_group.pilot_fuel.isin(fuels_in_group)
            compare_rows_group.loc[mask, 'demand_fuel_mj_year'] += compare_rows_group.loc[mask, 'pilot_fuel_consumption_mj_year_cohort']
            if has_fuel_mix:
                mask = compare_rows_group.fuel_1.isin(fuels_in_group)
                compare_rows_group.loc[mask, 'demand_fuel_mj_year'] += compare_rows_group.loc[mask, 'share_fuel_1_final'] * compare_rows_group.loc[mask, 'fuel_consumption_mj_year_cohort']
                mask = compare_rows_group.fuel_2.isin(fuels_in_group)
                compare_rows_group.loc[mask, 'demand_fuel_mj_year'] += compare_rows_group.loc[mask, 'share_fuel_2_final'] * compare_rows_group.loc[mask, 'fuel_consumption_mj_year_cohort']
                mask = compare_rows_group.fuel_3.isin(fuels_in_group)
                compare_rows_group.loc[mask, 'demand_fuel_mj_year'] += compare_rows_group.loc[mask, 'share_fuel_3_final'] * compare_rows_group.loc[mask, 'fuel_consumption_mj_year_cohort']
            assert len(compare_rows_group[(compare_rows_group.demand_fuel_mj_year == 0)&(compare_rows_group.real_ships_nb>0)]) <= 0, (
                f"Zero demand_fuel_mj_year for fuel_group={fuel_group}. "
                f"ship_year_ids={compare_rows_group.loc[compare_rows_group.demand_fuel_mj_year <= 0].head()}"
            )

            # Build cumulative demand over route rows; any ship with at least one row
            # above the cap is marked for second choice.
            compare_rows_group['demand_fuel_mj_year_cumulative'] = compare_rows_group['demand_fuel_mj_year'].cumsum()
            availability_cap = fuel_group_avail[fuel_group_avail.fuel_group == fuel_group]['availability'].iloc[0]
            second_choice_ship_year_ids = compare_rows_group[
                compare_rows_group.demand_fuel_mj_year_cumulative > availability_cap
            ].ship_year_id.unique()
            
            # Update ship-level and route-level flags:
            # ship is marked unavailable if any of its route rows exceeds the cap.
            mask = df_unassigned_ships[df_unassigned_ships.ship_year_id.isin(second_choice_ship_year_ids)].index
            df_unassigned_ships.loc[mask,'fuel_chosen_available_flag'] = False
            mask = compare_rows[compare_rows.ship_year_id.isin(second_choice_ship_year_ids)].index
            compare_rows.loc[mask,'fuel_chosen_available_flag'] = False

            # Store per-ship fuel group consumption on df_unassigned_ships
            col_name = f"fuel_cons_{fuel_group}"
            ship_demand = (
                compare_rows_group
                .groupby('ship_year_id', as_index=False)['demand_fuel_mj_year']
                .sum()
                .rename(columns={'demand_fuel_mj_year': col_name})
            )
            df_unassigned_ships = df_unassigned_ships.drop(columns=[col_name], errors='ignore')
            check_len = len(df_unassigned_ships)
            df_unassigned_ships = df_unassigned_ships.merge(ship_demand, on='ship_year_id', how='left')
            assert len(df_unassigned_ships) == check_len, f"Not the same len after merge: {check_len}, {len(df_unassigned_ships)}"
            df_unassigned_ships[col_name] = df_unassigned_ships[col_name].fillna(0.0)

            if len(second_choice_ship_year_ids) > 0:
                mask = fuel_group_avail[fuel_group_avail.fuel_group == fuel_group].index
                fuel_group_avail.loc[mask, 'availability_reached'] = True
                all_fuels_available = False

        # Step 4: keep ships which have access to fuels; rerun ships marked unavailable in Step 3
        mask_first_choice = df_unassigned_ships['fuel_chosen_available_flag']

        selected_ships_iter = df_unassigned_ships[mask_first_choice].copy()
        selected_ships_iter['allocation_iter'] = allocation_iter
        selected_ships_iter['fuel_group_excluded'] = excluded_groups_label

        selected_rrc_iter = compare_rows[compare_rows['fuel_chosen_available_flag']].copy()
        selected_rrc_iter['allocation_iter'] = allocation_iter
        selected_rrc_iter['fuel_group_excluded'] = excluded_groups_label

        df_selected_ships = pd.concat((df_selected_ships, selected_ships_iter), ignore_index=True)
        df_selected_rrc_rows = pd.concat((df_selected_rrc_rows, selected_rrc_iter), ignore_index=True)
        df_unassigned_ships = df_unassigned_ships[~mask_first_choice].copy()
       
        # If the availability limits are never reached, stop there
        # Otherwise, update the inputs to the next look
        if all_fuels_available :
            break
        else:
            # with saturated fuel groups removed from allowed_fuels.
            # restrict the fuels allowed to those which have not met the availability limit
            saturated_group_fuels = FUEL_GROUPS_AVAIL[
                FUEL_GROUPS_AVAIL.fuel_group.isin(fuel_group_avail[fuel_group_avail.availability_reached]['fuel_group'].unique())
            ].fuel.unique()
            user_params_work['allowed_fuels'] = [fuel for fuel in user_params_work['allowed_fuels'] if fuel not in saturated_group_fuels]

            # Track loop-level allocation metadata on selected rows.
            excluded_groups_so_far = sorted(
                fuel_group_avail.loc[fuel_group_avail['availability_reached'], 'fuel_group'].dropna().astype(str).tolist()
            )
            excluded_groups_label = '|'.join(excluded_groups_so_far) if excluded_groups_so_far else ''
    
    # Step 5: run some checks
    missing_ship_year_ids = [x for x in df_year_active.ship_year_id.unique() if x not in df_selected_ships.ship_year_id.unique()]
    assert len(missing_ship_year_ids) == 0, f"Missing ship_year_id after the fuel availability/fuel choice loops: {df_year_active[df_year_active.ship_year_id.isin(missing_ship_year_ids)].head()}"
    assert len(df_year_active) == len(df_selected_ships), f"Longer df_selected_ships than expected: {df_selected_ships}, {df_year_active}"

    return df_selected_ships, df_selected_rrc_rows

def build_yearly_fuel_group_caps(prev_year_fuel_group_avail, df_growth_infra, year_step=1):
    min_avail_pct = 0  # minimum availability as a fraction of total fleet fuel demand
    fuel_group_avail = (
        df_growth_infra
        .merge(prev_year_fuel_group_avail, on='fuel_group', how='left')
    )
    assert len(fuel_group_avail) == len(df_growth_infra), (
        f"Row count changed after merging prev_year_fuel_group_avail: expected {len(df_growth_infra)}, got {len(fuel_group_avail)}"
    )
    assert "fuel_consumption_fleet_mj_year" in fuel_group_avail.columns, (
        f"prev_year_fuel_group_avail must contain a 'fuel_consumption_fleet_mj_year' column. "
        f"Available columns: {fuel_group_avail.columns.tolist()}"
    )

    # Compute min_avail as a % of total fleet fuel demand
    total_fleet_demand = fuel_group_avail["fuel_consumption_fleet_mj_year"].sum()
    min_avail = max(min_avail_pct * total_fleet_demand, 0.1)  # absolute floor of 0.1 MJ/year as fallback

    # Fill missing fuel groups (absent from last year's fleet) with minimum availability before computing growth
    fuel_group_avail = fuel_group_avail.copy()
    fuel_group_avail["fuel_consumption_fleet_mj_year"] = fuel_group_avail["fuel_consumption_fleet_mj_year"].fillna(min_avail).clip(lower=min_avail)

    # Compute this year's availability: last year's consumption × (1 + growth_rate_pa)^year_step
    fuel_group_avail["availability"] = (
        fuel_group_avail["fuel_consumption_fleet_mj_year"] * (1 + fuel_group_avail["growth_rate_pa"]) ** year_step
    )
    fuel_group_avail["availability"] = fuel_group_avail["availability"].clip(lower=min_avail)
    assert fuel_group_avail["availability"].notna().all(), (
        f"NaN availability after growth computation. "
        f"fuel_groups affected: {fuel_group_avail[fuel_group_avail['availability'].isna()]['fuel_group'].unique().tolist()}"
    )
    fuel_group_avail = fuel_group_avail.drop(columns=["growth_rate_pa"])

    return fuel_group_avail

def update_prev_year_fuel_group_avail(df_year_active, prev_year_fuel_group_avail, fuel_group_avail):
    """Update fuel-group availability baseline for next year (non-decreasing by group).

    Uses the per-ship fuel_cons_{group} columns written by choose_fuel_given_su.
    For saturated groups (availability_reached == True), the carried-forward
    baseline uses the availability cap rather than the lower realised consumption,
    so the growth trajectory is not artificially dampened by the constraint.
    """
    fuel_cons_cols = [c for c in df_year_active.columns if c.startswith('fuel_cons_')]
    fuel_group_avail_current = pd.DataFrame({
        'fuel_group': [c.replace('fuel_cons_', '') for c in fuel_cons_cols],
        'fuel_consumption_fleet_mj_year': [df_year_active[c].sum() for c in fuel_cons_cols],
    })

    # For saturated groups, use the availability cap instead of realised consumption
    saturated = fuel_group_avail.loc[fuel_group_avail['availability_reached'], ['fuel_group', 'availability']]
    if len(saturated) > 0:
        fuel_group_avail_current = fuel_group_avail_current.merge(saturated, on='fuel_group', how='left')
        mask = fuel_group_avail_current['availability'].notna()
        fuel_group_avail_current.loc[mask, 'fuel_consumption_fleet_mj_year'] = (
            fuel_group_avail_current.loc[mask, 'availability']
        )
        fuel_group_avail_current = fuel_group_avail_current.drop(columns=['availability'])

    prev_series = prev_year_fuel_group_avail.set_index('fuel_group')['fuel_consumption_fleet_mj_year']
    curr_series = fuel_group_avail_current.set_index('fuel_group')['fuel_consumption_fleet_mj_year']
    all_groups = prev_series.index.union(curr_series.index)
    prev_aligned = prev_series.reindex(all_groups, fill_value=0.0)
    curr_aligned = curr_series.reindex(all_groups, fill_value=0.0)
    return (
        np.maximum(prev_aligned, curr_aligned)
        .rename('fuel_consumption_fleet_mj_year')
        .reset_index()
    )


def iterate_su_market(
        df_year_active, user_params,game_year, 
        engines_design_mapping, engine_fuels_dict,
        df_drop_in_fuels_ref, df_energy_density_ref,
        df_imo_targets_data, df_fueleu_targets_data,
        df_wtt_carbon_intensity_ref, df_ttw_carbon_intensity_ref,
        df_reward_ref, df_tc_rates_ref, df_pilot_fuel_input_ref,
        ru_price_data, initial_su_price_data, fuel_price_data,
        df_growth_infra, df_capex_ref, df_retrofit_cost_ref,
        df_opex_conventional_ref, df_opex_nonconv_wages_ref,
        df_opex_nonconv_insurance_ref,
        df_opex_nonconv_ssf_ref, df_opex_nonconv_isf_ref,
        df_opex_nonconv_spares_ref, df_operational_data_ref,
        max_iterations, 
        # planning_horizon_years, 
        prev_year_fuel_group_avail,
        df_speed_ref, df_efficiency_ref, df_engine_efficiency_ref,
    ):
    """Iteratively find SU price and re‑optimise fuel choices until
    convergence.

    First, a first guess of the SU price is taken as the marginal abatement cost of the marginal SU supply vessel (the one that would just clear the market).
    However, in most cases, because the NZF is not symetrical around the direct compliance (lost space between the base and the compliance targets), most ships would rather switch to the high emitter option than to sell SU with their low carbon option.
    Therefore, this function iteratively searches for the SU price that clears the market by adjusting the price up or down based on whether there is excess supply or demand, and re-running the fuel choice optimization at each step.
    The loop stops when the SU price converges; either because it is stuck at the RU2 price (undersupply of SU), or because SU supply broadly equals to SU demand.
    Note that because the supply curve is made of few discrete steps, without corrections, the supply is almost never equal to the demand.
    For example, if the fleet is fully conventional, it will either only run on bio-MGO or on LSFO.
    Therefore, a manual correction step is added, by which the ships setting the marginal price (i.e. the last ones to switch fuels) are split randomly between SU buyers and SU sellers, so that the SU supply ~ SU demand if SU price < RU2
    """
    debug = bool(user_params.get('DEBUG', False))
    df_year_active['design_chosen'] = True

    # Build this year's fuel-group availability caps from previous-year realised demand
    # and configured growth rates. This table is reused across SU-price iterations.
    year_step = int(user_params.get('year_step', 1))
    fuel_group_avail = build_yearly_fuel_group_caps(prev_year_fuel_group_avail, df_growth_infra, year_step)
    
    # start price at previousinitial_su_price (from the scenario inputs) or the previuos year's su price
    ru2_val = ru_price_data[ru_price_data['year'] == game_year]['ru2'].values[0]
    assert not pd.isna(ru2_val), f"RU2 price for year {game_year} is NaN"
    initial_su_price = initial_su_price_data[
        (initial_su_price_data['year'] == game_year)
    ]['value'].iloc[0]

    # For bisection search: track price bounds that bracket equilibrium
    price_low = 0  # Minimum price floor
    price_high = ru2_val  # Maximum price cap
    su_price = max(min(user_params.get('su_price', initial_su_price), ru2_val), price_low)
    assert not pd.isna(su_price), f"Initial SU price for year {game_year}, scenario {user_params['scenario_number']} is NaN"
   
    # Track comprehensive history: store all fuel choices and metrics across iterations
    history_records = []
    df_year_full = pd.DataFrame()
    df_rrc_fuel_full = []
    df_year_active.engine_type.unique()
    reason_break = "max_iterations"
    for su_iter in range(max_iterations):

        df_year_active, df_rrc_fuel = choose_fuel_given_su(
            df_year_active, su_price, user_params,
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
            game_year,
            fuel_group_avail,
            max_iterations,
            df_speed_ref=df_speed_ref, df_efficiency_ref=df_efficiency_ref,
            df_engine_efficiency_ref=df_engine_efficiency_ref,
        )
        df_year_active.engine_type.unique()
        # Record this iteration's results in history
        # Store key columns: identifiers, fuel choice, and SU/RU metrics
        df_year_active['su_loop_iteration'] = su_iter
        df_rrc_fuel['su_loop_iteration'] = su_iter
        df_iter = df_year_active[[
            'ship_year_id',
            'ship_name',
            'dcf_id', 
            'engine_type', 
            'su_tco2_cohort', 
            'ru_super_emitter_no_biofuel_tco2_cohort', 
            'su_price',
            'su_loop_iteration',
        ]].copy()
        if debug:
            # collect the chosen fuels per ship/route for the debug log
            to_merge = df_rrc_fuel.groupby(['dcf_id','ship_year_id']).agg({'fuel': list}).reset_index()
            outdir = Path(user_params.get('RUN_OUTPUT_DIR', '.'))
            outdir.mkdir(parents=True, exist_ok=True)
            logpath = outdir / 'debug_log.txt'
            with open(logpath, 'a') as f:
                f.write(f"SU price at the beginning of round {su_iter}: su_price = {su_price}, price_low = {price_low}, price_high = {price_high} \n")
            to_merge = df_rrc_fuel.groupby(['dcf_id','ship_year_id']).agg({'fuel': list}).reset_index()
            check_len = len(df_iter)
            df_iter = df_iter.merge(
                to_merge,
                on = ['dcf_id','ship_year_id'],
                how='left',
            )
            assert len(df_iter) == check_len, f"Length mismatch {len(df_iter),check_len}"
            assert len(df_iter[df_iter.fuel.isna()]) == 0, f"{df_iter[df_iter.fuel.isna()]}"
        history_records.append(df_iter)

        tot_su = df_year_active['su_tco2_cohort'].sum() if 'su_tco2_cohort' in df_year_active else None
        tot_ru = df_year_active['ru_super_emitter_no_biofuel_tco2_cohort'].sum() if 'ru_super_emitter_no_biofuel_tco2_cohort' in df_year_active else None

        imbalance = abs(tot_su - tot_ru) / max(tot_ru, 1e-6)
        if debug:
            outdir = Path(user_params.get('RUN_OUTPUT_DIR', '.'))
            outdir.mkdir(parents=True, exist_ok=True)
            logpath = outdir / 'debug_log.txt'
            with open(logpath, 'a') as f:
                f.write(f"tot_su: {tot_su}, tot_ru: {tot_ru}, imbalance: {imbalance} \n ")
                # f.write("df_iter:\n")
                # f.write(df_iter.to_string(index=False))
                # f.write(f"\n{'='*80}\n")
        if imbalance < 0.05:
            df_year_full = pd.concat((df_year_full, df_year_active), ignore_index = True)
            df_rrc_fuel_full.append(df_rrc_fuel)
            reason_break = "supply_demand_eq"
            break

        # Use bisection search to narrow in on equilibrium price
        # Adjust bounds based on imbalance direction, then try midpoint
        if tot_su > tot_ru:
            # Too much supply: price is too low, move upper bound down
            price_high = su_price
        elif tot_su < tot_ru:
            # Too little supply: price is too high, move lower bound up
            price_low = su_price
        
        # Try midpoint of current bounds
        su_price_new = (price_low + price_high) / 2.0
        
        # sanity: price must be finite and within [0, ru2]
        assert not pd.isna(su_price_new), f"SU price became NaN, price_low: su_loop_iteration: {su_iter}, price low: {price_low}, price_high: {price_high}"
        assert 0 <= su_price_new <= ru2_val, f"SU price {su_price_new} outside bounds 0-{ru2_val}"
        
        # compile the results into dataframes
        df_year_full = pd.concat((df_year_full, df_year_active), ignore_index = True)
        df_rrc_fuel_full.append(df_rrc_fuel)

        if abs(su_price - su_price_new) / max(su_price_new, 1e-6) < 0.01:
            reason_break = "price_convergence"
            break

        su_price = su_price_new
    
    # Consolidate history into single dataframe
    df_history = pd.concat(history_records, ignore_index=True) if history_records else pd.DataFrame()

    if reason_break == "max_iterations":
        print(f"The SU market reached max iterations without convergence in scenario {user_params['scenario_number']}, year {df_year_full['game_year'].unique()}")

    if debug and not df_history.empty:
        print(f"Debugging iterate_su_market for scenario {user_params['scenario_number']}, game year {game_year}:")
        outdir = Path(user_params.get('RUN_OUTPUT_DIR', '.'))
        outdir.mkdir(parents=True, exist_ok=True)
        logpath = outdir / 'debug_log.txt'
        with open(logpath, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"GAME YEAR: {game_year}\n")
            f.write(f"Final SU price: {su_price}\n\n")
            f.write(f"Imbabance: {imbalance} \n")
            if reason_break == "supply_demand_eq":
                f.write(f"Reason for SU loop break: Supplied and demanded SU are sufficiently close: tot_su = {tot_su}, tot_ru = {tot_ru} \n")
            elif reason_break == "price_convergence":
                f.write(f"Reason for SU loop break: SU price is converging: su_price = {su_price}, su_price_new = {su_price_new} \n")
            elif reason_break == "max_iterations":
                f.write(f"Reason for SU loop break: Maximum iterations reached \n")
            else:
                f.write(f"Invalid reason_break:{reason_break}")
        
            f.write("Iteration History:\n")
            f.write(df_history.to_string(index=False))
            f.write(f"\n{'='*80}\n")

    # Initialize default values
    switched_choice_keys = []
    non_switched_choice_keys = _ship_choice_key(df_year_active)
    
    # if su_tot <> ru_tot, then the market is not clearing.
    # this is likely due to large portions of the fleet switching suddenly from one fuel to the other
    # to have su_tot ~ ru_tot, we need to split the last switching fuels between SU buyers and seller
    if not df_history.empty and abs(su_price - ru2_val)/ru2_val >= 0.01 and reason_break != "supply_demand_eq":
        switched_choice_keys, non_switched_choice_keys = split_marginal_su_ships_to_meet_su_market_eq(
            df_year_active, df_history, user_params
        )

    df_year_full['design_chosen'] = True
    df_year_full['su_chosen'] = False
    df_year_full['switched_to_balance_the_su_market'] = False

    # Select from intended states only:
    # - non-switched keys from final SU-loop iteration
    # - switched rows by previous (ship_year_id, dcf_id) keys returned from split_marginal...
    df_year_full['_ship_choice_key'] = _ship_choice_key(df_year_full)

    mask_non_switched = df_year_full['_ship_choice_key'].isin(non_switched_choice_keys)
    df_year_full.loc[mask_non_switched, 'su_chosen'] = True

    if switched_choice_keys:
        mask_switched = df_year_full['_ship_choice_key'].isin(switched_choice_keys)
        df_year_full.loc[mask_switched, 'su_chosen'] = True
        df_year_full.loc[mask_switched, 'switched_to_balance_the_su_market'] = True

    # Ensure that all ships have one dcf_id and only one
    assert len(df_year_full[df_year_full.su_chosen == True]) == len(df_year_full.ship_name.unique()), f"The number of chosen SU and unique ships differ: {len(df_year_full[df_year_full.su_chosen == True]), len(df_year_full.ship_name.unique())}"

    # Select the subset of ships which are chosen to go forward
    df_year_active = df_year_full[df_year_full.su_chosen == True].copy()
    if '_ship_choice_key' in df_year_active.columns:
        df_year_active = df_year_active.drop(columns=['_ship_choice_key'])
    if '_ship_choice_key' in df_year_full.columns:
        df_year_full = df_year_full.drop(columns=['_ship_choice_key'])
    df_year_active['su_price'] = su_price
    
    # verify the market cleared: SU and RU should match (within tolerance)
    sold = df_year_active['su_tco2_cohort'].sum()
    bought = df_year_active['ru_super_emitter_no_biofuel_tco2_cohort'].sum()
    if debug:
        if bought > 0:
            if abs((sold - bought)/bought) > 0.05:
                outdir = Path(user_params.get('RUN_OUTPUT_DIR', '.'))
                outdir.mkdir(parents=True, exist_ok=True)
                logpath = outdir / 'debug_log.txt'
                with open(logpath, 'a') as f:
                    f.write(f"\n Post-market SU/RU mismatch: sold {sold} vs bought {bought}, number of iterations {su_iter}, final SU price {su_price}\n")
        
    # Concatenate list of DataFrames into single DataFrame
    df_rrc_fuel_full = pd.concat(df_rrc_fuel_full, ignore_index=True) if df_rrc_fuel_full else pd.DataFrame()

    # Compute fuel availability baseline for next year from per-ship fuel_cons_{group} columns.
    prev_year_fuel_group_avail = update_prev_year_fuel_group_avail(df_year_active, prev_year_fuel_group_avail, fuel_group_avail)
    
    return df_year_active, su_price, df_year_full, df_rrc_fuel_full, prev_year_fuel_group_avail

def split_marginal_su_ships_to_meet_su_market_eq(df_year_in, df_history, user_params):
    """Allocate SU/RU trades using iteration history to identify marginal vessels.

    Marginal vessels are the ones which have switched fuels most recently and towards the opposite direction of the imbalance.
    For these vessels, we examine their last 2 fuel choices from history and
    selectively switch some vessels back to create market balance.

    Args:
        df_year_active: Current fleet state
        df_history: Complete iteration history with columns: iteration, ship_name, 
                    dcf_id, engine_type, su_tco2_cohort, ru_super_emitter_no_biofuel_tco2_cohort
        su_price: Current SU price
        user_params: User configuration

    Returns:
        Updated df_year_active with balanced SU/RU allocation
    """
    df_year_active = df_year_in.copy()
    last_choice_keys_without_switch = _ship_choice_key(df_year_active)
    debug = bool(user_params.get('DEBUG', False))

    # let's first check that a ship is noth both buying and selling SU
    should_be_empty = df_history[
        (df_history.su_tco2_cohort > 0)
        & (df_history.ru_super_emitter_no_biofuel_tco2_cohort > 0)
    ]
    assert len(should_be_empty) == 0, f"Those ships both emit and buy SU: {should_be_empty}"

    #---- 1. Filter ships which can be switched to balanced the market ----

    # we calculate the SU balance. 
    # a positive su balance means that the ship is a net emitter of SU (su_tco2_cohort > 0)
    # a negative su balance means that the ship is a net buyer of SU (ru_super_emitter_no_biofuel_tco2_cohort < 0)
    df_history['su_balance'] = df_history['su_tco2_cohort'] - df_history['ru_super_emitter_no_biofuel_tco2_cohort']
    assert len(df_history[df_history.su_balance.isna()]) == 0, f"{df_history[df_history.su_balance.isna()].head()}"

    # Identify fuel switches: for each vessel, find when it last changed fuel in the direction which is helpful
    # we identify them by sorting them which have seen a change in their SU balance
    assert len(df_history) > 0, "History is empty, cannot allocate SU buyers/sellers without any iterations"
    df_history = df_history.sort_values(['ship_name', 'su_loop_iteration'])
    target = df_year_active['su_tco2_cohort'].sum() - df_year_active['ru_super_emitter_no_biofuel_tco2_cohort'].sum()
   
    # ...existing code...
    #---- 1. Build stable reference iteration from market-level history ----
    assert len(df_history) > 0, "History is empty, cannot allocate SU buyers/sellers"

    iter_summary = (
        df_history
        .groupby('su_loop_iteration', as_index=False)
        .agg(
            su_tco2_cohort=('su_tco2_cohort', 'sum'),
            ru_tco2=('ru_super_emitter_no_biofuel_tco2_cohort', 'sum'),
            su_price=('su_price', 'mean'),
        )
    )
    iter_summary['su_balance'] = iter_summary['su_tco2_cohort'] - iter_summary['ru_tco2']

    current_iter = int(iter_summary['su_loop_iteration'].max())
    current_row = iter_summary.loc[iter_summary['su_loop_iteration'] == current_iter].iloc[0]
    current_balance = float(current_row['su_balance'])
    current_price = float(current_row['su_price'])

    # same definition as before
    target = current_balance

    # choose iterations on the opposite side of zero vs current target
    if target > 0:
        cand = iter_summary[iter_summary['su_balance'] < 0].copy()
    elif target < 0:
        cand = iter_summary[iter_summary['su_balance'] > 0].copy()
    else:
        return [], last_choice_keys_without_switch

    if cand.empty:
        if debug:
            outdir = Path(user_params.get('RUN_OUTPUT_DIR', '.'))
            outdir.mkdir(parents=True, exist_ok=True)
            with open(outdir / 'debug_log.txt', 'a') as f:
                f.write(f"No opposite-sign iteration found; target={target}\n")
        return [], last_choice_keys_without_switch

    # closest SU price to current iteration
    cand['price_gap'] = (cand['su_price'] - current_price).abs()
    cand = cand.sort_values(['price_gap', 'su_loop_iteration'])
    ref_iter = int(cand.iloc[0]['su_loop_iteration'])

    #---- 2. Identify marginal ships as changed choice between ref_iter and current_iter ----
    cols_state = [
        'ship_year_id', 'ship_name', 'dcf_id',
        'su_tco2_cohort', 'ru_super_emitter_no_biofuel_tco2_cohort'
    ]

    df_current = (
        df_history[df_history['su_loop_iteration'] == current_iter][cols_state]
        .drop_duplicates(subset=['ship_year_id'])
        .rename(columns={
            'dcf_id': 'dcf_id_current',
            'su_tco2_cohort': 'current_su',
            'ru_super_emitter_no_biofuel_tco2_cohort': 'current_ru',
        })
    )
    df_prev = (
        df_history[df_history['su_loop_iteration'] == ref_iter][cols_state]
        .drop_duplicates(subset=['ship_year_id'])
        .rename(columns={
            'dcf_id': 'prev_dcf_id',
            'su_tco2_cohort': 'prev_su',
            'ru_super_emitter_no_biofuel_tco2_cohort': 'prev_ru',
        })
    )

    states = df_current.merge(
        df_prev[['ship_year_id', 'prev_dcf_id', 'prev_su', 'prev_ru']],
        on='ship_year_id',
        how='left'
    )

    states['current_su_balance'] = states['current_su'] - states['current_ru']
    states['prev_su_balance'] = states['prev_su'] - states['prev_ru']
    states['delta'] = states['current_su_balance'] - states['prev_su_balance']

    changed = states[
        states['prev_dcf_id'].notna()
        & (states['dcf_id_current'] != states['prev_dcf_id'])
    ].copy()

    # keep only helpful deltas
    if target > 0:
        changed = changed[changed['delta'] > 0]
    else:
        changed = changed[changed['delta'] < 0]

    if changed.empty:
        if debug:
            outdir = Path(user_params.get('RUN_OUTPUT_DIR', '.'))
            outdir.mkdir(parents=True, exist_ok=True)
            with open(outdir / 'debug_log.txt', 'a') as f:
                f.write(f"No marginal ships with helpful delta. ref_iter={ref_iter}, target={target}\n")
        return [], last_choice_keys_without_switch

    marginal_ships = changed['ship_name'].unique().tolist()
    mask_marg = df_year_active['ship_name'].isin(marginal_ships)
    df_marg = df_year_active.loc[mask_marg].copy()
    assert len(df_marg) > 0, f"No marginal vessels found with ref_iter = {ref_iter}, marginal_ships = {marginal_ships}, df_history = {df_history}"

    # Non-marginal vessels
    df_nonm = df_year_active.loc[~mask_marg]
    total_su_nonm = df_nonm['su_tco2_cohort'].sum()
    total_ru_nonm = df_nonm['ru_super_emitter_no_biofuel_tco2_cohort'].sum()
    assert total_su_nonm >= 0 and total_ru_nonm >= 0, "Negative SU/RU in non-marginal fleet should have been clipped to zero"
    non_switched_choice_keys = _ship_choice_key(df_nonm)

    # Marginal ships and their current totals
    total_su_marg = df_marg['su_tco2_cohort'].sum()
    total_ru_marg = df_marg['ru_super_emitter_no_biofuel_tco2_cohort'].sum()
    
    denom = max(total_su_nonm + total_su_marg, total_ru_nonm + total_ru_marg)
    if denom == 0:
        imbalance = 0
    else:
        imbalance = (
            (total_ru_nonm + total_ru_marg) - (total_su_nonm + total_su_marg)
        )/denom

    # --- DEBUG LOGGING: Write early to capture allocation attempt ---
    if debug:
        outdir = Path(user_params.get('RUN_OUTPUT_DIR', '.'))
        outdir.mkdir(parents=True, exist_ok=True)
        logpath = outdir / 'debug_log.txt'
        with open(logpath, 'a') as f:
            f.write(f"\n[split_marginal_su_ships_to_meet_su_market_eq]\n")
            f.write(f"Previous state iteration = {ref_iter}")
            f.write(f"\n \n[split_marginal_su_ships_to_meet_su_market_eq]\n")
            f.write(f"total_su_nonm={total_su_nonm:.2f}, total_ru_nonm={total_ru_nonm:.2f}\n")
            f.write(f"total_su_marg={total_su_marg:.2f}, total_ru_marg={total_ru_marg:.2f}\n")
            f.write(f"imbalance={imbalance:.4f}\n")
            f.write(f"marginal_ships={marginal_ships}\n \n")
    
    if abs(imbalance) < 0.05:
        if debug:
            outdir = Path(user_params.get('RUN_OUTPUT_DIR', '.'))
            outdir.mkdir(parents=True, exist_ok=True)
            logpath = outdir / 'debug_log.txt'
            with open(logpath, 'a') as f:
                f.write(f"Imbalance small ({imbalance:.4f}) or no marginal vessels, skipping reallocation.\n")
        return [], last_choice_keys_without_switch

    #---- 3. Build marginal states directly from changed ----
    # changed already contains ship_year_id, ship_name, dcf_id_current, prev_dcf_id, delta, current/prev balances
    df_marg_states = changed[[
        'ship_year_id', 'ship_name', 'dcf_id_current', 'prev_dcf_id',
        'current_su_balance', 'prev_su_balance', 'delta'
    ]].copy()

    df_marg_states = df_marg_states.rename(columns={'dcf_id_current': 'dcf_id'})

    # Keep only ships present in active year frame
    df_marg_states = df_marg_states.merge(
        df_marg[['ship_year_id']].drop_duplicates(),
        on=['ship_year_id'],
        how='inner'
    )

    assert len(df_marg_states) > 0, (
        f"No marginal vessels found with ref_iter={ref_iter}, "
        f"target={target}, marginal_ships={marginal_ships}"
    )
    assert df_marg_states['prev_dcf_id'].notna().all(), "All marginal ships must have prev_dcf_id"
    
    #---- 4. Randomly allocate fuel switches to the marginal vessels ----
    # Use ship_year_id as a stable unique key for lookups.
    df_marg_states = (
        df_marg_states
        .drop_duplicates(subset=['ship_year_id'])
        .set_index('ship_year_id', drop=True)
    )

    # Draws from the scenario's own rng (not global NumPy state), so this is reproducible
    # and independent of other concurrently running scenarios.
    available_ship_year_ids = df_marg_states.index.to_numpy().copy()
    user_params['rng'].shuffle(available_ship_year_ids)

    cumulative = 0.0
    newcum = 0.0
    switched_choice_keys = []
    marginal_non_switched_choice_keys = list(zip(df_marg_states.index, df_marg_states['dcf_id']))
    
    for ship_year_id in available_ship_year_ids:
        delta_val = float(df_marg_states.at[ship_year_id, 'delta'])
        switched_choice_key = (
            ship_year_id,
            df_marg_states.at[ship_year_id, 'prev_dcf_id'],
        )
        remove_from_non_switched_key = (
            ship_year_id,
            df_marg_states.at[ship_year_id, 'dcf_id']
        )

        newcum = cumulative + delta_val
        switched_choice_keys.append(switched_choice_key)
        marginal_non_switched_choice_keys = [x for x in marginal_non_switched_choice_keys if x != remove_from_non_switched_key]

        # Check if we've reached or exceeded the target (considering sign)
        if target > 0:
            # Moving toward positive target, stop when we reach or exceed it
            if newcum < target:
                cumulative = newcum
            else:
                break
        else:
            # Moving toward negative target, stop when we reach or go below it
            if newcum > target:
                cumulative = newcum
            else:
                break
    
    non_switched_choice_keys = list(marginal_non_switched_choice_keys) + list(non_switched_choice_keys)
    
    # --- DEBUG LOGGING: Write allocation results ---
    if debug:
        outdir = Path(user_params.get('RUN_OUTPUT_DIR', '.'))
        outdir.mkdir(parents=True, exist_ok=True)
        logpath = outdir / 'debug_log.txt'
        with open(logpath, 'a') as f:
            f.write(f"\n target={target:.2f}\n")
            f.write(f"newcum={newcum:.2f}\n \n")
            f.write(f"Random order of ids: {available_ship_year_ids.tolist()}\n \n")
            f.write(f"switched_choice_keys (previous choice keys): {switched_choice_keys}\n")
            f.write(f"Non switched keys of potentially marginal ships: {marginal_non_switched_choice_keys}\n")
            f.write(f"All non switched keys: {non_switched_choice_keys}\n")
            if not df_marg_states.empty:
                f.write("\\n Marginal vessel states (helpful direction only):\\n")
                f.write(df_marg_states.to_string())
                f.write("\n \n")
        
    if len(switched_choice_keys) == 0 and debug:
        outdir = Path(user_params.get('RUN_OUTPUT_DIR', '.'))
        outdir.mkdir(parents=True, exist_ok=True)
        logpath = outdir / 'debug_log.txt'
        with open(logpath, 'a') as f:
            f.write("No vessels switched_choice_keys for switching back, but imbalance exists. Check logic for selecting helpful vessels.\n")

    return switched_choice_keys, non_switched_choice_keys