import pandas as pd

def update_engine_type(df_scenarios):
    '''
    Update the engine type in scenarios df.
    For LNG vessels, depending on num_strokes and year_built, update  engine_type
    '''

    df_scenarios['engine_type'] = 'All'
    df_scenarios['engine_type_label'] = ''

    # --- LNG + 2-stroke ---
    mask = (
            df_scenarios['engine'].str.contains('df_ice_lng', case=False, regex=False)
            & (df_scenarios['n_stroke'] == '2stroke')
    )
    df_scenarios.loc[mask, 'engine_type'] = 'LNG Diesel'
    df_scenarios.loc[mask, 'engine_type_label'] = '2024+'

    # --- LNG + 4-stroke + year < 2024 ---
    mask = (
            df_scenarios['engine'].str.contains('df_ice_lng', case=False, regex=False)
            & (df_scenarios['n_stroke'] == '4stroke')
            & (df_scenarios['year_built'] < 2024)
    )
    df_scenarios.loc[mask, 'engine_type'] = 'LNG Otto MSD'
    df_scenarios.loc[mask, 'engine_type_label'] = '<2024'

    # --- LNG + 4-stroke + year ≥ 2024 ---
    mask = (
            df_scenarios['engine'].str.contains('df_ice_lng', case=False, regex=False)
            & (df_scenarios['n_stroke'] == '4stroke')
            & (df_scenarios['year_built'] >= 2024)
    )
    df_scenarios.loc[mask, 'engine_type'] = 'LNG Otto MSD'
    df_scenarios.loc[mask, 'engine_type_label'] = '2024+'
    assert len(df_scenarios[(df_scenarios.engine_type.isna())|(df_scenarios.engine_type_label.isna())]) == 0, f"Empty engine types: {df_scenarios[(df_scenarios.engine_type.isna())|(df_scenarios.engine_type_label.isna())]}"

    return df_scenarios

def get_retrofit_options(df_retrofit_cost_ref, vessel_row, user_retrofit_options):

    ship = vessel_row.copy(deep=True)  # ensures it’s a standalone DataFrame/Series
    ship = ship if isinstance(ship, pd.Series) else ship.iloc[0]
    ship.initial_fuel_design = ship.prev_engine
    ship.wind = "" if pd.isna(ship.wind) or ship.wind in (None, {}, "") else ship.wind

    avail_retrofits = get_available_retrofits(ship, df_retrofit_cost_ref)
    avail_retrofits.remove(ship.initial_fuel_design)
    retrofit_options = [x for x in avail_retrofits if x in user_retrofit_options]

    return retrofit_options

# --- SHARED FUNCTIONS V2 ---

def get_available_retrofits(ship, retrofit_cost):
    """=== Available Designs ===
    # This contains the list of designs which a ship can take theoretically throughout its lifetime (through retrofits)
    # for ships without wind or CCS, this includes all DF, CCS, and one of the wind options
    # for ships with wind or CCS, they cant uninstall the wind or CCS"""

    segment_size_id = ship.segment_size_id

    if ship.wind not in (None, {}, ""):
        avail_designs = set(
            retrofit_cost[
                (retrofit_cost.segment_size_id == segment_size_id)
                & (retrofit_cost.retrofitted_wind == ship.wind)
            ].retrofitted_fuel_design.unique()
        )

    elif str(getattr(ship, "ccs", getattr(ship, "CCS", ""))).lower() == "_+_ccs":
        avail_designs = set(
            retrofit_cost[
                (retrofit_cost.segment_size_id == segment_size_id)
                & (retrofit_cost.retrofitted_fuel_design.str.contains("ccs"))
            ].retrofitted_fuel_design.unique()
        )
    else:
        avail_designs = set(
            retrofit_cost[
                (retrofit_cost.segment_size_id == segment_size_id)
            ].retrofitted_fuel_design.unique()
        )

    # Always include the ship’s starting design
    avail_designs.add(ship.initial_fuel_design)

    # Convert to sorted list for stable indexing
    avail_designs = sorted(avail_designs)

    return avail_designs
