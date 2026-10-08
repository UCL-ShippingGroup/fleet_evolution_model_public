import pandas as pd
import numpy as np
from datetime import datetime
from ship_econ import assign_first_capex_opex

pd.set_option('display.max_columns', None)


def assign_archetypes(df, early_fraction=0.05, *, rng):
    """Randomly assign 'early' or 'mass' archetype to each ship in *df* in-place.

    Draws from *rng* (a per-scenario np.random.Generator), not the global NumPy RNG,
    so concurrent scenarios running in different threads do not share RNG state.

    Args:
        df (pd.DataFrame): Ships DataFrame; modified in place.
        early_fraction (float): Fraction of ships assigned 'early'. Default 0.05.
        rng (np.random.Generator): Per-scenario RNG, e.g. user_params['rng'].
    """
    df['archetype'] = 'mass'
    mask = rng.random(len(df)) < early_fraction
    df.loc[mask, 'archetype'] = 'early'


# --- VESSEL ---
def load_rep_ships(user_params, model_excel_db, df_scrap_rates_ref, df_capex_ref):
    'load representative ships from model database'
    df_rep_ships = pd.read_excel(model_excel_db, sheet_name="fem_initial_fleet", engine="openpyxl")
    df_rep_ships.columns = df_rep_ships.columns.str.lower().str.replace(' ', '_')

    # Initialisation
    df_rep_ships = df_rep_ships.rename(columns={'generation': 'year_built'})
    df_rep_ships['vessel_age'] = user_params['start_year'] - df_rep_ships['year_built']
    df_rep_ships['segment_size_id'] = (df_rep_ships['imo_type'].astype(str) + '_'+ df_rep_ships['size_id'].astype(str))

    mask = df_rep_ships[df_rep_ships.ccs.isna()].index
    df_rep_ships.loc[mask,'ccs'] = ""
    mask = df_rep_ships[df_rep_ships.wind.isna()].index
    df_rep_ships.loc[mask,'wind'] = ""

    df_rep_ships['prev_engine'] = df_rep_ships['initial_fuel_design']
    df_rep_ships['engine'] = df_rep_ships['prev_engine']

    ## scrapping value and age
    df_rep_ships = df_rep_ships.merge(df_scrap_rates_ref,
        on=['imo_type', 'size_id'],
        how='left'
    )
    df_rep_ships = df_rep_ships.rename(columns ={'scrap_age':'lifetime'})
    df_rep_ships['prev_vessel_year_built'] = df_rep_ships['year_built'] - df_rep_ships['lifetime']
    df_rep_ships['prev_vessel_age'] = df_rep_ships['lifetime']
    df_rep_ships['salvage_value_usd'] = df_rep_ships['dwt'] * df_rep_ships['usd_per_dwt']

    # Assign initial CAPEX
    df_rep_ships = assign_first_capex_opex(user_params, df_rep_ships, df_capex_ref)

    df_rep_ships['real_ships_nb'] = df_rep_ships['real_ships_nb'] / df_rep_ships['modelled_ships_nb']

    # DUPLICATE ROWS ACCORDING TO modelled_ships_nb
    df_vessels = df_rep_ships.loc[
        df_rep_ships.index.repeat(df_rep_ships['modelled_ships_nb'])
    ].reset_index(drop=True)

    # Player archetypes
    # A per-scenario Generator (not the global NumPy RNG) drives this and all later
    # random draws in the run (growth-ship archetypes, su_market fuel-priority shuffles).
    # Stored on user_params so concurrent scenarios (e.g. ThreadPoolExecutor) each keep
    # their own independent RNG state instead of sharing NumPy's global state.
    assert 'random_seed' in user_params, "user_params must include 'random_seed' (set per scenario in scenarios_config)"
    assert 'early_movers_fraction' in user_params, "user_params must include 'early_movers_fraction' (set per scenario in scenarios_config)"
    user_params['rng'] = np.random.default_rng(int(user_params['random_seed']))
    assign_archetypes(df_vessels, early_fraction=float(user_params['early_movers_fraction']), rng=user_params['rng'])

    # ADD UNIQUE SHIP NAME
    df_vessels['seq'] = (
        df_vessels
        .groupby(['imo_type', 'size_id'])
        .cumcount()
        .add(1)
    )

    df_vessels['ship_name'] = (
            df_vessels['imo_type'].str.replace(" ", "-") +
            "_" +
            df_vessels['size_id'].astype(str) +
            "_" +
            df_vessels['seq'].astype(str)
    )

    df_vessels = df_vessels.drop(columns=['seq'])

    return df_vessels

def load_imo_size_map(model_excel_db):
    'load IMO size map'

    imo_size_map = pd.read_excel(model_excel_db, sheet_name="imo_size_map", engine="openpyxl")
    imo_size_map.columns = imo_size_map.columns.str.lower().str.replace(' ', '_')

    return imo_size_map


# --- POLICY ---
def load_gfi_targets_data(model_excel_db):
    """Load GFI targets data from Excel file"""
    df_gfi_targets_data = pd.read_excel(model_excel_db, sheet_name="policy_targets", engine="openpyxl")
    df_eu_compliance = pd.read_excel(model_excel_db, sheet_name="eu_compliance", engine="openpyxl")

    for df in [df_gfi_targets_data, df_eu_compliance]:
        df.columns = df.columns.str.lower().str.replace(' ', '_')

    # IMO TARGETS DATA
    df_imo_targets_data = df_gfi_targets_data[df_gfi_targets_data['policy'] == 'IMO NZF']
    # Pivot so that each variable becomes a column
    df_imo_targets_data = df_imo_targets_data.pivot_table(
        index=['year', 'scenario'],  # keep these as identifiers
        columns='variable',  # each unique variable becomes a column
        values='gfi'  # values to fill
    ).reset_index()
    df_imo_targets_data.columns.name = None
    # rename columns
    df_imo_targets_data = df_imo_targets_data.rename(columns={
        'Base': 'base_compliance_target',
        'Direct': 'direct_compliance_target',
        'ZNZ threshold': 'znz'
    })
    # Reorder columns
    df_imo_targets_data = df_imo_targets_data[
        ['year', 'scenario', 'base_compliance_target', 'direct_compliance_target', 'znz']]
    df_imo_targets_data = df_imo_targets_data.sort_values(by=['scenario', 'year'])

    # FUELEU TARGETS DATA
    df_fueleu_targets_data = df_gfi_targets_data[
        (df_gfi_targets_data['policy'] == 'FuelEU') & (df_gfi_targets_data['variable'] == 'Target')]
    df_fueleu_targets_data = df_fueleu_targets_data[['scenario', 'year', 'gfi']]

    df_fueleu_targets_data = df_fueleu_targets_data.pivot_table(
        index=['year', 'scenario'],  # keep these as identifiers
        values='gfi'  # values to fill
    ).reset_index()
    df_fueleu_targets_data = df_fueleu_targets_data.rename(columns={'gfi': 'FuelEU'})
    df_eu_compliance = df_eu_compliance[['year', 'fueleu_penalty', 'ets_price']]
    df_fueleu_targets_data = df_fueleu_targets_data.merge(df_eu_compliance, on='year', how='left')

    return df_imo_targets_data, df_fueleu_targets_data

def load_ru_price_data(model_excel_db):
    """Load RU price data from Excel file"""
    df_ru_price_data = pd.read_excel(model_excel_db, sheet_name="imo_ru", engine="openpyxl")
    df_ru_price_data.columns = df_ru_price_data.columns.str.lower().str.replace(' ', '_')

    # Pivot the table
    df_ru_price_data = df_ru_price_data.pivot_table(
        index=['year', 'scenario'],  # keep year and scenario as rows
        columns='variable',  # each variable becomes a column
        values='value'  # fill values from 'value' column
    ).reset_index()
    df_ru_price_data.columns.name = None

    # Formatting
    df_ru_price_data = df_ru_price_data.rename(columns=lambda x: x.lower())
    df_ru_price_data = df_ru_price_data.sort_values(by=['scenario', 'year'])

    return df_ru_price_data

def load_initial_su_price(model_excel_db):
    """Load RU price data from Excel file"""
    df_initial_su_price = pd.read_excel(model_excel_db, sheet_name="imo_su", engine="openpyxl")
    df_initial_su_price.columns = df_initial_su_price.columns.str.lower().str.replace(' ', '_')

    # Formatting
    df_initial_su_price = df_initial_su_price.rename(columns=lambda x: x.lower())
    df_initial_su_price = df_initial_su_price.sort_values(by=['scenario', 'year'])

    return df_initial_su_price

def load_carbon_intensity(model_excel_db):
    'Load TTW and WTT data from DB'
    # --- Read in data from DB ---
    ttw_ef = pd.read_excel(model_excel_db, sheet_name="TTW_EF")  # in g CO2e/MJ
    wtt_ef = pd.read_excel(model_excel_db, sheet_name="WTT_EF")  # in g CO2e/MJ

    # --- Clean and format data ---
    for df in [ttw_ef, wtt_ef]:
        df.columns = df.columns.str.lower().str.replace(' ', '_')

    # convert string data to lower for matching.
    for df in [ttw_ef, wtt_ef]:
        df['fuel'] = df['fuel'].str.lower().str.replace(" ", "_")

    wtt_ef = wtt_ef.pivot_table(
        index=['year', 'scenario'],  # keep year and scenario as rows
        columns='fuel',  # each fuel becomes a column
        values='value'  # fill values from 'value' column
    ).reset_index()
    wtt_ef.columns.name = None

    # Reshape the TTW emission factors
    ttw_ef = ttw_ef.melt(
        id_vars=["fuel", "engine_type"],
        value_vars=["<2024", "2024+"],
        var_name="built_before",
        value_name="ttw_gco2e_p_mj",
    )
    mask = ttw_ef[~(ttw_ef.engine_type == "All")].index
    ttw_ef.loc[mask, "engine_type"] = (
            ttw_ef.loc[mask, "engine_type"] + " " + ttw_ef.loc[mask, "built_before"]
    )
    ttw_ef = ttw_ef.drop(columns=["built_before"]).drop_duplicates()

    # --- Create engine fuel mapping ---
    fuel_engine_mapping = ttw_ef[["engine_type", "fuel"]].drop_duplicates()

    return wtt_ef, ttw_ef, fuel_engine_mapping  # df_carbon_intensity_ref,

def load_imo_nzf_reward_data(model_excel_db):
    """load IMO NZF rewards data"""

    df_reward_ref = pd.read_excel(model_excel_db, sheet_name="imo_znzreward", engine="openpyxl")
    df_reward_ref.columns = df_reward_ref.columns.str.lower().str.replace(' ', '_')

    return df_reward_ref


# --- ENGINE ---
def load_engine_fuels_dict(model_excel_db, allowed_fuels):
    """Load & process drop in fuels data from Excel DB"""

    df_drop_in_fuels_ref = pd.read_excel(model_excel_db, sheet_name="drop_in_fuels", engine="openpyxl")
    df_drop_in_fuels_ref.columns = df_drop_in_fuels_ref.columns.str.lower().str.replace(' ', '_')

    # format strings
    for col in ['fuel', 'fuel_design']:
        df_drop_in_fuels_ref[col] = df_drop_in_fuels_ref[col].str.lower().str.replace(" ", "_")

    # make pivot & format
    df_drop_in_fuels_ref = df_drop_in_fuels_ref.drop(columns='engine_type')
    df_drop_in_fuels_ref = df_drop_in_fuels_ref.drop_duplicates()
    df_drop_in_fuels_ref['drop_in_fuel'] = True

    df_drop_in_fuels_ref= df_drop_in_fuels_ref.pivot(
        index="fuel",
        columns="fuel_design",
        values="drop_in_fuel"
    )
    df_drop_in_fuels_ref.columns.name = None
    df_drop_in_fuels_ref = df_drop_in_fuels_ref.reset_index()
    df_drop_in_fuels_ref = df_drop_in_fuels_ref.fillna(False)
    df_drop_in_fuels_ref = df_drop_in_fuels_ref.rename(
        columns={'mf_ice_w_scrubber': 'conventional'}
    )
    # keep only rows for allowed fuels
    df_drop_in_fuels_ref = df_drop_in_fuels_ref[df_drop_in_fuels_ref['fuel'].isin(allowed_fuels)]

    # # --- Build dictionary of fuels for each engine ---
    engine_fuels_dict = {}

    for col in df_drop_in_fuels_ref.columns:
        if col == "fuel":
            continue
        engine_name = col
        engine_fuels_dict[engine_name] = df_drop_in_fuels_ref.loc[df_drop_in_fuels_ref[col] == True, "fuel"].tolist()

    return engine_fuels_dict, df_drop_in_fuels_ref

def load_engine_design_mapping(model_excel_db):
    'Load & process engine design mapping'

    engines_design_mapping = pd.read_excel(model_excel_db, sheet_name="engines", engine="openpyxl")
    engines_design_mapping.columns = engines_design_mapping.columns.str.lower().str.replace(' ', '_')
    engines_design_mapping = engines_design_mapping[['engine_type', 'fuel_design']]
    engines_design_mapping['fuel_design'] = engines_design_mapping['fuel_design'].str.lower().str.replace(' ','_')

    # drop the classification of other fuel designs, and the engine description
    mask = engines_design_mapping[engines_design_mapping.fuel_design != "df_ice_lng"].index
    engines_design_mapping.loc[mask, "engine_type"] = "All"
    engines_design_mapping = engines_design_mapping[
        ["fuel_design", "engine_type"]].drop_duplicates()

    mask = engines_design_mapping[engines_design_mapping.fuel_design == "df_ice_lng"].index
    engines_design_mapping.loc[mask, "engine_type"] = (
            engines_design_mapping.loc[mask, "engine_type"] + " <2024"
    )
    to_add = engines_design_mapping.loc[mask, :]
    to_add["engine_type"] = to_add["engine_type"].str.replace("<2024", "2024+")
    engines_design_mapping = pd.concat((engines_design_mapping, to_add))
    mask = engines_design_mapping[engines_design_mapping.fuel_design == "mf_ice_w_scrubber"].index
    engines_design_mapping.loc[mask,'fuel_design'] = "conventional"
    
    return engines_design_mapping

def load_fuel_price_data_usd_tonne_lshfoe(model_excel_db):
    """Load fuel price data from Excel file"""

    # Read in fuel price data from csv
    df_fuel_price_data = pd.read_excel(model_excel_db, sheet_name="fuel_prices", engine="openpyxl")
    df_fuel_price_data.columns = df_fuel_price_data.columns.str.lower().str.replace(' ', '_')
    # Remove completely blank rows
    df_fuel_price_data = df_fuel_price_data[df_fuel_price_data['scenario']!='ignore']
    df_fuel_price_data = df_fuel_price_data.dropna(how='all')

    # Clean the data - format strings and convert to numeric
    df_fuel_price_data['fuel'] = df_fuel_price_data['fuel'].str.lower().str.replace(' ', '_')
    df_fuel_price_data['value'] = pd.to_numeric(df_fuel_price_data['value'])
    df_fuel_price_data["year"] = df_fuel_price_data["year"].astype(int)

    # Convert 'USD/GJ' to USD/tonne LSHFOe
    # Select the rows where unit is 'USD/GJ', that are not already included in data
    usd_tlsfoe_scenarios = df_fuel_price_data.loc[df_fuel_price_data['unit'] == 'USD/tonne LSHFOe', 'scenario'].unique()

    usd_gj_rows = df_fuel_price_data[
        (df_fuel_price_data['unit'] == 'USD/GJ') 
        &(~df_fuel_price_data['scenario'].isin(usd_tlsfoe_scenarios))
    ].copy()
    usd_gj_rows['value'] = usd_gj_rows['value'] * 41.2
    usd_gj_rows['unit'] = 'USD/tonne LSHFOe'

    # Append the new rows to the original dataframe
    df_fuel_price_data = pd.concat([df_fuel_price_data, usd_gj_rows], ignore_index=True)

    # Select only data in USD/tonne LSHFOe
    df_fuel_price_data = df_fuel_price_data[df_fuel_price_data['unit']=='USD/tonne LSHFOe']

    # Pivot the data
    df_fuel_price_data = df_fuel_price_data.pivot(index=['year', 'scenario'], columns='fuel', values='value').reset_index()
    df_fuel_price_data.columns.name = None

    df_fuel_price_data = df_fuel_price_data.sort_values(by=['scenario', 'year'])

    return df_fuel_price_data


# --- FUEL ---

def load_energy_density_table_alt(model_excel_db):
    """Load energy density data from Excel DB"""
    df_energy_density_ref = pd.read_excel(model_excel_db, sheet_name="energy_density", engine="openpyxl")
    df_energy_density_ref.columns = df_energy_density_ref.columns.str.lower().str.replace(' ', '_')

    # reformat fuel names
    df_energy_density_ref['fuel'] = df_energy_density_ref['fuel'].str.lower().str.replace(" ", "_")

    # rename columns
    df_energy_density_ref = df_energy_density_ref.rename(columns={'density_gj_p_tfuel': 'gj_p_tfuel',
                                                                  'density_t_p_m3': 't_p_m3'})

    df_energy_density_ref["gj_p_m3"] = df_energy_density_ref["gj_p_tfuel"] * df_energy_density_ref["t_p_m3"]
    df_energy_density_ref[['gj_p_m3','t_p_m3','gj_p_tfuel']] = df_energy_density_ref[['gj_p_m3','t_p_m3','gj_p_tfuel']].round(2)

    return df_energy_density_ref

def load_pilot_fuel_input(model_excel_db):
    df_pilot_fuel_input_ref = pd.read_excel(model_excel_db, sheet_name="pilot_fuel", engine="openpyxl")
    df_pilot_fuel_input_ref.columns = df_pilot_fuel_input_ref.columns.str.lower().str.replace(' ', '_')

    df_pilot_fuel_input_ref['fuel'] = df_pilot_fuel_input_ref['fuel'].str.lower().str.replace(" ", "_")
    df_pilot_fuel_input_ref = df_pilot_fuel_input_ref.dropna()

    return df_pilot_fuel_input_ref

def load_operational_data(model_excel_db):
    'load fuel consumption data'

    df_operational_data_ref = pd.read_excel(model_excel_db, sheet_name="operational_data", engine="openpyxl")
    df_operational_data_ref.columns = df_operational_data_ref.columns.str.lower().str.replace(' ', '_')

    return df_operational_data_ref


# --- FINANCIAL ---
def load_capex_data(model_excel_db):
    'load and process capex related data (Newbuild, retrofit and capex change over time)'

    tech_name_dict = {'mf_ice_w_scrubber': 'conventional',
                      'ammonia_df': 'df_ice_ammonia',
                      'lng_df': 'df_ice_lng',
                      'methanol_df': 'df_ice_methanol'
                      }

    df_capex_ref = pd.read_excel(model_excel_db, sheet_name="newbuild_capex", engine="openpyxl")
    df_capex_reduction = pd.read_excel(model_excel_db, sheet_name="capex_reduction", engine="openpyxl")
    df_retrofit_cost_ref = pd.read_excel(model_excel_db, sheet_name="retrofitting_cost", engine="openpyxl")
    df_imo_size_map = pd.read_excel(model_excel_db, sheet_name="imo_size_map", engine="openpyxl").drop(columns='unit')

    for df in [df_capex_ref, df_capex_reduction, df_retrofit_cost_ref]:
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        df.drop(columns=df.filter(regex="^unnamed").columns,  inplace=True)

    # format capex table
    df_capex_ref = df_capex_ref.rename(
        columns=lambda col: next(
            (col.replace(k, v) for k, v in tech_name_dict.items() if k in col),
            col
        )
    )
    df_capex_ref = df_capex_ref.merge(df_imo_size_map, on=['size_id', 'imo_type'], how='left')

    # format retrofit table
    df_retrofit_cost_ref['segment_size_id'] = (df_retrofit_cost_ref['imo_type'].astype(str) + '_'+ df_retrofit_cost_ref['size_id'].astype(str))
    for col in ['initial_fuel_design', 'retrofitted_fuel_design']:
        df_retrofit_cost_ref[col] = df_retrofit_cost_ref[col].str.replace('mf_ice_w_scrubber', 'conventional')

    for col in ["initial_wind", "retrofitted_wind"]:
        if col in df_retrofit_cost_ref.columns:
            df_retrofit_cost_ref[col] = (
                df_retrofit_cost_ref[col]
                .replace({None: ""})
                .fillna("")
            )

    return df_capex_ref, df_retrofit_cost_ref, df_capex_reduction

def load_opex_data(model_excel_db):
    'load OPEX data from model database'

    df_opex_conventional_ref = pd.read_excel(model_excel_db, sheet_name="opex_conv", engine="openpyxl")
    df_opex_nonconv_wages_ref = pd.read_excel(model_excel_db, sheet_name="opex_nonconv_wages", engine="openpyxl")
    df_opex_nonconv_insurance_ref = pd.read_excel(model_excel_db, sheet_name="opex_nonconv_insurance", engine="openpyxl")
    df_opex_nonconv_ssf_ref = pd.read_excel(model_excel_db, sheet_name="opex_nonconv_ssf", engine="openpyxl")
    df_opex_nonconv_isf_ref = pd.read_excel(model_excel_db, sheet_name="opex_nonconv_isf", engine="openpyxl")
    df_opex_nonconv_spares_ref = pd.read_excel(model_excel_db, sheet_name="opex_nonconv_spares", engine="openpyxl")

    for df in [df_opex_conventional_ref, df_opex_nonconv_wages_ref, df_opex_nonconv_insurance_ref,
               df_opex_nonconv_ssf_ref, df_opex_nonconv_isf_ref, df_opex_nonconv_spares_ref]:
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        df.drop(columns=df.filter(regex="^unnamed").columns,  inplace=True)
        df.drop(columns=['vessel_type', 'size_category'], inplace=True)

    return (df_opex_conventional_ref, df_opex_nonconv_wages_ref, df_opex_nonconv_insurance_ref,
               df_opex_nonconv_ssf_ref, df_opex_nonconv_isf_ref, df_opex_nonconv_spares_ref)

def load_scrap_value_data(model_excel_db):
    'load scrap value data'

    df_scrap_rates_ref = pd.read_excel(model_excel_db, sheet_name="scrappage_data", engine="openpyxl")
    df_scrap_rates_ref.columns = df_scrap_rates_ref.columns.str.lower().str.replace(' ', '_')

    return df_scrap_rates_ref

def load_tc_rate_ref(model_excel_db):
    'load TC rates table'
    df_tc_rates_ref = pd.read_excel(model_excel_db, sheet_name="fem_initial_fleet", engine="openpyxl")
    df_tc_rates_ref.columns = df_tc_rates_ref.columns.str.lower().str.replace(' ', '_')

    df_tc_rates_ref = df_tc_rates_ref[['imo_type', 'size_id', 'tc_usd_per_day']]

    return df_tc_rates_ref


def load_reference_data(model_excel_db):
    df_imo_targets_data, df_fueleu_targets_data = load_gfi_targets_data(model_excel_db)

    (
        df_wtt_carbon_intensity_ref,
        df_ttw_carbon_intensity_ref,
        fuel_engine_mapping,
    ) = load_carbon_intensity(model_excel_db)

    return {
        "df_imo_targets_data": df_imo_targets_data,
        "df_fueleu_targets_data": df_fueleu_targets_data,
        "ru_price_data": load_ru_price_data(model_excel_db),
        "initial_su_price_data": load_initial_su_price(model_excel_db),
        "fuel_price_data": load_fuel_price_data_usd_tonne_lshfoe(model_excel_db),
        "df_operational_data_ref": load_operational_data(model_excel_db),
        "df_reward_ref": load_imo_nzf_reward_data(model_excel_db),
        "df_scrap_rates_ref": load_scrap_value_data(model_excel_db),
        "imo_size_map": load_imo_size_map(model_excel_db),
        "df_tc_rates_ref": load_tc_rate_ref(model_excel_db),
        "df_energy_density_ref": load_energy_density_table_alt(model_excel_db),
        "df_pilot_fuel_input_ref": load_pilot_fuel_input(model_excel_db),
        "df_wtt_carbon_intensity_ref": df_wtt_carbon_intensity_ref,
        "df_ttw_carbon_intensity_ref": df_ttw_carbon_intensity_ref,
        "fuel_engine_mapping": fuel_engine_mapping,
        "engines_design_mapping": load_engine_design_mapping(model_excel_db),
        "df_opex_data": load_opex_data(model_excel_db),
        "df_growth_infra": load_growth_infra(model_excel_db),
        "prev_year_fuel_group_avail": load_initial_fuel_cons(model_excel_db),
        "df_speed_ref": load_speed_reduction(model_excel_db),
        "df_efficiency_ref": load_energy_efficiency(model_excel_db),
        "df_engine_efficiency_ref": load_engine_efficiency_data(model_excel_db),
    }


def load_initial_fuel_cons(model_excel_db):
    """Load initial fuel-group consumption shares used to seed availability constraints.

    The Excel sheet ``initial_fuel_cons`` contains *shares* of total fleet
    consumption per fuel group (column ``share_fuel_cons``).  Absolute MJ
    values are computed later in ``fleet_adoption_game.main()`` using the
    actual modelled fleet's total consumption.
    """
    df_initial_fuel_cons = pd.read_excel(model_excel_db, sheet_name="initial_fuel_cons", engine="openpyxl")
    df_initial_fuel_cons.columns = df_initial_fuel_cons.columns.str.lower().str.replace(' ', '_')
    df_initial_fuel_cons["fuel_group"] = df_initial_fuel_cons["fuel_group"].astype(str).str.strip().str.lower()
    df_initial_fuel_cons["share_fuel_cons"] = pd.to_numeric(
        df_initial_fuel_cons["share_fuel_cons"], errors="coerce"
    )

    assert df_initial_fuel_cons["fuel_group"].notna().all(), "NaN fuel_group in initial_fuel_cons"
    assert df_initial_fuel_cons["share_fuel_cons"].notna().all(), (
        "NaN share_fuel_cons in initial_fuel_cons"
    )
    assert not df_initial_fuel_cons.duplicated(subset=["fuel_group"]).any(), (
        f"Duplicate fuel_group rows in initial_fuel_cons: "
        f"{df_initial_fuel_cons[df_initial_fuel_cons.duplicated(subset=['fuel_group'], keep=False)]['fuel_group'].tolist()}"
    )

    return df_initial_fuel_cons

def load_growth_infra(model_excel_db):
    """Load growth rate of infrastructure (fuel prod, bunkering) data from Excel file"""
    df_growth_infra = pd.read_excel(model_excel_db, sheet_name="growth_infra", engine="openpyxl")
    df_growth_infra.columns = df_growth_infra.columns.str.lower().str.replace(' ', '_')
    df_growth_infra = df_growth_infra.rename(columns=lambda x: x.lower())
    df_growth_infra = df_growth_infra.sort_values(by='scenario')
    df_growth_infra['fuel_group'] = df_growth_infra['fuel_group'].str.lower().str.replace(" ", "_")

    return df_growth_infra


def load_fleet_growth(model_excel_db):
    """Load annual fleet growth rates by (imo_type, scenario, year) from Excel file."""
    df_fleet_growth = pd.read_excel(model_excel_db, sheet_name="fleet_growth", engine="openpyxl")
    df_fleet_growth.columns = df_fleet_growth.columns.str.lower().str.replace(' ', '_')
    df_fleet_growth = df_fleet_growth.sort_values(by=['scenario', 'year'])
    return df_fleet_growth


# --- ENERGY EFFICIENCY / SPEED REDUCTION ---
def load_speed_reduction(model_excel_db):
    """Load speed-reduction scenario data from Excel tab ``speed_reduction``.

    Columns expected: scenario, year, speed_reduction (0–0.13 fractional reduction
    of design speed; 0 = no reduction, 0.13 = 13 % slower than design speed).
    """
    df = pd.read_excel(model_excel_db, sheet_name="speed_reduction", engine="openpyxl")
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    df = df.sort_values(by=['scenario', 'year'])
    return df


def load_energy_efficiency(model_excel_db):
    """Load energy-efficiency improvement scenario data from Excel tab ``energy_efficiency_impr``.

    Columns expected: scenario, year, energy_efficiency_impr (0–0.23 fractional
    improvement in energy use per tonne-mile; 0 = no improvement).
    """
    df = pd.read_excel(model_excel_db, sheet_name="energy_efficiency_impr", engine="openpyxl")
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    df = df.sort_values(by=['scenario', 'year'])
    return df


def load_engine_efficiency_data(model_excel_db):
    """Load engine brake-thermal-efficiency (BTE) data from Excel tab ``engine_energy_efficiency``.

    Columns expected: fuel_design, n_stroke (2 or 4), bte (fractional efficiency).
    Not scenario/year dependent — a static fuel_design x n_stroke lookup table.
    """
    df = pd.read_excel(model_excel_db, sheet_name="engine_energy_efficiency", engine="openpyxl")
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    df = df.drop_duplicates(subset=['fuel_design', 'n_stroke']).reset_index(drop=True)

    assert df['bte'].notna().all(), "NaN bte values in engine_energy_efficiency"
    assert not df.duplicated(subset=['fuel_design', 'n_stroke']).any(), (
        "Duplicate fuel_design/n_stroke rows in engine_energy_efficiency"
    )
    return df
