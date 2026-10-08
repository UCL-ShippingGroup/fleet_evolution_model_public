import pandas as pd

VALUE_SEGMENTS = [
    "Oil tanker",
    "Bulk carrier",
    "Chemical tanker",
    "Cruise",
    "Liquefied gas tanker",
    "Container",
    "General Cargo",
    "Ferry-RoPax",
    "Vehicle",
    "Ro-Ro",
    'Refrigerated bulk',
]

STUDY_SEGMENTS = [
    "Oil tanker",
    "Bulk carrier",
    "Chemical tanker",
    "Liquefied gas tanker",
    "Container",
    "General Cargo",
    "Refrigerated bulk",
    "Ro-Ro",
    "Vehicle",
]

OPERATING_FUELS = pd.DataFrame(
    {
        "fuel": [
            "lsfo",
            "bio-mgo",
            "e-mgo",
            "lng",
            "bio-lng",
            "e-lng",
            "blue_ammonia",
            "e-ammonia",
            "bio-methanol",
            "e-methanol",
        ],
        "fuel_group": [
            "gas",
            "bio",
            "e-fuels",
            "gas",
            "bio",
            "e-fuels",
            "gas",
            "e-fuels",
            "bio",
            "e-fuels",
        ],
    }
)

FUEL_GROUPS_AVAIL = pd.DataFrame(
    {
        "fuel": [
            "lsfo",
            "e-mgo",
            "blue_ammonia",
            "bio-mgo",
            "lng",
            "bio-lng",
            "e-lng",
            "e-ammonia",
            "bio-methanol",
            "e-methanol",
        ],
        "fuel_group": [
            "fossil",
            "e-fuels",
            "blue",
            "bio",
            "fossil",
            "bio",
            "e-fuels",
            "e-fuels",
            "bio",
            "e-fuels",
        ],
    }
)

PILOT_FUELS = ["lsfo", "bio-mgo", "e-mgo"]

DEPRECIATION_TIME = 25 # we assume a depreciation time lower than the actual ship's lifetime, to represent the fact that shipowners assume a lower lifetime

RANGE_REDUCTION = 0.3
TIME_OFF_HIRE = 90 # days needed for retrofits
REG_REG_CLASSES = ["eu_to_eu", "eu_to_noneu", "noneu_to_noneu"]

FUEL_DESIGNS = [
    "conventional",
    "df_ice_methanol",
    "df_ice_ammonia",
    "df_ice_lng",
    "df_ice_methanol_+_rigid_sail",
    "df_ice_ammonia_+_rigid_sail",
    "df_ice_lng_+_rigid_sail",
    "conventional_+_rigid_sail",
    "df_ice_methanol_+_flettner_rotors",
    "df_ice_ammonia_+_flettner_rotors",
    "df_ice_lng_+_flettner_rotors",
    "conventional_+_flettner_rotors",
    "df_ice_methanol_+_kite",
    "df_ice_ammonia_+_kite",
    "df_ice_lng_+_kite",
    "conventional_+_kite",
    "conventional_+_ccs",
    "df_ice_lng_+_ccs",
]

FD_AVAIL_FOR_RETRO = [
    "df_ice_methanol",
    "df_ice_ammonia",
    "df_ice_lng",
    "df_ice_methanol_+_rigid_sail",
    "df_ice_ammonia_+_rigid_sail",
    "df_ice_lng_+_rigid_sail",
    "conventional_+_rigid_sail",
    "df_ice_methanol_+_kite",
    "df_ice_ammonia_+_kite",
    "df_ice_lng_+_kite",
    "conventional_+_kite",
    "conventional_+_ccs",
    "df_ice_lng_+_ccs",
]

LEARNING_FUELS = [x for x in FUEL_DESIGNS if "ammonia" in x.lower()]
HIGH_WACC_ENGINES = FD_AVAIL_FOR_RETRO
CARBON_CAPTURE_RATE = 0.27
CCS_ADD_FUEL_CONS = 0.1

WIND_ENERGY_SAVINGS = pd.DataFrame(
    {
        "wind": [
            "rigid_sail",
            "kite",
            "flettner_rotors",
        ],
        "fuel_savings": [
            0.06,
            0.04,
            0.06,
        ],
    }
)

WIND_TECHNO_LIST = ["rigid_sail", "kite", "flettner_rotors"]

COLS_SEGMENT_SIZE_ID = ["imo_type", "bunker_t_lsfo_e", "mcr","generation","dwt", "speed_exponent"]