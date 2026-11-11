The repository contains the code behind :
- the Fleet Evolution Model (FEM), described in "Beyond cost-optimal transitions: behavioural decision-making and policy feedback in shipping decarbonisation" (DOI : 10.1016/j.jclepro.2026.149608) which models fleet-wide investment decisions over time.
- the transition risk metric (CRM), described in "Assessing climate change transitional risk in shipping", which computes a transition risk metric at the IMO number level.

Organisation of the folder:
fleet_adoption_game.py: orchestration of the evolution of the fleet
fem_runner.ipynb: notebook calling the scenarios to run.
fem_plotting.ipynb: plotting the results
crm_runner.ipynb: notebook running the climate risk metrics for a sample of IMO numbers and case studies
climate_risk_metric_functions.py: functions specific to the climate risk metric
default_assumptions.py: lists and hard coded value
fuel.py: functions related to fuel costs, emission factors and consumptions
load_data.py: data loading functions from the Model_Database
policy.py: functions related to policy compliance and costs (EU and IMO policies)
ship_econ.py: TCO-related functions (CAPEX, OPEX, profitability etc)
su_market.py: modelling of an SU market around the IMO's GFI
technology.py: available technology, engine type, retrofits onboard ships
utility_functions.py: cleaning and housekeeping functions

# Input assumptions
The assumptions common to both the FEM and the CRM are in data>input>Model_database.xlsx
newbuild_premium_rep_ships, rep_ships, retrofit_cost_rep_ships are the input assumptions representing the case study for the CRM paper
scenario_variables_crm contains the building of the scenario space of the CRM