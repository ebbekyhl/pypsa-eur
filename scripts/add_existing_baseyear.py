# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Adds existing power and heat generation capacities for initial planning
horizon.
"""

import logging
import re
from types import SimpleNamespace

import country_converter as coco
import numpy as np
import pandas as pd
import geopandas as gpd
import powerplantmatching as pm
import pypsa
import xarray as xr
from shapely import Point

from scripts._helpers import (
    configure_logging,
    sanitize_custom_columns,
    set_scenario_config,
    update_config_from_wildcards,
)
from scripts.add_electricity import load_costs, sanitize_carriers
from scripts.build_energy_totals import cartesian
from scripts.definitions.heat_system import HeatSystem
from scripts.prepare_sector_network import cluster_heat_buses, define_spatial

logger = logging.getLogger(__name__)
cc = coco.CountryConverter()
idx = pd.IndexSlice
spatial = SimpleNamespace()


def add_UK_conventional_powerplants(df_agg,df_OIM_pp,renewable_carriers,baseyear):
    """
    This function adds conventional power plants from the Open Infrastructure Map (OIM) dataset for the
    United Kingdom (UK) to the existing conventional power plant dataset (df_agg) in the PyPSA-Eur workflow.
    Parameters
    ----------
    df_agg : pd.DataFrame
        DataFrame containing existing conventional power plant data 
    df_OIM_pp : pd.DataFrame
        DataFrame containing cleaned UK power plants data from OIM
    renewable_carriers: list
        List of renewable carriers in the network
    Returns
    -------
    df_agg : pd.DataFrame
        Updated DataFrame with UK conventional power plants added
    """

    # filter UK power plants from df_agg
    df_agg_uk = df_agg.query("Country == 'GB'")

    # non-renewable UK power plants, e.g., conventional power plants
    df_agg_uk_non_RE = df_agg_uk.Fueltype[~df_agg_uk.Fueltype.isin(renewable_carriers)].index

    # drop conventional UK power plants from df_agg to replace with OIM data
    df_agg_uk_dropped = df_agg.loc[df_agg_uk_non_RE] # save for later comparison between old and new data
    df_agg.drop(index = df_agg_uk_non_RE, inplace=True)

    # get data from OIM data for UK conventional power plants only
    uk_pp_new = df_OIM_pp.loc[~df_OIM_pp.Technology.isin(renewable_carriers)]
    uk_pp_new.set_index("Name", inplace=True)

    # rename technologies 
    uk_pp_new.loc[:, "Technology"] = uk_pp_new["Technology"].replace({"biomass CHP": 'urban central solid biomass CHP',
                                                                    "biogas CHP": 'urban central biogas CHP',
                                                                    "gas CHP": 'urban central gas CHP',
                                                                    'Natural Gas': 'CCGT',
                                                                    "biomass": "urban central solid biomass CHP",
                                                                    "biogas": "urban central biogas CHP",
                                                                    "waste": "waste CHP",
                                                                    "combustion": "CCGT",
                                                                    "anaerobic_digestion": "urban central biogas CHP",
                                                                    'diesel': "other",
                                                                    "Diesel": "other",
                                                                    'tidal': "hydro", 
                                                                    "Waste": "waste CHP",
                                                                    "thermal": "waste CHP" # in dataset, only waste is categorized as thermal
                                                                    })
 
    # drop miscategorized power plants as they do not have a defined technology
    uk_pp_new = uk_pp_new.loc[~ uk_pp_new.Technology.isin(["other", 
                                                            'other generators',
                                                            ])]

    # for later use, copy Technology to Fueltype
    uk_pp_new.loc[:, "Fueltype"] = uk_pp_new["Technology"].copy()

    # info on how much capacity has changed from old to new data source 
    logger.info(f"Replacing {df_agg_uk_dropped.Capacity.sum()/1e3:<.1f} GW with {uk_pp_new.Capacity.sum()/1e3:<.1f} GW")

    # get intersect between df_agg.columns and uk_pp_new.columns
    intersecting_cols = df_agg.columns.intersection(uk_pp_new.columns)
    uk_pp_new = uk_pp_new[intersecting_cols]

    # manually add missing columns with NaN values
    missing_cols = [col for col in df_agg.columns if col not in intersecting_cols]
    uk_pp_new.loc[:, missing_cols] = np.nan
    uk_pp_new.loc[:, "resource_class"] = 0

    all_columns = df_agg.columns
    uk_pp_new = uk_pp_new[all_columns]

    lifetime_for_existing_UK_conventional_powerplants = 40 # years
    date_out = uk_pp_new["DateIn"] + lifetime_for_existing_UK_conventional_powerplants

    # where date_out is earlier than baseyear, set to baseyear + 1
    date_out_adjusted = date_out.loc[date_out < baseyear]
    date_out.loc[date_out_adjusted.index] = baseyear + 1

    uk_pp_new.loc[:, "DateOut"] = date_out

    # add updated UK power plants to df_agg
    df_agg = pd.concat([df_agg, uk_pp_new], ignore_index=False)

    return df_agg

def read_and_clean_OIM_UK_powerplants(uk_regions, baseyear):
    """
    This function reads and cleans the Open Infrastructure Map (OIM) data for UK power plants. 
    It adds information such as commission and decommission years, classifies technologies, 
    and prepares the data to be included in PyPSA-Eur. 

    Parameters
    ----------
    uk_regions : gpd.GeoDataFrame
        GeoDataFrame containing UK regions
    baseyear : int
        First year of the planning horizon
    Returns
    -------
    df_OIM_powerplants : pd.DataFrame
        DataFrame containing cleaned UK power plants data
    """

    # Data source: Open Street Map / Open Infrastructure Map
    # This data file contains the raw data from OIM, with name of powerplant, capacity, and location.
    # This has been validated for major powerplants >200 MW, for which we have also added commission and decommission years.
    # Read OIM coordinates data
    df_OIM_coords = pd.read_csv("data/data_UK/uk_powerplants_oim_with_coords.csv")
    df_OIM_coords = df_OIM_coords.loc[df_OIM_coords["Output"].dropna().index]

    # Data source: Automatic extraction of data from wiki
    # This data file contains information generated with a Python script, searching 
    # for information on commission and decommission years from Wikipedia. For this reason, some data 
    # might not be correct. Some information, validated for major powerplants >200 MW, is used instead.
    df_OIM = pd.read_csv("data/data_UK/uk_powerplants_oim_wiki.csv")
    df_OIM.loc[:, "lat"] = df_OIM_coords["Latitude"]
    df_OIM.loc[:, "lon"] = df_OIM_coords["Longitude"]
    df_OIM.rename(columns = {"output_oim_mw": "Capacity"}, inplace=True)
    df_OIM.loc[df_OIM_coords.index, "source_oim"] = df_OIM_coords["Source"] 

    # add capacity from OIM data
    df_OIM_capacity = df_OIM_coords.loc[df_OIM_coords["Output"].dropna().index]
    df_OIM.loc[df_OIM_capacity.index, "Capacity"] = df_OIM_capacity["Output"].astype(float)
    df_OIM = df_OIM.loc[df_OIM["Capacity"].dropna().index] # if no capacity is given, drop the entry

    # add commission year (where available)
    df_OIM_years_in = df_OIM_coords.loc[df_OIM_coords["Year_in"].dropna().index]
    df_OIM.loc[df_OIM_years_in.index, "commission_year"] = df_OIM_capacity["Year_in"]

    # add decommission year (where available)
    df_OIM_years_out = df_OIM_coords.loc[df_OIM_coords["Year_out"].dropna().index]
    df_OIM.loc[df_OIM_years_out.index, "decommission_year"] = df_OIM_capacity["Year_out"]

    # add storage capacity (where available)
    df_OIM_storage = df_OIM_coords.loc[df_OIM_coords["Storage"].dropna().index]
    df_OIM.loc[df_OIM_storage.index, "storage_capacity_mwh"] = df_OIM_storage["Storage"].astype(float)

    # assign names to unnamed power plants    
    df_OIM_unnamed = df_OIM.query("name == '[unnamed]'").copy()
    df_OIM_unnamed.loc[:, "name"] = df_OIM_unnamed["source_oim"] + " " + df_OIM_unnamed.index.astype(str)
    df_OIM.loc[df_OIM_unnamed.index, "name"] = df_OIM_unnamed["name"]

    # ensure only storage are included as storage and not power plants
    for storage_techs in ["water-storage", "flywheel", "battery"]:
        df_OIM_powerplants_s = df_OIM.query("type == @storage_techs")
        df_OIM.loc[df_OIM_powerplants_s.index, "facility"] = "storage"

    # check from names if storage
    df_OIM_powerplants_s = df_OIM.loc[df_OIM.name.str.contains("Storage")]
    df_OIM.loc[df_OIM_powerplants_s.index, "facility"] = "storage"

    # Read dataset with power plants under construction
    df_OIM_coords_construction = pd.read_csv("data/data_UK/uk_powerplants_oim_with_coords_construction.csv")
    df_OIM_coords_construction = df_OIM_coords_construction.loc[df_OIM_coords_construction["Output"].dropna().index]
    df_OIM_coords_construction.rename(columns = {"Latitude": "lat", 
                                                "Longitude": "lon",
                                                "Name": "name",
                                                "Output": "Capacity",
                                                "Source": "source_oim",
                                                "Completion Date": "commission_year"
                                                }, inplace=True)

    # drop entries without capacities listed
    df_OIM_coords_construction = df_OIM_coords_construction.loc[df_OIM_coords_construction["Capacity"].notna()]
    df_OIM_coords_construction.loc[:, "Capacity"] = df_OIM_coords_construction["Capacity"].astype(float)

    # set status to under construction (unless commission year is before or equal to baseyear)
    df_OIM_coords_construction.loc[:, "status"] = "under construction"
    df_OIM_coords_construction_completed = df_OIM_coords_construction.loc[df_OIM_coords_construction["commission_year"] <= baseyear]
    df_OIM_coords_construction.loc[df_OIM_coords_construction_completed.index, "status"] = "online"

    # split battery storage from power plants
    df_OIM_coords_construction.loc[:, "facility"] = "power plant"
    df_OIM_coords_construction_battery = df_OIM_coords_construction.query("source_oim == 'battery'").index
    df_OIM_coords_construction.loc[df_OIM_coords_construction_battery, "facility"] = "storage"

    # merge datasets of existing and under construction power plants
    df_OIM_powerplants = pd.concat([df_OIM, df_OIM_coords_construction], ignore_index=True)

    # if no commission year is given, set to baseyear - 1
    df_OIM_powerplants_wo_cy = df_OIM_powerplants.loc[df_OIM_powerplants["commission_year"].isna()].copy()
    df_OIM_powerplants.loc[df_OIM_powerplants_wo_cy.index, "commission_year"] = baseyear - 1

    # for renewables, we replace missing decommission years with commission year + lifetime
    lifetime = 30 # years
    df_OIM_pp_RES = df_OIM_powerplants.loc[df_OIM_powerplants.source_oim.isin(["wind", "solar"])]
    df_OIM_pp_RES_wo_dy = df_OIM_pp_RES.loc[df_OIM_pp_RES["decommission_year"].isna()]
    df_OIM_powerplants.loc[df_OIM_pp_RES_wo_dy.index, "decommission_year"] = df_OIM_pp_RES_wo_dy["commission_year"] + lifetime

    # rename technologies
    tech_dict = {'Nuclear (AGR)': "nuclear", 
                'Biomass': "biomass", 
                'Gas (CCGT)': "CCGT", 
                'Gas': "CCGT", 
                'Solar': "solar", 
                'Oil': "oil",
                'Gas (OCGT)': "OCGT", 
                'Gas (CHP)': "gas CHP", 
                'Hydro': "hydro", 
                'Waste-to-energy': "waste", 
                'diesel': "other",
                "Diesel": "other",
                'tidal': "hydro", 
                "Waste": "waste",
                'fission': "nuclear", 
                'photovoltaic': "solar", 
                "Gas": "CCGT",
                "gas": "CCGT",
                'run-of-the-river': "hydro",
                "stream": "hydro",
                }

    # rename technology types
    df_OIM_powerplants.loc[:, "type"] = df_OIM_powerplants["type"].replace(tech_dict)

    # categorize technologies
    chp_units = df_OIM_powerplants.loc[df_OIM_powerplants.name.str.contains("CHP")]
    df_OIM_powerplants.loc[chp_units.index, "type"] = chp_units.source_oim + " CHP"

    for type in ["combustion", "anaerobic_digestion", "thermal"]:
        df_OIM_pp_type = df_OIM_powerplants.query("type == '@type'")
        df_OIM_powerplants.loc[df_OIM_pp_type.index, "type"] = df_OIM_pp_type.source_oim

    # drop entries without coordinates
    pp_wo_coords = df_OIM_powerplants.loc[df_OIM_powerplants["lat"].isna()]
    pp_wo_coords = pd.concat([pp_wo_coords, df_OIM_powerplants.loc[df_OIM_powerplants["lon"].loc[df_OIM_powerplants["lon"].isna()].index]])
    pp_wo_coords = pp_wo_coords[~pp_wo_coords.index.duplicated(keep='first')]
    df_OIM_powerplants = df_OIM_powerplants.drop(pp_wo_coords.index)

    # copy type from source_oim for power plants under construction
    df_OIM_pp_uc = df_OIM_powerplants.query("status == 'under construction'")
    df_OIM_powerplants.loc[df_OIM_pp_uc.index, "type"] = df_OIM_pp_uc["source_oim"] 

    # if type is missing
    df_OIM_pp_type_missing = df_OIM_powerplants.loc[df_OIM_powerplants.type.isna()]

    # replace by "source_oim" if possible
    df_OIM_pp_type_missing_1 = df_OIM_pp_type_missing.loc[~df_OIM_pp_type_missing.source_oim.isna()]
    df_OIM_powerplants.loc[df_OIM_pp_type_missing_1.index, "type"] = df_OIM_pp_type_missing_1.source_oim

    # for remaining missing types, try to infer from name
    df_OIM_pp_type_missing_2 = df_OIM_pp_type_missing.loc[df_OIM_pp_type_missing.source_oim.isna()]

    for idx in df_OIM_pp_type_missing_2.index:
        index_name = df_OIM_pp_type_missing_2.loc[idx, "name"].split(" ")
        pp_type = "solar" if ("solar" in index_name or "Solar" in index_name) else None
        pp_type = "wind" if ("wind" in index_name or "Wind" in index_name) else pp_type

        df_OIM_powerplants.loc[idx, "type"] = pp_type

    # if still missing, remove the entry
    df_OIM_powerplants = df_OIM_powerplants.loc[df_OIM_powerplants.type.dropna().index]

    # some entries contain multiple fuel types; we categorize them as 'other'
    df_OIM_pp_multifuel = df_OIM_powerplants.loc[df_OIM_powerplants.type.str.contains(";")].copy()
    df_OIM_powerplants.loc[df_OIM_pp_multifuel.index, "type"] = "other"

    # rename some types
    df_OIM_powerplants["type"] = df_OIM_powerplants["type"].replace({"gas": "Natural Gas",
                                                                        "geothermal": "other generators"})

    # split wind into onshore and offshore based on location
    df_OIM_pp_wind = df_OIM_powerplants.query("source_oim == 'wind'").copy()
    uk_boundary = uk_regions.dissolve()
    df_OIM_pp_wind.loc[:, "within_uk"] = df_OIM_pp_wind.apply(lambda row: uk_boundary.contains(Point(row["lon"], row["lat"])), axis=1)
    df_OIM_pp_offshore_wind = df_OIM_pp_wind[df_OIM_pp_wind["within_uk"] == False].copy()
    df_OIM_pp_onshore_wind = df_OIM_pp_wind[df_OIM_pp_wind["within_uk"] == True].copy()
    df_OIM_powerplants.loc[df_OIM_pp_offshore_wind.index, "type"] = "Offshore Wind"
    df_OIM_powerplants.loc[df_OIM_pp_onshore_wind.index, "type"] = "Onshore Wind"

    # drop unused types    
    df_OIM_powerplants.drop(index = df_OIM_powerplants.query("type == 'landfill_gas'").index, inplace=True)

    # final renaming of columns to match PyPSA-Eur conventions
    df_OIM_powerplants.rename(columns = {"commission_year": "DateIn",
                                        "decommission_year": "DateOut",
                                        "type": "Technology",
                                        "source_oim": "Fueltype",
                                        "name": "Name",
                                        }, 
                                        inplace=True
                                        )

    # add new column "Country" with "GB" for all entries
    df_OIM_powerplants.loc[:, "Country"] = "GB"

    # split powerplants into CHP and PP units to match PyPSA-Eur conventions
    df_OIM_powerplants_pp = df_OIM_powerplants.query("facility == 'power plant'")
    chp_units = df_OIM_powerplants_pp[df_OIM_powerplants_pp.Technology.str.endswith("CHP")]
    pp_units = df_OIM_powerplants_pp[~df_OIM_powerplants_pp.Technology.str.endswith("CHP")]
    df_OIM_powerplants.loc[chp_units.index, "Set"] = "CHP"
    df_OIM_powerplants.loc[pp_units.index, "Set"] = "PP"

    # Add storage identifier
    df_OIM_powerplants_pp = df_OIM_powerplants.query("facility == 'storage'")
    df_OIM_powerplants.loc[df_OIM_powerplants_pp.index, "Set"] = "S"

    # final clean-up of Fueltype column
    df_OIM_powerplants.loc[:, "Fueltype"] =df_OIM_powerplants["Fueltype"].astype(str).replace({"nuclear": "Nuclear",
                                                                                                "gas": "Natural Gas",
                                                                                                "biomass": "Bioenergy",
                                                                                                "biogas": "Bioenergy",
                                                                                                "hydro": "Hydro",
                                                                                                "waste": "Waste",
                                                                                                "unknown;gas": "Natural Gas",
                                                                                                "biogas;gas;sludge": "Natural Gas",
                                                                                                "diesel": "Other",
                                                                                                "tidal": "Hydro",
                                                                                                "methane": "Natural Gas",
                                                                                                "biomass;waste": "Bioenergy",
                                                                                                "waste;biomass": "Bioenergy",
                                                                                                "biomass;gas": "Bioenergy",
                                                                                                "wind;solar": "Wind", # only applies to "Chelveston Renewable Energy Park"
                                                                                                "nan": "Other",
                                                                                                "oil": "Oil",
                                                                                                "oil;gas": "Natural Gas",
                                                                                                "geothermal": "Geothermal",
                                                                                                "solar": "Solar",
                                                                                                "wind": "Wind",
                                                                                                })

    # clean-up hydro storage technology data
    df_OIM_powerplants.loc[:, "Technology"] =df_OIM_powerplants["Technology"].astype(str).replace({"Hydro (pumped-storage)": "water-storage"})
    
    df_OIM_powerplants_pumped_hydro = df_OIM_powerplants.query("Technology == 'water-storage'")
    df_OIM_powerplants_pumped_hydro_missing_storage = df_OIM_powerplants_pumped_hydro.loc[df_OIM_powerplants_pumped_hydro["storage_capacity_mwh"].isna()]
    df_OIM_powerplants.loc[df_OIM_powerplants_pumped_hydro_missing_storage.index, "storage_capacity_mwh"] = df_OIM_powerplants_pumped_hydro_missing_storage["Capacity"] * 4 # assume 4 hours duration if no data is given

    # manually correct some special cases
    special_cases = {'Periwinkle Hall - Links Solar photovoltaic farm & Battery storage': {"facility": "power plant",
                                                                                        "Set": "PP",
                                                                                        "Name": "Periwinkle Hall - Links Solar photovoltaic farm"},

                    'Battery Point Power Station': {"Fueltype": "battery",
                                                        "Technology": "battery",
                                                        "Set": "S"},

                    "EFDA JET Fusion Flywheel": {"facility": "storage",
                                                "Set": "S"}, 
                    
                    "Bessy Bell Wind Farm": {"facility": "power plant",
                                         "Set": "PP"} 
                    }
                                                                                        
    for item, props in special_cases.items():
        df_OIM_powerplants_special_case = df_OIM_powerplants.query("Name == @item")
        for key, value in props.items():
            df_OIM_powerplants.loc[df_OIM_powerplants_special_case.index, 
                                key] = value

    # save cleaned data to csv 
    df_OIM_powerplants.drop(columns = ["type_oim", "Operator", "Method", "Wikidata"], inplace=True)
    df_OIM_powerplants.to_csv("data/data_UK/uk_powerplants_oim_cleaned.csv", index=False)

    return df_OIM_powerplants

def add_UK_gas_storage_data(n, onshore, offshore, baseyear):
    """
    
    """
    onshore_regions =  gpd.read_file(onshore)
    onshore_regions.set_index("name", inplace=True)
    onshore_regions = onshore_regions.loc[onshore_regions.index.str.contains("GB")]

    offshore_regions =  gpd.read_file(offshore)
    offshore_regions.set_index("name", inplace=True)
    offshore_regions = offshore_regions.loc[offshore_regions.index.str.contains("GB")]

    gas_storage_UK = pd.read_csv("data/data_UK/UK_gas_storage_capacity.csv")
    gas_storage_UK['points'] = gpd.points_from_xy(gas_storage_UK.lon, gas_storage_UK.lat)
    gas_storage_UK = gpd.GeoDataFrame(gas_storage_UK, geometry='points')

    # transform points (lat, lon) to the same CRS as regions
    gas_storage_UK = gas_storage_UK.set_crs(epsg=4326)  # assuming the original CRS is WGS84
    gas_storage_UK = gas_storage_UK.to_crs(onshore_regions.crs)

    joined_onshore = gpd.sjoin(
        gas_storage_UK,
        onshore_regions,
        how="left",
        predicate="within"
    )

    joined_offshore = gpd.sjoin(
        gas_storage_UK,
        offshore_regions,
        how="left",
        predicate="within"
    )

    # get indices with nan values for onshore 
    nan_onshore_indices = joined_onshore[joined_onshore['name'].isna()].index

    # check if nan_onshore_indices are in joined_offshore
    # If so, replace the nan values in joined_onshore with the corresponding values from joined_offshore
    for idx in nan_onshore_indices:
        if idx in joined_offshore.index:
            joined_onshore.at[idx, 'name'] = joined_offshore.at[idx, 'name']

    joined_onshore.set_index("name", inplace=True)

    # copy stores elements from network
    df = getattr(n, "stores")
    gas_stores = df.query("carrier == 'gas'")
    UK_gas_stores = gas_stores.query("bus.str.contains('GB')")

    # include more recent data for UK gas storage capacity
    UK_gas_stores_capacity = joined_onshore["Capacity (MWh)"].groupby(joined_onshore.index).sum()
    UK_gas_stores_capacity.index = UK_gas_stores_capacity.index + " gas Store" + f"-{baseyear}"
    uk_gas_nodes = UK_gas_stores_capacity.index.intersection(UK_gas_stores.index)
    uk_non_gas_nodes = UK_gas_stores.index.drop(uk_gas_nodes)
    df.loc[uk_gas_nodes, "e_nom"] = UK_gas_stores_capacity.loc[uk_gas_nodes]
    df.loc[uk_gas_nodes, "e_nom_min"] = UK_gas_stores_capacity.loc[uk_gas_nodes]
    df.loc[uk_non_gas_nodes, "e_nom"] = 0
    df.loc[uk_non_gas_nodes, "e_nom_min"] = 0

    logger.info(f"Total gas storage capacity in UK of {df.loc[uk_gas_nodes, "e_nom"].sum()} added!")

    # # For the base year, we allow gas storage capacity in locations outside of UK 
    # # to be expanded, to calibrate the model. This is to address any inadequate data 
    # # in countries outside of UK, which could lead to infeasible scenarios.
    # UK_gas_stores = gas_stores.query("bus.str.contains('GB')")
    # non_UK_gas_stores = gas_stores.drop(index = UK_gas_stores.index).index
    # df.loc[non_UK_gas_stores, "e_nom_extendable"] = False

def attach_to_buses(df_OIM_pp, uk_offshore_regions, uk_regions):
    oim_points = gpd.GeoDataFrame(
        df_OIM_pp,
        geometry=gpd.points_from_xy(df_OIM_pp['lon'], df_OIM_pp['lat']),
        crs=uk_offshore_regions.crs        # use the same CRS as the polygons
    )

    plants_in_regions = gpd.sjoin(
        oim_points,
        uk_offshore_regions[['name','geometry']],
        how='left',
        predicate='within'                  # or 'intersects' if you prefer
    ).rename(columns={'name':'offshore_region'}).drop(columns='index_right')

    # 3) Assign each plant to the polygon it falls within for the onshore regions
    plants_in_regions = gpd.sjoin(
        plants_in_regions,
        uk_regions[['name','geometry']],
        how='left',
        predicate='within'                  # or 'intersects' if you prefer
    ).rename(columns={'name':'onshore_region'}).drop(columns='index_right')

    plants_in_regions["offshore_region"] = plants_in_regions["offshore_region"].fillna("")
    plants_in_regions["onshore_region"] = plants_in_regions["onshore_region"].fillna("")

    # create a new column "region"
    plants_in_regions["region"] = plants_in_regions["offshore_region"] + plants_in_regions["onshore_region"]
    plants_in_regions.drop(columns=["geometry","lon","lat"], inplace=True)
    plants_in_regions.drop(index = plants_in_regions.loc[plants_in_regions.region == ""].index , inplace=True)

    df_OIM_pp.loc[:, "bus"] = plants_in_regions["region"]

    df_OIM_pp.loc[:, "Technology"] = df_OIM_pp["Technology"].replace({"Onshore Wind": "onwind",
                                                                    "Offshore Wind": "offwind-ac",
                                                                    "Solar": "solar"})

    return df_OIM_pp

def calculate_uk_fraction(df_OIM_pp, carrier, group):

    """
    This function is used to calculate nodal fractions of the total installed capacity, as fractions of the total deployment potential. 
    This is needed to address how renewable energy sources are represented in PyPSA-Eur.
    """

    # Numerator (capacity per bus)
    fraction_num = df_OIM_pp.query("Technology == @carrier").set_index("bus").sort_index().groupby("bus").Capacity.sum()
    
    # Denominator (total capacity)
    fraction_denom = fraction_num.sum()
    
    # Calculate the nodal fractions of the total installed capacity 
    fraction_installed = fraction_num / fraction_denom

    # Create new dataframe with maximum potentials per bus
    group_df = pd.DataFrame(group.p_nom_max)
    group_df["bus"] = group.bus
    p_num_max_per_bus = group_df["p_nom_max"].groupby(group_df.bus).sum()
    group_df = group_df.reset_index().set_index("bus")
    group_df.loc[p_num_max_per_bus.index, "p_nom_max_per_bus"] = p_num_max_per_bus

    # Calculate nodal fractions of the total deployment potential
    fractions_potential = (group_df["p_nom_max"] / group_df["p_nom_max_per_bus"]).values

    # allocate the nodal fractions of the total installed capacity 
    group_df.loc[fraction_installed.index, "fraction_UK_regions"] = fraction_installed
    group_df.set_index("Generator", inplace=True)

    # allocate nodal fractions of the total deployment potential
    group_df.loc[:, "fraction_potential"] = fractions_potential

    # Now, the total fraction is the product of both fractions
    group_df.loc[:, "fraction"] = group_df["fraction_UK_regions"] * group_df["fraction_potential"]

    fraction = group_df["fraction"]

    return fraction

def add_build_year_to_new_assets(n: pypsa.Network, baseyear: int) -> None:
    """
    Add build year to new assets in the network.

    Parameters
    ----------
    n : pypsa.Network
        Network to modify
    baseyear : int
        Year in which optimized assets are built
    """
    # Give assets with lifetimes and no build year the build year baseyear
    for c in n.iterate_components(["Link", "Generator", "Store"]):
        assets = c.df.index[(c.df.lifetime != np.inf) & (c.df.build_year == 0)]
        c.df.loc[assets, "build_year"] = baseyear

        # add -baseyear to name
        rename = pd.Series(c.df.index, c.df.index)
        rename[assets] += f"-{str(baseyear)}"
        c.df.rename(index=rename, inplace=True)

        # rename time-dependent
        selection = n.component_attrs[c.name].type.str.contains(
            "series"
        ) & n.component_attrs[c.name].status.str.contains("Input")
        for attr in n.component_attrs[c.name].index[selection]:
            c.pnl[attr] = c.pnl[attr].rename(columns=rename)


def add_existing_renewables(
    n: pypsa.Network,
    costs: pd.DataFrame,
    df_agg: pd.DataFrame,
    countries: list[str],
    renewable_carriers: list[str],
    df_OIM_pp: pd.DataFrame,
    uk_settings: dict[str, bool],
) -> None:
    """
    Add existing renewable capacities to conventional power plant data.

    Parameters
    ----------
    df_agg : pd.DataFrame
        DataFrame containing conventional power plant data
    costs : pd.DataFrame
        Technology cost data with 'lifetime' column indexed by technology
    n : pypsa.Network
        Network containing topology and generator data
    countries : list
        List of country codes to consider
    renewable_carriers: list
        List of renewable carriers in the network

    Returns
    -------
    None
        Modifies df_agg in-place
    """
    tech_map = {"solar": "PV", "onwind": "Onshore", "offwind-ac": "Offshore"}

    irena = pm.data.IRENASTAT().powerplant.convert_country_to_alpha2()
    irena = irena.query("Country in @countries")
    irena = irena.groupby(["Technology", "Country", "Year"]).Capacity.sum()

    irena = irena.unstack().reset_index()

    for carrier, tech in tech_map.items():
        if carrier not in renewable_carriers:
            continue
        df = (
            irena[irena.Technology.str.contains(tech)]
            .drop(columns=["Technology"])
            .set_index("Country")
        )

        # add more recent year based on OIM data (only available for UK)
        if "GB" in countries and uk_settings["uk_new_powerplants_data"]:
            df.loc["GB", "2024"] = df_OIM_pp.query("Technology == @carrier").Capacity.sum()

        df.columns = df.columns.astype(int)

        # calculate yearly differences
        df.insert(loc=0, value=0.0, column="1999")
        df = df.diff(axis=1).drop("1999", axis=1).clip(lower=0)

        # distribute capacities among generators potential (p_nom_max)
        gen_i = n.generators.query("carrier == @carrier").index
        carrier_gens = n.generators.loc[gen_i]
        res_capacities = []
        for country, group in carrier_gens.groupby(carrier_gens.bus.map(n.buses.country)):
            if country != "GB" or not uk_settings["uk_new_powerplants_data"]:
                fraction = group.p_nom_max / group.p_nom_max.sum()
            else:
                fraction = calculate_uk_fraction(df_OIM_pp, 
                                                carrier, 
                                                group)
            
            res_capacities.append(cartesian(df.loc[country], fraction))

        res_capacities = pd.concat(res_capacities, axis=1).T

        for year in res_capacities.columns:
            for gen in res_capacities.index:
                bus_bin = re.sub(f" {carrier}.*", "", gen)
                bus, bin_id = bus_bin.rsplit(" ", maxsplit=1)
                name = f"{bus_bin} {carrier}-{year}"
                capacity = res_capacities.loc[gen, year]
                if capacity > 0.0:
                    cost_key = carrier.split("-", maxsplit=1)[0]
                    df_agg.at[name, "Fueltype"] = carrier
                    df_agg.at[name, "Technology"] = carrier
                    df_agg.at[name, "Set"] = "PP"
                    df_agg.at[name, "Capacity"] = capacity
                    df_agg.at[name, "DateIn"] = year
                    df_agg.at[name, "lifetime"] = costs.at[cost_key, "lifetime"]
                    df_agg.at[name, "DateOut"] = (
                        year + costs.at[cost_key, "lifetime"] - 1
                    )
                    df_agg.at[name, "bus"] = bus
                    df_agg.at[name, "resource_class"] = bin_id

    if "GB" in countries and uk_settings["uk_new_powerplants_data"]:
        df_agg.loc[df_agg[df_agg.bus.str.contains("GB")].index,
                   "Country"] = "GB"

    df_agg["resource_class"] = df_agg["resource_class"].fillna(0)


def add_power_capacities_installed_before_baseyear(
    n: pypsa.Network,
    costs: pd.DataFrame,
    grouping_years: list[int],
    baseyear: int,
    powerplants_file: str,
    countries: list[str],
    capacity_threshold: float,
    lifetime_values: dict[str, float],
    renewable_carriers: list[str],
) -> None:
    """
    Add power generation capacities installed before base year.

    Parameters
    ----------
    n : pypsa.Network
        Network to modify
    costs : pd.DataFrame
        Technology costs
    grouping_years : list
        Intervals to group existing capacities
    baseyear : int
        Base year for analysis
    powerplants_file : str
        Path to powerplants CSV file
    countries : list
        List of countries to consider
    capacity_threshold : float
        Minimum capacity threshold
    lifetime_values : dict
        Default values for missing data
    renewable_carriers: list
        List of renewable carriers in the network
    """
    logger.debug(f"Adding power capacities installed before {baseyear}")

    df_agg = pd.read_csv(powerplants_file, index_col=0)

    rename_fuel = {
        "Hard Coal": "coal",
        "Lignite": "lignite",
        "Nuclear": "nuclear",
        "Oil": "oil",
        "OCGT": "OCGT",
        "CCGT": "CCGT",
        "Bioenergy": "urban central solid biomass CHP",
    }

    # Replace Fueltype "Natural Gas" with the respective technology (OCGT or CCGT)
    df_agg.loc[df_agg["Fueltype"] == "Natural Gas", "Fueltype"] = df_agg.loc[
        df_agg["Fueltype"] == "Natural Gas", "Technology"
    ]

    fueltype_to_drop = [
        "Hydro",
        "Wind",
        "Solar",
        "Geothermal",
        "Waste",
        "Other",
        "CCGT, Thermal",
    ]

    technology_to_drop = ["Pv", "Storage Technologies"]

    # drop unused fueltypes and technologies
    df_agg.drop(df_agg.index[df_agg.Fueltype.isin(fueltype_to_drop)], inplace=True)
    df_agg.drop(df_agg.index[df_agg.Technology.isin(technology_to_drop)], inplace=True)
    df_agg.Fueltype = df_agg.Fueltype.map(rename_fuel)

    # Intermediate fix for DateIn & DateOut
    # Fill missing DateIn
    biomass_i = df_agg.loc[df_agg.Fueltype == "urban central solid biomass CHP"].index
    mean = df_agg.loc[biomass_i, "DateIn"].mean()
    df_agg.loc[biomass_i, "DateIn"] = df_agg.loc[biomass_i, "DateIn"].fillna(int(mean))
    # Fill missing DateOut
    dateout = df_agg.loc[biomass_i, "DateIn"] + lifetime_values["lifetime"]
    df_agg.loc[biomass_i, "DateOut"] = df_agg.loc[biomass_i, "DateOut"].fillna(dateout)

    # include renewables in df_agg
    regions = gpd.read_file(snakemake.input.onshore_regions).set_index("name")
    offshore_regions = gpd.read_file(snakemake.input.offshore_regions)

    uk_regions = regions[regions.index.str.contains("GB")]
    uk_offshore_regions = offshore_regions.loc[offshore_regions["name"].str[0:2] == "GB"]
    
    df_OIM_pp_all = read_and_clean_OIM_UK_powerplants(uk_regions, baseyear)

    # attach power plants to buses
    df_OIM_pp_all_w_buses = attach_to_buses(df_OIM_pp_all, uk_offshore_regions, uk_regions.reset_index())

    # reading power plants already constructed
    df_OIM_pp_online = df_OIM_pp_all_w_buses.query("facility == 'power plant'").query("status == 'online'")
    df_OIM_pp_uc = df_OIM_pp_all_w_buses.query("facility == 'power plant'").query("status == 'under construction'")
    df_OIM_storage = df_OIM_pp_all_w_buses.query("facility == 'storage'")

    df_OIM_pp_uc.to_csv(snakemake.output.uk_brownfield_power_plant_under_construction, index=False)
    df_OIM_storage.to_csv(snakemake.output.uk_brownfield_storage, index=False)

    # get intersecting columns of df_OIM_pp_all_w_buses and powerplants
    intersecting_cols = df_OIM_pp_online.columns.intersection(df_agg.columns)
    # order intersecting columns as in powerplants
    intersecting_cols = [col for col in df_agg.columns if col in intersecting_cols]
    df_OIM_pp = df_OIM_pp_online[intersecting_cols]

    # missing columns in df_OIM_pp_all that are in powerplants
    missing_cols = [col for col in df_agg.columns if col not in intersecting_cols]
    df_OIM_pp.loc[:, missing_cols] = np.nan
    df_OIM_pp.loc[:, missing_cols] = df_OIM_pp[missing_cols].astype(df_agg[missing_cols].dtypes)

    uk_settings = snakemake.params["uk_settings"]

    add_existing_renewables(
        df_agg=df_agg,
        costs=costs,
        n=n,
        countries=countries,
        renewable_carriers=renewable_carriers,
        df_OIM_pp=df_OIM_pp,
        uk_settings=uk_settings,
    )

    # replace powerplants UK
    if uk_settings["uk_new_powerplants_data"]:
        df_agg = add_UK_conventional_powerplants(df_agg,df_OIM_pp,renewable_carriers,baseyear)

    # drop assets which are already phased out / decommissioned
    phased_out = df_agg[df_agg["DateOut"] < baseyear].index
    df_agg.drop(phased_out, inplace=True)

    df_agg["DateIn"] = df_agg.DateIn.fillna(max(grouping_years))

    newer_assets = (df_agg.DateIn > max(grouping_years)).sum()
    if newer_assets:
        logger.warning(
            f"There are {newer_assets} assets with build year "
            f"after last power grouping year {max(grouping_years)}. "
            "These assets are dropped and not considered."
            "Consider to redefine the grouping years to keep them."
        )
        to_drop = df_agg[df_agg.DateIn > max(grouping_years)].index
        df_agg.drop(to_drop, inplace=True)

    df_agg["grouping_year"] = np.take(
        grouping_years, np.digitize(df_agg.DateIn, grouping_years, right=True)
    )

    # calculate (adjusted) remaining lifetime before phase-out (+1 because assuming
    # phase out date at the end of the year)
    df_agg["lifetime"] = df_agg.DateOut - df_agg["grouping_year"] + 1

    df = df_agg.pivot_table(
        index=["grouping_year", "Fueltype", "resource_class"],
        columns="bus",
        values="Capacity",
        aggfunc="sum",
    )

    lifetime = df_agg.pivot_table(
        index=["grouping_year", "Fueltype", "resource_class"],
        columns="bus",
        values="lifetime",
        aggfunc="mean",  # currently taken mean for clustering lifetimes
    )

    carrier = {
        "OCGT": "gas",
        "CCGT": "gas",
        "coal": "coal",
        "oil": "oil",
        "lignite": "lignite",
        "nuclear": "uranium",
        "urban central solid biomass CHP": "biomass",
        "urban central gas CHP": "gas",
        "urban central biogas CHP": "gas",
        "waste CHP": "msw",
    }

    for grouping_year, generator, resource_class in df.index:
        # capacity is the capacity in MW at each node for this
        capacity = df.loc[grouping_year, generator, resource_class]
        capacity = capacity[~capacity.isna()]
        capacity = capacity[capacity > capacity_threshold]
        suffix = "-ac" if generator == "offwind" else ""
        name_suffix = f" {generator}{suffix}-{grouping_year}"
        asset_i = capacity.index + name_suffix
        
        if generator in ["solar", "onwind", "offwind-ac"]:
            asset_i = capacity.index + " " + resource_class + name_suffix
            name_suffix = " " + resource_class + name_suffix
            cost_key = generator.split("-")[0]
            # to consider electricity grid connection costs or a split between
            # solar utility and rooftop as well, rather take cost assumptions
            # from existing network than from the cost database
            capital_cost = n.generators.loc[
                n.generators.carrier == generator + suffix, "capital_cost"
            ].mean()
            marginal_cost = n.generators.loc[
                n.generators.carrier == generator + suffix, "marginal_cost"
            ].mean()
            # check if assets are already in network (e.g. for 2020)
            already_build = n.generators.index.intersection(asset_i)
            new_build = asset_i.difference(n.generators.index)

            # this is for the year 2020
            if not already_build.empty:
                n.generators.loc[already_build, "p_nom"] = n.generators.loc[
                    already_build, "p_nom_min"
                ] = capacity.loc[already_build.str.replace(name_suffix, "")].values
            new_capacity = capacity.loc[new_build.str.replace(name_suffix, "")]

            name_suffix_by = f" {resource_class} {generator}{suffix}-{baseyear}"
            p_max_pu = n.generators_t.p_max_pu[capacity.index + name_suffix_by]

            if not new_build.empty:
                n.add(
                    "Generator",
                    new_capacity.index,
                    suffix=name_suffix,
                    bus=new_capacity.index,
                    carrier=generator,
                    p_nom=new_capacity,
                    marginal_cost=marginal_cost,
                    capital_cost=capital_cost,
                    efficiency=costs.at[cost_key, "efficiency"],
                    p_max_pu=p_max_pu.rename(columns=n.generators.bus),
                    build_year=grouping_year,
                    lifetime=costs.at[cost_key, "lifetime"],
                )

        else:
            bus0 = vars(spatial)[carrier[generator]].nodes
            if "EU" not in vars(spatial)[carrier[generator]].locations:
                bus0 = bus0.intersection(capacity.index + " " + carrier[generator])

            # check for missing bus
            missing_bus = pd.Index(bus0).difference(n.buses.index)
            if not missing_bus.empty:
                logger.info(f"add buses {bus0}")
                n.add(
                    "Bus",
                    bus0,
                    carrier=generator,
                    location=vars(spatial)[carrier[generator]].locations,
                    unit="MWh_el",
                )

            already_build = n.links.index.intersection(asset_i)
            new_build = asset_i.difference(n.links.index)
            try:
                lifetime_assets = lifetime.loc[
                    grouping_year, generator, resource_class
                ].dropna()
            except:
                print(f"Missing lifetime for {grouping_year}, {generator}, {resource_class}")
                continue

            # this is for the year 2020
            if not already_build.empty:
                n.links.loc[already_build, "p_nom_min"] = capacity.loc[
                    already_build.str.replace(name_suffix, "")
                ].values

            if not new_build.empty:
                new_capacity = capacity.loc[new_build.str.replace(name_suffix, "")]

                # get indices of new_capacity that is in lifetime_assets.index
                matching_capacity_index = new_capacity.index.intersection(lifetime_assets.index)
                missing_lifetime = new_capacity.index.difference(matching_capacity_index)
                lifetime_assets_new_capacity = lifetime_assets.loc[matching_capacity_index]

                if not missing_lifetime.empty:
                    # add default lifetime if missing
                    default_lifetime = 30
                    lifetime_assets_new_capacity = pd.concat([
                        lifetime_assets_new_capacity,
                        pd.Series(default_lifetime, index=missing_lifetime)
                    ])

                # align indices of lifetime_assets_new_capacity with new_capacity
                lifetime_assets_new_capacity = lifetime_assets_new_capacity.loc[new_capacity.index]

                if generator not in ["urban central solid biomass CHP", 
                                      "urban central biogas CHP",
                                      "urban central gas CHP",
                                      "waste CHP"]:
                    n.add(
                        "Link",
                        new_capacity.index,
                        suffix=name_suffix,
                        bus0=bus0,
                        bus1=new_capacity.index,
                        bus2="co2 atmosphere",
                        carrier=generator,
                        marginal_cost=costs.at[generator, "efficiency"]
                        * costs.at[generator, "VOM"],  # NB: VOM is per MWel
                        capital_cost=costs.at[generator, "efficiency"]
                        * costs.at[
                            generator, "capital_cost"
                        ],  # NB: fixed cost is per MWel
                        p_nom=new_capacity / costs.at[generator, "efficiency"],
                        efficiency=costs.at[generator, "efficiency"],
                        efficiency2=costs.at[carrier[generator], "CO2 intensity"],
                        build_year=grouping_year,
                        lifetime=lifetime_assets_new_capacity,
                    )
                else:
                    spatial_dic = {"urban central solid biomass CHP": spatial.biomass.df.loc[new_capacity.index]["nodes"].values,
                                    "urban central biogas CHP": spatial.gas.df.loc[new_capacity.index]["nodes"].values,
                                    "urban central gas CHP": spatial.gas.df.loc[new_capacity.index]["nodes"].values,
                                    "waste CHP":  spatial.msw.df.loc[new_capacity.index]["nodes"].values}
                    
                    key = "central solid biomass CHP" # for cost technology lookup, we assume the same cost as for biomass CHP
                    central_heat = n.buses.query(
                        "carrier == 'urban central heat'"
                    ).location.unique()
                    heat_buses = new_capacity.index.map(
                        lambda i: i + " urban central heat" if i in central_heat else ""
                    )

                    n.add(
                        "Link",
                        new_capacity.index,
                        suffix=name_suffix,
                        bus0=spatial_dic[generator],
                        bus1=new_capacity.index,
                        bus2=heat_buses,
                        carrier=generator,
                        p_nom=new_capacity / costs.at[key, "efficiency"],
                        capital_cost=costs.at[key, "capital_cost"]
                        * costs.at[key, "efficiency"],
                        marginal_cost=costs.at[key, "VOM"],
                        efficiency=costs.at[key, "efficiency"],
                        build_year=grouping_year,
                        efficiency2=costs.at[key, "efficiency-heat"],
                        lifetime=lifetime_assets_new_capacity,
                    )
        # check if existing capacities are larger than technical potential
        existing_large = n.generators[
            n.generators["p_nom_min"] > n.generators["p_nom_max"]
        ].index
        if len(existing_large):
            logger.warning(
                f"Existing capacities larger than technical potential for {existing_large},\
                           adjust technical potential to existing capacities"
            )
            n.generators.loc[existing_large, "p_nom_max"] = n.generators.loc[
                existing_large, "p_nom_min"
            ]

def add_storage_capacities_installed_before_baseyear(n, baseyear):
    """
    This function adds exosting electricity storage capacities for UK.
    It currently includes pumped hydro storage and battery storage.
    Parameters
    ----------
    n : pypsa.Network
        Network to modify
    baseyear : int
        Base year for analysis
    Returns
    -------
    n : pypsa.Network
        Modified network with existing storage capacities added 
    """
    # read existing storage power plants from cleaned OIM data
    df_OIM_storage = pd.read_csv(snakemake.output.uk_brownfield_storage)

    # only consider storage already constructed and online
    df_OIM_storage_online = df_OIM_storage.query("status == 'online'")
    battery = df_OIM_storage_online.query("Technology == 'battery'")

    battery = battery.groupby("bus").agg({"Capacity": "sum",
                                "DateIn": "mean",
                                "DateOut": "mean"})

    battery.index = battery.index + " battery discharger-" + str(baseyear)
    phs_power = df_OIM_storage_online.query("Technology == 'water-storage'")[["Capacity", "bus"]].groupby("bus").sum()
    phs_reservoir = df_OIM_storage_online.query("Technology == 'water-storage'")[["storage_capacity_mwh", "bus"]].groupby("bus").sum()
    max_hours = phs_reservoir["storage_capacity_mwh"] / phs_power["Capacity"]

    # read pumped hydro storage in PyPSA-Eur network
    n_phs = n.storage_units.query("carrier == 'PHS'")
    n_UK_phs = n_phs.loc[n_phs.index.str.contains("GB")]

    # overwrite PHS capacities with existing ones from OIM data
    phs_power_df = pd.DataFrame(columns = n_UK_phs.columns, index = phs_power.index + " PHS")
    phs_power_df.loc[:,:] = n_UK_phs.values
    phs_power_df.loc[:, "bus"] = phs_power.index
    phs_power_df.loc[:, "p_nom"] = phs_power["Capacity"].values
    phs_power_df.loc[:, "max_hours"] = max_hours.values
    phs_power_df.index.name = "StorageUnit"

    # attach to network
    n.remove("StorageUnit", n_UK_phs.index)
    n.add("StorageUnit", phs_power_df.index, **phs_power_df.T.to_dict(orient="index"))

    # update p_nom_min for existing battery storage units (if they exist)
    existing_batteries = n.links.index.intersection(battery.index)
    battery_duration = 6 # assume 6 hours discharge time for battery storage units
    if not existing_batteries.empty:
        
        # Update discharging power capacity
        n.links.loc[battery.index, 
                    "p_nom_min"] = battery["Capacity"].values
        
        # Update energy capacity
        n.stores.loc[existing_batteries.str.replace(" discharger", ""), 
                     "e_nom_min"] = battery["Capacity"].values * battery_duration

        logger.info(f"Updated existing battery storage units: {existing_batteries.tolist()}")

    else:
        logger.info("No existing battery storage units found in the network to update.")
    return n

def get_efficiency(
    heat_system: HeatSystem,
    carrier: str,
    nodes: pd.Index,
    efficiencies: dict[str, float],
    costs: pd.DataFrame,
) -> pd.Series | float:
    """
    Computes the heating system efficiency based on the sector and carrier
    type.

    Parameters
    ----------
    heat_system : object
    carrier : str
        The type of fuel or energy carrier (e.g., 'gas', 'oil').
    nodes : pandas.Series
        A pandas Series containing node information used to match the heating efficiency data.
    efficiencies : dict
        A dictionary containing efficiency values for different carriers and sectors.
    costs : pandas.DataFrame
        A DataFrame containing boiler cost and efficiency data for different heating systems.

    Returns
    -------
    efficiency : pandas.Series or float
        A pandas Series mapping the efficiencies based on nodes for residential and services sectors, or a single
        efficiency value for other heating systems (e.g., urban central).

    Notes
    -----
    - For residential and services sectors, efficiency is mapped based on the nodes.
    - For other sectors, the default boiler efficiency is retrieved from the `costs` database.
    """

    if heat_system.value == "urban central":
        boiler_costs_name = getattr(heat_system, f"{carrier}_boiler_costs_name")
        efficiency = costs.at[boiler_costs_name, "efficiency"]
    elif heat_system.sector.value == "residential":
        key = f"{carrier} residential space efficiency"
        efficiency = nodes.str[:2].map(efficiencies[key])
    elif heat_system.sector.value == "services":
        key = f"{carrier} services space efficiency"
        efficiency = nodes.str[:2].map(efficiencies[key])
    else:
        raise ValueError(f"Heat system {heat_system} not defined.")

    return efficiency


def add_heating_capacities_installed_before_baseyear(
    n: pypsa.Network,
    costs: pd.DataFrame,
    baseyear: int,
    grouping_years: list[int],
    existing_capacities: pd.DataFrame,
    heat_pump_cop: xr.DataArray,
    heat_pump_source_types: dict[str, list[str]],
    efficiency_file: str,
    use_time_dependent_cop: bool,
    default_lifetime: int,
    energy_totals_year: int,
    capacity_threshold: float,
    use_electricity_distribution_grid: bool,
) -> None:
    """
    Add heating capacities installed before base year.

    Parameters
    ----------
    n : pypsa.Network
        Network to modify
    costs : pd.DataFrame
        Technology costs
    baseyear : int
        Base year for analysis
    grouping_years : list
        Intervals to group capacities
    heat_pump_cop : xr.DataArray
        Heat pump coefficients of performance
    use_time_dependent_cop : bool
        Use time-dependent COPs
    heating_default_lifetime : int
        Default lifetime for heating systems
    existing_capacities : pd.DataFrame
        Existing heating capacity distribution
    heat_pump_source_types : dict
        Heat pump sources by system type
    efficiency_file : str
        Path to heating efficiencies file
    energy_totals_year : int
        Year for energy totals
    capacity_threshold : float
        Minimum capacity threshold
    use_electricity_distribution_grid : bool
        Whether to use electricity distribution grid
    """
    logger.debug(f"Adding heating capacities installed before {baseyear}")

    # Load heating efficiencies
    heating_efficiencies = pd.read_csv(efficiency_file, index_col=[1, 0]).loc[
        energy_totals_year
    ]

    ratios = []
    valid_grouping_years = []

    for heat_system in existing_capacities.columns.get_level_values(0).unique():
        heat_system = HeatSystem(heat_system)

        nodes = pd.Index(
            n.buses.location[n.buses.index.str.contains(f"{heat_system} heat")]
        )

        if (
            not heat_system == HeatSystem.URBAN_CENTRAL
        ) and use_electricity_distribution_grid:
            nodes_elec = nodes + " low voltage"
        else:
            nodes_elec = nodes

            too_large_grouping_years = [
                gy for gy in grouping_years if gy >= int(baseyear)
            ]
            if too_large_grouping_years:
                logger.warning(
                    f"Grouping years >= baseyear are ignored. Dropping {too_large_grouping_years}."
                )
            valid_grouping_years = pd.Series(
                [
                    int(grouping_year)
                    for grouping_year in grouping_years
                    if int(grouping_year) + default_lifetime > int(baseyear)
                    and int(grouping_year) < int(baseyear)
                ]
            )

            assert valid_grouping_years.is_monotonic_increasing

            if len(valid_grouping_years) == 0:
                logger.warning(
                    f"No valid grouping years found for {heat_system}. "
                    "No existing capacities will be added."
                )
                ratios = []
            else:
                # get number of years of each interval
                _years = valid_grouping_years.diff()
                # Fill NA from .diff() with value for the first interval
                _years[0] = valid_grouping_years[0] - baseyear + default_lifetime
                # Installation is assumed to be linear for the past
                ratios = _years / _years.sum()

        for ratio, grouping_year in zip(ratios, valid_grouping_years):
            # Add heat pumps
            for heat_source in heat_pump_source_types[heat_system.system_type.value]:
                costs_name = heat_system.heat_pump_costs_name(heat_source)

                efficiency = (
                    heat_pump_cop.sel(
                        heat_system=heat_system.system_type.value,
                        heat_source=heat_source,
                        name=nodes,
                    )
                    .to_pandas()
                    .reindex(index=n.snapshots)
                    if use_time_dependent_cop
                    else costs.at[costs_name, "efficiency"]
                )

                n.add(
                    "Link",
                    nodes,
                    suffix=f" {heat_system} {heat_source} heat pump-{grouping_year}",
                    bus0=nodes_elec,
                    bus1=nodes + " " + heat_system.value + " heat",
                    carrier=f"{heat_system} {heat_source} heat pump",
                    efficiency=efficiency,
                    capital_cost=costs.at[costs_name, "efficiency"]
                    * costs.at[costs_name, "capital_cost"],
                    p_nom=existing_capacities.loc[
                        nodes, (heat_system.value, f"{heat_source} heat pump")
                    ]
                    * ratio
                    / costs.at[costs_name, "efficiency"],
                    build_year=int(grouping_year),
                    lifetime=costs.at[costs_name, "lifetime"],
                )

            # add resistive heater, gas boilers and oil boilers
            n.add(
                "Link",
                nodes,
                suffix=f" {heat_system} resistive heater-{grouping_year}",
                bus0=nodes_elec,
                bus1=nodes + " " + heat_system.value + " heat",
                carrier=heat_system.value + " resistive heater",
                efficiency=costs.at[
                    heat_system.resistive_heater_costs_name, "efficiency"
                ],
                capital_cost=(
                    costs.at[heat_system.resistive_heater_costs_name, "efficiency"]
                    * costs.at[heat_system.resistive_heater_costs_name, "capital_cost"]
                ),
                p_nom=(
                    existing_capacities.loc[
                        nodes, (heat_system.value, "resistive heater")
                    ]
                    * ratio
                    / costs.at[heat_system.resistive_heater_costs_name, "efficiency"]
                ),
                build_year=int(grouping_year),
                lifetime=costs.at[heat_system.resistive_heater_costs_name, "lifetime"],
            )

            efficiency = get_efficiency(
                heat_system, "gas", nodes, heating_efficiencies, costs
            )

            n.add(
                "Link",
                nodes,
                suffix=f" {heat_system} gas boiler-{grouping_year}",
                bus0="EU gas" if "EU gas" in spatial.gas.nodes else nodes + " gas",
                bus1=nodes + " " + heat_system.value + " heat",
                bus2="co2 atmosphere",
                carrier=heat_system.value + " gas boiler",
                efficiency=efficiency,
                efficiency2=costs.at["gas", "CO2 intensity"],
                capital_cost=(
                    costs.at[heat_system.gas_boiler_costs_name, "efficiency"]
                    * costs.at[heat_system.gas_boiler_costs_name, "capital_cost"]
                ),
                p_nom=(
                    existing_capacities.loc[nodes, (heat_system.value, "gas boiler")]
                    * ratio
                    / costs.at[heat_system.gas_boiler_costs_name, "efficiency"]
                ),
                build_year=int(grouping_year),
                lifetime=costs.at[heat_system.gas_boiler_costs_name, "lifetime"],
            )

            efficiency = get_efficiency(
                heat_system, "oil", nodes, heating_efficiencies, costs
            )

            n.add(
                "Link",
                nodes,
                suffix=f" {heat_system} oil boiler-{grouping_year}",
                bus0=spatial.oil.nodes,
                bus1=nodes + " " + heat_system.value + " heat",
                bus2="co2 atmosphere",
                carrier=heat_system.value + " oil boiler",
                efficiency=efficiency,
                efficiency2=costs.at["oil", "CO2 intensity"],
                capital_cost=costs.at[heat_system.oil_boiler_costs_name, "efficiency"]
                * costs.at[heat_system.oil_boiler_costs_name, "capital_cost"],
                p_nom=(
                    existing_capacities.loc[nodes, (heat_system.value, "oil boiler")]
                    * ratio
                    / costs.at[heat_system.oil_boiler_costs_name, "efficiency"]
                ),
                build_year=int(grouping_year),
                lifetime=costs.at[
                    f"{heat_system.central_or_decentral} gas boiler", "lifetime"
                ],
            )

            # delete links with p_nom=nan corresponding to extra nodes in country
            n.remove(
                "Link",
                [
                    index
                    for index in n.links.index.to_list()
                    if str(grouping_year) in index and np.isnan(n.links.p_nom[index])
                ],
            )

            # delete links with capacities below threshold
            n.remove(
                "Link",
                [
                    index
                    for index in n.links.index.to_list()
                    if str(grouping_year) in index
                    and n.links.p_nom[index] < capacity_threshold
                ],
            )


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "add_existing_baseyear",
            configfiles="config/test/config.myopic.yaml",
            clusters="5",
            opts="",
            sector_opts="",
            planning_horizons=2030,
        )

    configure_logging(snakemake)  # pylint: disable=E0606
    set_scenario_config(snakemake)

    update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    options = snakemake.params.sector

    renewable_carriers = snakemake.params.carriers

    baseyear = snakemake.params.baseyear

    n = pypsa.Network(snakemake.input.network)

    # define spatial resolution of carriers
    spatial = define_spatial(n.buses[n.buses.carrier == "AC"].index, options)
    add_build_year_to_new_assets(n, baseyear)

    Nyears = n.snapshot_weightings.generators.sum() / 8760.0
    costs = load_costs(
        snakemake.input.costs,
        snakemake.params.costs,
        nyears=Nyears,
    )

    grouping_years_power = snakemake.params.existing_capacities["grouping_years_power"]
    grouping_years_heat = snakemake.params.existing_capacities["grouping_years_heat"]
    add_power_capacities_installed_before_baseyear(
        n=n,
        costs=costs,
        grouping_years=grouping_years_power,
        baseyear=baseyear,
        powerplants_file=snakemake.input.powerplants,
        countries=snakemake.config["countries"],
        capacity_threshold=snakemake.params.existing_capacities["threshold_capacity"],
        lifetime_values=snakemake.params.costs["fill_values"],
        renewable_carriers=renewable_carriers,
    )

    n = add_storage_capacities_installed_before_baseyear(n, baseyear)

    if options["heating"]:
        # one could use baseyear here instead (but dangerous if no data)
        fn = snakemake.input.heating_efficiencies
        year = int(snakemake.params["energy_totals_year"])
        heating_efficiencies = pd.read_csv(fn, index_col=[1, 0]).loc[year]

        add_heating_capacities_installed_before_baseyear(
            n=n,
            costs=costs,
            baseyear=baseyear,
            grouping_years=grouping_years_heat,
            heat_pump_cop=xr.open_dataarray(snakemake.input.cop_profiles),
            use_time_dependent_cop=options["time_dep_hp_cop"],
            default_lifetime=snakemake.params.existing_capacities[
                "default_heating_lifetime"
            ],
            existing_capacities=pd.read_csv(
                snakemake.input.existing_heating_distribution,
                header=[0, 1],
                index_col=0,
            ),
            heat_pump_source_types=snakemake.params.heat_pump_sources,
            efficiency_file=snakemake.input.heating_efficiencies,
            energy_totals_year=snakemake.params["energy_totals_year"],
            capacity_threshold=snakemake.params.existing_capacities[
                "threshold_capacity"
            ],
            use_electricity_distribution_grid=options["electricity_distribution_grid"],
        )

    uk_settings = snakemake.params.uk_settings
    if uk_settings["uk_new_gas_storage_data"]:
        logger.info("Adding UK gas storage data for storage facilities already existing in the base year")
        onshore = snakemake.input.onshore_regions
        offshore = snakemake.input.offshore_regions
        add_UK_gas_storage_data(n, onshore, offshore, baseyear)

    if options.get("cluster_heat_buses", False):
        cluster_heat_buses(n)

    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))

    sanitize_custom_columns(n)
    sanitize_carriers(n, snakemake.config)
    n.export_to_netcdf(snakemake.output.network)
