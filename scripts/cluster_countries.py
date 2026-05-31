from scripts._helpers import configure_logging, set_scenario_config

import pypsa
import geopandas as gpd
import pandas as pd
import numpy as np
import cartopy.crs as ccrs
import functools
import operator
import re

import logging
import warnings
warnings.filterwarnings(action="ignore", category=UserWarning)
idx = pd.IndexSlice
logger = logging.getLogger(__name__)
from pyproj import Transformer
transformer = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)

def most_common(s: pd.Series):
    m = s.mode(dropna=True)
    if len(m):
        return m.iloc[0]
    # fallback to first non-null
    return s.dropna().iloc[0] if s.notna().any() else np.nan

def rename_buses(links_region, key, value, type="link"):
    # change bus0 name
    links_region_bus0 = links_region.loc[links_region.bus0.str.split(" ", expand=True)[0].str.startswith(value)]

    if len(links_region_bus0) > 0:
        bus0_split = links_region_bus0.bus0.str.split(" ", expand=True)
        bus0_split[0] = key
        bus0_split = bus0_split.fillna("")
        bus0_split = bus0_split.astype(str).agg(' '.join, axis=1).str.rstrip()
        if len(bus0_split) > 0:
            df = pd.DataFrame()
            df["bus0_old"] = links_region_bus0.bus0
            df["bus0_new"] = bus0_split
            df.set_index("bus0_old", inplace=True)
            bus0_dict = df["bus0_new"].to_dict()
            links_region.replace(bus0_dict, inplace=True)

    # change bus1 name
    links_region_bus1 = links_region.loc[links_region.bus1.str.split(" ", expand=True)[0].str.startswith(value)]
    if len(links_region_bus1) > 0:
        bus1_split = links_region_bus1.bus1.str.split(" ", expand=True)
        bus1_split[0] = key
        bus1_split = bus1_split.fillna("")
        bus1_split = bus1_split.astype(str).agg(' '.join, axis=1).str.rstrip()
        if len(bus1_split) > 0:
            df = pd.DataFrame()
            df["bus1_old"] = links_region_bus1.bus1
            df["bus1_new"] = bus1_split
            df.set_index("bus1_old", inplace=True)
            bus1_dict = df["bus1_new"].to_dict()
            links_region.replace(bus1_dict, inplace=True)

    if type == "link":
        # change bus2 name
        links_region_bus2 = links_region.loc[links_region.bus2.str.split(" ", expand=True)[0].str.startswith(value)]
        bus2_split = links_region_bus2.bus2.str.split(" ", expand=True)
        bus2_split[0] = key
        bus2_split = bus2_split.fillna("")
        if len(bus2_split) > 0:
            bus2_split = bus2_split.astype(str).agg(' '.join, axis=1).str.rstrip()
            df = pd.DataFrame()
            df["bus2_old"] = links_region_bus2.bus2
            df["bus2_new"] = bus2_split
            df.set_index("bus2_old", inplace=True)
            bus2_dict = df["bus2_new"].to_dict()
            links_region.replace(bus2_dict, inplace=True)

        # change bus3 name
        links_region_bus3 = links_region.loc[links_region.bus3.str.split(" ", expand=True)[0].str.startswith(value)]
        bus3_split = links_region_bus3.bus3.str.split(" ", expand=True)
        bus3_split[0] = key
        bus3_split = bus3_split.fillna("")
        if len(bus3_split) > 0:
            bus3_split = bus3_split.astype(str).agg(' '.join, axis=1).str.rstrip()
            df = pd.DataFrame()
            df["bus3_old"] = links_region_bus3.bus3
            df["bus3_new"] = bus3_split
            df.set_index("bus3_old", inplace=True)
            bus3_dict = df["bus3_new"].to_dict()
            links_region.replace(bus3_dict, inplace=True)

    return links_region

# 1. Map buses according to dct1
def map_buses(n, n_mapped, dct1, centroids): 

    buses = n.buses.copy()

    buses_dict = {}

    for key, value in dct1.items():

        print(value)
        
        if type(value) == str:
            buses_region = buses.loc[buses.index.str.startswith(value)]
        else:
            mask = functools.reduce(
                operator.or_,
                [n.buses.index.str.startswith(v) for v in value]
            )
            buses_region = n.buses.loc[mask]

        index_old = buses_region.index

        buses_region["name"] = buses_region.index
        buses_region_name = buses_region["name"].str.split(" ", expand=True)
        buses_region_name[0] = key
        buses_region_name = buses_region_name.fillna("")
        buses_region_name = buses_region_name.astype(str).agg(' '.join, axis=1).str.rstrip()
        buses_region["name"] = buses_region_name
        buses_region.set_index("name", drop=True, inplace=True)

        buses_region["x"] = centroids[key][0]
        buses_region["y"] = centroids[key][1]
        buses_region["location"] = key + " 0" 
        buses_region["control"] = "PQ"
        buses_region["country"] = key
        buses_region["generator"] = ""

        buses_dict = {**buses_dict, **pd.Series(buses_region.index, index_old).to_dict()}

        buses_region_grouped = buses_region.drop_duplicates()

        # add new buses
        buses = pd.concat([buses, buses_region_grouped])

        # remove old buses
        print("dropping ", index_old)
        buses.drop(index=index_old, inplace=True)

    n_mapped.buses = buses

    buses_t = n.buses_t.copy()

    for k, v in buses_t.items():

        dtypes = n.buses_t[k].dtypes

        if dtypes.shape[0] > 0:

            if dtypes.unique()[0] == 'object' or dtypes.unique()[0] == 'bool':
                buses_t_i = buses_t[k].rename(columns=buses_dict)
                buses_t_i_grouped = buses_t_i.groupby(buses_t_i.columns, axis=1).agg(most_common)

            elif k in ["v_mag_pu_set", "v_mag_pu", "v_ang", "marginal_price"]:
                weight_renamed = n.buses_t["p"].rename(columns=buses_dict)
                weight_grouped = weight_renamed.groupby(weight_renamed.columns, axis=1, ).sum()
                weighted_sum = n.buses_t["p"].multiply(buses_t[k]).rename(columns=buses_dict)
                weighted_sum_grouped = weighted_sum.groupby(weighted_sum.columns, axis=1).sum()
                weighted_average_grouped = weighted_sum_grouped.div(weight_grouped, axis=0)
                buses_t_i_grouped = weighted_average_grouped.fillna(0).copy()

            elif k in ["p", "q"]:
                buses_t_i = buses_t[k].rename(columns=buses_dict)
                buses_t_i_grouped = buses_t_i.groupby(buses_t_i.columns, axis=1).sum()

            else:
                buses_t_i = buses_t[k].rename(columns=buses_dict)
                buses_t_i_grouped = buses_t_i.groupby(buses_t_i.columns, axis=1).agg(most_common)

        else:
            continue

        buses_t[k] = buses_t_i_grouped.copy()

    n_mapped.buses_t = buses_t

    return n_mapped 

# 2. Map links according to dct1
def map_links(n, n_mapped, dct1, dct1_rev):
    links = n.links.copy()

    links_old = links.copy()

    links_dict = {}

    for key, value in dct1.items():
        
        if type(value) == str:
            links_region = links.loc[links.bus0.str.startswith(value) | links.bus1.str.startswith(value) | links.bus2.str.startswith(value) | links.bus3.str.startswith(value) | links.bus4.str.startswith(value)]
        else:
            bus_cols = [col for col in links.columns if col.startswith("bus")]
            pattern = "|".join(f"^{re.escape(v)}" for v in value)

            mask = functools.reduce(
                operator.or_,
                [links[col].str.contains(pattern, regex=True, na=False) for col in bus_cols]
            )

            links_region = links.loc[mask]

        index_old = links_region.index

        # change index name
        links_region["name"] = links_region.index
        links_region_name = links_region["name"].str.split(" ", expand=True)

        links_region_name_renamed = links_region_name[0].str[0:4].replace(dct1_rev)
        
        links_region_name_renamed = links_region_name_renamed.loc[links_region_name_renamed.str.startswith("GB")]
        links_region_name.loc[links_region_name_renamed.index, 0] = links_region_name_renamed
        links_region_name = links_region_name.fillna("")
        links_region_name = links_region_name.astype(str).agg(' '.join, axis=1).str.rstrip()
        links_region["name"] = links_region_name
        links_region.set_index("name", drop=True, inplace=True)

        # change bus name
        if type(value) == list:
            for v in value:
                links_region = rename_buses(links_region, key, v)

        else:
            links_region = rename_buses(links_region, key, value) 

        links_dict = {**links_dict, **pd.Series(links_region.index, index_old).to_dict()}

        # group duplicate indices
        aggregate = links_region.columns
        agg = {}
        for c in aggregate: 
            if links_region.dtypes.loc[c] == 'object' or links_region.dtypes.loc[c] == 'bool':
                agg[c] = most_common
            elif c.startswith("p_") or c in ["length"]:
                agg[c] = "sum"
            elif "cost" in c or c in ["lifetime", "efficiency", "build_year"]:
                agg[c] = "mean"
            else:
                agg[c] = most_common

        links_region = links_region.groupby(links_region.index).agg(agg)

        # remove old links
        links.drop(index=index_old, inplace=True)

        # add new links
        links = pd.concat([links, links_region])

    n_mapped.links = links
    links_new = links.index

    links_t = n.links_t.copy()

    for k, v in links_t.items():

        if links_t[k].empty:
            print(f"links_t[{k}] is empty, skipping...")
            continue
        else:
            print(f"Processing links_t[{k}]...")

        dtypes = n.links_t[k].dtypes

        if dtypes.shape[0] > 0:

            if dtypes.unique()[0] == 'object' or dtypes.unique()[0] == 'bool':
                links_t_i = links_t[k].rename(columns=links_dict)
                links_t_i_grouped = links_t_i.groupby(links_t_i.columns, axis=1).agg(most_common)

            elif "efficiency" in k or "cost" in k or "mu" in k or k in ["start_up", "shut_down"]:
                links_t_i = links_t[k].rename(columns=links_dict)

                if n.links_t["p1"].empty:
                    print("Warning: n.links_t['p1'] is empty. Cannot perform weighted average for links_t[{}].".format(k))
                    continue

                load = n.links_t["p1"]
                variable = links_t[k]
                
                weight_renamed = load[variable.columns].rename(columns=links_dict)
                weight_grouped = weight_renamed.groupby(weight_renamed.columns, axis=1).sum()

                weighted_sum =(load[variable.columns]*variable).rename(columns=links_dict)

                weighted_sum_grouped = weighted_sum.groupby(weighted_sum.columns, axis=1).sum()
                weighted_average_grouped = weighted_sum_grouped.div(weight_grouped, axis=0)
                links_t_i_grouped = weighted_average_grouped.copy()

            elif "p" in k:
                links_t_i = links_t[k].rename(columns=links_dict)
                links_t_i_grouped = links_t_i.groupby(links_t_i.columns, axis=1).sum()
            
            else:
                links_t_i = links_t[k].rename(columns=links_dict)
                links_t_i_grouped = links_t_i.groupby(links_t_i.columns, axis=1).agg(most_common)

        else:
            continue

        links_t[k] = links_t_i_grouped.copy()

    n_mapped.links_t = links_t

    return n_mapped

# 3. Map lines according to dct1
def map_lines(n, n_mapped, dct1):
    lines = n.lines.copy()

    lines_dict = {}

    for key, value in dct1.items():
        
        if type(value) == str:
            lines_region = lines.loc[lines.bus0.str.startswith(value) | lines.bus1.str.startswith(value) ]
        else:
            bus_cols = [col for col in lines.columns if col.startswith("bus")]

            mask = functools.reduce(
                operator.or_,
                [lines[col].str.startswith(v) for col in bus_cols for v in value]
            )

            lines_region = lines.loc[mask]

        index_old = lines_region.index

        # change bus name
        if type(value) == list:
            for v in value:
                lines_region = rename_buses(lines_region, key, v, type="line")
        else:
            lines_region = rename_buses(lines_region, key, value, type = "line")

        lines_dict = {**lines_dict, **pd.Series(lines_region.index, index_old).to_dict()}

        # group duplicate indices
        aggregate = lines_region.columns
        agg = {}
        for c in aggregate: 
            if lines_region.dtypes.loc[c] == 'object' or lines_region.dtypes.loc[c] == 'bool':
                agg[c] = most_common
            elif c.startswith("s_") or c in ["length"]:
                agg[c] = "sum"
            elif "cost" in c or c in ["lifetime", "efficiency", "build_year"]:
                agg[c] = "mean"
            else:
                agg[c] = most_common

        lines_region = lines_region.groupby(lines_region.index).agg(agg)

        # remove old lines
        lines.drop(index=index_old, inplace=True)

        # add new lines
        lines = pd.concat([lines, lines_region])

    n_mapped.lines = lines
    lines_new = lines.index

    lines_t = n.lines_t.copy()

    for k, v in lines_t.items():

        if lines_t[k].empty:
            continue

        dtypes = n.lines_t[k].dtypes

        if dtypes.shape[0] > 0:

            if dtypes.unique()[0] == 'object' or dtypes.unique()[0] == 'bool':
                lines_t_i = lines_t[k].rename(columns=lines_dict)
                lines_t_i_grouped = lines_t_i.groupby(lines_t_i.columns, axis=1).agg(most_common)

            elif "loss" in k or "cost" in k or "mu" in k or k in ["s_max_pu"]:
                lines_t_i = lines_t[k].rename(columns=lines_dict)

                load = n.lines_t["p1"]
                variable = lines_t[k]

                weight_renamed = load[variable.columns].rename(columns=lines_dict)
                weight_grouped = weight_renamed.groupby(weight_renamed.columns, axis=1).sum()

                weighted_sum =(load[variable.columns]*variable).rename(columns=lines_dict)

                weighted_sum_grouped = weighted_sum.groupby(weighted_sum.columns, axis=1).sum()
                weighted_average_grouped = weighted_sum_grouped.div(weight_grouped, axis=0)
                lines_t_i_grouped = weighted_average_grouped.copy()

            elif "p" in k or "q" in k:
                lines_t_i = lines_t[k].rename(columns=lines_dict)
                lines_t_i_grouped = lines_t_i.groupby(lines_t_i.columns, axis=1).sum()

            else:
                lines_t_i = lines_t[k].rename(columns=lines_dict)
                lines_t_i_grouped = lines_t_i.groupby(lines_t_i.columns, axis=1).agg(most_common)
        else:
            continue

        lines_t[k] = lines_t_i_grouped.copy()

    n_mapped.lines_t = lines_t

    return n_mapped

# 4. Map generators according to dct1
def map_generators(n, n_mapped, dct1):
    generators = n.generators.copy()

    generators_dict = {}

    for key, value in dct1.items():
        
        if type(value) == str:
            generators_region = generators.loc[generators.index.str.startswith(value)]
        else:
            mask = functools.reduce(
            operator.or_,
            [generators.index.str.startswith(v) for v in value]
            )

            generators_region = generators.loc[mask]

        index_old = generators_region.index

        generators_region["name"] = generators_region.index
        generators_region_name = generators_region["name"].str.split(" ", expand=True)
        generators_region_name[0] = key
        generators_region_name = generators_region_name.fillna("")
        generators_region_name = generators_region_name.astype(str).agg(' '.join, axis=1).str.rstrip()
        generators_region["name"] = generators_region_name
        generators_region.set_index("name", drop=True, inplace=True)
        
        generatores_region_bus = generators_region["bus"].str.split(" ", expand=True)
        generatores_region_bus[0] = key
        generatores_region_bus = generatores_region_bus.fillna("")
        generatores_region_bus = generatores_region_bus.astype(str).agg(' '.join, axis=1).str.rstrip()
        generators_region["bus"] = generatores_region_bus

        generators_dict = {**generators_dict, **pd.Series(generators_region.index, index_old).to_dict()}
        
        # group duplicate indices
        aggregate = generators_region.columns
        agg = {}
        for c in aggregate: 
            if generators_region.dtypes.loc[c] == 'object' or generators_region.dtypes.loc[c] == 'bool':
                agg[c] = most_common
            elif "_pu" in c:
                agg[c] = "mean"
            elif c.startswith("p_") or c.startswith("e_"):
                agg[c] = "sum"
            elif "cost" in c or c in ["lifetime", "efficiency"]:
                agg[c] = "mean"
            else:
                agg[c] = most_common

        generators_region = generators_region.groupby(generators_region.index).agg(agg)

        # generators_region_grouped = generators_region.drop_duplicates()

        # remove old lines
        generators.drop(index=index_old, inplace=True)

        # add new lines
        generators = pd.concat([generators, generators_region])

    n_mapped.generators = generators

    generators_t = n.generators_t.copy()

    for k, v in generators_t.items():

        if generators_t[k].empty:
            continue

        dtypes = n.generators_t[k].dtypes

        if dtypes.shape[0] > 0:

            if dtypes.unique()[0] == 'object' or dtypes.unique()[0] == 'bool':
                generators_t_i = generators_t[k].rename(columns=generators_dict)
                generators_t_i_grouped = generators_t_i.groupby(generators_t_i.columns, axis=1).agg(most_common)
            # Average for normalized variables
            elif "cost" in k or "mu" in k or k in ["start_up", "shut_down", "efficiency","p_max_pu", "p_min_pu",]:
                generators_t_i = generators_t[k].rename(columns=generators_dict)
                generators_t_i_grouped = generators_t_i.groupby(generators_t_i.columns, axis=1).mean()
            # Sum for absolute variables
            elif k in ["p", "q", "p_set", "q_set", 'ramp_limit_up', 'ramp_limit_down']:
                generators_t_i = generators_t[k].rename(columns=generators_dict)
                generators_t_i_grouped = generators_t_i.groupby(generators_t_i.columns, axis=1).sum()
            # Most common for string variables
            else:
                generators_t_i = generators_t[k].rename(columns=generators_dict)
                generators_t_i_grouped = generators_t_i.groupby(generators_t_i.columns, axis=1).agg(most_common)

        else:
            continue

        generators_t[k] = generators_t_i_grouped.copy()

    n_mapped.generators_t = generators_t

    return n_mapped

# 5. Map storage_units according to dct1
def map_storage_units(n, n_mapped, dct1):
    storage_units = n.storage_units.copy()

    storage_units_dict = {}

    for key, value in dct1.items():
        
        if type(value) == str:
            storage_units_region = storage_units.loc[storage_units.index.str.startswith(value)]
        else:
            mask = functools.reduce(
            operator.or_,
            [storage_units.index.str.startswith(v) for v in value]
            )

            storage_units_region = storage_units.loc[mask]

        if len(storage_units_region) == 0:
            continue

        index_old = storage_units_region.index

        storage_units_region["name"] = storage_units_region.index
        storage_units_region_name = storage_units_region["name"].str.split(" ", expand=True)
        storage_units_region_name[0] = key
        storage_units_region_name = storage_units_region_name.fillna("")
        storage_units_region_name = storage_units_region_name.astype(str).agg(' '.join, axis=1).str.rstrip()
        storage_units_region["name"] = storage_units_region_name
        storage_units_region.set_index("name", drop=True, inplace=True)

        storage_units_region_bus = storage_units_region["bus"].str.split(" ", expand=True)
        storage_units_region_bus[0] = key
        storage_units_region_bus = storage_units_region_bus.fillna("")
        storage_units_region_bus = storage_units_region_bus.astype(str).agg(' '.join, axis=1).str.rstrip()
        storage_units_region["bus"] = storage_units_region_bus

        storage_units_dict = {**storage_units_dict, **pd.Series(storage_units_region.index, index_old).to_dict()}

        # group duplicate indices
        aggregate = storage_units_region.columns
        agg = {}
        for c in aggregate: 
            if storage_units_region.dtypes.loc[c] == 'object' or storage_units_region.dtypes.loc[c] == 'bool':
                agg[c] = most_common
            elif c.startswith("p_") or c.startswith("e_"):
                agg[c] = "sum"
            elif "cost" in c or c in ["lifetime", "efficiency", "build_year"]:
                agg[c] = "mean"
            else:
                agg[c] = most_common

        storage_units_region = storage_units_region.groupby(storage_units_region.index).agg(agg)

        # storage_units_region_grouped = storage_units_region.drop_duplicates()

        # remove old lines
        storage_units.drop(index=index_old, inplace=True)

        # add new lines
        storage_units = pd.concat([storage_units, storage_units_region])

    n_mapped.storage_units = storage_units

    storage_units_t = n.storage_units_t.copy()

    for k, v in storage_units_t.items():

        if storage_units_t[k].empty:
            continue

        dtypes = n.storage_units_t[k].dtypes

        if dtypes.shape[0] > 0:

            if dtypes.unique()[0] == 'object' or dtypes.unique()[0] == 'bool':
                storage_units_t_i = storage_units_t[k].rename(columns=storage_units_dict)
                storage_units_t_i_grouped = storage_units_t_i.groupby(storage_units_t_i.columns, axis=1).agg(most_common)

            # Weighted average for normalized variables
            elif "cost" in k or "mu" in k or "efficiency" in k or k in ["start_up", "shut_down", "p_max_pu", "p_min_pu"]:
                weight_renamed = n.storage_units_t["p"].rename(columns=storage_units_dict)
                weight_grouped = weight_renamed.groupby(weight_renamed.columns, axis=1).sum()
                weighted_sum = n.storage_units_t["p"].multiply(storage_units_t[k]).rename(columns=storage_units_dict)
                weighted_sum_grouped = weighted_sum.groupby(weighted_sum.columns, axis=1).sum()
                weighted_average_grouped = weighted_sum_grouped.div(weight_grouped, axis=0)
                storage_units_t_i_grouped = weighted_average_grouped.copy()
            # Sum for absolute variables
            elif "p_" in k or k in ["p", "q", "q_set", 'ramp_limit_up', 'ramp_limit_down', "standing_loss", "inflow", "state_of_charge", "spill"]:
                storage_units_t_i = storage_units_t[k].rename(columns=storage_units_dict)
                storage_units_t_i_grouped = storage_units_t_i.groupby(storage_units_t_i.columns, axis=1).sum()
            # Most common for string variables
            else:
                storage_units_t_i = storage_units_t[k].rename(columns=storage_units_dict)
                storage_units_t_i_grouped = storage_units_t_i.groupby(storage_units_t_i.columns, axis=1).agg(most_common)
        else:
            continue

        storage_units_t[k] = storage_units_t_i_grouped.copy()

    n_mapped.storage_units_t = storage_units_t

    return n_mapped 

# 6. Map stores according to dct1
def map_stores(n, n_mapped, dct1):
    stores = n.stores.copy()

    stores_dict = {}

    for key, value in dct1.items():
        
        if type(value) == str:
            stores_region = stores.loc[stores.index.str.startswith(value)]
        else:
            mask = functools.reduce(
            operator.or_,
            [stores.index.str.startswith(v) for v in value]
            )

            stores_region = stores.loc[mask]

        if len(stores_region) == 0:
            continue

        index_old = stores_region.index

        stores_region["name"] = stores_region.index
        stores_region_name = stores_region["name"].str.split(" ", expand=True)
        stores_region_name[0] = key
        stores_region_name = stores_region_name.fillna("")
        stores_region_name = stores_region_name.astype(str).agg(' '.join, axis=1).str.rstrip()
        stores_region["name"] = stores_region_name
        stores_region.set_index("name", drop=True, inplace=True)

        stores_region_bus = stores_region["bus"].str.split(" ", expand=True)
        stores_region_bus[0] = key
        stores_region_bus = stores_region_bus.fillna("")
        stores_region_bus = stores_region_bus.astype(str).agg(' '.join, axis=1).str.rstrip()
        stores_region["bus"] = stores_region_bus

        stores_dict = {**stores_dict, **pd.Series(stores_region.index, index_old).to_dict()}

        # group duplicate indices
        aggregate = stores_region.columns
        agg = {}
        for c in aggregate: 
            if stores_region.dtypes.loc[c] == 'object' or stores_region.dtypes.loc[c] == 'bool':
                agg[c] = most_common
            elif c.startswith("p_") or c.startswith("e_"):
                agg[c] = "sum"
            elif "cost" in c or c in ["lifetime", "efficiency", "build_year"]:
                agg[c] = "mean"
            else:
                agg[c] = most_common

        stores_region = stores_region.groupby(stores_region.index).agg(agg)

        # storage_units_region_grouped = storage_units_region.drop_duplicates()

        # remove old lines
        stores.drop(index=index_old, inplace=True)

        # add new lines
        stores = pd.concat([stores, stores_region])

    n_mapped.stores = stores

    stores_t = n.stores_t.copy()

    for k, v in stores_t.items():

        if stores_t[k].empty:
            continue

        dtypes = n.stores_t[k].dtypes

        if dtypes.shape[0] > 0:

            # Weighted average for normalized variables
            if dtypes.unique()[0] == 'object' or dtypes.unique()[0] == 'bool':
                stores_t_i = stores_t[k].rename(columns=stores_dict)
                stores_t_i_grouped = stores_t_i.groupby(stores_t_i.columns, axis=1).agg(most_common)
            elif "cost" in k or "mu" in k or "efficiency" in k or k in ["start_up", "shut_down", "e_max_pu", "e_min_pu"]:
                weight_renamed = n.stores_t["e"].rename(columns=stores_dict)
                weight_grouped = weight_renamed.groupby(weight_renamed.columns, axis=1).sum()
                weighted_sum = n.stores_t["e"].multiply(stores_t[k]).rename(columns=stores_dict)
                weighted_sum_grouped = weighted_sum.groupby(weighted_sum.columns, axis=1).sum()
                weighted_average_grouped = weighted_sum_grouped.div(weight_grouped, axis=0)
                stores_t_i_grouped = weighted_average_grouped.copy()
                
            # Sum for absolute variables
            elif "e_" in k or "p_" in k or "q_" in k or k in ["e","p","q",'ramp_limit_up', 'ramp_limit_down', "standing_loss", "inflow", "state_of_charge", "spill"]:
                stores_t_i = stores_t[k].rename(columns=stores_dict)
                stores_t_i_grouped = stores_t_i.groupby(stores_t_i.columns, axis=1).sum()
            # Most common for string variables
            else:
                stores_t_i = stores_t[k].rename(columns=stores_dict)
                stores_t_i_grouped = stores_t_i.groupby(stores_t_i.columns, axis=1).agg(most_common)

        else:
            continue

        stores_t[k] = stores_t_i_grouped.copy()

    n_mapped.stores_t = stores_t

    return n_mapped 

# 7. Map loads according to dct1
def map_loads(n, n_mapped, dct1):
    loads = n.loads.copy()

    loads_dict = {}

    for key, value in dct1.items():
        
        if type(value) == str:
            loads_region = loads.loc[loads.index.str.startswith(value)]
        else:
            mask = functools.reduce(
            operator.or_,
            [loads.index.str.startswith(v) for v in value]
            )

            loads_region = loads.loc[mask]

        if len(loads_region) == 0:
            continue

        index_old = loads_region.index

        loads_region["name"] = loads_region.index
        loads_region_name = loads_region["name"].str.split(" ", expand=True)
        loads_region_name[0] = key
        loads_region_name = loads_region_name.fillna("")
        loads_region_name = loads_region_name.astype(str).agg(' '.join, axis=1).str.rstrip()
        loads_region["name"] = loads_region_name
        loads_region.set_index("name", drop=True, inplace=True)

        loads_region_bus = loads_region["bus"].str.split(" ", expand=True)
        loads_region_bus[0] = key
        loads_region_bus = loads_region_bus.fillna("")
        loads_region_bus = loads_region_bus.astype(str).agg(' '.join, axis=1).str.rstrip()
        loads_region["bus"] = loads_region_bus

        loads_dict = {**loads_dict, **pd.Series(loads_region.index, index_old).to_dict()}

        # group duplicate indices
        aggregate = loads_region.columns
        agg = {}
        for c in aggregate: 
            if loads_region.dtypes.loc[c] == 'object' or loads_region.dtypes.loc[c] == 'bool':
                agg[c] = most_common
            elif c.startswith("p_") or c.startswith("e_"):
                agg[c] = "sum"
            elif "cost" in c or c in ["lifetime", "efficiency", "build_year"]:
                agg[c] = "mean"
            else:
                agg[c] = most_common

        loads_region = loads_region.groupby(loads_region.index).agg(agg)

        # storage_units_region_grouped = storage_units_region.drop_duplicates()

        # remove old lines
        loads.drop(index=index_old, inplace=True)

        # add new lines
        loads = pd.concat([loads, loads_region])

    n_mapped.loads = loads

    loads_t = n.loads_t.copy()

    for k, v in loads_t.items():

        if loads_t[k].empty:
            continue

        dtypes = n.loads_t[k].dtypes

        if dtypes.shape[0] > 0:

            if dtypes.unique()[0] == 'object' or dtypes.unique()[0] == 'bool':
                loads_t_i = loads_t[k].rename(columns=loads_dict)
                loads_t_i_grouped = loads_t_i.groupby(loads_t_i.columns, axis=1).agg(most_common)

            elif k in ["v_mag_pu_set", "v_mag_pu", "v_ang", "marginal_price"]:
                weight_renamed = n.loads_t["p"].rename(columns=loads_dict)
                weight_grouped = weight_renamed.groupby(weight_renamed.columns, axis=1).sum()
                weighted_sum = n.loads_t["p"].multiply(loads_t[k]).rename(columns=loads_dict)
                weighted_sum_grouped = weighted_sum.groupby(weighted_sum.columns, axis=1).sum()
                weighted_average_grouped = weighted_sum_grouped.div(weight_grouped, axis=0)
                loads_t_i_grouped = weighted_average_grouped.copy()

            elif k in ["p", "q", "p_set", "q_set"]:
                loads_t_i = loads_t[k].rename(columns=loads_dict)
                loads_t_i_grouped = loads_t_i.groupby(loads_t_i.columns, axis=1).sum()

            else:
                loads_t_i = loads_t[k].rename(columns=loads_dict)
                loads_t_i_grouped = loads_t_i.groupby(loads_t_i.columns, axis=1).agg(most_common)

        else:
            continue

        loads_t[k] = loads_t_i_grouped.copy()

    n_mapped.loads_t = loads_t

    return n_mapped

def consistency_check(n_mapped, n):
    errors = 0

    # buses
    print("buses duplicated") if n_mapped.buses.index.duplicated().any() else None
    print("buses shape mismatch") if (n_mapped.buses.shape[1] - n.buses.shape[1]) != 0 else None
    errors += 1 if n_mapped.buses.index.duplicated().any() else errors
    for c in n_mapped.buses_t.keys():
        if n_mapped.buses_t[c].columns.duplicated().any():
            print("buses ", c)
            errors += 1

    # loads
    print("loads duplicated") if n_mapped.loads.index.duplicated().any() else None
    print("loads shape mismatch") if (n_mapped.loads.shape[1] - n.loads.shape[1]) != 0 else None
    errors += 1 if n_mapped.loads.index.duplicated().any() else errors
    for c in n_mapped.loads_t.keys():
        if n_mapped.loads_t[c].columns.duplicated().any():
            print("loads ", c)
            errors += 1

    # generators
    print("generators duplicated") if n_mapped.generators.index.duplicated().any() else None
    print("generators shape mismatch") if (n_mapped.generators.shape[1] - n.generators.shape[1]) != 0 else None
    errors += 1 if n_mapped.generators.index.duplicated().any() else errors
    for c in n_mapped.generators_t.keys():
        if n_mapped.generators_t[c].columns.duplicated().any():
            print("generators ", c)
            errors += 1

    # links
    print("links duplicated") if n_mapped.links.index.duplicated().any() else None
    print("links shape mismatch") if (n_mapped.links.shape[1] - n.links.shape[1]) != 0 else None
    errors += 1 if n_mapped.links.index.duplicated().any() else errors
    for c in n_mapped.links_t.keys():
        if n_mapped.links_t[c].columns.duplicated().any():
            print("links ", c)
            errors += 1

    # lines
    print("lines duplicated") if n_mapped.lines.index.duplicated().any() else None
    print("lines shape mismatch") if (n_mapped.lines.shape[1] - n.lines.shape[1]) != 0 else None
    errors += 1 if n_mapped.lines.index.duplicated().any() else errors
    for c in n_mapped.lines_t.keys():
        if n_mapped.lines_t[c].columns.duplicated().any():
            print("lines ", c)
            errors += 1

    # storage units
    print("storage_units duplicated") if n_mapped.storage_units.index.duplicated().any() else None
    print("storage_units shape mismatch") if (n_mapped.storage_units.shape[1] - n.storage_units.shape[1]) != 0 else None
    errors += 1 if n_mapped.storage_units.index.duplicated().any() else errors
    for c in n_mapped.storage_units_t.keys():
        if n_mapped.storage_units_t[c].columns.duplicated().any():
            print("storage units ", c)
            errors += 1
    # stores
    print("stores duplicated") if n_mapped.stores.index.duplicated().any() else None
    print("stores shape mismatch") if (n_mapped.stores.shape[1] - n.stores.shape[1]) != 0 else None
    errors += 1 if n_mapped.stores.index.duplicated().any() else errors
    for c in n_mapped.stores_t.keys():
        if n_mapped.stores_t[c].columns.duplicated().any():
            print("stores ", c)
            errors += 1

    logger.info("A total of {errors} errors were found during the consistency check.")

def calculate_centroids(dct):
    # Make new regions file and calculate centroids of new regions
    centroids = {}
    regions = gpd.GeoDataFrame()
    for key, value in dct.items():

        region = value

        # in center of the region, plot number i
        x = region.dissolve().geometry.centroid.x
        y = region.dissolve().geometry.centroid.y
        centroids[key] = transformer.transform(x.values[0], y.values[0])

        region_df = region.dissolve()
        region_df["name"] = key
        region_df.set_index("name", inplace=True)

        regions = pd.concat([regions, region_df])

    return centroids, regions

def map_countries_to_regions(n, dct1, dct1_rev, centroids):

    n_mapped = n.copy()

    n_mapped = map_buses(n, n_mapped, dct1, centroids) # 1

    n_mapped = map_links(n, n_mapped, dct1, dct1_rev) # 2

    n_mapped = map_lines(n, n_mapped, dct1) # 3

    n_mapped = map_generators(n, n_mapped, dct1) # 4

    n_mapped = map_storage_units(n, n_mapped, dct1) # 5

    n_mapped = map_stores(n, n_mapped, dct1) # 6

    n_mapped = map_loads(n, n_mapped, dct1) # 7

    consistency_check(n_mapped, n)

    return n_mapped

if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("cluster_network", clusters=60)
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    dct1 = {"Balkan": ["HR", "RS", "BG", 'AL', 'BA', 'GR', "MK", "ME"],
            "Baltic": ["EE", "LV", "LT"],
            "East": ["RO", "HU", "SK", "CZ", "PL"]
            }

    dct1_rev = {country: region for region, countries in dct1.items() for country in countries}

    regions = gpd.read_file(f"C:\\Users\\egtske\\Documents\\pypsa-uk/n110/regions_onshore_base_s_110.geojson")
    regions["country"] = regions["name"].str[0:2]
    data_crs = ccrs.epsg(3035)
    regions = regions.to_crs(data_crs)

    dct = {region: regions.loc[regions.country.isin(countries)] for region, countries in dct1.items()}

    centroids, clustered_regions = calculate_centroids(dct)

    n = pypsa.Network(snakemake.input.network)
    n_mapped = map_countries_to_regions(n, dct1, dct1_rev, centroids)

    n_mapped.export_to_netcdf(snakemake.output.network)
