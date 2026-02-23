# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Cluster gas transmission network to clustered model regions.
"""

import logging

import geopandas as gpd
import pandas as pd
import numpy as np
from pypsa.geo import haversine_pts
from shapely import wkt

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)


def concat_gdf(gdf_list, crs="EPSG:4326"):
    """
    Concatenate multiple geopandas dataframes with common coordinate reference
    system (crs).
    """
    return gpd.GeoDataFrame(pd.concat(gdf_list), crs=crs)


def load_bus_regions(onshore_path, offshore_path):
    """
    Load pypsa-eur on- and offshore regions and concat.
    """
    bus_regions_offshore = gpd.read_file(offshore_path)
    bus_regions_onshore = gpd.read_file(onshore_path)
    bus_regions = concat_gdf([bus_regions_offshore, bus_regions_onshore])
    bus_regions = bus_regions.dissolve(by="name", aggfunc="sum")

    return bus_regions

def extract_non_connected_pipes(df, countries, projection0 = "EPSG:4326", projection1 = "EPSG:3035"):
    """
    This function builds on top of the current methodology of clustering network. The current rule is that 
    if bus0 or bus1 is nan, or bus0 == bus1, the pipeline is dropped. This does, however, not account for 
    pipelines overseas which are represented as sequences in the dataset, causing them to be missing either a 
    bus0 or bus1 label. This function identifies such pipelines and ensures that they are not dropped.  
    """

    dropped_gas_lines_by_mistake = gpd.GeoDataFrame(columns = list(df.columns) + ["country"])
    for country in countries:

        # rows that will be dropped based on columns with buses
        df_dropped = df.loc[df.bus0.isna() | df.bus1.isna() | (df.bus1 == df.bus0)]

        c_index_0 = df_dropped.bus0.dropna().loc[df_dropped.bus0.str.contains(country).dropna()].index
        c_index_1 = df_dropped.bus1.dropna().loc[df_dropped.bus1.str.contains(country).dropna()].index

        # Gas network: all pipelines that appear to be connected to the specified country (based on buses)
        gas_network_part0 = df_dropped.loc[c_index_0].loc[df_dropped.loc[c_index_0].bus1.isna()]
        gas_network_part1 = df_dropped.loc[c_index_1].loc[df_dropped.loc[c_index_1].bus0.isna()]
        gas_network = pd.concat([gas_network_part0, 
                                gas_network_part1])
        gas_network["geometry"] = gpd.GeoSeries.from_wkt(gas_network["geometry"], crs=projection0)
        gas_network = gpd.GeoDataFrame(gas_network, geometry='geometry')
        gas_network = gas_network.to_crs(projection1)
        gas_network["klevel"] = 0

        # check if any of the pipelines are in fact connected other lines (based on the geometry)
        inters_reduced_df = gpd.GeoDataFrame(columns = gas_network.columns)
        k = 1 
        kmax = 5 # number of sequences to search for
        while k < kmax:
            print(k)
            # Gas network: all of the rest 
            index_to_drop = list(gas_network.index) + list(inters_reduced_df.index)
            gas_network_0 = df.drop(index = index_to_drop)
            gas_network_0["geometry"] = gpd.GeoSeries.from_wkt(gas_network_0["geometry"], crs=projection0)
            gas_network_0 = gpd.GeoDataFrame(gas_network_0, geometry='geometry')
            gas_network_0 = gas_network_0.to_crs(projection1)
            
            if k == 1: # first sequence
                print(gas_network)
                for geo_k0 in gas_network.geometry:
                    inters = gas_network_0.loc[gas_network_0.geometry.intersects(geo_k0)]
                    inters_reduced = inters.loc[inters.bus0 != inters.bus1]
                    # save the intersecting pipelines
                    inters_reduced["klevel"] = k
                    inters_reduced_df = pd.concat([inters_reduced_df, 
                                                    inters_reduced])

            else: # subsequent sequences
                # loop over inters_reduced (determined from the last iteration)
                for geo in inters_reduced.geometry:
                    inters = gas_network_0.loc[gas_network_0.geometry.intersects(geo)]
                    inters_reduced = inters.loc[inters.bus0 != inters.bus1]
                    inters_reduced["klevel"] = k
                    inters_reduced_df = pd.concat([inters_reduced_df, 
                                                    inters_reduced])
            k += 1

        # first, drop pipelines that are internal 
        inters_reduced_df = inters_reduced_df.query("bus0.isna() | bus1.isna()")

        inters_reduced_df = pd.concat([inters_reduced_df, 
                                       gas_network])
        
        if inters_reduced_df.empty:
            continue

        # remove if end of line is still nan after search
        lines_end = inters_reduced_df.query(f"klevel == {inters_reduced_df.klevel.max()}")
        lines_end_remove = lines_end.loc[lines_end.bus0.isna() & lines_end.bus1.isna() | lines_end.bus0.str.contains(country) | lines_end.bus1.str.contains(country)]

        inters_reduced_df.drop(lines_end_remove.index, inplace=True)

        if inters_reduced_df.empty:
            continue
        
        while not lines_end_remove.empty:
            lines_end = inters_reduced_df.query(f"klevel == {inters_reduced_df.klevel.max()}")
            lines_end_remove = lines_end.loc[lines_end.bus0.isna() & lines_end.bus1.isna()]
            inters_reduced_df.drop(lines_end_remove.index, inplace=True)

        if len(inters_reduced_df.bus0.dropna().unique()) == 1 and len(inters_reduced_df.bus1.dropna().unique()) == 1:
            inters_reduced_df.drop(index = inters_reduced_df.index, inplace=True)
            continue

        # then, check if any of the identified pipelines are dead ends
        for k_ref in range(1, inters_reduced_df.klevel.max() + 1):
            reference = inters_reduced_df.query(f"klevel == {k_ref}")
            test = inters_reduced_df.query(f"klevel == {k_ref - 1}")
            intersects = pd.DataFrame(index = inters_reduced_df.query(f"klevel == {k_ref - 1}").index, 
                                    columns = reference.index)
            for i in range(len(reference)):
                intersects.iloc[:,i] = test.geometry.intersects(reference.iloc[i].geometry)

            klevel0_to_drop = (intersects == False).all(axis = 1)
            index_to_drop = klevel0_to_drop.loc[klevel0_to_drop].index
            if len(index_to_drop) > 0:
                inters_reduced_df.drop(index = index_to_drop, inplace=True)

        # repeat, but the other way around (i.e. check if any of the identified pipelines are dead ends from the other direction)
        for k_ref in range(inters_reduced_df.klevel.max(), 0, -1):
            reference = inters_reduced_df.query(f"klevel == {k_ref - 1}")
            test = inters_reduced_df.query(f"klevel == {k_ref}")
            intersects = pd.DataFrame(index = inters_reduced_df.query(f"klevel == {k_ref}").index, 
                                    columns = reference.index)
            for i in range(len(reference)):
                intersects.iloc[:,i] = test.geometry.intersects(reference.iloc[i].geometry)

            klevel0_to_drop = (intersects == False).all(axis = 1)
            index_to_drop = klevel0_to_drop.loc[klevel0_to_drop].index
            if len(index_to_drop) > 0:
                inters_reduced_df.drop(index = index_to_drop, inplace=True)

        # inters_reduced_df.drop(columns = ["klevel"], inplace=True)
        inters_reduced_df["country"] = country
        dropped_gas_lines_by_mistake = pd.concat([dropped_gas_lines_by_mistake, 
                                                inters_reduced_df])
        print("adding ", len(inters_reduced_df), " pipelines for ", country, " back to the gas network")
    
        return dropped_gas_lines_by_mistake
    
def attach_non_connected_pipes(df, dropped_gas_lines_by_mistake, countries, projection0 = "EPSG:4326", projection1 = "EPSG:3035"):
    from shapely.ops import linemerge
    new_lines = gpd.GeoDataFrame(columns = df.columns)
    for country in countries:
        dropped_gas_lines_by_mistake_c = dropped_gas_lines_by_mistake.query("country == @country")
        kmax_c = dropped_gas_lines_by_mistake_c.klevel.max()
        columns = ["bus0", "bus1", "bidirectional"]
        end_points = dropped_gas_lines_by_mistake_c.query(f"klevel == {kmax_c}")[columns]
        end_points_unique = end_points[columns].drop_duplicates()

        for row in end_points_unique.index:
            print(row)
            # drop all other end points than "row" 
            dropped_gas_lines_by_mistake_c_i = dropped_gas_lines_by_mistake_c.drop(end_points.drop(row).index).copy()

            intersects_index = [row]
            intersects_i = pd.DataFrame(index = [])
            for i in range(kmax_c, 0, -1):
                print("klevel =",i)
                df_ref = dropped_gas_lines_by_mistake_c_i.drop(index = [row] + list(intersects_i.index))

                # intersection between "row" and every other connecting pipelines
                comp = row if i == kmax_c else intersects_i.index
                ind = df_ref.geometry.dwithin(dropped_gas_lines_by_mistake_c_i.loc[comp].geometry,
                                            distance = 20000 # 20 km distance threshold for intersection
                                            )
                intersects_i = df_ref.loc[ind] 

                intersects_index += list(intersects_i.index)

                if i == 1:

                    new_line_i = dropped_gas_lines_by_mistake.loc[intersects_index].query("country == @country")
                    new_line_i["length"] = new_line_i["length_haversine"]

                    bus_condition_1 = new_line_i.query(f"klevel == {new_line_i.klevel.min()}").bus0.isna().item() # True
                    bus_condition_2 = new_line_i.query(f"klevel == {new_line_i.klevel.min()}").bus1.isna().item() # False
                    bus_condition_3 = new_line_i.query(f"klevel == {new_line_i.klevel.max()}").bus0.isna().item() # True
                    bus_condition_4 = new_line_i.query(f"klevel == {new_line_i.klevel.max()}").bus1.isna().item() # False
                    
                    
                    #                           min(k), bus0 != nan                         max(k), bus0 == nan    
                    pipe_leaving_country = (not bus_condition_1 and bus_condition_2) and (bus_condition_3 and not bus_condition_4)
                    #                                               min(k), bus1 == nan                           max(k), bus1 != nan

                    #                           min(k), bus0 == nan                         max(k), bus0 != nan    
                    pipe_entering_country = (bus_condition_1 and not bus_condition_2) and (not bus_condition_3 and bus_condition_4)
                    #                                               min(k), bus1 != nan                           max(k), bus1 == nan

                    if pipe_leaving_country:
                        print("pipe is leaving ", country)
                    elif pipe_entering_country:
                        print("pipe is entering ", country)
                    elif bus_condition_3 and not bus_condition_4:
                        new_line_i.loc[new_line_i.query(f"klevel == {new_line_i.klevel.min()}").index, "bus0"] = new_line_i.loc[new_line_i.query(f"klevel == {new_line_i.klevel.min()}").index, "bus1"]
                        new_line_i.loc[new_line_i.query(f"klevel == {new_line_i.klevel.min()}").index, "bus1"] = np.nan

                    new_line_i = new_line_i.to_crs(projection0)
                    new_line_i_dissolved = new_line_i.dissolve(aggfunc = {"name": "first",
                                        "diameter_mm": "mean",
                                        "H_gas": "mean",
                                        "bidirectional": "first",
                                        "length": "sum",
                                        "p_nom": "min",
                                        "max_pressure_bar": "mean",
                                        "build_year": "mean",
                                        "point0": "first",
                                        "point1": "first",
                                        "p_nom_diameter": "min",
                                        "length_haversine": "sum",
                                        "bus0": "first",
                                        "bus1": "first",
                                        "klevel": "first",
                                        "country": "first"
                                        })

                    new_line_i_dissolved["geometry"] = str(new_line_i_dissolved.geometry.apply(linemerge).item())
                    new_lines = pd.concat([new_lines, new_line_i_dissolved.drop(columns = ["klevel", "country"])], ignore_index = True)

            print("")

    df = pd.concat([df, new_lines], ignore_index = True)
    for i in [0, 1]:
        df[f"point{i}"] = df[f"bus{i}"].map(
                    bus_regions.to_crs(projection1).centroid.to_crs(projection0)
                )

    return df

def build_clustered_gas_network(df, bus_regions, length_factor=1.25, projection0 = "EPSG:4326", projection1 = "EPSG:3035"):
    for i in [0, 1]:
        gdf = gpd.GeoDataFrame(geometry=df[f"point{i}"], crs=projection0)

        bus_mapping = gpd.sjoin(gdf, bus_regions, how="left", predicate="within")[
            "name"
        ]
        bus_mapping = bus_mapping.groupby(bus_mapping.index).first()

        df[f"bus{i}"] = bus_mapping

        df[f"point{i}"] = df[f"bus{i}"].map(
            bus_regions.to_crs(projection1).centroid.to_crs(projection0)
        )

    ###########
    # add function representing the operations below
    # We are currently only using this function to include the connection between Ireland and GB North West, 
    # which would have been dropped otherwise.
    countries = ["IE"]
    dropped_gas_lines_by_mistake = extract_non_connected_pipes(df, countries, projection0 = projection0, projection1 = projection1)
    df = attach_non_connected_pipes(df, dropped_gas_lines_by_mistake, countries, projection0 = projection0)
    #########

    # drop pipes where not both buses are inside regions
    df = df.loc[~df.bus0.isna() & ~df.bus1.isna()]

    # drop pipes within the same region
    df = df.loc[df.bus1 != df.bus0]

    if df.empty:
        return df

    # recalculate lengths as center to center * length factor
    df["length"] = df.apply(
        lambda p: length_factor
        * haversine_pts([p.point0.x, p.point0.y], [p.point1.x, p.point1.y]),
        axis=1,
    )

    # tidy and create new numbered index
    df.drop(["point0", "point1"], axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def reindex_pipes(df, prefix="gas pipeline"):
    def make_index(x):
        connector = " <-> " if x.bidirectional else " -> "
        return prefix + " " + x.bus0 + connector + x.bus1

    df.index = df.apply(make_index, axis=1)

    df["p_min_pu"] = df.bidirectional.apply(lambda bi: -1 if bi else 0)
    df.drop("bidirectional", axis=1, inplace=True)

    df.sort_index(axis=1, inplace=True)


def aggregate_parallel_pipes(df):
    strategies = {
        "bus0": "first",
        "bus1": "first",
        "p_nom": "sum",
        "p_nom_diameter": "sum",
        "max_pressure_bar": "mean",
        "build_year": "mean",
        "diameter_mm": "mean",
        "length": "mean",
        "name": " ".join,
        "p_min_pu": "min",
    }
    return df.groupby(df.index).agg(strategies)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("cluster_gas_network", clusters="37")
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    fn = snakemake.input.cleaned_gas_network
    df = pd.read_csv(fn, index_col=0)
    for col in ["point0", "point1"]:
        df[col] = df[col].apply(wkt.loads)

    bus_regions = load_bus_regions(
        snakemake.input.regions_onshore, snakemake.input.regions_offshore
    )

    gas_network = build_clustered_gas_network(df, bus_regions, projection0 = "EPSG:4326", projection1 = "EPSG:3035")

    reindex_pipes(gas_network)
    gas_network = aggregate_parallel_pipes(gas_network)

    gas_network.to_csv(snakemake.output.clustered_gas_network)
