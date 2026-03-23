# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Prepares brownfield data from previous planning horizon.
"""

import logging

import numpy as np
import pandas as pd
import geopandas as gpd
import pypsa
import xarray as xr

from scripts._helpers import (
    configure_logging,
    get_snapshots,
    sanitize_custom_columns,
    set_scenario_config,
    update_config_from_wildcards,
)
from scripts.add_electricity import flatten, sanitize_carriers
from scripts.add_existing_baseyear import add_build_year_to_new_assets

logger = logging.getLogger(__name__)
idx = pd.IndexSlice


def add_brownfield(
    n,
    n_p,
    year,
    h2_retrofit=False,
    h2_retrofit_capacity_per_ch4=None,
    capacity_threshold=None,
):
    """
    Add brownfield capacity from previous network.

    Parameters
    ----------
    n : pypsa.Network
        Network to add brownfield to
    n_p : pypsa.Network
        Previous network to get brownfield from
    year : int
        Planning year
    h2_retrofit : bool
        Whether to allow hydrogen pipeline retrofitting
    h2_retrofit_capacity_per_ch4 : float
        Ratio of hydrogen to methane capacity for pipeline retrofitting
    capacity_threshold : float
        Threshold for removing assets with low capacity
    """
    logger.info(f"Preparing brownfield for the year {year}")

    # electric transmission grid set optimised capacities of previous as minimum
    n.lines.s_nom_min = n_p.lines.s_nom_opt
    dc_i = n.links[n.links.carrier == "DC"].index
    dc_i_intersect = dc_i.intersection(n_p.links.index)
    n.links.loc[dc_i_intersect, "p_nom_min"] = n_p.links.loc[dc_i_intersect, "p_nom_opt"]

    for c in n_p.iterate_components(["Link", "Generator", "Store"]):
        attr = "e" if c.name == "Store" else "p"

        # first, remove generators, links and stores that track
        # CO2 or global EU values since these are already in n
        n_p.remove(c.name, c.df.index[c.df.lifetime == np.inf])

        # remove assets whose build_year + lifetime <= year
        n_p.remove(c.name, c.df.index[c.df.build_year + c.df.lifetime <= year])

        # remove assets if their optimized nominal capacity is lower than a threshold
        # since CHP heat Link is proportional to CHP electric Link, make sure threshold is compatible
        chp_heat = c.df.index[
            (c.df[f"{attr}_nom_extendable"] & c.df.index.str.contains("urban central"))
            & c.df.index.str.contains("CHP")
            & c.df.index.str.contains("heat")
        ]

        if not chp_heat.empty:
            threshold_chp_heat = (
                capacity_threshold
                * c.df.efficiency[chp_heat.str.replace("heat", "electric")].values
                * c.df.p_nom_ratio[chp_heat.str.replace("heat", "electric")].values
                / c.df.efficiency[chp_heat].values
            )
            n_p.remove(
                c.name,
                chp_heat[c.df.loc[chp_heat, f"{attr}_nom_opt"] < threshold_chp_heat],
            )

        n_p.remove(
            c.name,
            c.df.index[
                (c.df[f"{attr}_nom_extendable"] & ~c.df.index.isin(chp_heat))
                & (c.df[f"{attr}_nom_opt"] < capacity_threshold)
            ],
        )

        # copy over assets but fix their capacity
        c.df[f"{attr}_nom"] = c.df[f"{attr}_nom_opt"]
        c.df[f"{attr}_nom_extendable"] = False

        n.add(c.name, c.df.index, **c.df)

        # copy time-dependent
        selection = n.component_attrs[c.name].type.str.contains(
            "series"
        ) & n.component_attrs[c.name].status.str.contains("Input")
        for tattr in n.component_attrs[c.name].index[selection]:
            # TODO: Needs to be rewritten to
            n._import_series_from_df(c.pnl[tattr], c.name, tattr)

    # deal with gas network
    if h2_retrofit:
        # subtract the already retrofitted from the maximum capacity
        h2_retrofitted_fixed_i = n.links[
            (n.links.carrier == "H2 pipeline retrofitted")
            & (n.links.build_year != year)
        ].index
        h2_retrofitted = n.links[
            (n.links.carrier == "H2 pipeline retrofitted")
            & (n.links.build_year == year)
        ].index

        # pipe capacity always set in prepare_sector_network to todays gas grid capacity * H2_per_CH4
        # and is therefore constant up to this point
        pipe_capacity = n.links.loc[h2_retrofitted, "p_nom_max"]
        # already retrofitted capacity from gas -> H2
        already_retrofitted = (
            n.links.loc[h2_retrofitted_fixed_i, "p_nom"]
            .rename(lambda x: x.split("-2")[0] + f"-{year}")
            .groupby(level=0)
            .sum()
        )
        remaining_capacity = pipe_capacity - already_retrofitted.reindex(
            index=pipe_capacity.index
        ).fillna(0)
        n.links.loc[h2_retrofitted, "p_nom_max"] = remaining_capacity

        # reduce gas network capacity
        gas_pipes_i = n.links[n.links.carrier == "gas pipeline"].index
        if not gas_pipes_i.empty:
            # subtract the already retrofitted from today's gas grid capacity
            pipe_capacity = n.links.loc[gas_pipes_i, "p_nom"]
            fr = "H2 pipeline retrofitted"
            to = "gas pipeline"
            CH4_per_H2 = 1 / h2_retrofit_capacity_per_ch4
            already_retrofitted.index = already_retrofitted.index.str.replace(fr, to)
            remaining_capacity = (
                pipe_capacity
                - CH4_per_H2
                * already_retrofitted.reindex(index=pipe_capacity.index).fillna(0)
            )
            n.links.loc[gas_pipes_i, "p_nom"] = remaining_capacity
            n.links.loc[gas_pipes_i, "p_nom_max"] = remaining_capacity

def add_planned_generation_capacities(n, year, file, onshore_regions_file):
    """
    Adding planned generation capacities under construction to the network.
    This includes both renewable and conventional power plants.
    Parameters
    ----------
    n : pypsa.Network
        The network to which planned capacities will be added.
    year : int
        The planning year for which the capacities are being added.
    Returns
    -------
    None
        This function modifies the network in place and does not return a value.
    """
    # Read regions
    onshore_regions = gpd.read_file(onshore_regions_file).set_index("name").to_crs(3857)

    # Read data with planned power plants under construction
    df_OIM_pp_uc = pd.read_csv(file)
    df_OIM_pp_uc.Technology = df_OIM_pp_uc.Technology.replace({"Natural Gas": "CCGT",
                                                                "biomass": "urban central solid biomass CHP",
                                                                "offwind-ac": "offwind-dc"}) # assumption: new offshore wind farms are DC-connected

    # Add planned capacities to network
    planning_horizon = snakemake.config["scenario"]["planning_horizons"]
    previous_year = planning_horizon[planning_horizon.index(year) - 1]
    for tech in df_OIM_pp_uc.Technology.unique():
        df_tech = df_OIM_pp_uc.query("Technology == @tech")[["DateIn", "DateOut", "Capacity", "bus"]]
        
        # check if DateIn is both > previous_year and <= year
        df_tech_in = df_tech.loc[(df_tech["DateIn"] > previous_year) & (df_tech["DateIn"] <= year)]

        # group by bus 
        df_tech_in_grouped = df_tech_in.groupby("bus").agg({"Capacity": "sum"})

        if df_tech_in_grouped["Capacity"].sum() == 0:
            continue

        capacity_added = df_tech_in_grouped["Capacity"].sum()

        logger.info(f"{tech} {capacity_added}MW will be added")

        if tech in ["onwind", "offwind-dc", "offwind-ac", "solar", "solar rooftop"]:    
            # We need to treat renewables separately, as they are included at different resource classes, representing 
            # different resource quality and thus different capacity factors. Each level has its own p_nom_max. We will 
            # thus need to distribute the planned capacity over the different resource levels.

            tech_already_added = False
            resource_classes = snakemake.config["renewable"][tech]["resource_classes"] 
            
            res_levels = range(resource_classes-1, -1, -1)

            res_level = res_levels[0]
            df_tech_in_grouped_index = df_tech_in_grouped.index
            df_tech_in_grouped.index = df_tech_in_grouped_index + f" {res_level} " + tech + "-" + str(year)

            planned_capacity= df_tech_in_grouped["Capacity"].copy()

            # in the case that some capacity is located in a cell where the technology is not feasible, then move it to the neighboring region
            onshore_regions_available = onshore_regions.loc[n.generators.query("carrier == @tech").bus]
            not_contained = planned_capacity.index.difference(n.generators.query("carrier == @tech").index)
            if len(not_contained) > 0:
                not_contained_index = list(pd.DataFrame(not_contained)[0].str.split(f"{res_level} {tech}", expand = True)[0].str.strip())
                logger.info("Planned capacity at resource level ", res_level, " for technology ", tech, " that cannot be added at the location specified in the data set: ", not_contained_index)
                for t in not_contained_index:
                    target = onshore_regions.loc[[t]]

                    nearest = gpd.sjoin_nearest(
                        target,
                        onshore_regions,
                        how="left",
                        distance_col="dist_m"
                    ).query("name_right != @t")

                    nearest = nearest.query("name_right.isin(@onshore_regions_available.index)").sort_values("dist_m")["name_right"].iloc[0]
                    planned_capacity.index = planned_capacity.index.str.replace(t, nearest)
                    planned_capacity = planned_capacity.groupby(planned_capacity.index).sum()

            stop = False
            i = 0
            while not stop:

                p_nom_max = n.generators.loc[planned_capacity.index, "p_nom_max"]

                # First, check if planned capaity exceeds the estimated technical potential
                if (planned_capacity > p_nom_max).any() and tech not in ["offwind-dc", "offwind-ac", "offwind"]: # make exception for offshore wind as we assume all planned capacity can be added at most abundant resource level
                    
                    if res_level < res_levels[0]:
                        print("Planned capacity still exceeds technical potential at resource level ", res_level, " for technology ", tech)

                    tech_already_added = True

                    if res_level != 0:

                        for idx in planned_capacity.index:
                            
                            p_nom_max_n = n.generators.loc[idx, "p_nom_max"]

                            if planned_capacity.loc[idx] > p_nom_max_n:
                                p_nom_max_df = planned_capacity.loc[idx]
                                n.generators.loc[idx, "p_nom_max"] = p_nom_max_df
                                n.generators.loc[idx, "p_nom_min"] = p_nom_max_df
                                planned_capacity.loc[idx] = planned_capacity.loc[idx] - p_nom_max_n if planned_capacity.loc[idx] - p_nom_max_n > 0 else 0

                            else:
                                n.generators.loc[idx, "p_nom_min"] = planned_capacity.loc[idx]

                        i += 1
                        res_level = res_levels[i]
                        planned_capacity.index = df_tech_in_grouped_index + f" {res_level} " + tech + "-" + str(year)

                    elif res_level == 0:
                        for idx in planned_capacity.index:
                            n.generators.loc[idx, "p_nom_min"] = planned_capacity.loc[idx]

                            # Adjust p_nom_max if planned capacity exceeds it
                            p_nom_max_n = n.generators.loc[idx, "p_nom_max"]
                            if planned_capacity.loc[idx] > p_nom_max_n:
                                p_nom_max_df = planned_capacity.loc[idx]
                                n.generators.loc[idx, "p_nom_max"] = p_nom_max_df

                        stop = True
                        
                elif not tech_already_added:
                    n.generators.loc[planned_capacity.index, "p_nom_min"] = planned_capacity.values

                    # Adjust p_nom_max if planned capacity exceeds it
                    if tech in ["offwind-dc", "offwind-ac"] and (planned_capacity > p_nom_max).any():
                        for idx in planned_capacity.index:
                            p_nom_max_new = planned_capacity.loc[idx]
                            p_nom_max_old = n.generators.loc[idx, "p_nom_max"]
                            if p_nom_max_new > p_nom_max_old:
                                n.generators.loc[idx, "p_nom_max"] = p_nom_max_new

                    stop = True

                else:
                    print("For", tech, ", total planned capacity added: ", capacity_added, " MW in resource level , stopped at ", res_levels[i-1])
                    stop = True
                    

        elif tech in ["CCGT", 
                        "nuclear", 
                        "urban central solid biomass CHP"
                        ]:
            
            planned_capacity = df_tech_in_grouped["Capacity"].copy()
            planned_capacity.index = planned_capacity.index + " " + tech + "-" + str(year)
            efficiency = n.links.loc[planned_capacity.index, "efficiency"]
            n.links.loc[planned_capacity.index, "p_nom_min"] = (planned_capacity / efficiency).values

            print("For", tech, ", total planned capacity added: ", capacity_added, " MW")

def add_planned_storage_capacities(n, year, file):
    """
    Adding planned electricity storage capacities under construction to the network.
    This includes battery storage and pumped hydro storage (PHS) power plants.
    Parameters
    ----------
    n : pypsa.Network
        The network to which planned storage capacities will be added.
    year : int
        The planning year for which the capacities are being added.
    Returns
    -------
    None
        This function modifies the network in place and does not return a value.
    """

    # Read cleaned data set with storage power plants 
    df_OIM_storage = pd.read_csv(file)

    # only consider storage planned and not yet in operation
    df_OIM_storage_uc = df_OIM_storage.query("status == 'under construction'")

    planning_horizon = snakemake.config["scenario"]["planning_horizons"]
    previous_year = planning_horizon[planning_horizon.index(year) - 1]

    # rename technologies to match pypsa-eur naming
    tech_rename = {"battery": "battery discharger",
                   "water-storage": "PHS"}
    for tech in df_OIM_storage_uc.Technology.unique():
        df_tech = df_OIM_storage_uc.query("Technology == @tech")
        df_tech_in = df_tech.loc[(df_tech["DateIn"] > previous_year) & (df_tech["DateIn"] <= year)].groupby("bus").agg({"Capacity": "sum"})

        df_tech_in_index = df_tech_in.index + " " + tech_rename[tech] + "-" + str(year)

        if tech in ["battery"]:
            battery_duration = 6 # assume 6 hours discharge time for battery storage units

            # Update power capacity
            n.links.loc[df_tech_in_index, 
                        "p_nom_min"] = df_tech_in["Capacity"].values
            
            # Update energy capacity
            n.stores.loc[df_tech_in_index.str.replace(" discharger", ""), 
                        "e_nom_min"] = df_tech_in["Capacity"].values * battery_duration
            
            logger.info("planned battery storage capacity added")

        elif tech in ["water-storage"]:
            df_tech_in_reservoir = df_tech.loc[(df_tech["DateIn"] > previous_year) & (df_tech["DateIn"] <= year)].groupby("bus").agg({"storage_capacity_mwh": "sum"})

            max_hours = df_tech_in_reservoir["storage_capacity_mwh"] / df_tech_in["Capacity"]

            storage_units = n.storage_units.copy()
            n_phs = storage_units.query("carrier == 'PHS'")
            
            # add new pumped hydro storage units
            n_UK_phs = n_phs.loc[n_phs.index.str.contains("GB")].iloc[0]
            
            uk_phs_df = pd.DataFrame(columns = n_UK_phs.index,
                                    index = df_tech_in_index)
            
            uk_phs_df.loc[:, :] = n_UK_phs.values
            uk_phs_df.loc[:, "bus"] = df_tech_in.index
            uk_phs_df.loc[:, "p_nom"] = df_tech_in["Capacity"].values
            uk_phs_df.loc[:, "max_hours"] = max_hours.values

            storage_units = pd.concat([storage_units, uk_phs_df], ignore_index=False).sort_index()
            n.storage_units = storage_units

            logger.info("planned PHS capacity added")

def disable_grid_expansion_if_limit_hit(n):
    """
    Check if transmission expansion limit is already reached; then turn off.

    In particular, this function checks if the total transmission
    capital cost or volume implied by s_nom_min and p_nom_min are
    numerically close to the respective global limit set in
    n.global_constraints. If so, the nominal capacities are set to the
    minimum and extendable is turned off; the corresponding global
    constraint is then dropped.
    """
    types = {"expansion_cost": "capital_cost", "volume_expansion": "length"}
    for limit_type in types:
        glcs = n.global_constraints.query(f"type == 'transmission_{limit_type}_limit'")

        for name, glc in glcs.iterrows():
            total_expansion = (
                (
                    n.lines.query("s_nom_extendable")
                    .eval(f"s_nom_min * {types[limit_type]}")
                    .sum()
                )
                + (
                    n.links.query("carrier == 'DC' and p_nom_extendable")
                    .eval(f"p_nom_min * {types[limit_type]}")
                    .sum()
                )
            ).sum()

            # Allow small numerical differences
            if np.abs(glc.constant - total_expansion) / glc.constant < 1e-6:
                logger.info(
                    f"Transmission expansion {limit_type} is already reached, disabling expansion and limit"
                )
                extendable_acs = n.lines.query("s_nom_extendable").index
                n.lines.loc[extendable_acs, "s_nom_extendable"] = False
                n.lines.loc[extendable_acs, "s_nom"] = n.lines.loc[
                    extendable_acs, "s_nom_min"
                ]

                extendable_dcs = n.links.query(
                    "carrier == 'DC' and p_nom_extendable"
                ).index
                n.links.loc[extendable_dcs, "p_nom_extendable"] = False
                n.links.loc[extendable_dcs, "p_nom"] = n.links.loc[
                    extendable_dcs, "p_nom_min"
                ]

                n.global_constraints.drop(name, inplace=True)


def adjust_renewable_profiles(n, input_profiles, params, year):
    """
    Adjusts renewable profiles according to the renewable technology specified,
    using the latest year below or equal to the selected year.
    """

    # temporal clustering
    dr = get_snapshots(params["snapshots"], params["drop_leap_day"])
    snapshotmaps = (
        pd.Series(dr, index=dr).where(lambda x: x.isin(n.snapshots), pd.NA).ffill()
    )

    for carrier in params["carriers"]:
        if carrier == "hydro":
            continue

        with xr.open_dataset(getattr(input_profiles, "profile_" + carrier)) as ds:
            if ds.indexes["bus"].empty or "year" not in ds.indexes:
                continue

            ds = ds.stack(bus_bin=["bus", "bin"])

            closest_year = max(
                (y for y in ds.year.values if y <= year), default=min(ds.year.values)
            )

            p_max_pu = ds["profile"].sel(year=closest_year).to_pandas()
            p_max_pu.columns = p_max_pu.columns.map(flatten) + f" {carrier}"

            # temporal_clustering
            p_max_pu = p_max_pu.groupby(snapshotmaps).mean()

            # replace renewable time series
            n.generators_t.p_max_pu.loc[:, p_max_pu.columns] = p_max_pu


def update_heat_pump_efficiency(n: pypsa.Network, n_p: pypsa.Network, year: int):
    """
    Update the efficiency of heat pumps from previous years to current year
    (e.g. 2030 heat pumps receive 2040 heat pump COPs in 2030).

    Parameters
    ----------
    n : pypsa.Network
        The original network.
    n_p : pypsa.Network
        The network with the updated parameters.
    year : int
        The year for which the efficiency is being updated.

    Returns
    -------
    None
        This function updates the efficiency in place and does not return a value.
    """

    # get names of heat pumps in previous iteration that cannot be replaced by direct utilisation in this iteration
    heat_pump_idx_previous_iteration = n_p.links.index[
        n_p.links.index.str.contains("heat pump")
        & n_p.links.index.str[:-4].isin(
            n.links_t.efficiency.columns.str.rstrip(  # sources that can be directly used are no longer represented by heat pumps in the dynamic efficiency dataframe
                str(year)
            )
        )
    ]
    # construct names of same-technology heat pumps in the current iteration
    corresponding_idx_this_iteration = heat_pump_idx_previous_iteration.str[:-4] + str(
        year
    )
    # update efficiency of heat pumps in previous iteration in-place to efficiency in this iteration
    n_p.links_t["efficiency"].loc[:, heat_pump_idx_previous_iteration] = (
        n.links_t["efficiency"].loc[:, corresponding_idx_this_iteration].values
    )

    # Change efficiency2 for heat pumps that use an explicitly modelled heat source
    previous_iteration_columns = heat_pump_idx_previous_iteration.intersection(
        n_p.links_t["efficiency2"].columns
    )
    current_iteration_columns = corresponding_idx_this_iteration.intersection(
        n.links_t["efficiency2"].columns
    )
    n_p.links_t["efficiency2"].loc[:, previous_iteration_columns] = (
        n.links_t["efficiency2"].loc[:, current_iteration_columns].values
    )


def update_dynamic_ptes_capacity(
    n: pypsa.Network, n_p: pypsa.Network, year: int
) -> None:
    """
    Updates dynamic pit storage capacity based on district heating temperature changes.

    Parameters
    ----------
    n : pypsa.Network
        Original network.
    n_p : pypsa.Network
        Network with updated parameters.
    year : int
        Target year for capacity update.

    Returns
    -------
    None
        Updates capacity in-place.
    """
    # pit storages in previous iteration
    dynamic_ptes_idx_previous_iteration = n_p.stores.index[
        n_p.stores.index.str.contains("water pits")
    ]
    # construct names of same-technology dynamic pit storage in the current iteration
    corresponding_idx_this_iteration = dynamic_ptes_idx_previous_iteration.str[
        :-4
    ] + str(year)
    # update pit storage capacity in previous iteration in-place to capacity in this iteration
    n_p.stores_t.e_max_pu[dynamic_ptes_idx_previous_iteration] = n.stores_t.e_max_pu[
        corresponding_idx_this_iteration
    ].values


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "add_brownfield",
            clusters="39",
            opts="",
            sector_opts="",
            planning_horizons=2050,
        )

    configure_logging(snakemake)  # pylint: disable=E0606
    set_scenario_config(snakemake)

    update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    logger.info(f"Preparing brownfield from the file {snakemake.input.network_p}")

    year = int(snakemake.wildcards.planning_horizons)

    n = pypsa.Network(snakemake.input.network)

    adjust_renewable_profiles(n, snakemake.input, snakemake.params, year)

    add_build_year_to_new_assets(n, year)

    n_p = pypsa.Network(snakemake.input.network_p)

    update_heat_pump_efficiency(n, n_p, year)

    if snakemake.params.tes and snakemake.params.dynamic_ptes_capacity:
        update_dynamic_ptes_capacity(n, n_p, year)

    add_brownfield(
        n,
        n_p,
        year,
        h2_retrofit=snakemake.params.H2_retrofit,
        h2_retrofit_capacity_per_ch4=snakemake.params.H2_retrofit_capacity_per_CH4,
        capacity_threshold=snakemake.params.threshold_capacity,
    )

    file_powerplants = snakemake.input.uk_brownfield_power_plant_under_construction
    onshore_regions = snakemake.input.onshore_regions
    add_planned_generation_capacities(n, year, file_powerplants, onshore_regions)
    file_storage = snakemake.input.uk_brownfield_storage
    add_planned_storage_capacities(n, year, file_storage)

    disable_grid_expansion_if_limit_hit(n)

    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))

    sanitize_custom_columns(n)
    sanitize_carriers(n, snakemake.config)
    n.export_to_netcdf(snakemake.output.network)
