# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Solves optimal operation and capacity for a network with the option to
iteratively optimize while updating line reactances.

This script is used for optimizing the electrical network as well as the
sector coupled network.

Description
-----------

Total annual system costs are minimised with PyPSA. The full formulation of the
linear optimal power flow (plus investment planning
is provided in the
`documentation of PyPSA <https://pypsa.readthedocs.io/en/latest/optimal_power_flow.html#linear-optimal-power-flow>`_.

The optimization is based on the :func:`network.optimize` function.
Additionally, some extra constraints specified in :mod:`solve_network` are added.

.. note::

    The rules ``solve_elec_networks`` and ``solve_sector_networks`` run
    the workflow for all scenarios in the configuration file (``scenario:``)
    based on the rule :mod:`solve_network`.
"""

import importlib
import logging
import os
import re
import sys
from functools import partial
from typing import Any

import linopy
import numpy as np
import pandas as pd
import pypsa
import xarray as xr
import yaml
from pypsa.descriptors import get_activity_mask
from pypsa.descriptors import get_switchable_as_dense as get_as_dense

from scripts.prepare_sector_network import determine_emission_sectors, build_carbon_budget, add_co2limit
from scripts._benchmark import memory_logger
from scripts._helpers import (
    PYPSA_V1,
    configure_logging,
    get,
    set_scenario_config,
    update_config_from_wildcards,
)

logger = logging.getLogger(__name__)

# Allow for PyPSA versions <0.35
if PYPSA_V1:
    pypsa.network.power_flow.logger.setLevel(logging.WARNING)
else:
    pypsa.pf.logger.setLevel(logging.WARNING)


class ObjectiveValueError(Exception):
    pass

def calculate_co2_limit(investment_year, options, countries):
    co2_budget = snakemake.params.co2_budget
    if isinstance(co2_budget, str) and co2_budget.startswith("cb"):
        fn = "results/" + snakemake.params.RDIR + "/csvs/carbon_budget_distribution.csv"
        if not os.path.exists(fn):
            emissions_scope = snakemake.params.emissions_scope
            input_co2 = snakemake.input.co2

            build_carbon_budget(
                co2_budget,
                snakemake.input.eurostat,
                fn,
                emissions_scope,
                input_co2,
                options,
                countries,
                snakemake.params.planning_horizons,
            )
        co2_cap = pd.read_csv(fn, index_col=0).squeeze()
        limit = co2_cap.loc[investment_year]
    else:
        limit = get(co2_budget, investment_year)

    return limit

def add_land_use_constraint_perfect(n: pypsa.Network) -> None:
    """
    Add global constraints for tech capacity limit.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network instance

    Returns
    -------
    pypsa.Network
        Network with added land use constraints
    """
    logger.info("Add land-use constraint for perfect foresight")

    def compress_series(s):
        def process_group(group):
            if group.nunique() == 1:
                return pd.Series(group.iloc[0], index=[None])
            else:
                return group

        return s.groupby(level=[0, 1]).apply(process_group)

    def new_index_name(t):
        # Convert all elements to string and filter out None values
        parts = [str(x) for x in t if x is not None]
        # Join with space, but use a dash for the last item if not None
        return " ".join(parts[:2]) + (f"-{parts[-1]}" if len(parts) > 2 else "")

    def check_p_min_p_max(p_nom_max):
        p_nom_min = n.generators[ext_i].groupby(grouper).sum().p_nom_min
        p_nom_min = p_nom_min.reindex(p_nom_max.index)
        check = (
            p_nom_min.groupby(level=[0, 1]).sum()
            > p_nom_max.groupby(level=[0, 1]).min()
        )
        if check.sum():
            logger.warning(
                f"summed p_min_pu values at node larger than technical potential {check[check].index}"
            )

    grouper = [n.generators.carrier, n.generators.bus, n.generators.build_year]
    ext_i = n.generators.p_nom_extendable
    # get technical limit per node and investment period
    p_nom_max = n.generators[ext_i].groupby(grouper).min().p_nom_max
    # drop carriers without tech limit
    p_nom_max = p_nom_max[~p_nom_max.isin([np.inf, np.nan])]
    # carrier
    carriers = p_nom_max.index.get_level_values(0).unique()
    gen_i = n.generators[(n.generators.carrier.isin(carriers)) & (ext_i)].index
    n.generators.loc[gen_i, "p_nom_min"] = 0
    # check minimum capacities
    check_p_min_p_max(p_nom_max)
    # drop multi entries in case p_nom_max stays constant in different periods
    # p_nom_max = compress_series(p_nom_max)
    # adjust name to fit syntax of nominal constraint per bus
    df = p_nom_max.reset_index()
    df["name"] = df.apply(
        lambda row: f"nom_max_{row['carrier']}"
        + (f"_{row['build_year']}" if row["build_year"] is not None else ""),
        axis=1,
    )

    for name in df.name.unique():
        df_carrier = df[df.name == name]
        bus = df_carrier.bus
        n.buses.loc[bus, name] = df_carrier.p_nom_max.values


def add_land_use_constraint(n: pypsa.Network, planning_horizons: str) -> None:
    """
    Add land use constraints for renewable energy potential.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network instance
    planning_horizons : str
        The planning horizon year as string

    Returns
    -------
    pypsa.Network
        Modified PyPSA network with constraints added
    """
    # warning: this will miss existing offwind which is not classed AC-DC and has carrier 'offwind'

    for carrier in [
        "solar",
        "solar rooftop",
        "solar-hsat",
        "onwind",
        "offwind-ac",
        "offwind-dc",
        "offwind-float",
    ]:
        ext_i = (n.generators.carrier == carrier) & ~n.generators.p_nom_extendable
        grouper = n.generators.loc[ext_i].index.str.replace(
            f" {carrier}.*$", "", regex=True
        )
        existing = n.generators.loc[ext_i, "p_nom"].groupby(grouper).sum()
        existing.index += f" {carrier}-{planning_horizons}"
        n.generators.loc[existing.index, "p_nom_max"] -= existing

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

    n.generators["p_nom_max"] = n.generators["p_nom_max"].clip(lower=0)


def add_solar_potential_constraints(n: pypsa.Network, config: dict) -> None:
    """
    Add constraint to make sure the sum capacity of all solar technologies (fixed, tracking, ets. ) is below the region potential.

    Example:
    ES1 0: total solar potential is 10 GW, meaning:
           solar potential : 10 GW
           solar-hsat potential : 8 GW (solar with single axis tracking is assumed to have higher land use)
    The constraint ensures that:
           solar_p_nom + solar_hsat_p_nom * 1.13 <= 10 GW
    """
    land_use_factors = {
        "solar-hsat": config["renewable"]["solar"]["capacity_per_sqkm"]
        / config["renewable"]["solar-hsat"]["capacity_per_sqkm"],
    }
    rename = {} if PYPSA_V1 else {"Generator-ext": "Generator"}

    solar_carriers = ["solar", "solar-hsat"]
    solar = n.generators[
        n.generators.carrier.isin(solar_carriers) & n.generators.p_nom_extendable
    ].index

    solar_today = n.generators[
        (n.generators.carrier == "solar") & (n.generators.p_nom_extendable)
    ].index
    solar_hsat = n.generators[(n.generators.carrier == "solar-hsat")].index

    if solar.empty:
        return

    land_use = pd.DataFrame(1, index=solar, columns=["land_use_factor"])
    for carrier, factor in land_use_factors.items():
        land_use = land_use.apply(
            lambda x: (x * factor) if carrier in x.name else x, axis=1
        )

    location = pd.Series(n.buses.index, index=n.buses.index)
    ggrouper = n.generators.loc[solar].bus
    rhs = (
        n.generators.loc[solar_today, "p_nom_max"]
        .groupby(n.generators.loc[solar_today].bus.map(location))
        .sum()
        - n.generators.loc[solar_hsat, "p_nom"]
        .groupby(n.generators.loc[solar_hsat].bus.map(location))
        .sum()
        * land_use_factors["solar-hsat"]
    ).clip(lower=0)

    lhs = (
        (n.model["Generator-p_nom"].rename(rename).loc[solar] * land_use.squeeze())
        .groupby(ggrouper)
        .sum()
    )

    logger.info("Adding solar potential constraint.")
    n.model.add_constraints(lhs <= rhs, name="solar_potential")


def algebra_process_emissions(n, index):
    pe = n.model["Link-p"].loc[:, index]*n.links.loc[index].efficiency*n.snapshot_weightings["generators"]
    pe_sum = pe.sum()

    return pe_sum

def algebra_generation_emissions(n, index):
    ge = n.model["Link-p"].loc[:, index]*n.links.loc[index].efficiency2*n.snapshot_weightings["generators"]
    ge_sum = ge.sum()

    return ge_sum

def algebra_carboncapture(n, index, sign = "negative"):
    # bus0 is the electricity bus
    # bus1 is the urban heating bus 
    # bus2 is the CO2 atmosphere
    # bus3 is the CO2 storage bus
    # sign convention: when power is being discharged from bus0, then p0 (or "p" in the constraint) is positive
    
    cc = n.model["Link-p"].loc[:, index]*n.links.loc[index].efficiency3*n.snapshot_weightings["generators"]
    
    cc_sum = cc.sum() if sign == "positive" else -cc.sum()

    return cc_sum

def algebra_imports(n, index, sign = "positive"):
    # bus0 is the co2 atmosphere
    # bus1 is the imported commodity  
    # sign convention: when power is being discharged from bus0, then p0 (or "p" in the constraint) is positive
    # no conversion efficiency, since it already is in units of CO2 emissions
    
    im = n.model["Link-p"].loc[:, index]*n.snapshot_weightings["generators"]
    
    im_sum = im.sum() if sign == "positive" else -im.sum()

    return im_sum

def algebra_bio_gas(n, index):

    bg = n.model["Link-p"].loc[:, index]*n.links.loc[index].efficiency3*n.snapshot_weightings["generators"]
    bg_sum = bg.sum()

    return bg_sum

def update_UK_gas_price(n):
    """
    Use specific gas price  

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network instance
    Returns
    -------
    pypsa.Network
        Modified PyPSA network with updated marginal costs
    """

    # Set UK gas price
    uk_gas_price = 24.5 # in Eur/MWh

    # UK gas fuel subsidy for existing gas power plants (CCGT)
    uk_gas_fuel_subsidy = 1.0 # fraction of gas price subsidised

    # Set marginal cost of UK gas turbines
    marginal_cost_CCGT = 2.6 # Eur/MWh

    # Set marginal cost of UK gas CCGT for pre-built plants
    links = n.links.copy()
    gas_CCGT = links.query("carrier == 'CCGT'")
    uk_gas_CCGT = gas_CCGT.loc[gas_CCGT.index.str.contains("GB")]
    uk_gas_CCGT_prebuilt = uk_gas_CCGT.loc[uk_gas_CCGT.build_year < 2025]
    links.loc[uk_gas_CCGT_prebuilt.index, "marginal_cost"] = marginal_cost_CCGT - uk_gas_fuel_subsidy*uk_gas_price # assuming gas fuel expenditures are covered by subsidies for existing gas power plants
    n.links = links

    # Set marginal cost of UK gas production and extraction
    generators = n.generators.copy()
    uk_generators = generators.loc[generators.index.str.contains("GB")]
    uk_gas_generators = uk_generators.loc[uk_generators.carrier == "gas"]   
    generators.loc[uk_gas_generators.index, "marginal_cost"] = uk_gas_price
    n.generators = generators 

    return n

def add_UK_fixed_electricity_generation_mix(n, base_year):
    investment_year = int(snakemake.wildcards.planning_horizons)

    if investment_year > base_year:
        logger.info("Planning year greater than base year, skipping UK minimum capacity factor constraint.")
        return

    # Exogenous electricity load
    elec_loads = n.loads.loc[n.loads.carrier.str.contains("electricity")]
    uk_elec_loads = elec_loads.loc[elec_loads.index.str.contains("GB")]

    load = (n.loads_t.p_set[uk_elec_loads.query("carrier == 'electricity'").index].sum().sum() + 
            (uk_elec_loads.loc[uk_elec_loads.carrier != 'electricity'].p_set*len(n.snapshots)).sum()
            )

    # Additional electricity demand from sector-coupling
    links_wo_transmission = n.links.drop(n.links.query("carrier == 'DC'").index)
    electricity_buses = list(n.buses.query('carrier == "AC"').index) + list(
        n.buses.query('carrier == "low voltage"').index
    )
    boolean_elec_demand_via_links = [
        links_wo_transmission.bus0[i] in electricity_buses
        for i in range(len(links_wo_transmission.bus0))
    ]
    boolean_elec_demand_via_links_series = pd.Series(boolean_elec_demand_via_links)
    elec_demand_via_links = links_wo_transmission.iloc[
        boolean_elec_demand_via_links_series[boolean_elec_demand_via_links_series].index
    ]

    # Drop storage dischargers as they are not a part of the generation mix
    elec_demand_via_links = elec_demand_via_links.drop(
        elec_demand_via_links.index[elec_demand_via_links.index.str.contains("discharge")]
    )

    # Drop distribution links
    elec_demand_via_links = elec_demand_via_links.drop(
        elec_demand_via_links.index[
            elec_demand_via_links.index.str.contains("distribution")
        ]
    )

    # Access data for UK
    elec_demand_via_links = elec_demand_via_links.loc[elec_demand_via_links.index.str.contains("GB")]

    uk_electricity_demand = n.model.variables["Link-p"].loc[:, elec_demand_via_links.index].sum() + load

    # https://grid.iamkate.com/, https://www.energyoasis.org.uk/blog/uk-renewable-energy-mix-2024  
    carriers = {
                "nuclear": 20, #[10, 20],
                "wind": 35, # [30, 35],
                "solar": 10,# [4, 8],
                # "gas": 30, #[25, 35]
                # "urban central solid biomass CHP": 8,
                }

    uk_buses = n.buses.loc[n.buses.index.str.contains("GB")].query("carrier == 'AC'")
    for carrier, c_range in carriers.items():

        print(carrier)

        if carrier in ["wind", "solar"]:

            uk_generators_t = n.generators.index[n.generators.index.str.contains(carrier) & n.generators.index.str.contains("GB")]

            # drop solar thermal 
            if carrier == "solar":
                uk_generators_t = uk_generators_t.drop(
                    uk_generators_t[uk_generators_t.str.contains("thermal")]
                )

            lhs = n.model.variables["Generator-p"].loc[:, uk_generators_t].sum()

        else:
            uk_power_generation_links = n.links.loc[n.links.bus1.isin(uk_buses)]
            carriers = f"{carrier} | OCGT | CCGT" if carrier == "gas" else carrier
            uk_power_generation_links = uk_power_generation_links.loc[uk_power_generation_links.index.str.contains(carriers)] 

            lhs = n.model.variables["Link-p"].loc[:, uk_power_generation_links.index].sum()

        # n.model.add_constraints(
        #     lhs >= (uk_electricity_demand * c_range / 100)
        #     ,
        #     name="lower_generation_limit_" + carrier,
        #     )

        n.model.add_constraints(
            lhs <= (uk_electricity_demand * c_range / 100)
            ,
            name="upper_generation_limit_" + carrier,
            )

        # undefine lhs 
        del lhs

        logger.info(f"Added generation limits for {carrier} in UK: {c_range}% of total electricity generation")

# def add_UK_greenfield_minimum_capacity_factors(n, capacity_factors):

#     T = len(n.snapshots)
#     for tech, minimum_capacity_factor in capacity_factors.items():

#         minimum_capacity_factor_t = minimum_capacity_factor[tech]

#         # Select UK links of that tech
#         tech_uk = n.links[
#             n.links.index.str.contains("GB") & (n.links.carrier == tech)
#         ]

#         # Prebuilt (brownfield) capacities only
#         tech_uk_prebuilt = tech_uk[tech_uk.build_year < investment_year] if not (investment_year == base_year) else tech_uk[tech_uk.build_year <= investment_year] 

#         # If no prebuilt capacity, skip constraint
#         if tech_uk_prebuilt.empty:
#             continue

#         # Total prebuilt capacity (check for zero or NaN)
#         tech_uk_cap = tech_uk_prebuilt.p_nom.sum()

#         if (tech_uk_cap is None) or (tech_uk_cap == 0) or (pd.isna(tech_uk_cap)):
#             # log about zero capacity
#             logger.info(f"No pre-built {tech} capacity in UK, skipping constraint.")
#             continue

#         # Total production of those links over all snapshots
#         tech_uk_prod = n.model["Link-p"].loc[:, tech_uk_prebuilt.index].sum()

#         # Minimum energy requirement: CF * capacity * time
#         min_energy = minimum_capacity_factor_t_y * tech_uk_cap * T

#         lhs = -tech_uk_prod + min_energy

#         # enforce: tech_uk_prod >= min_energy
#         n.model.add_constraints(lhs <= 0, name=f"greenfield_capacity_factor_{tech}")

#         # log
#         logger.info(f"Added minimum capacity factor constraint for UK {tech}: {minimum_capacity_factor}")


def add_UK_brownfield_minimum_capacity_factors(n, capacity_factors, base_year):

    # For example:
    # CCGT: 0.5 # https://wattdirection.substack.com/p/uk-combined-cycle-gas-power-stations-748
    # nuclear: 0.7 # https://assets.publishing.service.gov.uk/media/5a75a748e5274a545822d2b9/Nuclear_Capacity_in_the_UK.pdf 

    investment_year = int(snakemake.wildcards.planning_horizons)

    T = len(n.snapshots)
    for tech, minimum_capacity_factor in capacity_factors.items():

        if not investment_year in minimum_capacity_factor.keys():
            logger.info(f"No minimum capacity factor specified for {tech} in year {investment_year}, skipping constraint.")
            continue
        
        logger.info("Adding UK brownfield minimum capacity factors.")
        
        minimum_capacity_factor_t_y = minimum_capacity_factor[investment_year]

        # Select UK links of that tech
        tech_uk = n.links[
            n.links.index.str.contains("GB") & (n.links.carrier == tech)
        ]

        # Prebuilt (brownfield) capacities only
        tech_uk_prebuilt = tech_uk[tech_uk.build_year < investment_year] if not (investment_year == base_year) else tech_uk[tech_uk.build_year <= investment_year] 

        # If no prebuilt capacity, skip constraint
        if tech_uk_prebuilt.empty:
            continue

        # Total prebuilt capacity (check for zero or NaN)
        tech_uk_cap = tech_uk_prebuilt.p_nom.sum()

        if (tech_uk_cap is None) or (tech_uk_cap == 0) or (pd.isna(tech_uk_cap)):
            # log about zero capacity
            logger.info(f"No pre-built {tech} capacity in UK, skipping constraint.")
            continue

        # Total production of those links over all snapshots
        tech_uk_prod = n.model["Link-p"].loc[:, tech_uk_prebuilt.index].sum()

        # Minimum energy requirement: CF * capacity * time
        min_energy = minimum_capacity_factor_t_y * tech_uk_cap * T

        lhs = -tech_uk_prod + min_energy

        # enforce: tech_uk_prod >= min_energy
        n.model.add_constraints(lhs <= 0, name=f"capacity_factor_{tech}")

        # log
        logger.info(f"Added minimum capacity factor constraint for UK {tech}: {minimum_capacity_factor}")

def add_UK_deployment_rate_limits(n, deployment_rate_limits, base_year):

    investment_year = int(snakemake.wildcards.planning_horizons)

    if not investment_year > base_year:
        logger.info("Planning year equals base year where capacities are fixed, and, thus, deployment rate limits are not needed.")
        return

    uk_generators = n.generators.query("bus.str.contains('GB')")
    uk_links = n.links.loc[n.links.index.str.contains("GB")]
    for tech in deployment_rate_limits.keys():

        if type(deployment_rate_limits[tech]) not in [int, float]:
            return 

        elif tech in ["heat pump"]:
            # by 2025, UK has 250,000 heat pumps installed (https://www.edie.net/uk-passes-250000-heat-pump-milestone/)            
            uk_links_hp = uk_links.loc[uk_links.index.str.contains(tech)]   
            uk_links_hp_extend = uk_links_hp.query("p_nom_extendable == True")
            COP_avg = 3 # assumed average COP of heat pumps
            lhs = n.model["Link-p_nom"].loc[uk_links_hp_extend.index].sum() * COP_avg
            rhs = deployment_rate_limits[tech] 

        elif tech in ["transmission"]:
            # Transmission lines are divided into AC and DC. The expansion limits should be applied to the total transmission volume, which is the sum of AC and DC lines. 
            power_factor = 0.7 # conversion from line capacity in MVA to MW
            uk_transmission_links = n.links.query("(bus0.str.contains('GB') and bus1.str.contains('GB')) and carrier == 'DC' and not index.str.contains('rev')") # including only internal lines
            uk_transmission_lines = n.lines.query("bus0.str.contains('GB') and bus1.str.contains('GB')") # including only internal lines

            # Today's total transmission volume in the UK (in MW-km)
            dc_links_vol_today = (n.links.loc[uk_transmission_links.index].p_nom_min * uk_transmission_links.length).sum()
            ac_lines_vol_today = power_factor * (n.lines.loc[uk_transmission_lines.index].s_nom_min * uk_transmission_lines.length).sum()
            transmission_vol_today = dc_links_vol_today + ac_lines_vol_today

            # Extendable subsets
            ext_dc = uk_transmission_links[uk_transmission_links.p_nom_extendable]
            ext_ac = uk_transmission_lines[uk_transmission_lines.s_nom_extendable]

            # Variables
            dc_var = n.model["Link-p_nom"].loc[ext_dc.index]
            ac_var = n.model["Line-s_nom"].loc[ext_ac.index]

            # Existing lines/links capacity
            dc_existing = ext_dc.p_nom_min
            ac_existing = ext_ac.s_nom_min

            # Existing lines/links length
            dc_length = ext_dc.length
            ac_length = ext_ac.length

            # Build linear expressions
            dc_expansion = ((dc_var - dc_existing) * dc_length).sum()
            ac_expansion = power_factor * ((ac_var - ac_existing) * ac_length).sum()

            # The expansion limit is defined as a fraction of today's total transmission volume in the UK. 
            expansion_fraction = deployment_rate_limits[tech] # fraction of today's transmission volume that can be added per 5-year period
            lhs = dc_expansion + ac_expansion
            rhs = expansion_fraction * transmission_vol_today

        else:
            if tech in ["offwind"]:
                uk_generators_vre = uk_generators.loc[uk_generators.index.str.contains(tech)]   
            else:
                uk_generators_vre = uk_generators.query("carrier == @tech")
    
            uk_generators_vre_extend = uk_generators_vre.query("p_nom_extendable == True")

            if uk_generators_vre_extend.empty:
                logger.info(f"No extendable {tech} capacity in UK, skipping deployment rate limit constraint.")
                continue

            lhs = n.model["Generator-p_nom"].loc[uk_generators_vre_extend.index].sum()
            rhs = deployment_rate_limits[tech] 
        
        n.model.add_constraints(lhs <= rhs, name=f"deployment_rate_limit_{tech}_{investment_year}")
        logger.info(f"Added deployment rate limit for UK {tech}: {rhs} MW / 5-year period")

def add_split_co2_constraints(n: pypsa.Network, local_co2: dict) -> None:
    """
    This function adds local net co2 emissions constraints, according to the limits 
    specified in the "local_co2" variable in the config file. 
    For the remaining countries not listed, a collective co2 emisssions constraint is 
    added, with the limit equivalent to the carbon budget "co2_budget" defined in the 
    config file. 
    The current version excludes the region specified in the local_co2 from the collective 
    constraint. Later, we should make it an option of either including or excluding the region
    in the collective constraint.
    """
    co2_totals_file = snakemake.input.co2_totals
    co2_totals = 1e6 * pd.read_csv(co2_totals_file, index_col=0) # convert Mt to tCO2

    options = snakemake.params.sector
    sectors = determine_emission_sectors(options)
    nhours = n.snapshot_weightings.generators.sum()
    nyears = nhours / 8760

    investment_year = int(snakemake.wildcards.planning_horizons)
    local_co2_regions = list(local_co2.keys())

    expr = n.optimize.expressions.energy_balance(
                                                bus_carrier="co2",
                                                groupby=False
                                                )

    # PyPSA 1.x labels the component dimension "name"; earlier versions used the
    # component name ("Link"). Selecting on the wrong label raises KeyError.
    link_dim = "name" if "name" in expr.coords else "Link"

    # add local CO2 emissions constraints
    i = 0
    for region in local_co2_regions:

        if (not region in co2_totals.index) or (investment_year not in local_co2[region].keys()):
            local_co2_regions.remove(region)
            logger.info(f"Region {region} specified in local_co2 not found in co2_totals, skipping local CO2 constraint for this region.")
            continue
        
        # calculating the net CO2 emissions allowance for the country
        limit = local_co2[region][investment_year]

        co2_1990 = co2_totals.loc[region, sectors].sum() # tCO2 emissions per year
        net_co2_allowance = co2_1990 * limit * nyears
        logger.info(f"Net CO2 emissions cap for {region}: {net_co2_allowance}")

        # Add expression containing all CO2 sources and sinks attached to the country
        mask = expr.coords[link_dim].str.contains(region)
        expr_c_group = expr.sel({link_dim: mask, "group": 0}) # group 0 corresponds to links

        lhs = expr_c_group.sum()
        rhs = net_co2_allowance

        n.model.add_constraints(
            lhs <= rhs,
            name="local co2 emissions constraint " + region ,
        )

        if i == 0:
            mask_merged = mask.copy()
        else:
            mask_merged = mask_merged | mask

        i += 1

    # Add global co2 emissions constraint
    all_countries = snakemake.params.countries
    collective = [x for x in all_countries if x not in local_co2_regions]
    limit = calculate_co2_limit(investment_year, options, collective)
    co2_1990 = co2_totals.loc[collective, sectors].sum().sum()
    net_co2_allowance = co2_1990 * limit * nyears
    logger.info(f"Collective CO2 emissions budget for {len(collective)} countries: {net_co2_allowance}")

    # Exclude the regions already covered by a local constraint. If none were
    # added, mask_merged was never assigned and the collective budget covers
    # everything. Previously a bare except handled both this and the KeyError
    # from the wrong dimension label, silently applying the collective budget to
    # every region -- double-counting the ones with a local cap.
    if i > 0:
        expr_collective_group = expr.sel({link_dim: ~mask_merged, "group": 0})
    else:
        expr_collective_group = expr.sel({"group": 0})

    lhs = expr_collective_group.sum()
    rhs = net_co2_allowance

    n.model.add_constraints(
        lhs <= rhs,
        name="collective co2 emissions constraint",
    )

def add_co2_sequestration_limit(
    n: pypsa.Network,
    limit_dict: dict[str, float],
    planning_horizons: str | None,
) -> None:
    """
    Add a global constraint on the amount of Mt CO2 that can be sequestered.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network instance
    limit_dict : dict[str, float]
        CO2 sequestration potential limit constraints by year.
    planning_horizons : str, optional
        The current planning horizon year or None in perfect foresight
    """

    if not n.investment_periods.empty:
        nyears = n.snapshot_weightings.groupby(level="period").generators.sum() / 8760
        periods = n.investment_periods
        limit = pd.Series(
            {period: nyears[period] * get(limit_dict, period) for period in periods}
        )
        limit.index = limit.index.map(lambda s: f"co2_sequestration_limit-{s}")
        names = limit.index
    else:
        nyears = n.snapshot_weightings.generators.sum() / 8760
        limit = get(limit_dict, int(planning_horizons)) * nyears
        periods = np.nan
        names = "co2_sequestration_limit"

    n.add(
        "GlobalConstraint",
        names,
        sense=">=",
        constant=-limit * 1e6,
        type="operational_limit",
        carrier_attribute="co2 sequestered",
        investment_period=periods,
    )


def add_carbon_constraint(n: pypsa.Network, snapshots: pd.DatetimeIndex) -> None:
    glcs = n.global_constraints.query('type == "co2_atmosphere"')
    if glcs.empty:
        return
    for name, glc in glcs.iterrows():
        carattr = glc.carrier_attribute
        emissions = n.carriers.query(f"{carattr} != 0")[carattr]

        if emissions.empty:
            continue

        # stores
        bus_carrier = n.stores.bus.map(n.buses.carrier)
        stores = n.stores[bus_carrier.isin(emissions.index) & ~n.stores.e_cyclic]
        if not stores.empty:
            last = n.snapshot_weightings.reset_index().groupby("period").last()
            last_i = last.set_index([last.index, last.timestep]).index
            final_e = n.model["Store-e"].loc[last_i, stores.index]
            time_valid = int(glc.loc["investment_period"])
            time_i = pd.IndexSlice[time_valid, :]
            lhs = final_e.loc[time_i, :] - final_e.shift(snapshot=1).loc[time_i, :]

            rhs = glc.constant
            n.model.add_constraints(lhs <= rhs, name=f"GlobalConstraint-{name}")


def add_carbon_budget_constraint(n: pypsa.Network, snapshots: pd.DatetimeIndex) -> None:
    glcs = n.global_constraints.query('type == "Co2Budget"')
    if glcs.empty:
        return
    for name, glc in glcs.iterrows():
        carattr = glc.carrier_attribute
        emissions = n.carriers.query(f"{carattr} != 0")[carattr]

        if emissions.empty:
            continue

        # stores
        bus_carrier = n.stores.bus.map(n.buses.carrier)
        stores = n.stores[bus_carrier.isin(emissions.index) & ~n.stores.e_cyclic]
        if not stores.empty:
            last = n.snapshot_weightings.reset_index().groupby("period").last()
            last_i = last.set_index([last.index, last.timestep]).index
            final_e = n.model["Store-e"].loc[last_i, stores.index]
            time_valid = int(glc.loc["investment_period"])
            time_i = pd.IndexSlice[time_valid, :]
            weighting = n.investment_period_weightings.loc[time_valid, "years"]
            lhs = final_e.loc[time_i, :] * weighting

            rhs = glc.constant
            n.model.add_constraints(lhs <= rhs, name=f"GlobalConstraint-{name}")


def add_max_growth(n: pypsa.Network, opts: dict) -> None:
    """
    Add maximum growth rates for different carriers.
    """

    # take maximum yearly difference between investment periods since historic growth is per year
    factor = n.investment_period_weightings.years.max() * opts["factor"]
    for carrier in opts["max_growth"].keys():
        max_per_period = opts["max_growth"][carrier] * factor
        logger.info(
            f"set maximum growth rate per investment period of {carrier} to {max_per_period} GW."
        )
        n.carriers.loc[carrier, "max_growth"] = max_per_period * 1e3

    for carrier in opts["max_relative_growth"].keys():
        max_r_per_period = opts["max_relative_growth"][carrier]
        logger.info(
            f"set maximum relative growth per investment period of {carrier} to {max_r_per_period}."
        )
        n.carriers.loc[carrier, "max_relative_growth"] = max_r_per_period


def add_retrofit_gas_boiler_constraint(
    n: pypsa.Network, snapshots: pd.DatetimeIndex
) -> None:
    """
    Allow retrofitting of existing gas boilers to H2 boilers and impose load-following must-run condition on existing gas boilers.
    Modifies the network in place, no return value.

    n : pypsa.Network
        The PyPSA network to be modified
    snapshots : pd.DatetimeIndex
        The snapshots of the network
    """
    c = "Link"
    logger.info("Add constraint for retrofitting gas boilers to H2 boilers.")
    # existing gas boilers
    mask = n.links.carrier.str.contains("gas boiler") & ~n.links.p_nom_extendable
    gas_i = n.links[mask].index
    mask = n.links.carrier.str.contains("retrofitted H2 boiler")
    h2_i = n.links[mask].index

    n.links.loc[gas_i, "p_nom_extendable"] = True
    p_nom = n.links.loc[gas_i, "p_nom"]
    n.links.loc[gas_i, "p_nom"] = 0

    # heat profile
    cols = n.loads_t.p_set.columns[
        n.loads_t.p_set.columns.str.contains("heat")
        & ~n.loads_t.p_set.columns.str.contains("industry")
        & ~n.loads_t.p_set.columns.str.contains("agriculture")
    ]
    profile = n.loads_t.p_set[cols].div(
        n.loads_t.p_set[cols].groupby(level=0).max(), level=0
    )
    # to deal if max value is zero
    profile.fillna(0, inplace=True)
    profile.rename(columns=n.loads.bus.to_dict(), inplace=True)
    profile = profile.reindex(columns=n.links.loc[gas_i, "bus1"])
    profile.columns = gas_i

    rhs = profile.mul(p_nom)

    dispatch = n.model["Link-p"]
    active = get_activity_mask(n, c, snapshots, gas_i)
    rhs = rhs[active]
    if PYPSA_V1:
        p_gas = dispatch.sel(name=gas_i)
        p_h2 = dispatch.sel(name=h2_i)
    else:
        p_gas = dispatch.sel(Link=gas_i)
        p_h2 = dispatch.sel(Link=h2_i)

    lhs = p_gas + p_h2

    n.model.add_constraints(lhs == rhs, name="gas_retrofit")

def prepare_network(
    n: pypsa.Network,
    solve_opts: dict,
    foresight: str,
    planning_horizons: str | None,
    co2_sequestration_potential: dict[str, float],
    limit_max_growth: dict[str, Any] | None = None,
) -> None:
    """
    Prepare network with various constraints and modifications.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network instance
    solve_opts : Dict
        Dictionary of solving options containing clip_p_max_pu, load_shedding etc.
    foresight : str
        Planning foresight type ('myopic' or 'perfect')
    planning_horizons : str or None
        The current planning horizon year or None for perfect foresight
    co2_sequestration_potential : Dict[str, float]
        CO2 sequestration potential constraints by year

    Returns
    -------
    pypsa.Network
        Modified PyPSA network with added constraints
    """
    if "clip_p_max_pu" in solve_opts:
        for df in (
            n.generators_t.p_max_pu,
            n.generators_t.p_min_pu,
            n.links_t.p_max_pu,
            n.links_t.p_min_pu,
            n.storage_units_t.inflow,
        ):
            df.where(df > solve_opts["clip_p_max_pu"], other=0.0, inplace=True)

    if load_shedding := solve_opts.get("load_shedding"):
        # intersect between macroeconomic and surveybased willingness to pay
        # http://journal.frontiersin.org/article/10.3389/fenrg.2015.00055/full
        # TODO: retrieve color and nice name from config
        n.add("Carrier", "load", color="#dd2e23", nice_name="Load shedding")
        buses_i = n.buses.index
        if not np.isscalar(load_shedding):
            # TODO: do not scale via sign attribute (use Eur/MWh instead of Eur/kWh)
            load_shedding = 1e2  # Eur/kWh

        n.add(
            "Generator",
            buses_i,
            " load",
            bus=buses_i,
            carrier="load",
            sign=1e-3,  # Adjust sign to measure p and p_nom in kW instead of MW
            marginal_cost=load_shedding,  # Eur/kWh
            p_nom=1e9,  # kW
        )

    if solve_opts.get("curtailment_mode"):
        n.add("Carrier", "curtailment", color="#fedfed", nice_name="Curtailment")
        n.generators_t.p_min_pu = n.generators_t.p_max_pu
        buses_i = n.buses.query("carrier == 'AC'").index
        n.add(
            "Generator",
            buses_i,
            suffix=" curtailment",
            bus=buses_i,
            p_min_pu=-1,
            p_max_pu=0,
            marginal_cost=-0.1,
            carrier="curtailment",
            p_nom=1e6,
        )

    if solve_opts.get("noisy_costs"):
        for t in n.iterate_components():
            # if 'capital_cost' in t.df:
            #    t.df['capital_cost'] += 1e1 + 2.*(np.random.random(len(t.df)) - 0.5)
            if "marginal_cost" in t.df:
                t.df["marginal_cost"] += 1e-2 + 2e-3 * (
                    np.random.random(len(t.df)) - 0.5
                )

        for t in n.iterate_components(["Line", "Link"]):
            t.df["capital_cost"] += (
                1e-1 + 2e-2 * (np.random.random(len(t.df)) - 0.5)
            ) * t.df["length"]

    if solve_opts.get("nhours"):
        nhours = solve_opts["nhours"]
        n.set_snapshots(n.snapshots[:nhours])
        n.snapshot_weightings[:] = 8760.0 / nhours

    if foresight == "myopic":
        add_land_use_constraint(n, planning_horizons)

    if foresight == "perfect":
        add_land_use_constraint_perfect(n)
        if limit_max_growth is not None and limit_max_growth["enable"]:
            add_max_growth(n, limit_max_growth)

    if n.stores.carrier.eq("co2 sequestered").any():
        limit_dict = co2_sequestration_potential
        add_co2_sequestration_limit(
            n, limit_dict=limit_dict, planning_horizons=planning_horizons
        )


def add_CCL_constraints(
    n: pypsa.Network, config: dict, planning_horizons: str | None
) -> None:
    """
    Add CCL (country & carrier limit) constraint to the network.

    Add minimum and maximum levels of generator nominal capacity per carrier
    for individual countries. Opts and path for agg_p_nom_minmax.csv must be defined
    in config.yaml. Default file is available at data/agg_p_nom_minmax.csv.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network instance
    config : dict
        Configuration dictionary
    planning_horizons : str, optional
        The current planning horizon year or None in perfect foresight

    Example
    -------
    scenario:
        opts: [Co2L-CCL-24h]
    electricity:
        agg_p_nom_limits: data/agg_p_nom_minmax.csv
    """

    assert planning_horizons is not None, (
        "add_CCL_constraints are not implemented for perfect foresight, yet"
    )

    agg_p_nom_minmax = pd.read_csv(
        config["solving"]["agg_p_nom_limits"]["file"], index_col=[0, 1], header=[0, 1]
    )[planning_horizons]
    logger.info("Adding generation capacity constraints per carrier and country")
    p_nom = n.model["Generator-p_nom"]

    gens = n.generators.query("p_nom_extendable")

    if not PYPSA_V1:
        gens = gens.rename_axis(index="Generator-ext")

    if config["solving"]["agg_p_nom_limits"]["agg_offwind"]:
        rename_offwind = {
            "offwind-ac": "offwind-all",
            "offwind-dc": "offwind-all",
            "offwind-float": "offwind-all",
            "offwind": "offwind-all",
        }
        gens = gens.replace(rename_offwind)
    if config["solving"]["agg_p_nom_limits"]["agg_solar"]:
        rename_solar = {
            "solar": "solar-all",
            "solar-hsat": "solar-all",
            "solar rooftop": "solar-all",
        }
        gens = gens.replace(rename_solar)
    grouper = pd.concat([gens.bus.map(n.buses.country), gens.carrier], axis=1)
    lhs = p_nom.groupby(grouper).sum().rename(bus="country")

    if config["solving"]["agg_p_nom_limits"]["include_existing"]:
        gens_cst = n.generators.query("~p_nom_extendable").rename_axis(
            index="Generator-cst"
        )
        gens_cst = gens_cst[
            (gens_cst["build_year"] + gens_cst["lifetime"]) >= int(planning_horizons)
        ]
        if config["solving"]["agg_p_nom_limits"]["agg_offwind"]:
            gens_cst = gens_cst.replace(rename_offwind)
        if config["solving"]["agg_p_nom_limits"]["agg_solar"]:
            gens_cst = gens_cst.replace(rename_solar)
        rhs_cst = (
            pd.concat(
                [gens_cst.bus.map(n.buses.country), gens_cst[["carrier", "p_nom"]]],
                axis=1,
            )
            .groupby(["bus", "carrier"])
            .sum()
        )
        rhs_cst.index = rhs_cst.index.rename({"bus": "country"})
        rhs_min = agg_p_nom_minmax["min"].dropna()
        idx_min = rhs_min.index.join(rhs_cst.index, how="left")
        rhs_min = rhs_min.reindex(idx_min).fillna(0)
        rhs = (rhs_min - rhs_cst.reindex(idx_min).fillna(0).p_nom).dropna()
        rhs[rhs < 0] = 0
        minimum = xr.DataArray(rhs).rename(dim_0="group")
    else:
        minimum = xr.DataArray(agg_p_nom_minmax["min"].dropna()).rename(dim_0="group")

    index = minimum.indexes["group"].intersection(lhs.indexes["group"])
    if not index.empty:
        n.model.add_constraints(
            lhs.sel(group=index) >= minimum.loc[index], name="agg_p_nom_min"
        )

    if config["solving"]["agg_p_nom_limits"]["include_existing"]:
        rhs_max = agg_p_nom_minmax["max"].dropna()
        idx_max = rhs_max.index.join(rhs_cst.index, how="left")
        rhs_max = rhs_max.reindex(idx_max).fillna(0)
        rhs = (rhs_max - rhs_cst.reindex(idx_max).fillna(0).p_nom).dropna()
        rhs[rhs < 0] = 0
        maximum = xr.DataArray(rhs).rename(dim_0="group")
    else:
        maximum = xr.DataArray(agg_p_nom_minmax["max"].dropna()).rename(dim_0="group")

    index = maximum.indexes["group"].intersection(lhs.indexes["group"])
    if not index.empty:
        n.model.add_constraints(
            lhs.sel(group=index) <= maximum.loc[index], name="agg_p_nom_max"
        )


def add_EQ_constraints(n, o, scaling=1e-1):
    """
    Add equity constraints to the network.

    Currently this is only implemented for the electricity sector only.

    Opts must be specified in the config.yaml.

    Parameters
    ----------
    n : pypsa.Network
    o : str

    Example
    -------
    scenario:
        opts: [Co2L-EQ0.7-24h]

    Require each country or node to on average produce a minimal share
    of its total electricity consumption itself. Example: EQ0.7c demands each country
    to produce on average at least 70% of its consumption; EQ0.7 demands
    each node to produce on average at least 70% of its consumption.
    """
    # TODO: Generalize to cover myopic and other sectors?
    float_regex = r"[0-9]*\.?[0-9]+"
    level = float(re.findall(float_regex, o)[0])
    if o[-1] == "c":
        ggrouper = n.generators.bus.map(n.buses.country)
        lgrouper = n.loads.bus.map(n.buses.country)
        sgrouper = n.storage_units.bus.map(n.buses.country)
    else:
        ggrouper = n.generators.bus
        lgrouper = n.loads.bus
        sgrouper = n.storage_units.bus
    load = (
        n.snapshot_weightings.generators
        @ n.loads_t.p_set.groupby(lgrouper, axis=1).sum()
    )
    inflow = (
        n.snapshot_weightings.stores
        @ n.storage_units_t.inflow.groupby(sgrouper, axis=1).sum()
    )
    inflow = inflow.reindex(load.index).fillna(0.0)
    rhs = scaling * (level * load - inflow)
    p = n.model["Generator-p"]
    lhs_gen = (
        (p * (n.snapshot_weightings.generators * scaling))
        .groupby(ggrouper.to_xarray())
        .sum()
        .sum("snapshot")
    )
    # TODO: double check that this is really needed, why do have to subtract the spillage
    if not n.storage_units_t.inflow.empty:
        spillage = n.model["StorageUnit-spill"]
        lhs_spill = (
            (spillage * (-n.snapshot_weightings.stores * scaling))
            .groupby(sgrouper.to_xarray())
            .sum()
            .sum("snapshot")
        )
        lhs = lhs_gen + lhs_spill
    else:
        lhs = lhs_gen
    n.model.add_constraints(lhs >= rhs, name="equity_min")


def add_BAU_constraints(n: pypsa.Network, config: dict) -> None:
    """
    Add business-as-usual (BAU) constraints for minimum capacities.

    Parameters
    ----------
    n : pypsa.Network
        PyPSA network instance
    config : dict
        Configuration dictionary containing BAU minimum capacities
    """
    mincaps = pd.Series(config["electricity"]["BAU_mincapacities"])
    p_nom = n.model["Generator-p_nom"]
    ext_i = n.generators.query("p_nom_extendable")
    ext_carrier_i = xr.DataArray(ext_i.carrier)
    if not PYPSA_V1:
        ext_carrier_i = ext_carrier_i.rename_axis("Generator-ext")
    lhs = p_nom.groupby(ext_carrier_i).sum()
    rhs = mincaps[lhs.indexes["carrier"]].rename_axis("carrier")
    n.model.add_constraints(lhs >= rhs, name="bau_mincaps")


# TODO: think about removing or make per country
def add_SAFE_constraints(n, config):
    """
    Add a capacity reserve margin of a certain fraction above the peak demand.
    Renewable generators and storage do not contribute. Ignores network.

    Parameters
    ----------
        n : pypsa.Network
        config : dict

    Example
    -------
    config.yaml requires to specify opts:

    scenario:
        opts: [Co2L-SAFE-24h]
    electricity:
        SAFE_reservemargin: 0.1
    Which sets a reserve margin of 10% above the peak demand.
    """
    peakdemand = n.loads_t.p_set.sum(axis=1).max()
    margin = 1.0 + config["electricity"]["SAFE_reservemargin"]
    reserve_margin = peakdemand * margin
    conventional_carriers = config["electricity"]["conventional_carriers"]  # noqa: F841
    ext_gens_i = n.generators.query(
        "carrier in @conventional_carriers & p_nom_extendable"
    ).index
    p_nom = n.model["Generator-p_nom"].loc[ext_gens_i]
    lhs = p_nom.sum()
    exist_conv_caps = n.generators.query(
        "~p_nom_extendable & carrier in @conventional_carriers"
    ).p_nom.sum()
    rhs = reserve_margin - exist_conv_caps
    n.model.add_constraints(lhs >= rhs, name="safe_mintotalcap")


def add_operational_reserve_margin(n, sns, config):
    """
    Build reserve margin constraints based on the formulation given in
    https://genxproject.github.io/GenX/dev/core/#Reserves.

    Parameters
    ----------
        n : pypsa.Network
        sns: pd.DatetimeIndex
        config : dict

    Example:
    --------
    config.yaml requires to specify operational_reserve:
    operational_reserve: # like https://genxproject.github.io/GenX/dev/core/#Reserves
        activate: true
        epsilon_load: 0.02 # percentage of load at each snapshot
        epsilon_vres: 0.02 # percentage of VRES at each snapshot
        contingency: 400000 # MW
    """
    reserve_config = config["electricity"]["operational_reserve"]
    EPSILON_LOAD = reserve_config["epsilon_load"]
    EPSILON_VRES = reserve_config["epsilon_vres"]
    CONTINGENCY = reserve_config["contingency"]

    # Reserve Variables
    n.model.add_variables(
        0, np.inf, coords=[sns, n.generators.index], name="Generator-r"
    )
    reserve = n.model["Generator-r"]
    summed_reserve = reserve.sum("Generator")

    # Share of extendable renewable capacities
    ext_i = n.generators.query("p_nom_extendable").index
    vres_i = n.generators_t.p_max_pu.columns
    if not ext_i.empty and not vres_i.empty:
        capacity_factor = n.generators_t.p_max_pu[vres_i.intersection(ext_i)]
        p_nom_vres = n.model["Generator-p_nom"].loc[vres_i.intersection(ext_i)]
        if not PYPSA_V1:
            p_nom_vres = p_nom_vres.rename({"Generator-ext": "Generator"})
        lhs = summed_reserve + (
            p_nom_vres * (-EPSILON_VRES * xr.DataArray(capacity_factor))
        ).sum("Generator")

        # Total demand per t
        demand = get_as_dense(n, "Load", "p_set").sum(axis=1)

        # VRES potential of non extendable generators
        capacity_factor = n.generators_t.p_max_pu[vres_i.difference(ext_i)]
        renewable_capacity = n.generators.p_nom[vres_i.difference(ext_i)]
        potential = (capacity_factor * renewable_capacity).sum(axis=1)

        # Right-hand-side
        rhs = EPSILON_LOAD * demand + EPSILON_VRES * potential + CONTINGENCY

        n.model.add_constraints(lhs >= rhs, name="reserve_margin")

    # additional constraint that capacity is not exceeded
    gen_i = n.generators.index
    ext_i = n.generators.query("p_nom_extendable").index
    fix_i = n.generators.query("not p_nom_extendable").index

    dispatch = n.model["Generator-p"]
    reserve = n.model["Generator-r"]

    capacity_variable = n.model["Generator-p_nom"]
    if not PYPSA_V1:
        capacity_variable = capacity_variable.rename({"Generator-ext": "Generator"})
    capacity_fixed = n.generators.p_nom[fix_i]

    p_max_pu = get_as_dense(n, "Generator", "p_max_pu")

    lhs = dispatch + reserve - capacity_variable * xr.DataArray(p_max_pu[ext_i])

    rhs = (p_max_pu[fix_i] * capacity_fixed).reindex(columns=gen_i, fill_value=0)

    n.model.add_constraints(lhs <= rhs, name="Generator-p-reserve-upper")


def add_TES_energy_to_power_ratio_constraints(n: pypsa.Network) -> None:
    """
    Add TES constraints to the network.

    For each TES storage unit, enforce:
        Store-e_nom - etpr * Link-p_nom == 0

    Parameters
    ----------
    n : pypsa.Network
        A PyPSA network with TES and heating sectors enabled.

    Raises
    ------
    ValueError
        If no valid TES storage or charger links are found.
    RuntimeError
        If the TES storage and charger indices do not align.
    """
    indices_charger_p_nom_extendable = n.links.index[
        n.links.index.str.contains("water tanks charger|water pits charger")
        & n.links.p_nom_extendable
    ]

    indices_stores_e_nom_extendable = n.stores.index[
            n.stores.index.str.contains("water tanks|water pits")
            & n.stores.e_nom_extendable
        ]

    if indices_charger_p_nom_extendable.empty or indices_stores_e_nom_extendable.empty:
        return 

    # Ensure indices of chargers and stores match
    # Name the columns explicitly: pd.DataFrame(<Index>) names the column after
    # the index, which PyPSA 1.x calls "name" for every component (pre-1.0 used
    # the component name, i.e. "Link"/"Store"), breaking the lookups below.
    indices_charger_p_nom_extendable_df = pd.DataFrame(
        {"Link": indices_charger_p_nom_extendable}
    )
    indices_charger_p_nom_extendable_df["index_reduced"] = indices_charger_p_nom_extendable.str.split(" charger", expand = True).get_level_values(0)
    indices_stores_e_nom_extendable_df = pd.DataFrame(
        {"Store": indices_stores_e_nom_extendable}
    )
    indices_stores_e_nom_extendable_df["index_reduced"] = indices_stores_e_nom_extendable.str.split("-", expand = True).get_level_values(0)

    indices_stores_e_nom_extendable_df = indices_stores_e_nom_extendable_df.reset_index().set_index("index_reduced")
    indices_charger_p_nom_extendable_df = indices_charger_p_nom_extendable_df.reset_index().set_index("index_reduced")

    index_in_common = indices_charger_p_nom_extendable_df.index.intersection(indices_stores_e_nom_extendable_df.index)

    indices_charger_p_nom_extendable = pd.Index(indices_charger_p_nom_extendable_df.loc[index_in_common]["Link"].values)
    indices_stores_e_nom_extendable = pd.Index(indices_stores_e_nom_extendable_df.loc[index_in_common]["Store"].values)

    if indices_charger_p_nom_extendable.empty or indices_stores_e_nom_extendable.empty:
        logger.warning(
            "No valid extendable charger links or stores found for TES energy-to-power constraints.Not enforcing TES energy-to-power ratio constraints!"
        )
        return

    energy_to_power_ratio_values = n.links.loc[
        indices_charger_p_nom_extendable, "energy to power ratio"
    ].values

    linear_expr_list = []
    for charger, tes, energy_to_power_value in zip(
        indices_charger_p_nom_extendable,
        indices_stores_e_nom_extendable,
        energy_to_power_ratio_values,
    ):
        charger_var = n.model["Link-p_nom"].loc[charger]
        if not tes == charger.replace(" charger", ""):
            # e.g. "DE0 0 urban central water tanks charger-2050" -> "DE0 0 urban central water tanks-2050"
            raise RuntimeError(
                f"Charger {charger} and TES {tes} do not match. "
                "Ensure that the charger and TES are in the same location and refer to the same technology."
            )
        store_var = n.model["Store-e_nom"].loc[tes]
        linear_expr = store_var - energy_to_power_value * charger_var
        linear_expr_list.append(linear_expr)

    # Merge the individual expressions
    dim = "Store-ext, Link-ext" if PYPSA_V1 else "name"
    merged_expr = linopy.expressions.merge(
        linear_expr_list, dim=dim, cls=type(linear_expr_list[0])
    )

    n.model.add_constraints(merged_expr == 0, name="TES_energy_to_power_ratio")


def add_TES_charger_ratio_constraints(n: pypsa.Network) -> None:
    """
    Add TES charger ratio constraints.

    For each TES unit, enforce:
        Link-p_nom(charger) - efficiency * Link-p_nom(discharger) == 0

    Parameters
    ----------
    n : pypsa.Network
        A PyPSA network with TES and heating sectors enabled.

    Raises
    ------
    ValueError
        If no valid TES discharger or charger links are found.
    RuntimeError
        If the charger and discharger indices do not align.
    """
    indices_charger_p_nom_extendable = n.links.index[
        n.links.index.str.contains(
            "water tanks charger|water pits charger|aquifer thermal energy storage charger"
        )
        & n.links.p_nom_extendable
    ]
    indices_discharger_p_nom_extendable = n.links.index[
        n.links.index.str.contains(
            "water tanks discharger|water pits discharger|aquifer thermal energy storage discharger"
        )
        & n.links.p_nom_extendable
    ]

    if (
        indices_charger_p_nom_extendable.empty
        or indices_discharger_p_nom_extendable.empty
    ):
        logger.warning(
            "No valid extendable TES discharger or charger links found for TES charger ratio constraints. Not enforcing TES charger_ratio constraints."
        )
        return

    for charger, discharger in zip(
        indices_charger_p_nom_extendable, indices_discharger_p_nom_extendable
    ):
        if not charger.replace(" charger", " ") == discharger.replace(
            " discharger", " "
        ):
            # e.g. "DE0 0 urban central water tanks charger-2050" -> "DE0 0 urban central water tanks-2050"
            raise RuntimeError(
                f"Charger {charger} and discharger {discharger} do not match. "
                "Ensure that the charger and discharger are in the same location and refer to the same technology."
            )

    eff_discharger = n.links.efficiency[indices_discharger_p_nom_extendable].values
    lhs = (
        n.model["Link-p_nom"].loc[indices_charger_p_nom_extendable]
        - n.model["Link-p_nom"].loc[indices_discharger_p_nom_extendable]
        * eff_discharger
    )

    n.model.add_constraints(lhs == 0, name="TES_charger_ratio")


def add_battery_constraints(n, battery_techs):
    """
    Add constraint ensuring that charger = discharger, i.e.
    1 * charger_size - efficiency * discharger_size = 0
    """
    if not n.links.p_nom_extendable.any():
        return

    discharger_bool = n.links.index.str.contains("|".join([f"{tech} discharger" for tech in battery_techs]))
    charger_bool = n.links.index.str.contains("|".join([f"{tech} charger" for tech in battery_techs]))

    dischargers_ext = n.links[discharger_bool].query("p_nom_extendable").index
    chargers_ext = n.links[charger_bool].query("p_nom_extendable").index

    eff = n.links.efficiency[dischargers_ext].values
    lhs = (
        n.model["Link-p_nom"].loc[chargers_ext]
        - n.model["Link-p_nom"].loc[dischargers_ext] * eff
    )

    n.model.add_constraints(lhs == 0, name="Link-charger_ratio")


def add_lossy_bidirectional_link_constraints(n):
    if not n.links.p_nom_extendable.any() or not any(n.links.get("reversed", [])):
        return

    carriers = n.links.loc[n.links.reversed, "carrier"].unique()  # noqa: F841
    backwards = n.links.query(
        "carrier in @carriers and p_nom_extendable and reversed"
    ).index
    forwards = backwards.str.replace("-reversed", "")
    lhs = n.model["Link-p_nom"].loc[backwards]
    rhs = n.model["Link-p_nom"].loc[forwards]
    n.model.add_constraints(lhs == rhs, name="Link-bidirectional_sync")


def add_chp_constraints(n):
    electric = (
        n.links.index.str.contains("urban central")
        & n.links.index.str.contains("CHP")
        & n.links.index.str.contains("electric")
    )
    heat = (
        n.links.index.str.contains("urban central")
        & n.links.index.str.contains("CHP")
        & n.links.index.str.contains("heat")
    )

    electric_ext = n.links[electric].query("p_nom_extendable").index
    heat_ext = n.links[heat].query("p_nom_extendable").index

    electric_fix = n.links[electric].query("~p_nom_extendable").index
    heat_fix = n.links[heat].query("~p_nom_extendable").index

    p = n.model["Link-p"]  # dimension: [time, link]

    # output ratio between heat and electricity and top_iso_fuel_line for extendable
    if not electric_ext.empty:
        p_nom = n.model["Link-p_nom"]

        lhs = (
            p_nom.loc[electric_ext]
            * (n.links.p_nom_ratio * n.links.efficiency)[electric_ext].values
            - p_nom.loc[heat_ext] * n.links.efficiency[heat_ext].values
        )
        n.model.add_constraints(lhs == 0, name="chplink-fix_p_nom_ratio")

        rename = {} if PYPSA_V1 else {"Link-ext": "Link"}
        lhs = (
            p.loc[:, electric_ext]
            + p.loc[:, heat_ext]
            - p_nom.rename(rename).loc[electric_ext]
        )
        n.model.add_constraints(lhs <= 0, name="chplink-top_iso_fuel_line_ext")

    # top_iso_fuel_line for fixed
    if not electric_fix.empty:
        lhs = p.loc[:, electric_fix] + p.loc[:, heat_fix]
        rhs = n.links.p_nom[electric_fix]
        n.model.add_constraints(lhs <= rhs, name="chplink-top_iso_fuel_line_fix")

    # back-pressure
    if not electric.empty:
        lhs = (
            p.loc[:, heat] * (n.links.efficiency[heat] * n.links.c_b[electric].values)
            - p.loc[:, electric] * n.links.efficiency[electric]
        )
        n.model.add_constraints(lhs <= rhs, name="chplink-backpressure")


def add_pipe_retrofit_constraint(n):
    """
    Add constraint for retrofitting existing CH4 pipelines to H2 pipelines.
    """
    if "reversed" not in n.links.columns:
        n.links["reversed"] = False
    gas_pipes_i = n.links.query(
        "carrier == 'gas pipeline' and p_nom_extendable and ~reversed"
    ).index
    h2_retrofitted_i = n.links.query(
        "carrier == 'H2 pipeline retrofitted' and p_nom_extendable and ~reversed"
    ).index

    if h2_retrofitted_i.empty or gas_pipes_i.empty:
        return

    p_nom = n.model["Link-p_nom"]

    CH4_per_H2 = 1 / n.config["sector"]["H2_retrofit_capacity_per_CH4"]
    lhs = p_nom.loc[gas_pipes_i] + CH4_per_H2 * p_nom.loc[h2_retrofitted_i]
    rhs = n.links.p_nom[gas_pipes_i]
    if not PYPSA_V1:
        rhs = rhs.rename_axis("Link-ext")

    n.model.add_constraints(lhs == rhs, name="Link-pipe_retrofit")


def add_flexible_egs_constraint(n):
    """
    Upper bounds the charging capacity of the geothermal reservoir according to
    the well capacity.
    """
    well_index = n.links.loc[n.links.carrier == "geothermal heat"].index
    storage_index = n.storage_units.loc[
        n.storage_units.carrier == "geothermal heat"
    ].index

    p_nom_rhs = n.model["Link-p_nom"].loc[well_index]
    p_nom_lhs = n.model["StorageUnit-p_nom"].loc[storage_index]

    n.model.add_constraints(
        p_nom_lhs <= p_nom_rhs,
        name="upper_bound_charging_capacity_of_geothermal_reservoir",
    )


def add_cbam_transmission_capacity_constraints(n: pypsa.Network, sns: pd.DatetimeIndex) -> None:
    """
    Restrict the CO2 intensity copies of an interconnector to the rating of the
    single physical cable they represent.

    ``prepare_sector_network.duplicate_transmission`` replicates every HVDC link
    once per CO2 intensity class so that imported electricity keeps its emission
    label when it crosses a border. Each copy is a fully-fledged link with its
    own capacity, so on its own the split hands the model N times the physical
    transfer capability of every interconnector for free -- and, under myopic
    foresight, N times the capacity carried into the next planning horizon. The
    copies are therefore tied together here to behave as one asset:

        sum_c p_{c,t} <= p_max_pu_t * p_nom_owner    for every snapshot t
        sum_c p_{c,t} >= p_min_pu_t * p_nom_owner    (HVDC links are bidirectional)
        p_nom_c        = p_nom_owner                 (one capacity per cable)

    where "owner" is the copy that kept the capital cost of the cable, so a
    border expansion is decided and paid for once but usable by every class.
    """
    links = n.links

    if "cbam_owner" not in links.columns:
        return

    dup = links[links.cbam_owner.fillna("").ne("")]
    if dup.empty:
        return

    # Capacity-owning copies, and per class the copies of the same cables in the
    # same order, so that the flows summed below always refer to one cable.
    owners = dup.index[dup.index.values == dup.cbam_owner.values]
    if owners.empty:
        logger.warning(
            "Duplicated DC links found but none of them owns a capacity, "
            "skipping the joint transmission capacity constraint."
        )
        return

    by_level = {
        lvl: pd.Series(g.index.values, index=g.cbam_owner.values).reindex(owners)
        for lvl, g in dup.groupby("cbam_level")
    }
    incomplete = [lvl for lvl, c in by_level.items() if c.isna().any()]
    if incomplete:
        logger.warning(
            f"CO2 intensity levels {incomplete} do not cover every duplicated DC "
            "link, skipping the joint transmission capacity constraint."
        )
        return

    p = n.model["Link-p"]
    dim = "name" if "name" in p.dims else "Link"
    # PyPSA <1 puts the capacity variables on their own "Link-ext" dimension.
    rename = {} if PYPSA_V1 else {"Link-ext": dim}

    # Total flow per cable. linopy stacks the terms positionally and keeps the
    # coordinates of the first summand, which are the capacity-owning copies.
    flow = p.loc[:, by_level[dup.loc[owners[0], "cbam_level"]].values]
    for lvl, copies in by_level.items():
        if lvl == dup.loc[owners[0], "cbam_level"]:
            continue
        flow = flow + p.loc[:, copies.values]

    p_max_pu = get_as_dense(n, "Link", "p_max_pu", sns)[owners]
    p_min_pu = get_as_dense(n, "Link", "p_min_pu", sns)[owners]

    def as_da(df: pd.DataFrame) -> xr.DataArray:
        return xr.DataArray(
            df.values,
            dims=["snapshot", dim],
            coords={"snapshot": sns, dim: df.columns.values},
        )

    extendable = links.p_nom_extendable[owners].values
    ext, fix = owners[extendable], owners[~extendable]

    if not ext.empty:
        p_nom = n.model["Link-p_nom"].rename(rename).loc[ext]
        lhs = flow.sel({dim: ext})
        n.model.add_constraints(
            lhs - p_nom * as_da(p_max_pu[ext]) <= 0,
            name="Link-cbam_joint_capacity_upper_ext",
        )
        n.model.add_constraints(
            lhs - p_nom * as_da(p_min_pu[ext]) >= 0,
            name="Link-cbam_joint_capacity_lower_ext",
        )

    if not fix.empty:
        p_nom = links.p_nom[fix]
        lhs = flow.sel({dim: fix})
        n.model.add_constraints(
            lhs <= as_da(p_max_pu[fix] * p_nom),
            name="Link-cbam_joint_capacity_upper_fix",
        )
        n.model.add_constraints(
            lhs >= as_da(p_min_pu[fix] * p_nom),
            name="Link-cbam_joint_capacity_lower_fix",
        )

    # Let every copy report the capacity of the cable it shares. Without this the
    # capacity variables of the copies that carry no capital cost are free within
    # their bounds, and myopic foresight would hand those arbitrary values to the
    # next planning horizon as existing capacity.
    others = dup.index[dup.index.values != dup.cbam_owner.values]
    others = others[links.p_nom_extendable[others].values]
    matched = dup.loc[others, "cbam_owner"]
    shared = links.p_nom_extendable[matched.values].values
    others, matched = others[shared], matched[shared]

    if not others.empty:
        p_nom = n.model["Link-p_nom"].rename(rename)
        n.model.add_constraints(
            p_nom.loc[others] - p_nom.loc[matched.values] == 0,
            name="Link-cbam_shared_capacity",
        )

    logger.info(
        f"Added joint capacity constraints for {len(owners)} duplicated DC links "
        f"across {len(by_level)} CO2 intensity levels."
    )


def add_import_limit_constraint(n: pypsa.Network, sns: pd.DatetimeIndex, type: str, limit_sense: str = "<=", limit: dict | None = None) -> None:
    """
    Add constraint for limiting green energy imports (synthetic and biomass).
    Does not include fossil fuel imports.
    """

    nyears = n.snapshot_weightings.generators.sum() / 8760
    weightings = n.snapshot_weightings.loc[sns, "generators"]

    import_links = n.links.loc[n.links.carrier.str.contains("import")].index
    import_gens = n.generators.loc[n.generators.carrier.str.contains("import")].index

    if (import_links.empty and import_gens.empty):
        return

    if isinstance(limit, dict):

        for c, lim in limit.items():

            import_gens_c = import_gens[import_gens.str.contains(c)]
            import_links_c = import_links[import_links.str.contains(c)]

            # everything needs to be in MWh_fuel
            eff = n.links.loc[import_links_c, "efficiency"]

            p_gens = n.model["Generator-p"].loc[sns, import_gens_c]
            p_links = n.model["Link-p"].loc[sns, import_links_c]

            lhs = (p_gens * weightings).sum() + (p_links * eff * weightings).sum()

            rhs = lim * 1e6 * nyears

            n.model.add_constraints(lhs, limit_sense, rhs, name=f"{type}_import_limit_{c}")
            logger.info(f"Added {type} import limit constraint for {c} with {limit_sense} {lim} TWh_fuel/year")


def add_co2_atmosphere_constraint(n, snapshots):
    logger.info("Adding global CO2 constraint.")
    glcs = n.global_constraints[n.global_constraints.type == "co2_atmosphere"]

    if glcs.empty:
        return
    for name, glc in glcs.iterrows():
        carattr = glc.carrier_attribute
        emissions = n.carriers.query(f"{carattr} != 0")[carattr]

        if emissions.empty:
            continue

        # stores
        bus_carrier = n.stores.bus.map(n.buses.carrier)
        stores = n.stores[bus_carrier.isin(emissions.index) & ~n.stores.e_cyclic]
        if not stores.empty:
            last_i = snapshots[-1]
            lhs = n.model["Store-e"].loc[last_i, stores.index]
            rhs = glc.constant

            n.model.add_constraints(lhs <= rhs, name=f"GlobalConstraint-{name}")


def extra_functionality(
    n: pypsa.Network, snapshots: pd.DatetimeIndex, planning_horizons: str | None = None
) -> None:
    """
    Add custom constraints and functionality.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network instance with config and params attributes
    snapshots : pd.DatetimeIndex
        Simulation timesteps
    planning_horizons : str, optional
        The current planning horizon year or None in perfect foresight

    Collects supplementary constraints which will be passed to
    ``pypsa.optimization.optimize``.

    If you want to enforce additional custom constraints, this is a good
    location to add them. The arguments ``opts`` and
    ``snakemake.config`` are expected to be attached to the network.
    """
    config = n.config
    constraints = config["solving"].get("constraints", {})
    if constraints["BAU"] and n.generators.p_nom_extendable.any():
        add_BAU_constraints(n, config)
    if constraints["SAFE"] and n.generators.p_nom_extendable.any():
        add_SAFE_constraints(n, config)
    if constraints["CCL"] and n.generators.p_nom_extendable.any():
        add_CCL_constraints(n, config, planning_horizons)

    reserve = config["electricity"].get("operational_reserve", {})
    if reserve.get("activate"):
        add_operational_reserve_margin(n, snapshots, config)

    if EQ_o := constraints["EQ"]:
        add_EQ_constraints(n, EQ_o.replace("EQ", ""))

    if {"solar-hsat", "solar"}.issubset(
        config["electricity"]["renewable_carriers"]
    ) and {"solar-hsat", "solar"}.issubset(
        config["electricity"]["extendable_carriers"]["Generator"]
    ):
        add_solar_potential_constraints(n, config)

    if n.config.get("sector", {}).get("tes", False):
        if n.buses.index.str.contains(
            r"urban central heat|urban decentral heat|rural heat",
            case=False,
            na=False,
        ).any():
            add_TES_energy_to_power_ratio_constraints(n)
            add_TES_charger_ratio_constraints(n)

    battery_techs = ["battery", "redox flow battery"] 
    add_battery_constraints(n, battery_techs)
    add_lossy_bidirectional_link_constraints(n)
    add_pipe_retrofit_constraint(n)
    if n._multi_invest:
        # add_carbon_constraint(n, snapshots)
        # add_carbon_budget_constraint(n, snapshots)
        add_retrofit_gas_boiler_constraint(n, snapshots)
    elif not isinstance(config["local_co2"], dict):
        add_co2_atmosphere_constraint(n, snapshots)

    if config["sector"]["enhanced_geothermal"]["enable"]:
        add_flexible_egs_constraint(n)

    if config["sector"]["green_imports"]["enable"]:
        limit = config["sector"]["green_imports"]["limit"]
        limit_sense = config["sector"]["green_imports"]["limit_sense"]
        add_import_limit_constraint(n, snapshots, "green", limit_sense, limit)

    if config["sector"]["fossil_imports"]["enable"]:
        limit = config["sector"]["fossil_imports"]["limit"]
        limit_sense = config["sector"]["fossil_imports"]["limit_sense"]
        add_import_limit_constraint(n, snapshots, "fossil", limit_sense, limit)

    base_year = snakemake.config["scenario"]["planning_horizons"][0]
    uk_settings = snakemake.params.uk_settings

    if uk_settings["uk_fixed_electricity_generation_mix"]:
        logger.info("Adding UK fixed electricity generation mix.")
        add_UK_fixed_electricity_generation_mix(n, base_year)
    
    if isinstance(uk_settings["uk_brownfield_minimum_capacity_factors"], dict):
        capacity_factors = uk_settings["uk_brownfield_minimum_capacity_factors"]
        add_UK_brownfield_minimum_capacity_factors(n, capacity_factors, base_year)

    # if isinstance(uk_settings["uk_greenfield_minimum_capacity_factors"], dict):
    #     logger.info("Adding UK greenfield minimum capacity factors.")
    #     capacity_factors = uk_settings["uk_greenfield_minimum_capacity_factors"]
    #     add_UK_greenfield_minimum_capacity_factors(n, capacity_factors)

    if isinstance(uk_settings["uk_deployment_rate_limits"], dict):
        logger.info("Adding UK deployment rates.")
        add_UK_deployment_rate_limits(n, uk_settings["uk_deployment_rate_limits"], base_year)

    if isinstance(config["local_co2"], dict):
        logger.info("Adding local CO2 constraint.")
        add_split_co2_constraints(n, config["local_co2"])

    if snakemake.params.uk_settings_prepare.get("cbam", False):
        add_cbam_transmission_capacity_constraints(n, snapshots)

    if n.params.custom_extra_functionality:
        source_path = n.params.custom_extra_functionality
        assert os.path.exists(source_path), f"{source_path} does not exist"
        sys.path.append(os.path.dirname(source_path))
        module_name = os.path.splitext(os.path.basename(source_path))[0]
        module = importlib.import_module(module_name)
        custom_extra_functionality = getattr(module, module_name)
        custom_extra_functionality(n, snapshots, snakemake)  # pylint: disable=E0601

def check_objective_value(n: pypsa.Network, solving: dict) -> None:
    """
    Check if objective value matches expected value within tolerance.

    Parameters
    ----------
    n : pypsa.Network
        Network with solved objective
    solving : Dict
        Dictionary containing objective checking parameters

    Raises
    ------
    ObjectiveValueError
        If objective value differs from expected value beyond tolerance
    """
    check_objective = solving["check_objective"]
    if check_objective["enable"]:
        atol = check_objective["atol"]
        rtol = check_objective["rtol"]
        expected_value = check_objective["expected_value"]
        if not np.isclose(n.objective, expected_value, atol=atol, rtol=rtol):
            raise ObjectiveValueError(
                f"Objective value {n.objective} differs from expected value "
                f"{expected_value} by more than {atol}."
            )

def save_co2_constraint_duals(n: pypsa.Network) -> None:
    """
    Save dual values of CO2 constraints to CSV files.
    """
    # model specs
    investment_year = snakemake.wildcards.planning_horizons

    # get constraints
    constraints = pd.Series(n.model.constraints)
    co2_constraints = constraints.loc[constraints.str.contains("co2", case=False)]

    # loop over constraints and save duals
    for constraint_name in co2_constraints:
        df = pd.Series(n.model.dual[constraint_name].values)
        coord = list(n.model.dual[constraint_name].coords)

        if len(coord) == 1:
            df.index = pd.Series(n.model.dual[constraint_name].coords[coord[0]])
            df.index.name = coord[0]
        
        if len(coord) == 2:
            df.index = pd.Series(n.model.dual[constraint_name].coords[coord[0]])
            df.columns = pd.Series(n.model.dual[constraint_name].coords[coord[1]])

            df.index.name = coord[0]
            df.columns.name = coord[1]

        df.to_csv(f"results/{snakemake.params.RDIR}/networks/{constraint_name}_{investment_year}.csv")

def add_load_shedding(n):
    buses = n.buses.drop(index = n.buses.query("carrier.str.contains('co2')").index)

    n.add("Carrier", 
        "load",#
        color="#000000", 
        nice_name="Load shedding")
    
    n.add("Generator", 
        buses.index + " load shedding",
        bus=buses.index,
        carrier='load',
        marginal_cost=1e5, # Eur/MWh
        # intersect between macroeconomic and surveybased willingness to pay
        # http://journal.frontiersin.org/article/10.3389/fenrg.2015.00055/full
        p_nom_extendable = True,
        capital_cost = 0)


def freeze_uk_wind_projects(n, dictionary):
    investment_year = int(snakemake.wildcards.planning_horizons)

    for tech, years in dictionary.items():
        if not investment_year in years:
            continue

        # generators
        uk_generators = n.generators.query("bus.str.contains('GB')")
        
        elements = {"generators": uk_generators}
        attributes = {"generators": "p"}

        conditions = "carrier.str.contains(@tech)"

        for element_name, element in elements.items():
            att = attributes[element_name]
            df = getattr(n, element_name)
            elements_to_freeze = df.loc[element.index].query(conditions)

            # before freezing, merge p_nom_min and p_nom
            element_tf_special = elements_to_freeze.loc[elements_to_freeze[f"{att}_nom_min"] != elements_to_freeze[f"{att}_nom"]]
            df.loc[element_tf_special.index, f"{att}_nom"] = df.loc[element_tf_special.index, f"{att}_nom"] + df.loc[element_tf_special.index, f"{att}_nom_min"]
            df.loc[elements_to_freeze.index, f"{att}_nom_extendable"] = False

def remove_storage(n, carrier):
    df = getattr(n, "stores")
    uk_stores = n.stores.query("carrier.str.contains(@carrier) and bus.str.contains('GB')")
    df.loc[uk_stores.index, "e_nom_extendable"] = False

def freeze_uk_capacities(n, base_year):
    investment_year = int(snakemake.wildcards.planning_horizons)
    if investment_year > base_year:
        logger.info("Freezing capacities in UK only applies to the base year. Skipping.")
        return
    else:
        logger.info("Freezing capacities in UK for the baseyear.")

    # Freeze AC and DC transmission lines everywhere
    transmission_dic = {"AC": ["lines", "s_nom_extendable"], 
                        "DC": ["links", "p_nom_extendable"]}
    
    for c, element in transmission_dic.items():
        df = getattr(n, element[0])
        df_carrier = df.query("carrier == @c")
        df.loc[df_carrier.index, element[1]] = False

    # Freeze emerging technologies everywhere
    emerging_techs = ["CC", 
                      "DAC",
                      "methanol",
                      "Methanol",
                      "H2 Electrolysis", 
                      "H2 turbine", 
                      "H2 Fuel Cell", 
                      "methanolisation",
                      "redox flow", 
                      "compressed air", 
                      "molten salt",
                      "liquid air",
                      "biomass to liquid",
                      "solid biomass for industry heat",
                      "Sabatier",
                      "Fischer-Tropsch",
                      "NH3 turbine"]
    
    for et in emerging_techs:
        condition = "carrier.str.contains(@et)" if not et == "CC" else "carrier.str.endswith(@et)"
        df = getattr(n, "links")
        df_carrier = df.query(condition)
        df.loc[df_carrier.index, "p_nom_extendable"] = False

    # add load shedding everywhere
    add_load_shedding(n)

    # UK generators
    uk_generators = n.generators.query("bus.str.contains('GB')")
    
    # UK lines
    uk_lines = n.lines.query("bus0.str.contains('GB') or bus1.str.contains('GB')")
    
    # UK links
    carriers = ["AC", "urban central heat", "urban decentral heat", "rural heat"]
    uk_buses = n.buses.query("index.str.contains('GB') and carrier.isin(@carriers)")
    uk_links = n.links.query("bus1.isin(@uk_buses.index)")
    uk_links = uk_links.query("carrier.str.contains('OCGT') | carrier.str.contains('CCGT') | carrier.str.contains('nuclear') | carrier.str.contains('lignite') | carrier.str.contains('coal') | carrier.str.contains('oil') | carrier.str.contains('CHP') | carrier.str.contains('heat pump')") 
    pipelines = n.links.query("carrier.str.contains('pipeline') and (bus0.str.contains('GB') or bus1.str.contains('GB'))")
    uk_links = pd.concat([uk_links, pipelines])

    # stores
    uk_stores = n.stores.query("bus.str.contains('GB')")
    
    # storage units
    uk_storage_units = n.storage_units.query("bus.str.contains('GB')")
    
    elements = {
                "generators": uk_generators, 
                "lines": uk_lines, 
                "links": uk_links, 
                "stores": uk_stores, 
                "storage_units": uk_storage_units
                }
    attributes = {"generators": "p", "lines": "s", "links": "p", "stores": "e", "storage_units": "p"}

    for element_name, element in elements.items():
        att = attributes[element_name]
        df = getattr(n, element_name)
        conditions = f"capital_cost > 0 and {att}_nom_extendable"
        elements_to_freeze = df.loc[element.index].query(conditions)

        if element_name == "links":
            elements_to_freeze_rev_index = elements_to_freeze.index.str[:-4] + f"reversed-{base_year}"
            elements_to_freeze = pd.concat([elements_to_freeze, df.loc[df.index.intersection(elements_to_freeze_rev_index)]])

        # before freezing, merge p_nom_min and p_nom
        element_tf_special = elements_to_freeze.loc[elements_to_freeze[f"{att}_nom_min"] != elements_to_freeze[f"{att}_nom"]]
        df.loc[element_tf_special.index, f"{att}_nom"] = df.loc[element_tf_special.index, f"{att}_nom"] + df.loc[element_tf_special.index, f"{att}_nom_min"]
        
        df.loc[elements_to_freeze.index, f"{att}_nom_extendable"] = False

def remove_ie_from_network(n):
    """ 
    Remove Ireland from the network.

    This is a quick workaround of the error when running GB as the only country. 
    The workflow requires an IDEEES country to be present when building the 
    network (in this case, Ireland). Before solving the network, we remove all 
    components connected to Ireland, including buses, lines, links, generators, to 
    model Great Britain only.
    """
    # Remove all buses in Ireland
    ireland_buses = n.buses.loc[n.buses.index.str.contains("IE")].index
    n.remove("Bus", ireland_buses)

    # Remove all lines connected to these buses
    ireland_lines = n.lines[n.lines.bus0.isin(ireland_buses) | n.lines.bus1.isin(ireland_buses)].index
    n.remove("Line", ireland_lines)

    # Remove all links connected to these buses
    ireland_links = n.links[n.links.bus0.isin(ireland_buses) | n.links.bus1.isin(ireland_buses)].index
    n.remove("Link", ireland_links)

    # Remove all generators connected to these buses
    ireland_generators = n.generators[n.generators.bus.isin(ireland_buses)].index
    n.remove("Generator", ireland_generators)

    # Remove all loads connected to these buses
    ireland_loads = n.loads[n.loads.bus.isin(ireland_buses)].index
    n.remove("Load", ireland_loads)

    # Remove all storage units connected to these buses
    ireland_storage_units = n.storage_units[n.storage_units.bus.isin(ireland_buses)].index
    n.remove("StorageUnit", ireland_storage_units)

    # Remove all stores connected to these buses
    ireland_stores = n.stores[n.stores.bus.isin(ireland_buses)].index
    n.remove("Store", ireland_stores)
    
def solve_network(
    n: pypsa.Network,
    config: dict,
    params: dict,
    solving: dict,
    rule_name: str | None = None,
    planning_horizons: str | None = None,
    **kwargs,
) -> None:
    """
    Solve network optimization problem.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network instance
    config : Dict
        Configuration dictionary containing solver settings
    params : Dict
        Dictionary of solving parameters
    solving : Dict
        Dictionary of solving options and configuration
    rule_name : str, optional
        Name of the snakemake rule being executed
    planning_horizons : str, optional
            The current planning horizon year or None in perfect foresight
    **kwargs
        Additional keyword arguments passed to the solver

    Returns
    -------
    n : pypsa.Network
        Solved network instance
    status : str
        Solution status
    condition : str
        Termination condition

    Raises
    ------
    RuntimeError
        If solving status is infeasible or warning
    ObjectiveValueError
        If objective value differs from expected value
    """
    set_of_options = solving["solver"]["options"]
    cf_solving = solving["options"]

    kwargs["multi_investment_periods"] = config["foresight"] == "perfect"
    kwargs["solver_options"] = (
        solving["solver_options"][set_of_options] if set_of_options else {}
    )
    kwargs["solver_name"] = solving["solver"]["name"]
    kwargs["extra_functionality"] = partial(
        extra_functionality, planning_horizons=planning_horizons
    )
    kwargs["transmission_losses"] = cf_solving.get("transmission_losses", False)
    kwargs["linearized_unit_commitment"] = cf_solving.get(
        "linearized_unit_commitment", False
    )
    kwargs["assign_all_duals"] = cf_solving.get("assign_all_duals", False)
    kwargs["io_api"] = cf_solving.get("io_api", None)

    model_kwargs = cf_solving.get("model_kwargs", {})
    model_kwargs["solver_dir"] = os.environ.get('TMPDIR')
    kwargs["model_kwargs"] = model_kwargs

    kwargs["keep_files"] = cf_solving.get("keep_files", False)

    if kwargs["solver_name"] == "gurobi":
        logging.getLogger("gurobipy").setLevel(logging.CRITICAL)

    rolling_horizon = cf_solving.pop("rolling_horizon", False)
    skip_iterations = cf_solving.pop("skip_iterations", False)
    if not n.lines.s_nom_extendable.any():
        skip_iterations = True
        logger.info("No expandable lines found. Skipping iterative solving.")

    # add to network for extra_functionality
    n.config = config
    n.params = params

    if rolling_horizon and rule_name == "solve_operations_network":
        kwargs["horizon"] = cf_solving.get("horizon", 365)
        kwargs["overlap"] = cf_solving.get("overlap", 0)
        n.optimize.optimize_with_rolling_horizon(**kwargs)
        status, condition = "", ""
    elif skip_iterations:
        status, condition = n.optimize(**kwargs)
    else:
        kwargs["track_iterations"] = cf_solving["track_iterations"]
        kwargs["min_iterations"] = cf_solving["min_iterations"]
        kwargs["max_iterations"] = cf_solving["max_iterations"]
        if cf_solving["post_discretization"].pop("enable"):
            logger.info("Add post-discretization parameters.")
            kwargs.update(cf_solving["post_discretization"])
        status, condition = n.optimize.optimize_transmission_expansion_iteratively(
            **kwargs
        )

    if not rolling_horizon:
        if status != "ok":
            logger.warning(
                f"Solving status '{status}' with termination condition '{condition}'"
            )
        check_objective_value(n, solving)

    if "warning" in condition:
        raise RuntimeError("Solving status 'warning'. Discarding solution.")

    if "infeasible" in condition:
        labels = n.model.compute_infeasibilities()
        logger.info(f"Labels:\n{labels}")
        n.model.print_infeasibilities()
        raise RuntimeError("Solving status 'infeasible'. Infeasibilities computed.")

    save_co2_constraint_duals(n)

if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "solve_sector_network",
            opts="",
            clusters="5",
            configfiles="config/test/config.overnight.yaml",
            sector_opts="",
            planning_horizons="2030",
        )
    configure_logging(snakemake)
    set_scenario_config(snakemake)
    update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    solve_opts = snakemake.params.solving["options"]
    
    uk_settings = snakemake.params.uk_settings

    np.random.seed(solve_opts.get("seed", 123))

    n = pypsa.Network(snakemake.input.network)

    countries = snakemake.params.countries
    # "uk_only" is configured under uk_settings.prepare, whereas uk_settings
    # above is bound to uk_settings.solve.
    if snakemake.params.uk_settings_prepare["uk_only"]:
        countries.remove("IE")
        remove_ie_from_network(n)

    planning_horizons = snakemake.wildcards.get("planning_horizons", None)

    prepare_network(
        n,
        solve_opts=snakemake.params.solving["options"],
        foresight=snakemake.params.foresight,
        planning_horizons=planning_horizons,
        co2_sequestration_potential=snakemake.params["co2_sequestration_potential"],
        limit_max_growth=snakemake.params.get("sector", {}).get("limit_max_growth"),
    )

    base_year = snakemake.config["scenario"]["planning_horizons"][0]
    if uk_settings["uk_freeze_base_year_capacities"]:
        freeze_uk_capacities(n, base_year)

    if isinstance(uk_settings["uk_freeze_wind_projects"], dict):
        freeze_uk_wind_projects(n, uk_settings["uk_freeze_wind_projects"])

    storage_to_remove = uk_settings.get("uk_remove_storage", None)
    if isinstance(storage_to_remove, str):
        remove_storage(n, storage_to_remove)

    if not isinstance(snakemake.config["local_co2"], dict):
        countries = snakemake.params.countries
        co2_budget = snakemake.params.co2_budget
        options = snakemake.params.sector
        investment_year = int(snakemake.wildcards.planning_horizons)
        nhours = n.snapshot_weightings.generators.sum()
        nyears = nhours / 8760

        if isinstance(co2_budget, str) and co2_budget.startswith("cb"):
            fn = "results/" + snakemake.params.RDIR + "/csvs/carbon_budget_distribution.csv"
            if not os.path.exists(fn):
                emissions_scope = snakemake.params.emissions_scope
                input_co2 = snakemake.input.co2
                build_carbon_budget(
                    co2_budget,
                    snakemake.input.eurostat,
                    fn,
                    emissions_scope,
                    input_co2,
                    options,
                    countries,
                    snakemake.params.planning_horizons,
                )
            co2_cap = pd.read_csv(fn, index_col=0).squeeze()
            limit = co2_cap.loc[investment_year]
        else:
            limit = get(co2_budget, investment_year)

        add_co2limit(
            n,
            options,
            snakemake.input.co2_totals,
            countries,
            nyears,
            limit,
        )

    logging_frequency = snakemake.config.get("solving", {}).get(
        "mem_logging_frequency", 30
    )
    with memory_logger(
        filename=getattr(snakemake.log, "memory", None), interval=logging_frequency
    ) as mem:
        solve_network(
            n,
            config=snakemake.config,
            params=snakemake.params,
            solving=snakemake.params.solving,
            planning_horizons=planning_horizons,
            rule_name=snakemake.rule,
            log_fn=snakemake.log.solver,
        )

    logger.info(f"Maximum memory usage: {mem.mem_usage}")

    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output.network)

    with open(snakemake.output.config, "w") as file:
        yaml.dump(
            n.meta,
            file,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
