# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Distribute country-level energy demands by population.
"""

import logging

import pandas as pd

from scripts._helpers import configure_logging, get_snapshots, set_scenario_config

idx = pd.IndexSlice

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "build_population_weighted_energy_totals",
            kind="heat",
            clusters=60,
        )
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    config = snakemake.config["energy"]

    if snakemake.wildcards.kind == "heat":
        snapshots = get_snapshots(
            snakemake.params.snapshots, snakemake.params.drop_leap_day
        )
        data_years = snapshots.year.unique()
    else:
        data_years = int(config["energy_totals_year"])

    pop_layout = pd.read_csv(snakemake.input.clustered_pop_layout, index_col=0)
    
    totals_ungrouped = pd.read_csv(snakemake.input.energy_totals, index_col=[0, 1])
    totals = totals_ungrouped.loc[idx[:, data_years], :].groupby("country").mean()

    # read UK settings
    data_years_uk = snakemake.config["uk_settings"].get("uk_energy_balance_year", False)     
    if data_years_uk and data_years_uk != data_years:
        
        uk_year = data_years_uk
        uk_index = idx["GB", uk_year]
        
        while uk_index not in totals_ungrouped.index:
            uk_year -= 1
            uk_index = idx["GB", uk_year]

        if uk_year != data_years_uk:
            logger.info(f"Using UK totals from year {uk_index[1]} instead of {data_years_uk} as specified in the config.")

        if snakemake.wildcards.kind == "heat":
            totals_UK = totals_ungrouped.loc[idx["GB", uk_year], :]
            columns_intersect = totals.columns.intersection(totals_UK.index)
            totals.loc["GB", columns_intersect] = totals_UK.loc[columns_intersect]
        else:
            totals_UK = totals_ungrouped.loc[idx["GB", uk_year], :].groupby("country").mean()
            columns_intersect = totals.columns.intersection(totals_UK.columns)
            totals.loc["GB", columns_intersect] = totals_UK.loc["GB", columns_intersect]

    nodal_totals = totals.loc[pop_layout.ct].fillna(0.0)
    nodal_totals.index = pop_layout.index
    nodal_totals = nodal_totals.multiply(pop_layout.fraction, axis=0)

    nodal_totals.to_csv(snakemake.output[0])
