# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Build depleted gas fields potentials for hydrogen storage.

"""

import logging

import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)

def cluster_depleted_gas_fields(depleted_gas_fields_file, 
                                clustered_depleted_gas_fields,
                                fn_onshore, 
                                fn_offshore):

    regions = gpd.read_file(fn_onshore)
    offshoreregions = gpd.read_file(fn_offshore)
    
    df = pd.read_csv(depleted_gas_fields_file, index_col=0)
    fields_gdf = gpd.GeoDataFrame(
                                df,
                                geometry=gpd.points_from_xy(df["lon"], df["lat"]),
                                crs="EPSG:4326" # lat and lon coordinates are in WGS84 (EPSG:4326)
                                )

    regions = regions.to_crs("EPSG:27700")
    offshore_regions = offshoreregions.to_crs("EPSG:27700")
    fields_gdf = fields_gdf.to_crs("EPSG:27700")

    # UK shorelines
    shoreline_union = unary_union(regions.query("name.str.contains('GB')").geometry)

    # Calculate distance from each field point to the shoreline (in metres)
    fields_gdf["distance_to_shore_m"] = fields_gdf.geometry.apply(
        lambda point: point.distance(shoreline_union)
    )

    # Convert to kilometres for readability
    fields_gdf["distance_to_shore_km"] = fields_gdf["distance_to_shore_m"] / 1000

    # Distribute offshore gas fields to clustered offshore regions
    fields_gdf = gpd.sjoin(
        fields_gdf,
        offshore_regions,
        how="left",
        predicate="within"
    )

    # Export to CSV
    fields_gdf.drop(columns = ["distance_to_shore_m", "geometry", "index_right"], inplace=True)
    fields_gdf.rename(columns = {"name": "offshore_region"}, inplace=True)
    fields_gdf.loc[df.index, "capacity_TWh"] = df["capacity_TWh"]
    fields_gdf.to_csv(clustered_depleted_gas_fields)

if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("build_depleted_gas_field_potentials", clusters="37")

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    fn_onshore = snakemake.input.regions_onshore
    fn_offshore = snakemake.input.regions_offshore

    cluster_depleted_gas_fields(
        depleted_gas_fields_file=snakemake.input.depleted_gas_fields,
        clustered_depleted_gas_fields=snakemake.output.clustered_depleted_gas_fields,
        fn_onshore = fn_onshore, 
        fn_offshore = fn_offshore,
    )
