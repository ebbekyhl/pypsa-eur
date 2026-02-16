# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT


"""
Creates networks clustered to ``{cluster}`` number of zones with aggregated
buses and transmission corridors.

Outputs
-------

- ``resources/regions_onshore_base_s_{clusters}.geojson``:

    .. image:: img/regions_onshore_base_s_X.png
        :scale: 33 %

- ``resources/regions_offshore_base_s_{clusters}.geojson``:

    .. image:: img/regions_offshore_base_s_X.png
        :scale: 33 %

- ``resources/busmap_base_s_{clusters}.csv``: Mapping of buses from ``networks/base.nc`` to ``networks/base_s_{clusters}.nc``;
- ``resources/linemap_base_s_{clusters}.csv``: Mapping of lines from ``networks/base.nc`` to ``networks/base_s_{clusters}.nc``;
- ``networks/base_s_{clusters}.nc``:

    .. image:: img/base_s_X.png
        :scale: 40  %

Description
-----------

.. note::

    **Is it possible to run the model without the** ``simplify_network`` **rule?**

        No, the network clustering methods in the PyPSA module
        `pypsa.clustering.spatial <https://github.com/PyPSA/PyPSA/blob/master/pypsa/clustering/spatial.py>`_
        do not work reliably with multiple voltage levels and transformers.

Exemplary unsolved network clustered to 512 nodes:

.. image:: img/base_s_512.png
    :scale: 40  %
    :align: center

Exemplary unsolved network clustered to 256 nodes:

.. image:: img/base_s_256.png
    :scale: 40  %
    :align: center

Exemplary unsolved network clustered to 128 nodes:

.. image:: img/base_s_128.png
    :scale: 40  %
    :align: center

Exemplary unsolved network clustered to 37 nodes:

.. image:: img/base_s_37.png
    :scale: 40  %
    :align: center
"""

import logging
import warnings
from functools import reduce

import geopandas as gpd
import linopy
import numpy as np
import pandas as pd
import pypsa
import tqdm
import xarray as xr
from packaging.version import Version, parse
from pypsa.clustering.spatial import (
    busmap_by_greedy_modularity,
    busmap_by_hac,
    busmap_by_kmeans,
    get_clustering_from_busmap,
)
from scipy.sparse.csgraph import connected_components
from shapely.algorithms.polylabel import polylabel
from shapely.geometry import MultiPolygon, Polygon

from scripts._helpers import configure_logging, set_scenario_config

PD_GE_2_2 = parse(pd.__version__) >= Version("2.2")

warnings.filterwarnings(action="ignore", category=UserWarning)
idx = pd.IndexSlice
logger = logging.getLogger(__name__)

GEO_CRS = "EPSG:4326"
DISTANCE_CRS = "EPSG:3035"
BUS_TOL = 500  # meters

central_england_short = ["GBH11", "GBH12", "GBH21", "GBH23","GBH24", "GBF24", "GBF25", "GBJ14", "GBJ11", "GBH25"]
scotland_short = "GBM"
north_west_short = "GBD"
north_east_yorkshire_humber_short = ["GBC", "GBE"]
east_midland_short = ["GBE13", "GBF"]
west_midland_short = "GBG"
east_short = "GBH"
south_east_short = "GBJ"
south_west_short = "GBK"
wales_cymru_short = "GBL"
greater_london_short = "GBI"
north_ireland_short = "GBN"

dct1 = {
        "GB central england": central_england_short, 
        "GB east midland": east_midland_short, 
        "GB scotland": scotland_short, 
        "GB north west": north_west_short, 
        "GB north east yorkshire humber": north_east_yorkshire_humber_short, 
        "GB west midland": west_midland_short, 
        "GB east": east_short,
        "GB south east": south_east_short, 
        "GB south west": south_west_short, 
        "GB wales cymru": wales_cymru_short, 
        "GB greater london": greater_london_short,
        "GB north ireland": north_ireland_short
        }

dct1_rev = {}
for region, codes in dct1.items():
    if isinstance(codes, list):
        for code in codes:
            dct1_rev[code] = region
    else:
        dct1_rev[codes] = region

def create_neighbors_matrix(regions):
    neighbors_dct = {}
    for a in regions.admin.unique():

        regions_a = regions.query("admin == @a")

        neighbors_matrix = pd.DataFrame(columns = regions_a.index, 
                                    index = regions_a.index)

        for i in range(len(regions_a)):
            for j in range(len(regions_a)):

                if i != j:
                    intersect_ij = regions_a.iloc[i].geometry.boundary.intersection(regions_a.iloc[j].geometry.boundary)

                    if not intersect_ij.is_empty:
                        neighbors_matrix.loc[regions_a.index[i], regions_a.index[j]] = 1

        neighbors_dct[a] = neighbors_matrix

    return neighbors_dct

def collect_small_regions(regions, neighbors_dct):

    regions_merge = regions.copy()

    indices_dropped = []

    for c in regions_merge.index:

        if c in regions_merge.index:

            neighbors_dct_c = neighbors_dct[regions_merge.loc[c, "admin"]]

            if regions_merge.loc[c, "colors"] == 1:

                c_neighbors = neighbors_dct_c.loc[c].dropna()

                c_neighbors_index = c_neighbors.index[c_neighbors.index.isin(regions_merge.index)]

                regions_neighbors = regions_merge.loc[c_neighbors_index]

                # only consider neighbors that are also small in size:
                regions_neighbors_small = regions_neighbors.query("colors == 1")

                if not regions_neighbors_small.empty:
                    # add c to the list of neighbors:
                    regions_neighbors_c = pd.concat([regions_neighbors_small, pd.DataFrame(regions_merge.loc[c]).T])

                else:
                    # if no small neighbors, then merge to the largest of the large neighbors:
                    # add c to the list of neighbors:
                    regions_neighbors_c = pd.concat([regions_neighbors, pd.DataFrame(regions_merge.loc[c]).T])

                # largest region in the group of large/small neighbors:
                largest = regions_neighbors_c["size"].idxmax()

                # substations
                substations = regions_neighbors_c["substations"].sum()

                # if one region has already been merged, then drop it:
                regions_to_merge = regions_neighbors_c.loc[~regions_neighbors_c.index.isin(indices_dropped)]

                if not regions_to_merge.empty:

                    indices_to_drop = list(regions_to_merge.index.values)

                    regions_merge.loc[regions_to_merge.index, "admin1"] = largest
                    regions_merge.loc[regions_to_merge.index, "substations"] = substations

                    indices_dropped += indices_to_drop

        else:
            if c in indices_dropped:
                continue
            else:
                raise ValueError(c, "Region not found")

    # sanity check
    if regions_merge.query("colors == 1").loc[regions_merge.query("colors == 1")["admin1"].isna()].empty:
        print("All small regions merged - sanity check passed!")
    else:
        raise ValueError("Some small regions not merged - sanity check 2/2 failed!")

    regions_merge.loc[regions_merge.query("colors == 0").index, "admin1"] = regions_merge.query("colors == 0").index

    no_reduced = len(regions.index) - len(regions_merge["admin1"].unique())
    print(no_reduced, "regions reduced after merging small regions")

    return regions_merge

def merge_small_regions(regions, regions_post, neighbors_dct):

    # logger.info(f"Number of regions after merging step: {len(regions_post['admin1'].unique())}")
    
    for a1 in regions_post["admin1"].unique():
        regions_post_a1 = regions_post.query("admin1 == @a1")

        if len(regions_post_a1) > 1:
            index_to_drop = regions_post_a1.index

            regions_post_a1_merged = regions_post_a1.dissolve()
            regions_post_a1_merged.index = [a1]
            regions_post_a1_merged["colors"] = 0

            regions_post.drop(index=index_to_drop, inplace=True)

            regions_post = pd.concat([regions_post, regions_post_a1_merged])

        elif len(regions_post_a1) == 1 and regions_post_a1["colors"].values[0] == 1:

            try: 
                dct1_rev_a1 = dct1_rev[a1[0:5]]

            except:
                dct1_rev_a1 = dct1_rev[a1[0:3]]
                
            number_of_small_neighbors = regions.loc[neighbors_dct[dct1_rev_a1].loc[a1].dropna().index]["colors"].sum()
            
            if number_of_small_neighbors > 0:
                print(f"Region {a1} is small, it has {number_of_small_neighbors} small neighbors, and was for some reason not merged.")

        else:
            continue

    return regions_post

def aggregate_small_admin_subregions(admin_shapes_all, area_threshold, country = "GB"):

    admin_shapes_all = admin_shapes_all.to_crs(DISTANCE_CRS)

    admin_shapes = admin_shapes_all.query("country == @country")
    admin_shapes_c_index = admin_shapes.index

    # add color column for small (1) and large (0) regions
    admin_shapes_size = admin_shapes.geometry.area / 1e6 # km2
    admin_shapes["size"] = admin_shapes_size

    admin_shapes_size.loc[admin_shapes_size < area_threshold] = 1
    admin_shapes_size.loc[admin_shapes_size > area_threshold] = 0

    admin_shapes["colors"] = admin_shapes_size
    admin_shapes_new = admin_shapes.copy()

    for key, value in dct1.items():

        if key in ['GB central england', 'GB east midland']:
            admin_key = pd.concat([admin_shapes.query("contains.str.contains(@value[@i])") for i in range(len(value))]).drop_duplicates()
        elif type(value) == str:
            admin_key = admin_shapes.loc[admin_shapes.index.str.startswith(value)]
        else:
            admin_key = admin_shapes.loc[admin_shapes.index.str.startswith(value[0]) | admin_shapes.index.str.contains(value[1])]

        if "admin" in admin_shapes_new.columns:
            condition = admin_shapes_new.loc[admin_key.index, "admin"].isna()
            if condition.any():
                indices_to_replace = admin_shapes_new.loc[admin_key.index].loc[condition].index
            else:
                continue
        else:
            indices_to_replace = admin_key.index

        admin_shapes_new.loc[indices_to_replace, "admin"] = key

    admin_shapes_update = admin_shapes_new.copy()
    logger.info(f"Colors sum: {admin_shapes_update['colors'].sum()}")
    i = 0
    while admin_shapes_update["colors"].sum() > 3 and i < 10:
        # create neighbor matrix
        neighbors_dct = create_neighbors_matrix(admin_shapes_update)
        
        # collect all adjacent small regions 
        regions_uk_post = collect_small_regions(admin_shapes_update, neighbors_dct)
        
        # merge small neighboring regions
        admin_shapes_update = merge_small_regions(admin_shapes_update, regions_uk_post, neighbors_dct)
        
        # update sizes 
        admin_shapes_size = admin_shapes_update.geometry.area / 1e6 # km2
        admin_shapes_size.loc[admin_shapes_size < area_threshold] = 1
        admin_shapes_size.loc[admin_shapes_size > area_threshold] = 0

        admin_shapes_update["colors"] = admin_shapes_size

        i += 1

    # print admin_shapes_update
    logger.info(f"admin_shapes_update columns: {admin_shapes_update.columns}")
    logger.info(f"admin_shapes_update admin: {admin_shapes_update.admin.unique()}")

    admin_shapes_update.drop(columns=["size", "colors","admin", "admin1"], inplace=True)

    # drop indices for country
    admin_shapes_all.drop(admin_shapes_c_index, inplace=True)

    # add new admin regions for country
    admin_shapes_all = pd.concat([admin_shapes_all, admin_shapes_update])

    return admin_shapes_all.to_crs(GEO_CRS)

def group_clusters(n, region):
    """
    Group the buses in a region to one bus.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network to modify.
    region : str
        The region code (e.g. 'GBNI', 'IE') for which to correct the clusters.
    """
    ni_buses = n.buses.query("index.str.startswith(@region)")

    ni_bus_name = 'new_bus' # temporary name for country bus

    # add new bus for country
    n.add("Bus", 
            ni_bus_name, 
            country=region[0:2], 
            v_nom=380.0,
            x=ni_buses.x.mean(), 
            y=ni_buses.y.mean())

    # drop remaining buses in country and replace with new bus
    n.buses = n.buses.drop(ni_buses.index)
    new_bus = region + " 0"  # e.g. "GBNI 0"
    buses_copy = n.buses.copy()
    buses_copy.rename({ni_bus_name: new_bus}, inplace=True)
    n.buses = buses_copy

    # drop internal lines within listed region
    lines_country = n.lines.loc[n.lines.bus1.isin(ni_buses.index)]
    lines_country_0 = n.lines.loc[n.lines.bus0.isin(ni_buses.index)]
    lines_country_1 = lines_country_0.loc[lines_country_0.bus1.isin(ni_buses.index)]
    n.lines = n.lines.drop(lines_country_1.index)

    # rename bus0 and bus1 on interconnections
    for line in n.lines.itertuples():
        if line.bus0 in ni_buses.index:
            n.lines.at[line.Index, 'bus0'] = new_bus
            print("Bus 0 changed for lines")
        if line.bus1 in ni_buses.index:
            n.lines.at[line.Index, 'bus1'] = new_bus
            print("Bus 1 changed for lines")
        
    for link in n.links.itertuples():    
        if link.bus0 in ni_buses.index:
            n.links.at[link.Index, 'bus0'] = new_bus
            print("Bus 0 changed for links")
        if link.bus1 in ni_buses.index:
            n.links.at[link.Index, 'bus1'] = new_bus
            print("Bus 1 changed for links")

    return new_bus, n

def correct_busmap(busmap, bus):
    """
    Corrects the busmap for Northern Ireland to point to the new bus.
    
    Parameters
    ----------
    busmap : pd.Series
        The busmap to correct.
    """
    busmap_subset = busmap.loc[busmap.str.contains(bus[0:3])]

    busmap.loc[busmap_subset.index] = [bus] * len(busmap_subset)

    return busmap

def normed(x):
    return (x / x.sum()).fillna(0.0)


def weighting_for_country(df: pd.DataFrame, weights: pd.Series) -> pd.Series:
    w = normed(weights.reindex(df.index, fill_value=0))
    return (w * (100 / w.max())).clip(lower=1).astype(int)


def busmap_from_shapes(
    n: pypsa.Network,
    shapes: gpd.GeoDataFrame,
    buses: pd.DataFrame = None,
    cluster_names: str = "name",
    per_country: bool = False,
) -> pd.Series:
    """
    Create a busmap from target shapes.

    This function takes into account the coordinates of the buses assigns the buses to
    the closest, preferably covering shape in the set of target shapes.

    For the subset of buses which are not covered by target shapes, the geographically
    nearest shape is assigned.

    If "per_country" is True, the function assigns buses to shapes based on the country of the buses and the shapes.

    Parameters
    ----------
    n : pypsa.Network
        Non-clustered network.
    shapes : geopandas.GeoDataFrame
        Non-overlapping target shapes.
    buses : pd.DataFrame, optional
        Buses to be assigned to target shapes. If None, n.buses is used.
    cluster_names : str, optional
        Column name of the shapes to be used as cluster names.
    per_country : Bool, optional
        Apply the function to buses based on country.

    Returns
    -------
    pd.Series
        busmap with index of buses and values of shape names.
    """
    if not isinstance(shapes, gpd.GeoDataFrame):
        raise TypeError("Shapes must be a gpd.GeoDataFrame object")

    if buses is None:
        buses = n.buses

    if per_country:
        logger.info("Assigning buses to target shapes based on country.")
        if "country" not in shapes.columns:
            raise ValueError(
                "Shapes must contain a 'country' column for per-country assignment."
            )
        if not set(shapes.country).issuperset(buses.country):
            logger.warning("Not all countries in buses are covered by target shapes.")
        busmaps = []
        for country in buses.country.unique():
            country_buses = buses[buses.country == country]
            country_shapes = shapes[shapes.country == country]
            busmaps.append(
                busmap_from_shapes(
                    n,
                    country_shapes,
                    country_buses,
                    cluster_names=cluster_names,
                    per_country=False,
                )
            )
        busmap = pd.concat(busmaps).reindex(n.buses.index)

    else:
        shapes = shapes.set_index(cluster_names)
        points = gpd.points_from_xy(**buses[["x", "y"]], crs=GEO_CRS)
        coords = gpd.GeoDataFrame(geometry=points, index=buses.index)
        busmap = gpd.sjoin(coords, shapes, how="left")[cluster_names].rename("busmap")

        if busmap.isnull().any():
            unassigned = coords[busmap.isnull()]
            # Take a projection which properly handles distances for European areas.
            unassigned_converted = unassigned.to_crs(DISTANCE_CRS)
            shapes_converted = shapes.to_crs(DISTANCE_CRS)
            for i, row in unassigned_converted.iterrows():
                dists = shapes_converted.distance(row.geometry)
                busmap.at[i] = dists.idxmin()

    return busmap


def copperplate_buses(n: pypsa.Network, copperplate_regions: list[list[str]]):
    """
    Copperplate buses that belong to the same group.

    Based on the input pandas series, buses are grouped together into market zone by
    replacing existing connections between the buses with a new connection of infinite capacity.

    Parameters
    ----------
    n : pypsa.Network
    copperplate_regions : list[list[str]]
        List of groups of regions to copperplate
    """
    buses_to_regions_raw = {
        bus: "_".join(region) for region in copperplate_regions for bus in region
    }
    n.buses["zone"] = n.buses.index.map(lambda bus: buses_to_regions_raw.get(bus, bus))
    buses_to_regions = n.buses["zone"]
    regions_to_buses = buses_to_regions.groupby(buses_to_regions).apply(
        lambda x: set(x.index)
    )

    # Remove connections between buses in the same zone
    for c in n.branch_components:
        df = n.static(c)
        bus0_zones = df.bus0.map(buses_to_regions).values
        bus1_zones = df.bus1.map(buses_to_regions).values
        to_remove = df.index[bus0_zones == bus1_zones]
        if len(to_remove) > 0:
            n.remove(c, to_remove)

    # Add new lines with infinite capacity within each zone
    for zone, buses in regions_to_buses.items():
        if len(buses) > 1:
            logging.info(
                f"Copperplating together the following buses: {', '.join(buses)}"
            )

            # Create lines between first bus and all others
            first_bus = list(buses)[0]
            other_buses = list(buses)[1:]

            for i, bus in enumerate(other_buses):
                n.add(
                    "Link",
                    f"copper_{zone}_{i}",
                    carrier="copper",
                    bus0=first_bus,
                    bus1=bus,
                    p_nom=float("inf"),
                    p_min_pu=-1,
                    underwater_fraction=0.0,
                    under_construction=0.0,
                )


def get_feature_data_for_hac(fn: str) -> pd.DataFrame:
    ds = xr.open_dataset(fn)
    feature_data = (
        pd.concat([ds[var].to_pandas() for var in ds.data_vars], axis=0).fillna(0.0).T
    )
    feature_data.columns = feature_data.columns.astype(str)
    return feature_data


def fix_country_assignment_for_hac(n: pypsa.Network) -> None:
    # overwrite country of nodes that are disconnected from their country-topology
    for country in n.buses.country.unique():
        m = n[n.buses.country == country].copy()

        _, labels = connected_components(m.adjacency_matrix(), directed=False)

        component = pd.Series(labels, index=m.buses.index)
        component_sizes = component.value_counts()

        if len(component_sizes) > 1:
            disconnected_bus = component[component == component_sizes.index[-1]].index[
                0
            ]

            neighbor_bus = n.lines.query(
                "bus0 == @disconnected_bus or bus1 == @disconnected_bus"
            ).iloc[0][["bus0", "bus1"]]
            new_country = list(set(n.buses.loc[neighbor_bus].country) - {country})[0]

            logger.info(
                f"overwriting country `{country}` of bus `{disconnected_bus}` "
                f"to new country `{new_country}`, because it is disconnected "
                "from its initial inter-country transmission grid."
            )
            n.buses.at[disconnected_bus, "country"] = new_country


def distribute_n_clusters_to_countries(
    n: pypsa.Network,
    n_clusters: int,
    cluster_weights: pd.Series,
    focus_weights: dict | None = None,
    solver_name: str = "scip",
) -> pd.Series:
    """
    Determine the number of clusters per country.
    """
    L = (
        cluster_weights.groupby([n.buses.country, n.buses.sub_network])
        .sum()
        .pipe(normed)
    )

    N = n.buses.groupby(["country", "sub_network"]).size()[L.index]

    assert n_clusters >= len(N) and n_clusters <= N.sum(), (
        f"Number of clusters must be {len(N)} <= n_clusters <= {N.sum()} for this selection of countries."
    )

    if isinstance(focus_weights, dict):
        total_focus = sum(list(focus_weights.values()))

        assert total_focus <= 1.0, (
            "The sum of focus weights must be less than or equal to 1."
        )

        for country, weight in focus_weights.items():
            L[country] = weight / len(L[country])

        remainder = [
            c not in focus_weights.keys() for c in L.index.get_level_values("country")
        ]
        L[remainder] = L.loc[remainder].pipe(normed) * (1 - total_focus)

        logger.warning("Using custom focus weights for determining number of clusters.")

    assert np.isclose(L.sum(), 1.0, rtol=1e-3), (
        f"Country weights L must sum up to 1.0 when distributing clusters. Is {L.sum()}."
    )

    m = linopy.Model()
    clusters = m.add_variables(
        lower=1, upper=N, coords=[L.index], name="n", integer=True
    )
    m.add_constraints(clusters.sum() == n_clusters, name="tot")
    # leave out constant in objective (L * n_clusters) ** 2
    m.objective = (clusters * clusters - 2 * clusters * L * n_clusters).sum()
    if solver_name == "gurobi":
        logging.getLogger("gurobipy").propagate = False
    elif solver_name not in ["scip", "cplex", "xpress", "copt", "mosek"]:
        logger.info(
            f"The configured solver `{solver_name}` does not support quadratic objectives. Falling back to `scip`."
        )
        solver_name = "scip"
    m.solve(solver_name=solver_name)
    return m.solution["n"].to_series().astype(int)

def allocate_integer_clusters(
    weights: pd.Series,
    n_total: int,
    min_per_group: int = 0,
) -> pd.Series:
    """
    Allocate n_total integer clusters across groups proportionally to weights.

    - Ensures sum == n_total
    - Enforces min_per_group where possible
    """
    weights = weights.fillna(0.0)

    if len(weights) == 0:
        return weights.astype(int)

    # If everything is zero, fall back to equal weights
    if weights.sum() <= 0:
        weights = pd.Series(1.0, index=weights.index)

    # Enforce minimums if requested
    min_total = min_per_group * len(weights)
    if min_total > n_total:
        raise ValueError(
            f"Cannot allocate {n_total} clusters with min_per_group={min_per_group} "
            f"across {len(weights)} groups (need at least {min_total})."
        )

    # Start with minimums
    alloc = pd.Series(min_per_group, index=weights.index, dtype=int)
    remaining = n_total - alloc.sum()
    if remaining == 0:
        return alloc

    # Proportional allocation on remaining
    w = weights / weights.sum()
    raw = w * remaining
    floored = np.floor(raw).astype(int)
    alloc += floored
    remainder = remaining - floored.sum()

    if remainder > 0:
        # Largest remainder method
        frac = (raw - np.floor(raw)).sort_values(ascending=False)
        winners = frac.index[:remainder]
        alloc.loc[winners] += 1

    # Safety
    assert alloc.sum() == n_total
    return alloc

def make_country_subregion_labels(
    n: pypsa.Network,
    country: str,
    subregions_geojson: str,
    id_field: str = "name",
    on_unassigned: str = "nearest",
) -> pd.Series:
    """
    Returns a Series indexed by n.buses.index with values:
      - "<country>:<subregion_id>" for buses in the selected country
      - NaN for other buses
    """
    shapes = gpd.read_file(subregions_geojson)
    shapes.to_crs(GEO_CRS, inplace=True)

    if id_field not in shapes.columns:
        raise ValueError(
            f"GeoJSON does not contain id_field='{id_field}'. Columns: {list(shapes.columns)}"
        )

    # Optional: if geojson contains multiple countries, filter
    if "country" in shapes.columns:
        shapes = shapes[shapes["country"] == country]

    # Assign only that country's buses
    buses_c = n.buses[n.buses.country == country][["x", "y", "country"]].copy()

    # Use existing helper: assigns by polygon containment; if missing, assigns nearest polygon.
    busmap = busmap_from_shapes(
        n,
        shapes,
        buses=buses_c,
        cluster_names=id_field,
        per_country=False,
    )

    # busmap_from_shapes always fills by nearest if unassigned; if you want strictness, check:
    if on_unassigned == "error":
        # Recompute without nearest fallback would require extra logic;
        # simplest: detect if any were outside (not possible here because we already fill nearest).
        # Alternative: you can implement "error" by doing a pure sjoin first and raising if null.
        pass

    return busmap.astype(str).rename("region")

def busmap_for_n_clusters_by_bus_groups(
    n: pypsa.Network,
    bus_groups: pd.Series,          # index=buses, values=region label
    n_clusters_g: pd.Series,        # MultiIndex: (region, sub_network) -> int
    cluster_weights: pd.Series,
    algorithm: str = "kmeans",
    features: pd.DataFrame | None = None,
    **algorithm_kwds,
) -> pd.Series:
    """
    Like busmap_for_n_clusters, but groups buses by (bus_groups, sub_network)
    instead of (country, sub_network).
    """

    if algorithm == "hac" and features is None:
        raise ValueError("For HAC clustering, features must be provided.")

    if algorithm == "kmeans":
        algorithm_kwds.setdefault("n_init", 1000)
        algorithm_kwds.setdefault("max_iter", 30000)
        algorithm_kwds.setdefault("tol", 1e-6)
        algorithm_kwds.setdefault("random_state", 0)

    # Ensure aligned
    bus_groups = bus_groups.reindex(n.buses.index)

    # Create a working dataframe of buses with grouping keys
    df = n.buses[["sub_network"]].copy()
    df["region"] = bus_groups

    def busmap_for_group(x):
        region, sub_network = x.name
        prefix = f"{region} {sub_network} "
        k = int(n_clusters_g.loc[(region, sub_network)])

        logger.debug(
            f"Determining busmap for group {region}/{sub_network} "
            f"from {len(x)} buses to {k}."
        )

        if len(x) == 1:
            return pd.Series(prefix + "0", index=x.index)

        weight = weighting_for_country(x, cluster_weights)

        if algorithm == "kmeans":
            return prefix + busmap_by_kmeans(
                n, weight, k, buses_i=x.index, **algorithm_kwds
            )
        elif algorithm == "hac":
            return prefix + busmap_by_hac(
                n,
                k,
                buses_i=x.index,
                feature=features.reindex(x.index, fill_value=0.0),
            )
        elif algorithm == "modularity":
            return prefix + busmap_by_greedy_modularity(
                n, k, buses_i=x.index
            )
        else:
            raise ValueError(
                f"`algorithm` must be one of 'kmeans' or 'hac' or 'modularity'. Is {algorithm}."
            )

    compat_kws = dict(include_groups=False) if PD_GE_2_2 else {}

    df.to_csv("debug_busmap.csv")
    grouped = df.groupby(["region", "sub_network"], group_keys=False)
    df.to_csv("debug_busmap_grouped.csv")

    # return grouped.apply(busmap_for_group, **compat_kws).squeeze().rename("busmap")

    busmap = (
        grouped
        .apply(busmap_for_group, **compat_kws)
        .squeeze()
        .rename("busmap")
    )

    # Ensure all buses are present
    busmap = busmap.reindex(n.buses.index)

    if busmap.isna().any():
        missing = busmap.index[busmap.isna()]
        sample = missing[:20].tolist()
        raise ValueError(
            f"busmap is incomplete: {len(missing)} buses were not assigned to any cluster. "
            f"Sample: {sample}. "
            "This usually means region/sub_network had NaN or there is no "
            "n_clusters entry for a group."
        )

    return busmap.astype(str)

def busmap_for_n_clusters(
    n: pypsa.Network,
    n_clusters_c: pd.Series,
    cluster_weights: pd.Series,
    algorithm: str = "kmeans",
    features: pd.DataFrame | None = None,
    **algorithm_kwds,
) -> pd.Series:
    if algorithm == "hac" and features is None:
        raise ValueError("For HAC clustering, features must be provided.")

    if algorithm == "kmeans":
        algorithm_kwds.setdefault("n_init", 1000)
        algorithm_kwds.setdefault("max_iter", 30000)
        algorithm_kwds.setdefault("tol", 1e-6)
        algorithm_kwds.setdefault("random_state", 0)

    def busmap_for_country(x):
        prefix = x.name[0] + x.name[1] + " "
        logger.debug(
            f"Determining busmap for country {prefix[:-1]} "
            f"from {len(x)} buses to {n_clusters_c[x.name]}."
        )
        if len(x) == 1:
            return pd.Series(prefix + "0", index=x.index)
        weight = weighting_for_country(x, cluster_weights)

        if algorithm == "kmeans":
            return prefix + busmap_by_kmeans(
                n, weight, n_clusters_c[x.name], buses_i=x.index, **algorithm_kwds
            )
        elif algorithm == "hac":
            return prefix + busmap_by_hac(
                n,
                n_clusters_c[x.name],
                buses_i=x.index,
                feature=features.reindex(x.index, fill_value=0.0),
            )
        elif algorithm == "modularity":
            return prefix + busmap_by_greedy_modularity(
                n, n_clusters_c[x.name], buses_i=x.index
            )
        else:
            raise ValueError(
                f"`algorithm` must be one of 'kmeans' or 'hac' or 'modularity'. Is {algorithm}."
            )

    compat_kws = dict(include_groups=False) if PD_GE_2_2 else {}

    return (
        n.buses.groupby(["country", "sub_network"], group_keys=False)
        .apply(busmap_for_country, **compat_kws)
        .squeeze()
        .rename("busmap")
    )


def clustering_for_n_clusters(
    n: pypsa.Network,
    busmap: pd.Series,
    aggregation_strategies: dict | None = None,
) -> pypsa.clustering.spatial.Clustering:
    if aggregation_strategies is None:
        aggregation_strategies = dict()

    line_strategies = aggregation_strategies.get("lines", dict())

    bus_strategies = aggregation_strategies.get("buses", dict())
    bus_strategies.setdefault("substation_lv", lambda x: bool(x.sum()))
    bus_strategies.setdefault("substation_off", lambda x: bool(x.sum()))

    # TODO Quick Fix for osm-prebuilt-version 0.6
    for way_i in ["way/140248154", "way/975637991"]:
        if way_i in n.buses.index:
            n.buses.loc[way_i, "carrier"] = "AC"

    clustering = get_clustering_from_busmap(
        n,
        busmap,
        bus_strategies=bus_strategies,
        line_strategies=line_strategies,
        custom_line_groupers=["build_year"],
    )

    return clustering


def cluster_regions(
    busmaps: tuple | list, regions: gpd.GeoDataFrame, with_country: bool = False
) -> gpd.GeoDataFrame:
    """
    Cluster regions based on busmaps.

    Parameters
    ----------
        - busmaps (list) : A list of busmaps used for clustering.
        - regions (gpd.GeoDataFrame) : The regions to cluster.
        - with_country (bool) : Whether to keep country column.

    Returns
    -------
    gpd.GeoDataFrame: The clustered regions.
    """
    busmap = reduce(lambda x, y: x.map(y), busmaps[1:], busmaps[0])
    columns = ["name", "country", "geometry"] if with_country else ["name", "geometry"]
    regions = regions.reindex(columns=columns).set_index("name")
    regions_c = regions.dissolve(busmap)
    regions_c.index.name = "name"
    return regions_c.reset_index()


def busmap_for_admin_regions(
    n: pypsa.Network,
    admin_shapes: str,
    countries: list,
    administrative: dict,
    area_threshold = 0,
) -> pd.Series:
    """
    Create a busmap based on administrative regions using the NUTS3 shapefile.

    Parameters
    ----------
        - n (pypsa.Network) : The network to cluster.
        - admin_shapes (str) : The path to the administrative regions.
        - params (dict) : The parameters for clustering.

    Returns
    -------
        busmap (pd.Series): Busmap mapping each bus to an administrative region.
    """
    admin_regions = gpd.read_file(admin_shapes)

    admin_regions = admin_regions.set_index("admin")

    if area_threshold > 0:
        admin_regions = aggregate_small_admin_subregions(admin_regions, area_threshold, country = "GB")

    # log info about how many regions are in GB
    n_GB = len(admin_regions.query('country == "GB"'))
    logger.info(f"Number of administrative regions in GB before merging small regions: {n_GB}")

    admin_regions = admin_regions.reset_index().rename(columns={"index": "admin"})
    
    # overwrite admin_shapes file
    admin_regions.to_file(admin_shapes, driver="GeoJSON")

    level = administrative.get("level", 0)
    logger.info(f"Clustering at administrative level {level}.")

    # check if BA, MD, UA, or XK are in the network
    adm1_countries = ["BA", "MD", "UA", "XK"]
    buses = n.buses[["x", "y", "country"]].copy()

    # Find the intersection of adm1_countries and n.buses.country
    adm1_countries = list(set(adm1_countries).intersection(buses["country"].unique()))

    if adm1_countries:
        logger.info(
            f"Note that the following countries can only be clustered at a maximum administration level of 1: {adm1_countries}."
        )

    country_level = {
        k: v for k, v in administrative.items() if (k != "level") and (k in countries)
    }
    if country_level:
        country_level_list = "\n".join(
            [f"- {k}: level {v}" for k, v in country_level.items()]
        )
        logger.info(
            f"Setting individual administrative levels for:\n{country_level_list}"
        )

    buses["geometry"] = gpd.points_from_xy(buses["x"], buses["y"])
    buses = gpd.GeoDataFrame(buses, geometry="geometry", crs="EPSG:4326")
    buses["busmap"] = ""

    # Map based for each country
    logger.info("Mapping buses to administrative regions.")
    for country in countries:
        buses_subset = buses.loc[buses["country"] == country]

        buses.loc[buses_subset.index, "busmap"] = gpd.sjoin_nearest(
            buses_subset.to_crs(epsg=3857),
            admin_regions.loc[admin_regions["country"] == country].to_crs(epsg=3857),
            how="left",
        )["admin"]

    return buses["busmap"]


def keep_largest_polygon(geometry: MultiPolygon) -> Polygon:
    """
    Checks for each MultiPolygon if it contains multiple Polygons and returns the one with the largest area.

    Parameters
    ----------
        geometry (MultiPolygon) : The MultiPolygon to check.

    Returns
    -------
        geometry (Polygon) : The Polygon with the largest area.
    """
    if isinstance(geometry, MultiPolygon):
        # Find the polygon with the largest area in the MultiPolygon
        largest_polygon = max(geometry.geoms, key=lambda poly: poly.area)

        return largest_polygon
    else:
        # If it's a Polygon, return it as is
        return geometry


def update_bus_coordinates(
    n: pypsa.Network,
    busmap: pd.Series,
    admin_shapes: str,
    geo_crs: str = GEO_CRS,
    distance_crs: str = DISTANCE_CRS,
    tol: float = BUS_TOL,
) -> None:
    """
    Updates the x, y coordinates of the buses in the original network based on the busmap and the administrative regions.
    Using the Pole of Inaccessibility (PoI) to determine internal points of the administrative regions.

    Parameters
    ----------
        - n (pypsa.Network) : The original network.
        - busmap (pd.Series) : The busmap mapping each bus to an administrative region.
        - admin_shapes (str) : The path to the administrative regions.
        - geo_crs (str) : The geographic coordinate reference system.
        - distance_crs (str) : The distance coordinate reference system.
        - tol (float) : The tolerance in meters for the PoI calculation.

    Returns
    -------
        None
    """
    logger.info("Updating x, y coordinates of buses based on administrative regions.")
    admin_regions = gpd.read_file(admin_shapes).set_index("admin")
    admin_regions["geometry"] = (
        admin_regions["geometry"]
        .to_crs(distance_crs)
        .apply(keep_largest_polygon)
        .to_crs(geo_crs)
    )
    admin_regions["poi"] = (
        admin_regions["geometry"]
        .to_crs(distance_crs)
        .apply(lambda polygon: polylabel(polygon, tolerance=tol / 2))
        .to_crs(geo_crs)
    )
    admin_regions["x"] = admin_regions["poi"].x
    admin_regions["y"] = admin_regions["poi"].y

    busmap_df = pd.DataFrame(busmap)
    busmap_df = pd.merge(
        busmap_df,
        admin_regions[["x", "y"]],
        left_on="busmap",
        right_index=True,
        how="left",
    )

    # Update x, y coordinates of original network
    buses = n.buses.copy()
    update_buses = buses.loc[busmap.index]
    update_buses["x"] = busmap_df["x"]
    update_buses["y"] = busmap_df["y"]

    buses.loc[update_buses.index, :] = update_buses

    n.buses = buses

def temporarily_split_countries_into_subcountries(
    n: pypsa.Network,
    split_cfg: dict,
) -> pd.Series:
    """
    Temporarily overwrite n.buses.country for selected countries
    using subregion polygons defined in split_cfg.

    Returns a copy of the original country Series for restoration.
    """
    original_country = n.buses.country.copy()

    for country, cfg in split_cfg.items():

        shapes = gpd.read_file(cfg["geojson"])
        if shapes.crs is None:
            shapes = shapes.set_crs(GEO_CRS)

        id_field = cfg.get("id_field", "name")

        if id_field not in shapes.columns:
            raise ValueError(
                f"{country} geojson missing id_field='{id_field}'. "
                f"Columns: {list(shapes.columns)}"
            )

        shapes[id_field] = shapes[id_field].astype(str).str.strip()

        if shapes[id_field].isna().any() or (shapes[id_field] == "").any():
            raise ValueError(
                f"{country} geojson has missing/blank '{id_field}' values."
            )

        # Select buses in this country
        buses_c = n.buses[n.buses.country == country][["x", "y", "country"]].copy()

        if buses_c.empty:
            continue

        # Assign each bus to subregion
        subregion_for_bus = busmap_from_shapes(
            n,
            shapes,
            buses=buses_c,
            cluster_names=id_field,
            per_country=False,
        )

        if subregion_for_bus.isna().any():
            missing = subregion_for_bus.index[subregion_for_bus.isna()].tolist()[:10]
            raise ValueError(
                f"Some buses in {country} were not assigned to any subregion. "
                f"Sample: {missing}"
            )

        # Overwrite country column with subregion IDs
        n.buses.loc[buses_c.index, "country"] = subregion_for_bus.astype(str).values

if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("cluster_network", clusters=60)
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    params = snakemake.params
    mode = params.mode
    administrative = params.administrative
    countries = params.countries
    n_clusters = int(snakemake.wildcards.clusters)
    solver_name = snakemake.config["solving"]["solver"]["name"]

    algorithm = params.cluster_network["algorithm"]
    features = None

    n = pypsa.Network(snakemake.input.network)
    buses_prev, lines_prev, links_prev = len(n.buses), len(n.lines), len(n.links)

    load = (
        xr.open_dataarray(snakemake.input.load)
        .mean(dim="time")
        .to_pandas()
        .reindex(n.buses.index, fill_value=0.0)
    )

    if snakemake.wildcards.clusters == "all":
        # Fast-path if no clustering is necessary
        busmap = n.buses.index.to_series()
        linemap = n.lines.index.to_series()
        clustering = pypsa.clustering.spatial.Clustering(n, busmap, linemap)
    else:
        Nyears = n.snapshot_weightings.objective.sum() / 8760

        if mode == "administrative_mixed":
            administrative_df = pd.Series(administrative)
            admin_countries = list(administrative_df.index[administrative_df.index.isin(countries)])

            # print admin_countries
            logger.info(f"Administrative countries: {admin_countries}")

            busmap_a = busmap_for_admin_regions(
                                                n,
                                                snakemake.input.admin_shapes,
                                                admin_countries,
                                                administrative,
                                                area_threshold = params.area_threshold,
                                            )

            # print busmap_a
            logger.info(f"Busmap for administrative regions: {busmap_a}")

            n_clusters = int(n_clusters) - busmap_a[busmap_a != ""].unique().shape[0] # withdraw administrative regions

            logger.info(f"Number of clusters after withdrawing administrative regions: {n_clusters}")

            buses = n.buses.copy()
            lines = n.lines.copy()
            links = n.links.copy()
            load_reduced = load.copy()
            admin_buses_lst = []
            for admin_country in admin_countries:
                admin_buses = buses.query("country == @admin_country").index
                buses.drop(index=admin_buses, inplace=True)

                admin_lines_0 = lines[lines.bus0.isin(admin_buses)].index
                admin_lines_1 = lines[lines.bus1.isin(admin_buses)].index
                lines.drop(index=admin_lines_0.union(admin_lines_1), inplace=True)

                admin_links_0 = links[links.bus0.isin(admin_buses)].index
                admin_links_1 = links[links.bus1.isin(admin_buses)].index
                links.drop(index=admin_links_0.union(admin_links_1), inplace=True)

                load_reduced = load.drop(admin_buses)

                admin_buses_lst += list(admin_buses)

            n_temp = n.copy()
            n_temp.buses = buses
            n_temp.lines = lines
            n_temp.links = links

            if algorithm == "hac":
                features = get_feature_data_for_hac(snakemake.input.hac_features)
                fix_country_assignment_for_hac(n)

            n_temp.determine_network_topology()

            n_clusters_c = distribute_n_clusters_to_countries(
                n_temp,
                n_clusters,
                load_reduced,
                focus_weights=params.focus_weights,
                solver_name=solver_name,
            )
            busmap_c = busmap_for_n_clusters(
                n_temp,
                n_clusters_c,
                cluster_weights=load_reduced,
                algorithm=algorithm,
                features=features,
            )
            logger.info(f"Busmap for the rest: {busmap_c}")

            busmap = pd.concat([busmap_c, busmap_a.loc[pd.Index(admin_buses_lst)]])
            logger.info(f"Combined busmap: {busmap}")

            # Update x, y coordinates, ensuring that bus locations are inside the administrative region
            update_bus_coordinates(
                n,
                busmap_a[busmap_a != ""],
                snakemake.input.admin_shapes,
            )

            logger.info(f"Updated x-coords: {n.buses.x}")
            logger.info(f"Updated y-coords: {n.buses.y}")

        elif mode == "administrative":
            busmap = busmap_for_admin_regions(
                n,
                snakemake.input.admin_shapes,
                countries,
                administrative,
            )
            # Update x, y coordinates, ensuring that bus locations are inside the administrative region
            update_bus_coordinates(
                n,
                busmap,
                snakemake.input.admin_shapes,
            )
        elif mode == "custom_busshapes":
            n.determine_network_topology()
            custom_shapes = gpd.read_file(snakemake.input.custom_busshapes)
            custom_busmap = busmap_from_shapes(
                n,
                custom_shapes,
            )
            logger.info(
                f"Imported custom shapes from {snakemake.input.custom_busshapes}"
            )

            busmap = custom_busmap
        elif mode == "custom_busmap":
            custom_busmap = pd.read_csv(
                snakemake.input.custom_busmap, index_col=0
            ).squeeze()
            custom_busmap.index = custom_busmap.index.astype(str)
            logger.info(f"Imported custom busmap from {snakemake.input.custom_busmap}")
            busmap = custom_busmap
        else:
            if algorithm == "hac":
                features = get_feature_data_for_hac(snakemake.input.hac_features)
                fix_country_assignment_for_hac(n)

            cluster_subregions = params.cluster_network.get("subregions", {})
            if cluster_subregions:
                temporarily_split_countries_into_subcountries(n, cluster_subregions)

            n.determine_network_topology()

            n_clusters_c = distribute_n_clusters_to_countries(
                n,
                n_clusters,
                load,
                focus_weights=params.focus_weights,
                solver_name=solver_name,
            )

            busmap = busmap_for_n_clusters(
                n,
                n_clusters_c,
                cluster_weights=load,
                algorithm=algorithm,
                features=features,
            )

        clustering = clustering_for_n_clusters(
            n,
            busmap,
            aggregation_strategies=params.aggregation_strategies,
        )

    nc = clustering.n

    if snakemake.params.copperplate_regions:
        copperplate_buses(nc, snakemake.params.copperplate_regions)

    for attr in ["linemap"]:
        getattr(clustering, attr).to_csv(snakemake.output[attr])

    nc.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))

    busmap_clustering = clustering.busmap.copy()
    # group clusters in country
    if params.group_clusters:
        group_regions = params.group_clusters
        for gr in group_regions:
            country_bus, nc = group_clusters(nc, gr)
            busmap_clustering = correct_busmap(busmap_clustering, country_bus)
        
    busmap_clustering.index.name = "Bus"
    busmap_clustering.to_csv(snakemake.output.busmap, index=True)

    # nc.shapes = n.shapes.copy()
    for which in ["regions_onshore", "regions_offshore"]:
        regions = gpd.read_file(snakemake.input[which])
        clustered_regions = cluster_regions((busmap_clustering,), regions)
        clustered_regions.to_file(snakemake.output[which])
        # append_bus_shapes(nc, clustered_regions, type=which.split("_")[1])

    nc.buses.country = nc.buses.country.str[0:2]
    nc.export_to_netcdf(snakemake.output.network)

    logger.info(
        f"Clustered network:\n"
        f"Buses: {buses_prev} to {len(nc.buses)}\n"
        f"Lines: {lines_prev} to {len(nc.lines)}\n"
        f"Links: {links_prev} to {len(nc.links)}"
    )
