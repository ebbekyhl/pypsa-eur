# SPDX-FileCopyrightText: : 2023- The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT

import logging

import pandas as pd
from xarray import DataArray

logger = logging.getLogger(__name__)


def add_expansion_rate_limit(n, snapshots, snakemake):
    limits = snakemake.config["industry"].get("max_new_capacity_MW", {})
    if not limits:
        return

    year = int(snakemake.wildcards.planning_horizons)
    var = n.model["Link-p_nom"]
    ext = pd.Index(var.indexes[var.dims[0]])

    for carrier, spec in limits.items():
        cap = spec.get(year) if isinstance(spec, dict) else spec
        if cap is None:
            continue
        idx = n.links.index[(n.links.carrier == carrier) & n.links.p_nom_extendable]
        idx = idx.intersection(ext)
        if idx.empty:
            logger.info("Expansion limit for %s in %s: no extendable links", carrier, year)
            continue
        n.model.add_constraints(
            var.loc[idx].sum() <= float(cap),
            name=f"expansion_rate_{carrier.replace(' ', '_').replace('-', '_')}",
        )
        logger.info("Expansion limit %s %s: %.0f MW over %d links", carrier, year, float(cap), len(idx))

def add_import_share_limit(n, snapshots, snakemake):
    shares = snakemake.config["industry"].get("max_import_share", {})
    if not shares:
        return

    year = int(snakemake.wildcards.planning_horizons)
    w = n.snapshot_weightings.generators
    wd = DataArray(w.values, coords={"snapshot": n.snapshots}, dims=["snapshot"])
    var = n.model["Link-p"]
    all_links = pd.Index(var.indexes[var.dims[-1]])

    for product, spec in shares.items():
        share = spec.get(year) if isinstance(spec, dict) else spec
        if share is None:
            continue

        loads = n.loads.index[n.loads.carrier == product]
        if loads.empty:
            logger.info("Import share %s %s: no loads found", product, year)
            continue
        varying = [c for c in loads if c in n.loads_t.p_set.columns]
        if varying:
            demand = float((n.loads_t.p_set[varying].sum(axis=1) * w).sum())
        else:
            demand = float(n.loads.loc[loads, "p_set"].sum() * w.sum())

        imp = n.links.index[n.links.carrier == f"{product} import"].intersection(all_links)
        if imp.empty or demand <= 0:
            logger.info("Import share %s %s: no import links or zero demand", product, year)
            continue

        n.model.add_constraints(
            (var.loc[:, imp] * wd).sum() <= share * demand,
            name=f"import_share_{product.replace(' ', '_')}",
        )
        logger.info(
            "Import share limit %s %s: %.0f%% of %.1f TWh over %d links",
            product, year, share * 100, demand / 1e6, len(imp),
        )

def add_fossil_feed_limit(n, snapshots, snakemake):
    """
    Cap fossil feedstock for HVC at (1 - x) of what this year's HVC demand
    would need at today's yield.

    Counted as fossil:
      - ethane, propane and butane import generators (feedstock, 1:1)
      - HVC import generators (ethylene, propylene, butadiene, aromatics),
        converted to the feedstock they would have needed at today's yield,
        so imports cannot be used to get round the cap
      - naphtha cracked from the oil bus, as far as it exceeds the synthetic
        oil supply (book and claim: petrochemicals may claim Fischer-Tropsch,
        biomass-to-liquid and import oil first)

    Not counted: recycled naphtha from pyrolysis, which enters the naphtha bus
    directly (mass balance, recycled naphtha going to the crackers first), and
    methanol, ethanol and plastic waste routes. Physical emissions are not
    affected by this allocation.

    The reference is this horizon's HVC demand, which is primary demand only:
    exogenous recycling has already been removed through HVC_primary_fraction.
    Today's yield comes from the config.
    """
    cf = snakemake.config["industry"]
    spec = cf.get("hvc_fossil_feed_reduction", {})
    if not spec:
        return

    year = int(snakemake.wildcards.planning_horizons)
    x = spec.get(year) if isinstance(spec, dict) else spec
    if x is None:
        return

    products = {
        "ethylene": ("ethylene_production_today", "MWh_Ethylene_per_tEthylene"),
        "propylene": ("propylene_production_today", "MWh_Propylene_per_tPropylene"),
        "aromatics": ("aromatics_production_today", "MWh_Aromatics_per_tAromatics"),
        "butadiene": ("butadiene_production_today", "MWh_Butadiene_per_tButadiene"),
    }

    # --- reference: feedstock this year's demand would need at today's yield ---
    today_t = sum(float(cf[t]) for t, _ in products.values()) * 1e6
    feed_today = (
        today_t
        * float(cf["t_naphtha_per_t_steam_cracker_product"])
        * float(cf["MWh_Naphtha_per_tNaphtha"])
    )
    hvc_today = sum(float(cf[t]) * 1e6 * float(cf[e]) for t, e in products.values())
    yield_today = hvc_today / feed_today  # ~0.727 MWh HVC per MWh feed

    w = n.snapshot_weightings.generators
    loads = n.loads.index[n.loads.carrier.isin(list(products))]
    varying = [c for c in loads if c in n.loads_t.p_set.columns]
    if varying:
        demand = float((n.loads_t.p_set[varying].sum(axis=1) * w).sum())
    else:
        demand = float(n.loads.loc[loads, "p_set"].sum() * w.sum())
    if demand <= 0:
        logger.info("Fossil HVC feed %s: no HVC demand, no cap", year)
        return

    cap = (1 - x) * demand / yield_today

    wd = DataArray(w.values, coords={"snapshot": n.snapshots}, dims=["snapshot"])
    gp, lp = n.model["Generator-p"], n.model["Link-p"]
    gen_dim, link_dim = gp.dims[-1], lp.dims[-1]
    gen_idx = pd.Index(gp.indexes[gen_dim])
    link_idx = pd.Index(lp.indexes[link_dim])

    def link_flow(idx, port):
        """Annual MWh leaving the links at `port`, summed."""
        attr = "efficiency" if port == 1 else f"efficiency{port}"
        eff = DataArray(n.links.loc[idx, attr].values, coords={link_dim: idx}, dims=[link_dim])
        return (lp.loc[:, idx] * eff * wd).sum()

    # --- 1. generators: gas liquids as feedstock, HVC imports as feedstock equivalent ---
    weights = {
        "EU ethane import": 1.0,
        "EU propane import": 1.0,
        "EU butane import": 1.0,
        **{f"EU {p} import": 1.0 / yield_today for p in products},
    }
    gens = pd.Index(list(weights)).intersection(gen_idx)
    coef = DataArray([weights[g] for g in gens], coords={gen_dim: gens}, dims=[gen_dim])
    fossil = (gp.loc[:, gens] * coef * wd).sum()

    # --- 2. naphtha cracked from the oil bus (net of naphtha used elsewhere) ---
    oil_naphtha = n.links.index[n.links.carrier == "naphtha for industry"].intersection(link_idx)
    residual = n.links.index[n.links.carrier == "residual naphtha"].intersection(link_idx)

    cracked = 0
    if len(oil_naphtha):
        cracked = link_flow(oil_naphtha, 1)
    if len(residual):
        cracked = cracked - (lp.loc[:, residual] * wd).sum()

    # --- 3. synthetic oil fed into the oil bus, which petrochemicals may claim ---
    nonfossil = cf.get(
        "oil_nonfossil_carriers",
        ["Fischer-Tropsch", "biomass to liquid", "biomass to liquid CC", "import oil"],
    )
    oil_buses = n.buses.index[n.buses.carrier == "oil"]
    synthetic = 0
    for i in range(1, 10):
        col = f"bus{i}"
        if col not in n.links.columns:
            continue
        idx = n.links.index[
            n.links.carrier.isin(nonfossil) & n.links[col].isin(oil_buses)
        ].intersection(link_idx)
        if len(idx):
            synthetic = synthetic + link_flow(idx, i)
    sgen = n.generators.index[
        n.generators.carrier.isin(nonfossil) & n.generators.bus.isin(oil_buses)
    ].intersection(gen_idx)
    if len(sgen):
        synthetic = synthetic + (gp.loc[:, sgen] * wd).sum()

    # --- 4. book and claim: only naphtha beyond the synthetic supply is fossil ---
    if len(oil_naphtha):
        fossil_naphtha = n.model.add_variables(lower=0, name="hvc_fossil_naphtha")
        n.model.add_constraints(
            fossil_naphtha >= cracked - synthetic, name="hvc_fossil_naphtha_def"
        )
        fossil = fossil + fossil_naphtha

    n.model.add_constraints(fossil <= cap, name="hvc_fossil_feed_cap")
    logger.info(
        "Fossil HVC feed %s: at most %.1f TWh (%.0f%% below the %.1f TWh that %.1f TWh "
        "of HVC demand would need at a yield of %.3f); %d generators, %d oil-naphtha links",
        year, cap / 1e6, x * 100, demand / yield_today / 1e6,
        demand / 1e6, yield_today, len(gens), len(oil_naphtha),
    )

def custom_extra_functionality(n, snapshots, snakemake):
    """
    Add custom extra functionality constraints.
    """
    add_expansion_rate_limit(n, snapshots, snakemake)
    add_import_share_limit(n, snapshots, snakemake)
    add_fossil_feed_limit(n, snapshots, snakemake)