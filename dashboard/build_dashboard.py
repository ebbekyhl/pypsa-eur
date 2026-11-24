#!/usr/bin/env python
# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Build one interactive, self-contained HTML dashboard from solved PyPSA networks.

Usage
-----
    python dashboard/build_dashboard.py results/Run-0511-cbam-168h
    python dashboard/build_dashboard.py \
        "Base=results/Run-A/all/networks" "CBAM=results/Run-B/all/networks" \
        -o results/dashboard_compare.html

Each argument is PATH or LABEL=PATH. PATH can be a run folder, a networks folder
or a glob. The HTML needs no server and no install: open it from disk or send it
on. Only the extracted numbers go into the file, not the networks.

How the numbers are built
-------------------------
Everything starts from one table: the net flow into every bus, per technology.
A link's injection into bus_i is -p_i on every port, so inputs and outputs use
the same expression and each bus balance closes. Views and checks read that table.

* Yield shift keeps its ethylene withdrawal (port 0), so it shows as negative
  ethylene and positive co-products, and supply plus disposal equals demand.
* Prices are weighted by the energy through each bus, so idle buses do not
  dilute them. The whiskers show the range across the active buses.
* "Objective" is what the solver minimised. "System cost" adds the annuities of
  sunk assets the optimiser never weighed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import logging
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly
import pypsa

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("dashboard")
logging.getLogger("pypsa").setLevel(logging.ERROR)

HERE = Path(__file__).resolve().parent

HVC = ["ethylene", "propylene", "butadiene", "aromatics", "benzene", "toluene", "xylene"]

FEEDSTOCKS = [
    "naphtha for industry", "ethane for industry", "propane for industry",
    "butane for industry", "methanol", "industry methanol", "solid biomass",
    "ethanol for industry", "thermochemical recycling feed", "pyrolysis feed",
    "gas", "H2", "AC",
]

# link carriers that belong to the petrochemical sector
PETRO = [
    "steam cracker", "methanol-to-olefin", "methanol-to-aromatic", "ethanol-to-x",
    "biomass-to-ethanol", "pyrolysis", "yield shift", "on purpose propylene",
    "on purpose butadiene", "waste-to-methanol", "waste to methanol",
    "petrochemical cc", "process emissions petrochemicals",
]
EOL = ("hvc to air", "waste chp", "municipal solid waste")

# used only when the network's own config has no {product}_import_co2
IMPORT_CO2 = {"ethylene": 0.0763, "propylene": 0.0786, "butadiene": 0.0807, "aromatics": 0.0887}

TRANSFERS = ["capex subsidy", "opex new build", "capex grant (cash)", "opex existing", "tariff revenue"]
YEARS = (2025, 2030, 2035, 2040, 2045, 2050)


# --- small helpers -----------------------------------------------------------


def sig(x, digits=4):
    """Round to significant digits and make the value JSON-safe."""
    if x is None:
        return None
    x = float(x)
    if not math.isfinite(x):
        return None
    return 0.0 if x == 0 else float(f"{x:.{digits}g}")


def sigs(values, digits=4):
    return [sig(v, digits) for v in values]


def blocks(values, size):
    """Average a series into blocks, so very long runs do not bloat the file."""
    values = np.asarray(values, dtype=float)
    if size <= 1:
        return values
    return np.array([values[i:i + size].mean() for i in range(0, len(values), size)])


def cfg_get(spec, year, default=0.0):
    if not isinstance(spec, dict):
        return float(spec) if spec is not None else default
    return float(spec.get(year, spec.get(str(year), default)) or default)


def is_petro(tech):
    t = str(tech).lower()
    if "gate" in t or any(e in t for e in EOL):
        return False
    return any(p in t for p in PETRO)


def source_of(tech):
    """Which origin a technology on an HVC bus counts towards; None if neither."""
    t = str(tech).lower()
    if t.startswith(("demand ", "slack ")) or "transport" in t or "disposal" in t:
        return None
    if "import" in t:
        return "Imports"
    if t in ("steam cracker", "yield shift"):
        return "Existing fleet"
    if t == "steam cracker extension":
        return "Extended fleet"
    return "New builds"


def flatten(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}{k}."))
    else:
        out[prefix[:-1]] = json.dumps(d, default=str)
    return out


# --- flows -------------------------------------------------------------------


def tech_labels(n):
    """Link carriers, with yield shift on new electric crackers kept apart."""
    tech = n.links.carrier.astype(str).copy()
    electric = (tech == "yield shift") & n.links.index.str.contains("Electric Cracker")
    tech[electric] = "yield shift (new electric)"
    return tech


def flow_frames(n):
    """Yield (flows into the bus [snapshots x assets], bus per asset, tech per asset)."""
    links_tech = tech_labels(n)
    for col in sorted((c for c in n.links.columns if re.fullmatch(r"bus\d+", c)), key=lambda c: int(c[3:])):
        p = n.links_t.get("p" + col[3:])
        if p is None or p.empty:
            continue
        bus = n.links[col].astype(str)
        idx = p.columns.intersection(bus.index[bus.isin(n.buses.index)])
        if len(idx):
            yield -p[idx], bus[idx], links_tech[idx]

    gen = n.generators
    if not gen.empty and not n.generators_t.p.empty:
        idx = n.generators_t.p.columns
        tech = gen.loc[idx, "carrier"].astype(str)
        # generators sitting directly on a product bus are the feasibility slack
        tech = tech.where(~tech.isin(HVC), "slack " + tech)
        yield n.generators_t.p[idx] * gen.loc[idx, "sign"], gen.loc[idx, "bus"], tech

    if not n.loads.empty and not n.loads_t.p.empty:
        idx = n.loads_t.p.columns
        yield -n.loads_t.p[idx], n.loads.loc[idx, "bus"], "demand " + n.loads.loc[idx, "carrier"].astype(str)

    for name, comp, dyn in (("stores", n.stores, n.stores_t), ("storage_units", n.storage_units, n.storage_units_t)):
        if not comp.empty and not dyn.p.empty:
            idx = dyn.p.columns
            yield dyn.p[idx], comp.loc[idx, "bus"], comp.loc[idx, "carrier"].astype(str)

    if not n.lines.empty and not n.lines_t.p0.empty:
        idx = n.lines_t.p0.columns
        label = pd.Series("AC line", index=idx)
        yield -n.lines_t.p0[idx], n.lines.loc[idx, "bus0"], label
        yield -n.lines_t.p1[idx], n.lines.loc[idx, "bus1"], label


def collect_flows(n, w):
    """Return annual balance, link port use, time series per carrier and bus throughput."""
    bus_carrier = n.buses.carrier.astype(str)
    annual, ports, series = [], [], []
    throughput = pd.Series(0.0, index=n.buses.index)

    for flow, bus, tech in flow_frames(n):
        energy = flow.mul(w, axis=0).sum()
        annual.append(pd.DataFrame({"bus": bus.values, "tech": tech.values, "value": energy.values}))

        gross = flow.abs().mul(w, axis=0).sum()
        throughput = throughput.add(gross.groupby(bus.values).sum(), fill_value=0.0)

        keys = [bus.map(bus_carrier).values, tech.values]
        series.append(flow.T.groupby(keys).sum())

        if flow.columns.isin(n.links.index).all():
            ports.append(pd.DataFrame({
                "tech": tech.values,
                "carrier": bus.map(bus_carrier).values,
                "used": (-energy).clip(lower=0).values,
                "made": energy.clip(lower=0).values,
            }))

    balance = pd.concat(annual).groupby(["bus", "tech"], sort=False)["value"].sum().reset_index()
    ports = pd.concat(ports).groupby(["tech", "carrier"], sort=False)[["used", "made"]].sum().reset_index()
    series = pd.concat(series).groupby(level=[0, 1]).sum()
    return balance, ports, series, throughput


def capacities(n):
    """Optimal capacity per technology and node."""
    nodes = set(n.buses.index[n.buses.carrier == "AC"])
    loc = n.buses.location.astype(str)
    tech = tech_labels(n)

    node = pd.Series(np.nan, index=n.links.index, dtype=object)
    for col in sorted((c for c in n.links.columns if re.fullmatch(r"bus\d+", c)), key=lambda c: int(c[3:])):
        here = n.links[col].astype(str).map(loc)
        here = here.where(here.isin(nodes))
        node = node.fillna(here)

    # yield shift p_nom is only a ceiling (1e6 on new crackers), so it is not capacity
    keep = ~tech.str.startswith("yield shift")
    frames = [pd.DataFrame({
        "tech": tech[keep], "node": node[keep].fillna("EU"),
        "value": n.links.loc[keep, "p_nom_opt"], "unit": "MW",
    })]
    gen = n.generators
    frames.append(pd.DataFrame({"tech": gen.carrier, "node": gen.bus.map(loc), "value": gen.p_nom_opt, "unit": "MW"}))
    su = n.storage_units
    frames.append(pd.DataFrame({"tech": su.carrier, "node": su.bus.map(loc), "value": su.p_nom_opt, "unit": "MW"}))
    st = n.stores
    frames.append(pd.DataFrame({"tech": st.carrier, "node": st.bus.map(loc), "value": st.e_nom_opt, "unit": "MWh"}))

    cap = pd.concat(frames)
    cap = cap[cap.value.abs() > 1e-3]
    cap["node"] = cap.node.where(cap.node.isin(nodes), "EU")
    return cap.groupby(["tech", "node", "unit"], sort=False)["value"].sum().reset_index()


def crackers(n, w):
    """One row per existing cracker or life extension."""
    idx = n.links.index[n.links.carrier.isin(["steam cracker", "steam cracker extension"])]
    if idx.empty:
        return []
    loc = n.buses.location.astype(str)
    carrier = n.buses.carrier.astype(str)

    outputs = pd.Series(0.0, index=idx)
    for col in [c for c in n.links.columns if re.fullmatch(r"bus[1-9]\d*", c)]:
        p = n.links_t.get("p" + col[3:])
        if p is None or p.empty:
            continue
        here = idx.intersection(p.columns)
        on_hvc = n.links.loc[here, col].astype(str).map(carrier).isin(HVC).reindex(here)
        here = here[on_hvc.values]
        outputs[here] += -(p[here].mul(w, axis=0).sum())

    feed = n.links_t.p0.reindex(columns=idx).fillna(0.0).mul(w, axis=0).sum()
    hours = float(w.sum())
    rows = []
    for link in idx:
        row = n.links.loc[link]
        plant = re.sub(r" steam cracker( extension)?(-\d+)?$", "", link)
        cap = float(row.p_nom_opt)
        rows.append({
            "plant": plant,
            "ext": row.carrier == "steam cracker extension",
            "node": str(loc.get(row.bus0, "")),
            "feed": str(carrier.get(row.bus0, "")).replace(" for industry", ""),
            "cap": sig(cap),
            "cf": sig(feed[link] / (cap * hours), 3) if cap > 0 else None,
            "out": sig(outputs[link] / 1e6),
            "build": int(row.build_year),
            "retire": int(row.build_year + row.lifetime) if math.isfinite(row.lifetime) else None,
        })
    return rows


def yield_shift_check(n, w):
    """Shifted ethylene against the limit from solve_network.add_yield_shift_limit."""
    shift = n.links.index[n.links.carrier == "yield shift"]
    if shift.empty:
        return {"status": "info", "detail": "No yield shift links in this network."}
    if "max_shift" not in n.links.columns or n.links.loc[shift, "max_shift"].isna().all():
        return {"status": "warn", "detail": "Yield shift links have no max_shift value: network predates the current code."}

    crack = n.links.index[n.links.carrier.isin(["steam cracker", "steam cracker extension", "electric steam cracker naphtha"])]
    used = allowed = 0.0
    worst = 0.0
    for link in shift:
        base = link.removesuffix(" yield shift")
        prefix = base if base.endswith("Electric Cracker Naphtha") else base + " steam cracker"
        eth_bus = n.links.at[link, "bus0"]
        production = pd.Series(0.0, index=n.snapshots)
        for c in crack[crack.str.startswith(prefix)]:
            for i in range(1, 9):
                if n.links.at[c, f"bus{i}"] == eth_bus:
                    production = production - n.links_t[f"p{i}"].get(c, 0.0)
                    break
        limit = n.links.at[link, "max_shift"] * production
        p0 = n.links_t.p0.get(link, pd.Series(0.0, index=n.snapshots))
        used += float(p0.mul(w).sum())
        allowed += float(limit.mul(w).sum())
        worst = max(worst, float((p0 - limit).max()))

    detail = f"{used / 1e6:.1f} TWh ethylene shifted of {allowed / 1e6:.1f} TWh allowed."
    if worst > 1:
        return {"status": "warn", "detail": detail + f" Limit exceeded by up to {worst:.0f} MW."}
    return {"status": "ok", "detail": detail}


# --- policy transfers (logic from hvc_results.py) ------------------------------


def hvc_out(n, links, w):
    """MWh of HVC injected per link, outputs only."""
    total = pd.Series(0.0, index=links)
    carrier = n.buses.carrier.astype(str)
    for col in [c for c in n.links.columns if re.fullmatch(r"bus[1-9]\d*", c)]:
        p = n.links_t.get("p" + col[3:])
        if p is None or p.empty:
            continue
        here = pd.Index(links).intersection(p.columns)
        here = here[n.links.loc[here, col].astype(str).map(carrier).isin(HVC).values]
        total[here] += -(p[here].mul(w, axis=0).sum())
    return total


def policy_transfers(n, cf, year, w):
    out = dict.fromkeys(TRANSFERS, 0.0)
    new = n.links.index[n.links.carrier.isin(cf.get("hvc_expansion_carriers", []) or [])]

    # each vintage keeps its build-year rate
    sched = cf.get("hvc_opex_subsidy")
    if len(new) and isinstance(sched, dict) and any(cfg_get(sched, y) for y in YEARS):
        by = n.links.loc[new, "build_year"].replace(0, year).fillna(year)
        out["opex new build"] = float((hvc_out(n, new, w) * by.map(lambda b: cfg_get(sched, int(b)))).sum())

    # the existing fleet is re-subsidised at the current rate every horizon
    rate = cfg_get(cf.get("hvc_existing_opex_subsidy"), year)
    if rate:
        old = n.links.index[n.links.carrier.isin(cf.get("hvc_existing_carriers", []) or []) & ~n.links.p_nom_extendable]
        out["opex existing"] = float((hvc_out(n, old, w) * rate).sum())

    # annual equivalent: the discount recurs because capital_cost is an annuity
    sched_c = cf.get("hvc_capex_subsidy")
    if len(new) and any(cfg_get(sched_c, y) for y in YEARS):
        by = n.links.loc[new, "build_year"].replace(0, year).fillna(year)
        share = by.map(lambda b: cfg_get(sched_c, int(b))).clip(0.0, 0.99)
        annual = n.links.loc[new, "p_nom_opt"] * n.links.loc[new, "capital_cost"]
        out["capex subsidy"] = float((annual * share / (1.0 - share)).sum())

    # cash view for the budget narrative, not part of the net transfer
    s_now = cfg_get(sched_c, year)
    if 0 < s_now < 1:
        ext = new[n.links.loc[new, "p_nom_extendable"]]
        if len(ext):
            life = n.links.loc[ext, "lifetime"].fillna(25).clip(lower=1)
            annuity = 0.07 / (1 - (1 + 0.07) ** -life)
            invest = n.links.loc[ext, "p_nom_opt"] * n.links.loc[ext, "capital_cost"] / annuity
            out["capex grant (cash)"] = float((invest * s_now / (1 - s_now)).sum())

    tariff = cfg_get(cf.get("hvc_import_tariff"), year)
    if tariff:
        imp = n.links.index[n.links.carrier.isin([f"{p} import" for p in HVC])]
        imp = imp.intersection(n.links_t.p0.columns)
        if len(imp):
            out["tariff revenue"] = -tariff * float(n.links_t.p0[imp].sum(axis=1).mul(w).sum())
    return out


# --- one network ---------------------------------------------------------------


def extract(path, year, max_snapshots):
    logger.info("reading %s", path)
    n = pypsa.Network(str(path))
    w = n.snapshot_weightings.objective
    meta = n.meta if isinstance(n.meta, dict) else {}
    cf = meta.get("industry", {}) or {}

    balance, ports, series, throughput = collect_flows(n, w)
    bc = n.buses.carrier.astype(str)
    balance["carrier"] = balance.bus.map(bc)
    by_carrier = balance.groupby(["carrier", "tech"])["value"].sum()

    def carrier_totals(carrier):
        return by_carrier.xs(carrier) if carrier in by_carrier.index.get_level_values(0) else pd.Series(dtype=float)

    # --- HVC supply ---
    hvc = balance[balance.carrier.isin(HVC)]
    per_tech = hvc.groupby("tech")["value"].sum()
    demand = -float(per_tech[per_tech.index.str.startswith("demand ")].sum())
    disposal = -float(per_tech[per_tech.index.str.contains("disposal")].sum())
    slack_all = balance[balance.tech.str.startswith("slack ")]
    origin = {"Existing fleet": 0.0, "Extended fleet": 0.0, "New builds": 0.0, "Imports": 0.0}
    for tech, value in per_tech.items():
        src = source_of(tech)
        if src:
            origin[src] += float(value)

    # --- waste fate: resolve the recycling gates into the routes behind them ---
    def sinks(carrier):
        s = carrier_totals(carrier)
        return -s[s < 0]

    waste = sinks("non-sequestered HVC")
    gates = [t for t in waste.index if "gate" in t]
    if gates:
        behind = sinks("thermochemical recycling feed")
        inner = [t for t in behind.index if "gate" in t]
        if inner:
            behind = pd.concat([behind.drop(inner), sinks("pyrolysis feed")])
        if behind.sum() > 0:
            waste = pd.concat([waste.drop(gates), behind]).groupby(level=0).sum()

    # --- petrochemical CO2 ---
    charged = free = captured = 0.0
    for row in ports.itertuples():
        if not is_petro(row.tech):
            continue
        net = row.made - row.used
        if row.carrier == "co2":
            charged += net
        elif row.carrier == "co2 unpriced":
            free += net
        elif row.carrier in ("co2 stored", "co2 sequestered"):
            captured += net

    cbam = cfg_get(cf.get("cbam_factor"), year)
    embedded = 0.0
    for product in HVC:
        value = float(per_tech.get(f"{product} import", 0.0))
        fallback = IMPORT_CO2.get(product, 0.09)
        embedded += value * cfg_get(cf.get(f"{product}_import_co2", fallback), year, fallback)
    imp_charged, imp_free = embedded * cbam, embedded * (1 - cbam)

    def store_end(carrier):
        idx = n.stores.index[n.stores.carrier == carrier].intersection(n.stores_t.e.columns)
        return float(n.stores_t.e[idx].iloc[-1].sum()) if len(idx) else 0.0

    # --- cost ---
    capex = n.statistics.capex(nice_names=False)
    opex = n.statistics.opex(nice_names=False)
    capex_c = capex.groupby(level=-1).sum()
    opex_c = opex.groupby(level=-1).sum()
    cost = capex_c.add(opex_c, fill_value=0.0)
    asset = float(cost[[is_petro(t) for t in cost.index]].sum())

    avg_price = n.buses_t.marginal_price.mul(w, axis=0).sum() / w.sum()
    delivered = 0.0
    for row in hvc[hvc.tech.str.startswith("demand ")].itertuples():
        delivered += -row.value * float(avg_price.get(row.bus, 0.0))

    # --- prices, weighted by throughput ---
    prices = {}
    series_price = {}
    mp = n.buses_t.marginal_price
    for carrier, buses in n.buses.groupby("carrier").groups.items():
        cols = mp.columns.intersection(buses)
        if cols.empty:
            continue
        flow = throughput.reindex(cols).fillna(0.0)
        active = cols[flow > 1]
        if active.empty:
            continue
        weight = flow[active]
        prices[carrier] = {
            "mean": sig((avg_price[active] * weight).sum() / weight.sum()),
            "min": sig(avg_price[active].min()),
            "max": sig(avg_price[active].max()),
        }
        series_price[carrier] = mp[active].mul(weight, axis=1).sum(axis=1) / weight.sum()

    co2_mu = 0.0
    if "CO2Limit" in n.global_constraints.index:
        co2_mu = -float(n.global_constraints.at["CO2Limit", "mu"])

    # --- fleet ---
    fleet = crackers(n, w)
    open_gw = sum(r["cap"] or 0 for r in fleet) / 1e3

    # --- transfers ---
    transfers = policy_transfers(n, cf, year, w)
    net_transfer = sum(v for k, v in transfers.items() if k != "capex grant (cash)")

    capex_total, opex_total = float(capex.sum()), float(opex.sum())
    kpi = {
        "Objective [bn €/a]": float(n.objective) / 1e9,
        "System cost [bn €/a]": (capex_total + opex_total) / 1e9,
        "Capex [bn €/a]": capex_total / 1e9,
        "Opex [bn €/a]": opex_total / 1e9,
        "System cost excl. transfers [bn €/a]": (capex_total + opex_total + net_transfer) / 1e9,
        "Net transfer [bn €/a]": net_transfer / 1e9,
        **{f"Transfer: {k} [bn €/a]": v / 1e9 for k, v in transfers.items()},
        "HVC demand [TWh]": demand / 1e6,
        "Existing fleet [TWh]": origin["Existing fleet"] / 1e6,
        "Extended fleet [TWh]": origin["Extended fleet"] / 1e6,
        "New builds [TWh]": origin["New builds"] / 1e6,
        "Imports [TWh]": origin["Imports"] / 1e6,
        "Import share [%]": 100 * origin["Imports"] / demand if demand else 0.0,
        "Surplus disposal [TWh]": disposal / 1e6,
        "Open cracker capacity [GW]": open_gw,
        "Domestic CO2 charged [Mt]": charged / 1e6,
        "Domestic CO2 not charged [Mt]": free / 1e6,
        "Imported CO2 charged [Mt]": imp_charged / 1e6,
        "Imported CO2 not charged [Mt]": imp_free / 1e6,
        "Petrochemical CO2 total [Mt]": (charged + free + imp_charged + imp_free) / 1e6,
        "Petrochemical CO2 captured [Mt]": captured / 1e6,
        "CO2 price [€/t]": co2_mu,
        "System CO2 emissions [Mt]": store_end("co2") / 1e6,
        "System CO2 sequestered [Mt]": store_end("co2 sequestered") / 1e6,
        "Petrochemical asset cost [bn €/a]": asset / 1e9,
        "Delivered HVC value [bn €/a]": delivered / 1e9,
    }

    # --- checks ---
    checks = []
    slack_twh = float(slack_all.value.clip(lower=0).sum()) / 1e6
    shed = n.generators.index[n.generators.index.str.contains("load shedding|slack", case=False)]
    shed_twh = float(n.generators_t.p.reindex(columns=shed).fillna(0).mul(w, axis=0).sum().sum()) / 1e6
    checks.append({"name": "Slack and load shedding", "status": "ok" if slack_twh + shed_twh < 0.01 else "fail",
                   "detail": f"{slack_twh + shed_twh:.3f} TWh served by slack."})

    gap = sum(origin.values()) - disposal - demand + float(slack_all[slack_all.carrier.isin(HVC)].value.sum())
    transport = float(per_tech[per_tech.index.str.contains("transport")].sum())
    gap += transport
    checks.append({"name": "HVC supply equals demand", "status": "ok" if abs(gap) < 1e-3 * max(demand, 1) else "warn",
                   "detail": f"Supply − disposal − demand = {gap / 1e6:.3f} TWh; net transport {transport / 1e6:.3f} TWh."})

    residual = balance.groupby("bus")["value"].sum()
    worst = residual.abs().idxmax()
    closes = abs(residual[worst]) < max(1e3, 1e-4 * throughput.get(worst, 0.0))
    checks.append({"name": "Every bus balance closes", "status": "ok" if closes else "warn",
                   "detail": f"Largest residual {residual[worst] / 1e6:.4f} TWh at {worst}."})

    shift = yield_shift_check(n, w)
    checks.append({"name": "Yield shift within limit", **shift})

    placeholder = [r["plant"] for r in fleet if r["build"] == 1979]
    if placeholder:
        checks.append({"name": "Placeholder start years", "status": "warn",
                       "detail": f"{len(placeholder)} crackers have build year 1979: {', '.join(placeholder[:4])}…"})

    gap_cost = (capex_total + opex_total) - float(n.objective)
    checks.append({"name": "Objective vs system cost", "status": "info",
                   "detail": f"System cost is {gap_cost / 1e9:.1f} bn €/a above the objective (sunk annuities and constants)."})

    # --- time series ---
    block = max(1, math.ceil(len(n.snapshots) / max_snapshots))
    ts_bal = {}
    for carrier, frame in series.groupby(level=0):
        frame = frame.droplevel(0)
        peak = frame.abs().to_numpy().max()
        if peak <= 0:
            continue
        kept = frame[frame.abs().max(axis=1) > 1e-3 * peak]
        ts_bal[carrier] = {tech: sigs(blocks(row.values, block), 3) for tech, row in kept.iterrows()}

    snaps = n.snapshots[::block]
    balance = balance[balance.value.abs() >= 1]
    cap = capacities(n)

    return {
        "n": n,
        "meta": meta,
        "balance": balance,
        "ports": ports,
        "cap": cap,
        "avg_price": avg_price,
        "out": {
            "kpi": {k: sig(v, 5) for k, v in kpi.items()},
            "origin": {k: sig(v / 1e6) for k, v in origin.items()},
            "prices": prices,
            "cost": {t: [sig(capex_c.get(t, 0.0) / 1e9), sig(opex_c.get(t, 0.0) / 1e9)] for t in cost.index if abs(cost[t]) > 1e3},
            "waste": {t: sig(v / 1e6) for t, v in waste.items() if v > 1},
            "fleet": fleet,
            "checks": checks,
            "ts": {
                "t": [str(s)[:13] for s in snaps],
                "bal": ts_bal,
                "price": {c: sigs(blocks(s.values, block), 3) for c, s in series_price.items()},
            },
        },
    }


# --- map outlines --------------------------------------------------------------


def load_regions(path):
    import geopandas as gpd

    regions = gpd.read_file(path).set_index("name")
    out = []
    for name, geom in regions.geometry.simplify(0.03).items():
        for poly in getattr(geom, "geoms", [geom]):
            xy = np.asarray(poly.exterior.coords)
            if len(xy) >= 4:
                out.append({"n": name, "x": np.round(xy[:, 0], 2).tolist(), "y": np.round(xy[:, 1], 2).tolist()})
    return out


def find_regions(path, meta):
    """Look for the onshore regions next to the run's results folder."""
    run = meta.get("run", {}) if isinstance(meta, dict) else {}
    wild = meta.get("wildcards", {}) if isinstance(meta, dict) else {}
    prefix, name, clusters = run.get("prefix", ""), run.get("name", ""), wild.get("clusters", "")
    for parent in Path(path).resolve().parents:
        candidate = parent / "resources" / str(prefix) / str(name) / f"regions_onshore_base_s_{clusters}.geojson"
        if candidate.exists():
            return candidate
    return None


# --- inputs --------------------------------------------------------------------


def year_of(path):
    match = re.search(r"_(20\d{2})$", path.stem) or re.search(r"(20\d{2})(?!.*20\d{2})", path.stem)
    return int(match.group(1)) if match else None


def find_files(spec):
    label, raw = spec.split("=", 1) if "=" in spec else (None, spec)
    path = Path(raw)
    if path.is_dir():
        files = sorted(path.glob("*.nc")) or sorted(path.glob("*/networks/*.nc")) or sorted(path.glob("networks/*.nc"))
    else:
        files = sorted(Path(f) for f in glob.glob(raw))
    if not files:
        raise SystemExit(f"No .nc files found at {raw}")
    if label is None:
        parts = [p for p in path.parts if p not in ("all", "networks", ".", "results")]
        label = parts[-1] if parts and path.is_dir() else files[0].parent.name
    return label, files


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", metavar="[LABEL=]PATH")
    ap.add_argument("-o", "--output", type=Path, default=None)
    ap.add_argument("--regions", type=Path, default=None, help="regions_onshore geojson for the map outlines")
    ap.add_argument("--max-snapshots", type=int, default=400, help="time series are averaged down to about this many steps")
    args = ap.parse_args()

    jobs = []
    for spec in args.runs:
        label, files = find_files(spec)
        for f in files:
            year = year_of(f)
            if year is None:
                logger.warning("skipping %s: no year in the filename", f.name)
                continue
            jobs.append((label, year, f))
    jobs.sort(key=lambda j: (j[0], j[1]))

    # several sector_opts in one folder: keep them apart
    seen = {}
    for label, year, f in jobs:
        seen.setdefault((label, year), []).append(f)
    multi = {k for k, v in seen.items() if len(v) > 1}
    if multi:
        jobs = [(f"{l} {f.stem.rsplit('_', 1)[0]}" if (l, y) in multi else l, y, f) for l, y, f in jobs]

    bus_id, tech_id = {}, {}
    bus_rows = []

    def bus_index(name, n):
        if name not in bus_id:
            bus_id[name] = len(bus_rows)
            row = n.buses.loc[name]
            bus_rows.append([name, str(row.carrier), str(row.location), str(row.get("unit", ""))])
        return bus_id[name]

    def tech_index(name):
        return tech_id.setdefault(name, len(tech_id))

    nets, configs, nodes = {}, {}, {}
    regions_path = args.regions
    for label, year, f in jobs:
        res = extract(f, year, args.max_snapshots)
        n, out = res["n"], res["out"]

        if regions_path is None:
            regions_path = find_regions(f, res["meta"])

        for name, row in n.buses[n.buses.carrier == "AC"].iterrows():
            nodes[name] = [sig(row.x, 6), sig(row.y, 6)]

        bal = res["balance"]
        out["bal"] = {
            "b": [bus_index(b, n) for b in bal.bus],
            "t": [tech_index(t) for t in bal.tech],
            "v": sigs(bal.value.values),
        }
        cap = res["cap"]
        out["cap"] = {
            "t": [tech_index(t) for t in cap.tech],
            "n": cap.node.tolist(),
            "u": cap.unit.tolist(),
            "v": sigs(cap.value.values),
        }
        ports = res["ports"]
        ports = ports[(ports["used"] > 1) | (ports["made"] > 1)]
        out["ports"] = [[r.tech, r.carrier, sig(r.used / 1e6), sig(r.made / 1e6)] for r in ports.itertuples()]
        prices = res["avg_price"]
        prices = prices[prices.abs() > 1e-3]
        known = prices.index.intersection(n.buses.index)
        out["pb"] = {"b": [bus_index(b, n) for b in known], "v": sigs(prices[known].values)}

        key = f"{label}|{year}"
        nets[key] = out
        configs[key] = flatten({k: v for k, v in res["meta"].items() if k not in ("plotting", "wildcards")})
        del res, n

    # config values that differ between runs, per year
    labels = sorted({j[0] for j in jobs})
    years = sorted({j[1] for j in jobs})
    all_keys = sorted(set().union(*(c.keys() for c in configs.values())))
    diff = {}
    if len(labels) > 1:
        for k in all_keys:
            if k.startswith(("run.", "version", "logging", "remote", "tutorial")):
                continue
            values = {lab: next((configs[f"{lab}|{y}"].get(k) for y in years if f"{lab}|{y}" in configs), None) for lab in labels}
            if len(set(values.values())) > 1:
                diff[k] = values
    industry = {k: v for k, v in next(iter(configs.values())).items() if k.startswith("industry.")}

    regions = []
    if regions_path and Path(regions_path).exists():
        logger.info("map outlines from %s", regions_path)
        regions = load_regions(regions_path)
    else:
        logger.warning("no regions geojson found: the map shows nodes only (use --regions)")

    data = {
        "generated": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "labels": labels,
        "years": years,
        "buses": bus_rows,
        "techs": list(tech_id),
        "nodes": nodes,
        "regions": regions,
        "nets": nets,
        "configDiff": diff,
        "industry": industry,
        "feedstocks": FEEDSTOCKS,
        "hvc": HVC,
        "idleThreshold": cfg_get(json.loads(next(iter(configs.values())).get("industry.cracker_idle_threshold", "0.4")), 0),
    }

    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    plotly_js = (Path(plotly.__file__).parent / "package_data" / "plotly.min.js").read_text(encoding="utf-8").replace("</script", "<\\/script")
    html = (HERE / "template.html").read_text(encoding="utf-8")
    html = html.replace("/*__PLOTLY__*/", plotly_js).replace("/*__DATA__*/", payload)

    output = args.output
    if output is None:
        slug = re.sub(r"[^A-Za-z0-9_-]+", "-", "_".join(labels))[:60]
        output = (Path("results") if Path("results").is_dir() else Path(".")) / f"dashboard_{slug}.html"
    output.write_text(html, encoding="utf-8")
    logger.info("wrote %s (%.1f MB, %d networks)", output, output.stat().st_size / 1e6, len(nets))


if __name__ == "__main__":
    main()
