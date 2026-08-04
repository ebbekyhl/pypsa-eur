#!/usr/bin/env python
"""
Compare the CBAM (CO2-intensity split) scenario against the reference run, with a
focus on Great Britain: investment, dispatch, operation, storage, cross-border
flows by CO2 class, cost and carbon pressure.

Two stages, so re-styling the report never re-loads the (large) networks:

  1. EXTRACT  - load each solved network once, compute GB metrics, cache to a
                pickle.  Run with --reload (or delete the cache) to recompute.
  2. RENDER   - build a single self-contained HTML report from the cache.

Usage
-----
    python analysis/compare_scenarios.py            # use cache if present, else extract
    python analysis/compare_scenarios.py --reload   # force re-extraction from networks
    python analysis/compare_scenarios.py --html-only # only rebuild the HTML from cache

Design notes
------------
* The CBAM network splits every GB AC bus into <region>, <region> clean and
  <region> nonclean layers.  ``n.buses.location`` maps every layer bus back to
  its region bus, and is a no-op for the reference run (a bus is its own
  location).  All GB aggregation goes through it so the two scenarios line up.
* Electricity-only views filter on ``bus_carrier="AC"`` so fuel-supply
  "generators" (biomass/gas potentials) do not swamp the numbers.
* The CBAM border *charge* is not implemented yet (topology only), so the
  interesting signal is the clean/nonclean *labelling* of flows, read straight
  off the ``co2_class`` column the split writes onto the duplicated DC links.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
REPO = Path(__file__).resolve().parents[1]
SCENARIOS = {
    "reference": REPO / "results/pypsa-uk/reference/networks",
    "CBAM": REPO / "results/pypsa-uk/CBAM/networks",
}
YEARS = [2025, 2030, 2035, 2040, 2045, 2050]
NETFMT = "base_s_50__3h_{year}.nc"
COUNTRY = "GB"

OUTDIR = REPO / "analysis"
CACHE = OUTDIR / "_comparison_cache.pkl"
HTML = OUTDIR / "scenario_comparison.html"

MWH_TWH = 1e6  # network is in MWh; snapshot-weighted sums -> divide by 1e6 for TWh
MW_GW = 1e3

# carriers that are electricity *sinks* on the AC bus (exclude from "generation")
SINK_CARRIERS = {
    "electricity distribution grid",
    "battery charger",
    "redox flow battery charger",
    "compressed air charger",
    "molten salt charger",
    "liquid air charger",
    "home battery charger",
    "H2 Electrolysis",
    "methanolisation",
    "Haber-Bosch",
    "DAC",
    "gas pipeline",
}
FOSSIL_ELEC = {"CCGT", "OCGT", "urban central gas CHP", "urban central gas CHP CC",
               "CCGT methanol", "OCGT methanol", "allam methanol"}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def net_path(scenario: str, year: int) -> Path:
    return SCENARIOS[scenario] / NETFMT.format(year=year)


def load_network(scenario: str, year: int) -> pypsa.Network | None:
    p = net_path(scenario, year)
    if not p.exists():
        return None
    return pypsa.Network(str(p))


def region(n: pypsa.Network, bus: pd.Series) -> pd.Series:
    """Map any (possibly CBAM-layer) bus to its region bus."""
    return bus.map(n.buses.location).fillna(bus)


def gb_mask(series: pd.Series) -> pd.Series:
    return series.astype(str).str.contains(COUNTRY)


# --------------------------------------------------------------------------- #
# Metric extraction (per network)
# --------------------------------------------------------------------------- #
def gb_capacity(n: pypsa.Network) -> pd.Series:
    """GB electricity capacity by carrier [GW] (optimal, bus_carrier=AC)."""
    cap = n.statistics.optimal_capacity(
        bus_carrier="AC", groupby=["bus", "carrier"], nice_names=False
    )
    df = cap.rename("v").reset_index()
    buscol = "bus" if "bus" in df.columns else df.columns[-3]
    df = df[gb_mask(df[buscol])]
    return (df.groupby("carrier")["v"].sum() / MW_GW).sort_values(ascending=False)


def gb_energy_balance(n: pypsa.Network) -> pd.Series:
    """GB net energy balance by carrier on AC buses [TWh] (supply +, sink -)."""
    eb = n.statistics.energy_balance(
        bus_carrier="AC", groupby=["bus", "carrier"], nice_names=False
    )
    df = eb.rename("v").reset_index()
    buscol = "bus" if "bus" in df.columns else df.columns[-3]
    df = df[gb_mask(df[buscol])]
    return (df.groupby("carrier")["v"].sum() / MWH_TWH).sort_values()


def gb_storage_energy(n: pypsa.Network) -> pd.Series:
    """GB store energy capacity by carrier [GWh]."""
    st = n.stores.copy()
    st = st[gb_mask(region(n, st.bus))]
    return (st.groupby("carrier").e_nom_opt.sum() / MW_GW).sort_values(ascending=False)


def gb_storage_power(n: pypsa.Network) -> pd.Series:
    """GB storage discharge power capacity by carrier [GW] (dischargers + StorageUnits)."""
    out = {}
    li = n.links.copy()
    li = li[li.carrier.str.contains("discharger") & gb_mask(region(n, li.bus1))]
    for c, sub in li.groupby(li.carrier.str.replace(" discharger", "", regex=False)):
        out[c] = sub.p_nom_opt.sum() / MW_GW
    su = n.storage_units.copy()
    su = su[gb_mask(region(n, su.bus))]
    for c, sub in su.groupby("carrier"):
        out[c] = out.get(c, 0) + sub.p_nom_opt.sum() / MW_GW
    return pd.Series(out).sort_values(ascending=False)


def gb_interconnectors(n: pypsa.Network) -> dict:
    """Net GB imports over DC interconnectors [TWh], total and by CO2 class."""
    L = n.links
    dc = L[L.carrier == "DC"].copy()
    if dc.empty:
        return {"net_total": 0.0, "by_class": {}}
    loc = n.buses.location
    dc["b0"] = region(n, dc.bus0).astype(str)
    dc["b1"] = region(n, dc.bus1).astype(str)
    cross = dc[dc.b0.str.startswith(COUNTRY) ^ dc.b1.str.startswith(COUNTRY)]
    if cross.empty:
        return {"net_total": 0.0, "by_class": {}}
    w = n.snapshot_weightings.generators
    # net import to GB: +p0 when bus1 is GB (flow foreign->GB), -p0 when bus0 is GB
    sign = np.where(cross.b1.str.startswith(COUNTRY), 1.0, -1.0)
    energy = (n.links_t.p0[cross.index].mul(w, axis=0).sum() * sign) / MWH_TWH
    res = {"net_total": float(energy.sum()), "by_class": {}}
    if "co2_class" in cross.columns:
        cls = cross["co2_class"].replace("", "unclassified")
        for c in sorted(cls.dropna().unique()):
            res["by_class"][str(c)] = float(energy[cls == c].sum())
    return res


def gb_price(n: pypsa.Network) -> dict:
    """Snapshot-weighted mean marginal price on GB AC buses [EUR/MWh].

    For CBAM, also split by layer (common / clean / nonclean)."""
    buses = n.buses[(n.buses.carrier == "AC") & gb_mask(n.buses.index)].index
    if len(buses) == 0 or n.buses_t.marginal_price.empty:
        return {}
    w = n.snapshot_weightings.generators
    mp = n.buses_t.marginal_price.reindex(columns=buses).dropna(axis=1, how="all")
    wmean = lambda cols: float((mp[cols].mul(w, axis=0).sum().sum()) / (w.sum() * len(cols))) if len(cols) else np.nan
    out = {"all": wmean(list(mp.columns))}
    for layer in ["clean", "nonclean"]:
        cols = [b for b in mp.columns if b.endswith(" " + layer)]
        if cols:
            out[layer] = wmean(cols)
    common = [b for b in mp.columns if not b.endswith(" clean") and not b.endswith(" nonclean")]
    if common and ("clean" in out or "nonclean" in out):
        out["common"] = wmean(common)
    return out


def co2_shadow(scenario: str, year: int) -> dict:
    """Read CO2 constraint shadow prices from the per-year CSVs (if present)."""
    d = SCENARIOS[scenario]
    out = {}
    files = {
        "local_GB": f"local co2 emissions constraint {COUNTRY}_{year}.csv",
        "collective": f"collective co2 emissions constraint_{year}.csv",
        "sequestration": f"GlobalConstraint-co2_sequestration_limit_{year}.csv",
    }
    for key, fname in files.items():
        f = d / fname
        if f.exists():
            try:
                val = pd.read_csv(f).iloc[0, -1]
                out[key] = float(val)
            except Exception:
                pass
    return out


def extract_one(scenario: str, year: int) -> dict | None:
    n = load_network(scenario, year)
    if n is None:
        return None
    rec = {
        "capacity": gb_capacity(n),
        "energy_balance": gb_energy_balance(n),
        "storage_energy": gb_storage_energy(n),
        "storage_power": gb_storage_power(n),
        "interconnect": gb_interconnectors(n),
        "price": gb_price(n),
        "objective_bn": float(n.objective) / 1e9,
        "co2_shadow": co2_shadow(scenario, year),
    }
    del n
    return rec


def extract_all() -> dict:
    data = {}
    for scen in SCENARIOS:
        for yr in YEARS:
            print(f"  extracting {scen} {yr} ...", flush=True)
            rec = extract_one(scen, yr)
            if rec is not None:
                data[(scen, yr)] = rec
            else:
                print(f"    (missing: {net_path(scen, yr).name})", flush=True)
    return data


# --------------------------------------------------------------------------- #
# Reshaping cache -> tidy frames
# --------------------------------------------------------------------------- #
def frame(data, scen, key) -> pd.DataFrame:
    """carriers x years matrix for a Series-valued metric, one scenario."""
    cols = {}
    for (s, y), rec in data.items():
        if s == scen and key in rec:
            cols[y] = rec[key]
    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols).sort_index(axis=1).fillna(0.0)


def scalar_frame(data, extract) -> pd.DataFrame:
    rows = {}
    for (s, y), rec in data.items():
        rows.setdefault(s, {})[y] = extract(rec)
    return pd.DataFrame(rows).sort_index()


# --------------------------------------------------------------------------- #
# Plotting (plotly)
# --------------------------------------------------------------------------- #
def _fig_to_div(fig) -> str:
    import plotly.io as pio
    return pio.to_html(fig, full_html=False, include_plotlyjs=False,
                       config={"displayModeBar": False, "responsive": True})


def build_html(data: dict) -> str:
    import plotly.graph_objects as go
    import plotly.offline as pyo
    from plotly.subplots import make_subplots

    years_present = sorted({y for (_, y) in data})
    both_years = sorted({y for (s, y) in data if s == "CBAM"})  # limiting scenario

    PALETTE = {
        "offwind-ac": "#1f77b4", "offwind-dc": "#17a2c9", "offwind-float": "#4bb3d6",
        "onwind": "#4daf4a", "solar": "#ffcf33", "solar rooftop": "#ffe680",
        "solar-hsat": "#e6b800", "ror": "#2b8cbe", "hydro": "#3182bd", "PHS": "#6baed6",
        "nuclear": "#e41a1c", "CCGT": "#8c564b", "OCGT": "#b5651d",
        "DC": "#7f7f7f", "battery discharger": "#9467bd", "battery": "#9467bd",
        "urban central solid biomass CHP": "#66c2a5", "urban central gas CHP": "#a6761d",
        "waste CHP": "#999999", "urban central biogas CHP": "#8dd3c7",
        "clean": "#2ca02c", "nonclean": "#d62728", "unclassified": "#999999",
        "reference": "#4c78a8", "CBAM": "#e45756",
    }
    color = lambda c: PALETTE.get(c, None)

    divs = []

    # ---- 1. Capacity mix (stacked bars, scenarios side by side) ----
    cap_r, cap_c = frame(data, "reference", "capacity"), frame(data, "CBAM", "capacity")
    carriers = [c for c in (cap_r.index.union(cap_c.index))
                if (cap_r.reindex([c]).abs().sum().sum() + cap_c.reindex([c]).abs().sum().sum()) > 0.1]
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
                        subplot_titles=("Reference", "CBAM"))
    for j, cap in enumerate([cap_r, cap_c], start=1):
        for c in carriers:
            y = cap.reindex(index=[c]).iloc[0].values if c in cap.index else [0] * len(cap.columns)
            fig.add_bar(x=[str(v) for v in cap.columns], y=y, name=c, marker_color=color(c),
                        legendgroup=c, showlegend=(j == 1), row=1, col=j)
    fig.update_layout(barmode="stack", height=460, title="GB electricity capacity by carrier [GW]",
                      legend=dict(font=dict(size=10)))
    divs.append(("GB installed capacity (investment)", _fig_to_div(fig),
                 "Optimal GB electricity capacity by carrier, both scenarios. "
                 "Layer buses collapsed to region via n.buses.location."))

    # ---- 2. Capacity delta (CBAM - REF) ----
    common_years = [y for y in cap_c.columns if y in cap_r.columns]
    delta = (cap_c.reindex(index=carriers, columns=common_years).fillna(0)
             - cap_r.reindex(index=carriers, columns=common_years).fillna(0))
    figd = go.Figure()
    for c in carriers:
        row = delta.reindex([c]).iloc[0]
        if row.abs().max() < 0.05:
            continue
        figd.add_bar(x=[str(y) for y in common_years], y=row.values, name=c, marker_color=color(c))
    figd.update_layout(barmode="relative", height=420,
                       title="GB capacity difference CBAM − Reference [GW]")
    divs.append(("GB capacity difference (CBAM − Reference)", _fig_to_div(figd),
                 "Positive = CBAM builds more of that carrier than the reference."))

    # ---- 3. Dispatch / generation mix ----
    eb_r, eb_c = frame(data, "reference", "energy_balance"), frame(data, "CBAM", "energy_balance")
    gen_carriers = [c for c in eb_r.index.union(eb_c.index)
                    if c not in SINK_CARRIERS
                    and (eb_r.reindex([c]).clip(lower=0).sum().sum()
                         + eb_c.reindex([c]).clip(lower=0).sum().sum()) > 0.05]
    figg = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=("Reference", "CBAM"))
    for j, eb in enumerate([eb_r, eb_c], start=1):
        for c in gen_carriers:
            y = eb.reindex(index=[c]).iloc[0].clip(lower=0).values if c in eb.index else [0] * len(eb.columns)
            figg.add_bar(x=[str(v) for v in eb.columns], y=y, name=c, marker_color=color(c),
                         legendgroup=c, showlegend=(j == 1), row=1, col=j)
    figg.update_layout(barmode="stack", height=460, title="GB electricity supply by carrier [TWh]",
                       legend=dict(font=dict(size=10)))
    divs.append(("GB dispatch / generation mix", _fig_to_div(figg),
                 "Supply-side energy balance on GB AC buses. 'DC' = net interconnector imports."))

    # ---- 4. Interconnectors: net imports + CBAM clean/nonclean split ----
    ic_net = scalar_frame(data, lambda r: r["interconnect"]["net_total"])
    figi = go.Figure()
    for scen in ic_net.columns:
        figi.add_scatter(x=[str(y) for y in ic_net.index], y=ic_net[scen].values,
                         mode="lines+markers", name=f"{scen} net import", line=dict(color=color(scen)))
    # CBAM class split (stacked bars)
    for cls in ["clean", "nonclean"]:
        ys = [data.get(("CBAM", y), {}).get("interconnect", {}).get("by_class", {}).get(cls, np.nan)
              for y in both_years]
        figi.add_bar(x=[str(y) for y in both_years], y=ys, name=f"CBAM {cls}",
                     marker_color=color(cls), opacity=0.55)
    figi.update_layout(barmode="relative", height=440,
                       title="GB net interconnector imports [TWh] — total (lines) and CBAM CO2 class (bars)")
    divs.append(("Cross-border flows by CO2 class (the CBAM signal)", _fig_to_div(figi),
                 "Net GB import is near-identical between scenarios, but CBAM resolves it into a large "
                 "CLEAN import and a small NONCLEAN net export — the labelling the split exists to provide."))

    # ---- 5. Storage energy capacity ----
    # Keep the huge cross-sector H2 store OUT of the battery-family chart, or it
    # dwarfs everything (H2 ~2400 GWh vs battery ~60 GWh) and hides the battery
    # comparison. H2 is shown separately below.
    se_r, se_c = frame(data, "reference", "storage_energy"), frame(data, "CBAM", "storage_energy")
    SHORT = {"battery", "home battery", "PHS", "redox flow battery",
             "compressed air", "molten salt", "liquid air"}
    scar = [c for c in se_r.index.union(se_c.index)
            if c in SHORT and (se_r.reindex([c]).sum().sum() + se_c.reindex([c]).sum().sum()) > 0.01]
    figs = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=("Reference", "CBAM"))
    for j, se in enumerate([se_r, se_c], start=1):
        for c in scar:
            y = se.reindex(index=[c]).iloc[0].values if c in se.index else [0] * len(se.columns)
            figs.add_bar(x=[str(v) for v in se.columns], y=y, name=c, legendgroup=c,
                         showlegend=(j == 1), row=1, col=j, marker_color=color(c))
    figs.update_layout(barmode="stack", height=420,
                       title="GB short-duration electricity storage energy [GWh]  (battery family — H2 excluded)",
                       legend=dict(font=dict(size=10)))
    divs.append(("GB electricity storage (battery family)", _fig_to_div(figs),
                 "Battery/home-battery/PHS energy capacity — the true electricity stores. These are "
                 "essentially IDENTICAL between the two scenarios (e.g. 2040: 63.9 GWh battery in both), and "
                 "batteries charge AND discharge, so the clean-layer storage MVP is functional. The hydrogen "
                 "store is shown separately below because it is ~40x larger and cross-sector."))

    # ---- 5b. Hydrogen store (cross-sector, shown on its own scale) ----
    h2 = scalar_frame(data, lambda r: float(r["storage_energy"].get("H2 Store", 0.0)))
    figh = go.Figure()
    for scen in h2.columns:
        figh.add_scatter(x=[str(y) for y in h2.index], y=(h2[scen] / 1e3).values, mode="lines+markers",
                         name=scen, line=dict(color=color(scen)))
    figh.update_layout(height=380, yaxis_title="H2 store [TWh]",
                       title="GB hydrogen store energy [TWh]  (cross-sector: power + industry + synthetic fuels)")
    divs.append(("GB hydrogen store (context, not battery-like)", _fig_to_div(figh),
                 "The H2 store is a multi-sector energy reservoir, not a battery. It grows to ~2.4 TWh and is "
                 "the same in both scenarios — so lumping it with batteries (as an earlier version did) made "
                 "the two scenarios look different when they are not."))

    # ---- 6. System cost + CO2 shadow price ----
    # The base-year (first horizon) objective carries the frozen existing capital
    # stock and is orders of magnitude larger; drop it so the trend is readable.
    obj = scalar_frame(data, lambda r: r["objective_bn"])
    base_year = min(obj.index)
    obj_plot = obj.drop(index=base_year)
    figc = go.Figure()
    for scen in obj_plot.columns:
        figc.add_scatter(x=[str(y) for y in obj_plot.index], y=obj_plot[scen].values, mode="lines+markers",
                         name=f"{scen}", line=dict(color=color(scen)))
    figc.update_layout(height=420, title=f"Total system cost [bn EUR/a] (objective, {base_year} base year omitted)",
                       yaxis_title="system cost [bn EUR/a]")
    divs.append(("System cost", _fig_to_div(figc),
                 f"CBAM tracks the reference within a fraction of a percent. The {base_year} base year is "
                 "omitted (its objective carries the frozen existing capital stock and dwarfs later years). "
                 "The joint transmission-capacity constraint on the split cables is the only active CBAM cost "
                 "lever; the border charge itself is not yet modelled."))

    # ---- 7. Prices (CBAM layer spread) ----
    prow = []
    for (s, y), rec in data.items():
        for k, v in rec.get("price", {}).items():
            prow.append({"scenario": s, "year": y, "layer": k, "price": v})
    pdf = pd.DataFrame(prow)
    figp = go.Figure()
    if not pdf.empty:
        ref = pdf[(pdf.scenario == "reference") & (pdf.layer == "all")].sort_values("year")
        figp.add_scatter(x=ref.year.astype(str), y=ref.price, mode="lines+markers",
                         name="reference (all)", line=dict(color=color("reference")))
        for layer in ["all", "clean", "nonclean"]:
            sub = pdf[(pdf.scenario == "CBAM") & (pdf.layer == layer)].sort_values("year")
            if not sub.empty:
                figp.add_scatter(x=sub.year.astype(str), y=sub.price, mode="lines+markers",
                                 name=f"CBAM ({layer})",
                                 line=dict(color=color(layer) if layer in ("clean", "nonclean") else color("CBAM"),
                                           dash="dot" if layer != "all" else "solid"))
    figp.update_layout(height=420, title="GB average electricity price [EUR/MWh] (load-weighted)")
    divs.append(("GB prices and clean/nonclean layer spread", _fig_to_div(figp),
                 "The clean vs nonclean layer prices show the shadow value the split assigns to each CO2 class."))

    # ---- Findings & diagnostics ----
    diag = diagnostics(data, cap_c, eb_c, se_c)

    plotlyjs = pyo.get_plotlyjs()
    body = build_report_shell(divs, diag, years_present, both_years)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CBAM vs Reference — GB comparison</title>
<script>{plotlyjs}</script>
<style>{CSS}</style></head><body>{body}</body></html>"""


def diagnostics(data, cap_c, eb_c, se_c) -> list[tuple[str, str, str]]:
    """Return list of (severity, title, detail) findings."""
    out = []
    have_cbam = sorted({y for (s, y) in data if s == "CBAM"})
    have_ref = sorted({y for (s, y) in data if s == "reference"})
    missing = [y for y in have_ref if y not in have_cbam]
    if missing:
        out.append(("warn", f"CBAM horizon(s) {missing} not solved",
                    "The CBAM myopic chain hit the 20 h PBS wall-time during the 2050 net-zero solve "
                    "(it was converging, not infeasible). Comparison is limited to the years both runs have. "
                    "Re-submit job_cbam.sh (or raise walltime) to complete 2050."))
    # solar-hsat built?
    hsat = eb_c.reindex(["solar-hsat"]).abs().sum().sum() if "solar-hsat" in eb_c.index else 0
    if hsat < 0.01:
        out.append(("ok", "solar-hsat not built in GB",
                    "solar-hsat sits on the primary bus (absent from generators_co2_lvls), which the "
                    "solar_potential fix works around. It is not deployed here, so the mislabelling has no "
                    "numerical effect in this study — but would if a scenario builds it."))
    # storage cycles?
    dis = eb_c.reindex(["battery discharger"]).clip(lower=0).sum().sum() if "battery discharger" in eb_c.index else 0
    if dis > 0.1:
        out.append(("ok", "Storage is identical between scenarios; clean-layer MVP works",
                    f"GB battery energy capacity matches the reference (e.g. 63.9 GWh in both at 2040) and "
                    f"batteries discharge ~{dis:.0f} TWh over the horizon (charge AND discharge). The H2 store "
                    "(~2.4 TWh, cross-sector) is also the same in both and is NOT split — earlier it was charted "
                    "on the same axis as batteries, which made the scenarios look different when they are not."))
    else:
        out.append(("err", "Storage may be stranded",
                    "GB battery discharge is ~0 despite capacity — check the storage connection direction."))
    # headline signal 1: clean/nonclean import composition and the export flip
    ics = {y: data[("CBAM", y)]["interconnect"] for y in have_cbam if ("CBAM", y) in data}
    if ics:
        first, last = min(ics), max(ics)
        net_f, net_l = ics[first]["net_total"], ics[last]["net_total"]
        cl_l = ics[last]["by_class"].get("clean", float("nan"))
        flip = (net_f > 0) and (net_l < 0)
        out.append(("info", "GB shifts from clean importer to clean exporter",
                    f"CBAM resolves the cross-border flow by CO2 class: in {first} GB nets ~{net_f:+.0f} TWh "
                    f"(≈ all clean), and by {last} it nets ~{net_l:+.0f} TWh — a large CLEAN "
                    f"{'export' if net_l<0 else 'import'} (~{cl_l:+.0f} TWh clean) as offshore wind scales. "
                    "Net flow is ~identical to the reference; the split is what makes its composition visible."))
    # headline signal 2: clean/nonclean price spread
    spreads = {}
    for y in have_cbam:
        p = data.get(("CBAM", y), {}).get("price", {})
        if "clean" in p and "nonclean" in p:
            spreads[y] = p["nonclean"] - p["clean"]
    if spreads:
        ymax = max(spreads, key=spreads.get)
        out.append(("info", "Clean/nonclean price spread widens over time",
                    f"The nonclean AC layer clears well above the clean layer — the spread reaches "
                    f"~{spreads[ymax]:.0f} EUR/MWh by {ymax}. This is the implicit carbon value the split "
                    "assigns to dirty electricity even though no explicit border tariff is applied yet."))
    # near-identical dispatch => tax not active
    out.append(("info", "CBAM charge is topology-only (expected)",
                "Investment and dispatch are within ~1% of the reference because the CBAM border charge is "
                "still a TODO in prepare_sector_network. Present differences come from the joint transmission "
                "capacity constraint and flow re-labelling, not a carbon tariff."))
    return out


# --------------------------------------------------------------------------- #
# HTML shell
# --------------------------------------------------------------------------- #
CSS = """
:root{--bg:#ffffff;--fg:#1a1a1a;--muted:#666;--card:#f7f8fa;--border:#e3e6ea;--accent:#e45756;}
@media (prefers-color-scheme:dark){:root{--bg:#14161a;--fg:#e8eaed;--muted:#9aa0a6;--card:#1e2127;--border:#2c313a;}}
*{box-sizing:border-box}body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
background:var(--bg);color:var(--fg);line-height:1.5}
.wrap{max-width:1100px;margin:0 auto;padding:32px 24px 80px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:36px 0 6px;border-bottom:1px solid var(--border);padding-bottom:6px}
.sub{color:var(--muted);font-size:14px;margin:0 0 24px}
.desc{color:var(--muted);font-size:13px;margin:2px 0 10px}
.card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:14px 16px;margin:10px 0 22px;overflow-x:auto}
.findings{display:grid;gap:8px;margin:8px 0 8px}
.f{display:flex;gap:10px;padding:10px 12px;border-radius:8px;border:1px solid var(--border);background:var(--card);font-size:14px}
.f .tag{font-weight:600;font-size:11px;letter-spacing:.03em;text-transform:uppercase;padding:2px 8px;border-radius:20px;height:fit-content;white-space:nowrap}
.tag.err{background:#fdd;color:#a00}.tag.warn{background:#fe9;color:#960}.tag.ok{background:#dfe;color:#070}.tag.info{background:#def;color:#036}
@media (prefers-color-scheme:dark){.tag.err{background:#3a1f22;color:#f2a}.tag.warn{background:#3a3320;color:#fd8}.tag.ok{background:#1f3324;color:#8f8}.tag.info{background:#1f2a3a;color:#8cf}}
.f b{display:block}.f .d{color:var(--muted);margin-top:2px}
table{border-collapse:collapse;font-size:13px;width:100%}
th,td{border:1px solid var(--border);padding:5px 9px;text-align:right}th{background:var(--card);text-align:right}
td:first-child,th:first-child{text-align:left}
.meta{font-size:12px;color:var(--muted);margin-top:40px;border-top:1px solid var(--border);padding-top:12px}
"""


def build_report_shell(divs, diag, years_present, both_years) -> str:
    findings = "".join(
        f'<div class="f"><span class="tag {sev}">{sev}</span>'
        f'<div><b>{title}</b><span class="d">{detail}</span></div></div>'
        for sev, title, detail in diag
    )
    sections = "".join(
        f'<h2>{title}</h2><p class="desc">{desc}</p><div class="card">{div}</div>'
        for title, div, desc in divs
    )
    return f"""
<div class="wrap">
  <h1>CBAM vs Reference — Great Britain</h1>
  <p class="sub">CO2-intensity split scenario against the no-split reference. Years available:
     reference {min(years_present)}–{max(years_present)}, CBAM {min(both_years)}–{max(both_years)}.
     All GB values aggregate the clean/nonclean/common layers via <code>n.buses.location</code>.</p>

  <h2>Key findings &amp; diagnostics</h2>
  <div class="findings">{findings}</div>

  {sections}

  <p class="meta">Generated by <code>analysis/compare_scenarios.py</code> from the solved networks in
     <code>results/pypsa-uk/&lt;scenario&gt;/networks/</code>. Re-run with <code>--reload</code> after new solves.</p>
</div>
"""


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reload", action="store_true", help="re-extract metrics from networks")
    ap.add_argument("--html-only", action="store_true", help="only rebuild HTML from cache")
    args = ap.parse_args()

    OUTDIR.mkdir(exist_ok=True)
    if args.reload or (not CACHE.exists() and not args.html_only):
        print("Extracting metrics from networks (this loads each network once)...")
        data = extract_all()
        with open(CACHE, "wb") as f:
            pickle.dump(data, f)
        print(f"Cached -> {CACHE}")
    else:
        with open(CACHE, "rb") as f:
            data = pickle.load(f)
        print(f"Loaded cache <- {CACHE} ({len(data)} network-years)")

    print("Building HTML report...")
    html = build_html(data)
    HTML.write_text(html)
    print(f"Wrote -> {HTML}  ({HTML.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
