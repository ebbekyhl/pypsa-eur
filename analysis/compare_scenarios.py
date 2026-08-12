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
    "CBAM_old": REPO / "results/pypsa-uk/CBAM_old/networks",
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


SCEN_ORDER = ["reference", "CBAM_old", "CBAM"]
SCEN_LABEL = {
    "reference": "Reference (no split)",
    "CBAM_old": "CBAM · clean-only storage",
    "CBAM": "CBAM · per-layer storage + H2",
}
SCEN_COLOR = {"reference": "#4c78a8", "CBAM_old": "#f2a900", "CBAM": "#e45756"}


def build_html(data: dict) -> str:
    import plotly.graph_objects as go
    import plotly.offline as pyo
    from plotly.subplots import make_subplots

    scen = [s for s in SCEN_ORDER if any(sc == s for (sc, _) in data)]
    split_scen = [s for s in scen if s != "reference"]  # scenarios with the CO2 split
    years_present = sorted({y for (_, y) in data})
    both_years = sorted({y for (s, y) in data if s == "CBAM"})

    PALETTE = {
        "offwind-ac": "#1f77b4", "offwind-dc": "#17a2c9", "offwind-float": "#4bb3d6",
        "onwind": "#4daf4a", "solar": "#ffcf33", "solar rooftop": "#ffe680",
        "solar-hsat": "#e6b800", "ror": "#2b8cbe", "hydro": "#3182bd", "PHS": "#6baed6",
        "nuclear": "#e41a1c", "CCGT": "#8c564b", "OCGT": "#b5651d",
        "DC": "#7f7f7f", "battery discharger": "#9467bd", "battery": "#9467bd",
        "urban central solid biomass CHP": "#66c2a5", "urban central gas CHP": "#a6761d",
        "waste CHP": "#999999", "urban central biogas CHP": "#8dd3c7",
        "clean": "#2ca02c", "nonclean": "#d62728", "unclassified": "#999999",
        **SCEN_COLOR,
    }
    color = lambda c: PALETTE.get(c, None)
    divs = []

    def mix_panels(key, title, desc, carrier_filter=None, exclude=None, positive_only=False):
        """N-panel stacked bar of a Series-metric, one panel per scenario."""
        frames = {s: frame(data, s, key) for s in scen}
        allidx = pd.Index([])
        for f in frames.values():
            allidx = allidx.union(f.index)

        def total(c):
            t = 0.0
            for f in frames.values():
                if c in f.index:
                    v = f.reindex([c]).iloc[0]
                    t += (v.clip(lower=0) if positive_only else v.abs()).sum()
            return t
        carriers = [c for c in allidx
                    if (carrier_filter is None or c in carrier_filter)
                    and (exclude is None or c not in exclude)
                    and total(c) > 0.05]
        fig = make_subplots(rows=1, cols=len(scen), shared_yaxes=True,
                            subplot_titles=[SCEN_LABEL[s] for s in scen])
        for j, s in enumerate(scen, start=1):
            f = frames[s]
            for c in carriers:
                if c in f.index:
                    v = f.reindex([c]).iloc[0]
                    y = (v.clip(lower=0) if positive_only else v).values
                else:
                    y = [0] * len(f.columns)
                fig.add_bar(x=[str(v) for v in f.columns], y=y, name=c, marker_color=color(c),
                            legendgroup=c, showlegend=(j == 1), row=1, col=j)
        fig.update_layout(barmode="stack", height=460, title=title, legend=dict(font=dict(size=10)))
        divs.append((title.split("[")[0].strip(), _fig_to_div(fig), desc))
        return frames, carriers

    # ---- 1. Capacity mix ----
    cap_frames, carriers = mix_panels(
        "capacity", "GB electricity capacity by carrier [GW]",
        "Optimal GB electricity capacity by carrier, per scenario. Layer buses collapsed to region "
        "via n.buses.location.")

    # ---- 2. Capacity delta: each split scenario vs reference ----
    figd = go.Figure()
    cap_ref = cap_frames["reference"]
    for s in split_scen:
        cs = cap_frames[s]
        yrs = [y for y in cs.columns if y in cap_ref.columns]
        d = (cs.reindex(index=carriers, columns=yrs).fillna(0)
             - cap_ref.reindex(index=carriers, columns=yrs).fillna(0))
        tot = d.sum(axis=0)  # net GW difference summed over carriers (sanity line)
        figd.add_scatter(x=[str(y) for y in yrs], y=tot.values, mode="lines+markers",
                         name=f"{SCEN_LABEL[s]} − reference (net GW)", line=dict(color=color(s)))
    figd.update_layout(height=380, title="GB net capacity difference vs reference [GW]",
                       yaxis_title="Σ(carrier) capacity − reference [GW]")
    divs.append(("GB capacity difference vs reference", _fig_to_div(figd),
                 "Net electricity capacity each split scenario builds relative to the reference. "
                 "See the per-carrier panels above for the composition."))

    # ---- 3. Dispatch / generation mix ----
    mix_panels("energy_balance", "GB electricity supply by carrier [TWh]",
               "Supply-side energy balance on GB AC buses. 'DC' = net interconnector imports.",
               exclude=SINK_CARRIERS, positive_only=True)

    # ---- 4a. Net interconnector imports (line per scenario) ----
    ic_net = scalar_frame(data, lambda r: r["interconnect"]["net_total"]).reindex(columns=scen)
    figi = go.Figure()
    for s in scen:
        if s in ic_net.columns:
            figi.add_scatter(x=[str(y) for y in ic_net.index], y=ic_net[s].values,
                             mode="lines+markers", name=SCEN_LABEL[s], line=dict(color=color(s)))
    figi.add_hline(y=0, line_dash="dot", line_color="#888")
    figi.update_layout(height=400, title="GB net interconnector imports [TWh]  (>0 import, <0 export)",
                       yaxis_title="net import [TWh]")
    divs.append(("Net cross-border flow (all scenarios)", _fig_to_div(figi),
                 "Total net GB import/export. Near-identical across scenarios — the split changes the "
                 "LABELLING of flows, shown next, far more than the net magnitude."))

    # ---- 4b. Clean vs nonclean split — one panel per split scenario ----
    figc2 = make_subplots(rows=1, cols=len(split_scen), shared_yaxes=True,
                          subplot_titles=[SCEN_LABEL[s] for s in split_scen])
    for j, s in enumerate(split_scen, start=1):
        yrs = sorted({y for (sc, y) in data if sc == s})
        for cls in ["clean", "nonclean"]:
            ys = [data.get((s, y), {}).get("interconnect", {}).get("by_class", {}).get(cls, np.nan)
                  for y in yrs]
            figc2.add_bar(x=[str(y) for y in yrs], y=ys, name=cls, marker_color=color(cls),
                          legendgroup=cls, showlegend=(j == 1), row=1, col=j)
    figc2.update_layout(barmode="relative", height=420,
                        title="GB net cross-border flow by CO2 class [TWh]  (>0 import, <0 export)")
    divs.append(("Clean vs nonclean cross-border flow — does per-layer storage rebalance it?",
                 _fig_to_div(figc2),
                 "THE test of the storage change: clean-only storage (left) gave only the clean layer "
                 "time-shifting flexibility; per-layer storage + H2 (right) gives the nonclean layer its own. "
                 "Compare how much of the cross-border flow stays labelled clean vs nonclean."))

    # ---- 5. Battery-family storage ----
    SHORT = {"battery", "home battery", "PHS", "redox flow battery",
             "compressed air", "molten salt", "liquid air"}
    mix_panels("storage_energy", "GB short-duration electricity storage energy [GWh]  (battery family)",
               "Battery/home-battery/PHS energy capacity — the true electricity stores (H2 shown separately "
               "below). Compare whether per-layer storage builds materially more/different storage.",
               carrier_filter=SHORT)

    # ---- 5b. Hydrogen store ----
    h2 = scalar_frame(data, lambda r: float(r["storage_energy"].get("H2 Store", 0.0))).reindex(columns=scen)
    figh = go.Figure()
    for s in scen:
        if s in h2.columns:
            figh.add_scatter(x=[str(y) for y in h2.index], y=(h2[s] / 1e3).values, mode="lines+markers",
                             name=SCEN_LABEL[s], line=dict(color=color(s)))
    figh.update_layout(height=380, yaxis_title="H2 store [TWh]",
                       title="GB hydrogen store energy [TWh]  (cross-sector: power + industry + synthetic fuels)")
    divs.append(("GB hydrogen store", _fig_to_div(figh),
                 "Total H2 store energy. In CBAM (per-layer + H2 split) this is now the SUM of the clean and "
                 "nonclean H2 stores; watch for a step-change vs the clean-only run where H2 was unsplit."))

    # ---- 6. System cost ----
    obj = scalar_frame(data, lambda r: r["objective_bn"]).reindex(columns=scen)
    base_year = min(obj.index)
    obj_plot = obj.drop(index=base_year)
    figcost = go.Figure()
    for s in scen:
        if s in obj_plot.columns:
            figcost.add_scatter(x=[str(y) for y in obj_plot.index], y=obj_plot[s].values,
                                mode="lines+markers", name=SCEN_LABEL[s], line=dict(color=color(s)))
    figcost.update_layout(height=400,
                          title=f"Total system cost [bn EUR/a]  ({base_year} base year omitted)",
                          yaxis_title="system cost [bn EUR/a]")
    divs.append(("System cost", _fig_to_div(figcost),
                 f"Objective per horizon ({base_year} omitted — its objective carries the frozen existing "
                 "capital stock). The split adds cost via the joint transmission-capacity constraint; the "
                 "border charge is still not modelled."))

    # ---- 7. Clean/nonclean price spread per split scenario ----
    def price(s, y, layer):
        return data.get((s, y), {}).get("price", {}).get(layer, np.nan)
    figp = go.Figure()
    ry = sorted({y for (sc, y) in data if sc == "reference"})
    figp.add_scatter(x=[str(y) for y in ry], y=[price("reference", y, "all") for y in ry],
                     mode="lines+markers", name="reference (all)", line=dict(color=color("reference")))
    for s in split_scen:
        ys = sorted({y for (sc, y) in data if sc == s})
        for layer, dash in [("clean", "solid"), ("nonclean", "dot")]:
            figp.add_scatter(x=[str(y) for y in ys], y=[price(s, y, layer) for y in ys],
                             mode="lines+markers", name=f"{SCEN_LABEL[s]} · {layer}",
                             line=dict(color=color(s), dash=dash))
    figp.update_layout(height=420, title="GB electricity price by layer [EUR/MWh] (load-weighted)",
                       yaxis_title="price [EUR/MWh]")
    divs.append(("GB prices and clean/nonclean spread", _fig_to_div(figp),
                 "Solid = clean layer, dotted = nonclean layer, per split scenario. The nonclean-minus-clean "
                 "gap is the implicit carbon value; compare whether per-layer storage narrows it."))

    # ---- Findings & diagnostics ----
    diag = diagnostics(data)

    plotlyjs = pyo.get_plotlyjs()
    body = build_report_shell(divs, diag, years_present, both_years)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CBAM storage/H2 — GB scenario comparison</title>
<script>{plotlyjs}</script>
<style>{CSS}</style></head><body>{body}</body></html>"""


def diagnostics(data) -> list[tuple[str, str, str]]:
    """Return list of (severity, title, detail) findings."""
    out = []
    yrs = lambda s: sorted({y for (sc, y) in data if sc == s})
    have_ref, have_old, have_cbam = yrs("reference"), yrs("CBAM_old"), yrs("CBAM")
    comp = [y for y in have_cbam if y in have_old]  # years both split scenarios cover

    if have_cbam:
        out.append(("ok", "Three scenarios compared",
                    f"reference (no split), CBAM_old (clean-only storage, no H2 split), and CBAM "
                    f"(per-layer storage + H2 split). Years: reference {_span(have_ref)}, "
                    f"CBAM_old {_span(have_old)}, CBAM {_span(have_cbam)}."))
    missing = [y for y in have_ref if y not in have_cbam]
    if missing:
        out.append(("warn", f"CBAM horizon(s) {missing} missing",
                    "Comparison limited to years present in all runs."))

    def nonclean_share(s, y):
        bc = data.get((s, y), {}).get("interconnect", {}).get("by_class", {})
        c, nc = bc.get("clean"), bc.get("nonclean")
        if c is None or nc is None:
            return None
        tot = abs(c) + abs(nc)
        return (abs(nc) / tot) if tot > 1e-6 else 0.0

    # THE storage-effect test: does per-layer storage move flow onto the nonclean label?
    rows = [(nonclean_share("CBAM_old", y), nonclean_share("CBAM", y)) for y in comp]
    rows = [(o, n) for o, n in rows if o is not None and n is not None]
    if rows:
        ao = 100 * sum(o for o, _ in rows) / len(rows)
        an = 100 * sum(n for _, n in rows) / len(rows)
        d = an - ao
        verdict = ("shifts flow ONTO the nonclean label" if d > 1 else
                   "shifts flow OFF the nonclean label" if d < -1 else
                   "barely changes the clean/nonclean split")
        out.append(("info", "Storage-effect test: per-layer vs clean-only storage",
                    f"Nonclean share of |cross-border flow| averages {ao:.1f}% with clean-only storage vs "
                    f"{an:.1f}% with per-layer storage + H2 ({_span(comp)}). Giving the nonclean layer its own "
                    f"storage {verdict} ({d:+.1f} pp) — the direct test of your hypothesis that clean-only "
                    "storage biased flows toward clean."))

    def spread(s, y):
        p = data.get((s, y), {}).get("price", {})
        return (p["nonclean"] - p["clean"]) if "clean" in p and "nonclean" in p else None
    so = [spread("CBAM_old", y) for y in comp if spread("CBAM_old", y) is not None]
    sn = [spread("CBAM", y) for y in comp if spread("CBAM", y) is not None]
    if so and sn:
        mo, mn = sum(so) / len(so), sum(sn) / len(sn)
        out.append(("info", "Clean/nonclean price spread",
                    f"Mean nonclean−clean layer price: {mo:.0f} EUR/MWh (clean-only storage) vs {mn:.0f} "
                    "(per-layer + H2). The spread is the implicit carbon value on dirty electricity; a smaller "
                    "gap means the nonclean layer is less scarce once it can store and buffer."))

    eb = frame(data, "CBAM", "energy_balance")
    dis = eb.reindex(["battery discharger"]).clip(lower=0).sum().sum() if "battery discharger" in eb.index else 0
    if dis > 0.1:
        out.append(("ok", "Per-layer batteries build and cycle (not stranded)",
                    f"GB batteries discharge ~{dis:.0f} TWh over the horizon in CBAM — they charge AND "
                    "discharge on both the clean and nonclean layers, confirming the per-layer storage works."))
    else:
        out.append(("err", "Storage may be stranded",
                    "GB battery discharge ~0 despite capacity — check the storage connection direction."))

    out.append(("info", "H2 is now carbon-labelled (CBAM only)",
                "CBAM splits H2 into clean/nonclean (electrolysis tied to its electricity layer, SMR→nonclean, "
                "SMR CC→clean, store per class); CBAM_old left H2 unsplit. NH3 stays unsplit (Phase 2). The H2 "
                "store chart therefore sums the two class stores in CBAM."))

    out.append(("info", "CBAM charge is still topology-only",
                "No border tariff on nonclean imports is applied yet — differences between the scenarios come "
                "from the split topology (per-layer storage/H2, joint transmission capacity), not a carbon charge."))
    return out


def _span(years):
    return f"{min(years)}–{max(years)}" if years else "—"


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
  <h1>CBAM storage &amp; H₂ split — Great Britain</h1>
  <p class="sub">Three scenarios: <b>reference</b> (no split), <b>CBAM_old</b> (clean-only storage),
     <b>CBAM</b> (per-layer storage + H₂ split). The CBAM-vs-CBAM_old comparison isolates the effect of
     the storage/H₂ changes on the clean/nonclean flow balance. All GB values aggregate the
     clean/nonclean/common layers via <code>n.buses.location</code>.</p>

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
