import os
import numpy as np
from scipy.stats import gaussian_kde
import plotly.graph_objects as go
import plotly.offline as pyo

from src.data import demo_series
from src.regimes import (
    fit_regimes,
    labels_to_series,
    transition_matrix,
    distance_matrix,
    embed_2d,
    silhouette,
    elbow_scan,
)
from src.wkmeans import MomentKMeans
from src.mmd import cluster_mmd


BG = "#0A0E17"
PANEL = "#111726"
INK = "#E7ECF4"
SUB = "#93A1B5"
FAINT = "#5C6B82"
LINE = "#1E2A3E"
GRID = "#182338"
REGIME = ["#2DD4BF", "#FBBF24", "#FB7185", "#A78BFA", "#38BDF8"]
REGIME_NAME = ["Calm", "Choppy", "Stress", "Regime D", "Regime E"]

K = 3
W = 21
STEP = 21


def apply_dark(fig, height):
    fig.update_layout(
        height=height,
        margin=dict(l=48, r=24, t=16, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono, monospace", size=12, color=SUB),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=SUB, size=11),
                    orientation="h", yanchor="bottom", y=1.02, x=0),
        hoverlabel=dict(bgcolor=PANEL, bordercolor=LINE,
                        font=dict(family="JetBrains Mono, monospace", color=INK)),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=LINE,
                     tickcolor=LINE, color=SUB)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=LINE,
                     tickcolor=LINE, color=SUB)
    return fig


def contiguous_segments(series):
    segments = []
    start = 0
    for i in range(1, len(series)):
        if series[i] != series[start]:
            segments.append((start, i - 1, int(series[start])))
            start = i
    segments.append((start, len(series) - 1, int(series[start])))
    return segments


def fig_timeline(dates, prices, series):
    fig = go.Figure()
    for a, b, lab in contiguous_segments(series):
        fig.add_vrect(x0=dates[a], x1=dates[b], fillcolor=REGIME[lab],
                      opacity=0.14, line_width=0, layer="below")
    fig.add_trace(go.Scatter(
        x=dates, y=prices, mode="lines",
        line=dict(color=INK, width=1.6),
        hovertemplate="%{x|%b %d, %Y}<br>%{y:.2f}<extra></extra>", name="Price"))
    for lab in range(K):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                 marker=dict(size=10, color=REGIME[lab], symbol="square"),
                                 name=REGIME_NAME[lab]))
    apply_dark(fig, 320)
    fig.update_yaxes(title_text="Index level")
    return fig


def fig_ridgeline(returns, window_labels, windows, starts):
    per_cluster = {c: [] for c in range(K)}
    for lab, s in zip(window_labels, starts):
        per_cluster[int(lab)].append(returns[s:s + W])
    pooled = {c: np.concatenate(v) if v else np.array([0.0]) for c, v in per_cluster.items()}
    lo = np.percentile(returns, 1)
    hi = np.percentile(returns, 99)
    xs = np.linspace(lo, hi, 300)
    fig = go.Figure()
    offset = 0.0
    step = 1.0
    ticks = []
    for c in range(K):
        data = pooled[c]
        kde = gaussian_kde(data)
        dens = kde(xs)
        dens = dens / dens.max()
        base = offset
        fig.add_trace(go.Scatter(
            x=xs, y=dens + base, mode="lines",
            line=dict(color=REGIME[c], width=1.8),
            fill="tonexty" if c > 0 else "tozeroy",
            fillcolor=_rgba(REGIME[c], 0.28),
            name=REGIME_NAME[c],
            hovertemplate="return %{x:.3f}<extra>" + REGIME_NAME[c] + "</extra>"))
        ticks.append((base + 0.4, REGIME_NAME[c]))
        offset += step
    apply_dark(fig, 320)
    fig.update_xaxes(title_text="Daily log-return", tickformat=".2%")
    fig.update_yaxes(tickvals=[t[0] for t in ticks],
                     ticktext=[t[1] for t in ticks], title_text="")
    fig.update_layout(showlegend=False)
    return fig


def _rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def fig_heatmap(dmat):
    scale = [[0.0, "#0A0E17"], [0.35, "#0E3A44"], [0.65, "#128597"],
             [0.85, "#22D3EE"], [1.0, "#A78BFA"]]
    fig = go.Figure(go.Heatmap(
        z=dmat, colorscale=scale, showscale=True,
        colorbar=dict(outlinecolor=LINE, tickcolor=LINE, tickfont=dict(color=SUB, size=10),
                      thickness=10, len=0.85),
        hovertemplate="w%{x} vs w%{y}<br>W2 = %{z:.4f}<extra></extra>"))
    apply_dark(fig, 320)
    fig.update_xaxes(title_text="Window", showgrid=False)
    fig.update_yaxes(title_text="Window", showgrid=False, autorange="reversed")
    return fig


def fig_mds(coords, window_labels):
    fig = go.Figure()
    for c in range(K):
        m = window_labels == c
        fig.add_trace(go.Scatter(
            x=coords[m, 0], y=coords[m, 1], mode="markers",
            marker=dict(size=11, color=REGIME[c], line=dict(color=BG, width=1),
                        opacity=0.9),
            name=REGIME_NAME[c],
            hovertemplate=REGIME_NAME[c] + "<extra></extra>"))
    apply_dark(fig, 320)
    fig.update_xaxes(title_text="MDS-1", showgrid=True, showticklabels=False)
    fig.update_yaxes(title_text="MDS-2", showgrid=True, showticklabels=False)
    return fig


def fig_transitions(T):
    labels = REGIME_NAME[:K]
    text = [[f"{T[i, j]*100:.0f}%" for j in range(K)] for i in range(K)]
    scale = [[0.0, "#0A0E17"], [0.5, "#134E4A"], [1.0, "#2DD4BF"]]
    fig = go.Figure(go.Heatmap(
        z=T, x=labels, y=labels, colorscale=scale, showscale=False,
        text=text, texttemplate="%{text}",
        textfont=dict(family="JetBrains Mono", color=INK, size=13),
        hovertemplate="from %{y} to %{x}<br>%{z:.2f}<extra></extra>"))
    apply_dark(fig, 320)
    fig.update_xaxes(title_text="To", showgrid=False, side="bottom")
    fig.update_yaxes(title_text="From", showgrid=False, autorange="reversed")
    return fig


def fig_comparison(wk, moment):
    cats = ["Between-cluster", "Within-cluster"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=cats, y=[wk["between"], wk["within"]],
                         marker_color="#22D3EE", name="WK-means",
                         hovertemplate="%{y:.4f}<extra>WK-means</extra>"))
    fig.add_trace(go.Bar(x=cats, y=[moment["between"], moment["within"]],
                         marker_color=FAINT, name="Moment k-means",
                         hovertemplate="%{y:.4f}<extra>Moment</extra>"))
    apply_dark(fig, 320)
    fig.update_yaxes(title_text="MMD")
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.12)
    return fig


def fig_kselect(ks, inertias, sils):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ks, y=inertias, mode="lines+markers",
                             line=dict(color="#22D3EE", width=2),
                             marker=dict(size=8, color="#22D3EE"),
                             name="Inertia", yaxis="y"))
    fig.add_trace(go.Scatter(x=ks, y=sils, mode="lines+markers",
                             line=dict(color="#FBBF24", width=2, dash="dot"),
                             marker=dict(size=8, color="#FBBF24"),
                             name="Silhouette", yaxis="y2"))
    fig.add_vline(x=K, line=dict(color=REGIME[2], width=1.5, dash="dash"))
    apply_dark(fig, 320)
    fig.update_xaxes(title_text="k", dtick=1)
    fig.update_yaxes(title_text="Inertia")
    fig.update_layout(yaxis2=dict(title="Silhouette", overlaying="y", side="right",
                                  gridcolor="rgba(0,0,0,0)", color=SUB))
    return fig


def card(kicker, title, div):
    return f"""
    <div class="card">
      <div class="card-head"><span class="kick">{kicker}</span><span class="ctitle">{title}</span></div>
      <div class="plot">{div}</div>
    </div>"""


def metric(value, label, accent=INK):
    return f"""
    <div class="metric">
      <div class="mval" style="color:{accent}">{value}</div>
      <div class="mlab">{label}</div>
    </div>"""


def build():
    dates, prices, returns, truth = demo_series()
    model, windows, starts = fit_regimes(returns, w=W, step=STEP, k=K, random_state=0)
    wlabels = model.labels_
    series = labels_to_series(wlabels, starts, W, returns.size)

    T = transition_matrix(wlabels, K)
    dmat = distance_matrix(windows, p=2)
    coords = embed_2d(dmat)
    sil = silhouette(dmat, wlabels)

    wk_mmd = cluster_mmd(windows, wlabels)
    mom = MomentKMeans(k=K, random_state=0).fit(windows)
    mom_mmd = cluster_mmd(windows, mom.labels_)

    ks, inertias = elbow_scan(windows, [2, 3, 4, 5, 6], random_state=0)
    sils = [silhouette(dmat, fit_regimes(returns, w=W, step=STEP, k=kk, random_state=0)[0].labels_)
            for kk in ks]

    uplift = wk_mmd["ratio"] / mom_mmd["ratio"] if mom_mmd["ratio"] > 0 else float("inf")

    figs = {
        "timeline": fig_timeline(dates, prices, series),
        "ridge": fig_ridgeline(returns, wlabels, windows, starts),
        "heat": fig_heatmap(dmat),
        "mds": fig_mds(coords, wlabels),
        "trans": fig_transitions(T),
        "cmp": fig_comparison(wk_mmd, mom_mmd),
        "ksel": fig_kselect(ks, inertias, sils),
    }
    config = {"displayModeBar": False, "responsive": True}
    divs = {name: pyo.plot(f, include_plotlyjs=False, output_type="div", config=config)
            for name, f in figs.items()}

    plotlyjs = pyo.get_plotlyjs()

    metrics = "".join([
        metric(f"{len(windows)}", "Windows clustered"),
        metric(f"{K}", "Regimes detected", REGIME[0]),
        metric(f"{wk_mmd['ratio']:.1f}x", "WK-means separation", "#22D3EE"),
        metric(f"{mom_mmd['ratio']:.1f}x", "Moment baseline", FAINT),
        metric(f"{sil:.2f}", "Silhouette (WK)", "#FBBF24"),
        metric(f"{uplift:.1f}x", "WK uplift vs baseline", REGIME[0]),
    ])

    body = f"""
    <div class="topbar">
      <div>
        <div class="kick">QUANT-LAB // P04 // WASSERSTEIN REGIME CLUSTERING</div>
        <div class="h1">Market Regime Clustering</div>
        <div class="sub">Wasserstein k-means on empirical return distributions</div>
      </div>
      <div class="badge">SYNTHETIC DEMO SERIES</div>
    </div>
    <div class="metrics">{metrics}</div>
    <div class="full">{card("HERO // PRICE x REGIME", "Regime-shaded timeline", divs["timeline"])}</div>
    <div class="grid2">
      {card("DISTRIBUTIONS", "Per-cluster return ridgelines", divs["ridge"])}
      {card("OPTIMAL TRANSPORT", "Wasserstein distance matrix", divs["heat"])}
    </div>
    <div class="grid2">
      {card("EMBEDDING", "2D regime map (MDS)", divs["mds"])}
      {card("DYNAMICS", "Regime transition matrix", divs["trans"])}
    </div>
    <div class="grid2">
      {card("VALIDATION", "WK-means vs moment baseline", divs["cmp"])}
      {card("MODEL SELECTION", "k by inertia and silhouette", divs["ksel"])}
    </div>
    <div class="foot">
      <span>Method: Horvath, Issa &amp; Muguruza (2021), arXiv:2110.11848.</span>
      <span>Distances via 1D optimal transport; centroids are Wasserstein barycenters.</span>
      <span class="warn">Not financial advice.</span>
    </div>"""

    html = TEMPLATE.replace("__PLOTLYJS__", plotlyjs).replace("__BODY__", body)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(root, "output", "P04_regime_dashboard.html")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print("windows", len(windows), "sil", round(sil, 3),
          "wk", round(wk_mmd["ratio"], 2), "mom", round(mom_mmd["ratio"], 2),
          "uplift", round(uplift, 2))
    print("wrote", out)


TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>P04 - Market Regime Clustering</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Sora:wght@300;400;600;800&display=swap" rel="stylesheet">
<script>__PLOTLYJS__</script>
<style>
  :root{
    --bg:#0A0E17; --panel:#111726; --panel2:#0E1420; --ink:#E7ECF4;
    --sub:#93A1B5; --faint:#5C6B82; --line:#1E2A3E; --cyan:#22D3EE;
  }
  *{box-sizing:border-box;}
  body{margin:0;background:var(--bg);color:var(--ink);
       font-family:'Sora',sans-serif;padding:32px 28px 48px;}
  .kick{font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:1.5px;
        color:var(--cyan);text-transform:uppercase;}
  .h1{font-family:'Sora',sans-serif;font-weight:800;font-size:34px;margin-top:6px;letter-spacing:-0.5px;}
  .sub{color:var(--sub);font-size:15px;font-weight:300;margin-top:2px;}
  .topbar{display:flex;justify-content:space-between;align-items:flex-start;
          border-bottom:1px solid var(--line);padding-bottom:22px;margin-bottom:24px;}
  .badge{font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:1.5px;
         color:var(--cyan);border:1px solid var(--line);border-radius:999px;
         padding:7px 14px;background:var(--panel);align-self:center;}
  .metrics{display:grid;grid-template-columns:repeat(6,1fr);gap:14px;margin-bottom:24px;}
  .metric{background:var(--panel);border:1px solid var(--line);border-radius:14px;
          padding:18px 16px;}
  .mval{font-family:'JetBrains Mono',monospace;font-size:26px;font-weight:700;line-height:1;}
  .mlab{color:var(--faint);font-size:11px;margin-top:8px;letter-spacing:0.3px;
        font-family:'JetBrains Mono',monospace;text-transform:uppercase;}
  .full{margin-bottom:18px;}
  .grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:18px;}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:16px;
        padding:18px 18px 10px;box-shadow:0 12px 40px rgba(0,0,0,0.35);}
  .card-head{display:flex;flex-direction:column;gap:4px;margin-bottom:6px;}
  .ctitle{font-family:'Sora',sans-serif;font-weight:600;font-size:16px;color:var(--ink);}
  .plot{width:100%;}
  .foot{display:flex;gap:22px;flex-wrap:wrap;border-top:1px solid var(--line);
        margin-top:14px;padding-top:18px;color:var(--faint);
        font-family:'JetBrains Mono',monospace;font-size:11px;}
  .warn{color:#FB7185;margin-left:auto;}
  @media(max-width:900px){.metrics{grid-template-columns:repeat(2,1fr);}
    .grid2{grid-template-columns:1fr;}}
</style>
</head>
<body>__BODY__</body>
</html>"""


if __name__ == "__main__":
    build()
