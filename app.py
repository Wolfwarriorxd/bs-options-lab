"""
app.py — Black-Scholes Options Lab
Full suite: pricing, Greeks, IV surface, Monte Carlo, Vol smile
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from bs_engine import (
    bs_price, bs_greeks, implied_vol,
    iv_surface, price_grid,
    monte_carlo_paths, mc_option_price,
)

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="BS Options Lab",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Base ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main .block-container { padding: 2rem 2.5rem 3rem; max-width: 1400px; }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #0d1117 0%, #0f1b2d 50%, #0d1117 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 2.4rem 2.8rem 2rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 70% 50%, rgba(99,179,237,0.06) 0%, transparent 60%);
    pointer-events: none;
}
.hero-badge {
    display: inline-block;
    background: rgba(99,179,237,0.12);
    border: 1px solid rgba(99,179,237,0.3);
    color: #63b3ed;
    font-size: 11px; font-weight: 600; letter-spacing: 2px;
    text-transform: uppercase;
    padding: 4px 12px; border-radius: 20px; margin-bottom: 12px;
}
.hero-title {
    font-size: 2.2rem; font-weight: 700;
    color: #e2e8f0; margin: 0 0 8px;
    line-height: 1.2;
}
.hero-title span { color: #63b3ed; }
.hero-sub {
    color: #718096; font-size: 0.9rem; font-weight: 400;
    margin: 0; letter-spacing: 0.2px;
}
.hero-pills {
    display: flex; gap: 8px; flex-wrap: wrap; margin-top: 16px;
}
.hero-pill {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    color: #a0aec0; font-size: 11px; font-weight: 500;
    padding: 4px 10px; border-radius: 6px;
}

/* ── Metric Cards ── */
.card-row { display: flex; gap: 14px; margin-bottom: 14px; }
.mcard {
    flex: 1;
    background: #0d1117;
    border: 1px solid #1a2332;
    border-radius: 12px;
    padding: 18px 20px 14px;
    position: relative; overflow: hidden;
    transition: border-color 0.2s;
}
.mcard:hover { border-color: #2d4a6e; }
.mcard::after {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #63b3ed, #4299e1);
    border-radius: 12px 12px 0 0;
}
.mcard.green::after { background: linear-gradient(90deg, #48bb78, #38a169); }
.mcard.amber::after { background: linear-gradient(90deg, #f6ad55, #ed8936); }
.mcard.purple::after { background: linear-gradient(90deg, #9f7aea, #805ad5); }
.mcard.red::after   { background: linear-gradient(90deg, #fc8181, #f56565); }
.card-label {
    color: #4a5568; font-size: 10px; font-weight: 600;
    letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 6px;
}
.card-value {
    color: #e2e8f0; font-size: 1.65rem; font-weight: 700;
    font-family: 'JetBrains Mono', monospace; line-height: 1;
    margin-bottom: 6px;
}
.card-sub { color: #4a5568; font-size: 11px; }

/* ── Moneyness badge ── */
.badge {
    display: inline-block;
    padding: 3px 10px; border-radius: 20px;
    font-size: 11px; font-weight: 700; letter-spacing: 1px;
    text-transform: uppercase;
}
.badge-ATM  { background: rgba(99,179,237,0.15); color: #63b3ed; border: 1px solid rgba(99,179,237,0.3); }
.badge-ITM  { background: rgba(72,187,120,0.15); color: #48bb78; border: 1px solid rgba(72,187,120,0.3); }
.badge-OTM  { background: rgba(252,129,129,0.15); color: #fc8181; border: 1px solid rgba(252,129,129,0.3); }

/* ── Greek cards ── */
.greek-row { display: flex; gap: 10px; margin: 4px 0 20px; }
.gcard {
    flex: 1;
    background: #0d1117;
    border: 1px solid #1a2332;
    border-radius: 10px;
    padding: 14px 16px;
    text-align: center;
}
.gcard-name  { color: #4a5568; font-size: 10px; font-weight: 600; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 8px; }
.gcard-value { font-size: 1.1rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; margin-bottom: 4px; }
.gcard-desc  { color: #4a5568; font-size: 10px; }
.pos { color: #48bb78; }
.neg { color: #fc8181; }

/* ── Section headers ── */
.sec-header {
    color: #a0aec0; font-size: 11px; font-weight: 600;
    letter-spacing: 2px; text-transform: uppercase;
    border-bottom: 1px solid #1a2332;
    padding-bottom: 8px; margin: 24px 0 16px;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #080d14;
    border-right: 1px solid #1a2332;
}
section[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }
.sidebar-logo {
    font-size: 1.1rem; font-weight: 700; color: #63b3ed;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 4px;
}
.sidebar-sub { color: #4a5568; font-size: 11px; margin-bottom: 20px; }
.sidebar-section {
    color: #4a5568; font-size: 10px; font-weight: 600;
    letter-spacing: 1.5px; text-transform: uppercase;
    margin: 18px 0 8px; padding-top: 14px;
    border-top: 1px solid #1a2332;
}

/* ── Tab styling ── */
button[data-baseweb="tab"] {
    font-size: 12px !important; font-weight: 600 !important;
    letter-spacing: 0.5px !important;
}

/* ── Dataframe ── */
.stDataFrame { border: 1px solid #1a2332 !important; border-radius: 8px !important; }

/* ── Streamlit overrides ── */
div[data-testid="stMetric"] { background: transparent !important; }
div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">⚡ BS Options Lab</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Black-Scholes Pricing Engine</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Underlying</div>', unsafe_allow_html=True)
    S = st.number_input("Spot Price (S)", value=100.0, min_value=1.0, step=1.0)
    K = st.number_input("Strike Price (K)", value=100.0, min_value=1.0, step=1.0)
    r = st.slider("Risk-Free Rate", 0.0, 0.20, 0.05, 0.001, format="%.3f")

    st.markdown('<div class="sidebar-section">Option</div>', unsafe_allow_html=True)
    T_days = st.slider("Days to Expiry", 1, 730, 90)
    T = T_days / 365
    sigma = st.slider("Volatility (σ)", 0.01, 1.50, 0.20, 0.01, format="%.2f")
    option_type = st.radio("Type", ["call", "put"], horizontal=True)

    st.markdown('<div class="sidebar-section">Monte Carlo</div>', unsafe_allow_html=True)
    n_paths = st.select_slider("Paths", [100, 500, 1000, 5000, 10000], value=1000)
    n_steps = st.slider("Time Steps", 50, 252, 252)
    mc_seed = st.number_input("Seed", value=42, step=1)

    st.markdown('<div class="sidebar-section">About</div>', unsafe_allow_html=True)
    st.caption("European options only · GBM assumption · Not financial advice")

# ─────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">Quantitative Finance</div>
    <div class="hero-title">Black-<span>Scholes</span> Options Lab</div>
    <div class="hero-sub">Real-time European option pricing · Greeks · IV Surface · Monte Carlo simulation</div>
    <div class="hero-pills">
        <span class="hero-pill">📐 Analytic PDE</span>
        <span class="hero-pill">🔬 Brentq IV Calibration</span>
        <span class="hero-pill">🎲 GBM Monte Carlo</span>
        <span class="hero-pill">📊 3D Vol Surface</span>
        <span class="hero-pill">⚡ &lt;1e-5 MAE</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "⚡ Pricer & Greeks",
    "🌡️ Price Heatmap",
    "📐 IV Surface",
    "🎲 Monte Carlo",
    "📈 Vol Smile",
])

# ══════════════════════════════════════════════
# TAB 1 — Pricer & Greeks
# ══════════════════════════════════════════════
with tab1:
    price   = bs_price(S, K, T, r, sigma, option_type)
    greeks  = bs_greeks(S, K, T, r, sigma, option_type)
    mc_snap = mc_option_price(S, K, r, sigma, T, option_type, n_paths=5000, n_steps=n_steps, seed=int(mc_seed))

    intrinsic   = max(S - K, 0) if option_type == "call" else max(K - S, 0)
    time_value  = price - intrinsic
    moneyness   = ("ATM" if abs(S - K) / K < 0.005
                   else ("ITM" if (S > K and option_type == "call") or (S < K and option_type == "put")
                         else "OTM"))
    mc_diff = mc_snap['price'] - price

    st.markdown(f"""
    <div class="card-row">
        <div class="mcard">
            <div class="card-label">Black-Scholes Price</div>
            <div class="card-value">{price:.4f}</div>
            <div class="card-sub">{option_type.upper()} · S={S:.0f} · K={K:.0f} · T={T_days}d</div>
        </div>
        <div class="mcard green">
            <div class="card-label">Monte Carlo Price</div>
            <div class="card-value">{mc_snap['price']:.4f}</div>
            <div class="card-sub">95% CI [{mc_snap['ci_lower']:.3f}, {mc_snap['ci_upper']:.3f}] · diff {mc_diff:+.4f}</div>
        </div>
        <div class="mcard amber">
            <div class="card-label">Time Value</div>
            <div class="card-value">{time_value:.4f}</div>
            <div class="card-sub">Intrinsic {intrinsic:.2f} · <span class="badge badge-{moneyness}">{moneyness}</span></div>
        </div>
        <div class="mcard purple">
            <div class="card-label">Implied Vol (input)</div>
            <div class="card-value">{sigma*100:.1f}%</div>
            <div class="card-sub">r = {r*100:.2f}% · σ = {sigma:.3f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-header">Option Greeks</div>', unsafe_allow_html=True)

    greek_meta = [
        ("Delta", "Δ", "delta", "∂P/∂S"),
        ("Gamma", "Γ", "gamma", "∂²P/∂S²"),
        ("Vega",  "ν", "vega",  "∂P/∂σ (per 1%)"),
        ("Theta", "Θ", "theta", "∂P/∂t (per day)"),
        ("Rho",   "ρ", "rho",   "∂P/∂r (per 1%)"),
    ]

    greek_html = '<div class="greek-row">'
    for name, symbol, key, desc in greek_meta:
        val   = greeks[key]
        cls   = "pos" if val >= 0 else "neg"
        greek_html += f"""
        <div class="gcard">
            <div class="gcard-name">{name} {symbol}</div>
            <div class="gcard-value {cls}">{val:+.5f}</div>
            <div class="gcard-desc">{desc}</div>
        </div>"""
    greek_html += '</div>'
    st.markdown(greek_html, unsafe_allow_html=True)

    st.markdown('<div class="sec-header">Theta Decay — Price Decomposition vs TTM</div>', unsafe_allow_html=True)

    ttm_arr    = np.linspace(0.003, 2.0, 120)
    prices_ttm = [bs_price(S, K, t, r, sigma, option_type) for t in ttm_arr]
    intr_ttm   = float(intrinsic)
    tv_ttm     = [p - intr_ttm for p in prices_ttm]

    fig_decay = go.Figure()
    fig_decay.add_trace(go.Scatter(
        x=ttm_arr * 365, y=prices_ttm, name="Option Price",
        fill="tozeroy", fillcolor="rgba(99,179,237,0.06)",
        line=dict(color="#63b3ed", width=2.5)))
    fig_decay.add_trace(go.Scatter(
        x=ttm_arr * 365, y=tv_ttm, name="Time Value",
        line=dict(color="#9f7aea", width=2, dash="dash")))
    fig_decay.add_hline(y=intr_ttm, line=dict(color="#fc8181", width=1, dash="dot"),
                        annotation_text="Intrinsic", annotation_font_color="#fc8181")
    fig_decay.add_vline(x=T_days,
                        line=dict(color="#f6ad55", width=1.5, dash="dash"),
                        annotation_text=f"Now ({T_days}d)",
                        annotation_font_color="#f6ad55")
    fig_decay.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#080d14",
        xaxis=dict(title="Days to Expiry", gridcolor="#1a2332", zeroline=False),
        yaxis=dict(title="Price", gridcolor="#1a2332", zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        height=360, margin=dict(l=40, r=20, t=10, b=40),
        font=dict(family="Inter", color="#718096"),
    )
    st.plotly_chart(fig_decay, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 2 — Price Heatmap
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="sec-header">Option Price — Spot × Volatility Grid</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        spot_pct = st.slider("Spot range (% of S)", 50, 200, (70, 130))
        n_spot   = st.slider("Spot grid points", 10, 60, 35)
    with c2:
        vol_pct = st.slider("Vol range (%)", 5, 150, (5, 80))
        n_vol   = st.slider("Vol grid points", 10, 60, 35)

    spots = np.linspace(S * spot_pct[0] / 100, S * spot_pct[1] / 100, n_spot)
    vols  = np.linspace(vol_pct[0] / 100, vol_pct[1] / 100, n_vol)
    grid  = price_grid(spots, vols, K, T, r, option_type)

    fig_heat = go.Figure(data=go.Heatmap(
        z=grid, x=np.round(vols * 100, 1), y=np.round(spots, 1),
        colorscale="Viridis",
        colorbar=dict(title=dict(text="Price", font=dict(size=11)), tickfont=dict(size=10)),
        hovertemplate="Vol: %{x:.1f}%<br>Spot: %{y:.1f}<br>Price: %{z:.4f}<extra></extra>",
    ))
    fig_heat.add_scatter(
        x=[sigma * 100], y=[S], mode="markers",
        marker=dict(color="#f6ad55", size=12, symbol="x-thin-open", line=dict(width=3)),
        name="Current params", showlegend=True)
    fig_heat.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#080d14",
        xaxis=dict(title="Implied Volatility (%)", gridcolor="#1a2332"),
        yaxis=dict(title="Spot Price", gridcolor="#1a2332"),
        height=480, margin=dict(l=40, r=20, t=10, b=40),
        font=dict(family="Inter", color="#718096"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown('<div class="sec-header">Greeks Heatmaps</div>', unsafe_allow_html=True)
    gc1, gc2 = st.columns(2)
    for gcol, gkey, gtitle, cscale in [
        (gc1, "delta", "Delta (Δ)", "RdBu"),
        (gc2, "gamma", "Gamma (Γ)", "Plasma"),
    ]:
        gg = np.array([[bs_greeks(s, K, T, r, v, option_type)[gkey] for v in vols] for s in spots])
        fig_g = go.Figure(data=go.Heatmap(
            z=gg, x=np.round(vols * 100, 1), y=np.round(spots, 1),
            colorscale=cscale,
            colorbar=dict(title=dict(text=gtitle, font=dict(size=11)), tickfont=dict(size=10)),
            hovertemplate=f"Vol: %{{x:.1f}}%<br>Spot: %{{y:.1f}}<br>{gtitle}: %{{z:.5f}}<extra></extra>",
        ))
        fig_g.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#080d14",
            xaxis=dict(title="Vol (%)", gridcolor="#1a2332"),
            yaxis=dict(title="Spot", gridcolor="#1a2332"),
            height=360, margin=dict(l=40, r=20, t=30, b=40),
            title=dict(text=gtitle, font=dict(size=13, color="#a0aec0")),
            font=dict(family="Inter", color="#718096"),
        )
        with gcol:
            st.plotly_chart(fig_g, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 3 — IV Surface
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="sec-header">3D Implied Volatility Surface</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        vol_base   = st.slider("Base Vol (%)", 5, 80, 20, key="vb") / 100
        skew       = st.slider("Skew", -0.30, 0.10, -0.05, 0.01)
    with c2:
        smile_curv = st.slider("Smile Curvature", 0.0, 0.20, 0.02, 0.005)
        n_k        = st.slider("Strike grid points", 10, 60, 30)

    k_arr  = np.linspace(S * 0.60, S * 1.40, n_k)
    t_arr  = np.array([0.08, 0.17, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0])
    surf   = iv_surface(S, r, k_arr, t_arr, option_type, vol_base, skew, smile_curv)
    mon_lb = np.round(k_arr / S, 3)
    ttm_lb = np.round(t_arr * 365).astype(int)

    fig_surf = go.Figure(data=go.Surface(
        z=surf * 100, x=mon_lb, y=ttm_lb,
        colorscale="Turbo",
        colorbar=dict(title=dict(text="IV (%)", font=dict(size=11)), tickfont=dict(size=10)),
        hovertemplate="K/S: %{x:.3f}<br>TTM: %{y}d<br>IV: %{z:.2f}%<extra></extra>",
        lighting=dict(ambient=0.7, diffuse=0.6, specular=0.2),
    ))
    fig_surf.update_layout(
        scene=dict(
            xaxis=dict(title="Moneyness K/S", backgroundcolor="#080d14",
                       gridcolor="#1a2332", showbackground=True),
            yaxis=dict(title="TTM (days)", backgroundcolor="#080d14",
                       gridcolor="#1a2332", showbackground=True),
            zaxis=dict(title="IV (%)", backgroundcolor="#080d14",
                       gridcolor="#1a2332", showbackground=True),
            bgcolor="#080d14",
        ),
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        height=560, margin=dict(l=0, r=0, t=10, b=0),
        font=dict(family="Inter", color="#718096"),
    )
    st.plotly_chart(fig_surf, use_container_width=True)

    st.markdown('<div class="sec-header">Brentq IV Calibration — Convergence Audit</div>', unsafe_allow_html=True)
    test_k = np.linspace(S * 0.80, S * 1.20, 20)
    rows = []
    for Kt in test_k:
        true_iv = max(vol_base + skew * np.log(Kt / S) + smile_curv * np.log(Kt / S)**2, 0.01)
        mp      = bs_price(S, Kt, T, r, true_iv, option_type)
        cal_iv  = implied_vol(mp, S, Kt, T, r, option_type)
        mae     = abs(cal_iv - true_iv) if not np.isnan(cal_iv) else np.nan
        rows.append({
            "Strike": f"{Kt:.2f}",
            "K/S": f"{Kt/S:.4f}",
            "True IV": f"{true_iv*100:.4f}%",
            "Calibrated IV": f"{cal_iv*100:.4f}%" if not np.isnan(cal_iv) else "—",
            "MAE": f"{mae:.2e}" if not np.isnan(mae) else "—",
            "Status": "✅" if (not np.isnan(mae) and mae < 1e-5) else "❌",
        })

    df = pd.DataFrame(rows)
    ok = df["Status"].eq("✅").sum()
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Strikes tested", len(rows))
    col_b.metric("Converged", ok)
    col_c.metric("Max MAE", f"{max(float(r['MAE'].replace('e','e')) for r in rows if r['MAE'] != '—'):.2e}" if any(r['MAE'] != '—' for r in rows) else "—")
    st.dataframe(df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════
# TAB 4 — Monte Carlo
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<div class="sec-header">GBM Path Simulation</div>', unsafe_allow_html=True)
    with st.spinner(f"Simulating {n_paths:,} paths..."):
        paths   = monte_carlo_paths(S, r, sigma, T, n_paths=n_paths, n_steps=n_steps, seed=int(mc_seed))
        mc_full = mc_option_price(S, K, r, sigma, T, option_type, n_paths=n_paths, n_steps=n_steps, seed=int(mc_seed))

    t_ax  = np.linspace(0, T * 365, paths.shape[1])
    p5    = np.percentile(paths, 5,  axis=0)
    p25   = np.percentile(paths, 25, axis=0)
    p50   = np.percentile(paths, 50, axis=0)
    p75   = np.percentile(paths, 75, axis=0)
    p95   = np.percentile(paths, 95, axis=0)

    fig_mc = go.Figure()
    show_n = min(150, n_paths)
    for i in range(show_n):
        fig_mc.add_trace(go.Scatter(
            x=t_ax, y=paths[i], mode="lines",
            line=dict(color="rgba(99,179,237,0.08)", width=0.7),
            showlegend=False, hoverinfo="skip"))

    # Bands
    fig_mc.add_trace(go.Scatter(x=t_ax, y=p95, fill=None, mode="lines",
                                 line=dict(color="rgba(0,0,0,0)"), showlegend=False))
    fig_mc.add_trace(go.Scatter(x=t_ax, y=p5, fill="tonexty", mode="lines",
                                 line=dict(color="rgba(0,0,0,0)"),
                                 fillcolor="rgba(72,187,120,0.07)", name="5%–95%"))
    fig_mc.add_trace(go.Scatter(x=t_ax, y=p75, fill=None, mode="lines",
                                 line=dict(color="rgba(0,0,0,0)"), showlegend=False))
    fig_mc.add_trace(go.Scatter(x=t_ax, y=p25, fill="tonexty", mode="lines",
                                 line=dict(color="rgba(0,0,0,0)"),
                                 fillcolor="rgba(246,173,85,0.10)", name="25%–75%"))
    fig_mc.add_trace(go.Scatter(x=t_ax, y=p50, mode="lines",
                                 line=dict(color="#f6ad55", width=2.5), name="Median"))
    fig_mc.add_hline(y=K, line=dict(color="#fc8181", dash="dash", width=1.5),
                      annotation_text=f"Strike {K}", annotation_font_color="#fc8181")

    fig_mc.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#080d14",
        xaxis=dict(title="Days", gridcolor="#1a2332", zeroline=False),
        yaxis=dict(title="Spot Price", gridcolor="#1a2332", zeroline=False),
        height=440, margin=dict(l=40, r=20, t=10, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        font=dict(family="Inter", color="#718096"),
    )
    st.plotly_chart(fig_mc, use_container_width=True)

    bs_p = bs_price(S, K, T, r, sigma, option_type)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MC Price",  f"{mc_full['price']:.5f}")
    m2.metric("BS Price",  f"{bs_p:.5f}", delta=f"{mc_full['price'] - bs_p:+.5f}")
    m3.metric("95% CI",    f"[{mc_full['ci_lower']:.4f}, {mc_full['ci_upper']:.4f}]")
    m4.metric("Std Error", f"{mc_full['std_err']:.6f}")

    st.markdown('<div class="sec-header">Terminal Price Distribution</div>', unsafe_allow_html=True)
    ST = paths[:, -1]
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=ST, nbinsx=70, name="S(T)",
        marker=dict(color="rgba(99,179,237,0.5)", line=dict(color="rgba(99,179,237,0.8)", width=0.5))))
    fig_dist.add_vline(x=K,        line=dict(color="#fc8181", dash="dash", width=2),
                        annotation_text=f"K={K}", annotation_font_color="#fc8181")
    fig_dist.add_vline(x=ST.mean(), line=dict(color="#f6ad55", dash="dot", width=2),
                        annotation_text=f"μ={ST.mean():.1f}", annotation_font_color="#f6ad55")
    fig_dist.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#080d14",
        xaxis=dict(title="Terminal Spot S(T)", gridcolor="#1a2332", zeroline=False),
        yaxis=dict(title="Frequency", gridcolor="#1a2332", zeroline=False),
        height=320, margin=dict(l=40, r=20, t=10, b=40),
        font=dict(family="Inter", color="#718096"),
        showlegend=False,
    )
    st.plotly_chart(fig_dist, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 5 — Vol Smile
# ══════════════════════════════════════════════
with tab5:
    st.markdown('<div class="sec-header">Volatility Smile — IV Cross-Section by TTM</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        smile_ttms = st.multiselect("TTMs (days)", [7, 14, 30, 60, 90, 180, 365], default=[30, 90, 180])
        n_smile_k  = st.slider("Strike grid", 10, 80, 50)
    with c2:
        s_base  = st.slider("Base Vol (%)", 5, 80, 20, key="sb2") / 100
        s_skew  = st.slider("Skew", -0.30, 0.10, -0.05, 0.01, key="sk2")
        s_curv  = st.slider("Curvature", 0.0, 0.20, 0.03, 0.005, key="sc2")

    k_smile  = np.linspace(S * 0.60, S * 1.40, n_smile_k)
    mon_smile = k_smile / S
    COLORS = ["#63b3ed", "#48bb78", "#f6ad55", "#fc8181", "#9f7aea", "#ed64a6", "#4fd1c5"]

    fig_smile = go.Figure()
    for i, ttm_d in enumerate(smile_ttms):
        T_s = ttm_d / 365
        ivs = [max(s_base + s_skew * np.log(Ks / S) + s_curv * np.log(Ks / S)**2 / np.sqrt(T_s), 0.01) * 100
               for Ks in k_smile]
        fig_smile.add_trace(go.Scatter(
            x=mon_smile, y=ivs, mode="lines+markers",
            name=f"{ttm_d}d",
            line=dict(color=COLORS[i % len(COLORS)], width=2.5),
            marker=dict(size=4, color=COLORS[i % len(COLORS)]),
        ))

    fig_smile.add_vline(x=1.0, line=dict(color="rgba(255,255,255,0.15)", dash="dash"),
                         annotation_text="ATM", annotation_font_color="#4a5568")
    fig_smile.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#080d14",
        xaxis=dict(title="Moneyness K/S", gridcolor="#1a2332", zeroline=False),
        yaxis=dict(title="Implied Volatility (%)", gridcolor="#1a2332", zeroline=False),
        height=420, margin=dict(l=40, r=20, t=10, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        font=dict(family="Inter", color="#718096"),
    )
    st.plotly_chart(fig_smile, use_container_width=True)

    st.markdown('<div class="sec-header">Volatility Term Structure — ATM IV vs TTM</div>', unsafe_allow_html=True)
    t_ts   = np.linspace(0.05, 2.0, 100)
    atm_iv = [max(s_base, 0.01) * 100] * len(t_ts)   # ATM: log(K/S)=0, curvature term=0

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=t_ts * 365, y=atm_iv, mode="lines",
        fill="tozeroy", fillcolor="rgba(159,122,234,0.06)",
        line=dict(color="#9f7aea", width=2.5), name="ATM IV"))
    fig_ts.add_vline(x=T_days, line=dict(color="#f6ad55", dash="dash", width=1.5),
                      annotation_text=f"{T_days}d", annotation_font_color="#f6ad55")
    fig_ts.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#080d14",
        xaxis=dict(title="Days to Expiry", gridcolor="#1a2332", zeroline=False),
        yaxis=dict(title="ATM IV (%)", gridcolor="#1a2332", zeroline=False),
        height=300, margin=dict(l=40, r=20, t=10, b=40),
        font=dict(family="Inter", color="#718096"), showlegend=False,
    )
    st.plotly_chart(fig_ts, use_container_width=True)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; color:#2d3748; font-size:11px; margin-top:40px; padding-top:20px;
     border-top: 1px solid #1a2332; font-family:'Inter',sans-serif; letter-spacing:0.5px;">
    Black-Scholes Options Lab &nbsp;·&nbsp; European options only &nbsp;·&nbsp;
    GBM assumption &nbsp;·&nbsp; Not financial advice
</div>
""", unsafe_allow_html=True)
