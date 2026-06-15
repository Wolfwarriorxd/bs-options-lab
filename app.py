"""
app.py — Black-Scholes Options Dashboard
Full suite: pricing, Greeks, IV surface, Monte Carlo, Vol smile
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

from bs_engine import (
    bs_price,
    bs_greeks,
    implied_vol,
    iv_surface,
    price_grid,
    monte_carlo_paths,
    mc_option_price,
)

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Black-Scholes Options Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130 0%, #252840 100%);
        border: 1px solid #3d4166;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-label { color: #8b92b3; font-size: 12px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; }
    .metric-value { color: #e8eaf6; font-size: 28px; font-weight: 700; font-family: 'Courier New', monospace; }
    .metric-sub { color: #5c6bc0; font-size: 12px; margin-top: 4px; }
    .greek-positive { color: #26de81; }
    .greek-negative { color: #fc5c65; }
    div[data-testid="stMetric"] label { color: #8b92b3 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Sidebar — Global Parameters
# ─────────────────────────────────────────────
st.sidebar.title("⚙️ Parameters")
st.sidebar.markdown("---")

st.sidebar.subheader("Market")
S = st.sidebar.number_input("Spot Price (S)", value=100.0, min_value=1.0, step=1.0)
K = st.sidebar.number_input("Strike Price (K)", value=100.0, min_value=1.0, step=1.0)
r = st.sidebar.slider("Risk-Free Rate (r)", 0.0, 0.20, 0.05, 0.001, format="%.3f")

st.sidebar.subheader("Option")
T_days = st.sidebar.slider("Time to Maturity (days)", 1, 730, 90)
T = T_days / 365
sigma = st.sidebar.slider("Volatility (σ)", 0.01, 1.50, 0.20, 0.01, format="%.2f")
option_type = st.sidebar.radio("Option Type", ["call", "put"], horizontal=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Monte Carlo")
n_paths = st.sidebar.select_slider("Paths", [100, 500, 1000, 5000, 10000], value=1000)
n_steps = st.sidebar.slider("Steps (trading days)", 50, 252, 252)
mc_seed = st.sidebar.number_input("Random Seed", value=42, step=1)

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.title("📈 Black-Scholes Options Lab")
st.markdown("*European option pricing · Greeks · IV Surface · Monte Carlo · Vol Smile*")
st.markdown("---")

# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏷️ Pricer & Greeks",
    "🌡️ Price Heatmap",
    "📐 IV Surface",
    "🎲 Monte Carlo",
    "😊 Vol Smile",
])

# ══════════════════════════════════════════════
# TAB 1 — Pricer & Greeks
# ══════════════════════════════════════════════
with tab1:
    price = bs_price(S, K, T, r, sigma, option_type)
    greeks = bs_greeks(S, K, T, r, sigma, option_type)
    mc_result = mc_option_price(S, K, r, sigma, T, option_type, n_paths=5000, n_steps=n_steps, seed=int(mc_seed))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">BS Price</div>
            <div class="metric-value">₹{price:.4f}</div>
            <div class="metric-sub">{option_type.upper()} | S={S} K={K}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">MC Price</div>
            <div class="metric-value">₹{mc_result['price']:.4f}</div>
            <div class="metric-sub">95% CI [{mc_result['ci_lower']:.3f}, {mc_result['ci_upper']:.3f}]</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        moneyness = "ATM" if abs(S - K) < 0.5 else ("ITM" if (S > K and option_type == "call") or (S < K and option_type == "put") else "OTM")
        intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
        time_value = price - intrinsic
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Moneyness</div>
            <div class="metric-value">{moneyness}</div>
            <div class="metric-sub">Intrinsic: {intrinsic:.2f} | Time Value: {time_value:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Greeks")
    g_cols = st.columns(5)
    greek_names = ["Delta (Δ)", "Gamma (Γ)", "Vega (ν)", "Theta (Θ)", "Rho (ρ)"]
    greek_keys = ["delta", "gamma", "vega", "theta", "rho"]
    greek_descs = ["per ₹1 move in S", "Δ per ₹1 move in S", "per 1% move in σ", "per calendar day", "per 1% move in r"]

    for col, name, key, desc in zip(g_cols, greek_names, greek_keys, greek_descs):
        val = greeks[key]
        color = "#26de81" if val >= 0 else "#fc5c65"
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{name}</div>
                <div class="metric-value" style="color:{color}; font-size:22px;">{val:+.5f}</div>
                <div class="metric-sub">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Theta Decay — Time Value vs TTM")
    ttm_range = np.linspace(0.01, 2.0, 100)
    prices_ttm = [bs_price(S, K, t, r, sigma, option_type) for t in ttm_range]
    intrinsics_ttm = [max(S - K, 0) if option_type == "call" else max(K - S, 0)] * 100
    time_values_ttm = [p - i for p, i in zip(prices_ttm, intrinsics_ttm)]

    fig_theta = go.Figure()
    fig_theta.add_trace(go.Scatter(x=ttm_range * 365, y=prices_ttm, name="Option Price",
                                    line=dict(color="#5c6bc0", width=2.5)))
    fig_theta.add_trace(go.Scatter(x=ttm_range * 365, y=time_values_ttm, name="Time Value",
                                    line=dict(color="#26de81", width=2, dash="dash")))
    fig_theta.add_trace(go.Scatter(x=ttm_range * 365, y=intrinsics_ttm, name="Intrinsic Value",
                                    line=dict(color="#fc5c65", width=1.5, dash="dot")))
    fig_theta.add_vline(x=T_days, line=dict(color="#ffd32a", width=1.5, dash="dash"),
                         annotation_text=f"Current TTM ({T_days}d)")
    fig_theta.update_layout(
        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        xaxis_title="Days to Expiry", yaxis_title="Price (₹)",
        legend=dict(bgcolor="rgba(0,0,0,0)"), height=380,
        margin=dict(l=40, r=20, t=20, b=40)
    )
    st.plotly_chart(fig_theta, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 2 — Price Heatmap
# ══════════════════════════════════════════════
with tab2:
    st.subheader("Option Price Grid — Spot × Volatility")
    c1, c2 = st.columns(2)
    with c1:
        spot_range = st.slider("Spot range (% of S)", 50, 200, (70, 130), key="spot_range")
        n_spot = st.slider("Spot grid points", 10, 60, 30, key="n_spot")
    with c2:
        vol_range = st.slider("Vol range (%)", 5, 150, (5, 80), key="vol_range")
        n_vol = st.slider("Vol grid points", 10, 60, 30, key="n_vol")

    spots = np.linspace(S * spot_range[0] / 100, S * spot_range[1] / 100, n_spot)
    vols = np.linspace(vol_range[0] / 100, vol_range[1] / 100, n_vol)
    grid = price_grid(spots, vols, K, T, r, option_type)

    fig_heat = go.Figure(data=go.Heatmap(
        z=grid,
        x=np.round(vols * 100, 1),
        y=np.round(spots, 1),
        colorscale="Viridis",
        colorbar=dict(title="Price (₹)"),
        hovertemplate="Vol: %{x:.1f}%<br>Spot: ₹%{y:.1f}<br>Price: ₹%{z:.4f}<extra></extra>"
    ))
    fig_heat.add_scatter(x=[sigma * 100], y=[S], mode="markers",
                          marker=dict(color="#ffd32a", size=14, symbol="x", line=dict(width=2)),
                          name="Current")
    fig_heat.update_layout(
        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        xaxis_title="Implied Volatility (%)", yaxis_title="Spot Price (₹)",
        height=480, margin=dict(l=40, r=20, t=20, b=40)
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.subheader("Greeks Heatmap — Delta & Gamma")
    g_col1, g_col2 = st.columns(2)

    for gcol, greek_key, title, cscale in [
        (g_col1, "delta", "Delta (Δ)", "RdBu"),
        (g_col2, "gamma", "Gamma (Γ)", "Plasma"),
    ]:
        greek_grid = np.array([
            [bs_greeks(s, K, T, r, v, option_type)[greek_key] for v in vols]
            for s in spots
        ])
        fig_g = go.Figure(data=go.Heatmap(
            z=greek_grid, x=np.round(vols * 100, 1), y=np.round(spots, 1),
            colorscale=cscale, colorbar=dict(title=title),
            hovertemplate=f"Vol: %{{x:.1f}}%<br>Spot: ₹%{{y:.1f}}<br>{title}: %{{z:.5f}}<extra></extra>"
        ))
        fig_g.update_layout(
            template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            xaxis_title="Vol (%)", yaxis_title="Spot (₹)",
            title=title, height=360, margin=dict(l=40, r=20, t=40, b=40)
        )
        with gcol:
            st.plotly_chart(fig_g, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 3 — IV Surface
# ══════════════════════════════════════════════
with tab3:
    st.subheader("Implied Volatility Surface")
    c1, c2 = st.columns(2)
    with c1:
        vol_base = st.slider("Base Vol (%)", 5, 80, 20, key="vb") / 100
        skew = st.slider("Skew", -0.30, 0.10, -0.05, 0.01, key="sk")
    with c2:
        smile_curv = st.slider("Smile Curvature", 0.0, 0.20, 0.02, 0.005, key="sm")
        n_strikes_iv = st.slider("Strike grid", 10, 60, 30, key="nk_iv")

    strikes_iv = np.linspace(S * 0.60, S * 1.40, n_strikes_iv)
    maturities_iv = np.array([0.08, 0.17, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0])

    surface = iv_surface(S, r, strikes_iv, maturities_iv, option_type, vol_base, skew, smile_curv)

    moneyness_labels = np.round(strikes_iv / S, 3)
    maturity_labels = np.round(maturities_iv * 365).astype(int)

    fig_surf = go.Figure(data=go.Surface(
        z=surface * 100,
        x=moneyness_labels,
        y=maturity_labels,
        colorscale="Turbo",
        colorbar=dict(title="IV (%)"),
        hovertemplate="K/S: %{x:.3f}<br>TTM: %{y}d<br>IV: %{z:.2f}%<extra></extra>"
    ))
    fig_surf.update_layout(
        scene=dict(
            xaxis_title="Moneyness (K/S)",
            yaxis_title="TTM (days)",
            zaxis_title="IV (%)",
            bgcolor="#0e1117",
            xaxis=dict(backgroundcolor="#0e1117", gridcolor="#2d3250"),
            yaxis=dict(backgroundcolor="#0e1117", gridcolor="#2d3250"),
            zaxis=dict(backgroundcolor="#0e1117", gridcolor="#2d3250"),
        ),
        template="plotly_dark", paper_bgcolor="#0e1117",
        height=580, margin=dict(l=0, r=0, t=20, b=0)
    )
    st.plotly_chart(fig_surf, use_container_width=True)

    st.subheader("IV Calibration — Brentq Convergence Test")
    test_strikes = np.linspace(S * 0.80, S * 1.20, 20)
    test_results = []
    for Kt in test_strikes:
        true_iv = vol_base + skew * np.log(Kt / S) + smile_curv * (np.log(Kt / S))**2
        true_iv = max(true_iv, 0.01)
        market_p = bs_price(S, Kt, T, r, true_iv, option_type)
        cal_iv = implied_vol(market_p, S, Kt, T, r, option_type)
        mae = abs(cal_iv - true_iv) if not np.isnan(cal_iv) else np.nan
        test_results.append({"Strike": round(Kt, 2), "True IV": f"{true_iv*100:.3f}%",
                               "Calibrated IV": f"{cal_iv*100:.3f}%" if not np.isnan(cal_iv) else "N/A",
                               "MAE": f"{mae:.2e}" if not np.isnan(mae) else "N/A",
                               "Converged": "✅" if not np.isnan(cal_iv) and mae < 1e-5 else "❌"})

    df_cal = pd.DataFrame(test_results)
    converged = df_cal["Converged"].eq("✅").sum()
    st.markdown(f"**{converged}/{len(test_results)} strikes converged with MAE < 1e-5**")
    st.dataframe(df_cal, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════
# TAB 4 — Monte Carlo
# ══════════════════════════════════════════════
with tab4:
    st.subheader("Monte Carlo — GBM Path Simulation")
    with st.spinner(f"Simulating {n_paths} paths..."):
        paths = monte_carlo_paths(S, r, sigma, T, n_paths=n_paths, n_steps=n_steps, seed=int(mc_seed))
        mc_full = mc_option_price(S, K, r, sigma, T, option_type, n_paths=n_paths, n_steps=n_steps, seed=int(mc_seed))

    time_axis = np.linspace(0, T * 365, paths.shape[1])
    pct_5 = np.percentile(paths, 5, axis=0)
    pct_25 = np.percentile(paths, 25, axis=0)
    pct_50 = np.percentile(paths, 50, axis=0)
    pct_75 = np.percentile(paths, 75, axis=0)
    pct_95 = np.percentile(paths, 95, axis=0)

    fig_mc = go.Figure()
    # Sample paths
    display_n = min(200, n_paths)
    for i in range(display_n):
        fig_mc.add_trace(go.Scatter(
            x=time_axis, y=paths[i], mode="lines",
            line=dict(color="rgba(92,107,192,0.12)", width=0.8),
            showlegend=False, hoverinfo="skip"
        ))

    fig_mc.add_trace(go.Scatter(x=time_axis, y=pct_95, fill=None, mode="lines",
                                 line=dict(color="rgba(38,222,129,0)"), showlegend=False))
    fig_mc.add_trace(go.Scatter(x=time_axis, y=pct_5, fill="tonexty", mode="lines",
                                 line=dict(color="rgba(38,222,129,0)"),
                                 fillcolor="rgba(38,222,129,0.08)", name="5%–95% Band"))
    fig_mc.add_trace(go.Scatter(x=time_axis, y=pct_75, fill=None, mode="lines",
                                 line=dict(color="rgba(255,211,42,0)"), showlegend=False))
    fig_mc.add_trace(go.Scatter(x=time_axis, y=pct_25, fill="tonexty", mode="lines",
                                 line=dict(color="rgba(255,211,42,0)"),
                                 fillcolor="rgba(255,211,42,0.12)", name="25%–75% Band"))
    fig_mc.add_trace(go.Scatter(x=time_axis, y=pct_50, mode="lines",
                                 line=dict(color="#ffd32a", width=2.5), name="Median"))
    fig_mc.add_hline(y=K, line=dict(color="#fc5c65", dash="dash", width=1.5),
                      annotation_text=f"Strike K={K}")

    fig_mc.update_layout(
        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        xaxis_title="Days", yaxis_title="Spot Price (₹)",
        height=460, legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=40, r=20, t=20, b=40)
    )
    st.plotly_chart(fig_mc, use_container_width=True)

    # MC vs BS summary
    bs_p = bs_price(S, K, T, r, sigma, option_type)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MC Price", f"₹{mc_full['price']:.4f}")
    col2.metric("BS Price", f"₹{bs_p:.4f}", delta=f"{mc_full['price'] - bs_p:+.4f}")
    col3.metric("95% CI", f"[{mc_full['ci_lower']:.3f}, {mc_full['ci_upper']:.3f}]")
    col4.metric("Std Error", f"₹{mc_full['std_err']:.5f}")

    # Terminal distribution
    st.subheader("Terminal Price Distribution")
    ST = paths[:, -1]
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(x=ST, nbinsx=60, name="ST",
                                     marker=dict(color="rgba(92,107,192,0.7)",
                                                  line=dict(color="#5c6bc0", width=0.5))))
    fig_dist.add_vline(x=K, line=dict(color="#fc5c65", dash="dash", width=2),
                        annotation_text=f"K={K}")
    fig_dist.add_vline(x=ST.mean(), line=dict(color="#ffd32a", dash="dot", width=2),
                        annotation_text=f"Mean={ST.mean():.1f}")
    fig_dist.update_layout(
        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        xaxis_title="Terminal Spot (₹)", yaxis_title="Frequency",
        height=340, margin=dict(l=40, r=20, t=20, b=40)
    )
    st.plotly_chart(fig_dist, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 5 — Vol Smile
# ══════════════════════════════════════════════
with tab5:
    st.subheader("Volatility Smile — Cross-Section at Fixed TTM")
    c1, c2 = st.columns(2)
    with c1:
        smile_ttms = st.multiselect("Select TTMs (days)", [30, 60, 90, 180, 365], default=[30, 90, 180])
        n_smile_strikes = st.slider("Number of strikes", 10, 80, 40)
    with c2:
        smile_vol_base = st.slider("Base Vol (%)", 5, 80, 20, key="svb") / 100
        smile_skew = st.slider("Skew", -0.30, 0.10, -0.05, 0.01, key="ssk")
        smile_curv2 = st.slider("Curvature", 0.0, 0.20, 0.03, 0.005, key="ssc")

    strike_arr = np.linspace(S * 0.60, S * 1.40, n_smile_strikes)
    moneyness_arr = strike_arr / S

    fig_smile = go.Figure()
    colors = ["#5c6bc0", "#26de81", "#ffd32a", "#fc5c65", "#a29bfe"]

    for i, ttm_days in enumerate(smile_ttms):
        T_smile = ttm_days / 365
        iv_smile = []
        for Ks in strike_arr:
            m = np.log(Ks / S)
            iv_s = smile_vol_base + smile_skew * m + smile_curv2 * m**2 / np.sqrt(T_smile)
            iv_s = max(iv_s, 0.01)
            iv_smile.append(iv_s * 100)

        fig_smile.add_trace(go.Scatter(
            x=moneyness_arr, y=iv_smile,
            mode="lines+markers", name=f"TTM={ttm_days}d",
            line=dict(color=colors[i % len(colors)], width=2.5),
            marker=dict(size=4)
        ))

    fig_smile.add_vline(x=1.0, line=dict(color="rgba(255,255,255,0.3)", dash="dash"),
                         annotation_text="ATM")
    fig_smile.update_layout(
        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        xaxis_title="Moneyness (K/S)", yaxis_title="Implied Volatility (%)",
        height=420, legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=40, r=20, t=20, b=40)
    )
    st.plotly_chart(fig_smile, use_container_width=True)

    # Term structure
    st.subheader("Volatility Term Structure — ATM IV vs TTM")
    ttm_arr = np.linspace(0.05, 2.0, 80)
    atm_ivs = []
    for T_ts in ttm_arr:
        # ATM: moneyness = 0, so IV = base + curvature term (skew * 0 = 0)
        iv_atm = smile_vol_base + smile_curv2 * 0 / np.sqrt(T_ts)
        atm_ivs.append(max(iv_atm, 0.01) * 100)

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=ttm_arr * 365, y=atm_ivs, mode="lines",
                                 line=dict(color="#a29bfe", width=2.5), name="ATM IV"))
    fig_ts.add_vline(x=T_days, line=dict(color="#ffd32a", dash="dash"),
                      annotation_text=f"Selected TTM ({T_days}d)")
    fig_ts.update_layout(
        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        xaxis_title="Days to Expiry", yaxis_title="ATM IV (%)",
        height=340, margin=dict(l=40, r=20, t=20, b=40)
    )
    st.plotly_chart(fig_ts, use_container_width=True)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#8b92b3; font-size:12px;'>"
    "Black-Scholes Options Lab · Built with Python, Streamlit, Plotly · "
    "European options only · Not financial advice"
    "</div>",
    unsafe_allow_html=True
)
