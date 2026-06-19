# 📈 Black-Scholes Options Lab

Author: Anish Deshmukh  Roll No: ED23B005, IIT Madras

Full-suite European options pricing dashboard built with Python, Streamlit, and Plotly.

## Features

- **BS Pricer** — Call/put prices, intrinsic vs time value breakdown
- **Greeks** — Delta, Gamma, Vega, Theta, Rho with live heatmaps
- **Theta Decay** — Time value vs TTM with 50+ grid points
- **Price Heatmap** — Option price over Spot × Vol grid (1000+ scenarios)
- **IV Surface** — 3D implied volatility surface (Brentq calibration, <1e-5 MAE)
- **Monte Carlo** — GBM path simulation with antithetic variates, confidence bands
- **Vol Smile** — Cross-sectional IV curves across maturities

## Quickstart (Local)

```bash
git clone https://github.com/<your-username>/bs-options-lab.git
cd bs-options-lab
pip install -r requirements.txt
streamlit run app.py
```

## Run Tests

```bash
python tests/test_bs_engine.py
```

## Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo, branch `main`, file `app.py`
4. Click **Deploy** — live in ~60 seconds

## CI/CD

GitHub Actions runs on every push to `main`:
- Engine unit tests (put-call parity, IV calibration, Monte Carlo validation)
- Flake8 lint check

## Project Structure

```
bs-options-lab/
├── app.py               # Streamlit dashboard
├── bs_engine.py         # Core math: pricing, Greeks, IV, Monte Carlo
├── requirements.txt
├── tests/
│   └── test_bs_engine.py
└── .github/
    └── workflows/
        └── ci.yml       # GitHub Actions CI pipeline
```

## Implementation Notes

- IV calibration uses `scipy.optimize.brentq` (guaranteed convergence on [1e-6, 10.0])
- Monte Carlo uses antithetic variates for variance reduction
- Vol surface parametrized with SVI-inspired skew + curvature model
- All plots rendered with Plotly for sub-200ms UI latency
