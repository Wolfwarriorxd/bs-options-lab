"""
bs_engine.py
Core Black-Scholes engine: pricing, Greeks, IV calibration, Monte Carlo.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from typing import Literal

OptionType = Literal["call", "put"]


# ─────────────────────────────────────────────
# 1. Black-Scholes Price
# ─────────────────────────────────────────────

def bs_price(S: float, K: float, T: float, r: float, sigma: float, option_type: OptionType) -> float:
    """Black-Scholes closed-form price for European call/put."""
    if T <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
        return float(intrinsic)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return float(price)


# ─────────────────────────────────────────────
# 2. Greeks
# ─────────────────────────────────────────────

def bs_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: OptionType) -> dict:
    """Return all first/second-order Greeks."""
    if T <= 0 or sigma <= 0:
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    pdf_d1 = norm.pdf(d1)

    # Delta
    delta = norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1

    # Gamma (same for call and put)
    gamma = pdf_d1 / (S * sigma * sqrt_T)

    # Vega (per 1% move in vol, divided by 100)
    vega = S * pdf_d1 * sqrt_T / 100

    # Theta (per calendar day)
    if option_type == "call":
        theta = (
            -S * pdf_d1 * sigma / (2 * sqrt_T)
            - r * K * np.exp(-r * T) * norm.cdf(d2)
        ) / 365
    else:
        theta = (
            -S * pdf_d1 * sigma / (2 * sqrt_T)
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
        ) / 365

    # Rho (per 1% move in rates, divided by 100)
    if option_type == "call":
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),
        "rho": float(rho),
    }


# ─────────────────────────────────────────────
# 3. Implied Volatility (Brentq)
# ─────────────────────────────────────────────

def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: OptionType,
    tol: float = 1e-7,
    max_iter: int = 500,
) -> float:
    """Calibrate IV using Brent's method. Returns NaN if calibration fails."""
    if T <= 0 or market_price <= 0:
        return np.nan

    intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
    if market_price < intrinsic:
        return np.nan

    def objective(sigma):
        return bs_price(S, K, T, r, sigma, option_type) - market_price

    try:
        iv = brentq(objective, 1e-6, 10.0, xtol=tol, maxiter=max_iter)
        return float(iv)
    except (ValueError, RuntimeError):
        return np.nan


# ─────────────────────────────────────────────
# 4. IV Surface Grid
# ─────────────────────────────────────────────

def iv_surface(
    S: float,
    r: float,
    strikes: np.ndarray,
    maturities: np.ndarray,
    option_type: OptionType = "call",
    vol_base: float = 0.20,
    skew: float = -0.05,
    smile: float = 0.02,
) -> np.ndarray:
    """
    Simulate a realistic IV surface (SVI-inspired parametric vol surface).
    Returns (len(maturities), len(strikes)) array of IVs.
    """
    surface = np.zeros((len(maturities), len(strikes)))
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            moneyness = np.log(K / S)
            term_adj = np.sqrt(T)
            iv = vol_base + skew * moneyness + smile * moneyness**2 / term_adj
            iv = max(iv, 0.01)
            surface[i, j] = iv
    return surface


# ─────────────────────────────────────────────
# 5. Price Grid (spot × vol)
# ─────────────────────────────────────────────

def price_grid(
    spots: np.ndarray,
    vols: np.ndarray,
    K: float,
    T: float,
    r: float,
    option_type: OptionType,
) -> np.ndarray:
    """Compute BS price over a spot × vol grid. Returns (len(spots), len(vols)) array."""
    grid = np.zeros((len(spots), len(vols)))
    for i, S in enumerate(spots):
        for j, sigma in enumerate(vols):
            grid[i, j] = bs_price(S, K, T, r, sigma, option_type)
    return grid


# ─────────────────────────────────────────────
# 6. Monte Carlo (GBM)
# ─────────────────────────────────────────────

def monte_carlo_paths(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int = 1000,
    n_steps: int = 252,
    seed: int = 42,
) -> np.ndarray:
    """
    Simulate GBM paths. Returns array of shape (n_paths, n_steps+1).
    Uses antithetic variates for variance reduction.
    """
    np.random.seed(seed)
    dt = T / n_steps
    half = n_paths // 2

    Z = np.random.standard_normal((half, n_steps))
    Z = np.vstack([Z, -Z])  # antithetic variates

    log_returns = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.cumsum(log_returns, axis=1)
    paths = S0 * np.exp(np.hstack([np.zeros((n_paths, 1)), log_paths]))

    return paths


def mc_option_price(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: OptionType,
    n_paths: int = 10000,
    n_steps: int = 252,
    seed: int = 42,
) -> dict:
    """Monte Carlo option price with 95% CI."""
    paths = monte_carlo_paths(S0, r, sigma, T, n_paths, n_steps, seed)
    ST = paths[:, -1]

    if option_type == "call":
        payoffs = np.maximum(ST - K, 0)
    else:
        payoffs = np.maximum(K - ST, 0)

    discounted = np.exp(-r * T) * payoffs
    price = discounted.mean()
    stderr = discounted.std() / np.sqrt(n_paths)

    return {
        "price": float(price),
        "ci_lower": float(price - 1.96 * stderr),
        "ci_upper": float(price + 1.96 * stderr),
        "std_err": float(stderr),
    }
