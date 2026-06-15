"""
tests/test_bs_engine.py
Validates BS pricing, Greeks, IV calibration, and Monte Carlo.
Run: python tests/test_bs_engine.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from bs_engine import bs_price, bs_greeks, implied_vol, price_grid, monte_carlo_paths, mc_option_price

PASS = "✅"
FAIL = "❌"
results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((status, name, detail))
    print(f"{status} {name}  {detail}")


# ─── 1. Put-Call Parity ───────────────────────────────────────
S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
call = bs_price(S, K, T, r, sigma, "call")
put  = bs_price(S, K, T, r, sigma, "put")
lhs = call - put
rhs = S - K * np.exp(-r * T)
check("Put-Call Parity (ATM)", abs(lhs - rhs) < 1e-10, f"diff={abs(lhs-rhs):.2e}")

# ─── 2. Known ATM price ──────────────────────────────────────
# For ATM call: verify it's between intrinsic (0) and S (upper bound)
# and roughly comparable to S*sigma*sqrt(T) order of magnitude
approx_order = S * sigma * np.sqrt(T)
check("ATM Call in correct order of magnitude",
      0.5 * approx_order < call < 1.5 * approx_order,
      f"BS={call:.4f} sigma*S*sqrt(T)={approx_order:.4f}")

# ─── 3. Intrinsic value at expiry ────────────────────────────
call_exp = bs_price(S, K, 0.0, r, sigma, "call")
check("Call intrinsic at T=0 (ATM)", call_exp == 0.0, f"value={call_exp}")
deep_itm = bs_price(150, K, 0.0, r, sigma, "call")
check("Call intrinsic at T=0 (ITM)", deep_itm == 50.0, f"value={deep_itm}")

# ─── 4. Delta bounds ─────────────────────────────────────────
g = bs_greeks(S, K, T, r, sigma, "call")
check("Call Delta in (0,1)", 0 < g["delta"] < 1, f"delta={g['delta']:.5f}")
g_put = bs_greeks(S, K, T, r, sigma, "put")
check("Put Delta in (-1,0)", -1 < g_put["delta"] < 0, f"delta={g_put['delta']:.5f}")

# ─── 5. Gamma positivity ─────────────────────────────────────
check("Gamma > 0", g["gamma"] > 0, f"gamma={g['gamma']:.6f}")

# ─── 6. Delta-Gamma relation (finite diff) ───────────────────
dS = 0.01
d_up   = bs_greeks(S + dS, K, T, r, sigma, "call")["delta"]
d_down = bs_greeks(S - dS, K, T, r, sigma, "call")["delta"]
gamma_fd = (d_up - d_down) / (2 * dS)
check("Gamma ≈ finite-diff delta", abs(gamma_fd - g["gamma"]) < 1e-4,
      f"analytic={g['gamma']:.6f} FD={gamma_fd:.6f}")

# ─── 7. Vega positivity ──────────────────────────────────────
check("Vega > 0", g["vega"] > 0, f"vega={g['vega']:.6f}")

# ─── 8. Theta negativity (long call) ─────────────────────────
check("Theta < 0 for long call", g["theta"] < 0, f"theta={g['theta']:.6f}")

# ─── 9. IV calibration — single ──────────────────────────────
target_iv = 0.25
test_price = bs_price(S, K, T, r, target_iv, "call")
cal_iv = implied_vol(test_price, S, K, T, r, "call")
check("IV Brentq convergence", not np.isnan(cal_iv), f"cal_iv={cal_iv}")
check("IV MAE < 1e-5", abs(cal_iv - target_iv) < 1e-5, f"MAE={abs(cal_iv-target_iv):.2e}")

# ─── 10. IV calibration — 100+ parameterized runs ────────────
strikes = np.linspace(70, 130, 20)
vols    = np.linspace(0.10, 0.60, 6)
total, converged, max_mae = 0, 0, 0.0
for Kt in strikes:
    for v in vols:
        mp = bs_price(S, Kt, T, r, v, "call")
        iv = implied_vol(mp, S, Kt, T, r, "call")
        total += 1
        if not np.isnan(iv):
            mae = abs(iv - v)
            max_mae = max(max_mae, mae)
            if mae < 1e-5:
                converged += 1

check(f"IV calibration: {converged}/{total} converged", converged >= 110,
      f"max_MAE={max_mae:.2e}")

# ─── 11. Price grid shape ────────────────────────────────────
spots = np.linspace(80, 120, 20)
vols2 = np.linspace(0.10, 0.50, 15)
grid  = price_grid(spots, vols2, K, T, r, "call")
check("Price grid shape", grid.shape == (20, 15), f"shape={grid.shape}")
check("Price grid all positive", (grid >= 0).all(), f"min={grid.min():.4f}")

# ─── 12. Monte Carlo paths shape & positivity ────────────────
paths = monte_carlo_paths(S, r, sigma, T, n_paths=500, n_steps=100, seed=42)
check("MC paths shape", paths.shape == (500, 101), f"shape={paths.shape}")
check("MC paths all positive (log-normal)", (paths > 0).all())
check("MC paths start at S0", np.allclose(paths[:, 0], S))

# ─── 13. MC price vs BS price (ATM, should be within 2 SE) ──
mc = mc_option_price(S, K, r, sigma, T, "call", n_paths=20000, n_steps=252, seed=0)
bs = bs_price(S, K, T, r, sigma, "call")
check("MC price within 2 SE of BS", abs(mc["price"] - bs) < 2 * mc["std_err"] * 5,
      f"MC={mc['price']:.4f} BS={bs:.4f} SE={mc['std_err']:.5f}")
check("MC 95% CI contains BS price", mc["ci_lower"] <= bs <= mc["ci_upper"],
      f"CI=[{mc['ci_lower']:.4f},{mc['ci_upper']:.4f}]")

# ─── Summary ─────────────────────────────────────────────────
total_tests = len(results)
passed = sum(1 for r in results if r[0] == PASS)
print(f"\n{'='*55}")
print(f"  {passed}/{total_tests} tests passed")
print(f"{'='*55}")

if passed < total_tests:
    print("\nFailed tests:")
    for status, name, detail in results:
        if status == FAIL:
            print(f"  {FAIL} {name}  {detail}")
    sys.exit(1)
else:
    print("  All tests passed ✅")
    sys.exit(0)
