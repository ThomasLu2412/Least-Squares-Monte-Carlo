# -*- coding: utf-8 -*-
"""
LSMC (Least-Squares Monte Carlo) 
Ultimate Stable Version for Academic Presentation
===============================================
FIXED: Strict isolation of ITM paths during regression and evaluation 
to prevent out-of-domain polynomial explosion.
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. GBM Simulation (with Antithetic Variates)
# ============================================================
def simulate_gbm_paths(S0, r, sigma, T, M, N, seed=42):
    np.random.seed(seed)
    dt = T / M
    paths = np.zeros((M + 1, N))
    paths[0, :] = S0

    if N % 2 != 0:
        raise ValueError("N must be even for antithetic variates.")

    for k in range(1, M + 1):
        Z = np.random.standard_normal(N // 2)
        Z = np.concatenate([Z, -Z])
        paths[k, :] = paths[k-1, :] * np.exp(
            (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        )
    return paths, dt

# ============================================================
# 2. LSMC Pricing (STRICT ITM ISOLATION)
# ============================================================
def lsmc_price(paths, K, r, dt, degree=5):
    M, N = paths.shape[0] - 1, paths.shape[1]
    discount = np.exp(-r * dt)
    
    # Value at maturity
    V = np.maximum(K - paths[-1, :], 0.0)

    for k in range(M - 1, 0, -1):
        S = paths[k, :]
        payoff = np.maximum(K - S, 0.0)
        
        # Boolean mask for In-The-Money paths
        itm = payoff > 0 
        
        if np.sum(itm) > degree + 1:
            # 1. Scale X to [0, 1] for numerical stability
            X = S[itm] / K
            Y = V[itm] * discount
            
            # 2. Fit polynomial ONLY on ITM paths
            coefs = np.polyfit(X, Y, degree)
            
            # 3. Predict continuation value ONLY for ITM paths
            C = np.polyval(coefs, X)
            
            # 4. Exercise decision ONLY for ITM paths
            # Create a global exercise mask initialized to False
            exercise = np.zeros(N, dtype=bool)
            exercise[itm] = payoff[itm] > C
            
            # 5. Update Value matrix
            # If exercised: get immediate payoff. Otherwise: discount future value.
            V = np.where(exercise, payoff, V * discount)
        else:
            # If not enough ITM paths, everyone holds
            V = V * discount

    return np.mean(V * discount)

# ============================================================
# 3. Binomial Tree (True Benchmark)
# ============================================================
def binomial_american_put(S0, K, T, r, sigma, N=2000):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    discount = np.exp(-r * dt)

    S = S0 * (u ** (2 * np.arange(N + 1) - N))
    V = np.maximum(K - S, 0.0)

    for k in range(N - 1, -1, -1):
        S_k = S0 * (u ** (2 * np.arange(k + 1) - k))
        V = discount * (p * V[1:k+2] + (1 - p) * V[:k+1])
        V = np.maximum(V, K - S_k)
    return V[0]

# ============================================================
# 4. Exercise Boundary Extraction
# ============================================================
def exercise_boundary(S0, K, T, r, sigma, M, N, degree=5):
    paths, dt = simulate_gbm_paths(S0, r, sigma, T, M, N, seed=42)
    discount = np.exp(-r * dt)
    V = np.maximum(K - paths[-1, :], 0.0)
    
    boundary = []
    # Evaluate theoretical boundary purely within the ITM region
    S_grid = np.linspace(10, K, 500)
    payoff_grid = K - S_grid

    for k in range(M - 1, 0, -1):
        S = paths[k, :]
        payoff = np.maximum(K - S, 0.0)
        itm = payoff > 0

        if np.sum(itm) > degree + 1:
            X = S[itm] / K
            Y = V[itm] * discount
            coefs = np.polyfit(X, Y, degree)
            
            # Predict on theoretical ITM grid
            cont_grid = np.polyval(coefs, S_grid / K)
            exercise_bool = payoff_grid > cont_grid
            
            if np.any(exercise_bool):
                # The boundary is the highest stock price that triggers exercise
                boundary.append(S_grid[exercise_bool][-1])
            else:
                boundary.append(np.nan)

            # Standard V update
            exercise = np.zeros(N, dtype=bool)
            exercise[itm] = payoff[itm] > np.polyval(coefs, X)
            V = np.where(exercise, payoff, V * discount)
        else:
            boundary.append(np.nan)
            V = V * discount

    boundary.reverse()
    timesteps = np.arange(1, M) * dt
    return timesteps, np.array(boundary)

# ============================================================
# 5. Greeks via Common Random Number Finite Difference
# ============================================================
def compute_greeks(S0, K, T, r, sigma, M, N, degree=5, eps=0.01):
    # Using fixed seed guarantees Common Random Numbers (CRN)
    paths, dt = simulate_gbm_paths(S0, r, sigma, T, M, N, seed=100)
    V0 = lsmc_price(paths, K, r, dt, degree)

    paths_up, _ = simulate_gbm_paths(S0 * (1 + eps), r, sigma, T, M, N, seed=100)
    paths_down, _ = simulate_gbm_paths(S0 * (1 - eps), r, sigma, T, M, N, seed=100)
    V_up = lsmc_price(paths_up, K, r, dt, degree)
    V_down = lsmc_price(paths_down, K, r, dt, degree)
    delta = (V_up - V_down) / (2 * S0 * eps)
    
    paths_up_sig, _ = simulate_gbm_paths(S0, r, sigma * (1 + eps), T, M, N, seed=100)
    paths_down_sig, _ = simulate_gbm_paths(S0, r, sigma * (1 - eps), T, M, N, seed=100)
    V_up_sig = lsmc_price(paths_up_sig, K, r, dt, degree)
    V_down_sig = lsmc_price(paths_down_sig, K, r, dt, degree)
    vega = (V_up_sig - V_down_sig) / (2 * sigma * eps)

    return delta, vega

# ============================================================
# 6. Plot Generation
# ============================================================
def generate_plots():
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    S0, K, T, r, sigma = 36.0, 40.0, 1.0, 0.06, 0.2
    M, degree = 50, 5
    print("Calculating precise binomial benchmark...")
    benchmark_price = binomial_american_put(S0, K, T, r, sigma, 2000)

    # --------------------------------------------------------
    # (a) Convergence of Paths (N)
    # --------------------------------------------------------
    print("Computing Plot (a): N Convergence...")
    N_list = [1000, 5000, 10000, 20000, 50000, 100000]
    prices_N = []
    for n in N_list:
        paths, dt = simulate_gbm_paths(S0, r, sigma, T, M, n, seed=123)
        prices_N.append(lsmc_price(paths, K, r, dt, degree))
        
    axes[0, 0].semilogx(N_list, prices_N, 'bo-', linewidth=2, label='LSMC Price')
    axes[0, 0].axhline(y=benchmark_price, color='r', linestyle='--', label=f'Benchmark ({benchmark_price:.3f})')
    axes[0, 0].set_xlabel('Number of Monte Carlo Paths (N)')
    axes[0, 0].set_ylabel('Option Price')
    axes[0, 0].set_title('(a) Convergence: Path Count (N)')
    axes[0, 0].legend()

    # --------------------------------------------------------
    # (b) Convergence of Basis Degree (d)
    # --------------------------------------------------------
    print("Computing Plot (b): Degree Convergence...")
    degrees = [2, 3, 4, 5, 6, 7, 8]
    prices_d = []
    paths_fixed, dt = simulate_gbm_paths(S0, r, sigma, T, M, 50000, seed=123)
    for d in degrees:
        prices_d.append(lsmc_price(paths_fixed, K, r, dt, d))

    axes[0, 1].plot(degrees, prices_d, 'gs-', linewidth=2, label='LSMC Price')
    axes[0, 1].axhline(y=benchmark_price, color='r', linestyle='--', label='Benchmark')
    axes[0, 1].set_xlabel('Polynomial Degree (d)')
    axes[0, 1].set_ylabel('Option Price')
    axes[0, 1].set_title('(b) Truncation Error: Basis Degree (d)')
    axes[0, 1].set_ylim([4.40, 4.52]) # Narrow y-axis to show extreme stability!
    axes[0, 1].legend()

    # --------------------------------------------------------
    # (c) Exercise Boundary
    # --------------------------------------------------------
    print("Computing Plot (c): Exercise Boundary...")
    t_grid, bd = exercise_boundary(S0, K, T, r, sigma, M=100, N=100000, degree=5)
    axes[1, 0].plot(t_grid, bd, 'r-', linewidth=2.5, label='Optimal Boundary S*(t)')
    axes[1, 0].axhline(y=K, color='k', linestyle=':', label=f'Strike K={K}')
    axes[1, 0].set_xlabel('Time to Maturity (t)')
    axes[1, 0].set_ylabel('Critical Stock Price S*(t)')
    axes[1, 0].set_title('(c) Optimal Early Exercise Boundary')
    axes[1, 0].set_ylim([32, 41])
    axes[1, 0].legend()

    # --------------------------------------------------------
    # (d) Greeks vs S0
    # --------------------------------------------------------
    print("Computing Plot (d): Greeks Profiles...")
    S0_range = np.arange(30, 46, 2)
    deltas, vegas = [], []
    for s0 in S0_range:
        # Use a large N to smooth out finite difference noise
        d, v = compute_greeks(s0, K, T, r, sigma, M=50, N=50000, degree=5, eps=0.01)
        deltas.append(d)
        vegas.append(v)

    ax_d = axes[1, 1]
    color_d = 'tab:blue'
    ax_d.plot(S0_range, deltas, color=color_d, marker='o', linewidth=2)
    ax_d.set_ylabel('Delta', color=color_d, fontweight='bold')
    ax_d.tick_params(axis='y', labelcolor=color_d)
    
    ax_v = ax_d.twinx()
    color_v = 'tab:orange'
    ax_v.plot(S0_range, vegas, color=color_v, marker='s', linewidth=2)
    ax_v.set_ylabel('Vega', color=color_v, fontweight='bold')
    ax_v.tick_params(axis='y', labelcolor=color_v)

    axes[1, 1].set_xlabel('Initial Stock Price (S0)')
    axes[1, 1].set_title('(d) Risk Profiles: Delta & Vega vs S0')

    plt.tight_layout()
    plt.savefig('lsmc_presentation_plots_final.png', dpi=200, bbox_inches='tight')
    print("Success! Plots saved as 'lsmc_presentation_plots_final.png'")
    plt.show()

if __name__ == '__main__':
    generate_plots()