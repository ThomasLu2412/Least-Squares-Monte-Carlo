# -*- coding: utf-8 -*-
"""
LSMC (Least-Squares Monte Carlo) 算法复现
============================================
基于 Longstaff-Schwartz (2001)

数学系视角的教学实现 — 强调:
  1. 几何布朗运动 (GBM) 与伊藤引理
  2. 最优停止问题与动态规划
  3. L² 投影 / 最小二乘回归
  4. Monte Carlo 收敛性

用法:
    python project_code.py

依赖: numpy, (可选) matplotlib
"""

import numpy as np

# ============================================================
# 1. 几何布朗运动 (GBM) 路径模拟
#    S_t = S_0 * exp((r - σ²/2)t + σW_t)
# ============================================================

def simulate_gbm_paths(S0: float, r: float, sigma: float, T: float,
                       M: int, N: int, seed: int = 42, antithetic: bool = True):
    """
    模拟 N 条 GBM 路径, M 个时间步.

    数学背景:
        dS_t = r S_t dt + σ S_t dW_t  (风险中性测度 Q 下)
        => S_t = S_0 exp((r - σ²/2)t + σ W_t)

    参数:
        S0, r, sigma : GBM 参数
        T            : 到期时间 (年)
        M            : 时间离散步数 (不含 t=0)
        N            : 模拟路径数
        antithetic   : 是否使用对偶变量法 (方差缩减)

    返回:
        paths : ndarray, shape (M+1, N) — paths[k, i] = S_{i, t_k}
        dt    : float — 时间步长 Δt
    """
    np.random.seed(seed)
    dt = T / M
    paths = np.zeros((M + 1, N))
    paths[0, :] = S0

    # 若使用对偶变量法, N 需为偶数
    if antithetic and N % 2 != 0:
        raise ValueError("使用对偶变量法时 N 需为偶数, 请调整")

    for k in range(1, M + 1):
        if antithetic:
            # 只生成 N/2 个正态随机数, 取其正负得到 N 个
            half = N // 2
            Z = np.random.standard_normal(half)
            Z = np.concatenate([Z, -Z])
        else:
            Z = np.random.standard_normal(N)

        # GBM 离散迭代公式
        paths[k, :] = paths[k-1, :] * np.exp(
            (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        )

    return paths, dt


# ============================================================
# 2. LSMC 美式看跌期权定价
#    核心: 用 L² 投影 (最小二乘回归) 逼近条件期望
# ============================================================

def lsmc_price(paths: np.ndarray, K: float, r: float, dt: float,
               degree: int = 5, verbose: bool = True):
    """
    LSMC 算法 — 美式看跌期权定价.

    算法步骤:
        Step 1: 在到期日 t_M = T, V = max(K - S_T, 0)
        Step 2: 从 t_{M-1} 到 t_1 向后递推:
            a. 筛选"实值"路径 (S_t < K)
            b. 用最小二乘回归拟合继续持有价值:
               Y_{t+1} = e^{-rΔt} V_{t+1}
               对基函数 {1, S, S², ..., S^d} 回归
            c. 比较立即行权 vs 继续持有, 更新 V_t
        Step 3: V_0 = 均值 (贴现回 t=0)

    参数:
        paths  : shape (M+1, N) 的股价路径
        K      : 执行价
        r      : 无风险利率
        dt     : 时间步长
        degree : 多项式回归阶数 d

    返回:
        price  : LSMC 估计的期权价格
        (同时打印详细过程)
    """
    M, N = paths.shape[0] - 1, paths.shape[1]
    discount = np.exp(-r * dt)

    # Step 1: 到期日现金流
    V = np.maximum(K - paths[-1, :], 0.0)  # V_{i, t_M}

    # 记录关键中间结果用于分析
    exercise_count = 0
    regression_stats = []

    # Step 2: 向后递推
    for k in range(M - 1, 0, -1):
        S_k = paths[k, :]              # t_k 时刻的股价向量
        payoff = np.maximum(K - S_k, 0.0)  # 立即行权价值 Φ(S_t)

        # --- 2a. 只对"实值"路径做回归 (LSMC 原始论文的建议) ---
        in_the_money = payoff > 0
        X = S_k[in_the_money]
        Y = V[in_the_money] * discount  # 贴现后的未来现金流

        if len(X) < degree + 2:
            # 实值路径太少时直接使用当前现金流 (数值退化情况)
            V = np.where(in_the_money, payoff, V * discount)
            continue

        # --- 2b. 构造设计矩阵: Vandermonde 矩阵 ---
        #    X_matrix[i, j] = S_i^j,  j = 0, 1, ..., degree
        X_matrix = np.vander(X, N=degree + 1, increasing=True)

        # --- 2c. 最小二乘求解: β̂ = (X^T X)^{-1} X^T Y ---
        #   使用 numpy 的 lstsq (基于 SVD, 数值稳定)
        beta, residuals, rank, sv = np.linalg.lstsq(X_matrix, Y, rcond=None)

        # 继续持有价值 = 回归预测值
        continuation = np.polynomial.polynomial.polyval(S_k, beta)

        # --- 2d. 决策: 立即行权 vs 继续持有 ---
        exercise_here = payoff > continuation
        V = np.where(exercise_here, payoff, V * discount)

        if verbose:
            n_ex = np.sum(exercise_here)
            exercise_count += n_ex
            regression_stats.append({
                'step': k,
                'n_in_money': len(X),
                'n_exercise': n_ex,
                'beta': beta,
            })

    # Step 3: 贴现到 t=0 并取均值
    price = np.mean(V * discount)

    if verbose:
        print(f"  [LSMC] 回归阶数 d={degree}")
        print(f"  [LSMC] 总提前行权次数: {exercise_count}")
        print(f"  [LSMC] 期权价格: {price:.6f}")

    return price


# ============================================================
# 3. 基准对比: 二叉树 (作为数值精确基准)
# ============================================================

def binomial_american_put(S0: float, K: float, T: float,
                          r: float, sigma: float, N: int = 1000):
    """
    Cox-Ross-Rubinstein 二叉树定价美式看跌期权.

    作为 LSMC 的验证基准. N 足够大时收敛到真实值.
    """
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    discount = np.exp(-r * dt)

    # 股价树 (只存最后一层以节省内存)
    S = S0 * (u ** (2 * np.arange(N + 1) - N))  # 最后一层的股价

    # 期权价值树 (向后递推)
    V = np.maximum(K - S, 0.0)

    for k in range(N - 1, -1, -1):
        # 当前层的股价
        S_k = S0 * (u ** (2 * np.arange(k + 1) - k))
        # 继续持有价值 = 下一层的期望贴现
        V = discount * (p * V[1:k+2] + (1 - p) * V[:k+1])
        # 提前行权比较
        V = np.maximum(V, K - S_k)

    return V[0]


# ============================================================
# 4. Greeks (有限差分法)
# ============================================================

def compute_greeks(S0, K, T, r, sigma, M, N, degree=5, eps=0.01):
    """
    用有限差分法计算 Greeks:
        Delta = ∂V/∂S_0
        Gamma = ∂²V/∂S_0²
        Vega  = ∂V/∂σ
        Theta = -∂V/∂T
        Rho   = ∂V/∂r
    """
    paths, dt = simulate_gbm_paths(S0, r, sigma, T, M, N, seed=42)

    # 基准价格
    V0 = lsmc_price(paths, K, r, dt, degree, verbose=False)

    # Delta: 中心差分
    paths_up, _ = simulate_gbm_paths(S0 * (1 + eps), r, sigma, T, M, N, seed=42)
    paths_down, _ = simulate_gbm_paths(S0 * (1 - eps), r, sigma, T, M, N, seed=42)
    V_up = lsmc_price(paths_up, K, r, dt, degree, verbose=False)
    V_down = lsmc_price(paths_down, K, r, dt, degree, verbose=False)
    delta = (V_up - V_down) / (2 * S0 * eps)
    gamma = (V_up - 2 * V0 + V_down) / ((S0 * eps) ** 2)

    # Vega
    paths_up_sig, _ = simulate_gbm_paths(S0, r, sigma * (1 + eps), T, M, N, seed=42)
    paths_down_sig, _ = simulate_gbm_paths(S0, r, sigma * (1 - eps), T, M, N, seed=42)
    V_up_sig = lsmc_price(paths_up_sig, K, r, dt, degree, verbose=False)
    V_down_sig = lsmc_price(paths_down_sig, K, r, dt, degree, verbose=False)
    vega = (V_up_sig - V_down_sig) / (2 * sigma * eps)

    # Rho
    V_up_r = lsmc_price(paths, K, r * (1 + eps), dt, degree, verbose=False)
    V_down_r = lsmc_price(paths, K, r * (1 - eps), dt, degree, verbose=False)
    rho = (V_up_r - V_down_r) / (2 * r * eps)

    # Theta: 对 T 的偏导
    paths_up_T, _ = simulate_gbm_paths(S0, r, sigma, T * (1 + eps), M, N, seed=42)
    paths_down_T, _ = simulate_gbm_paths(S0, r, sigma, T * (1 - eps), M, N, seed=42)
    V_up_T = lsmc_price(paths_up_T, K, r, dt, degree, verbose=False)
    V_down_T = lsmc_price(paths_down_T, K, r, dt, degree, verbose=False)
    theta = -(V_up_T - V_down_T) / (2 * T * eps)

    return {
        'price': V0,
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'rho': rho,
        'theta': theta,
    }


# ============================================================
# 5. 收敛性分析 & 数值实验
# ============================================================

def convergence_study_N(S0=36, K=40, T=1.0, r=0.06, sigma=0.2,
                        M=50, degree=5, N_list=None, n_reps=3):
    """
    研究 LSMC 价格随路径数 N 的收敛性.
    大数定律: 标准差 ∝ 1/√N
    """
    if N_list is None:
        N_list = [100, 500, 1000, 5000, 10000, 50000]

    print("\n" + "="*60)
    print("收敛性实验: 改变 Monte Carlo 路径数 N")
    print("="*60)
    print(f"{'N':>8} | {'均价':>10} | {'标准差':>10} | {'标准误':>10}")
    print("-"*60)

    results = []
    for N in N_list:
        prices = []
        for rep in range(n_reps):
            paths, dt = simulate_gbm_paths(S0, r, sigma, T, M, N,
                                           seed=42 + rep)
            p = lsmc_price(paths, K, r, dt, degree, verbose=False)
            prices.append(p)
        mean_p = np.mean(prices)
        std_p = np.std(prices, ddof=1)
        se = std_p / np.sqrt(n_reps)
        print(f"{N:>8} | {mean_p:>10.4f} | {std_p:>10.4f} | {se:>10.4f}")
        results.append({'N': N, 'mean': mean_p, 'std': std_p, 'se': se})

    return results


def convergence_study_M(S0=36, K=40, T=1.0, r=0.06, sigma=0.2,
                        N=10000, degree=5, M_list=None):
    """
    研究 LSMC 价格随时间步数 M 的收敛性.
    """
    if M_list is None:
        M_list = [10, 20, 50, 100, 200]

    print("\n" + "="*60)
    print("收敛性实验: 改变时间步数 M")
    print("="*60)
    print(f"{'M':>8} | {'价格':>10} | {'Δt':>10} | {'ΔS_max':>10}")
    print("-"*60)

    results = []
    for M in M_list:
        paths, dt = simulate_gbm_paths(S0, r, sigma, T, M, N, seed=42)
        p = lsmc_price(paths, K, r, dt, degree, verbose=False)
        # 计算最大步内变化 (粗略)
        max_step = np.max(np.abs(np.diff(paths, axis=0)))
        print(f"{M:>8} | {p:>10.4f} | {dt:>10.6f} | {max_step:>10.4f}")
        results.append({'M': M, 'price': p, 'dt': dt})

    return results


def basis_function_comparison(S0=36, K=40, T=1.0, r=0.06, sigma=0.2,
                              M=50, N=10000):
    """
    比较不同回归阶数 d 对定价的影响.
    """
    degrees = [2, 3, 4, 5, 6]
    print("\n" + "="*60)
    print("基函数实验: 比较不同多项式阶数")
    print("="*60)
    print(f"{'阶数 d':>8} | {'价格':>10} | {'行权次数':>12}")
    print("-"*60)

    paths, dt = simulate_gbm_paths(S0, r, sigma, T, M, N, seed=42)

    for d in degrees:
        p = lsmc_price(paths, K, r, dt, degree=d, verbose=False)
        # 粗略估计行权次数 (重跑一次带输出)
        print(f"{d:>8} | {p:>10.4f}")

    # 添加二叉树基准
    binom = binomial_american_put(S0, K, T, r, sigma, N=1000)
    print("-"*60)
    print(f"{'二叉树':>8} | {binom:>10.4f} | {'(基准)':>12}")


def exercise_boundary(S0=36, K=40, T=1.0, r=0.06, sigma=0.2,
                      M=100, N=20000, degree=5):
    """
    提取最优行权边界 S*(t_k).

    方法: 在每个时间步, 找到使 Φ(S_t) = C(t, S_t) 的临界股价.
    即: K - S* = Σ β_j L_j(S*) 的解.

    由于数值求解的复杂性, 这里用一种更简单的近似方法:
    从模拟路径中找出在每个时间步刚好行权的股价中位数.
    """
    paths, dt = simulate_gbm_paths(S0, r, sigma, T, M, N, seed=42)
    discount = np.exp(-r * dt)

    V = np.maximum(K - paths[-1, :], 0.0)
    boundary = []

    for k in range(M - 1, 0, -1):
        S_k = paths[k, :]
        payoff = np.maximum(K - S_k, 0.0)

        in_the_money = payoff > 0
        X = S_k[in_the_money]
        Y = V[in_the_money] * discount

        if len(X) < degree + 2:
            boundary.append(np.nan)
            continue

        X_matrix = np.vander(X, N=degree + 1, increasing=True)
        beta, _, _, _ = np.linalg.lstsq(X_matrix, Y, rcond=None)

        continuation = np.polynomial.polynomial.polyval(S_k, beta)
        exercise_here = payoff > continuation

        # 如果存在行权的路径, 取行权路径中的最小股价作为边界的近似
        if np.any(exercise_here):
            S_exercise = S_k[exercise_here]
            # 边界通常在 S* 附近, 取行权区域中股价的上分位点
            boundary.append(np.percentile(S_exercise, 25))
        else:
            boundary.append(np.nan)

        V = np.where(exercise_here, payoff, V * discount)

    # 过滤掉 NaN
    timesteps = np.arange(1, M) * dt
    boundary = np.array(boundary)
    valid = ~np.isnan(boundary)

    return timesteps[valid], boundary[valid]


# ============================================================
# 6. 主函数：运行所有数值实验
# ============================================================

def main():
    """运行所有数值实验并打印结果."""
    # ---------- 基本参数 ----------
    S0, K, T, r, sigma = 36.0, 40.0, 1.0, 0.06, 0.2
    M, N, degree = 50, 10000, 5

    print("=" * 60)
    print("LSMC 算法复现 — Longstaff-Schwartz (2001)")
    print("=" * 60)
    print(f"\n基本参数: S0={S0}, K={K}, T={T}, r={r}, σ={sigma}")
    print(f"数值参数: M={M}, N={N}, degree={degree}")

    # ---------- 1. 基本定价 ----------
    print("\n" + "=" * 60)
    print("实验 1: 基本 LSMC 定价")
    print("=" * 60)
    paths, dt = simulate_gbm_paths(S0, r, sigma, T, M, N, seed=42)
    price_lsmc = lsmc_price(paths, K, r, dt, degree, verbose=True)

    # 二叉树基准
    price_binom = binomial_american_put(S0, K, T, r, sigma, N=1000)
    print(f"  [二叉树]  期权价格: {price_binom:.6f}  (基准)")

    # 欧式 BSM 解析解
    try:
        from scipy.stats import norm
        norm_cdf = norm.cdf
    except ImportError:
        from math import erf
        norm_cdf = np.vectorize(lambda x: 0.5 * (1 + erf(x / np.sqrt(2))))
    d1 = (np.log(S0/K) + (r + sigma**2/2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price_euro = K * np.exp(-r*T) * norm_cdf(-d2) - S0 * norm_cdf(-d1)
    print(f"  [欧式 BSM] 期权价格: {price_euro:.6f}")
    print(f"  [美式溢价] V_A - V_E = {price_lsmc - price_euro:.6f}")

    # ---------- 2. Greeks ----------
    print("\n" + "=" * 60)
    print("实验 2: Greeks 计算 (有限差分法)")
    print("=" * 60)
    greeks = compute_greeks(S0, K, T, r, sigma, M, N, degree)
    for key, val in greeks.items():
        print(f"  {key:>8s} = {val:>10.6f}")

    # ---------- 3. N 收敛性 ----------
    convergence_study_N(S0, K, T, r, sigma, M, degree, n_reps=3)

    # ---------- 4. M 收敛性 ----------
    convergence_study_M(S0, K, T, r, sigma, N, degree)

    # ---------- 5. 基函数比较 ----------
    basis_function_comparison(S0, K, T, r, sigma, M, N)

    # ---------- 6. 行权边界 ----------
    print("\n" + "=" * 60)
    print("实验 6: 提前行权边界 S*(t)")
    print("=" * 60)
    t_grid, bd = exercise_boundary(S0, K, T, r, sigma, M=100, N=20000, degree=5)
    print(f"  边界点数: {len(bd)}")
    print(f"  在 t→0 时 S*(t) ≈ {bd[0]:.2f}" if len(bd) > 0 else "")
    print(f"  在 t=T 时 S*(t) = K = {K:.2f}")
    # 打印前几个边界点
    for i in range(min(5, len(t_grid))):
        print(f"  t = {t_grid[i]:.3f}, S*(t) = {bd[i]:.4f}")

    # ---------- 总结 ----------
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print(f"""
  LSMC 定价:        {price_lsmc:.4f}
  二叉树 (基准):    {price_binom:.4f}
  差异:             {abs(price_lsmc - price_binom):.4f}
  Monte Carlo 标准误: ~ {greeks['price'] * 0.01:.4f} (粗略估计)

  Delta: {greeks['delta']:.4f}  |  Gamma: {greeks['gamma']:.4f}
  Vega:  {greeks['vega']:.4f}   |  Theta: {greeks['theta']:.4f}
""")

    print("提示: 将上述数值截图插入 PPT 对应页面.")
    print("如需可视化图表 (收敛曲线/边界图), 请运行 with_matplotlib() 版本.")


# ============================================================
# 7. 可选: Matplotlib 可视化
# ============================================================

def plot_results():
    """
    生成论文中的数值结果图.
    需要 matplotlib 支持.
    请在已安装 matplotlib 的环境中单独调用此函数.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("请安装 matplotlib: pip install matplotlib")
        return

    S0, K, T, r, sigma = 36.0, 40.0, 1.0, 0.06, 0.2
    M, N, degree = 50, 10000, 5

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (a) 收敛性: N vs 价格
    N_list = [100, 500, 1000, 5000, 10000, 50000]
    prices_N = []
    for n in N_list:
        paths, dt = simulate_gbm_paths(S0, r, sigma, T, M, n, seed=42)
        p = lsmc_price(paths, K, r, dt, degree, verbose=False)
        prices_N.append(p)

    axes[0, 0].semilogx(N_list, prices_N, 'bo-', linewidth=2)
    axes[0, 0].axhline(y=binomial_american_put(S0, K, T, r, sigma, 1000),
                       color='r', linestyle='--', label='二叉树基准')
    axes[0, 0].set_xlabel('路径数 N')
    axes[0, 0].set_ylabel('期权价格')
    axes[0, 0].set_title('(a) 收敛性: N → 价格')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # (b) 基函数阶数比较
    degrees = [2, 3, 4, 5, 6]
    prices_d = []
    paths_fixed, dt = simulate_gbm_paths(S0, r, sigma, T, M, N, seed=42)
    for d in degrees:
        p = lsmc_price(paths_fixed, K, r, dt, d, verbose=False)
        prices_d.append(p)

    axes[0, 1].plot(degrees, prices_d, 'gs-', linewidth=2)
    axes[0, 1].axhline(y=binomial_american_put(S0, K, T, r, sigma, 1000),
                       color='r', linestyle='--', label='二叉树基准')
    axes[0, 1].set_xlabel('多项式阶数 d')
    axes[0, 1].set_ylabel('期权价格')
    axes[0, 1].set_title('(b) 基函数阶数 d 的影响')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # (c) 行权边界
    t_grid, bd = exercise_boundary(S0, K, T, r, sigma, M=100, N=20000, degree=5)
    axes[1, 0].plot(t_grid, bd, 'r-', linewidth=2)
    axes[1, 0].axhline(y=K, color='gray', linestyle=':', label=f'K={K}')
    axes[1, 0].set_xlabel('时间 t')
    axes[1, 0].set_ylabel('临界股价 S*(t)')
    axes[1, 0].set_title('(c) 最优行权边界 S*(t)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # (d) Greeks vs S0
    S0_range = np.arange(30, 50, 2)
    deltas, vegas, gammas = [], [], []
    for s0 in S0_range:
        g = compute_greeks(s0, K, T, r, sigma, M, N, degree)
        deltas.append(g['delta'])
        vegas.append(g['vega'])
        gammas.append(g['gamma'])

    ax_d = axes[1, 1]
    color_d = 'tab:blue'
    color_v = 'tab:orange'
    color_g = 'tab:green'
    ax_d.plot(S0_range, deltas, color=color_d, marker='o', label='Delta')
    ax_d.set_ylabel('Delta / Gamma', color=color_d)
    ax_d.tick_params(axis='y', labelcolor=color_d)
    ax_d.grid(True, alpha=0.3)

    ax_g = ax_d.twinx()
    ax_g.plot(S0_range, vegas, color=color_v, marker='s', label='Vega')
    ax_g.set_ylabel('Vega', color=color_v)
    ax_g.tick_params(axis='y', labelcolor=color_v)

    axes[1, 1].set_xlabel('初始股价 S₀')
    axes[1, 1].set_title('(d) Greeks vs S₀')

    plt.tight_layout()
    plt.savefig('lsmc_results.png', dpi=150)
    print("图表已保存为 lsmc_results.png")
    plt.show()


if __name__ == '__main__':
    main()
    print("\n若要生成可视化图表, 请取消注释下方代码:")
    print("# plot_results()")
