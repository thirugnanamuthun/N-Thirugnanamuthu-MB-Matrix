"""
Universal Algebraic Matrix Framework - Advanced, verified version
-----------------------------------------------------------------
Model:  X(t+1) = A_eff(u) X(t) + noise,   A_eff(u) = colnorm( A + sum_k u_k T_k )

* A  : column-stochastic transition matrix (each column sums to 1)
* T  : impact tensor, T[:,:,k] = influence matrix of actor k
* X  : state distribution on the simplex (sum = 1)

Math tools: Perron-Frobenius, spectral gap, mixing time, stationary
distribution, actor sensitivity, Monte Carlo (stochastic) simulation.
This is a TOY MODEL with assumed parameters, not a forecast of real people.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
LABELS = ["Decision power", "Economy", "AI growth", "Human mindset", "Environment"]
ACTORS = ["Tech leader", "Investor", "Social activist"]
n, m = 5, 3


def colnorm(M):
    M = np.clip(M, 1e-12, None)
    return M / M.sum(axis=0, keepdims=True)


# ---------- Step 1: state vector (on the simplex) ----------
X0 = np.array([0.70, 0.60, 0.85, 0.50, 0.40])
X0 = X0 / X0.sum()                      # L1 normalisation: sum = 1

# ---------- Step 2: transition matrix A (column-stochastic) ----------
A = colnorm(np.array([
    [0.70, 0.10, 0.15, 0.05, 0.02],
    [0.15, 0.65, 0.10, 0.10, 0.02],
    [0.05, 0.10, 0.65, 0.05, 0.10],
    [0.05, 0.10, 0.05, 0.75, 0.06],
    [0.05, 0.05, 0.05, 0.05, 0.80],
]))
assert np.allclose(A.sum(axis=0), 1.0)

# ---------- Step 3: impact tensor T[i,j,k] ----------
T = rng.random((n, n, m)) * 0.05
T[2, 2, 0], T[1, 2, 0], T[3, 2, 0] = 0.95, 0.75, 0.60   # tech leader via AI
T[1, 1, 1], T[0, 1, 1] = 0.85, 0.55                      # investor via economy
T[3, 3, 2], T[4, 3, 2] = 0.80, 0.65                      # activist via mindset


def A_eff(u):
    return colnorm(A + sum(u[k] * T[:, :, k] for k in range(m)))


# ---------- Step 4: spectral analysis (Perron-Frobenius) ----------
def spectrum(M):
    vals, vecs = np.linalg.eig(M)
    order = np.argsort(-np.abs(vals))
    vals, vecs = vals[order], vecs[:, order]
    pi = np.real(vecs[:, 0])
    pi = pi / pi.sum()
    return vals, pi


vals, pi = spectrum(A)
rho = abs(vals[0])
lam2 = abs(vals[1])
gap = 1 - lam2
mix_time = np.log(0.01) / np.log(lam2)  # steps to shrink error to 1%

# ---------- Step 5: simulation (deterministic + stochastic) ----------
def simulate(u, steps=40, sigma=0.0, x0=X0, rng_=None):
    M = A_eff(u)
    X = x0.copy()
    out = [X.copy()]
    for _ in range(steps):
        X = M @ X
        if sigma > 0:
            X = np.clip(X + sigma * rng_.standard_normal(n), 1e-9, None)
            X = X / X.sum()
        out.append(X.copy())
    return np.array(out)


U = np.array([0.15, 0.05, 0.02])
STEPS = 40
base = simulate(np.zeros(m), STEPS)
ctrl = simulate(U, STEPS)
pi_ctrl = spectrum(A_eff(U))[1]

# Verification: distance to stationary distribution vs theoretical decay
err = np.abs(base - pi).sum(axis=1)
bound = err[0] * lam2 ** np.arange(STEPS + 1)

# ---------- Step 6: actor influence ----------
def influence(k, eps=0.1):
    u = np.zeros(m); u[k] = eps
    shift = np.abs(spectrum(A_eff(u))[1] - pi).sum() / eps
    return np.linalg.norm(T[:, :, k], "fro"), shift


infl = [influence(k) for k in range(m)]
order = np.argsort([-s for _, s in infl])

# ---------- Step 7: Monte Carlo ----------
RUNS = 300
mc = np.array([simulate(U, STEPS, sigma=0.01, rng_=rng) for _ in range(RUNS)])
mc_mean, mc_std = mc.mean(axis=0), mc.std(axis=0)

# ---------- Report ----------
np.set_printoptions(precision=4, suppress=True)
print("STEP 1  X0 (sum=%.3f):" % X0.sum(), X0)
print("\nSTEP 2  A (column sums):", A.sum(axis=0))
print(A)
print("\nSTEP 4  Spectral analysis")
print("  eigenvalues |l|:", np.abs(vals))
print(f"  spectral radius = {rho:.6f} (Perron-Frobenius: must be 1)")
print(f"  |lambda2| = {lam2:.4f}, spectral gap = {gap:.4f}, mixing time(1%) ~ {mix_time:.1f} steps")
print("  stationary distribution pi:", pi)
print("  pi with actor influence  :", pi_ctrl)
print("\nSTEP 6  Actor influence (Frobenius norm | stationary shift per unit u)")
for r, k in enumerate(order, 1):
    print(f"  {r}. {ACTORS[k]:16s} ||T_k||={infl[k][0]:.3f}  sensitivity={infl[k][1]:.4f}")
print("\nSTEP 7  Monte Carlo final state (mean +- std, %d runs):" % RUNS)
for i, l in enumerate(LABELS):
    print(f"  {l:15s} {mc_mean[-1, i]:.4f} +- {mc_std[-1, i]:.4f}")
print("\nCHECK   error vs bound holds:", bool(np.all(err <= bound * 1.0001 + 1e-9)))

# ---------- Dashboard ----------
fig, ax = plt.subplots(2, 3, figsize=(18, 10))
t = np.arange(STEPS + 1)
for i, l in enumerate(LABELS):
    ax[0, 0].plot(t, base[:, i], label=l)
    ax[0, 1].plot(t, ctrl[:, i], label=l)
ax[0, 0].set_title("Baseline evolution (no actors)")
ax[0, 1].set_title("With actor influence u=[.15,.05,.02]")
for a in ax[0, :2]:
    a.set_xlabel("t"); a.set_ylabel("share"); a.grid(alpha=.3); a.legend(fontsize=8)

ev = vals
th = np.linspace(0, 2 * np.pi, 200)
ax[0, 2].plot(np.cos(th), np.sin(th), "k--", lw=.8)
ax[0, 2].scatter(ev.real, ev.imag, c="crimson", s=60)
ax[0, 2].set_title(f"Eigenvalues (gap={gap:.3f})"); ax[0, 2].set_aspect("equal"); ax[0, 2].grid(alpha=.3)

ax[1, 0].semilogy(t, err + 1e-16, "o-", label="||X(t) - pi||_1")
ax[1, 0].semilogy(t, bound, "r--", label="|lambda2|^t bound")
ax[1, 0].set_title("Convergence check (theory vs simulation)"); ax[1, 0].legend(); ax[1, 0].grid(alpha=.3)

im = ax[1, 1].imshow(A, cmap="viridis")
ax[1, 1].set_xticks(range(n)); ax[1, 1].set_yticks(range(n))
ax[1, 1].set_xticklabels(range(1, n + 1)); ax[1, 1].set_yticklabels(range(1, n + 1))
ax[1, 1].set_title("Transition matrix A"); plt.colorbar(im, ax=ax[1, 1])

i_ai = 2
ax[1, 2].plot(t, mc_mean[:, i_ai], label="mean")
ax[1, 2].fill_between(t, mc_mean[:, i_ai] - 2 * mc_std[:, i_ai], mc_mean[:, i_ai] + 2 * mc_std[:, i_ai],
                      alpha=.3, label="±2σ (300 runs)")
ax[1, 2].set_title("Monte Carlo: AI growth share"); ax[1, 2].legend(); ax[1, 2].grid(alpha=.3)

plt.tight_layout()
plt.savefig("universal_matrix_dashboard.png", dpi=110)
print("\nSaved universal_matrix_dashboard.png")
