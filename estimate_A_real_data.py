"""
Estimate the transition matrix A from REAL world data (World Bank, World aggregate, 2005-2023)
Model: standardized annual changes  d(t+1) = B d(t) + e(t)   (ridge-regularised VAR(1))
Data source: World Bank Open Data API (api.worldbank.org), World aggregate "WLD".
"""
import numpy as np, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

years = list(range(2005, 2024))
# values fetched from the World Bank API, listed 2005..2023
gdp = [4.0591513939052, 4.48913510541212, 4.43123376835284, 2.09160448064915, -1.33303814192512,
       4.52085213873801, 3.31431839310659, 2.74743359443481, 2.88974765617328, 3.16849193058364,
       3.12721325699906, 2.78974074234588, 3.44658623167963, 3.28663086462807, 2.680308747553,
       -2.8862118545708, 6.48508887596732, 3.44001097265658, 2.85938493335971]
net = [15.6, 17.2, 20.2, 22.8, 25.3, 28.4, 30.9, 33.3, 35.3, 37.4, 39.9, 43.6, 46.3, 49.4, 53.9,
       60.1, 63.8, 67, 69.2]
rnd = [1.94742294314691, 1.95948046250741, 1.92478790620051, 1.97782056990498, 2.01385351337703,
       1.99889242984883, 1.98165761122278, 2.01288483345794, 1.975191691907, 2.08659703677765,
       2.10431830423067, 2.14865606825884, 2.14199917077902, 2.21817463707231, 2.26339308318271,
       2.45606936077903, 2.5227995854217, 2.5711591602204, 2.59737836465431]
co2 = [4.38846326687297, 4.47802062614128, 4.58831889715123, 4.56160067411296, 4.45009852599458,
       4.67132107098143, 4.76260892077127, 4.77415805922717, 4.80936209737091, 4.77725129095571,
       4.69833990155379, 4.65001262944222, 4.67223120041328, 4.73494565527389, 4.70120784176447,
       4.47451332276585, 4.68476409230163, 4.66789826409203, 4.67759995985565]
fdi = [3.25006740382534, 4.20905710925081, 5.26778582426019, 3.71639869792935, 2.42189804326271,
       2.70315400310448, 3.15441473783361, 2.65658518564651, 2.64643308813522, 2.40765405360375,
       3.60688353056102, 3.51689327049206, 2.70520870156727, 1.15929971742826, 2.15216224296887,
       1.33250811209163, 2.50690608217622, 1.86462748037312, 0.868642986055631]
NAMES = ["GDP growth %", "Internet users %", "R&D % GDP", "CO2 t/capita", "FDI inflow % GDP"]
Y = np.array([gdp, net, rnd, co2, fdi]).T          # (19, 5)
assert Y.shape == (19, 5) and not np.isnan(Y).any()

with open("world_data_2005_2023.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["year"] + NAMES)
    for y, row in zip(years, Y): w.writerow([y] + list(np.round(row, 5)))

D = np.diff(Y, axis=0)                              # (18, 5) annual changes
n_obs = D.shape[0]
rng = np.random.default_rng(7)


def zfit(Dtrain):
    return Dtrain.mean(0), Dtrain.std(0, ddof=1)


def ridge_B(Z, lam):
    """Z: standardized changes (T,5). Returns B with z(t+1)=B z(t)."""
    X, Yn = Z[:-1], Z[1:]
    k = X.shape[1]
    Bt = np.linalg.solve(X.T @ X + lam * np.eye(k), X.T @ Yn)
    return Bt.T


def forecast_errors(lam, start=8):
    """Rolling-origin one-step-ahead: train on D[:s], predict D[s]. Compare with mean baseline."""
    e_model, e_base = [], []
    for s in range(start, n_obs):
        mu, sd = zfit(D[:s])
        Z = (D[:s] - mu) / sd
        B = ridge_B(Z, lam)
        pred = B @ Z[-1]
        truth = (D[s] - mu) / sd
        e_model.append(truth - pred)
        e_base.append(truth - 0.0)                 # baseline: predict the training mean
    return np.array(e_model), np.array(e_base)


# ---- Step 1: choose ridge strength by rolling-origin CV
lams = [0.1, 0.3, 1, 3, 10, 30, 100]
cv = {l: np.mean(forecast_errors(l)[0] ** 2) for l in lams}
best = min(cv, key=cv.get)
em, eb = forecast_errors(best)
rmse_m = np.sqrt((em ** 2).mean(0)); rmse_b = np.sqrt((eb ** 2).mean(0))
skill = 1 - (em ** 2).mean(0) / (eb ** 2).mean(0)

# ---- Step 2: final fit on all data
mu, sd = zfit(D)
Z = (D - mu) / sd
B = ridge_B(Z, best)
vals = np.linalg.eigvals(B)
rho = np.max(np.abs(vals))

# ---- Step 3: bootstrap (pairs) for B and spectral radius
boot_B, boot_rho = [], []
for _ in range(2000):
    idx = rng.integers(0, Z.shape[0] - 1, Z.shape[0] - 1)
    X = Z[:-1][idx]; Yn = Z[1:][idx]
    Bt = np.linalg.solve(X.T @ X + best * np.eye(5), X.T @ Yn).T
    boot_B.append(Bt); boot_rho.append(np.max(np.abs(np.linalg.eigvals(Bt))))
boot_B = np.array(boot_B); boot_rho = np.array(boot_rho)
lo, hi = np.percentile(boot_B, [5, 95], axis=0)
rho_lo, rho_hi = np.percentile(boot_rho, [5, 95])

# ---- Step 4: stochastic-matrix view (comparable to the toy model's A)
# Influence-share matrix: column-normalised |B| (not a Markov matrix; a relative influence map)
S = np.abs(B) / np.abs(B).sum(axis=0, keepdims=True)

np.set_printoptions(precision=3, suppress=True)
print("Data: World Bank WLD, 2005-2023, n_years=19, n_transitions=%d" % (n_obs - 1))
print("Ridge lambda by rolling-origin CV:", {k: round(v, 3) for k, v in cv.items()}, "-> best", best)
print("\nB (rows=affected, cols=driver), z-scored annual changes:\n", B)
print("\n|eigenvalues|:", np.sort(np.abs(vals))[::-1])
print(f"spectral radius = {rho:.3f}  (90% bootstrap CI {rho_lo:.3f} - {rho_hi:.3f})")
print("\nOut-of-sample one-step forecast (rolling origin), RMSE in z-units")
for i, nme in enumerate(NAMES):
    print(f"  {nme:18s} model {rmse_m[i]:.3f}  mean-baseline {rmse_b[i]:.3f}  skill {skill[i]:+.2f}")
print(f"  overall skill = {1 - (em**2).mean() / (eb**2).mean():+.3f}  (positive = better than baseline)")
print("\nSignificant links (90% bootstrap interval excludes 0):")
found = False
for i in range(5):
    for j in range(5):
        if lo[i, j] > 0 or hi[i, j] < 0:
            found = True
            print(f"  {NAMES[j]} (t) -> {NAMES[i]} (t+1): {B[i, j]:+.2f}  [{lo[i, j]:+.2f}, {hi[i, j]:+.2f}]")
if not found:
    print("  none")

# ---- Figure
fig, ax = plt.subplots(2, 3, figsize=(18, 10))
for i in range(5):
    ax[0, 0].plot(years, (Y[:, i] - Y[:, i].mean()) / Y[:, i].std(), label=NAMES[i])
ax[0, 0].set_title("Real data (World, z-scored levels)"); ax[0, 0].legend(fontsize=8); ax[0, 0].grid(alpha=.3)
im = ax[0, 1].imshow(B, cmap="RdBu_r", vmin=-1, vmax=1)
ax[0, 1].set_xticks(range(5)); ax[0, 1].set_yticks(range(5))
ax[0, 1].set_xticklabels(["GDP", "Net", "R&D", "CO2", "FDI"]); ax[0, 1].set_yticklabels(["GDP", "Net", "R&D", "CO2", "FDI"])
for i in range(5):
    for j in range(5): ax[0, 1].text(j, i, f"{B[i, j]:.2f}", ha="center", va="center", fontsize=8)
ax[0, 1].set_title("Estimated B (driver = column)"); plt.colorbar(im, ax=ax[0, 1])
ax[0, 2].hist(boot_rho, bins=30, color="steelblue")
ax[0, 2].axvline(rho, color="k"); ax[0, 2].axvline(1, color="r", ls="--")
ax[0, 2].set_title("Bootstrap: spectral radius (red = 1)")
xs = np.arange(5)
ax[1, 0].bar(xs - .2, rmse_b, .4, label="mean baseline"); ax[1, 0].bar(xs + .2, rmse_m, .4, label="VAR(1) ridge")
ax[1, 0].set_xticks(xs); ax[1, 0].set_xticklabels(["GDP", "Net", "R&D", "CO2", "FDI"])
ax[1, 0].set_title("Out-of-sample RMSE (lower = better)"); ax[1, 0].legend()
ax[1, 1].semilogx(list(cv.keys()), list(cv.values()), "o-"); ax[1, 1].set_title("CV error vs ridge λ"); ax[1, 1].grid(alpha=.3)
th = np.linspace(0, 2 * np.pi, 200)
ax[1, 2].plot(np.cos(th), np.sin(th), "k--", lw=.8); ax[1, 2].scatter(vals.real, vals.imag, c="crimson", s=60)
ax[1, 2].set_aspect("equal"); ax[1, 2].set_title("Eigenvalues of estimated B"); ax[1, 2].grid(alpha=.3)
plt.tight_layout(); plt.savefig("real_data_matrix_estimation.png", dpi=110)
print("\nSaved real_data_matrix_estimation.png, world_data_2005_2023.csv")
