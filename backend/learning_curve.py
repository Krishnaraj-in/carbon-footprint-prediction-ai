import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import learning_curve

# ─── 1. Recreate your dataset (Section 3.3 DEFRA formula) ──────────────────
np.random.seed(42)
n = 8964

distance     = np.random.uniform(1.2, 119.7, n)
load         = np.random.uniform(10, 998, n)
traffic      = np.random.uniform(0.51, 1.99, n)
weather      = np.random.choice([1.00, 1.05, 1.12], n, p=[0.6, 0.3, 0.1])
rush_hour    = np.random.randint(0, 2, n)
base_factor  = np.random.choice(
    [0.14533, 0.19285, 0.02400, 0.11360, 0.00300], n,
    p=[0.50, 0.25, 0.15, 0.07, 0.03]
)
noise        = np.random.uniform(0.95, 1.05, n)
rush_penalty = np.where(rush_hour == 1, 1.15, 1.0)

# DEFRA formula from Section 3.3
emission = (distance * base_factor
            * (1 + load / 50)
            * (1 + traffic * 0.3)
            * weather * rush_penalty * noise)

# Feature matrix
X = np.column_stack([distance, load, traffic, weather, rush_hour, base_factor])
y = emission

# ─── 2. Define models (hyperparams from Section 3.7 grid search) ────────────
gbr = GradientBoostingRegressor(
    n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
)
rf = RandomForestRegressor(
    n_estimators=200, max_depth=10, min_samples_split=2, random_state=42
)

# ─── 3. Compute Learning Curves (5-fold CV matching Section 3.7) ────────────
train_sizes = np.linspace(0.1, 1.0, 10)
# Work around joblib multiprocessing issues on some Windows Python builds.
SAFE_N_JOBS = 1

train_sz_gbr, train_scores_gbr, val_scores_gbr = learning_curve(
    gbr, X, y,
    train_sizes=train_sizes,
    cv=5,
    scoring='r2',
    n_jobs=SAFE_N_JOBS
)

train_sz_rf, train_scores_rf, val_scores_rf = learning_curve(
    rf, X, y,
    train_sizes=train_sizes,
    cv=5,
    scoring='r2',
    n_jobs=SAFE_N_JOBS
)

# ─── 4. Plot ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
fig.suptitle(
    "Learning Curves — GBR vs Random Forest\n"
    "Carbon Emission Prediction (n=8,964 UK Delivery Scenarios)",
    fontsize=13, fontweight='bold', y=1.02
)

for ax, train_sz, tr_scores, val_scores, title, colour in [
    (axes[0], train_sz_gbr, train_scores_gbr, val_scores_gbr,
     "Gradient Boosting Regressor (GBR)", "#1F4E79"),
    (axes[1], train_sz_rf,  train_scores_rf,  val_scores_rf,
     "Random Forest (RF)",                "#82b366"),
]:
    tr_mean  = tr_scores.mean(axis=1)
    tr_std   = tr_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std  = val_scores.std(axis=1)

    ax.plot(train_sz, tr_mean,  'o-', color=colour,  label='Training R²',   linewidth=2)
    ax.fill_between(train_sz, tr_mean - tr_std,  tr_mean + tr_std,
                    alpha=0.15, color=colour)

    ax.plot(train_sz, val_mean, 's--', color='tomato', label='CV R² (5-fold)', linewidth=2)
    ax.fill_between(train_sz, val_mean - val_std, val_mean + val_std,
                    alpha=0.15, color='tomato')

    # Annotate final CV R² from report
    final_cv = 0.990 if 'GBR' in title else 0.883
    ax.axhline(final_cv, color='grey', linestyle=':', linewidth=1.2,
               label=f'Reported CV R² = {final_cv}')

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel("Training Set Size (scenarios)", fontsize=10)
    ax.set_ylabel("R² Score", fontsize=10)
    ax.set_ylim(0.5, 1.02)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("learning_curve.png", dpi=300, bbox_inches='tight')
plt.show()
print("Saved: learning_curve.png")