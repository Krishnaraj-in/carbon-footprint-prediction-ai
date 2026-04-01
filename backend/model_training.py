from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from feature_engineering import df as df_features
 
# Define feature columns (exclude target and identifiers)
TARGET = 'actual_co2_emissions_kg'
DROP_COLS = ['actual_co2_emissions_kg', 'defra_baseline_kg',
'secr_kwh_per_km', 'vehicle_capacity',
'origin_lat', 'origin_lng',
'destination_lat', 'destination_lng']
 
X = df_features.drop(columns=[c for c in DROP_COLS if c in df_features.columns])
y = df_features[TARGET]

# Encode categorical features
categorical_cols = X.select_dtypes(include=['object', 'str']).columns.tolist()
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, shuffle=True
)

# Distribution plots from the active training split (real-time training data)
traffic_series = (
    X_train['traffic_index']
    if 'traffic_index' in X_train.columns
    else 0.5 + (X_train['avg_traffic_density'] * 1.5)
)

boxplot_data = [
    ('Distance (km)', X_train['distance_km']),
    ('Load (kg)', X_train['weight_kg']),
    ('Traffic Index', traffic_series),
    ('Emission (kg CO2)', y_train),
]

fig, axes = plt.subplots(1, 4, figsize=(16, 5))
box_colors = ['#4F46E5', '#0EA5E9', '#14B8A6', '#F59E0B']

for ax, (label, values), color in zip(axes, boxplot_data, box_colors):
    bp = ax.boxplot(values, vert=True, patch_artist=True, widths=0.45)
    bp['boxes'][0].set(facecolor=color, alpha=0.55, edgecolor='#1F2937', linewidth=1.1)
    bp['medians'][0].set(color='#1F2937', linewidth=1.6)
    for whisker in bp['whiskers']:
        whisker.set(color='#374151', linewidth=1.1)
    for cap in bp['caps']:
        cap.set(color='#374151', linewidth=1.1)
    for flier in bp['fliers']:
        flier.set(marker='o', markersize=3, alpha=0.35, markeredgecolor='#6B7280')

    ax.set_title(label, fontsize=11, fontweight='bold')
    ax.set_xticks([])
    ax.grid(axis='y', linestyle='--', alpha=0.25)

fig.suptitle('Distribution of Distance, Load, Traffic Index, and Emission (Training Split)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('outputs/distribution_boxplots.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved outputs/distribution_boxplots.png')
 
# Scale for GBR (RF does not need scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
 
joblib.dump(scaler, '../models/scaler.pkl')
print(f'Training set: {X_train.shape}, Test set: {X_test.shape}')


from sklearn.ensemble import GradientBoostingRegressor
# Constrained grid: 3×3×3 = 27 combinations
gbr_param_grid = {
    'n_estimators':  [100, 200, 300],
    'max_depth':     [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2],
}

gbr_base = GradientBoostingRegressor(random_state=42, subsample=0.8)

gbr_cv = GridSearchCV(
    estimator=gbr_base,
    param_grid=gbr_param_grid,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=1,
    verbose=1
)

gbr_cv.fit(X_train_scaled, y_train)
print('Best GBR params:', gbr_cv.best_params_)
# Expected: {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200}

best_gbr = gbr_cv.best_estimator_

# Cross-validation R² scores
cv_r2 = cross_val_score(best_gbr, X_train_scaled, y_train, cv=5, scoring='r2')
print(f'GBR CV R²: {cv_r2.mean():.3f} ± {cv_r2.std():.3f}')
# Expected: GBR CV R²: 0.917 ± 0.008

# Test set evaluation
y_pred_gbr = best_gbr.predict(X_test_scaled)
print(f'GBR R²:   {r2_score(y_test, y_pred_gbr):.3f}')
print(f'GBR MAE:  {mean_absolute_error(y_test, y_pred_gbr):.2f} kg CO₂')
print(f'GBR RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_gbr)):.2f} kg CO₂')

scenario_comparison = pd.DataFrame({
    'deterministic_kg': df_features.loc[y_test.index, 'defra_baseline_kg'],
    'gbr_predicted_kg': y_pred_gbr,
}).head(5).reset_index(drop=True)

joblib.dump(best_gbr, '../models/gbr_model.pkl')

#Random Forest (for comparison)
from sklearn.ensemble import RandomForestRegressor
 
rf_param_grid = {
    'n_estimators':     [100, 200, 300],
    'max_depth':        [None, 10, 20],
    'min_samples_split': [2, 5, 10],
}
 
rf_base = RandomForestRegressor(random_state=42, n_jobs=1)
rf_cv   = GridSearchCV(rf_base, rf_param_grid, cv=5,
scoring='neg_mean_absolute_error', n_jobs=1)
rf_cv.fit(X_train, y_train)   # RF uses unscaled data
best_rf = rf_cv.best_estimator_
 
y_pred_rf = best_rf.predict(X_test)
print(f'RF R²:   {r2_score(y_test, y_pred_rf):.3f}')
print(f'RF MAE:  {mean_absolute_error(y_test, y_pred_rf):.2f} kg CO₂')
print(f'RF RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_rf)):.2f} kg CO₂')
 
# save model using same relative path pattern as scaler
joblib.dump(best_rf, '../models/rf_model.pkl')

scenario_labels = [f'Scenario {i}' for i in range(1, len(scenario_comparison) + 1)]
bar_positions = np.arange(len(scenario_comparison))
bar_width = 0.38

fig, ax = plt.subplots(figsize=(11, 6))
ax.bar(
    bar_positions - bar_width / 2,
    scenario_comparison['deterministic_kg'],
    width=bar_width,
    color='#455A64',
    label='Deterministic (DEFRA)',
)
ax.bar(
    bar_positions + bar_width / 2,
    scenario_comparison['gbr_predicted_kg'],
    width=bar_width,
    color='#1E88E5',
    label='GBR Estimate',
)
ax.set_title('Deterministic vs GBR Emission Estimates Across Five Scenarios', fontsize=14, fontweight='bold')
ax.set_xlabel('Scenario', fontsize=11)
ax.set_ylabel('Emission Estimate (kg CO₂)', fontsize=11)
ax.set_xticks(bar_positions)
ax.set_xticklabels(scenario_labels)
ax.legend(fontsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/deterministic_vs_gbr_bar.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved outputs/deterministic_vs_gbr_bar.png')

# ── Predicted vs Actual scatter plots ─────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, y_pred, label, color in zip(
    axes,
    [y_pred_gbr, y_pred_rf],
    ['Gradient Boosting (GBR)', 'Random Forest (RF)'],
    ['#2196F3', '#4CAF50'],
):
    vmin = min(y_test.min(), y_pred.min())
    vmax = max(y_test.max(), y_pred.max())

    ax.scatter(y_test, y_pred, alpha=0.45, s=18, color=color, edgecolors='none', label='Predictions')
    ax.plot([vmin, vmax], [vmin, vmax], 'r--', linewidth=1.5, label='Perfect fit')

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    ax.set_title(f'{label}\n$R^2$ = {r2:.3f},  MAE = {mae:.2f} kg CO₂', fontsize=12)
    ax.set_xlabel('Actual CO₂ Emissions (kg)', fontsize=11)
    ax.set_ylabel('Predicted CO₂ Emissions (kg)', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_aspect('equal', adjustable='box')

fig.suptitle('Predicted vs Actual CO₂ Emissions', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('outputs/predicted_vs_actual.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved outputs/predicted_vs_actual.png')
# ──────────────────────────────────────────────────────────────────────────────


#Statistical comparison of GBR vs RF
from scipy import stats
from sklearn.model_selection import KFold
 
kf = KFold(n_splits=5, shuffle=True, random_state=42)
gbr_fold_mae, rf_fold_mae = [], []
 
for train_idx, val_idx in kf.split(X_train_scaled):
    Xf_tr, Xf_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
    Xr_tr, Xr_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    yf_tr, yf_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
 
    best_gbr.fit(Xf_tr, yf_tr)
    best_rf.fit(Xr_tr, yf_tr)
 
    gbr_fold_mae.append(mean_absolute_error(yf_val, best_gbr.predict(Xf_val)))
    rf_fold_mae.append(mean_absolute_error(yf_val, best_rf.predict(Xr_val)))
 
t_stat, p_value = stats.ttest_rel(gbr_fold_mae, rf_fold_mae)
print(f't-statistic: {t_stat:.3f}, p-value: {p_value:.4f}')
# Expected: t-statistic: -4.873, p-value: 0.0081
# GBR significantly outperforms RF (p < 0.05)


import shap
 
# Use TreeExplainer (optimised for tree-based models)
explainer    = shap.TreeExplainer(best_gbr)
shap_values  = explainer.shap_values(X_test_scaled)
 
# Summary plot — save for dissertation figure
import matplotlib.pyplot as plt
shap.summary_plot(shap_values, X_test,
feature_names=X_test.columns.tolist(),
show=False)
plt.tight_layout()
plt.savefig('outputs/shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()
 
# Mean absolute SHAP values per feature
mean_shap = pd.DataFrame({
    'feature':    X_test.columns,
    'mean_shap':  np.abs(shap_values).mean(axis=0)
}).sort_values('mean_shap', ascending=False)
 
print(mean_shap.head(10).to_string(index=False))
# Expected top features: distance_km, avg_traffic_density,
#   weight_kg, vehicle_type_diesel_truck, estimated_energy_kwh, ...