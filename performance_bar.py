import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def compute_defra_baseline(row):
    vehicle_type = row['vehicle_type']
    fuel_type = row['fuel_type']
    if vehicle_type == 'diesel_van':
        base_factor = 0.27365
    elif vehicle_type == 'electric_van':
        base_factor = 0.0
    elif vehicle_type == 'diesel_truck':
        base_factor = 0.87296
    elif vehicle_type == 'motorcycle':
        base_factor = 0.11990
    else:
        base_factor = 0.0

    if base_factor == 0.0:
        return 0.0

    if 'van' in vehicle_type:
        weight_ceiling = 500.0
    elif 'motor' in vehicle_type:
        weight_ceiling = 50.0
    else:
        weight_ceiling = 20000.0

    load_modifier = 1.0 + (row['weight_kg'] / weight_ceiling)
    traffic_index = 0.5 + (row['avg_traffic_density'] * 1.5)
    traffic_modifier = 1.0 + (traffic_index * 0.3)

    if row['precipitation_mm'] > 3.0 or row['temperature_celsius'] < 5.0:
        weather_penalty = 1.12
    elif row['precipitation_mm'] > 1.0 or row['temperature_celsius'] < 10.0:
        weather_penalty = 1.05
    else:
        weather_penalty = 1.00

    rush_hour_penalty = 1.15 if row['is_rush_hour'] else 1.0

    emission = (row['distance_km'] * base_factor * load_modifier * traffic_modifier * weather_penalty * rush_hour_penalty)
    return round(emission, 4)


def get_secr_kwh(vehicle_type, fuel_type):
    if vehicle_type == 'diesel_van':
        return 1.08497
    if vehicle_type == 'electric_van':
        return 0.0
    if vehicle_type == 'diesel_truck':
        return 3.42464
    if vehicle_type == 'motorcycle':
        return 0.46265
    return 0.0


def engineer_features(df):
    df = df.copy()
    drop_cols = ['id', 'timestamp', 'customer_id', 'route', 'origin_city', 'destination_city']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df['is_rush_hour'] = df['is_rush_hour'].astype(int)
    df['is_weekend'] = df['is_weekend'].astype(int)
    df['is_fragile'] = df['is_fragile'].astype(int)
    df = df[df['vehicle_type'] != 'cargo_bike'].copy()
    df['defra_baseline_kg'] = df.apply(compute_defra_baseline, axis=1)
    df['secr_kwh_per_km'] = df.apply(lambda r: get_secr_kwh(r['vehicle_type'], r['fuel_type']), axis=1)
    df['estimated_energy_kwh'] = df['secr_kwh_per_km'] * df['distance_km']

    capacity_map = {'diesel_van': 500, 'electric_van': 500, 'diesel_truck': 20000, 'motorcycle': 50}
    df['vehicle_capacity'] = df['vehicle_type'].map(capacity_map).fillna(500)
    df['payload_ratio'] = df['weight_kg'] / df['vehicle_capacity']
    df['traffic_index'] = 0.5 + (df['avg_traffic_density'] * 1.5)

    conditions = [
        (df['precipitation_mm'] > 3.0) | (df['temperature_celsius'] < 5),
        (df['precipitation_mm'] > 1.0) | (df['temperature_celsius'] < 10),
    ]
    df['weather_penalty'] = np.select(conditions, [1.12, 1.05], default=1.0)

    df['is_urban'] = ((df['avg_traffic_density'] > 0.65) & (df['distance_km'] < 80)).astype(int)
    df['dist_traffic_interaction'] = df['distance_km'] * df['traffic_index']
    df['time_load_product'] = df['is_rush_hour'] * df['payload_ratio']
    df['vol_weight_density'] = df['weight_kg'] / (df['package_volume_liters'] + 0.001)

    df = pd.get_dummies(df, columns=['vehicle_type', 'fuel_type', 'delivery_method'], drop_first=False)
    return df


def evaluate_models():
    # consistent with backend/model_training.py
    TARGET = 'actual_co2_emissions_kg'
    DROP_COLS = ['actual_co2_emissions_kg', 'defra_baseline_kg',
                 'secr_kwh_per_km', 'vehicle_capacity',
                 'origin_lat', 'origin_lng', 'destination_lat', 'destination_lng']

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'synthetic_deliveries.csv')
    df = pd.read_csv(data_path)
    df_features = engineer_features(df)

    X = df_features.drop(columns=[c for c in DROP_COLS if c in df_features.columns])
    y = df_features[TARGET]

    categorical_cols = X.select_dtypes(include=['object', 'str']).columns.tolist()
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Align to scaler/model feature names exactly
    scaler = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'scaler.pkl'))
    expected_columns = list(scaler.feature_names_in_)

    for col in expected_columns:
        if col not in X_encoded.columns:
            X_encoded[col] = 0

    X_encoded = X_encoded[expected_columns]

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, shuffle=True
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir  = os.path.join(script_dir, 'models')

    gbr_model = joblib.load(os.path.join(model_dir, 'gbr_model.pkl'))
    rf_model  = joblib.load(os.path.join(model_dir, 'rf_model.pkl'))
    scaler    = joblib.load(os.path.join(model_dir, 'scaler.pkl'))

    X_test_scaled = scaler.transform(X_test)

    y_pred_gbr = gbr_model.predict(X_test_scaled)
    y_pred_rf  = rf_model.predict(X_test)

    metrics = {
        'GBR': {
            'r2': r2_score(y_test, y_pred_gbr),
            'mae': mean_absolute_error(y_test, y_pred_gbr),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_gbr)),
        },
        'RF': {
            'r2': r2_score(y_test, y_pred_rf),
            'mae': mean_absolute_error(y_test, y_pred_rf),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
        },
    }

    return metrics


def plot_performance(metrics, output_path=None):
    labels = ['R2 (x100)', 'MAE (kg CO₂)', 'RMSE (kg CO₂)']
    gbr_vals = [metrics['GBR']['r2'] * 100, metrics['GBR']['mae'], metrics['GBR']['rmse']]
    rf_vals  = [metrics['RF']['r2'] * 100, metrics['RF']['mae'], metrics['RF']['rmse']]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, gbr_vals, width, label='GBR', color='#0f766e')
    ax.bar(x + width/2, rf_vals,  width, label='RF', color='#64748b')

    ax.set_ylabel('Value')
    ax.set_title('GBR vs RF Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, max(max(gbr_vals), max(rf_vals)) * 1.2)

    for i, v in enumerate(gbr_vals):
        ax.text(i - width/2, v + 0.8, f'{v:.1f}', ha='center', va='bottom', fontsize=10)
    for i, v in enumerate(rf_vals):
        ax.text(i + width/2, v + 0.8, f'{v:.1f}', ha='center', va='bottom', fontsize=10)

    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=180)
        print(f'Saved bar chart to {output_path}')

    plt.show()


if __name__ == '__main__':
    metrics = evaluate_models()
    print('GBR metrics:', metrics['GBR'])
    print('RF metrics:', metrics['RF'])

    out_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gbr_vs_rf_performance.png')
    plot_performance(metrics, out_file)
