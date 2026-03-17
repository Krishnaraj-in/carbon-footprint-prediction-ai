import pandas as pd
import numpy as np
from data_loader import EMISSION_FACTORS, SECR_FACTORS
from emission_calculator import get_co2_factor
 
def load_and_clean(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
 
    # Drop non-predictive columns
    drop_cols = ['id', 'timestamp', 'customer_id', 'route',
                 'origin_city', 'destination_city']
    df = df.drop(columns=drop_cols)
 
    # Convert boolean columns
    df['is_rush_hour'] = df['is_rush_hour'].astype(int)
    df['is_weekend']   = df['is_weekend'].astype(int)
    df['is_fragile']   = df['is_fragile'].astype(int)
 

    df = df[df['vehicle_type'] != 'cargo_bike'].copy()
 
    print(f'Dataset shape after cleaning: {df.shape}')
    print(df['vehicle_type'].value_counts())
    return df
 
df = load_and_clean('../data/synthetic_deliveries.csv')

def compute_defra_baseline(row: pd.Series) -> float:
    """
    Compute deterministic DEFRA emission (kg CO2e) for one scenario.
    Formula: distance × base_factor × load_modifier
             × traffic_modifier × weather_penalty × rush_hour_penalty
    """
    base_factor = get_co2_factor(row['vehicle_type'], row['fuel_type'])
 

    weight_ceiling = 500 if 'van' in row['vehicle_type'] else 20000
    load_modifier = 1 + (row['weight_kg'] / weight_ceiling)
 

    traffic_index = 0.5 + (row['avg_traffic_density'] * 1.5)
    traffic_modifier = 1 + (traffic_index * 0.3)
 
    # Weather penalty from temperature and precipitation
    if row['precipitation_mm'] > 3.0 or row['temperature_celsius'] < 5:
        weather_penalty = 1.12   # Severe: heavy rain or cold
    elif row['precipitation_mm'] > 1.0 or row['temperature_celsius'] < 10:
        weather_penalty = 1.05   # Moderate weather
    else:
        weather_penalty = 1.00   # Normal conditions
 
    rush_hour_penalty = 1.15 if row['is_rush_hour'] else 1.0
 
    emission = (row['distance_km'] * base_factor * load_modifier
                * traffic_modifier * weather_penalty * rush_hour_penalty)
    return round(emission, 4)
 
df['defra_baseline_kg'] = df.apply(compute_defra_baseline, axis=1)
print(df[['distance_km','vehicle_type','actual_co2_emissions_kg',
          'defra_baseline_kg']].head(5))


def get_secr_kwh(vehicle_type: str, fuel_type: str) -> float:
    """Return SECR kWh per km for vehicle/fuel combination."""
    key = (vehicle_type, fuel_type)
    kwh_map = {
        ('diesel_van',   'diesel'):   SECR_FACTORS['van_class3']['diesel'],    # 1.08497
        ('electric_van', 'electric'): SECR_FACTORS['van_average']['electric'],  # 0.0
        ('diesel_truck', 'diesel'):   SECR_FACTORS['hgv_all']['laden_avg'],    # 3.42464
        ('motorcycle',   'petrol'):   SECR_FACTORS['motorbike_kwh']['average'], # 0.46265
    }
    return kwh_map.get(key, SECR_FACTORS['van_average']['diesel'])
 
df['secr_kwh_per_km'] = df.apply(
    lambda r: get_secr_kwh(r['vehicle_type'], r['fuel_type']), axis=1
)
df['estimated_energy_kwh'] = df['secr_kwh_per_km'] * df['distance_km']


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
 
    # 1. Payload ratio (weight relative to vehicle capacity)
    capacity_map = {'diesel_van': 500, 'electric_van': 500,
                    'diesel_truck': 20000, 'motorcycle': 50}
    df['vehicle_capacity'] = df['vehicle_type'].map(capacity_map).fillna(500)
    df['payload_ratio']    = df['weight_kg'] / df['vehicle_capacity']
 
    # 2. Traffic index (rescale avg_traffic_density 0.2–1.0 → 0.5–2.0)
    df['traffic_index'] = 0.5 + (df['avg_traffic_density'] * 1.5)
 
    # 3. Weather penalty (numeric)
    conditions = [
        (df['precipitation_mm'] > 3.0) | (df['temperature_celsius'] < 5),
        (df['precipitation_mm'] > 1.0) | (df['temperature_celsius'] < 10),
    ]
    df['weather_penalty'] = np.select(conditions, [1.12, 1.05], default=1.0)
 
    # 4. Urban proxy (high traffic + short distance)
    df['is_urban'] = ((df['avg_traffic_density'] > 0.65)
                      & (df['distance_km'] < 80)).astype(int)
 
    # 5. Interaction: distance × traffic
    df['dist_traffic_interaction'] = df['distance_km'] * df['traffic_index']
 
    # 6. Time-load product
    df['time_load_product'] = df['is_rush_hour'] * df['payload_ratio']
 
    # 7. Volume-weight density
    df['vol_weight_density'] = (df['weight_kg']
                                 / (df['package_volume_liters'] + 0.001))
 
    # 8. DEFRA baseline as a feature (teaches model the formula's anchor)
    df['defra_baseline_kg'] = df.apply(compute_defra_baseline, axis=1)
 
    # 9. SECR energy feature
    df['secr_kwh_per_km']    = df.apply(
        lambda r: get_secr_kwh(r['vehicle_type'], r['fuel_type']), axis=1)
    df['estimated_energy_kwh'] = df['secr_kwh_per_km'] * df['distance_km']
 
    # 10. One-hot encode categoricals
    df = pd.get_dummies(df,
        columns=['vehicle_type', 'fuel_type', 'delivery_method'],
        drop_first=False)
 
    return df
 
df_features = engineer_features(df)
print('Feature matrix shape:', df_features.shape)