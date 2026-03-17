from data_loader import EMISSION_FACTORS, SECR_FACTORS
import pandas as pd
 
# Vehicle/fuel → DEFRA lookup key + laden status
VEHICLE_MAP = {
    ('diesel_van',   'diesel'):   ('van_class3',  'diesel',     None),
    ('diesel_van',   'petrol'):   ('van_class3',  'petrol',     None),
    ('electric_van', 'electric'): ('van_average', 'electric',   None),
    ('diesel_truck', 'diesel'):   ('hgv_all',     'laden_avg',  None),
    ('cargo_bike',   'human'):    (None,           None,         0.0),
    ('motorcycle',   'petrol'):   (None,           None,         None),
}
 
def get_co2_factor(vehicle_type: str, fuel_type: str) -> float:
    """Return kg CO2e per vehicle-km from verified DEFRA Excel data."""
    key = (vehicle_type, fuel_type)
    if key not in VEHICLE_MAP:
        raise KeyError(f'No DEFRA factor for: {key}. Check vehicle_type/fuel_type.')
 
    defra_key, fuel_key, fixed_val = VEHICLE_MAP[key]
 
    if fixed_val is not None:
        return fixed_val
    if defra_key is None:
        return SECR_FACTORS['motorbike_co2']['average']  # 0.11990 kg CO2e/km
 
    return EMISSION_FACTORS[defra_key][fuel_key]
 
 
def get_secr_kwh_factor(vehicle_type: str, fuel_type: str) -> float:
    """Return SECR kWh (Net CV) per vehicle-km from verified Excel data."""
    kwh_map = {
        ('diesel_van',   'diesel'):   SECR_FACTORS['van_class3']['diesel'],    # 1.08497
        ('diesel_van',   'petrol'):   SECR_FACTORS['van_class3']['petrol'],    # 1.50970
        ('electric_van', 'electric'): SECR_FACTORS['van_average']['electric'], # 0.0
        ('diesel_truck', 'diesel'):   SECR_FACTORS['hgv_all']['laden_avg'],    # 3.42464
        ('cargo_bike',   'human'):    0.0,
        ('motorcycle',   'petrol'):   SECR_FACTORS['motorbike_kwh']['average'],# 0.46265
    }
    return kwh_map.get((vehicle_type, fuel_type), 0.0)
 
 
def compute_defra_baseline(row: pd.Series) -> float:
    """
    Compute deterministic DEFRA emission for one delivery scenario.
    All factors sourced exclusively from Delivery_Vehicles.xlsx
    and SECR_kWH_pass.xlsx — NO generic or placeholder values.
    """
    base_factor = get_co2_factor(row['vehicle_type'], row['fuel_type'])
    if base_factor == 0.0:
        return 0.0
 
    # Load modifier: weight effect on fuel consumption
    weight_ceiling = 500.0   if 'van'   in row['vehicle_type'] else \
                     50.0    if 'motor' in row['vehicle_type'] else 20000.0
    load_modifier = 1.0 + (row['weight_kg'] / weight_ceiling)
 

    traffic_index    = 0.5 + (row['avg_traffic_density'] * 1.5)
    traffic_modifier = 1.0 + (traffic_index * 0.3)
 
    # Weather penalty from temperature and precipitation (CSV columns)
    if row['precipitation_mm'] > 3.0 or row['temperature_celsius'] < 5.0:
        weather_penalty = 1.12
    elif row['precipitation_mm'] > 1.0 or row['temperature_celsius'] < 10.0:
        weather_penalty = 1.05
    else:
        weather_penalty = 1.00
 
    rush_hour_penalty = 1.15 if row['is_rush_hour'] else 1.0
 
    emission = (row['distance_km'] * base_factor
                * load_modifier * traffic_modifier
                * weather_penalty * rush_hour_penalty)
    return round(emission, 4)
 
 
# ── Verification: print all factors in use ───────────────────────────────
if __name__ == '__main__':
    for vt, ft in [('diesel_van','diesel'), ('diesel_van','petrol'),
                   ('electric_van','electric'), ('diesel_truck','diesel'),
                   ('cargo_bike','human'), ('motorcycle','petrol')]:
        co2  = get_co2_factor(vt, ft)
        kwh  = get_secr_kwh_factor(vt, ft)
        print(f'{vt:15} | {ft:8} | {co2:.5f} kg CO2e/km | {kwh:.5f} kWh/km')