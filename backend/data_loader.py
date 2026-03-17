import pandas as pd
import numpy as np
 
def load_defra_emission_factors(filepath: str) -> dict:
    """
    Load DEFRA 2024 GHG Conversion Factors from Delivery_Vehicles.xlsx.
    Row/column indices are 0-based from the raw Excel file.
    """
    raw = pd.read_excel(filepath, header=None)
 
    # ── VANS (rows 25, 27, 29, 31 — km rows only) ──────────────────────
    # Fuel columns: 3=Diesel, 7=Petrol, 23=PHEV, 27=BEV
    van_factors = {
        # Class I: up to 1.305 tonnes
        'van_class1': {
            'diesel':   float(raw.iloc[25, 3]),   # 0.15356 kg CO2e/km
            'petrol':   float(raw.iloc[25, 7]),   # 0.20071 kg CO2e/km
            'electric': float(raw.iloc[25, 27]),  # 0.0     kg CO2e/km (BEV tailpipe)
        },
        # Class II: 1.305 to 1.74 tonnes
        'van_class2': {
            'diesel':   float(raw.iloc[27, 3]),   # 0.18832 kg CO2e/km
            'petrol':   float(raw.iloc[27, 7]),   # 0.21709 kg CO2e/km
            'electric': float(raw.iloc[27, 27]),  # 0.0     kg CO2e/km
        },
        # Class III: 1.74 to 3.5 tonnes — DEFAULT for diesel_van in CSV
        'van_class3': {
            'diesel':   float(raw.iloc[29, 3]),   # 0.27365 kg CO2e/km
            'petrol':   float(raw.iloc[29, 7]),   # 0.34923 kg CO2e/km
            'phev':     float(raw.iloc[29, 23]),  # 0.13292 kg CO2e/km
            'electric': float(raw.iloc[29, 27]),  # 0.0     kg CO2e/km
        },
        # Average: up to 3.5 tonnes — fallback if class unknown
        'van_average': {
            'diesel':   float(raw.iloc[31, 3]),   # 0.25023 kg CO2e/km
            'petrol':   float(raw.iloc[31, 7]),   # 0.22095 kg CO2e/km
            'cng':      float(raw.iloc[31, 11]),  # 0.25120 kg CO2e/km
            'lpg':      float(raw.iloc[31, 15]),  # 0.27617 kg CO2e/km
            'phev':     float(raw.iloc[31, 23]),  # 0.13292 kg CO2e/km
            'electric': float(raw.iloc[31, 27]),  # 0.0     kg CO2e/km
        },
    }
 
    # ── HGVs (rows 37, 39, 41, 43, 45, 47, 49, 51 — km rows) ───────────
    # Laden columns: 3=0% Laden, 7=50% Laden, 11=100% Laden, 15=Avg Laden
    # Project default: Average Laden (col 15) for diesel_truck in CSV
    hgv_factors = {
        'hgv_rigid_3_5_7_5t': {
            'laden_0pct':  float(raw.iloc[37, 3]),   # 0.45380
            'laden_50pct': float(raw.iloc[37, 7]),   # 0.49279
            'laden_100pct':float(raw.iloc[37, 11]),  # 0.53178
            'laden_avg':   float(raw.iloc[37, 15]),  # 0.48733  ← project default
        },
        'hgv_rigid_7_5_17t': {
            'laden_0pct':  float(raw.iloc[39, 3]),   # 0.54426
            'laden_50pct': float(raw.iloc[39, 7]),   # 0.62106
            'laden_100pct':float(raw.iloc[39, 11]),  # 0.69787
            'laden_avg':   float(raw.iloc[39, 15]),  # 0.59495  ← project default
        },
        'hgv_rigid_17t_plus': {
            'laden_0pct':  float(raw.iloc[41, 3]),   # 0.74987
            'laden_50pct': float(raw.iloc[41, 7]),   # 0.91210
            'laden_100pct':float(raw.iloc[41, 11]),  # 1.07433
            'laden_avg':   float(raw.iloc[41, 15]),  # 0.97698  ← project default
        },
        'hgv_all_rigids': {
            'laden_avg':   float(raw.iloc[43, 15]),  # 0.82657
        },
        'hgv_artic_3_5_33t': {
            'laden_avg':   float(raw.iloc[45, 15]),  # 0.76642
        },
        'hgv_artic_33t_plus': {
            'laden_avg':   float(raw.iloc[47, 15]),  # 0.91247
        },
        # Overall HGV average — DEFAULT for diesel_truck in synthetic CSV
        'hgv_all': {
            'laden_0pct':  float(raw.iloc[51, 3]),   # 0.64392
            'laden_50pct': float(raw.iloc[51, 7]),   # 0.81517
            'laden_100pct':float(raw.iloc[51, 11]),  # 0.98641
            'laden_avg':   float(raw.iloc[51, 15]),  # 0.87296  ← project default
        },
    }
 
    return {**van_factors, **hgv_factors}
 
 
EMISSION_FACTORS = load_defra_emission_factors('../data/raw/Delivery_Vehicles.xlsx')
 
# Quick verification printout
print('Van Class III diesel:   ', EMISSION_FACTORS['van_class3']['diesel'])   # 0.27365
print('All HGVs avg laden:     ', EMISSION_FACTORS['hgv_all']['laden_avg'])   # 0.87296
print('Van Class I electric:   ', EMISSION_FACTORS['van_class1']['electric']) # 0.0


def load_secr_kwh_factors(filepath: str) -> dict:
    """
    Load SECR 2024 Energy Conversion Factors from SECR_kWH_pass.xlsx.
    All values: kWh (Net CV) per vehicle-km.
    Used for: (1) SECR energy reporting, (2) estimated_energy_kwh ML feature.
    """
    raw = pd.read_excel(filepath, header=None)
 
    # ── VANS (rows 63, 66, 69, 72 — km rows only) ───────────────────────
    # Fuel columns: 3=Diesel, 4=Petrol, 5=CNG, 6=LPG, 7=Unknown, 8=PHEV, 9=BEV
    van_kwh = {
        'van_class1': {
            'diesel':   float(raw.iloc[63, 3]),   # 0.60597 kWh/km
            'petrol':   float(raw.iloc[63, 4]),   # 0.86634 kWh/km
            'electric': float(raw.iloc[63, 9]),   # 0.0     kWh/km (BEV conventional fuel)
        },
        'van_class2': {
            'diesel':   float(raw.iloc[66, 3]),   # 0.74460 kWh/km
            'petrol':   float(raw.iloc[66, 4]),   # 0.93732 kWh/km
            'electric': float(raw.iloc[66, 9]),   # 0.0     kWh/km
        },
        'van_class3': {
            'diesel':   float(raw.iloc[69, 3]),   # 1.08497 kWh/km
            'petrol':   float(raw.iloc[69, 4]),   # 1.50970 kWh/km
            'phev':     float(raw.iloc[69, 8]),   # 0.56797 kWh/km
            'electric': float(raw.iloc[69, 9]),   # 0.0     kWh/km
        },
        'van_average': {
            'diesel':   float(raw.iloc[72, 3]),   # 0.99155 kWh/km
            'petrol':   float(raw.iloc[72, 4]),   # 0.95404 kWh/km
            'cng':      float(raw.iloc[72, 5]),   # 1.23317 kWh/km
            'lpg':      float(raw.iloc[72, 6]),   # 1.19846 kWh/km
            'unknown':  float(raw.iloc[72, 7]),   # 0.99049 kWh/km
            'phev':     float(raw.iloc[72, 8]),   # 0.56797 kWh/km
            'electric': float(raw.iloc[72, 9]),   # 0.0     kWh/km
        },
    }
 
    # ── HGVs (rows 79, 82, 85, 88, 91, 94, 97, 100 — km rows) ──────────
    # Laden columns: 3=0% Laden, 4=50% Laden, 5=100% Laden, 6=Average Laden
    hgv_kwh = {
        'hgv_rigid_3_5_7_5t': {
            'laden_0pct':  float(raw.iloc[79, 3]),  # 1.78508 kWh/km
            'laden_50pct': float(raw.iloc[79, 4]),  # 1.94030 kWh/km
            'laden_100pct':float(raw.iloc[79, 5]),  # 2.09552 kWh/km
            'laden_avg':   float(raw.iloc[79, 6]),  # 1.91857 kWh/km  ← project default
        },
        'hgv_rigid_7_5_17t': {
            'laden_avg':   float(raw.iloc[82, 6]),  # 2.33972 kWh/km
        },
        'hgv_rigid_17t_plus': {
            'laden_avg':   float(raw.iloc[85, 6]),  # 3.84574 kWh/km
        },
        'hgv_all_rigids': {
            'laden_avg':   float(raw.iloc[88, 6]),  # 3.25386 kWh/km
        },
        'hgv_artic_3_5_33t': {
            'laden_avg':   float(raw.iloc[91, 6]),  # 3.00114 kWh/km
        },
        'hgv_artic_33t_plus': {
            'laden_avg':   float(raw.iloc[94, 6]),  # 3.57097 kWh/km
        },
        # Overall HGV average — DEFAULT for diesel_truck in synthetic CSV
        'hgv_all': {
            'laden_0pct':  float(raw.iloc[100, 3]), # 2.51355 kWh/km
            'laden_50pct': float(raw.iloc[100, 4]), # 3.19476 kWh/km
            'laden_100pct':float(raw.iloc[100, 5]), # 3.87596 kWh/km
            'laden_avg':   float(raw.iloc[100, 6]), # 3.42464 kWh/km  ← project default
        },
    }
 
    # ── MOTORBIKE (rows 50, 52, 54, 56) — kWh/km ONLY ──────────────────
    # Only SECR kWh/km available. For kg CO2e/km, convert using petrol
    # net CV factor: 1 litre petrol = 8.93 kWh; petrol = 2.31392 kg CO2e/litre
    # => 2.31392 / 8.93 = 0.25912 kg CO2e/kWh (net CV).
    # Average motorbike petrol kg CO2e/km = 0.46265 × 0.25912 = 0.11990
    motorbike_kwh = {
        'small':   float(raw.iloc[50, 3]),   # 0.33622 kWh/km
        'medium':  float(raw.iloc[52, 3]),   # 0.40813 kWh/km
        'large':   float(raw.iloc[54, 3]),   # 0.54296 kWh/km
        'average': float(raw.iloc[56, 3]),   # 0.46265 kWh/km  ← project default
    }
    # Derived kg CO2e/km for motorcycle (petrol net CV conversion)
    PETROL_NET_CV_KG_CO2_PER_KWH = 0.25912  # kg CO2e per kWh (petrol net CV)
    motorbike_co2 = {k: round(v * PETROL_NET_CV_KG_CO2_PER_KWH, 5)
                     for k, v in motorbike_kwh.items()}
    # average: 0.46265 × 0.25912 = 0.11990 kg CO2e/km
 
    return {**van_kwh, **hgv_kwh,
            'motorbike_kwh': motorbike_kwh,
            'motorbike_co2': motorbike_co2}
 
SECR_FACTORS = load_secr_kwh_factors('../data/raw/SECR_kWH_pass.xlsx')
print('Van Class III diesel kWh/km:', SECR_FACTORS['van_class3']['diesel']) # 1.08497
print('All HGVs avg laden kWh/km: ', SECR_FACTORS['hgv_all']['laden_avg'])  # 3.42464
print('Motorbike avg kWh/km:       ', SECR_FACTORS['motorbike_kwh']['average']) # 0.46265
print('Motorbike avg kg CO2e/km:   ', SECR_FACTORS['motorbike_co2']['average']) # 0.11990