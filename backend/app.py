from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib, numpy as np, pandas as pd, os
from emission_calculator import get_co2_factor, compute_defra_baseline
from data_loader import EMISSION_FACTORS, SECR_FACTORS
from feature_engineering import engineer_features

app = Flask(__name__)
CORS(app)

# Load models at startup (models are in parent directory)
model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
GBR    = joblib.load(os.path.join(model_dir, 'gbr_model.pkl'))
RF     = joblib.load(os.path.join(model_dir, 'rf_model.pkl'))
SCALER = joblib.load(os.path.join(model_dir, 'scaler.pkl'))

# Load feature columns if available, else extract from scaler
feature_cols_path = os.path.join(model_dir, 'feature_cols.pkl')
if os.path.exists(feature_cols_path):
    FEATURE_COLS = joblib.load(feature_cols_path)
else:
    # Extract feature names from the scaler's fitted attributes
    FEATURE_COLS = list(SCALER.feature_names_in_) if hasattr(SCALER, 'feature_names_in_') else None

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'models': ['GBR', 'RF']})
 
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        row = pd.Series({
            'distance_km':         float(data['distance_km']),
            'weight_kg':           float(data['weight_kg']),
            'vehicle_type':        data['vehicle_type'],
            'fuel_type':           data['fuel_type'],
            'avg_traffic_density': float(data.get('avg_traffic_density', 0.6)),
            'temperature_celsius': float(data.get('temperature_celsius', 12)),
            'precipitation_mm':    float(data.get('precipitation_mm', 1.0)),
            'is_rush_hour':        int(data.get('is_rush_hour', 0)),
            'is_weekend':          int(data.get('is_weekend', 0)),
            'hour_of_day':         int(data.get('hour_of_day', 12)),
            'day_of_week':         int(data.get('day_of_week', 2)),
            'month':               int(data.get('month', 6)),
            'package_volume_liters': float(data.get('package_volume_liters', 5)),
            'wind_speed_kmh':      float(data.get('wind_speed_kmh', 15)),
            'delivery_method':     data.get('delivery_method', 'standard'),
            'is_fragile':          int(data.get('is_fragile', 0)),
        })
 
        # DEFRA deterministic baseline
        defra_emission = compute_defra_baseline(row)
 
        # Build feature dataframe, align columns to training set
        df_row = engineer_features(pd.DataFrame([row]))
        
        # Ensure all expected columns are present
        if FEATURE_COLS is not None:
            df_row = df_row.reindex(columns=FEATURE_COLS, fill_value=0)
        else:
            # If no feature columns saved, print available columns for debugging
            print(f"Available columns: {list(df_row.columns)}")
            print(f"Scaler expects: {list(SCALER.feature_names_in_) if hasattr(SCALER, 'feature_names_in_') else 'Unknown'}")
        
        X_scaled = SCALER.transform(df_row)
 
        gbr_pred = float(GBR.predict(X_scaled)[0])
        rf_pred  = float(RF.predict(df_row)[0])
 
        # Confidence interval: ±1.96 × per-tree std (GBR approximation)
        tree_preds = np.array([t.predict(X_scaled) for t in GBR.estimators_[:, 0]])
        ci_lower   = float(np.percentile(tree_preds, 2.5))
        ci_upper   = float(np.percentile(tree_preds, 97.5))
 
        return jsonify({
            'defra_emission_kg':  round(defra_emission, 3),
            'gbr_prediction_kg':  round(gbr_pred, 3),
            'rf_prediction_kg':   round(rf_pred, 3),
            'ci_lower_95':        round(ci_lower, 3),
            'ci_upper_95':        round(ci_upper, 3),
            'defra_factor_used':  get_co2_factor(
                                      data['vehicle_type'], data['fuel_type']),
            'secr_kwh_per_km':    SECR_FACTORS.get(
                                      data['vehicle_type'], {}).get('diesel', 0),
        })
    except Exception as e:
        print(f"Error in predict endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400
 
if __name__ == '__main__':
        app.run(host='0.0.0.0', port=5000, debug=False)