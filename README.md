# carbon-footprint-prediction-ai

A research project comparing ML-based and deterministic approaches to carbon emission estimation for UK e-commerce last-mile delivery. Built as part of my MSc Computer Science dissertation at Coventry University.

The core question I was trying to answer: does adding contextual delivery variables (traffic, load, weather, time of day) to an ML model actually give you meaningfully different emission estimates compared to just using the standard DEFRA distance × emission factor formula?

Short answer: yes, especially in urban conditions. The GBR model came in 51.7% higher than the deterministic baseline for a peak-hour urban delivery — which makes intuitive sense when you think about stop-start driving in congestion.

What this does

- Generates ~8,964 synthetic UK delivery scenarios with 15 variables
- Calculates emissions using the DEFRA 2024 deterministic formula as the baseline
- Trains a Gradient Boosting Regressor and Random Forest on the same data
- Compares ML estimates vs deterministic across different delivery profiles
- Exposes predictions via a Flask API connected to a React frontend
- Shows route on a Leaflet map using OpenRouteService for real UK route geometry
- Uses SHAP to explain which features drive predictions

Results

The GBR model ended up with R² = 0.995 and MAE of 5.43 kg CO₂ on the test set. The Random Forest was close behind at R² = 0.988. Both are approximating a known deterministic function with ±5% noise added, so high R² is expected — the more interesting finding is how much the estimates diverge on specific scenario types.

| Scenario | Deterministic | GBR | Difference |
|---|---|---|---|
| Suburban, low traffic (15km, idx 0.7) | 1.82 kg | 1.74 kg | −4.4% |
| Urban, peak hour (8km, idx 1.8) | 3.21 kg | 4.87 kg | **+51.7%** |
| Motorway, off-peak (80km, idx 0.9) | 18.34 kg | 17.91 kg | −2.3% |
| Urban, high load (12km, idx 1.6) | 8.76 kg | 12.43 kg | **+41.9%** |
| Rural, standard (45km, idx 0.8) | 8.92 kg | 8.67 kg | −2.8% |

Motorway and rural scenarios are nearly identical between the two approaches — which makes sense since DEFRA factors were calibrated under those kinds of conditions. The urban congestion cases are where the ML model picks up non-linear interactions that the linear deterministic formula misses.

SHAP analysis confirmed distance is the dominant feature (mean |SHAP| = 3.84), followed by traffic index (1.93) and vehicle load (1.67). The noise factor scored 0.09 — meaning the model learned the underlying structure rather than fitting to noise, which is what you want.


Project layout

```
├── api/                  # Node.js / Express gateway
│   └── index.js
├── backend/              # Python ML side
│   ├── app.py            # Flask entry point
│   ├── data_loader.py
│   ├── emission_calculator.py   # DEFRA formula
│   ├── feature_engineering.py  # derived features
│   ├── model_training.py
│   ├── learning_curve.py
│   └── outputs/          # saved charts
├── data/
│   └── synthetic_deliveries.csv
├── frontend/             # React app
│   └── src/
├── models/
│   └── gbr_model.pkl
├── graph.py
└── performance_bar.py
```

---

Stack

Backend: Python 3.10, Flask, Scikit-learn 1.3, SHAP, Pandas, NumPy, SciPy

Frontend: React 18, Tailwind CSS, Leaflet.js, Chart.js

API layer: Node.js, Express

Database: MongoDB Atlas

External: OpenRouteService API (route geometry)

Deployed on: Vercel (frontend) + Railway (Flask + Node)

---

Running it locally

You need Python 3.10+, Node 18+, a MongoDB Atlas URI and an ORS API key.

```bash
git clone https://github.com/Krishnaraj-in/carbon-footprint-prediction-ai.git
cd carbon-footprint-prediction-ai
```

Flask ML service:
```bash
cd backend
pip install -r requirements.txt
python model_training.py   # trains and saves the models
python app.py              # runs on :5000
```

Node API gateway:
```bash
cd api
npm install
node index.js              # runs on :3001
```

React frontend:
```bash
cd frontend
npm install
npm start                  # runs on :3000
```

Create a `.env` in `/api`:
```
ORS_API_KEY=your_key
MONGODB_URI=your_atlas_uri
FLASK_URL=http://localhost:5000
```


Dataset

Synthetic data generated with NumPy and Pandas. Each scenario has 15 variables — distance, load, traffic index, vehicle type, rush hour flag, weather penalty, ambient temp, time of day, day of week, route efficiency, noise factor, urban flag, stop count, vehicle capacity and payload ratio.

Emissions are computed from the DEFRA 2024 formula:

```
Emission = Distance × BaseFactor × (1 + Load/50) × (1 + Traffic × 0.3)
           × WeatherPenalty × RushHourPenalty × Noise
```

Base factors: diesel van = 0.14533 kg CO₂e/km, light HGV = 0.19285, HGV = 0.25547. A ±5% noise factor is added so the ML task is not trivially easy. 500 real UK routes from ORS are included to give the distance distribution some geographic grounding.

---

Limitations worth being upfront about

The dataset is entirely synthetic. The ML models are learning to approximate a known formula — they are not trained on real vehicle telematics. So while the methodology is sound for a simulation study, I would not use these specific models for real-world Scope 3 reporting without validation against actual data first.

The circular validation problem (generating labels with a formula then predicting them with ML) is acknowledged and discussed in the dissertation. The contribution is the framework and the finding that ML captures non-linear contextual interactions better than the linear deterministic approach — not a claim that it outperforms real-world measurement.

---

Methodology note

Follows Design Science Research (Hevner et al., 2004). Training used an 80/20 split with 5-fold CV and a constrained grid search (27 combinations per model). Paired t-test confirmed GBR outperforms RF at statistical significance (t = −13.469, p = 0.0002).

---

Context

MSc Computer Science dissertation — Coventry University 2025/26
---
