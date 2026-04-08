import { useState } from 'react';
import axios from 'axios';
import RouteMap from './RouteMap';
import EmissionChart from './EmissionChart';
 
const VEHICLE_OPTIONS = [
  { value: 'diesel_van',   label: 'Diesel Van',    fuel: 'diesel' },
  { value: 'electric_van', label: 'Electric Van',  fuel: 'electric' },
  { value: 'diesel_truck', label: 'Diesel Truck',  fuel: 'diesel' },
  { value: 'motorcycle',   label: 'Motorcycle',    fuel: 'petrol' },
];

// Function to geocode cities using OpenStreetMap Nominatim API
const geocodeCity = async (cityName) => {
  try {
    const response = await axios.get('https://nominatim.openstreetmap.org/search', {
      params: {
        q: `${cityName}, UK`,
        format: 'json',
        limit: 1,
        countrycodes: 'gb',
        addressdetails: 1
      },
      headers: {
        'User-Agent': 'CarbonFootprintPredictor/1.0'
      }
    });

    if (response.data && response.data.length > 0) {
      const result = response.data[0];
      return {
        lat: parseFloat(result.lat),
        lng: parseFloat(result.lon)
      };
    }

    // Fallback to hardcoded major cities if API doesn't find the city
    const fallbackCities = {
      'london': { lat: 51.5074, lng: -0.1278 },
      'manchester': { lat: 53.4808, lng: -2.2426 },
      'birmingham': { lat: 52.4862, lng: -1.8904 },
      'leeds': { lat: 53.8008, lng: -1.5491 },
      'glasgow': { lat: 55.8642, lng: -4.2518 },
      'sheffield': { lat: 53.3811, lng: -1.4701 },
      'liverpool': { lat: 53.4084, lng: -2.9916 },
      'bristol': { lat: 51.4545, lng: -2.5879 },
      'newcastle': { lat: 54.9783, lng: -1.6178 },
      'nottingham': { lat: 52.9548, lng: -1.1581 },
      'cardiff': { lat: 51.4816, lng: -3.1791 },
      'edinburgh': { lat: 55.9533, lng: -3.1883 },
      'brighton': { lat: 50.8225, lng: -0.1372 },
      'plymouth': { lat: 50.3755, lng: -4.1427 },
      'oxford': { lat: 51.7520, lng: -1.2577 },
      'cambridge': { lat: 52.2053, lng: 0.1218 }
    };

    const city = cityName.toLowerCase().trim();
    if (fallbackCities[city]) {
      return fallbackCities[city];
    }

    throw new Error(`City "${cityName}" not found. Try major UK cities like London, Manchester, Birmingham, etc.`);
  } catch (error) {
    console.error('Geocoding error:', error);
    throw new Error(`Failed to geocode city "${cityName}". Please check the city name and try again.`);
  }
};
 
export default function EmissionEstimator() {
  const [form, setForm] = useState({
    origin_city: '', destination_city: '',
    vehicle_type: 'diesel_van', weight_kg: 10,
    delivery_method: 'standard',
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
 
  const handleSubmit = async () => {
    setLoading(true);
    setError(null);
    try {
      // Geocode city names to lat/lng (use ORS geocoding or static UK city map)
      const origin      = await geocodeCity(form.origin_city);
      const destination = await geocodeCity(form.destination_city);
      const vehicle     = VEHICLE_OPTIONS.find(v => v.value === form.vehicle_type);
 
      const res = await axios.post('http://localhost:3001/api/estimate', {
        origin, destination,
        vehicle_type:    form.vehicle_type,
        fuel_type:       vehicle.fuel,
        weight_kg:       parseFloat(form.weight_kg),
        delivery_method: form.delivery_method,
        is_rush_hour:    0,
      });
      setResult(res.data);
    } catch (e) { 
      const errorMsg = e.response?.data?.error || e.message || 'Unknown error occurred';
      setError(errorMsg);
      console.error('API Error:', errorMsg);
    }
    finally { setLoading(false); }
  };
 
  return (
    <div className="results-page">
      <div className="hero-panel">
        <section className="panel form-panel">
          <span className="eyebrow">Prototype interface</span>
          <h1 className="page-title">Emission results page for deterministic and ML route estimates.</h1>
          <p className="page-subtitle">
            Enter an origin, destination, vehicle profile, and package weight to compare the DEFRA-style deterministic estimate against the model-led prediction layer.
          </p>

          <div className="form-grid">
            <div className="field">
              <label htmlFor="origin_city">Origin city</label>
              <input
                id="origin_city"
                placeholder="Manchester"
                value={form.origin_city}
                onChange={e => setForm({ ...form, origin_city: e.target.value })}
              />
            </div>
            <div className="field">
              <label htmlFor="destination_city">Destination city</label>
              <input
                id="destination_city"
                placeholder="London"
                value={form.destination_city}
                onChange={e => setForm({ ...form, destination_city: e.target.value })}
              />
            </div>
            <div className="field">
              <label htmlFor="vehicle_type">Vehicle type</label>
              <select
                id="vehicle_type"
                value={form.vehicle_type}
                onChange={e => setForm({ ...form, vehicle_type: e.target.value })}
              >
                {VEHICLE_OPTIONS.map(v => (
                  <option key={v.value} value={v.value}>{v.label}</option>
                ))}
              </select>
            </div>
            <div className="field">
              <label htmlFor="delivery_method">Delivery method</label>
              <select
                id="delivery_method"
                value={form.delivery_method}
                onChange={e => setForm({ ...form, delivery_method: e.target.value })}
              >
                <option value="standard">Standard</option>
                <option value="express">Express</option>
                <option value="same_day">Same day</option>
              </select>
            </div>
            <div className="field field-span-2">
              <label htmlFor="weight_kg">Package weight (kg)</label>
              <input
                id="weight_kg"
                type="number"
                min="0"
                step="0.1"
                placeholder="10"
                value={form.weight_kg}
                onChange={e => setForm({ ...form, weight_kg: e.target.value })}
              />
            </div>
          </div>

          <div className="submit-row">
            <button
              className="primary-button"
              onClick={handleSubmit}
              disabled={loading || !form.origin_city || !form.destination_city}
            >
              {loading ? 'Resolving route and emissions...' : 'Generate prototype results'}
            </button>
            <span className="form-note">Uses OpenStreetMap geocoding, OpenRouteService routing, and the GBR/RF backend models.</span>
          </div>

          {error && (
            <div className="error-banner">
              <strong>Request failed.</strong> {error}
            </div>
          )}
        </section>

        <section className="panel hero-summary">
          <div className="summary-topline">
            <div>
              <h2>Results overview</h2>
              <p className="summary-copy">
                The page balances a deterministic baseline, model-derived estimate, uncertainty band, and route context in one review surface.
              </p>
            </div>
            <span className="summary-status">{result ? 'Estimate ready' : 'Awaiting route'}</span>
          </div>

          <div className="hero-metrics">
            <div className="metric-card">
              <span className="metric-label">Distance</span>
              <span className="metric-value">{result ? `${result.distance_km} km` : '--'}</span>
              <span className="metric-detail">Live route length from OpenRouteService.</span>
            </div>
            <div className="metric-card">
              <span className="metric-label">GBR estimate</span>
              <span className="metric-value">{result ? `${result.gbr_prediction_kg} kg` : '--'}</span>
              <span className="metric-detail">Primary machine learning estimate returned by the backend.</span>
            </div>
            <div className="metric-card">
              <span className="metric-label">95% interval</span>
              <span className="metric-value">
                {result ? `${result.ci_lower_95} to ${result.ci_upper_95}` : '--'}
              </span>
              <span className="metric-detail">Approximate uncertainty derived from GBR trees.</span>
            </div>
          </div>

          <p className="summary-copy">
            {result
              ? `Comparing ${form.origin_city} to ${form.destination_city} for a ${form.weight_kg} kg shipment using ${VEHICLE_OPTIONS.find(v => v.value === form.vehicle_type)?.label.toLowerCase()}.`
              : 'Submit a route to populate the comparison panel, confidence interval, and route visualisation.'}
          </p>
        </section>
      </div>

      {result && (
        <div className="results-grid">
          <EmissionChart result={result} />
          <RouteMap
            geometry={result.route_geometry}
            distanceKm={result.distance_km}
            originLabel={form.origin_city}
            destinationLabel={form.destination_city}
          />
        </div>
      )}

      {!result && !loading && (
        <section className="panel empty-state">
          <div>
            <h2 className="card-title">Prototype results will appear here.</h2>
            <p className="card-copy">
              The final layout will show deterministic versus ML comparison, a confidence interval view, and the routed path on a Leaflet map once you submit a scenario.
            </p>
          </div>
        </section>
      )}
    </div>
  );
}