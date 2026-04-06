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
    <div className='max-w-4xl mx-auto p-6'>
      <h1 className='text-2xl font-bold text-blue-800 mb-4'>
        Carbon Footprint Estimator
      </h1>
      {/* Input form */}
      <div className='grid grid-cols-2 gap-4 mb-6'>
        <input className='border rounded p-2'
          placeholder='Origin city (e.g. Manchester) - OpenStreetMap geocoding'
          value={form.origin_city}
          onChange={e => setForm({...form, origin_city: e.target.value})} />
        <input className='border rounded p-2'
          placeholder='Destination city (e.g. London) - OpenStreetMap geocoding'
          value={form.destination_city}
          onChange={e => setForm({...form, destination_city: e.target.value})} />
        <select className='border rounded p-2'
          value={form.vehicle_type}
          onChange={e => setForm({...form, vehicle_type: e.target.value})}>
          {VEHICLE_OPTIONS.map(v => (
            <option key={v.value} value={v.value}>{v.label}</option>
          ))}
        </select>
        <input type='number' className='border rounded p-2'
          placeholder='Package weight (kg)'
          value={form.weight_kg}
          onChange={e => setForm({...form, weight_kg: e.target.value})} />
      </div>
      <button
        className='bg-blue-700 text-white px-6 py-2 rounded hover:bg-blue-800 disabled:opacity-50'
        onClick={handleSubmit} 
        disabled={loading || !form.origin_city || !form.destination_city}>
        {loading ? 'Calculating...' : 'Estimate Emissions'}
      </button>
      
      {error && (
        <div className='mt-4 p-4 bg-red-100 border border-red-400 text-red-700 rounded'>
          <p><strong>Error:</strong> {error}</p>
        </div>
      )}
      
      {result && (
        <div className='mt-8 grid grid-cols-2 gap-6'>
          <EmissionChart result={result} />
          <RouteMap geometry={result.route_geometry} />
        </div>
      )}
    </div>
  );
}