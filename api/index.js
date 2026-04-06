const express = require('express');
const axios   = require('axios');
const cors    = require('cors');
require('dotenv').config();
 
const app = express();
app.use(cors());
app.use(express.json());
 
const FLASK_URL = process.env.FLASK_URL || 'http://localhost:5000';
const ORS_KEY   = process.env.ORS_API_KEY || 'eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6ImJkNmY4OGI4NGEzODRmZWY4ZmI4MmRiMDYzZmQ1MjhiIiwiaCI6Im11cm11cjY0In0=';
 
// Route: fetch ORS geometry + predict emissions
app.post('/api/estimate', async (req, res) => {
    try {
        const { origin, destination, vehicle_type, fuel_type,
                weight_kg, delivery_method, is_rush_hour } = req.body;
 
        // 1. Fetch route from OpenRouteService
        const orsRes = await axios.post(
            'https://api.openrouteservice.org/v2/directions/driving-car/geojson',
            { coordinates: [
                [origin.lng, origin.lat],
                [destination.lng, destination.lat]
            ]},
            { headers: { Authorization: ORS_KEY } }
        );
 
        console.log('ORS Response:', orsRes.data);
        const feature     = orsRes.data.features[0];
        const distance_km = feature.properties.summary.distance / 1000;
        const geometry    = feature.geometry;
 
        // 2. Get current hour for rush-hour detection
        const now          = new Date();
        const hour_of_day  = now.getHours();
        const day_of_week  = now.getDay();
        const month        = now.getMonth() + 1;
        const auto_rush    = is_rush_hour !== undefined ? is_rush_hour
                             : (hour_of_day >= 7 && hour_of_day <= 9)
                               || (hour_of_day >= 16 && hour_of_day <= 18);
 
        // 3. Call Flask ML service
        const flaskRes = await axios.post(`${FLASK_URL}/predict`, {
            distance_km, weight_kg, vehicle_type, fuel_type,
            delivery_method, is_rush_hour: auto_rush ? 1 : 0,
            hour_of_day, day_of_week, month,
        });
 
        res.json({
            ...flaskRes.data,
            distance_km: Math.round(distance_km * 10) / 10,
            route_geometry: geometry,
        });
    } catch (err) {
        console.log(err.message);
        res.status(500).json({ error: err.message });
    }
});

app.get('/', async (req, res) => {
    try {
       
        res.json({
            message: 'Welcome to the Carbon Footprint Predictor API Gateway! Use POST /api/estimate with origin, destination, vehicle_type, fuel_type, weight_kg, delivery_method, and optionally is_rush_hour to get your carbon footprint estimate.'
        });
    } catch (err) {
        console.error(err.message);
        res.status(500).json({ error: err.message });
    }
});
 
app.listen(3001, () => console.log('API Gateway on port 3001'));
