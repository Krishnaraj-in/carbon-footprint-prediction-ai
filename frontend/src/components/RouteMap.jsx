import React, { useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Polyline } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';

// Fix for default markers in react-leaflet
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

const RouteMap = ({ geometry }) => {
  // Calculate map center and bounds from route geometry
  const getMapData = () => {
    if (!geometry || !geometry.coordinates || geometry.coordinates.length === 0) {
      return {
        center: [51.5074, -0.1278], // Default to London
        bounds: null,
        routeCoords: []
      };
    }

    // Convert [lng, lat] to [lat, lng] for Leaflet
    const routeCoords = geometry.coordinates.map(coord => [coord[1], coord[0]]);
    
    // Calculate bounds
    const lats = routeCoords.map(coord => coord[0]);
    const lngs = routeCoords.map(coord => coord[1]);
    const bounds = [
      [Math.min(...lats), Math.min(...lngs)],
      [Math.max(...lats), Math.max(...lngs)]
    ];

    return {
      center: routeCoords[Math.floor(routeCoords.length / 2)], // Center on middle of route
      bounds,
      routeCoords
    };
  };

  const { center, bounds, routeCoords } = getMapData();

  return (
    <div className="route-map p-4 border rounded bg-gray-50">
      <h3 className="text-xl font-bold mb-4">Route Map</h3>
      
      <div className="h-96 w-full rounded-lg overflow-hidden border">
        <MapContainer 
          center={center} 
          zoom={10} 
          style={{ height: '100%', width: '100%' }}
          bounds={bounds}
          boundsOptions={{ padding: [20, 20] }}
        >
          <TileLayer
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          />
          
          {routeCoords.length > 0 && (
            <>
              {/* Route polyline */}
              <Polyline 
                positions={routeCoords}
                color="blue"
                weight={4}
                opacity={0.7}
              />
              
              {/* Start marker */}
              <Marker position={routeCoords[0]}>
                <Popup>Start Point</Popup>
              </Marker>
              
              {/* End marker */}
              <Marker position={routeCoords[routeCoords.length - 1]}>
                <Popup>End Point</Popup>
              </Marker>
            </>
          )}
        </MapContainer>
      </div>
      
      {routeCoords.length > 0 && (
        <div className="mt-4 text-sm text-gray-600">
          <p><strong>Route Details:</strong></p>
          <p>• Distance: {routeCoords.length} coordinate points</p>
          <p>• Start: {routeCoords[0][0].toFixed(4)}, {routeCoords[0][1].toFixed(4)}</p>
          <p>• End: {routeCoords[routeCoords.length - 1][0].toFixed(4)}, {routeCoords[routeCoords.length - 1][1].toFixed(4)}</p>
        </div>
      )}
      
      {routeCoords.length === 0 && (
        <div className="text-center py-8 text-gray-500">
          <p className="text-lg mb-2">🗺️</p>
          <p>No route data available</p>
          <p className="text-sm mt-2">Route map will appear here after calculation</p>
        </div>
      )}
    </div>
  );
};

export default RouteMap;