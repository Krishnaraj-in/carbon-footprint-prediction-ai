import React from 'react';

const EmissionChart = ({ result }) => {
  if (!result) {
    return <div className="emission-chart"><p>No emission data available</p></div>;
  }

  return (
    <div className="emission-chart p-4 border rounded bg-gray-50">
      <h3 className="text-xl font-bold mb-4">Emission Results</h3>
      
      <div className="space-y-3">
        <div className="flex justify-between">
          <span className="font-semibold">Distance:</span>
          <span>{result.distance_km} km</span>
        </div>
        
        <div className="flex justify-between">
          <span className="font-semibold">DEFRA Baseline:</span>
          <span>{result.defra_emission_kg || 'N/A'} kg CO₂</span>
        </div>

        <div className="flex justify-between">
          <span className="font-semibold">GBR Prediction:</span>
          <span>{result.gbr_prediction_kg || 'N/A'} kg CO₂</span>
        </div>

        <div className="flex justify-between">
          <span className="font-semibold">RF Prediction:</span>
          <span>{result.rf_prediction_kg || 'N/A'} kg CO₂</span>
        </div>

        {result.ci_lower_95 && result.ci_upper_95 && (
          <div className="flex justify-between text-sm text-gray-600">
            <span>95% Confidence Interval:</span>
            <span>{result.ci_lower_95} - {result.ci_upper_95} kg CO₂</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default EmissionChart;