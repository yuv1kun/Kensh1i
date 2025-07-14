
import React from 'react';
import { TrafficPattern } from './types';

interface TrafficIndicatorProps {
  anomalyScore: number;
  trafficPatterns: TrafficPattern[];
}

const TrafficIndicator: React.FC<TrafficIndicatorProps> = ({ anomalyScore, trafficPatterns }) => {
  return (
    <div className="absolute top-4 left-4 bg-background/80 backdrop-blur-sm px-3 py-2 rounded-md text-xs z-20">
      <div className="flex items-center gap-2">
        <div className={`h-2 w-2 rounded-full ${
          anomalyScore > 0.25 ? 'bg-red-500 animate-pulse' : 
          anomalyScore > 0.15 ? 'bg-orange-500' : 'bg-green-500'
        }`}></div>
        <span className="font-medium">
          {anomalyScore > 0.25 ? 'Anomaly Detected' : 
           anomalyScore > 0.15 ? 'Suspicious Activity' : 'Normal Traffic'}
        </span>
      </div>
      <div className="text-muted-foreground mt-1">
        Score: {anomalyScore.toFixed(3)} | Patterns: {trafficPatterns.length}
      </div>
    </div>
  );
};

export default TrafficIndicator;
