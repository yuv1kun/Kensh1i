
import React from 'react';
import { Badge } from '@/components/ui/badge';

interface NeuronData {
  id: string;
  layer: string;
  spikeCount: number;
  connections: string[];
  lastFired: number | null;
}

interface HoverInfoProps {
  neuron: NeuronData | null;
}

const HoverInfo: React.FC<HoverInfoProps> = ({ neuron }) => {
  if (!neuron) return null;

  return (
    <div 
      className="absolute bg-background/70 backdrop-blur-md p-3 rounded-lg border border-border/30 shadow-lg z-10 text-xs"
      style={{ 
        top: '20px', 
        right: '20px',
        maxWidth: '200px'
      }}
    >
      <div className="font-medium mb-1">{neuron.id}</div>
      <div className="flex items-center gap-1.5 mb-1">
        <span className="font-semibold">Layer:</span> 
        <Badge variant="outline" className="capitalize">{neuron.layer}</Badge>
      </div>
      <div className="mb-1">
        <span className="font-semibold">Spikes:</span> {neuron.spikeCount}
      </div>
      <div className="mb-1">
        <span className="font-semibold">Connections:</span> {neuron.connections.length}
      </div>
      {neuron.lastFired && (
        <div className="text-kenshi-red">
          Last activated {((Date.now() / 1000) - neuron.lastFired).toFixed(1)}s ago
        </div>
      )}
    </div>
  );
};

export default HoverInfo;
