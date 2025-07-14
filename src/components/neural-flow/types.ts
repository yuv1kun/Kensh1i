
export interface NeuronData {
  id: string;
  position: number[];
  layer: string;
  connections: string[];
  lastFired: number | null;
  spikeCount: number;
  spikeHistory: number[];
  trafficState: 'normal' | 'suspicious' | 'anomaly';
  intensity: number;
  connectionStrengths: Record<string, number>;
  graphPosition?: {
    x: number;
    y: number;
    radius: number;
  };
}

export interface LayerVisibility {
  input: boolean;
  hidden: boolean;
  output: boolean;
}

export interface TrafficPattern {
  id: string;
  type: 'normal' | 'suspicious' | 'anomaly';
  intensity: number;
  duration: number;
  affectedNeurons: string[];
}
