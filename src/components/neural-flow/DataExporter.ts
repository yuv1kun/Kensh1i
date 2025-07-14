
import { NeuronData, TrafficPattern } from './types';

export const exportNeuronDataToCSV = (
  neurons: NeuronData[], 
  trafficPatterns: TrafficPattern[]
): void => {
  // Prepare neuron data
  const neuronRows = neurons.map(neuron => ({
    id: neuron.id,
    layer: neuron.layer,
    position_x: neuron.position[0],
    position_y: neuron.position[1],
    position_z: neuron.position[2],
    connections_count: neuron.connections.length,
    spike_count: neuron.spikeCount,
    traffic_state: neuron.trafficState,
    intensity: neuron.intensity,
    last_fired: neuron.lastFired ? new Date(neuron.lastFired * 1000).toISOString() : '',
    connections: neuron.connections.join(';')
  }));

  // Prepare traffic pattern data
  const patternRows = trafficPatterns.map(pattern => ({
    id: pattern.id,
    type: pattern.type,
    intensity: pattern.intensity,
    duration: pattern.duration,
    affected_neurons_count: pattern.affectedNeurons.length,
    affected_neurons: pattern.affectedNeurons.join(';')
  }));

  // Convert to CSV
  const neuronCSV = convertToCSV(neuronRows);
  const patternCSV = convertToCSV(patternRows);

  // Create and download files
  downloadCSV(neuronCSV, 'neural_network_neurons.csv');
  downloadCSV(patternCSV, 'traffic_patterns.csv');
};

const convertToCSV = (data: any[]): string => {
  if (data.length === 0) return '';
  
  const headers = Object.keys(data[0]);
  const csvRows = [
    headers.join(','),
    ...data.map(row => 
      headers.map(header => {
        const value = row[header];
        // Escape commas and quotes in values
        if (typeof value === 'string' && (value.includes(',') || value.includes('"'))) {
          return `"${value.replace(/"/g, '""')}"`;
        }
        return value;
      }).join(',')
    )
  ];
  
  return csvRows.join('\n');
};

const downloadCSV = (csvContent: string, fileName: string): void => {
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const link = document.createElement('a');
  
  if (link.download !== undefined) {
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', fileName);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }
};
