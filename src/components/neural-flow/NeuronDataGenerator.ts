
import { NeuronData } from './types';

export const generateInitialNeurons = (count: number = 100): NeuronData[] => {
  const neurons = [];
  const layers = ['input', 'hidden', 'hidden', 'output'];
  const layerNeuronCounts = {
    input: Math.floor(count * 0.2),
    hidden: Math.floor(count * 0.6),
    output: Math.floor(count * 0.2),
  };
  
  let neuronId = 0;
  
  // Create neurons for each layer with appropriate positioning
  layers.forEach((layer, layerIndex) => {
    const layerType = layer as 'input' | 'hidden' | 'output';
    const neuronCount = layer === 'hidden' 
      ? Math.floor(layerNeuronCounts.hidden / 2) 
      : layerNeuronCounts[layerType];
    
    for (let i = 0; i < neuronCount; i++) {
      const x = (layerIndex - 1.5) * 8;
      const spreadFactor = 5.5;
      const y = (Math.random() - 0.5) * spreadFactor;
      const z = (Math.random() - 0.5) * spreadFactor;
      
      neurons.push({
        id: `neuron-${neuronId++}`,
        position: [x, y, z],
        layer: layerType,
        connections: [],
        lastFired: null,
        spikeCount: 0,
        spikeHistory: [],
        trafficState: 'normal',
        intensity: 0.3,
        connectionStrengths: {},
      });
    }
  });
  
  // Create connections between neurons in adjacent layers
  for (let i = 0; i < layers.length - 1; i++) {
    const currentLayerType = layers[i] as 'input' | 'hidden' | 'output';
    const nextLayerType = layers[i + 1] as 'input' | 'hidden' | 'output';
    
    const currentLayerNeurons = neurons.filter(n => n.layer === currentLayerType);
    const nextLayerNeurons = neurons.filter(n => n.layer === nextLayerType);
    
    // Connect each neuron to ~3 neurons in the next layer
    currentLayerNeurons.forEach(neuron => {
      const connectionCount = Math.floor(Math.random() * 3) + 1;
      for (let j = 0; j < connectionCount; j++) {
        const targetIndex = Math.floor(Math.random() * nextLayerNeurons.length);
        if (targetIndex < nextLayerNeurons.length) {
          neuron.connections.push(nextLayerNeurons[targetIndex].id);
          // Initialize connection strength
          neuron.connectionStrengths[nextLayerNeurons[targetIndex].id] = 0.3;
        }
      }
    });
  }
  
  return neurons;
};
