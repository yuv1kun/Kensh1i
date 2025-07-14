import { brain, neataptic, LSTMTimeStep, nanoid } from './mock-dependencies';

// Export connection type for visualization-connector
export type NeuronState = SpikingNeuron;
export type SynapticConnection = SpikingConnection;

export interface SpikingNeuron {
  id: string;
  potential: number;
  threshold: number;
  refractoryPeriod: number;
  lastSpikeTime: number;
  connections: SpikingConnection[];
  layer: 'input' | 'hidden' | 'output';
  position: [number, number, number];
  spikeHistory: number[];
}

export interface SpikingConnection {
  id: string;
  sourceId: string;
  targetId: string;
  weight: number;
  delay: number;
  plasticityEnabled: boolean;
  lastActivity: number;
}

export interface SNNConfiguration {
  inputSize: number;
  hiddenLayers: number[];
  outputSize: number;
  learningRate: number;
  threshold: number;
  refractoryPeriod: number;
}

export interface SNNState {
  neurons: SpikingNeuron[];
  lastUpdateTime: number;
  currentInput: number[];
  currentOutput: number[];
  anomalyScore: number;
  detectedThreats: DetectedThreat[];
}

export interface DetectedThreat {
  id: string;
  timestamp: number;
  confidence: number;
  relatedNeurons: string[];
  anomalyPattern: number[];
  description: string;
  mitigationSuggested: boolean;
}

/**
 * SpikingNeuralNetwork - A neuromorphic computing implementation using
 * spiking neural networks for zero-day threat detection
 */
export class SpikingNeuralNetwork {
  private config: SNNConfiguration;
  private state: SNNState;
  private brain: LSTMTimeStep;
  private neatNetwork: any; // Neataptic network
  private learningEnabled: boolean = true;
  private anomalyThreshold: number = 0.75;
  private lastSpikePatterns: number[][] = [];
  private callbacks: {
    onStateUpdate?: (state: SNNState) => void;
    onThreatDetected?: (threat: DetectedThreat) => void;
  } = {};

  constructor(config: SNNConfiguration) {
    this.config = config;
    this.state = this.initializeState();
    
    // Initialize brain.js network for time-series anomaly detection
    this.brain = new brain.recurrent.LSTMTimeStep({
      inputSize: config.inputSize,
      hiddenLayers: config.hiddenLayers,
      outputSize: config.outputSize,
      learningRate: config.learningRate,
    });
    
    // Initialize neataptic network for adaptable network structure
    this.neatNetwork = neataptic.Architect.Perceptron(
      config.inputSize,
      config.hiddenLayers[0] || 10, // Use the first hidden layer or default to 10
      config.outputSize
    );
  }
  
  /**
   * Initialize the SNN state with neurons and connections
   */
  private initializeState(): SNNState {
    const neurons: SpikingNeuron[] = [];
    const inputNeurons = Array.from({ length: this.config.inputSize }, (_, i) => ({
      id: `input-${i}`,
      potential: 0,
      threshold: this.config.threshold,
      refractoryPeriod: this.config.refractoryPeriod,
      lastSpikeTime: 0,
      connections: [],
      layer: 'input' as const,
      position: [
        Math.random() * 100 - 50, // x position (-50 to 50)
        Math.random() * 100 - 50, // y position (-50 to 50)
        Math.random() * 100 - 50  // z position (-50 to 50)
      ] as [number, number, number],
      spikeHistory: [],
    }));
    neurons.push(...inputNeurons);
    
    let hiddenLayerOffset = 0;
    this.config.hiddenLayers.forEach((layerSize, layerIndex) => {
      const hiddenNeurons = Array.from({ length: layerSize }, (_, i) => ({
        id: `hidden-${layerIndex}-${i}`,
        potential: 0,
        threshold: this.config.threshold * (0.8 + Math.random() * 0.4),
        refractoryPeriod: this.config.refractoryPeriod,
        lastSpikeTime: 0,
        connections: [],
        layer: 'hidden' as const,
        position: [
          hiddenLayerOffset + (Math.random() * 2), 
          (Math.random() - 0.5) * 8, 
          (Math.random() - 0.5) * 8
        ] as [number, number, number],
        spikeHistory: [],
      }));
      neurons.push(...hiddenNeurons);
      hiddenLayerOffset += 3;
    });
    
    const outputNeurons = Array.from({ length: this.config.outputSize }, (_, i) => ({
      id: `output-${i}`,
      potential: 0,
      threshold: this.config.threshold * 1.2,
      refractoryPeriod: this.config.refractoryPeriod,
      lastSpikeTime: 0,
      connections: [],
      layer: 'output' as const,
      position: [
        hiddenLayerOffset + 3,
        (Math.random() - 0.5) * 8, 
        (Math.random() - 0.5) * 8
      ] as [number, number, number],
      spikeHistory: [],
    }));
    neurons.push(...outputNeurons);
    
    // Create connections between layers
    this.createConnections(neurons);
    
    return {
      neurons,
      lastUpdateTime: Date.now(),
      currentInput: new Array(this.config.inputSize).fill(0),
      currentOutput: new Array(this.config.outputSize).fill(0),
      anomalyScore: 0,
      detectedThreats: [],
    };
  }
  
  /**
   * Create connections between neurons following a feed-forward pattern
   */
  private createConnections(neurons: SpikingNeuron[]): void {
    const inputNeurons = neurons.filter(n => n.layer === 'input');
    const hiddenLayers: SpikingNeuron[][] = [];
    
    // Group hidden neurons by layer
    this.config.hiddenLayers.forEach((_, layerIndex) => {
      hiddenLayers.push(
        neurons.filter(n => n.layer === 'hidden' && n.id.startsWith(`hidden-${layerIndex}`))
      );
    });
    const outputNeurons = neurons.filter(n => n.layer === 'output');
    
    // Connect input to first hidden layer
    inputNeurons.forEach(inputNeuron => {
      hiddenLayers[0].forEach(hiddenNeuron => {
        if (Math.random() < 0.7) { // 70% connection probability
          const connection: SpikingConnection = {
            id: `conn-${inputNeuron.id}-${hiddenNeuron.id}`,
            sourceId: inputNeuron.id,
            targetId: hiddenNeuron.id,
            weight: Math.random() * 0.5 + 0.1,
            delay: Math.floor(Math.random() * 5) + 1,
            plasticityEnabled: true,
            lastActivity: 0
          };
          inputNeuron.connections.push(connection);
        }
      });
    });
    
    // Connect between hidden layers
    for (let i = 0; i < hiddenLayers.length - 1; i++) {
      hiddenLayers[i].forEach(sourceNeuron => {
        hiddenLayers[i + 1].forEach(targetNeuron => {
          if (Math.random() < 0.6) { // 60% connection probability
            const connection: SpikingConnection = {
              id: `conn-${sourceNeuron.id}-${targetNeuron.id}`,
              sourceId: sourceNeuron.id,
              targetId: targetNeuron.id,
              weight: Math.random() * 0.6 + 0.2,
              delay: Math.floor(Math.random() * 4) + 1,
              plasticityEnabled: true,
              lastActivity: 0
            };
            sourceNeuron.connections.push(connection);
          }
        });
      });
    }
    
    // Connect last hidden layer to output
    hiddenLayers[hiddenLayers.length - 1].forEach(hiddenNeuron => {
      outputNeurons.forEach(outputNeuron => {
        if (Math.random() < 0.8) { // 80% connection probability
          const connection: SpikingConnection = {
            id: `conn-${hiddenNeuron.id}-${outputNeuron.id}`,
            sourceId: hiddenNeuron.id,
            targetId: outputNeuron.id,
            weight: Math.random() * 0.7 + 0.3,
            delay: Math.floor(Math.random() * 3) + 1,
            plasticityEnabled: true,
            lastActivity: 0
          };
          hiddenNeuron.connections.push(connection);
        }
      });
    });
  }
  
  /**
   * Process a new network data sample through the SNN
   */
  processNetworkData(data: number[]): SNNState {
    const currentTime = Date.now();
    const elapsedMs = currentTime - this.state.lastUpdateTime;
    
    // Update input neurons with the new data
    const inputNeurons = this.state.neurons.filter(n => n.layer === 'input');
    data.forEach((value, i) => {
      if (i < inputNeurons.length) {
        // Add potential based on input value
        inputNeurons[i].potential += value * 2;
        
        // Check if neuron should spike
        if (inputNeurons[i].potential > inputNeurons[i].threshold && 
            currentTime - inputNeurons[i].lastSpikeTime > inputNeurons[i].refractoryPeriod) {
          this.triggerSpike(inputNeurons[i], currentTime);
        }
      }
    });
    
    this.state.currentInput = data;
    
    // Propagate spikes through the network
    this.propagateSpikes(currentTime);
    
    // Read output layer
    const outputNeurons = this.state.neurons.filter(n => n.layer === 'output');
    this.state.currentOutput = outputNeurons.map(n => 
      n.spikeHistory.filter(t => currentTime - t < 200).length / 5
    );
    
    // Train the brain.js network for anomaly detection
    if (this.learningEnabled && this.lastSpikePatterns.length > 10) {
      this.brain.train([this.lastSpikePatterns.slice(-10)]);
    }
    
    // Calculate anomaly score by comparing predicted vs. actual outputs
    if (this.lastSpikePatterns.length > 10) {
      const predicted = this.brain.forecast(
        this.lastSpikePatterns.slice(-9), 
        1
      )[0];
      
      const actual = this.state.currentOutput;
      const difference = actual.map((a, i) => Math.abs(a - (predicted[i] || 0)));
      const avgDiff = difference.reduce((sum, d) => sum + d, 0) / difference.length;
      
      // Smooth anomaly score with previous value
      this.state.anomalyScore = this.state.anomalyScore * 0.7 + avgDiff * 3 * 0.3;
      
      // Check for anomalies
      this.detectThreats(this.state.anomalyScore, this.state.currentOutput);
    }
    
    // Store current output pattern for future reference
    this.lastSpikePatterns.push(this.state.currentOutput);
    if (this.lastSpikePatterns.length > 100) {
      this.lastSpikePatterns.shift();
    }
    
    // Apply synaptic plasticity (learning)
    if (this.learningEnabled) {
      this.applySynapticPlasticity(elapsedMs);
    }
    
    this.state.lastUpdateTime = currentTime;
    
    // Notify listeners of state update
    if (this.callbacks.onStateUpdate) {
      this.callbacks.onStateUpdate(this.state);
    }
    
    return this.state;
  }
  
  /**
   * Trigger a spike in the specified neuron
   */
  private triggerSpike(neuron: SpikingNeuron, currentTime: number): void {
    neuron.lastSpikeTime = currentTime;
    neuron.spikeHistory.push(currentTime);
    if (neuron.spikeHistory.length > 20) {
      neuron.spikeHistory.shift();
    }
    
    // Reset potential after spike
    neuron.potential = 0;
  }
  
  /**
   * Propagate spikes through the network
   */
  private propagateSpikes(currentTime: number): void {
    // Process neurons in layer order (input -> hidden -> output)
    const layerOrder = ['input', 'hidden', 'output'];
    
    layerOrder.forEach(layer => {
      const layerNeurons = this.state.neurons.filter(n => n.layer === layer);
      
      layerNeurons.forEach(neuron => {
        // Skip if in refractory period
        if (currentTime - neuron.lastSpikeTime <= neuron.refractoryPeriod) {
          return;
        }
        
        // Process incoming connections
        const incomingConnections = this.state.neurons
          .flatMap(n => n.connections)
          .filter(c => c.targetId === neuron.id);
        
        incomingConnections.forEach(conn => {
          const sourceNeuron = this.state.neurons.find(n => n.id === conn.sourceId);
          if (!sourceNeuron) return;
          
          // Check if source neuron has spiked recently considering connection delay
          const recentSpikes = sourceNeuron.spikeHistory.filter(
            spikeTime => currentTime - spikeTime >= conn.delay && 
                         currentTime - spikeTime < conn.delay + 20
          );
          
          if (recentSpikes.length > 0) {
            // Increase potential based on connection weight and spike count
            neuron.potential += conn.weight * recentSpikes.length;
            conn.lastActivity = currentTime;
          }
        });
        
        // Decay potential over time
        neuron.potential *= 0.98;
        
        // Trigger spike if threshold reached
        if (neuron.potential > neuron.threshold) {
          this.triggerSpike(neuron, currentTime);
        }
      });
    });
  }
  
  /**
   * Apply synaptic plasticity (STDP - Spike-Timing-Dependent Plasticity)
   */
  private applySynapticPlasticity(elapsedMs: number): void {
    const currentTime = Date.now();
    
    // Apply plasticity to all connections
    this.state.neurons.forEach(neuron => {
      neuron.connections.forEach(conn => {
        if (!conn.plasticityEnabled) return;
        
        const sourceNeuron = this.state.neurons.find(n => n.id === conn.sourceId);
        const targetNeuron = this.state.neurons.find(n => n.id === conn.targetId);
        
        if (!sourceNeuron || !targetNeuron) return;
        
        // Get recent spikes
        const sourceRecentSpikes = sourceNeuron.spikeHistory
          .filter(t => currentTime - t < 200);
        const targetRecentSpikes = targetNeuron.spikeHistory
          .filter(t => currentTime - t < 200);
        
        if (sourceRecentSpikes.length > 0 && targetRecentSpikes.length > 0) {
          // Find closest spike pair
          let minDiff = Infinity;
          
          for (const sourceSpike of sourceRecentSpikes) {
            for (const targetSpike of targetRecentSpikes) {
              const diff = sourceSpike - targetSpike;
              
              if (Math.abs(diff) < Math.abs(minDiff)) {
                minDiff = diff;
              }
            }
          }
          
          // Apply STDP rule
          if (minDiff > 0 && minDiff < 50) {
            // Source fired after target - decrease weight
            conn.weight = Math.max(0.01, conn.weight - 0.001);
          } else if (minDiff < 0 && minDiff > -50) {
            // Source fired before target - increase weight
            conn.weight = Math.min(1, conn.weight + 0.002);
          }
        }
        
        // Decay unused connections
        if (currentTime - conn.lastActivity > 5000) {
          conn.weight *= 0.999;
        }
      });
    });
  }
  
  /**
   * Detect threats based on anomaly score and firing patterns
   */
  private detectThreats(anomalyScore: number, outputPattern: number[]): void {
    if (anomalyScore > this.anomalyThreshold) {
      // Find suspicious neurons (those with unusual activity)
      const suspiciousNeurons = this.state.neurons
        .filter(n => {
          const currentTime = Date.now();
          const recentSpikes = n.spikeHistory.filter(t => currentTime - t < 500).length;
          return recentSpikes > 5; // Unusually high activity
        })
        .map(n => n.id);
      
      // Create threat object
      const threat: DetectedThreat = {
        id: nanoid(),
        timestamp: Date.now(),
        confidence: Math.min(1, anomalyScore),
        relatedNeurons: suspiciousNeurons,
        anomalyPattern: [...outputPattern],
        description: this.generateThreatDescription(anomalyScore, suspiciousNeurons),
        mitigationSuggested: false
      };
      
      // Add to detected threats
      this.state.detectedThreats.push(threat);
      
      // Limit threat history
      if (this.state.detectedThreats.length > 20) {
        this.state.detectedThreats.shift();
      }
      
      // Notify listeners
      if (this.callbacks.onThreatDetected) {
        this.callbacks.onThreatDetected(threat);
      }
    }
  }
  
  /**
   * Generate a description for the detected threat
   */
  private generateThreatDescription(anomalyScore: number, suspiciousNeuronIds: string[]): string {
    const inputNeurons = suspiciousNeuronIds.filter(id => id.startsWith('input-'));
    const hiddenNeurons = suspiciousNeuronIds.filter(id => id.startsWith('hidden-'));
    const outputNeurons = suspiciousNeuronIds.filter(id => id.startsWith('output-'));
    
    let severityLevel = 'Low';
    if (anomalyScore > 0.9) severityLevel = 'Critical';
    else if (anomalyScore > 0.85) severityLevel = 'High';
    else if (anomalyScore > 0.8) severityLevel = 'Medium';
    
    return `${severityLevel} severity anomaly detected. Unusual activity in ${inputNeurons.length} input neurons, ${hiddenNeurons.length} hidden neurons, and ${outputNeurons.length} output neurons. Confidence: ${Math.round(anomalyScore * 100)}%.`;
  }
  
  /**
   * Register callbacks for state updates and threat detection
   */
  registerCallbacks(callbacks: {
    onStateUpdate?: (state: SNNState) => void;
    onThreatDetected?: (threat: DetectedThreat) => void;
  }) {
    this.callbacks = { ...this.callbacks, ...callbacks };
  }
  
  /**
   * Configure anomaly detection parameters
   */
  configureAnomalyDetection(options: {
    threshold?: number;
    learningEnabled?: boolean;
  }): void {
    if (options.threshold !== undefined) {
      this.anomalyThreshold = options.threshold;
    }
    if (options.learningEnabled !== undefined) {
      this.learningEnabled = options.learningEnabled;
    }
  }
  
  /**
   * Get the current state of the SNN
   */
  getState(): SNNState {
    return this.state;
  }
  
  /**
   * Convert SNN state to a format compatible with the visualization components
   */
  getVisualizationState() {
    return {
      neurons: this.state.neurons.map(n => ({
        id: n.id,
        layer: n.layer,
        position: n.position,
        connections: n.connections.map(c => c.targetId),
        lastFired: n.lastSpikeTime,
        spikeCount: n.spikeHistory.length,
        intensity: n.spikeHistory.filter(t => Date.now() - t < 500).length / 5,
        trafficState: this.getNeuronTrafficState(n)
      })),
      anomalyScore: this.state.anomalyScore,
      activeThreats: this.state.detectedThreats.length,
    };
  }
  
  /**
   * Get traffic state for a neuron based on its activity
   */
  private getNeuronTrafficState(neuron: SpikingNeuron): 'normal' | 'suspicious' | 'anomaly' {
    const currentTime = Date.now();
    const recentSpikes = neuron.spikeHistory.filter(t => currentTime - t < 500).length;
    
    if (recentSpikes > 6) {
      return 'anomaly';
    } else if (recentSpikes > 3) {
      return 'suspicious';
    }
    return 'normal';
  }
}
