import { AnomalyDetector } from './anomaly-detector';
import { SNNState, SpikingNeuralNetwork, NeuronState, SynapticConnection } from './snn-engine';
import { ThreatDetection, Anomaly, PipelineStatus } from './types';
import { Subject, BehaviorSubject } from './mock-dependencies';

// Types matching the existing visualization components
interface VisualNeuron {
  id: string;
  x: number;
  y: number;
  z: number;
  layer: string;
  size: number;
  connections: string[];
  state: 'inactive' | 'active' | 'refractory';
  activationLevel: number;
  anomalyLevel: number;
}

interface VisualConnection {
  id: string;
  sourceId: string;
  targetId: string;
  strength: number;
  active: boolean;
  highlighted: boolean;
}

interface VisualState {
  neurons: VisualNeuron[];
  connections: VisualConnection[];
  anomalyScore: number;
  isAnalysisActive: boolean;
  activeThreats: number;
}

/**
 * VisualizationConnector - Adapts the neuromorphic SNN engine outputs
 * to work with the existing visualization components
 */
export class VisualizationConnector {
  private anomalyDetector: AnomalyDetector;
  private visualState: VisualState;
  private stateSubject = new BehaviorSubject<VisualState | null>(null);
  private anomalySubject = new Subject<Anomaly>();
  private threatSubject = new Subject<ThreatDetection>();
  private statusSubject = new Subject<PipelineStatus>();

  private neuronPositions: Map<string, { x: number, y: number, z: number }> = new Map();
  private layerConfig = {
    input: { minZ: -300, maxZ: -200, spread: 300 },
    hidden1: { minZ: -100, maxZ: 0, spread: 250 },
    hidden2: { minZ: 100, maxZ: 200, spread: 200 },
    output: { minZ: 300, maxZ: 400, spread: 150 }
  };

  constructor(anomalyDetector: AnomalyDetector) {
    this.anomalyDetector = anomalyDetector;
    
    // Initialize with empty state
    this.visualState = {
      neurons: [],
      connections: [],
      anomalyScore: 0,
      isAnalysisActive: false,
      activeThreats: 0
    };

    // Subscribe to SNN state updates
    this.anomalyDetector.subscribeToSNNState(this.onSNNStateUpdate.bind(this));
    
    // Subscribe to anomaly events
    this.anomalyDetector.subscribeToAnomalies(anomaly => {
      this.anomalySubject.next(anomaly);
    });
    
    // Subscribe to threat events
    this.anomalyDetector.subscribeToThreats(threat => {
      this.threatSubject.next(threat);
    });
    
    // Subscribe to status updates
    this.anomalyDetector.subscribeToStatus(status => {
      this.statusSubject.next(status);
    });
  }

  /**
   * Start the anomaly detection and visualization pipeline
   */
  startAnalysis(networkInterface?: string): boolean {
    return this.anomalyDetector.start(networkInterface);
  }

  /**
   * Stop the anomaly detection pipeline
   */
  stopAnalysis(): void {
    this.anomalyDetector.stop();
  }

  /**
   * Handle SNN state updates and transform them into visualization-compatible format
   */
  private onSNNStateUpdate(snnState: SNNState): void {
    // Update visual state from SNN state
    this.visualState = this.transformSNNtoVisualState(snnState);
    
    // Emit updated visual state
    this.stateSubject.next(this.visualState);
  }

  /**
   * Transform SNN state to visual state compatible with existing components
   */
  private transformSNNtoVisualState(snnState: SNNState): VisualState {
    // Transform neurons
    const visualNeurons = this.transformNeurons(snnState.neurons);
    
    // Transform connections
    const visualConnections = this.transformConnections(snnState.connections);
    
    // Get other state information
    const visualState: VisualState = {
      neurons: visualNeurons,
      connections: visualConnections,
      anomalyScore: snnState.anomalyScore,
      isAnalysisActive: this.anomalyDetector.getStatus().isProcessing,
      activeThreats: this.anomalyDetector.getActiveThreatsCount()
    };
    
    return visualState;
  }

  /**
   * Transform SNN neurons to visual neurons
   */
  private transformNeurons(neurons: NeuronState[]): VisualNeuron[] {
    return neurons.map(neuron => {
      // Determine neuron layer
      let layer = 'input';
      if (neuron.type === 'output') {
        layer = 'output';
      } else if (neuron.type === 'hidden') {
        // Determine hidden layer based on neuron index
        layer = neuron.layerIndex === 0 ? 'hidden1' : 'hidden2';
      }

      // Get or generate position
      const position = this.getNeuronPosition(neuron.id, layer, neuron.index);
      
      // Determine neuron state
      let state: 'inactive' | 'active' | 'refractory' = 'inactive';
      if (neuron.isRefractory) {
        state = 'refractory';
      } else if (neuron.potential > neuron.threshold * 0.6) {
        state = 'active';
      }

      return {
        id: neuron.id,
        x: position.x,
        y: position.y,
        z: position.z,
        layer,
        size: neuron.anomalyContribution > 0.5 ? 1.5 : 1.0,
        connections: neuron.connections,
        state,
        activationLevel: neuron.potential / neuron.threshold,
        anomalyLevel: neuron.anomalyContribution
      };
    });
  }

  /**
   * Transform SNN connections to visual connections
   */
  private transformConnections(connections: SynapticConnection[]): VisualConnection[] {
    return connections.map(connection => {
      return {
        id: connection.id,
        sourceId: connection.sourceId,
        targetId: connection.targetId,
        strength: connection.weight,
        active: connection.lastActivation > Date.now() - 200, // Active if fired within 200ms
        highlighted: connection.plasticity > 0.5  // Highlight connections with high plasticity
      };
    });
  }

  /**
   * Get or generate a position for a neuron
   */
  private getNeuronPosition(id: string, layer: string, index: number): { x: number, y: number, z: number } {
    // Return cached position if available
    if (this.neuronPositions.has(id)) {
      return this.neuronPositions.get(id)!;
    }
    
    // Generate position based on layer and index
    const config = this.layerConfig[layer as keyof typeof this.layerConfig];
    const z = config.minZ + Math.random() * (config.maxZ - config.minZ);
    
    // Calculate x and y based on layer spread
    // This distributes neurons in a circular pattern within each layer
    const radius = config.spread;
    const angle = (index / 10) * Math.PI * 2; // Distribute neurons around the circle
    const x = Math.cos(angle) * radius + (Math.random() * 40 - 20); // Add some randomness
    const y = Math.sin(angle) * radius + (Math.random() * 40 - 20); // Add some randomness
    
    // Cache and return position
    const position = { x, y, z };
    this.neuronPositions.set(id, position);
    return position;
  }

  /**
   * Subscribe to visual state updates
   */
  subscribeToVisualState(callback: (state: VisualState) => void): { unsubscribe: () => void } {
    // Skip null values (initial state)
    const subscription = this.stateSubject.subscribe(state => {
      if (state !== null) {
        callback(state);
      }
    });
    return { unsubscribe: () => subscription.unsubscribe() };
  }

  /**
   * Subscribe to anomaly events
   */
  subscribeToAnomalies(callback: (anomaly: Anomaly) => void): { unsubscribe: () => void } {
    const subscription = this.anomalySubject.subscribe(callback);
    return { unsubscribe: () => subscription.unsubscribe() };
  }

  /**
   * Subscribe to threat events
   */
  subscribeToThreats(callback: (threat: ThreatDetection) => void): { unsubscribe: () => void } {
    const subscription = this.threatSubject.subscribe(callback);
    return { unsubscribe: () => subscription.unsubscribe() };
  }

  /**
   * Subscribe to status updates
   */
  subscribeToStatus(callback: (status: PipelineStatus) => void): { unsubscribe: () => void } {
    const subscription = this.statusSubject.subscribe(callback);
    return { unsubscribe: () => subscription.unsubscribe() };
  }

  /**
   * Get current visual state
   */
  getVisualState(): VisualState {
    return { ...this.visualState };
  }

  /**
   * Get a simplified representation of the current state for the neural activity indicator
   */
  getNeuralActivityState() {
    return {
      anomalyScore: this.visualState.anomalyScore,
      activeNeurons: this.visualState.neurons.filter(n => n.state === 'active').length,
      totalNeurons: this.visualState.neurons.length,
      isAnalysisActive: this.visualState.isAnalysisActive,
      activeThreats: this.visualState.activeThreats
    };
  }
}
