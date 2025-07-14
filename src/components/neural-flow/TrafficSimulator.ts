
import { NeuronData, TrafficPattern } from './types';

export class TrafficSimulator {
  private neurons: NeuronData[] = [];
  private anomalyScore: number = 0;
  private isAnalysisActive: boolean = false;

  constructor(neurons: NeuronData[], anomalyScore: number, isAnalysisActive: boolean) {
    this.neurons = neurons;
    this.anomalyScore = anomalyScore;
    this.isAnalysisActive = isAnalysisActive;
  }

  // Update internal state
  updateState(neurons: NeuronData[], anomalyScore: number, isAnalysisActive: boolean) {
    this.neurons = neurons;
    this.anomalyScore = anomalyScore;
    this.isAnalysisActive = isAnalysisActive;
  }

  // Generate traffic patterns based on analysis state
  generateTrafficPatterns(): TrafficPattern[] {
    const patterns: TrafficPattern[] = [];
    
    // Normal traffic - steady background activity
    patterns.push({
      id: 'normal-traffic',
      type: 'normal',
      intensity: this.isAnalysisActive ? 0.6 : 0.3,
      duration: 5000,
      affectedNeurons: this.getRandomNeurons(0.4)
    });

    // Suspicious traffic based on anomaly score
    if (this.anomalyScore > 0.15) {
      patterns.push({
        id: 'suspicious-traffic',
        type: 'suspicious',
        intensity: Math.min(this.anomalyScore * 2, 1),
        duration: 3000,
        affectedNeurons: this.getRandomNeurons(0.2)
      });
    }

    // Anomaly spikes - triggered by high anomaly scores
    if (this.anomalyScore > 0.25 || (this.isAnalysisActive && Math.random() > 0.85)) {
      patterns.push({
        id: 'anomaly-spike',
        type: 'anomaly',
        intensity: 1.0,
        duration: 1500,
        affectedNeurons: this.getRandomNeurons(0.1)
      });
    }

    return patterns;
  }

  // Update neuron states based on traffic patterns
  updateNeuronStates(patterns: TrafficPattern[]): NeuronData[] {
    return this.neurons.map(neuron => {
      let newState = { ...neuron };
      
      // Reset to normal state first
      newState.trafficState = 'normal';
      newState.intensity = 0.3;

      // Apply pattern effects
      patterns.forEach(pattern => {
        if (pattern.affectedNeurons.includes(neuron.id)) {
          newState.trafficState = pattern.type;
          newState.intensity = pattern.intensity;
          
          // Update firing based on traffic type
          if (pattern.type === 'anomaly') {
            newState.lastFired = Date.now() / 1000;
            newState.spikeCount += Math.floor(pattern.intensity * 5);
            newState.spikeHistory = [...newState.spikeHistory.slice(-5), Date.now()];
          } else if (pattern.type === 'suspicious' && Math.random() > 0.6) {
            newState.lastFired = Date.now() / 1000;
            newState.spikeCount += Math.floor(pattern.intensity * 2);
          }
        }
      });

      // Update connection strengths based on activity
      const updatedStrengths: Record<string, number> = {};
      neuron.connections.forEach(connId => {
        const baseStrength = 0.3;
        const activityBonus = newState.intensity * 0.7;
        updatedStrengths[connId] = Math.min(baseStrength + activityBonus, 1.0);
      });
      newState.connectionStrengths = updatedStrengths;

      return newState;
    });
  }

  private getRandomNeurons(percentage: number): string[] {
    const count = Math.floor(this.neurons.length * percentage);
    const shuffled = [...this.neurons].sort(() => 0.5 - Math.random());
    return shuffled.slice(0, count).map(n => n.id);
  }

  // Create cascading spike effects for anomalies
  createSpikesCascade(startNeuronId: string): string[] {
    const visited = new Set<string>();
    const cascade = [startNeuronId];
    
    const propagateSpike = (neuronId: string, depth: number) => {
      if (depth > 3 || visited.has(neuronId)) return;
      visited.add(neuronId);
      
      const neuron = this.neurons.find(n => n.id === neuronId);
      if (!neuron) return;
      
      // Propagate to connected neurons with decreasing probability
      neuron.connections.forEach(connId => {
        if (Math.random() < 0.4 / (depth + 1)) {
          cascade.push(connId);
          propagateSpike(connId, depth + 1);
        }
      });
    };
    
    propagateSpike(startNeuronId, 0);
    return cascade;
  }
}
