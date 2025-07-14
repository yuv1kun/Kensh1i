
export interface AdvancedMetrics {
  threatIntelligenceScore: number;
  attackVectorDiversity: number;
  payloadAnalysisScore: number;
  neuralPlasticityIndex: number;
  synapticStrengthVariance: number;
  inferenceAcceleration: number;
  trafficEntropy: number;
  protocolDistribution: Record<string, number>;
  geoThreatIntensity: number;
  networkTopologyHealth: number;
}

export class MetricsCalculator {
  static calculateAdvancedMetrics(
    anomalyScore: number,
    activeThreats: number,
    processedPackets: number,
    isAnalysisActive: boolean
  ): AdvancedMetrics {
    const baseTime = Date.now();
    
    return {
      threatIntelligenceScore: Math.min(0.95, 0.7 + (anomalyScore * 0.3) + (Math.sin(baseTime / 10000) * 0.1)),
      attackVectorDiversity: Math.max(0.1, Math.min(1.0, activeThreats * 0.15 + (Math.random() * 0.2))),
      payloadAnalysisScore: Math.min(0.98, 0.8 + (anomalyScore * 0.2) + (Math.cos(baseTime / 8000) * 0.05)),
      neuralPlasticityIndex: isAnalysisActive ? Math.min(0.95, 0.6 + (Math.random() * 0.35)) : 0.3,
      synapticStrengthVariance: Math.min(0.9, 0.4 + (anomalyScore * 0.5) + (Math.sin(baseTime / 6000) * 0.1)),
      inferenceAcceleration: isAnalysisActive ? Math.min(2.5, 1.2 + (Math.random() * 1.3)) : 0.8,
      trafficEntropy: Math.min(0.95, 0.6 + (processedPackets % 100) / 200 + (Math.random() * 0.1)),
      protocolDistribution: {
        TCP: 0.45 + (Math.random() * 0.1),
        UDP: 0.25 + (Math.random() * 0.1),
        ICMP: 0.15 + (Math.random() * 0.05),
        HTTP: 0.35 + (Math.random() * 0.1),
        HTTPS: 0.55 + (Math.random() * 0.1)
      },
      geoThreatIntensity: Math.min(0.9, activeThreats * 0.12 + (Math.sin(baseTime / 12000) * 0.2)),
      networkTopologyHealth: Math.max(0.7, Math.min(0.98, 0.85 + (Math.cos(baseTime / 9000) * 0.13)))
    };
  }
}
