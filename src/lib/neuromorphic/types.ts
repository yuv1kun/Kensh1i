/**
 * Types for neuromorphic computing and network data processing
 */

/**
 * Represents a captured network packet
 */
export interface NetworkPacket {
  id: string;
  timestamp: number;
  sourceIP: string;
  destinationIP: string;
  protocol: string;
  size: number;
  sourcePort: number;
  destinationPort: number;
  flags: string[];
  ttl: number;
  payload: Uint8Array | null;
  networkInterface: string; // Changed from 'interface' which is a reserved word
  direction: 'inbound' | 'outbound';
}

/**
 * Abstraction for a packet capture source
 */
export interface PacketSource {
  on(event: string, callback: (raw_packet: any) => void): void;
  close(): void;
}

/**
 * Statistics about captured network traffic
 */
export interface PacketStatistics {
  totalPackets: number;
  totalBytes: number;
  packetsPerSecond: number;
  bytesPerSecond: number;
  protocolDistribution: Record<string, number>;
  portDistribution: Record<string, number>;
}

/**
 * Feature vector extracted from network data for anomaly detection
 */
export interface FeatureVector {
  timestamp: number;
  protocol: number;          // Normalized protocol type
  packetSize: number;        // Normalized packet size
  sourcePortCategory: number;// Port category features
  destPortCategory: number;  // Port category features
  flagsVector: number[];     // Binary features for TCP flags
  payloadEntropy: number;    // Shannon entropy of payload
  srcIpEntropy: number;      // Source IP entropy
  dstIpEntropy: number;      // Destination IP entropy
  isIntranet: number;        // 1 if both IPs are private, 0 otherwise
  headerFields: number[];    // Header-specific features
  interPacketTime: number;   // Time since last packet (normalized)
  packetRatio: number;       // Ratio of different packet types
}

/**
 * An anomaly detected in network traffic
 */
export interface Anomaly {
  id: string;
  timestamp: number;
  confidence: number;
  relatedFeatures: string[];
  featureVector: FeatureVector;
  description: string;
}

/**
 * A detected threat based on anomaly analysis
 */
export interface ThreatDetection {
  id: string;
  timestamp: number;
  severity: 'low' | 'medium' | 'high' | 'critical';
  confidence: number;
  anomalies: string[]; // IDs of related anomalies
  sourceIps: string[];
  destinationIps: string[];
  ports: number[];
  protocols: number[];
  description: string;
  recommendedAction: string;
  isZeroDay: boolean;
}

/**
 * Parameters for configuring the SNN-based anomaly detection
 */
export interface AnomalyDetectionConfig {
  sensitivityThreshold: number;
  learningRate: number;
  temporalWindowSize: number;
  minSamples: number;
  adaptationRate: number;
  featureWeights: Record<string, number>;
}

/**
 * Response recommendation for detected threats
 */
export interface ResponseAction {
  id: string;
  threatId: string;
  action: 'monitor' | 'alert' | 'block' | 'isolate' | 'analyze';
  target: string;
  parameters: Record<string, any>;
  automatedExecutionAllowed: boolean;
  description: string;
}

/**
 * Configuration for the data preprocessing pipeline
 */
export interface PreprocessingConfig {
  normalizationMethod: 'minmax' | 'zscore' | 'log' | 'none';
  temporalAggregation: boolean;
  aggregationWindow: number; // in milliseconds
  featureSelection: string[];
  dimensionalityReduction: boolean;
  pcaComponents?: number;
}

/**
 * Status of the neuromorphic processing pipeline
 */
export interface PipelineStatus {
  isCapturing: boolean;
  isProcessing: boolean;
  isLearning: boolean;
  captureStartTime: number | null;
  packetsProcessed: number;
  anomaliesDetected: number;
  threatsIdentified: number;
  alertsGenerated: number;
  lastUpdateTime: number;
}
