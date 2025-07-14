import { SNNState, SpikingNeuralNetwork, DetectedThreat } from './snn-engine';
import { NetworkDataIngestion } from './data-ingestion';
import { FeatureExtractor } from './feature-extraction';
import { Anomaly, ThreatDetection, AnomalyDetectionConfig, ResponseAction, PipelineStatus } from './types';
import { nanoid, Subject } from './mock-dependencies';

/**
 * AnomalyDetector - Main class that orchestrates the zero-day threat detection pipeline
 * using neuromorphic computing principles
 */
export class AnomalyDetector {
  private snn: SpikingNeuralNetwork;
  private dataIngestion: NetworkDataIngestion;
  private featureExtractor: FeatureExtractor;
  private isActive: boolean = false;
  private anomalyThreshold: number = 0.75;
  private config: AnomalyDetectionConfig;
  private detectedAnomalies: Anomaly[] = [];
  private detectedThreats: ThreatDetection[] = [];
  private recommendedActions: ResponseAction[] = [];
  private status: PipelineStatus;
  private threatClusteringEnabled: boolean = true;
  
  // Observable subjects for monitoring
  private anomalySubject = new Subject<Anomaly>();
  private threatSubject = new Subject<ThreatDetection>();
  private actionSubject = new Subject<ResponseAction>();
  private statusSubject = new Subject<PipelineStatus>();
  private snnStateSubject = new Subject<SNNState>();

  constructor(config?: Partial<AnomalyDetectionConfig>) {
    // Initialize default configuration
    this.config = {
      sensitivityThreshold: 0.75,
      learningRate: 0.01,
      temporalWindowSize: 10,
      minSamples: 100,
      adaptationRate: 0.05,
      featureWeights: {},
      ...config
    };

    // Initialize the SNN with a configuration suitable for network data
    this.snn = new SpikingNeuralNetwork({
      inputSize: 20,
      hiddenLayers: [30, 20],
      outputSize: 5,
      learningRate: this.config.learningRate,
      threshold: 0.5,
      refractoryPeriod: 5
    });

    // Initialize data ingestion
    this.dataIngestion = new NetworkDataIngestion();

    // Initialize feature extractor
    this.featureExtractor = new FeatureExtractor({
      normalizationMethod: 'minmax',
      temporalAggregation: true,
      aggregationWindow: 3000
    });

    // Initialize pipeline status
    this.status = {
      isCapturing: false,
      isProcessing: false,
      isLearning: true,
      captureStartTime: null,
      packetsProcessed: 0,
      anomaliesDetected: 0,
      threatsIdentified: 0,
      alertsGenerated: 0,
      lastUpdateTime: Date.now()
    };

    // Set up the SNN to be notified of threats
    this.snn.registerCallbacks({
      onStateUpdate: (state) => {
        this.snnStateSubject.next(state);
      },
      onThreatDetected: (threat) => {
        this.handleSNNThreat(threat);
      }
    });

    // Configure the SNN anomaly detection
    this.snn.configureAnomalyDetection({
      threshold: this.config.sensitivityThreshold,
      learningEnabled: true
    });
  }

  /**
   * Start the anomaly detection pipeline
   */
  start(networkInterface?: string): boolean {
    if (this.isActive) {
      return true;
    }

    // Start data capture
    const captureSession = this.dataIngestion.startCapture(networkInterface);
    if (!captureSession) {
      return false;
    }

    this.isActive = true;
    this.status.isCapturing = true;
    this.status.isProcessing = true;
    this.status.captureStartTime = Date.now();
    
    // Subscribe to packet data
    this.dataIngestion.subscribeToPackets(packet => {
      this.status.packetsProcessed++;
      this.status.lastUpdateTime = Date.now();
    });

    // Subscribe to extracted features
    this.dataIngestion.subscribeToFeatures(featureVector => {
      // Process features
      const processedFeatures = this.featureExtractor.processFeatures(featureVector);

      if (processedFeatures.length > 0) {
        // Feed processed features to the SNN
        this.snn.processNetworkData(processedFeatures);
      }
    });

    // Notify status change
    this.updateAndEmitStatus();

    return true;
  }

  /**
   * Stop the anomaly detection pipeline
   */
  stop(): void {
    if (!this.isActive) {
      return;
    }

    this.dataIngestion.stopCapture();
    this.isActive = false;
    this.status.isCapturing = false;
    this.status.isProcessing = false;
    
    // Notify status change
    this.updateAndEmitStatus();
  }

  /**
   * Handle a threat detected by the SNN
   */
  private handleSNNThreat(threat: DetectedThreat): void {
    // Create an anomaly from the SNN threat
    const anomaly: Anomaly = {
      id: nanoid(),
      timestamp: threat.timestamp,
      confidence: threat.confidence,
      relatedFeatures: threat.relatedNeurons,
      featureVector: this.featureExtractor.getFeatureBuffer()[0] || {
        timestamp: Date.now(),
        protocol: 0,
        packetSize: 0,
        sourcePortCategory: 0,
        destPortCategory: 0,
        flagsVector: [],
        payloadEntropy: 0,
        srcIpEntropy: 0,
        dstIpEntropy: 0,
        isIntranet: 0,
        headerFields: [],
        interPacketTime: 0,
        packetRatio: 0
      },
      description: threat.description
    };

    // Add to detected anomalies
    this.detectedAnomalies.push(anomaly);
    if (this.detectedAnomalies.length > 100) {
      this.detectedAnomalies.shift();
    }

    // Emit anomaly event
    this.anomalySubject.next(anomaly);
    
    // Update counters
    this.status.anomaliesDetected++;
    this.updateAndEmitStatus();

    // Check if the anomaly should be considered a threat
    if (threat.confidence > this.anomalyThreshold) {
      this.createThreatFromAnomaly(anomaly);
    }
  }

  /**
   * Create a threat from an anomaly
   */
  private createThreatFromAnomaly(anomaly: Anomaly): void {
    // Check if we should cluster with existing threats
    if (this.threatClusteringEnabled) {
      const existingThreat = this.findRelatedThreat(anomaly);
      
      if (existingThreat) {
        // Update existing threat
        existingThreat.anomalies.push(anomaly.id);
        existingThreat.confidence = Math.max(existingThreat.confidence, anomaly.confidence);
        
        // Update severity based on new confidence level
        existingThreat.severity = this.calculateSeverity(existingThreat.confidence);
        
        // Update description
        existingThreat.description = `Updated: ${existingThreat.description}`;
        
        // Emit updated threat
        this.threatSubject.next(existingThreat);
        return;
      }
    }

    // Create new threat
    const threat: ThreatDetection = {
      id: nanoid(),
      timestamp: anomaly.timestamp,
      severity: this.calculateSeverity(anomaly.confidence),
      confidence: anomaly.confidence,
      anomalies: [anomaly.id],
      sourceIps: [],
      destinationIps: [],
      ports: [],
      protocols: [],
      description: `Potential zero-day threat detected with ${Math.round(anomaly.confidence * 100)}% confidence.`,
      recommendedAction: this.generateRecommendedAction(anomaly),
      isZeroDay: anomaly.confidence > 0.9
    };

    // Add to detected threats
    this.detectedThreats.push(threat);
    if (this.detectedThreats.length > 50) {
      this.detectedThreats.shift();
    }

    // Generate response action
    this.generateResponseAction(threat);

    // Emit threat event
    this.threatSubject.next(threat);
    
    // Update counters
    this.status.threatsIdentified++;
    this.updateAndEmitStatus();
  }

  /**
   * Find a related threat to an anomaly for clustering
   */
  private findRelatedThreat(anomaly: Anomaly): ThreatDetection | undefined {
    // Look for recent threats (within 5 minutes)
    const recentThreats = this.detectedThreats.filter(
      t => anomaly.timestamp - t.timestamp < 5 * 60 * 1000
    );

    // Find the most similar threat based on feature overlap
    return recentThreats.find(threat => {
      const anomalies = this.detectedAnomalies.filter(a => 
        threat.anomalies.includes(a.id)
      );

      // Calculate feature similarity
      const similarFeatures = anomaly.relatedFeatures.filter(feature => 
        anomalies.some(a => a.relatedFeatures.includes(feature))
      );

      // If enough features overlap, consider it part of the same threat
      return similarFeatures.length >= 3;
    });
  }

  /**
   * Calculate severity level based on confidence score
   */
  private calculateSeverity(confidence: number): 'low' | 'medium' | 'high' | 'critical' {
    if (confidence > 0.95) return 'critical';
    if (confidence > 0.85) return 'high';
    if (confidence > 0.75) return 'medium';
    return 'low';
  }

  /**
   * Generate a recommended action based on anomaly characteristics
   */
  private generateRecommendedAction(anomaly: Anomaly): string {
    const confidence = anomaly.confidence;
    
    if (confidence > 0.95) {
      return 'Immediate isolation and investigation required.';
    }
    
    if (confidence > 0.85) {
      return 'Block suspicious traffic and escalate to security team.';
    }
    
    if (confidence > 0.75) {
      return 'Monitor closely and collect additional data.';
    }
    
    return 'Analyze pattern for potential false positive.';
  }

  /**
   * Generate a response action for a threat
   */
  private generateResponseAction(threat: ThreatDetection): void {
    const action: ResponseAction = {
      id: nanoid(),
      threatId: threat.id,
      action: this.determineActionType(threat.severity),
      target: 'network',
      parameters: {
        duration: threat.severity === 'critical' ? 'permanent' : '24h',
        scope: threat.severity === 'critical' ? 'global' : 'specific'
      },
      automatedExecutionAllowed: threat.severity === 'critical' ? false : true,
      description: `${threat.severity.toUpperCase()} - ${threat.description}`
    };

    // Add to recommended actions
    this.recommendedActions.push(action);
    if (this.recommendedActions.length > 50) {
      this.recommendedActions.shift();
    }

    // Emit action event
    this.actionSubject.next(action);
    
    // Update counters
    this.status.alertsGenerated++;
    this.updateAndEmitStatus();
  }

  /**
   * Determine appropriate action type based on severity
   */
  private determineActionType(severity: string): 'monitor' | 'alert' | 'block' | 'isolate' | 'analyze' {
    switch (severity) {
      case 'critical':
        return 'isolate';
      case 'high':
        return 'block';
      case 'medium':
        return 'alert';
      default:
        return 'monitor';
    }
  }

  /**
   * Update and emit the current pipeline status
   */
  private updateAndEmitStatus(): void {
    this.status.lastUpdateTime = Date.now();
    this.statusSubject.next({ ...this.status });
  }

  /**
   * Get the current anomaly score from the SNN
   */
  getAnomalyScore(): number {
    return this.snn.getState().anomalyScore;
  }

  /**
   * Get the current active threats count
   */
  getActiveThreatsCount(): number {
    return this.detectedThreats.filter(
      t => Date.now() - t.timestamp < 30 * 60 * 1000 // Active within last 30 mins
    ).length;
  }

  /**
   * Get the current pipeline status
   */
  getStatus(): PipelineStatus {
    return { ...this.status };
  }

  /**
   * Get visualization state compatible with the UI components
   */
  getVisualizationState() {
    return {
      anomalyScore: this.getAnomalyScore(),
      isAnalysisActive: this.isActive,
      activeThreats: this.getActiveThreatsCount(),
      ...this.snn.getVisualizationState()
    };
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
   * Subscribe to action events
   */
  subscribeToActions(callback: (action: ResponseAction) => void): { unsubscribe: () => void } {
    const subscription = this.actionSubject.subscribe(callback);
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
   * Subscribe to SNN state updates
   */
  subscribeToSNNState(callback: (state: SNNState) => void): { unsubscribe: () => void } {
    const subscription = this.snnStateSubject.subscribe(callback);
    return { unsubscribe: () => subscription.unsubscribe() };
  }

  /**
   * Update anomaly detection configuration
   */
  updateConfig(config: Partial<AnomalyDetectionConfig>): void {
    this.config = {
      ...this.config,
      ...config
    };

    // Update SNN configuration
    this.snn.configureAnomalyDetection({
      threshold: this.config.sensitivityThreshold,
      learningEnabled: true
    });
  }
}
