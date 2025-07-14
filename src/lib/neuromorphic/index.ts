/**
 * Main entry point for the neuromorphic zero-day threat detection system
 * This module integrates all components and exposes a unified API
 */

import { SpikingNeuralNetwork } from './snn-engine';
import { NetworkDataIngestion } from './data-ingestion';
import { FeatureExtractor } from './feature-extraction';
import { AnomalyDetector } from './anomaly-detector';
import { VisualizationConnector } from './visualization-connector';
import { AlertSystem, Alert, AlertConfig, ActionResult } from './alert-system';
import { 
  NetworkPacket, 
  FeatureVector, 
  Anomaly, 
  ThreatDetection, 
  ResponseAction, 
  PipelineStatus,
  AnomalyDetectionConfig,
  PreprocessingConfig
} from './types';

/**
 * NeuromorphicThreatDetection - Main class that integrates all components
 */
export class NeuromorphicThreatDetection {
  private anomalyDetector: AnomalyDetector;
  private visualizationConnector: VisualizationConnector;
  private alertSystem: AlertSystem;
  
  constructor(config?: {
    anomalyDetection?: Partial<AnomalyDetectionConfig>;
    preprocessing?: Partial<PreprocessingConfig>;
    alerts?: Partial<AlertConfig>;
  }) {
    // Initialize the anomaly detector (which contains the SNN and data ingestion)
    this.anomalyDetector = new AnomalyDetector(config?.anomalyDetection);
    
    // Initialize the visualization connector
    this.visualizationConnector = new VisualizationConnector(this.anomalyDetector);
    
    // Initialize the alert system
    this.alertSystem = new AlertSystem(config?.alerts);
    
    // Connect anomaly detector threats/anomalies to alert system
    this.setupEventSubscriptions();
  }
  
  /**
   * Setup event subscriptions between components
   */
  private setupEventSubscriptions() {
    // Forward threats from anomaly detector to alert system
    this.anomalyDetector.subscribeToThreats(threat => {
      this.alertSystem.processThreatDetection(threat);
    });
    
    // Forward anomalies from anomaly detector to alert system
    this.anomalyDetector.subscribeToAnomalies(anomaly => {
      this.alertSystem.processAnomaly(anomaly);
    });
    
    // Process recommended actions
    this.anomalyDetector.subscribeToActions(action => {
      this.alertSystem.processResponseAction(action);
    });
  }
  
  /**
   * Start the neuromorphic threat detection pipeline
   */
  start(networkInterface?: string): boolean {
    return this.anomalyDetector.start(networkInterface);
  }
  
  /**
   * Stop the neuromorphic threat detection pipeline
   */
  stop(): void {
    this.anomalyDetector.stop();
  }
  
  /**
   * Get the current visual state for rendering in the UI
   * This is compatible with the existing visualization components
   */
  getVisualState() {
    return this.visualizationConnector.getVisualState();
  }
  
  /**
   * Get a simplified state for the neural activity indicator
   */
  getNeuralActivityState() {
    return this.visualizationConnector.getNeuralActivityState();
  }
  
  /**
   * Get the current status of the pipeline
   */
  getStatus(): PipelineStatus {
    return this.anomalyDetector.getStatus();
  }
  
  /**
   * Get active alerts
   */
  getActiveAlerts(): Alert[] {
    return this.alertSystem.getActiveAlerts();
  }
  
  /**
   * Get all alerts (including acknowledged ones)
   */
  getAllAlerts(): Alert[] {
    return this.alertSystem.getAllAlerts();
  }
  
  /**
   * Get pending response actions
   */
  getPendingActions(): ResponseAction[] {
    return this.alertSystem.getPendingActions();
  }
  
  /**
   * Execute a response action by its ID
   */
  executeAction(actionId: string): ActionResult {
    return this.alertSystem.executeAction(actionId);
  }
  
  /**
   * Acknowledge an alert
   */
  acknowledgeAlert(alertId: string): boolean {
    return this.alertSystem.acknowledgeAlert(alertId);
  }
  
  /**
   * Subscribe to visual state updates for UI components
   */
  subscribeToVisualState(callback: (state: any) => void) {
    return this.visualizationConnector.subscribeToVisualState(callback);
  }
  
  /**
   * Subscribe to alerts
   */
  subscribeToAlerts(callback: (alert: Alert) => void) {
    return this.alertSystem.subscribeToAlerts(callback);
  }
  
  /**
   * Subscribe to active alerts
   */
  subscribeToActiveAlerts(callback: (alerts: Alert[]) => void) {
    return this.alertSystem.subscribeToActiveAlerts(callback);
  }
  
  /**
   * Subscribe to action results
   */
  subscribeToActionResults(callback: (result: ActionResult) => void) {
    return this.alertSystem.subscribeToActionResults(callback);
  }
  
  /**
   * Update the anomaly detection configuration
   */
  updateAnomalyDetectionConfig(config: Partial<AnomalyDetectionConfig>): void {
    this.anomalyDetector.updateConfig(config);
  }
  
  /**
   * Update the alert system configuration
   */
  updateAlertConfig(config: Partial<AlertConfig>): void {
    this.alertSystem.updateConfig(config);
  }
}

// Export all components and types for direct usage if needed
export {
  SpikingNeuralNetwork,
  NetworkDataIngestion,
  FeatureExtractor,
  AnomalyDetector,
  VisualizationConnector,
  AlertSystem,
  // Types
  NetworkPacket,
  FeatureVector,
  Anomaly,
  ThreatDetection,
  ResponseAction,
  PipelineStatus,
  Alert,
  AlertConfig,
  ActionResult,
  AnomalyDetectionConfig,
  PreprocessingConfig
};
