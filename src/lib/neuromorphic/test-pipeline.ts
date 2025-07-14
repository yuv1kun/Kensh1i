/**
 * Test module to demonstrate the end-to-end neuromorphic
 * zero-day threat detection pipeline
 */

import { NeuromorphicThreatDetection } from './index';
import { FeatureVector } from './types';

/**
 * TestPipeline - Class to test and demonstrate the end-to-end pipeline
 */
export class TestPipeline {
  private threatDetection: NeuromorphicThreatDetection;
  private isRunning: boolean = false;
  private testScenarioInterval: any = null;
  private currentScenario: string = 'normal';
  private testScenarios = [
    'normal',          // Normal traffic pattern
    'highVolume',      // High volume but normal traffic
    'portScan',        // Port scanning activity
    'dataExfiltration',// Data exfiltration pattern
    'zeroDay',         // Simulated zero-day attack
    'normal'           // Return to normal
  ];
  private currentScenarioIndex: number = 0;
  private scenarioDurationMs: number = 30000; // 30 seconds per scenario
  private logCallback: ((message: string) => void) | null = null;

  constructor(logCallback?: (message: string) => void) {
    // Initialize the neuromorphic threat detection system
    this.threatDetection = new NeuromorphicThreatDetection({
      // Configure with settings optimized for testing
      anomalyDetection: {
        sensitivityThreshold: 0.7,
        learningRate: 0.02,
        temporalWindowSize: 8,
        minSamples: 50,
        adaptationRate: 0.1
      },
      alerts: {
        minSeverityToAlert: 'low', // Show all alerts for testing
        notificationSound: true,
        autoAcknowledgeAfterMs: 0  // Don't auto-acknowledge for testing
      }
    });

    if (logCallback) {
      this.logCallback = logCallback;
    }

    // Register event listeners
    this.setupEventListeners();
  }

  /**
   * Set up event listeners for the test
   */
  private setupEventListeners() {
    // Listen for alerts
    this.threatDetection.subscribeToAlerts(alert => {
      this.log(`🚨 ALERT: [${alert.severity.toUpperCase()}] ${alert.title} - ${alert.message}`);
    });

    // Listen for action results
    this.threatDetection.subscribeToActionResults(result => {
      const status = result.success ? '✅ SUCCESS' : '❌ FAILED';
      this.log(`${status}: Action result - ${result.message}`);
    });
  }

  /**
   * Start the test pipeline
   */
  start(): boolean {
    if (this.isRunning) {
      this.log('Test pipeline is already running.');
      return true;
    }

    this.log('Starting neuromorphic threat detection test pipeline...');
    
    // Start the threat detection system (with simulated data)
    const result = this.threatDetection.start();
    if (!result) {
      this.log('❌ Failed to start the threat detection system.');
      return false;
    }

    this.isRunning = true;
    this.log('✅ Neuromorphic threat detection system started successfully.');
    this.log('Initial learning phase with normal traffic...');

    // Start the test scenario progression
    this.startTestScenarios();

    return true;
  }

  /**
   * Stop the test pipeline
   */
  stop(): void {
    if (!this.isRunning) {
      return;
    }

    // Stop the test scenarios
    if (this.testScenarioInterval) {
      clearInterval(this.testScenarioInterval);
      this.testScenarioInterval = null;
    }

    // Stop the threat detection system
    this.threatDetection.stop();
    this.isRunning = false;

    this.log('Neuromorphic threat detection test pipeline stopped.');
  }

  /**
   * Start cycling through test scenarios
   */
  private startTestScenarios(): void {
    // Initial scenario
    this.currentScenarioIndex = 0;
    this.currentScenario = this.testScenarios[0];
    this.log(`Starting test scenario: ${this.formatScenarioName(this.currentScenario)}`);

    // Schedule scenario changes
    this.testScenarioInterval = setInterval(() => {
      this.currentScenarioIndex = (this.currentScenarioIndex + 1) % this.testScenarios.length;
      this.currentScenario = this.testScenarios[this.currentScenarioIndex];
      
      this.log(`Changing to scenario: ${this.formatScenarioName(this.currentScenario)}`);
      
      // If we've gone through all scenarios, stop the test
      if (this.currentScenarioIndex === 0) {
        this.log('Test cycle complete. Continuing with normal traffic...');
      }
    }, this.scenarioDurationMs);
  }

  /**
   * Format scenario name for display
   */
  private formatScenarioName(scenario: string): string {
    switch (scenario) {
      case 'normal':
        return 'Normal Traffic';
      case 'highVolume':
        return 'High Volume Traffic';
      case 'portScan':
        return 'Port Scanning Activity';
      case 'dataExfiltration':
        return 'Data Exfiltration Pattern';
      case 'zeroDay':
        return 'Simulated Zero-Day Attack';
      default:
        return scenario;
    }
  }

  /**
   * Log a message
   */
  private log(message: string): void {
    if (this.logCallback) {
      this.logCallback(`[Test Pipeline] ${message}`);
    } else {
      console.log(`[Test Pipeline] ${message}`);
    }
  }

  /**
   * Get the current test scenario
   */
  getCurrentScenario(): string {
    return this.currentScenario;
  }

  /**
   * Get the status of the pipeline
   */
  getStatus(): any {
    return {
      isRunning: this.isRunning,
      currentScenario: this.formatScenarioName(this.currentScenario),
      ...this.threatDetection.getStatus()
    };
  }

  /**
   * Get visualization state for UI
   */
  getVisualizationState(): any {
    return this.threatDetection.getVisualState();
  }

  /**
   * Get neural activity state for activity indicators
   */
  getNeuralActivityState(): any {
    return this.threatDetection.getNeuralActivityState();
  }

  /**
   * Get active alerts
   */
  getActiveAlerts(): any[] {
    return this.threatDetection.getActiveAlerts();
  }

  /**
   * Execute a pending response action
   */
  executeAction(actionId: string): any {
    return this.threatDetection.executeAction(actionId);
  }

  /**
   * Acknowledge an alert
   */
  acknowledgeAlert(alertId: string): boolean {
    return this.threatDetection.acknowledgeAlert(alertId);
  }

  /**
   * Set the scenario duration in milliseconds
   */
  setScenarioDuration(durationMs: number): void {
    if (durationMs < 5000) {
      // Minimum 5 seconds per scenario
      durationMs = 5000;
    }
    
    this.scenarioDurationMs = durationMs;
    
    // Reset the interval if already running
    if (this.isRunning && this.testScenarioInterval) {
      clearInterval(this.testScenarioInterval);
      this.startTestScenarios();
    }
  }

  /**
   * Subscribe to visual state updates
   */
  subscribeToVisualState(callback: (state: any) => void): { unsubscribe: () => void } {
    return this.threatDetection.subscribeToVisualState(callback);
  }
}
