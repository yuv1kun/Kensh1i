import { FeatureVector, NetworkPacket, PreprocessingConfig } from './types';
import { Subject } from './mock-dependencies';

/**
 * FeatureExtractor - Responsible for extracting and preprocessing features
 * from network data for use in neuromorphic anomaly detection
 */
export class FeatureExtractor {
  private config: PreprocessingConfig;
  private featureHistory: FeatureVector[] = [];
  private featureBuffer: FeatureVector[] = [];
  private statisticsWindow: Record<string, number[]> = {};
  private processedFeaturesSubject = new Subject<number[]>();
  private isInitialized = false;
  private minMaxValues: Record<string, { min: number; max: number }> = {};
  private meanStdValues: Record<string, { mean: number; std: number }> = {};

  constructor(config?: Partial<PreprocessingConfig>) {
    this.config = {
      normalizationMethod: 'minmax',
      temporalAggregation: true,
      aggregationWindow: 5000, // 5 seconds
      featureSelection: [],
      dimensionalityReduction: false,
      ...config
    };
  }

  /**
   * Initialize the feature extractor with initial data
   */
  initialize(initialFeatures: FeatureVector[]): void {
    if (initialFeatures.length > 0) {
      this.featureHistory = initialFeatures.slice(-100);
      this.calculateStatistics();
    }
    this.isInitialized = true;
  }

  /**
   * Process a new feature vector
   */
  processFeatures(features: FeatureVector): number[] {
    // Add to history
    this.featureHistory.push(features);
    if (this.featureHistory.length > 1000) {
      this.featureHistory.shift();
    }

    // Add to buffer for temporal aggregation
    this.featureBuffer.push(features);

    // Perform temporal aggregation if enabled
    if (this.config.temporalAggregation) {
      // Clean up old features
      const oldestValidTime = features.timestamp - this.config.aggregationWindow;
      this.featureBuffer = this.featureBuffer.filter(f => f.timestamp >= oldestValidTime);

      // Only process if we have enough samples
      if (this.featureBuffer.length < 3) {
        return [];
      }
    }

    // Extract the numerical feature vector
    let processedFeatures = this.extractNumericalFeatures(features);
    
    // Normalize features
    processedFeatures = this.normalizeFeatures(processedFeatures);
    
    // Apply feature selection if configured
    if (this.config.featureSelection.length > 0) {
      processedFeatures = this.selectFeatures(processedFeatures);
    }
    
    // Apply dimensionality reduction if enabled
    if (this.config.dimensionalityReduction && this.config.pcaComponents) {
      processedFeatures = this.reduceDimensionality(processedFeatures);
    }
    
    // Emit processed features
    this.processedFeaturesSubject.next(processedFeatures);
    
    return processedFeatures;
  }

  /**
   * Extract numerical features from a feature vector
   */
  private extractNumericalFeatures(features: FeatureVector): number[] {
    // Convert our feature vector to a flat array of numbers
    const numericalFeatures: number[] = [
      features.protocol,
      features.packetSize,
      features.sourcePortCategory,
      features.destPortCategory,
      ...features.flagsVector,
      features.payloadEntropy,
      features.srcIpEntropy,
      features.dstIpEntropy,
      features.isIntranet,
      ...features.headerFields,
      features.interPacketTime,
      features.packetRatio
    ];
    
    // If temporal aggregation is enabled, add time-based features
    if (this.config.temporalAggregation && this.featureBuffer.length >= 3) {
      // Calculate rate of change for key features
      const packetSizes = this.featureBuffer.map(f => f.packetSize);
      numericalFeatures.push(this.calculateSlope(packetSizes));
      
      // Entropy change over time
      const entropies = this.featureBuffer.map(f => f.payloadEntropy);
      numericalFeatures.push(this.calculateSlope(entropies));
      
      // Burstiness (variance in inter-packet times)
      const times = this.featureBuffer.map(f => f.interPacketTime);
      numericalFeatures.push(this.calculateVariance(times));
    }
    
    return numericalFeatures;
  }

  /**
   * Normalize features according to the configured method
   */
  private normalizeFeatures(features: number[]): number[] {
    if (!this.isInitialized) {
      return features;
    }
    
    switch (this.config.normalizationMethod) {
      case 'minmax':
        return this.applyMinMaxNormalization(features);
      case 'zscore':
        return this.applyZScoreNormalization(features);
      case 'log':
        return features.map(f => f > 0 ? Math.log(1 + f) : 0);
      default:
        return features;
    }
  }

  /**
   * Apply Min-Max normalization
   */
  private applyMinMaxNormalization(features: number[]): number[] {
    // If min/max values haven't been calculated yet, use current data
    if (Object.keys(this.minMaxValues).length === 0) {
      this.calculateMinMaxValues();
    }
    
    return features.map((value, index) => {
      const key = `feature_${index}`;
      const { min, max } = this.minMaxValues[key] || { min: 0, max: 1 };
      
      if (max === min) return 0.5;
      return (value - min) / (max - min);
    });
  }

  /**
   * Apply Z-Score normalization
   */
  private applyZScoreNormalization(features: number[]): number[] {
    // If mean/std values haven't been calculated yet, use current data
    if (Object.keys(this.meanStdValues).length === 0) {
      this.calculateMeanStdValues();
    }
    
    return features.map((value, index) => {
      const key = `feature_${index}`;
      const { mean, std } = this.meanStdValues[key] || { mean: 0, std: 1 };
      
      if (std === 0) return 0;
      return (value - mean) / std;
    });
  }

  /**
   * Apply feature selection based on configuration
   */
  private selectFeatures(features: number[]): number[] {
    if (this.config.featureSelection.length === 0) {
      return features;
    }
    
    // Define feature indices to keep
    // This would ideally be based on the feature names specified in featureSelection
    // For this implementation, we'll just use indices
    const selectedIndices: number[] = this.config.featureSelection
      .map(name => parseInt(name.replace('feature_', ''), 10))
      .filter(index => !isNaN(index) && index >= 0 && index < features.length);
    
    if (selectedIndices.length === 0) {
      return features;
    }
    
    return selectedIndices.map(index => features[index]);
  }

  /**
   * Apply dimensionality reduction (simplified PCA implementation)
   * Note: A real implementation would use a proper PCA library
   */
  private reduceDimensionality(features: number[]): number[] {
    // For demonstration, we'll just return a subset of features
    // In a real implementation, this would use PCA or another technique
    const n = this.config.pcaComponents || 5;
    return features.slice(0, Math.min(n, features.length));
  }

  /**
   * Calculate statistics for normalization
   */
  private calculateStatistics(): void {
    this.calculateMinMaxValues();
    this.calculateMeanStdValues();
  }

  /**
   * Calculate min and max values for each feature
   */
  private calculateMinMaxValues(): void {
    if (this.featureHistory.length === 0) {
      return;
    }
    
    // First, extract numerical features from all historical records
    const allFeatures = this.featureHistory.map(this.extractNumericalFeatures.bind(this));
    
    // For each feature position, calculate min and max
    if (!allFeatures || !Array.isArray(allFeatures) || allFeatures.length === 0) return;
    
    const numFeatures = allFeatures[0]?.length || 0;
    if (numFeatures === 0) return;
    
    for (let i = 0; i < numFeatures; i++) {
      const featureValues = allFeatures.map(features => features[i]);
      const min = Math.min(...featureValues);
      const max = Math.max(...featureValues);
      
      this.minMaxValues[`feature_${i}`] = { min, max };
    }
  }

  /**
   * Calculate mean and standard deviation for each feature
   */
  private calculateMeanStdValues(): void {
    if (this.featureHistory.length === 0) {
      return;
    }
    
    // First, extract numerical features from all historical records
    const allFeatures = this.featureHistory.map(this.extractNumericalFeatures.bind(this));
    
    // For each feature position, calculate mean and std
    if (!allFeatures || !Array.isArray(allFeatures) || allFeatures.length === 0) return;
    
    const numFeatures = allFeatures[0]?.length || 0;
    if (numFeatures === 0) return;
    
    for (let i = 0; i < numFeatures; i++) {
      const featureValues = allFeatures.map(features => features[i]);
      
      // Calculate mean
      const sum = featureValues.reduce((acc, val) => acc + val, 0);
      const mean = sum / featureValues.length;
      
      // Calculate standard deviation
      const squaredDiffs = featureValues.map(val => Math.pow(val - mean, 2));
      const avgSquaredDiff = squaredDiffs.reduce((acc, val) => acc + val, 0) / squaredDiffs.length;
      const std = Math.sqrt(avgSquaredDiff);
      
      this.meanStdValues[`feature_${i}`] = { mean, std };
    }
  }

  /**
   * Subscribe to processed feature events
   */
  subscribeToProcessedFeatures(callback: (features: number[]) => void): { unsubscribe: () => void } {
    const subscription = this.processedFeaturesSubject.subscribe(callback);
    return {
      unsubscribe: () => subscription.unsubscribe()
    };
  }

  /**
   * Calculate the slope of values over time (rate of change)
   */
  private calculateSlope(values: number[]): number {
    if (values.length < 2) return 0;
    
    // Simple linear regression to find slope
    const n = values.length;
    const xValues = Array.from({ length: n }, (_, i) => i);
    
    // Calculate means
    const xMean = xValues.reduce((a, b) => a + b, 0) / n;
    const yMean = values.reduce((a, b) => a + b, 0) / n;
    
    // Calculate slope
    let numerator = 0;
    let denominator = 0;
    
    for (let i = 0; i < n; i++) {
      numerator += (xValues[i] - xMean) * (values[i] - yMean);
      denominator += Math.pow(xValues[i] - xMean, 2);
    }
    
    if (denominator === 0) return 0;
    
    const slope = numerator / denominator;
    return slope;
  }

  /**
   * Calculate variance of a set of values
   */
  private calculateVariance(values: number[]): number {
    if (values.length < 2) return 0;
    
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const squaredDiffs = values.map(val => Math.pow(val - mean, 2));
    return squaredDiffs.reduce((a, b) => a + b, 0) / values.length;
  }

  /**
   * Update configuration
   */
  updateConfig(config: Partial<PreprocessingConfig>): void {
    this.config = {
      ...this.config,
      ...config
    };
    
    // Recalculate statistics if needed
    if (this.isInitialized) {
      this.calculateStatistics();
    }
  }

  /**
   * Get the current feature buffer
   */
  getFeatureBuffer(): FeatureVector[] {
    return [...this.featureBuffer];
  }
}
