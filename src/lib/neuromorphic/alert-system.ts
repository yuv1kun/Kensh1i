import { ThreatDetection, Anomaly, ResponseAction } from './types';
import { Subject, BehaviorSubject, nanoid } from './mock-dependencies';

/**
 * Types specific to the alert and response system
 */
export interface Alert {
  id: string;
  timestamp: number;
  title: string;
  message: string;
  severity: 'info' | 'warning' | 'critical' | 'error';
  source: 'anomaly' | 'threat' | 'system';
  relatedEntityId?: string;
  acknowledged: boolean;
  metadata: Record<string, any>;
}

export interface AlertConfig {
  minSeverityToAlert: 'low' | 'medium' | 'high' | 'critical';
  notificationSound: boolean;
  autoAcknowledgeAfterMs: number; // 0 means never auto-acknowledge
  maxAlertsToStore: number;
  autoExecuteActions: boolean;
}

export interface ActionResult {
  id: string;
  actionId: string;
  success: boolean;
  timestamp: number;
  message: string;
  data?: any;
}

/**
 * AlertSystem - Responsible for generating alerts from threats and anomalies
 * and tracking responses/actions
 */
export class AlertSystem {
  private alerts: Alert[] = [];
  private pendingActions: ResponseAction[] = [];
  private executedActions: ActionResult[] = [];
  private config: AlertConfig;
  
  // Observable subjects
  private alertSubject = new Subject<Alert>();
  private actionSubject = new Subject<ResponseAction>();
  private resultSubject = new Subject<ActionResult>();
  private activeAlertsSubject = new BehaviorSubject<Alert[]>([]);

  // Audio for notifications
  private notificationSound: HTMLAudioElement | null = null;

  constructor(config?: Partial<AlertConfig>) {
    // Set default configuration
    this.config = {
      minSeverityToAlert: 'medium',
      notificationSound: true,
      autoAcknowledgeAfterMs: 24 * 60 * 60 * 1000, // 24 hours
      maxAlertsToStore: 100,
      autoExecuteActions: false,
      ...config
    };
    
    // Initialize notification sound if in browser environment
    if (typeof window !== 'undefined' && this.config.notificationSound) {
      this.initializeNotificationSound();
    }
  }

  /**
   * Initialize notification sound for browser environments
   */
  private initializeNotificationSound(): void {
    try {
      this.notificationSound = new Audio();
      this.notificationSound.src = 'data:audio/mp3;base64,SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU4Ljc2LjEwMAAAAAAAAAAAAAAA/+M4wAAAAAAAAAAAAEluZm8AAAAPAAAAAwAAAbAAzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzM//////////////////////////////////////////////////////////////////8AAAAATGF2YzU4LjEzAAAAAAAAAAAAAAAAJAYHg/8='
      this.notificationSound.load();
    } catch (err) {
      console.error('Failed to initialize notification sound', err);
      this.notificationSound = null;
    }
  }

  /**
   * Process a new threat detection and generate appropriate alerts
   */
  processThreatDetection(threat: ThreatDetection): void {
    // Check if the threat meets the minimum severity for alerting
    const severityMap: Record<string, number> = {
      'low': 1,
      'medium': 2,
      'high': 3,
      'critical': 4
    };
    
    const threatSeverityLevel = severityMap[threat.severity] || 0;
    const minSeverityLevel = severityMap[this.config.minSeverityToAlert] || 0;
    
    if (threatSeverityLevel >= minSeverityLevel) {
      // Create an alert for the threat
      const alert: Alert = {
        id: nanoid(),
        timestamp: Date.now(),
        title: `${threat.severity.toUpperCase()} Threat Detected`,
        message: threat.description,
        severity: this.mapThreatSeverityToAlertSeverity(threat.severity),
        source: 'threat',
        relatedEntityId: threat.id,
        acknowledged: false,
        metadata: {
          confidence: threat.confidence,
          isZeroDay: threat.isZeroDay,
          protocols: threat.protocols,
          sourceIps: threat.sourceIps,
          destinationIps: threat.destinationIps
        }
      };
      
      // Add to alerts list
      this.addAlert(alert);
    }
  }

  /**
   * Process a new anomaly and generate appropriate alerts if needed
   */
  processAnomaly(anomaly: Anomaly): void {
    // Only alert on high-confidence anomalies
    if (anomaly.confidence > 0.9) {
      const alert: Alert = {
        id: nanoid(),
        timestamp: Date.now(),
        title: `High-Confidence Anomaly Detected`,
        message: `Network anomaly detected with ${Math.round(anomaly.confidence * 100)}% confidence.`,
        severity: 'warning',
        source: 'anomaly',
        relatedEntityId: anomaly.id,
        acknowledged: false,
        metadata: {
          confidence: anomaly.confidence,
          relatedFeatures: anomaly.relatedFeatures
        }
      };
      
      // Add to alerts list
      this.addAlert(alert);
    }
  }

  /**
   * Process a recommended action for a threat
   */
  processResponseAction(action: ResponseAction): void {
    // Add to pending actions
    this.pendingActions.push(action);
    
    // Emit action event
    this.actionSubject.next(action);
    
    // Auto-execute the action if configured and allowed
    if (this.config.autoExecuteActions && action.automatedExecutionAllowed) {
      this.executeAction(action.id);
    } else {
      // Create an alert for manual action
      const alert: Alert = {
        id: nanoid(),
        timestamp: Date.now(),
        title: `Action Required: ${this.formatActionType(action.action)}`,
        message: action.description,
        severity: 'info',
        source: 'system',
        relatedEntityId: action.id,
        acknowledged: false,
        metadata: {
          threatId: action.threatId,
          action: action.action,
          target: action.target,
          parameters: action.parameters
        }
      };
      
      // Add to alerts list
      this.addAlert(alert);
    }
  }

  /**
   * Execute a pending action by its ID
   */
  executeAction(actionId: string): ActionResult {
    // Find the action
    const actionIndex = this.pendingActions.findIndex(a => a.id === actionId);
    if (actionIndex === -1) {
      const result: ActionResult = {
        id: nanoid(),
        actionId,
        success: false,
        timestamp: Date.now(),
        message: 'Action not found'
      };
      this.resultSubject.next(result);
      return result;
    }
    
    // Remove from pending actions
    const action = this.pendingActions.splice(actionIndex, 1)[0];
    
    // Simulate action execution
    // In a real system, this would interface with security tools, firewalls, etc.
    const success = Math.random() > 0.1; // 90% success rate for simulation
    
    const result: ActionResult = {
      id: nanoid(),
      actionId: action.id,
      success,
      timestamp: Date.now(),
      message: success 
        ? `Successfully executed ${action.action} on ${action.target}` 
        : `Failed to execute ${action.action} on ${action.target}`,
      data: {
        actionType: action.action,
        target: action.target,
        parameters: action.parameters
      }
    };
    
    // Add to executed actions
    this.executedActions.push(result);
    if (this.executedActions.length > 100) {
      this.executedActions.shift();
    }
    
    // Emit result event
    this.resultSubject.next(result);
    
    // Create an alert for the action result
    const alert: Alert = {
      id: nanoid(),
      timestamp: Date.now(),
      title: success ? 'Action Completed' : 'Action Failed',
      message: result.message,
      severity: success ? 'info' : 'error',
      source: 'system',
      relatedEntityId: action.id,
      acknowledged: false,
      metadata: {
        threatId: action.threatId,
        actionType: action.action,
        target: action.target,
        success
      }
    };
    
    // Add to alerts list
    this.addAlert(alert);
    
    return result;
  }

  /**
   * Add a new alert to the system
   */
  private addAlert(alert: Alert): void {
    // Add to alerts list
    this.alerts.push(alert);
    
    // Trim if we have too many alerts
    if (this.alerts.length > this.config.maxAlertsToStore) {
      this.alerts.shift();
    }
    
    // Emit alert event
    this.alertSubject.next(alert);
    
    // Update active alerts
    this.updateActiveAlerts();
    
    // Play notification sound if enabled
    if (this.config.notificationSound && this.notificationSound) {
      try {
        this.notificationSound.play().catch(e => console.error('Error playing notification sound:', e));
      } catch (err) {
        console.error('Failed to play notification sound', err);
      }
    }
    
    // Set timeout for auto-acknowledge if configured
    if (this.config.autoAcknowledgeAfterMs > 0) {
      setTimeout(() => {
        this.acknowledgeAlert(alert.id);
      }, this.config.autoAcknowledgeAfterMs);
    }
  }

  /**
   * Acknowledge an alert by its ID
   */
  acknowledgeAlert(alertId: string): boolean {
    const alert = this.alerts.find(a => a.id === alertId);
    if (!alert) {
      return false;
    }
    
    alert.acknowledged = true;
    this.updateActiveAlerts();
    return true;
  }

  /**
   * Update the active alerts list
   */
  private updateActiveAlerts(): void {
    const activeAlerts = this.alerts.filter(a => !a.acknowledged);
    this.activeAlertsSubject.next(activeAlerts);
  }

  /**
   * Map threat severity to alert severity
   */
  private mapThreatSeverityToAlertSeverity(threatSeverity: string): 'info' | 'warning' | 'critical' | 'error' {
    switch (threatSeverity) {
      case 'critical':
        return 'critical';
      case 'high':
        return 'critical';
      case 'medium':
        return 'warning';
      case 'low':
      default:
        return 'info';
    }
  }

  /**
   * Format action type for display
   */
  private formatActionType(action: string): string {
    switch (action) {
      case 'monitor':
        return 'Monitor Traffic';
      case 'alert':
        return 'Security Alert';
      case 'block':
        return 'Block Traffic';
      case 'isolate':
        return 'Isolate System';
      case 'analyze':
        return 'Deep Analysis';
      default:
        return action.charAt(0).toUpperCase() + action.slice(1);
    }
  }

  /**
   * Get all alerts
   */
  getAllAlerts(): Alert[] {
    return [...this.alerts];
  }

  /**
   * Get active (unacknowledged) alerts
   */
  getActiveAlerts(): Alert[] {
    return this.alerts.filter(a => !a.acknowledged);
  }

  /**
   * Get pending actions
   */
  getPendingActions(): ResponseAction[] {
    return [...this.pendingActions];
  }

  /**
   * Get executed action results
   */
  getExecutedActions(): ActionResult[] {
    return [...this.executedActions];
  }

  /**
   * Subscribe to new alerts
   */
  subscribeToAlerts(callback: (alert: Alert) => void): { unsubscribe: () => void } {
    const subscription = this.alertSubject.subscribe(callback);
    return { unsubscribe: () => subscription.unsubscribe() };
  }

  /**
   * Subscribe to active alerts updates
   */
  subscribeToActiveAlerts(callback: (alerts: Alert[]) => void): { unsubscribe: () => void } {
    const subscription = this.activeAlertsSubject.subscribe(callback);
    return { unsubscribe: () => subscription.unsubscribe() };
  }

  /**
   * Subscribe to new actions
   */
  subscribeToActions(callback: (action: ResponseAction) => void): { unsubscribe: () => void } {
    const subscription = this.actionSubject.subscribe(callback);
    return { unsubscribe: () => subscription.unsubscribe() };
  }

  /**
   * Subscribe to action results
   */
  subscribeToActionResults(callback: (result: ActionResult) => void): { unsubscribe: () => void } {
    const subscription = this.resultSubject.subscribe(callback);
    return { unsubscribe: () => subscription.unsubscribe() };
  }

  /**
   * Update the alert system configuration
   */
  updateConfig(config: Partial<AlertConfig>): void {
    this.config = {
      ...this.config,
      ...config
    };
    
    // Re-initialize notification sound if setting changed
    if (typeof window !== 'undefined' && config.notificationSound !== undefined) {
      if (config.notificationSound && !this.notificationSound) {
        this.initializeNotificationSound();
      }
    }
  }
}
