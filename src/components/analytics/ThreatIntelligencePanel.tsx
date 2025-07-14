
import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Shield, AlertTriangle, Globe, Clock, Target, TrendingUp } from "lucide-react";

interface ThreatIntelData {
  source: string;
  threatType: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  confidence: number;
  timestamp: Date;
  geolocation: string;
  description: string;
}

interface ThreatIntelligencePanelProps {
  activeThreats: number;
  anomalyScore: number;
}

export function ThreatIntelligencePanel({ activeThreats, anomalyScore }: ThreatIntelligencePanelProps) {
  const [threatIntel, setThreatIntel] = useState<ThreatIntelData[]>([]);
  const [criticalAlerts, setCriticalAlerts] = useState<string[]>([]);
  const [lastUpdateTime, setLastUpdateTime] = useState<number>(Date.now());

  // Generate realistic threat intelligence data with proper timing intervals
  useEffect(() => {
    const currentTime = Date.now();
    const timeSinceLastUpdate = currentTime - lastUpdateTime;
    const minimumUpdateInterval = 30000; // 30 seconds minimum between updates
    
    // Only update if enough time has passed and there's significant activity
    const shouldUpdate = timeSinceLastUpdate > minimumUpdateInterval && 
                        (anomalyScore > 0.4 || activeThreats > 3);
    
    if (!shouldUpdate) {
      return; // Don't generate new threats too frequently
    }

    const generateThreatIntel = (): ThreatIntelData[] => {
      const sources = ['Shodan', 'VirusTotal', 'Mitre ATT&CK', 'AlienVault', 'Emerging Threats'];
      const threatTypes = ['Malware', 'Botnet', 'APT', 'Phishing', 'DDoS', 'Crypto Mining', 'Ransomware'];
      const geolocations = ['Russia', 'China', 'North Korea', 'Iran', 'Brazil', 'Unknown', 'USA'];
      const severities: ('low' | 'medium' | 'high' | 'critical')[] = ['low', 'medium', 'high', 'critical'];

      // Very conservative threat generation - only during high activity
      const isHighActivity = anomalyScore > 0.6 || activeThreats > 5;
      const isMediumActivity = anomalyScore > 0.4 || activeThreats > 3;
      
      if (!isMediumActivity) {
        // Keep existing threats but don't generate new ones during quiet periods
        return threatIntel;
      }

      // Generate very few threats - max 3 even during high activity
      const maxThreats = isHighActivity ? 3 : isMediumActivity ? 1 : 0;
      const threatCount = Math.floor(Math.random() * maxThreats) + (isHighActivity ? 1 : 0);
      
      if (threatCount === 0) {
        return threatIntel; // No new threats during this update
      }

      // Generate new threats to add to existing ones
      const newThreats = Array.from({ length: threatCount }, (_, i) => {
        // Weight severity based on anomaly score
        const severityWeights = anomalyScore > 0.7 ? ['high', 'critical'] : 
                               anomalyScore > 0.5 ? ['medium', 'high'] : 
                               ['low', 'medium'];
        
        return {
          source: sources[Math.floor(Math.random() * sources.length)],
          threatType: threatTypes[Math.floor(Math.random() * threatTypes.length)],
          severity: severityWeights[Math.floor(Math.random() * severityWeights.length)] as any,
          confidence: Math.floor(Math.random() * 15) + 80 + (anomalyScore * 10), // 80-95%, higher with more anomalies
          timestamp: new Date(currentTime - Math.random() * 600000), // Within last 10 minutes
          geolocation: geolocations[Math.floor(Math.random() * geolocations.length)],
          description: `Network anomaly detected with score ${anomalyScore.toFixed(2)} - ${isHighActivity ? 'coordinated attack pattern' : 'suspicious activity'}`
        };
      });

      // Combine with existing threats and limit total to 8
      const allThreats = [...newThreats, ...threatIntel].slice(0, 8);
      return allThreats.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
    };

    const intel = generateThreatIntel();
    setThreatIntel(intel);
    setLastUpdateTime(currentTime);

    // Generate critical alerts only for high-severity threats
    const alerts = intel
      .filter(threat => threat.severity === 'critical' || (threat.severity === 'high' && threat.confidence > 90))
      .map(threat => `${threat.threatType} from ${threat.geolocation} - ${threat.confidence}% confidence`)
      .slice(0, 2); // Limit to 2 critical alerts max
    
    setCriticalAlerts(alerts);
  }, [activeThreats, anomalyScore, lastUpdateTime, threatIntel]);

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'destructive';
      case 'high': return 'destructive';
      case 'medium': return 'secondary';
      case 'low': return 'outline';
      default: return 'outline';
    }
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'critical': return <AlertTriangle className="h-4 w-4 text-red-500" />;
      case 'high': return <Shield className="h-4 w-4 text-orange-500" />;
      case 'medium': return <Target className="h-4 w-4 text-yellow-500" />;
      case 'low': return <TrendingUp className="h-4 w-4 text-blue-500" />;
      default: return <Shield className="h-4 w-4" />;
    }
  };

  return (
    <div className="space-y-4">
      {/* Critical Alerts */}
      {criticalAlerts.length > 0 && (
        <Alert className="border-red-500/50 bg-red-500/10">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>
            <div className="space-y-1">
              <div className="font-semibold">Critical Threats Detected:</div>
              {criticalAlerts.map((alert, i) => (
                <div key={i} className="text-sm">• {alert}</div>
              ))}
            </div>
          </AlertDescription>
        </Alert>
      )}

      {/* Threat Intelligence Feed */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Shield className="h-5 w-5 text-blue-500" />
            Live Threat Intelligence Feed
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4 max-h-[400px] overflow-y-auto">
            {threatIntel.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                <Shield className="h-8 w-8 mx-auto mb-2 text-green-500" />
                <p className="text-sm">Network monitoring active - No threats detected</p>
                <p className="text-xs mt-1">System operating normally</p>
              </div>
            ) : (
              threatIntel.map((threat, i) => (
                <div key={i} className="p-3 border rounded-lg bg-card/50">
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex items-center gap-2">
                      {getSeverityIcon(threat.severity)}
                      <Badge variant={getSeverityColor(threat.severity) as any}>
                        {threat.severity.toUpperCase()}
                      </Badge>
                      <span className="font-medium">{threat.threatType}</span>
                    </div>
                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                      <Clock className="h-3 w-3" />
                      {threat.timestamp.toLocaleTimeString()}
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-muted-foreground">Source:</span> {threat.source}
                    </div>
                    <div className="flex items-center gap-1">
                      <Globe className="h-3 w-3" />
                      <span className="text-muted-foreground">Location:</span> {threat.geolocation}
                    </div>
                  </div>
                  
                  <div className="mt-2">
                    <div className="flex items-center justify-between text-sm mb-1">
                      <span className="text-muted-foreground">Confidence:</span>
                      <span>{threat.confidence}%</span>
                    </div>
                    <Progress value={threat.confidence} className="h-2" />
                  </div>
                  
                  <p className="text-sm text-muted-foreground mt-2">{threat.description}</p>
                </div>
              ))
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
