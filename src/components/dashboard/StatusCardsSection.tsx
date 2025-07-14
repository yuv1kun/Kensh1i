import React from 'react';
import { StatusCard } from "@/components/status-card";
import { AdvancedMetrics } from "@/components/advanced-metrics/MetricsCalculator";
import { Shield, Network, Activity, Settings, Heart, Brain, Layers, Cpu, Zap, Globe } from "lucide-react";

interface StatusCardsSectionProps {
  activeThreats: number;
  processedPackets: number;
  anomalyScore: number;
  bandwidthUsage: number;
  neuralEfficiency: number;
  systemHealth: number;
  blockedAttacks: number;
  advancedMetrics: AdvancedMetrics;
}

export function StatusCardsSection({
  activeThreats,
  processedPackets,
  anomalyScore,
  bandwidthUsage,
  neuralEfficiency,
  systemHealth,
  blockedAttacks,
  advancedMetrics
}: StatusCardsSectionProps) {
  return (
    <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-6 gap-4">
      {/* Existing Basic Metrics */}
      <StatusCard 
        title="Active Threats" 
        value={activeThreats}
        description="Currently monitored threats"
        trend={activeThreats > 2 ? "up" : "down"}
        trendValue={`${activeThreats > 2 ? "+" : "-"}${Math.abs(activeThreats - 2)} from baseline`}
        icon={<Shield className="h-4 w-4" />}
      />
      <StatusCard 
        title="Processed Packets" 
        value={processedPackets.toLocaleString()}
        description="Total packets analyzed"
        trend="neutral"
        trendValue="Normal traffic volume"
        icon={<Network className="h-4 w-4" />}
      />
      <StatusCard 
        title="Anomaly Score" 
        value={anomalyScore}
        description="Current network deviation"
        trend={anomalyScore > 0.20 ? "up" : "down"}
        trendValue={`${((anomalyScore - 0.15) * 100).toFixed(0)}% ${anomalyScore > 0.15 ? "increase" : "decrease"}`}
        icon={<Activity className="h-4 w-4" />}
      />
      <StatusCard 
        title="Detection Latency" 
        value="217ms"
        description="Average response time"
        trend="down"
        trendValue="56% faster than traditional"
        icon={<Settings className="h-4 w-4" />}
      />
      <StatusCard 
        title="Bandwidth Usage" 
        value={`${bandwidthUsage.toFixed(1)}%`}
        description="Network utilization"
        trend={bandwidthUsage > 80 ? "up" : bandwidthUsage < 50 ? "down" : "neutral"}
        trendValue={`${bandwidthUsage > 75 ? "High" : bandwidthUsage < 40 ? "Low" : "Normal"} utilization`}
        icon={<Network className="h-4 w-4" />}
      />
      <StatusCard 
        title="Neural Efficiency" 
        value={`${neuralEfficiency.toFixed(1)}%`}
        description="AI processing efficiency"
        trend={neuralEfficiency > 93 ? "up" : "down"}
        trendValue={`${neuralEfficiency > 93 ? "Optimal" : "Standard"} performance`}
        icon={<Activity className="h-4 w-4" />}
      />
      <StatusCard 
        title="System Health" 
        value={`${systemHealth.toFixed(1)}%`}
        description="Overall system status"
        trend={systemHealth > 95 ? "up" : systemHealth < 85 ? "down" : "neutral"}
        trendValue={`${systemHealth > 95 ? "Excellent" : systemHealth < 85 ? "Degraded" : "Good"} condition`}
        icon={<Heart className="h-4 w-4" />}
      />
      <StatusCard 
        title="Blocked Attacks" 
        value={blockedAttacks.toLocaleString()}
        description="Total attacks prevented"
        trend="up"
        trendValue={`+${Math.floor(blockedAttacks * 0.1)} this hour`}
        icon={<Shield className="h-4 w-4" />}
      />

      {/* New Advanced Metrics */}
      <StatusCard 
        title="Threat Intel Score" 
        value={advancedMetrics.threatIntelligenceScore.toFixed(3)}
        description="Intelligence quality index"
        trend={advancedMetrics.threatIntelligenceScore > 0.8 ? "up" : "down"}
        trendValue={`${advancedMetrics.threatIntelligenceScore > 0.8 ? "High" : "Standard"} quality`}
        icon={<Brain className="h-4 w-4" />}
      />
      <StatusCard 
        title="Attack Diversity" 
        value={advancedMetrics.attackVectorDiversity.toFixed(2)}
        description="Vector complexity index"
        trend={advancedMetrics.attackVectorDiversity > 0.5 ? "up" : "down"}
        trendValue={`${advancedMetrics.attackVectorDiversity > 0.5 ? "Complex" : "Simple"} patterns`}
        icon={<Layers className="h-4 w-4" />}
      />
      <StatusCard 
        title="Payload Analysis" 
        value={`${(advancedMetrics.payloadAnalysisScore * 100).toFixed(1)}%`}
        description="Deep inspection score"
        trend={advancedMetrics.payloadAnalysisScore > 0.85 ? "up" : "down"}
        trendValue={`${advancedMetrics.payloadAnalysisScore > 0.85 ? "Excellent" : "Good"} detection`}
        icon={<Cpu className="h-4 w-4" />}
      />
      <StatusCard 
        title="Neural Plasticity" 
        value={`${(advancedMetrics.neuralPlasticityIndex * 100).toFixed(1)}%`}
        description="Adaptation capability"
        trend={advancedMetrics.neuralPlasticityIndex > 0.6 ? "up" : "down"}
        trendValue={`${advancedMetrics.neuralPlasticityIndex > 0.6 ? "High" : "Low"} adaptability`}
        icon={<Brain className="h-4 w-4" />}
      />
      <StatusCard 
        title="Synaptic Strength" 
        value={advancedMetrics.synapticStrengthVariance.toFixed(3)}
        description="Connection variance"
        trend="neutral"
        trendValue="Dynamic learning active"
        icon={<Zap className="h-4 w-4" />}
      />
      <StatusCard 
        title="Inference Speed" 
        value={`${advancedMetrics.inferenceAcceleration.toFixed(1)}x`}
        description="Processing acceleration"
        trend={advancedMetrics.inferenceAcceleration > 1.5 ? "up" : "down"}
        trendValue={`${advancedMetrics.inferenceAcceleration > 1.5 ? "Accelerated" : "Standard"} processing`}
        icon={<Zap className="h-4 w-4" />}
      />
      <StatusCard 
        title="Traffic Entropy" 
        value={advancedMetrics.trafficEntropy.toFixed(3)}
        description="Randomness measure"
        trend={advancedMetrics.trafficEntropy > 0.7 ? "up" : "down"}
        trendValue={`${advancedMetrics.trafficEntropy > 0.7 ? "High" : "Low"} entropy`}
        icon={<Activity className="h-4 w-4" />}
      />
      <StatusCard 
        title="Geo-Threat Level" 
        value={`${(advancedMetrics.geoThreatIntensity * 100).toFixed(0)}%`}
        description="Geographic threat intensity"
        trend={advancedMetrics.geoThreatIntensity > 0.4 ? "up" : "down"}
        trendValue={`${advancedMetrics.geoThreatIntensity > 0.4 ? "Elevated" : "Normal"} regional risk`}
        icon={<Globe className="h-4 w-4" />}
      />
      <StatusCard 
        title="Topology Health" 
        value={`${(advancedMetrics.networkTopologyHealth * 100).toFixed(1)}%`}
        description="Network structure integrity"
        trend={advancedMetrics.networkTopologyHealth > 0.9 ? "up" : "down"}
        trendValue={`${advancedMetrics.networkTopologyHealth > 0.9 ? "Optimal" : "Degraded"} topology`}
        icon={<Network className="h-4 w-4" />}
      />
    </section>
  );
}
