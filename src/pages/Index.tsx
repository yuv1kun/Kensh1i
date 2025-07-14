
import React, { useEffect, useState, useReducer, useCallback } from "react";
import { Header } from "@/components/header";
import { SplineHero } from "@/components/spline-hero";
import { DataFlowRivers } from "@/components/cyberpunk-visuals/DataFlowRivers";
import { MetricsCalculator, AdvancedMetrics } from "@/components/advanced-metrics/MetricsCalculator";
import { sampleNetworkData, generateRandomNetworkData } from "@/data/sampleNetworkData";
import { toast } from "@/hooks/use-toast";
import { CommandPalette } from "@/components/command-interface/CommandPalette";
import { AIAssistant } from "@/components/command-interface/AIAssistant";
import { CommandTrigger } from "@/components/command-interface/CommandTrigger";
import { DeviceMonitoringVisualization } from "@/components/device-monitoring/DeviceMonitoringVisualization";

// Import refactored dashboard components
import { AIBrainSection } from "@/components/dashboard/AIBrainSection";
import { CyberpunkVisualizationsSection } from "@/components/dashboard/CyberpunkVisualizationsSection";
import { StatusCardsSection } from "@/components/dashboard/StatusCardsSection";
import { NeuralFlowSection } from "@/components/dashboard/NeuralFlowSection";
import { NetworkAnalysisSection } from "@/components/dashboard/NetworkAnalysisSection";
import { AdvancedAnalyticsPanel } from "@/components/analytics/AdvancedAnalyticsPanel";

// Updated Spline scene URL to use the public embed URL
const brainSceneUrl = "https://my.spline.design/particleaibrain-fRP1lexDTOPNTrZqEBAsVvOA/";

// Define types for analysis state management
type AnalysisState = {
  isRunning: boolean;
  error: string | null;
};

type Action =
  | { type: 'TRIGGER' }
  | { type: 'PAUSE' }
  | { type: 'ERROR'; payload: string };

// Define types for network data
type NetworkNode = {
  id: string;
  group: number;
};

type NetworkLink = {
  source: string | NetworkNode;
  target: string | NetworkNode;
  value: number;
};

const Index = () => {
  // Use reducer for analysis state management
  const [analysisState, dispatch] = useReducer(
    (state: AnalysisState, action: Action) => {
      switch (action.type) {
        case 'TRIGGER':
          return { ...state, isRunning: true, error: null };
        case 'PAUSE':
          return { ...state, isRunning: false };
        case 'ERROR':
          return { ...state, error: action.payload };
        default:
          return state;
      }
    },
    { isRunning: false, error: null }
  );

  const [networkData, setNetworkData] = useState(sampleNetworkData);
  const [activeThreats, setActiveThreats] = useState(3);
  const [processedPackets, setProcessedPackets] = useState(12847);
  const [anomalyScore, setAnomalyScore] = useState(0.23);
  const [showNeuronFlow, setShowNeuronFlow] = useState(false);
  
  // New metrics state
  const [bandwidthUsage, setBandwidthUsage] = useState(73.2);
  const [neuralEfficiency, setNeuralEfficiency] = useState(94.8);
  const [systemHealth, setSystemHealth] = useState(96.3);
  const [blockedAttacks, setBlockedAttacks] = useState(147);

  // New advanced metrics state
  const [advancedMetrics, setAdvancedMetrics] = useState<AdvancedMetrics>({
    threatIntelligenceScore: 0.78,
    attackVectorDiversity: 0.34,
    payloadAnalysisScore: 0.89,
    neuralPlasticityIndex: 0.67,
    synapticStrengthVariance: 0.45,
    inferenceAcceleration: 1.23,
    trafficEntropy: 0.72,
    protocolDistribution: { TCP: 0.45, UDP: 0.25, ICMP: 0.15, HTTP: 0.35, HTTPS: 0.55 },
    geoThreatIntensity: 0.29,
    networkTopologyHealth: 0.91
  });

  // New command interface state
  const [isCommandPaletteOpen, setIsCommandPaletteOpen] = useState(false);
  const [isAIAssistantVisible, setIsAIAssistantVisible] = useState(false);

  // Function to start analysis with simple click
  const startAnalysis = useCallback(() => {
    if (analysisState.isRunning) return;
    
    try {
      // In a real app, this would be an API call
      console.log("Starting neural analysis...");
      dispatch({ type: 'TRIGGER' });
      
      toast({
        title: "Neural Analysis Started",
        description: "Neural network is now processing network traffic",
      });
      
      // Show neuron flow visualization when analysis starts
      setShowNeuronFlow(true);
    } catch (error) {
      dispatch({ type: 'ERROR', payload: "Failed to start analysis" });
      
      toast({
        title: "Analysis Error",
        description: "Could not start neural analysis",
        variant: "destructive",
      });
    }
  }, [analysisState.isRunning]);

  // Function to pause analysis
  const pauseAnalysis = useCallback(() => {
    if (!analysisState.isRunning) return;
    
    try {
      // In a real app, this would be an API call
      console.log("Pausing neural analysis...");
      dispatch({ type: 'PAUSE' });
      
      toast({
        title: "Neural Analysis Paused",
        description: "Neural network processing has been paused",
      });
    } catch (error) {
      dispatch({ type: 'ERROR', payload: "Failed to pause analysis" });
    }
  }, [analysisState.isRunning]);

  // Command execution handler
  const handleExecuteCommand = useCallback((command: string, args?: any) => {
    console.log(`Executing command: ${command}`, args);
    
    switch (command) {
      case 'START_ANALYSIS':
        startAnalysis();
        break;
      case 'PAUSE_ANALYSIS':
        pauseAnalysis();
        break;
      case 'RESET_NEURAL':
        // Reset neural network state
        setAnomalyScore(0.23);
        setActiveThreats(1);
        toast({
          title: "Neural Network Reset",
          description: "Neural network has been reset to initial state",
        });
        break;
      case 'ISOLATE_THREATS':
        toast({
          title: "Threats Isolated",
          description: `${args?.count || activeThreats} threats have been quarantined`,
        });
        setActiveThreats(0);
        break;
      case 'DEEP_ANALYSIS':
        toast({
          title: "Deep Analysis Started",
          description: "Comprehensive threat vector analysis initiated",
        });
        break;
      case 'AUTO_RESPONSE':
        toast({
          title: "Auto-Response Enabled",
          description: "Automated threat response protocols activated",
        });
        break;
      case 'NETWORK_SCAN':
        toast({
          title: "Network Scan Initiated",
          description: "Full network topology scan in progress",
        });
        setNetworkData(generateRandomNetworkData());
        break;
      case 'TRAFFIC_ANALYSIS':
        toast({
          title: "Traffic Analysis",
          description: `Analyzing ${args?.packets?.toLocaleString() || processedPackets.toLocaleString()} packets`,
        });
        break;
      case 'OPTIMIZE_BANDWIDTH':
        toast({
          title: "Bandwidth Optimized",
          description: "Network bandwidth allocation has been optimized",
        });
        setBandwidthUsage(prev => Math.max(30, prev - 15));
        break;
      case 'HEALTH_CHECK':
        toast({
          title: "System Health Check",
          description: "Comprehensive system diagnostics completed",
        });
        setSystemHealth(98.5);
        break;
      case 'EXPORT_LOGS':
        toast({
          title: "Logs Exported",
          description: "System analysis logs have been exported",
        });
        break;
      case 'NEURAL_CONFIG':
        toast({
          title: "Neural Configuration",
          description: "Neural network parameters configuration opened",
        });
        break;
      case 'ASK_AI':
        setIsAIAssistantVisible(true);
        toast({
          title: "AI Assistant",
          description: `Processing query: "${args?.query}"`,
        });
        break;
      default:
        console.log(`Unknown command: ${command}`);
    }
  }, [startAnalysis, pauseAnalysis, activeThreats, processedPackets]);

  // Real-time data simulation - ONLY when analysis is running
  useEffect(() => {
    if (!analysisState.isRunning) {
      // When not running, just keep basic static updates
      return;
    }

    const interval = setInterval(() => {
      // Update network data every second for real-time feel
      setNetworkData(generateRandomNetworkData());
      
      // Realistic packet processing simulation
      setProcessedPackets(prev => prev + Math.floor(Math.random() * 150) + 50);
      
      // Dynamic anomaly score fluctuation
      const anomalyFluctuation = (Math.random() - 0.5) * 0.08;
      const newScore = Math.max(0.01, Math.min(0.95, anomalyScore + anomalyFluctuation));
      setAnomalyScore(parseFloat(newScore.toFixed(3)));
      
      // Dynamic threat count changes
      if (Math.random() > 0.85) {
        const threatChange = Math.floor(Math.random() * 3) - 1; // -1, 0, or 1
        setActiveThreats(prev => Math.max(0, Math.min(10, prev + threatChange)));
      }
      
      // Realistic bandwidth usage simulation
      setBandwidthUsage(prev => {
        const change = (Math.random() - 0.5) * 12;
        return Math.max(15, Math.min(98, prev + change));
      });
      
      // Neural efficiency with realistic variations
      setNeuralEfficiency(prev => {
        const change = (Math.random() - 0.5) * 4;
        return Math.max(85, Math.min(99.5, prev + change));
      });
      
      // System health with occasional dips
      setSystemHealth(prev => {
        const change = Math.random() > 0.9 ? -5 : (Math.random() - 0.5) * 2;
        return Math.max(80, Math.min(100, prev + change));
      });
      
      // Blocked attacks increment realistically
      if (Math.random() > 0.7) {
        setBlockedAttacks(prev => prev + Math.floor(Math.random() * 5) + 1);
      }

      // Update advanced metrics with realistic data
      setAdvancedMetrics(MetricsCalculator.calculateAdvancedMetrics(
        newScore,
        activeThreats,
        processedPackets,
        analysisState.isRunning
      ));
      
      // Console log for debugging real-time updates
      if (Math.random() > 0.95) {
        console.log('Real-time data update:', {
          anomalyScore: newScore,
          activeThreats,
          processedPackets,
          isAnalysisRunning: analysisState.isRunning
        });
      }
    }, 1000); // Update every second for real-time simulation

    return () => clearInterval(interval);
  }, [analysisState.isRunning, anomalyScore, activeThreats, processedPackets]);

  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      
      <main className="flex-1 container py-8 px-4 sm:px-8">
        <div className="flex flex-col gap-10">
          {/* New Spline Hero Section - Top Position */}
          <section className="w-full">
            <SplineHero />
          </section>

          {/* AI Brain Section */}
          <AIBrainSection
            brainSceneUrl={brainSceneUrl}
            analysisState={analysisState}
            anomalyScore={anomalyScore}
            activeThreats={activeThreats}
            onStartAnalysis={startAnalysis}
            onPauseAnalysis={pauseAnalysis}
          />

          {/* Cyberpunk Visualizations Section */}
          <CyberpunkVisualizationsSection
            activeThreats={activeThreats}
            anomalyScore={anomalyScore}
          />

          {/* Device Communication Monitoring Section */}
          <DeviceMonitoringVisualization
            isActive={analysisState.isRunning}
            anomalyScore={anomalyScore}
            activeThreats={activeThreats}
          />

          {/* Data Flow Rivers Visualization */}
          <section className="h-[200px] rounded-lg overflow-hidden border border-cyan-500/30 shadow-[0_0_20px_rgba(6,182,212,0.3)]">
            <DataFlowRivers 
              isActive={analysisState.isRunning}
              intensity={anomalyScore}
              className="w-full h-full"
            />
          </section>

          {/* Status Cards Section */}
          <StatusCardsSection
            activeThreats={activeThreats}
            processedPackets={processedPackets}
            anomalyScore={anomalyScore}
            bandwidthUsage={bandwidthUsage}
            neuralEfficiency={neuralEfficiency}
            systemHealth={systemHealth}
            blockedAttacks={blockedAttacks}
            advancedMetrics={advancedMetrics}
          />

          {/* Phase 5: Advanced Analytics Dashboard */}
          <AdvancedAnalyticsPanel
            activeThreats={activeThreats}
            anomalyScore={anomalyScore}
            isAnalysisActive={analysisState.isRunning}
            processedPackets={processedPackets}
          />

          {/* Neural Flow Section */}
          <NeuralFlowSection
            showNeuronFlow={showNeuronFlow}
            onToggleNeuronFlow={() => setShowNeuronFlow(!showNeuronFlow)}
            anomalyScore={anomalyScore}
            isAnalysisActive={analysisState.isRunning}
            activeThreats={activeThreats}
          />

          {/* Network Analysis Section */}
          <NetworkAnalysisSection networkData={networkData} />
        </div>
      </main>

      {/* Command Interface Components */}
      <CommandTrigger
        onOpenCommandPalette={() => setIsCommandPaletteOpen(true)}
        onOpenAIAssistant={() => setIsAIAssistantVisible(true)}
      />
      
      <CommandPalette
        open={isCommandPaletteOpen}
        onOpenChange={setIsCommandPaletteOpen}
        onExecuteCommand={handleExecuteCommand}
        analysisState={analysisState}
        metrics={{
          activeThreats,
          anomalyScore,
          processedPackets
        }}
      />
      
      <AIAssistant
        isVisible={isAIAssistantVisible}
        onClose={() => setIsAIAssistantVisible(false)}
        analysisState={analysisState}
        metrics={{
          activeThreats,
          anomalyScore,
          processedPackets
        }}
      />
    </div>
  );
};

export default Index;
