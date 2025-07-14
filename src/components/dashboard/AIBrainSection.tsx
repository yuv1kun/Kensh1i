
import React from 'react';
import { SplineViewer } from "@/components/spline-viewer";
import { NeuralActivityIndicator } from "@/components/neural-activity-indicator";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { AspectRatio } from "@/components/ui/aspect-ratio";
import { Pause, Play } from "lucide-react";

interface AIBrainSectionProps {
  brainSceneUrl: string;
  analysisState: { isRunning: boolean; error: string | null };
  anomalyScore: number;
  activeThreats: number;
  onStartAnalysis: () => void;
  onPauseAnalysis: () => void;
}

export function AIBrainSection({
  brainSceneUrl,
  analysisState,
  anomalyScore,
  activeThreats,
  onStartAnalysis,
  onPauseAnalysis
}: AIBrainSectionProps) {
  return (
    <section className="w-full rounded-lg neumorph overflow-hidden bg-background/50 backdrop-blur">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-0">
        {/* AI Brain Visualization - Takes up 2/3 of the width on large screens */}
        <div className="lg:col-span-2 min-h-[500px] relative">
          <AspectRatio ratio={16/9} className="lg:h-full">
            <SplineViewer scene={brainSceneUrl} className="rounded-l-lg" />
            <NeuralActivityIndicator isActive={analysisState.isRunning} anomalyScore={anomalyScore} />
          </AspectRatio>
          
          {analysisState.isRunning && (
            <div className="absolute top-4 left-4 bg-background/70 backdrop-blur-sm px-4 py-2 rounded-full z-20">
              <div className="flex items-center gap-3">
                <span className="animate-pulse h-3 w-3 bg-kenshi-red rounded-full"></span>
                <span className="font-medium text-sm">Neural Processing Active</span>
              </div>
            </div>
          )}
        </div>
        
        {/* AI Brain Information Panel */}
        <div className="p-6 flex flex-col justify-between">
          <div>
            <h2 className="text-2xl font-bold mb-4 bg-gradient-to-r from-kenshi-blue to-kenshi-red bg-clip-text text-transparent">
              Neuromorphic AI Core
            </h2>
            
            <p className="text-muted-foreground mb-6">
              This spiking neural network processes network traffic in real-time, 
              identifying patterns and anomalies using brain-inspired computing principles.
            </p>
            
            <div className="flex flex-wrap gap-3 mb-6">
              <Badge variant="outline" className="px-3 py-1">
                <span className={`mr-1 h-2 w-2 rounded-full inline-block ${analysisState.isRunning ? 'bg-kenshi-red' : 'bg-kenshi-blue'}`}></span> 
                {analysisState.isRunning ? 'Processing' : 'Monitoring'}
              </Badge>
              <Badge variant="outline" className="px-3 py-1">
                <span className="mr-1">●</span> {activeThreats} Active Threats
              </Badge>
              <Badge variant="outline" className="px-3 py-1">
                <span className="mr-1">●</span> Anomaly Score: {anomalyScore}
              </Badge>
            </div>
          </div>
          
          <div className="p-4 rounded-lg bg-background/50 backdrop-blur border border-border/50">
            <h3 className="text-sm font-medium flex items-center gap-2">
              <span className={`h-2 w-2 rounded-full ${analysisState.isRunning ? 'bg-kenshi-red animate-pulse' : 'bg-kenshi-blue'}`}></span>
              AI Thinking Process
            </h3>
            <p className="text-xs text-muted-foreground mt-2">
              {analysisState.isRunning 
                ? "Currently analyzing network patterns for anomalies using spike-timing-dependent plasticity..."
                : "Idle. Monitoring network traffic for suspicious patterns."}
            </p>
          </div>
          
          {/* Simple press button */}
          <div className="mt-6">
            {analysisState.isRunning ? (
              <Button 
                className="w-full bg-gradient-to-r from-kenshi-red to-kenshi-blue text-white hover:opacity-90 transition-opacity"
                onClick={onPauseAnalysis}
                aria-label="Pause neural analysis"
              >
                <Pause className="mr-2" size={16} />
                Pause Analysis
              </Button>
            ) : (
              <Button 
                className="w-full bg-gradient-to-r from-kenshi-blue to-kenshi-red text-white hover:opacity-90 transition-opacity"
                onClick={onStartAnalysis}
                aria-label="Trigger neural analysis"
              >
                <Play className="mr-2" size={16} />
                Press to Trigger Analysis
              </Button>
            )}
            
            {/* Accessibility announcement for screen readers */}
            <span className="sr-only">
              {analysisState.isRunning 
                ? "Neural analysis is currently running. Press the button to pause." 
                : "Neural analysis is paused. Press the button to start analysis."}
            </span>
          </div>
        </div>
      </div>
    </section>
  );
}
