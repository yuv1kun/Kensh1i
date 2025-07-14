
import React from 'react';
import { NeuronFlowVisualization } from "@/components/neural-flow";
import { Button } from "@/components/ui/button";
import { Layers } from "lucide-react";

interface NeuralFlowSectionProps {
  showNeuronFlow: boolean;
  onToggleNeuronFlow: () => void;
  anomalyScore: number;
  isAnalysisActive: boolean;
  activeThreats: number;
}

export function NeuralFlowSection({
  showNeuronFlow,
  onToggleNeuronFlow,
  anomalyScore,
  isAnalysisActive,
  activeThreats
}: NeuralFlowSectionProps) {
  return (
    <section className="rounded-lg neumorph overflow-hidden">
      <div className="bg-background p-6 border-b border-border/50">
        <div className="flex justify-between items-center">
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <Layers className="h-5 w-5 text-kenshi-blue" />
            Neural Flow Visualization
          </h2>
          <Button 
            variant="outline" 
            size="sm"
            onClick={onToggleNeuronFlow}
          >
            {showNeuronFlow ? 'Hide' : 'Show'} Neural Flow
          </Button>
        </div>
        <p className="text-muted-foreground mt-1">
          Real-time visualization of network traffic processing with anomaly spike detection
        </p>
      </div>
      
      {showNeuronFlow && (
        <div className="h-[500px] w-full">
          <NeuronFlowVisualization 
            anomalyScore={anomalyScore}
            isAnalysisActive={isAnalysisActive}
            activeThreats={activeThreats}
          />
        </div>
      )}
    </section>
  );
}
