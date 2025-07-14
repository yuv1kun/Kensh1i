
import React from 'react';
import { HolographicDisplay } from "@/components/cyberpunk-visuals/HolographicDisplay";
import { CyberThreatHeatMap } from "@/components/cyberpunk-visuals/CyberThreatHeatMap";
import { AttackVectorRadar } from "@/components/cyberpunk-visuals/AttackVectorRadar";

interface CyberpunkVisualizationsSectionProps {
  activeThreats: number;
  anomalyScore: number;
}

export function CyberpunkVisualizationsSection({
  activeThreats,
  anomalyScore
}: CyberpunkVisualizationsSectionProps) {
  return (
    <section className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <div className="overflow-hidden">
        <HolographicDisplay title="Threat Intelligence Hub" glowColor="red" className="h-[300px] w-full">
          <div className="h-full w-full overflow-hidden">
            <CyberThreatHeatMap 
              threats={activeThreats} 
              anomalyScore={anomalyScore}
              className="h-full w-full"
            />
          </div>
        </HolographicDisplay>
      </div>
      
      <div className="overflow-hidden">
        <HolographicDisplay title="Attack Vector Analysis" glowColor="blue" className="h-[300px] w-full">
          <div className="h-full w-full overflow-hidden">
            <AttackVectorRadar 
              threats={activeThreats}
              anomalyScore={anomalyScore}
              className="h-full w-full"
            />
          </div>
        </HolographicDisplay>
      </div>
    </section>
  );
}
