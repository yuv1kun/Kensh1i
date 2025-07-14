import React from 'react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { TooltipProvider, Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { Network } from "lucide-react";
import { Badge } from '@/components/ui/badge';
import { LayerVisibility } from './types';

interface NeuralControlsProps {
  simulationSpeed: number;
  setSimulationSpeed: (speed: number) => void;
  isPaused: boolean;
  setPaused: (paused: boolean) => void;
  viewMode: string;
  setViewMode: (mode: string) => void;
  layerVisibility: LayerVisibility;
  setLayerVisibility: (visibility: LayerVisibility) => void;
  neuronsCount: number;
  zoomLevel: number;
  setZoomLevel: (level: number) => void;
}

const NeuralControls: React.FC<NeuralControlsProps> = ({
  simulationSpeed,
  setSimulationSpeed,
  isPaused,
  setPaused,
  viewMode,
  setViewMode,
  layerVisibility,
  setLayerVisibility,
  neuronsCount,
  zoomLevel,
  setZoomLevel
}) => {
  return (
    <div className="space-y-3 p-2">
      <div className="text-sm font-medium">Neural Flow Controls</div>
      
      <div className="flex flex-wrap gap-2">
        <Button 
          size="sm" 
          variant={isPaused ? "outline" : "default"}
          onClick={() => setPaused(!isPaused)}
          className="bg-background/50 hover:bg-background/70"
        >
          {isPaused ? 'Resume' : 'Pause'} Flow
        </Button>
        
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button size="sm" variant="outline" className="ml-auto bg-background/50">
                <Network className="h-4 w-4 mr-1" />
                Node Count: {neuronsCount}
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              <p>Total neurons in the network</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>
      
      <div>
        <div className="mb-1 text-xs text-muted-foreground">Flow Speed</div>
        <Slider 
          value={[simulationSpeed]} 
          min={1} 
          max={10} 
          step={1} 
          onValueChange={(value) => setSimulationSpeed(value[0])}
          className="py-1"
        />
      </div>

      <div>
        <div className="mb-1 text-xs text-muted-foreground">Zoom Level</div>
        <Slider 
          value={[zoomLevel]} 
          min={1} 
          max={20} 
          step={1} 
          onValueChange={(value) => setZoomLevel(value[0])}
          className="py-1"
        />
      </div>
      
      <div>
        <div className="mb-1 text-xs text-muted-foreground">Layer Visibility</div>
        <div className="flex flex-wrap gap-1">
          <Badge 
            variant={layerVisibility.input ? "default" : "outline"}
            className="cursor-pointer bg-[#33C3F0]/50 hover:bg-[#33C3F0]/70"
            onClick={() => setLayerVisibility({ ...layerVisibility, input: !layerVisibility.input })}
          >
            Input Layer
          </Badge>
          <Badge 
            variant={layerVisibility.hidden ? "default" : "outline"}
            className="cursor-pointer bg-[#E5DEFF]/50 hover:bg-[#E5DEFF]/70 text-foreground"
            onClick={() => setLayerVisibility({ ...layerVisibility, hidden: !layerVisibility.hidden })}
          >
            Hidden Layers
          </Badge>
          <Badge 
            variant={layerVisibility.output ? "default" : "outline"}
            className="cursor-pointer bg-[#D3E4FD]/50 hover:bg-[#D3E4FD]/70 text-foreground"
            onClick={() => setLayerVisibility({ ...layerVisibility, output: !layerVisibility.output })}
          >
            Output Layer
          </Badge>
        </div>
      </div>
      
      <div>
        <div className="mb-1 text-xs text-muted-foreground">View Mode</div>
        <Tabs value={viewMode} onValueChange={setViewMode}>
          <TabsList className="grid grid-cols-2 bg-background/50">
            <TabsTrigger value="3d">3D View</TabsTrigger>
            <TabsTrigger value="graph">Graph View</TabsTrigger>
          </TabsList>
        </Tabs>
      </div>
      
      <div className="text-xs text-muted-foreground mt-2">
        <p>Hover over neurons to see details</p>
        <p>Drag to rotate • Scroll to zoom</p>
      </div>
    </div>
  );
};

export default NeuralControls;
