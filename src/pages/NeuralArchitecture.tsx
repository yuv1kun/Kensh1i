import { useState, useEffect, useRef } from "react";
import { Header } from "@/components/header";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Brain, Layers, ChartLine, Play, Pause, Eye, EyeOff, CircleIcon } from "lucide-react";
import { AspectRatio } from "@/components/ui/aspect-ratio";
import { toast } from "@/hooks/use-toast";

// Mock data for neural layers
const neuralLayers = [
  {
    id: "input",
    name: "Input Layer",
    neurons: 784,
    active: true,
    description: "Converts network packet features into spike-based representations for the SNN."
  },
  {
    id: "hidden1",
    name: "Hidden Layer 1",
    neurons: 256,
    active: true,
    description: "Performs feature extraction using leaky integrate-and-fire neurons with STDP learning."
  },
  {
    id: "hidden2",
    name: "Hidden Layer 2",
    neurons: 128,
    active: true,
    description: "Hierarchical feature processing through inhibitory and excitatory connections."
  },
  {
    id: "output",
    name: "Output Layer",
    neurons: 10,
    active: true,
    description: "Classification layer using winner-take-all competition to identify threat patterns."
  }
];

// Mock data for neuromorphic glossary
const glossaryTerms = [
  {
    term: "STDP",
    fullForm: "Spike-Timing-Dependent Plasticity",
    description: "Biological learning mechanism where synaptic strength is adjusted based on the relative timing of pre- and post-synaptic action potentials."
  },
  {
    term: "LTP/LTD",
    fullForm: "Long-Term Potentiation/Depression",
    description: "Persistent strengthening or weakening of synapses based on patterns of activity, forming cellular basis for learning and memory."
  },
  {
    term: "Izhikevich Model",
    fullForm: "Izhikevich Neuron Model",
    description: "Computationally efficient spiking neuron model capable of reproducing diverse firing patterns observed in cortical neurons."
  },
  {
    term: "SNN",
    fullForm: "Spiking Neural Network",
    description: "Neural network model that mimics natural neural networks by using discrete spikes for information transmission rather than continuous values."
  },
  {
    term: "Neuromorphic",
    fullForm: "Neuromorphic Computing",
    description: "Computing paradigm inspired by the structure and function of biological neural systems, often using specialized hardware."
  },
  {
    term: "Membrane Potential",
    fullForm: "Neuronal Membrane Potential",
    description: "The difference in electrical potential between the interior and exterior of a neuron, key to neural signal processing."
  }
];

// Sample weight matrix visualization data
const generateHeatmapData = (size: number) => {
  const data = [];
  for (let i = 0; i < size; i++) {
    const row = [];
    for (let j = 0; j < size; j++) {
      // Generate values between -1 and 1
      row.push(Math.tanh(Math.random() * 5 - 2.5));
    }
    data.push(row);
  }
  return data;
};

const NeuralArchitecture = () => {
  const [timeStep, setTimeStep] = useState(50);
  const [activeLayers, setActiveLayers] = useState<string[]>(neuralLayers.map(layer => layer.id));
  const [selectedNeuron, setSelectedNeuron] = useState<string | null>(null);
  const [showLabels, setShowLabels] = useState(true);
  const [isSimulationRunning, setIsSimulationRunning] = useState(false);
  const simulationInterval = useRef<number | null>(null);

  // Toggle layer visibility
  const toggleLayer = (layerId: string) => {
    if (activeLayers.includes(layerId)) {
      setActiveLayers(activeLayers.filter(id => id !== layerId));
    } else {
      setActiveLayers([...activeLayers, layerId]);
    }
  };

  // Generate sample spike timing data
  const generateSpikeTiming = () => {
    const data = [];
    const neurons = 30;
    const timeSteps = 100;
    
    for (let i = 0; i < neurons; i++) {
      const spikes = [];
      for (let t = 0; t < timeSteps; t++) {
        // More spikes around the current time step
        const probability = Math.exp(-Math.abs(t - timeStep) / 20) * 0.5;
        spikes.push(Math.random() < probability ? 1 : 0);
      }
      data.push(spikes);
    }
    
    return data;
  };

  // Handle run/pause simulation
  const toggleSimulation = () => {
    if (isSimulationRunning) {
      // Pause simulation
      if (simulationInterval.current !== null) {
        window.clearInterval(simulationInterval.current);
        simulationInterval.current = null;
      }
      setIsSimulationRunning(false);
      toast({
        title: "Simulation Paused",
        description: "The neural network simulation has been paused",
      });
    } else {
      // Run simulation
      simulationInterval.current = window.setInterval(() => {
        setTimeStep(prevStep => (prevStep + 1) % 100);
      }, 200);
      setIsSimulationRunning(true);
      toast({
        title: "Simulation Running",
        description: "The neural network simulation is now running",
      });
    }
  };

  // Clean up the interval on unmount
  useEffect(() => {
    return () => {
      if (simulationInterval.current !== null) {
        window.clearInterval(simulationInterval.current);
      }
    };
  }, []);

  const spikeTimingData = generateSpikeTiming();
  
  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      
      <main className="flex-1 container py-8 px-4 sm:px-8">
        <div className="flex flex-col gap-6">
          <div className="flex items-center justify-between">
            <h1 className="text-3xl font-bold bg-gradient-to-r from-kenshi-blue to-kenshi-red bg-clip-text text-transparent">
              Neural Architecture
            </h1>
            <Badge variant="outline" className="px-4 py-1 flex items-center gap-2">
              <div className="h-2 w-2 rounded-full bg-kenshi-red animate-pulse"></div>
              <span>SNN Processing Active</span>
            </Badge>
          </div>
          
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 neumorph rounded-lg overflow-hidden">
              <div className="p-4 border-b bg-background/50 backdrop-blur flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <Brain className="h-5 w-5 text-kenshi-blue" />
                  <h2 className="font-semibold">SNN Explorer</h2>
                </div>
                <div className="flex items-center gap-3">
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Button 
                          variant="ghost" 
                          size="sm"
                          onClick={() => setShowLabels(!showLabels)}
                          className="h-8 w-8 p-0 flex items-center justify-center"
                        >
                          {showLabels ? <Eye className="h-4 w-4" /> : <EyeOff className="h-4 w-4" />}
                        </Button>
                      </TooltipTrigger>
                      <TooltipContent>
                        <p>Toggle labels</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                  
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Button 
                          variant="ghost" 
                          size="sm" 
                          className="h-8 w-8 p-0 flex items-center justify-center"
                          onClick={toggleSimulation}
                          aria-label={isSimulationRunning ? "Pause simulation" : "Run simulation"}
                        >
                          {isSimulationRunning ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                        </Button>
                      </TooltipTrigger>
                      <TooltipContent>
                        <p>{isSimulationRunning ? "Pause simulation" : "Run simulation"}</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
              </div>
              
              <AspectRatio ratio={16/9}>
                <div className="w-full h-full bg-black/90 p-6 relative">
                  {/* Neural network visualization */}
                  <div className="absolute inset-0 flex items-center justify-center">
                    <svg width="100%" height="100%" viewBox="0 0 800 450">
                      <defs>
                        <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="0" refY="3.5" orient="auto">
                          <polygon points="0 0, 10 3.5, 0 7" fill="#4B5563" />
                        </marker>
                      </defs>
                      
                      <g className="network">
                        {/* Layers */}
                        {neuralLayers.map((layer, layerIndex) => {
                          if (!activeLayers.includes(layer.id)) return null;
                          
                          const x = 100 + (layerIndex * 200);
                          const totalNeurons = Math.min(layer.neurons, 12); // Show max 12 neurons visually
                          
                          return (
                            <g key={layer.id} className="layer">
                              {/* Layer label */}
                              {showLabels && (
                                <text 
                                  x={x} 
                                  y={30} 
                                  textAnchor="middle" 
                                  fill="#9CA3AF" 
                                  fontSize="14"
                                >
                                  {layer.name}
                                </text>
                              )}
                              
                              {/* Neurons */}
                              {[...Array(totalNeurons)].map((_, neuronIndex) => {
                                const neuronId = `${layer.id}-${neuronIndex}`;
                                const isSelected = selectedNeuron === neuronId;
                                const y = 80 + ((450 - 80) / (totalNeurons + 1)) * (neuronIndex + 1);
                                
                                // Determine if neuron is firing in current time step
                                const isFiring = layer.id === "input" && 
                                  spikeTimingData[neuronIndex % spikeTimingData.length][timeStep % 100] === 1;
                                
                                return (
                                  <g key={neuronId}>
                                    {/* Draw connections to next layer */}
                                    {layerIndex < neuralLayers.length - 1 && 
                                      activeLayers.includes(neuralLayers[layerIndex + 1].id) && 
                                      [...Array(Math.min(5, totalNeurons))].map((_, nextIdx) => {
                                        const nextY = 80 + ((450 - 80) / (totalNeurons + 1)) * (nextIdx + 1);
                                        
                                        // Only draw some connections for visual clarity
                                        if ((neuronIndex + nextIdx) % 3 !== 0) return null;
                                        
                                        return (
                                          <line 
                                            key={`${neuronId}-to-${nextIdx}`}
                                            x1={x + 20} 
                                            y1={y} 
                                            x2={x + 180} 
                                            y2={nextY}
                                            stroke={isFiring ? "#EA4335" : "#4B5563"}
                                            strokeWidth={isFiring ? 2 : 0.5}
                                            strokeOpacity={0.3}
                                            markerEnd="url(#arrowhead)"
                                          />
                                        );
                                      })
                                    }
                                    
                                    {/* Neuron circle */}
                                    <circle 
                                      cx={x} 
                                      cy={y} 
                                      r={isSelected ? 12 : 8}
                                      fill={isFiring ? "#EA4335" : "#1A73E8"}
                                      stroke={isSelected ? "#ffffff" : "none"}
                                      strokeWidth={2}
                                      opacity={isFiring ? 1 : 0.7}
                                      onClick={() => setSelectedNeuron(isSelected ? null : neuronId)}
                                      style={{ cursor: "pointer" }}
                                    />
                                    
                                    {/* Animation for firing neurons */}
                                    {isFiring && (
                                      <circle 
                                        cx={x} 
                                        cy={y} 
                                        r={20}
                                        fill="none"
                                        stroke="#EA4335"
                                        strokeWidth={2}
                                        opacity={0.5}
                                        className="animate-ping"
                                      />
                                    )}
                                  </g>
                                );
                              })}
                              
                              {/* Show indicator if more neurons exist than visualized */}
                              {layer.neurons > 12 && (
                                <text 
                                  x={x} 
                                  y={420} 
                                  textAnchor="middle" 
                                  fill="#9CA3AF" 
                                  fontSize="12"
                                >
                                  +{layer.neurons - 12} more neurons
                                </text>
                              )}
                            </g>
                          );
                        })}
                      </g>
                    </svg>
                  </div>
                  
                  {/* Time step control */}
                  <div className="absolute bottom-6 left-6 right-6 bg-background/20 backdrop-blur-sm p-4 rounded-md">
                    <div className="flex items-center gap-4">
                      <span className="text-white text-sm">Time Step: {timeStep}</span>
                      <div className="flex-1">
                        <Slider
                          value={[timeStep]}
                          min={0}
                          max={100}
                          step={1}
                          onValueChange={(value) => setTimeStep(value[0])}
                          className="w-full"
                        />
                      </div>
                      <Button 
                        variant="outline" 
                        size="sm" 
                        onClick={() => {
                          setTimeStep(0);
                          if (isSimulationRunning) {
                            toggleSimulation();
                          }
                        }}
                      >
                        Reset
                      </Button>
                    </div>
                  </div>
                </div>
              </AspectRatio>
            </div>
            
            <div className="space-y-6">
              {/* Layer controls */}
              <Card className="neumorph">
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg flex items-center gap-2">
                    <Layers className="h-5 w-5 text-kenshi-blue" />
                    Layer Toggles
                  </CardTitle>
                  <CardDescription>Toggle layers to visualize specific parts</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {neuralLayers.map(layer => (
                      <div key={layer.id} className="flex items-center justify-between">
                        <div>
                          <Label htmlFor={`layer-${layer.id}`} className="font-medium">
                            {layer.name}
                          </Label>
                          <p className="text-xs text-muted-foreground">{layer.neurons} neurons</p>
                        </div>
                        <Switch
                          id={`layer-${layer.id}`}
                          checked={activeLayers.includes(layer.id)}
                          onCheckedChange={() => toggleLayer(layer.id)}
                        />
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
              
              {/* Spike visualization */}
              <Card className="neumorph">
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg flex items-center gap-2">
                    <ChartLine className="h-5 w-5 text-kenshi-blue" />
                    Spike Activity
                  </CardTitle>
                  <CardDescription>Real-time spike encoding</CardDescription>
                </CardHeader>
                <CardContent className="p-0 overflow-hidden">
                  <div className="relative h-[120px] w-full bg-black">
                    <svg width="100%" height="100%" style={{ overflow: "visible" }}>
                      {/* Y-axis labels */}
                      <text x="5" y="15" fontSize="10" fill="#9CA3AF">Neurons</text>
                      
                      {/* X-axis labels */}
                      <text x="5" y="110" fontSize="10" fill="#9CA3AF">Time</text>
                      <line x1="0" y1="100" x2="100%" y2="100" stroke="#4B5563" strokeWidth="1" />
                      
                      {/* Spike visualization */}
                      {spikeTimingData.slice(0, 10).map((neuronSpikes, neuronIdx) => 
                        neuronSpikes.map((spike, timeIdx) => 
                          spike === 1 && (
                            <line 
                              key={`spike-${neuronIdx}-${timeIdx}`}
                              x1={timeIdx * 3}
                              y1={25 + neuronIdx * 7}
                              x2={timeIdx * 3}
                              y2={25 + neuronIdx * 7 + 5}
                              stroke="#EA4335"
                              strokeWidth="2"
                            />
                          )
                        )
                      )}
                      
                      {/* Current time indicator */}
                      <line 
                        x1={timeStep * 3} 
                        y1="0" 
                        x2={timeStep * 3} 
                        y2="100" 
                        stroke="#FFFFFF" 
                        strokeWidth="1" 
                        strokeDasharray="4,2"
                      />
                    </svg>
                  </div>
                </CardContent>
              </Card>
              
              {/* Synaptic weight heatmap */}
              <Card className="neumorph">
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg">Synaptic Weights</CardTitle>
                  <CardDescription>Weight matrix visualization</CardDescription>
                </CardHeader>
                <CardContent className="p-0">
                  <div className="p-4 flex justify-center">
                    <div className="grid grid-cols-10 gap-[1px]">
                      {generateHeatmapData(10).map((row, i) => 
                        row.map((value, j) => (
                          <div
                            key={`cell-${i}-${j}`}
                            className="w-4 h-4"
                            style={{
                              backgroundColor: value > 0 
                                ? `rgba(26, 115, 232, ${Math.abs(value)})`
                                : `rgba(234, 67, 53, ${Math.abs(value)})`,
                            }}
                            title={`Weight: ${value.toFixed(2)}`}
                          />
                        ))
                      )}
                    </div>
                  </div>
                  <div className="px-4 pb-4 flex justify-center items-center gap-6">
                    <div className="flex items-center">
                      <div className="w-3 h-3 bg-kenshi-red mr-1"></div>
                      <span className="text-xs">Inhibitory</span>
                    </div>
                    <div className="flex items-center">
                      <div className="w-3 h-3 bg-kenshi-blue mr-1"></div>
                      <span className="text-xs">Excitatory</span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
          
          {/* Neuromorphic Glossary */}
          <div className="neumorph rounded-lg p-6">
            <div className="flex items-center gap-3 mb-6">
              <CircleIcon className="h-5 w-5 text-kenshi-blue" />
              <h2 className="text-xl font-semibold">Neuromorphic Glossary</h2>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {glossaryTerms.map(term => (
                <Card key={term.term}>
                  <CardHeader className="py-3">
                    <CardTitle className="text-base">{term.term}</CardTitle>
                    <CardDescription>{term.fullForm}</CardDescription>
                  </CardHeader>
                  <CardContent className="pt-0">
                    <p className="text-sm">{term.description}</p>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default NeuralArchitecture;
