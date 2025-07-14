import React, { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { Activity, Brain, Zap, Download, Settings } from "lucide-react";
import { NeuralActivityIndicator } from "@/components/neural-activity-indicator";
import { generateInitialNeurons } from "./NeuronDataGenerator";
import { TrafficSimulator } from "./TrafficSimulator";

interface NeuronFlowVisualizationProps {
  anomalyScore: number;
  isAnalysisActive: boolean;
  activeThreats: number;
}

interface NeuralSignal {
  id: string;
  fromX: number;
  fromY: number;
  toX: number;
  toY: number;
  progress: number;
  speed: number;
  intensity: number;
  color: string;
  type: 'normal' | 'suspicious' | 'anomaly';
}

interface SynapticConnection {
  id: string;
  fromNeuron: string;
  toNeuron: string;
  strength: number;
  active: boolean;
  lastSignal: number;
}

export function NeuronFlowVisualization({ 
  anomalyScore, 
  isAnalysisActive, 
  activeThreats 
}: NeuronFlowVisualizationProps) {
  const [neurons, setNeurons] = useState(generateInitialNeurons(30));
  const [layerVisibility, setLayerVisibility] = useState({
    input: true,
    hidden: true,
    output: true
  });
  const [showConnections, setShowConnections] = useState(true);
  const [hoverNeuron, setHoverNeuron] = useState<any>(null);
  const [trafficSimulator] = useState(() => new TrafficSimulator(neurons, anomalyScore, isAnalysisActive));
  const [neuralSignals, setNeuralSignals] = useState<NeuralSignal[]>([]);
  const [synapticConnections, setSynapticConnections] = useState<SynapticConnection[]>([]);

  // Initialize synaptic connections between layers
  useEffect(() => {
    const inputNeurons = neurons.filter(n => n.layer === 'input').slice(0, 6);
    const hiddenNeurons = neurons.filter(n => n.layer === 'hidden').slice(0, 8);
    const outputNeurons = neurons.filter(n => n.layer === 'output').slice(0, 4);

    const connections: SynapticConnection[] = [];

    // Input to Hidden connections
    inputNeurons.forEach(inputNeuron => {
      hiddenNeurons.forEach(hiddenNeuron => {
        if (Math.random() > 0.4) { // 60% connection rate
          connections.push({
            id: `${inputNeuron.id}-${hiddenNeuron.id}`,
            fromNeuron: inputNeuron.id,
            toNeuron: hiddenNeuron.id,
            strength: 0.3 + Math.random() * 0.7,
            active: false,
            lastSignal: 0
          });
        }
      });
    });

    // Hidden to Output connections
    hiddenNeurons.forEach(hiddenNeuron => {
      outputNeurons.forEach(outputNeuron => {
        if (Math.random() > 0.3) { // 70% connection rate
          connections.push({
            id: `${hiddenNeuron.id}-${outputNeuron.id}`,
            fromNeuron: hiddenNeuron.id,
            toNeuron: outputNeuron.id,
            strength: 0.4 + Math.random() * 0.6,
            active: false,
            lastSignal: 0
          });
        }
      });
    });

    setSynapticConnections(connections);
  }, [neurons]);

  // Generate neural signals based on brain-like firing patterns
  useEffect(() => {
    if (!isAnalysisActive) {
      setNeuralSignals([]);
      return;
    }

    const interval = setInterval(() => {
      const currentTime = Date.now();
      const newSignals: NeuralSignal[] = [];

      // Simulate realistic neural firing patterns
      const inputNeurons = neurons.filter(n => n.layer === 'input').slice(0, 6);
      const hiddenNeurons = neurons.filter(n => n.layer === 'hidden').slice(0, 8);
      const outputNeurons = neurons.filter(n => n.layer === 'output').slice(0, 4);

      // Input layer spontaneous activity (like sensory input)
      inputNeurons.forEach((neuron, index) => {
        const shouldFire = Math.random() < (0.3 + anomalyScore * 0.4);
        if (shouldFire) {
          // Find connections from this input neuron
          const connections = synapticConnections.filter(conn => conn.fromNeuron === neuron.id);
          
          connections.forEach(connection => {
            const targetNeuron = hiddenNeurons.find(n => n.id === connection.toNeuron);
            if (targetNeuron && Math.random() < connection.strength) {
              let signalType: 'normal' | 'suspicious' | 'anomaly' = 'normal';
              let color = '#10B981'; // Green for input
              let speed = 0.015;

              if (anomalyScore > 0.7 && Math.random() < 0.3) {
                signalType = 'anomaly';
                color = '#EF4444';
                speed = 0.025;
              } else if (anomalyScore > 0.4 && Math.random() < 0.2) {
                signalType = 'suspicious';
                color = '#F59E0B';
                speed = 0.02;
              }

              newSignals.push({
                id: `signal-${currentTime}-${Math.random()}`,
                fromX: 150,
                fromY: 120 + (index * 40),
                toX: 350,
                toY: 100 + (hiddenNeurons.findIndex(n => n.id === targetNeuron.id) * 30),
                progress: 0,
                speed,
                intensity: connection.strength,
                color,
                type: signalType
              });

              // Update connection activity
              connection.active = true;
              connection.lastSignal = currentTime;
            }
          });
        }
      });

      // Hidden layer processing (integrate and fire)
      hiddenNeurons.forEach((neuron, index) => {
        const recentInputs = synapticConnections.filter(conn => 
          conn.toNeuron === neuron.id && 
          conn.active && 
          currentTime - conn.lastSignal < 1000
        );

        const totalInput = recentInputs.reduce((sum, conn) => sum + conn.strength, 0);
        const threshold = 0.5 + (Math.random() * 0.3);

        if (totalInput > threshold) {
          // Fire to output layer
          const outputConnections = synapticConnections.filter(conn => conn.fromNeuron === neuron.id);
          
          outputConnections.forEach(connection => {
            const targetNeuron = outputNeurons.find(n => n.id === connection.toNeuron);
            if (targetNeuron && Math.random() < connection.strength) {
              let color = '#3B82F6'; // Blue for hidden
              let speed = 0.018;

              // Propagate signal type based on inputs
              const hasAnomalyInput = recentInputs.some(conn => 
                synapticConnections.find(c => c.id === conn.id && c.lastSignal > currentTime - 500)
              );

              if (hasAnomalyInput || (anomalyScore > 0.6 && Math.random() < 0.4)) {
                color = '#EF4444';
                speed = 0.028;
              }

              newSignals.push({
                id: `signal-${currentTime}-${Math.random()}`,
                fromX: 350,
                fromY: 100 + (index * 30),
                toX: 550,
                toY: 140 + (outputNeurons.findIndex(n => n.id === targetNeuron.id) * 60),
                progress: 0,
                speed,
                intensity: connection.strength,
                color,
                type: hasAnomalyInput ? 'anomaly' : 'normal'
              });
            }
          });
        }
      });

      // Clean up old signals and update existing ones
      setNeuralSignals(prev => [
        ...prev.filter(signal => signal.progress < 1).map(signal => ({
          ...signal,
          progress: Math.min(1, signal.progress + signal.speed)
        })),
        ...newSignals.slice(0, 15) // Limit concurrent signals
      ]);

      // Decay connection activity
      setSynapticConnections(prev => 
        prev.map(conn => ({
          ...conn,
          active: conn.active && (currentTime - conn.lastSignal < 2000)
        }))
      );

    }, 150); // Realistic neural timing

    return () => clearInterval(interval);
  }, [isAnalysisActive, anomalyScore, neurons, synapticConnections]);

  // Update neurons based on analysis activity
  useEffect(() => {
    if (!isAnalysisActive) return;

    const interval = setInterval(() => {
      trafficSimulator.updateState(neurons, anomalyScore, isAnalysisActive);
      const patterns = trafficSimulator.generateTrafficPatterns();
      const updatedNeurons = trafficSimulator.updateNeuronStates(patterns);
      setNeurons(updatedNeurons);
    }, 800);

    return () => clearInterval(interval);
  }, [isAnalysisActive, anomalyScore, trafficSimulator]);

  const handleExportData = () => {
    const data = neurons.map(neuron => ({
      id: neuron.id,
      layer: neuron.layer,
      spikeCount: neuron.spikeCount,
      lastFired: neuron.lastFired,
      trafficState: neuron.trafficState,
      intensity: neuron.intensity,
      connections: neuron.connections.length
    }));

    const csv = [
      Object.keys(data[0]).join(','),
      ...data.map(row => Object.values(row).join(','))
    ].join('\n');

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'neural-flow-data.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  // Organize neurons by layers for proper positioning
  const inputNeurons = neurons.filter(n => n.layer === 'input').slice(0, 6);
  const hiddenNeurons = neurons.filter(n => n.layer === 'hidden').slice(0, 8);
  const outputNeurons = neurons.filter(n => n.layer === 'output').slice(0, 4);

  const renderNeuron = (neuron: any, x: number, y: number) => {
    let neuronColor = '#3B82F6';
    let intensity = neuron.intensity || 0.3;
    
    // Check if neuron is receiving signals
    const recentSignals = neuralSignals.filter(signal => 
      Math.abs(signal.toX - x) < 10 && Math.abs(signal.toY - y) < 10 && signal.progress > 0.8
    );
    
    const isReceivingSignal = recentSignals.length > 0;
    const signalIntensity = Math.min(recentSignals.reduce((sum, signal) => sum + signal.intensity, 0), 1); // Cap at 1
    
    switch (neuron.trafficState) {
      case 'anomaly':
        neuronColor = '#EF4444';
        break;
      case 'suspicious':
        neuronColor = '#F59E0B';
        break;
      default:
        switch (neuron.layer) {
          case 'input': neuronColor = '#10B981'; break;
          case 'hidden': neuronColor = '#3B82F6'; break;
          case 'output': neuronColor = '#8B5CF6'; break;
        }
    }

    const isActive = neuron.lastFired && (Date.now() / 1000) - neuron.lastFired < 2;
    
    // Fixed radius calculation to prevent excessive growth
    let baseRadius = 8;
    if (neuron.layer === 'input') baseRadius = 6;
    if (neuron.layer === 'output') baseRadius = 10;
    
    // Limit the maximum radius growth
    const maxGrowth = 4; // Maximum additional pixels
    const activityBonus = (isActive || isReceivingSignal) ? Math.min(signalIntensity * maxGrowth, maxGrowth) : 0;
    const radius = baseRadius + activityBonus;

    return (
      <g key={neuron.id}>
        {/* Reduced glow effect for active neurons */}
        {(isActive || isReceivingSignal) && (
          <circle
            cx={x}
            cy={y}
            r={radius * 1.5} // Reduced from radius * 2
            fill={neuronColor}
            opacity={0.1 + signalIntensity * 0.1} // Reduced opacity
          />
        )}
        {/* Main neuron with controlled size */}
        <circle
          cx={x}
          cy={y}
          r={radius}
          fill={neuronColor}
          opacity={isActive || isReceivingSignal ? 0.9 : 0.7}
          stroke={isActive || isReceivingSignal ? '#fff' : neuronColor}
          strokeWidth={isActive || isReceivingSignal ? 2 : 1}
          onMouseEnter={() => setHoverNeuron(neuron)}
          onMouseLeave={() => setHoverNeuron(null)}
          style={{ cursor: 'pointer' }}
        />
        {/* Controlled dendritic branches */}
        {(isActive || isReceivingSignal) && (
          <g stroke={neuronColor} strokeWidth="1" opacity="0.4">
            {[0, 1, 2, 3].map(i => {
              const angle = (i / 4) * Math.PI * 2;
              const length = radius * 1.2; // Reduced from radius * 1.5
              return (
                <line
                  key={i}
                  x1={x}
                  y1={y}
                  x2={x + Math.cos(angle) * length}
                  y2={y + Math.sin(angle) * length}
                />
              );
            })}
          </g>
        )}
        {/* Controlled activity pulse animation */}
        {(isActive || isReceivingSignal) && (
          <circle
            cx={x}
            cy={y}
            r={radius}
            fill="none"
            stroke={neuronColor}
            strokeWidth="2"
            opacity="0.6"
          >
            <animate
              attributeName="r"
              values={`${radius};${radius * 1.3};${radius}`} // Reduced from radius * 1.8
              dur="1s"
              repeatCount="indefinite"
            />
            <animate
              attributeName="opacity"
              values="0.6;0;0.6"
              dur="1s"
              repeatCount="indefinite"
            />
          </circle>
        )}
      </g>
    );
  };

  return (
    <div className="w-full h-full flex flex-col bg-slate-900 rounded-lg overflow-hidden">
      {/* Header */}
      <div className="flex justify-between items-center p-4 border-b border-slate-700 bg-slate-800">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <Brain className="h-5 w-5 text-blue-400" />
            <span className="font-semibold text-white">Neural Network Architecture</span>
            <Badge variant={isAnalysisActive ? "default" : "secondary"}>
              {isAnalysisActive ? "Processing" : "Idle"}
            </Badge>
          </div>
          
          <div className="flex items-center gap-4 text-sm text-slate-300">
            <div className="flex items-center gap-2">
              <Activity className="h-4 w-4 text-orange-400" />
              <span>Anomaly: {(anomalyScore * 100).toFixed(1)}%</span>
            </div>
            
            <div className="flex items-center gap-2">
              <Zap className="h-4 w-4 text-red-400" />
              <span>Threats: {activeThreats}</span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <Label htmlFor="connections" className="text-slate-300 text-sm">Connections</Label>
            <Switch 
              id="connections"
              checked={showConnections}
              onCheckedChange={setShowConnections}
            />
          </div>
          
          <Button variant="outline" size="sm" onClick={handleExportData} className="text-slate-300 border-slate-600">
            <Download className="h-4 w-4 mr-1" />
            Export
          </Button>
        </div>
      </div>

      {/* Main visualization area */}
      <div className="flex-1 relative bg-gradient-to-br from-slate-900 to-slate-800 overflow-hidden">
        {/* Neural Activity Base Layer */}
        <NeuralActivityIndicator 
          isActive={isAnalysisActive}
          anomalyScore={anomalyScore}
        />
        
        {/* Main Neural Network SVG */}
        <svg className="w-full h-full absolute inset-0" viewBox="0 0 700 400" style={{ zIndex: 5 }}>
          {/* Render synaptic connections with realistic appearance */}
          {showConnections && synapticConnections.map(connection => {
            const fromNeuron = [...inputNeurons, ...hiddenNeurons].find(n => n.id === connection.fromNeuron);
            const toNeuron = [...hiddenNeurons, ...outputNeurons].find(n => n.id === connection.toNeuron);
            
            if (!fromNeuron || !toNeuron) return null;
            
            let fromX = 150, fromY = 120, toX = 350, toY = 100;
            
            if (fromNeuron.layer === 'input') {
              fromX = 150;
              fromY = 120 + (inputNeurons.findIndex(n => n.id === fromNeuron.id) * 40);
            } else if (fromNeuron.layer === 'hidden') {
              fromX = 350;
              fromY = 100 + (hiddenNeurons.findIndex(n => n.id === fromNeuron.id) * 30);
            }
            
            if (toNeuron.layer === 'hidden') {
              toX = 350;
              toY = 100 + (hiddenNeurons.findIndex(n => n.id === toNeuron.id) * 30);
            } else if (toNeuron.layer === 'output') {
              toX = 550;
              toY = 140 + (outputNeurons.findIndex(n => n.id === toNeuron.id) * 60);
            }
            
            const isActive = connection.active;
            const opacity = isActive ? connection.strength * 0.8 : connection.strength * 0.2;
            
            return (
              <line
                key={connection.id}
                x1={fromX}
                y1={fromY}
                x2={toX}
                y2={toY}
                stroke={isActive ? "#00ff88" : "#334155"}
                strokeWidth={isActive ? 2 : 1}
                opacity={opacity}
                strokeDasharray={isActive ? "none" : "2,2"}
              />
            );
          })}

          {/* Neural signals flowing between neurons */}
          {neuralSignals.map(signal => {
            const currentX = signal.fromX + (signal.toX - signal.fromX) * signal.progress;
            const currentY = signal.fromY + (signal.toY - signal.fromY) * signal.progress;
            
            return (
              <g key={signal.id}>
                {/* Signal trail */}
                <circle
                  cx={currentX}
                  cy={currentY}
                  r="3"
                  fill={signal.color}
                  opacity={signal.intensity * 0.4}
                />
                {/* Main signal */}
                <circle
                  cx={currentX}
                  cy={currentY}
                  r="2"
                  fill={signal.color}
                  opacity={signal.intensity * 0.9}
                >
                  <animate
                    attributeName="r"
                    values="2;4;2"
                    dur="0.5s"
                    repeatCount="indefinite"
                  />
                </circle>
                {/* Enhanced glow for anomaly signals */}
                {signal.type === 'anomaly' && (
                  <circle
                    cx={currentX}
                    cy={currentY}
                    r="6"
                    fill={signal.color}
                    opacity="0.2"
                  />
                )}
              </g>
            );
          })}

          {/* Render Input Layer */}
          {layerVisibility.input && (
            <g>
              <text x="150" y="80" textAnchor="middle" fill="#10B981" fontSize="16" fontWeight="bold">
                Input Layer
              </text>
              <text x="150" y="95" textAnchor="middle" fill="#64748B" fontSize="12">
                Sensory Input
              </text>
              {inputNeurons.map((neuron, index) => 
                renderNeuron(neuron, 150, 120 + (index * 40))
              )}
            </g>
          )}

          {/* Render Hidden Layer */}
          {layerVisibility.hidden && (
            <g>
              <text x="350" y="80" textAnchor="middle" fill="#3B82F6" fontSize="16" fontWeight="bold">
                Processing Layer
              </text>
              <text x="350" y="95" textAnchor="middle" fill="#64748B" fontSize="12">
                Integration & Analysis
              </text>
              {hiddenNeurons.map((neuron, index) => 
                renderNeuron(neuron, 350, 100 + (index * 30))
              )}
            </g>
          )}

          {/* Render Output Layer */}
          {layerVisibility.output && (
            <g>
              <text x="550" y="80" textAnchor="middle" fill="#8B5CF6" fontSize="16" fontWeight="bold">
                Output Layer
              </text>
              <text x="550" y="95" textAnchor="middle" fill="#64748B" fontSize="12">
                Decision & Response
              </text>
              {outputNeurons.map((neuron, index) => 
                renderNeuron(neuron, 550, 140 + (index * 60))
              )}
            </g>
          )}
        </svg>
        
        {/* Layer Controls */}
        <div className="absolute top-4 left-4 bg-black/40 backdrop-blur-sm rounded-lg p-3 border border-slate-700">
          <div className="text-sm font-medium text-slate-200 mb-3">Layer Visibility</div>
          {Object.entries(layerVisibility).map(([layer, visible]) => (
            <div key={layer} className="flex items-center gap-2 mb-2">
              <Switch
                id={layer}
                checked={visible}
                onCheckedChange={(checked) =>
                  setLayerVisibility(prev => ({ ...prev, [layer]: checked }))
                }
              />
              <Label htmlFor={layer} className="text-slate-300 capitalize text-sm">
                {layer}
              </Label>
            </div>
          ))}
        </div>

        {/* Legend */}
        <div className="absolute top-4 right-4 bg-black/40 backdrop-blur-sm rounded-lg p-3 min-w-[200px] border border-slate-700">
          <div className="text-sm font-medium text-slate-200 mb-3">Legend</div>
          <div className="space-y-2 text-xs text-slate-300">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-green-500"></div>
              <span>Input Neurons</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-blue-500"></div>
              <span>Hidden Neurons</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-purple-500"></div>
              <span>Output Neurons</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-red-500"></div>
              <span>Anomaly Detected</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
              <span>Suspicious Activity</span>
            </div>
          </div>
        </div>

        {/* Hover Info */}
        {hoverNeuron && (
          <div className="absolute bottom-4 left-4 bg-black/80 backdrop-blur-sm rounded-lg p-3 text-white max-w-xs border border-slate-600">
            <div className="font-medium text-slate-200">Neuron {hoverNeuron.id}</div>
            <div className="text-sm text-slate-400 mt-1">
              <div>Layer: <span className="text-slate-300">{hoverNeuron.layer}</span></div>
              <div>State: <span className="text-slate-300">{hoverNeuron.trafficState || 'normal'}</span></div>
              <div>Intensity: <span className="text-slate-300">{((hoverNeuron.intensity || 0) * 100).toFixed(1)}%</span></div>
              <div>Spikes: <span className="text-slate-300">{hoverNeuron.spikeCount || 0}</span></div>
              <div>Connections: <span className="text-slate-300">{hoverNeuron.connections?.length || 0}</span></div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default NeuronFlowVisualization;
