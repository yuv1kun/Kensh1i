
import React, { useEffect, useRef } from 'react';
import { NeuronData, LayerVisibility } from './types';
import { renderConnections } from './ConnectionRenderer';
import { renderNeuron } from './NeuronRenderer';
import { createCanvasInteractions } from './CanvasInteractions';

interface GraphViewProps {
  neurons: NeuronData[];
  layerVisibility: Record<string, boolean>;
  setHoverNeuron: (neuron: NeuronData | null) => void;
  zoomLevel: number;
}

const GraphView: React.FC<GraphViewProps> = ({ 
  neurons, 
  layerVisibility, 
  setHoverNeuron,
  zoomLevel
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const scaleRef = useRef<number>(1);
  const offsetRef = useRef<{ x: number, y: number }>({ x: 0, y: 0 });

  // Update scale based on zoom level
  useEffect(() => {
    scaleRef.current = 0.7 + (zoomLevel / 10);
  }, [zoomLevel]);
  
  // Draw network graph with enhanced traffic visualization
  useEffect(() => {
    if (!canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Set canvas dimensions
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    canvas.width = width;
    canvas.height = height;
    
    // Clear canvas
    ctx.fillStyle = 'rgba(0, 0, 0, 0)';
    ctx.fillRect(0, 0, width, height);
    
    // Apply transformations for pan and zoom
    ctx.save();
    ctx.translate(offsetRef.current.x, offsetRef.current.y);
    ctx.scale(scaleRef.current, scaleRef.current);
    
    // Define layer positions
    const baseWidth = width / scaleRef.current;
    const baseHeight = height / scaleRef.current;
    
    const layerPositions = {
      'input': baseWidth * 0.2,
      'hidden': baseWidth * 0.5,
      'output': baseWidth * 0.8
    };
    
    // Draw enhanced connections with traffic flow visualization
    renderConnections(ctx, neurons, layerVisibility, layerPositions, baseHeight);
    
    // Draw enhanced neurons with traffic state visualization
    neurons.forEach(neuron => {
      if (!layerVisibility[neuron.layer]) return;
      
      // Calculate node position
      const x = layerPositions[neuron.layer];
      const y = (baseHeight * 0.1) + (baseHeight * 0.8) * (neurons.findIndex(n => n.id === neuron.id) % 20) / 20;
      
      renderNeuron(ctx, neuron, x, y, layerVisibility);
    });
    
    // Restore context
    ctx.restore();
    
  }, [neurons, layerVisibility, zoomLevel]);

  // Setup canvas interactions
  useEffect(() => {
    if (!canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const cleanup = createCanvasInteractions(
      canvas,
      neurons,
      layerVisibility,
      setHoverNeuron,
      scaleRef,
      offsetRef
    );
    
    return cleanup;
  }, [neurons, layerVisibility, setHoverNeuron]);

  return (
    <canvas
      ref={canvasRef}
      className="w-full h-full cursor-grab active:cursor-grabbing"
    />
  );
};

export default GraphView;
