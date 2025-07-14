
import { NeuronData } from './types';

export const renderConnections = (
  ctx: CanvasRenderingContext2D,
  neurons: NeuronData[],
  layerVisibility: Record<string, boolean>,
  layerPositions: Record<string, number>,
  baseHeight: number
) => {
  neurons.forEach(fromNeuron => {
    if (!layerVisibility[fromNeuron.layer]) return;
    
    // Calculate node position
    const fromX = layerPositions[fromNeuron.layer];
    const fromY = (baseHeight * 0.1) + (baseHeight * 0.8) * (neurons.findIndex(n => n.id === fromNeuron.id) % 20) / 20;
    
    fromNeuron.connections.forEach(toId => {
      const toNeuron = neurons.find(n => n.id === toId);
      if (!toNeuron || !layerVisibility[toNeuron.layer]) return;
      
      const toX = layerPositions[toNeuron.layer];
      const toY = (baseHeight * 0.1) + (baseHeight * 0.8) * (neurons.findIndex(n => n.id === toNeuron.id) % 20) / 20;
      
      // Connection strength affects line width and opacity
      const strength = fromNeuron.connectionStrengths[toId] || 0.3;
      const lineWidth = 0.5 + (strength * 2);
      const opacity = 0.2 + (strength * 0.6);
      
      // Base connection line
      ctx.strokeStyle = `rgba(211, 228, 253, ${opacity})`;
      ctx.lineWidth = lineWidth;
      ctx.beginPath();
      ctx.moveTo(fromX, fromY);
      
      const controlX = (fromX + toX) / 2;
      ctx.bezierCurveTo(controlX, fromY, controlX, toY, toX, toY);
      ctx.stroke();
      
      // Enhanced traffic flow visualization
      if (fromNeuron.lastFired && (Date.now() / 1000) - fromNeuron.lastFired < 2) {
        let flowColor;
        let flowIntensity = fromNeuron.intensity;
        
        // Color based on traffic state
        switch (fromNeuron.trafficState) {
          case 'anomaly':
            flowColor = `rgba(239, 68, 68, ${flowIntensity})`;
            break;
          case 'suspicious':
            flowColor = `rgba(245, 158, 11, ${flowIntensity})`;
            break;
          default:
            flowColor = `rgba(59, 130, 246, ${flowIntensity * 0.7})`;
        }
        
        // Animated pulse effect for anomalies
        if (fromNeuron.trafficState === 'anomaly') {
          ctx.strokeStyle = flowColor;
          ctx.lineWidth = lineWidth * 2;
          ctx.globalAlpha = 0.8;
          
          ctx.beginPath();
          ctx.moveTo(fromX, fromY);
          ctx.bezierCurveTo(controlX, fromY, controlX, toY, toX, toY);
          ctx.stroke();
          
          ctx.globalAlpha = 1;
        } else {
          // Regular traffic flow
          ctx.strokeStyle = flowColor;
          ctx.lineWidth = lineWidth * 1.5;
          ctx.beginPath();
          ctx.moveTo(fromX, fromY);
          ctx.bezierCurveTo(controlX, fromY, controlX, toY, toX, toY);
          ctx.stroke();
        }
      }
    });
  });
};
