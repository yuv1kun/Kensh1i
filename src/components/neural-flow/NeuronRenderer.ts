
import { NeuronData } from './types';

export const renderNeuron = (
  ctx: CanvasRenderingContext2D,
  neuron: NeuronData,
  x: number,
  y: number,
  layerVisibility: Record<string, boolean>
) => {
  if (!layerVisibility[neuron.layer]) return;
  
  // Enhanced color and size based on traffic state
  let color, glowColor;
  let radius = 5;
  
  switch (neuron.trafficState) {
    case 'anomaly':
      color = '#EF4444';
      glowColor = 'rgba(239, 68, 68, 0.6)';
      radius = 8 + Math.sin(Date.now() * 0.02) * 2; // Pulsing effect
      break;
    case 'suspicious':
      color = '#F59E0B';
      glowColor = 'rgba(245, 158, 11, 0.4)';
      radius = 6 + (neuron.intensity * 2);
      break;
    default:
      switch (neuron.layer) {
        case 'input': color = '#33C3F0'; break;
        case 'hidden': color = '#E5DEFF'; break;
        case 'output': color = '#D3E4FD'; break;
        default: color = '#F1F0FB';
      }
      glowColor = 'rgba(51, 195, 240, 0.3)';
      radius = neuron.lastFired ? 7 : 5;
  }
  
  // Enhanced glow effect for active neurons
  if (neuron.lastFired && (Date.now() / 1000) - neuron.lastFired < 3) {
    const glowRadius = radius * (2 + neuron.intensity);
    const gradient = ctx.createRadialGradient(x, y, radius * 0.5, x, y, glowRadius);
    gradient.addColorStop(0, glowColor);
    gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');
    
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(x, y, glowRadius, 0, Math.PI * 2);
    ctx.fill();
  }
  
  // Main neuron body
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.fill();
  
  // Spike visualization for anomalies
  if (neuron.trafficState === 'anomaly' && neuron.lastFired) {
    const spikeLength = radius * 1.5;
    const spikeCount = 6;
    
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    
    for (let i = 0; i < spikeCount; i++) {
      const angle = (i / spikeCount) * Math.PI * 2;
      const startX = x + Math.cos(angle) * radius;
      const startY = y + Math.sin(angle) * radius;
      const endX = x + Math.cos(angle) * (radius + spikeLength);
      const endY = y + Math.sin(angle) * (radius + spikeLength);
      
      ctx.beginPath();
      ctx.moveTo(startX, startY);
      ctx.lineTo(endX, endY);
      ctx.stroke();
    }
  }
  
  // Highlight for normal active neurons
  if (neuron.trafficState === 'normal' && neuron.lastFired) {
    ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
    ctx.beginPath();
    ctx.arc(x - radius * 0.3, y - radius * 0.3, radius * 0.25, 0, Math.PI * 2);
    ctx.fill();
  }
  
  // Store node position and data for hover detection
  neuron.graphPosition = { x, y, radius };
};
