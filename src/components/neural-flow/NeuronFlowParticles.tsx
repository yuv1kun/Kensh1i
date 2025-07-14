
import React, { useEffect, useState } from 'react';

interface FlowParticle {
  id: string;
  fromX: number;
  fromY: number;
  toX: number;
  toY: number;
  progress: number;
  speed: number;
  intensity: number;
  color: string;
}

interface NeuralFlowParticlesProps {
  neurons: any[];
  layerVisibility: Record<string, boolean>;
  isActive: boolean;
  anomalyScore: number;
}

export function NeuralFlowParticles({ 
  neurons, 
  layerVisibility, 
  isActive, 
  anomalyScore 
}: NeuralFlowParticlesProps) {
  const [flowParticles, setFlowParticles] = useState<FlowParticle[]>([]);

  // Generate flow particles based on neural activity
  useEffect(() => {
    if (!isActive) {
      setFlowParticles([]);
      return;
    }

    const interval = setInterval(() => {
      const newParticles: FlowParticle[] = [];
      
      neurons.forEach((neuron, index) => {
        if (!layerVisibility[neuron.layer] || !neuron.lastFired) return;
        
        const timeSinceLastFire = (Date.now() / 1000) - neuron.lastFired;
        if (timeSinceLastFire > 2) return;

        // Calculate neuron position
        const baseWidth = 100;
        const layerPositions = {
          'input': baseWidth * 0.2,
          'hidden': baseWidth * 0.5,
          'output': baseWidth * 0.8
        };
        
        const fromX = layerPositions[neuron.layer] || baseWidth * 0.5;
        const fromY = 10 + (80 * (index % 20)) / 20;

        // Create particles for each connection
        neuron.connections.slice(0, 2).forEach((connectionId: string) => {
          const targetNeuron = neurons.find(n => n.id === connectionId);
          if (!targetNeuron || !layerVisibility[targetNeuron.layer]) return;

          const targetIndex = neurons.findIndex(n => n.id === connectionId);
          const toX = layerPositions[targetNeuron.layer] || baseWidth * 0.5;
          const toY = 10 + (80 * (targetIndex % 20)) / 20;

          let color = '#3B82F6';
          let speed = 0.02;
          
          // Adjust based on traffic state
          switch (neuron.trafficState) {
            case 'anomaly':
              color = '#EF4444';
              speed = 0.05;
              break;
            case 'suspicious':
              color = '#F59E0B';
              speed = 0.035;
              break;
            default:
              color = '#3B82F6';
              speed = 0.02;
          }

          newParticles.push({
            id: `${neuron.id}-${connectionId}-${Date.now()}-${Math.random()}`,
            fromX,
            fromY,
            toX,
            toY,
            progress: 0,
            speed: speed + (Math.random() * 0.01),
            intensity: neuron.intensity || 0.5,
            color
          });
        });
      });

      setFlowParticles(prev => [
        ...prev.filter(p => p.progress < 1),
        ...newParticles.slice(0, 20) // Limit particles for performance
      ]);
    }, 300);

    return () => clearInterval(interval);
  }, [neurons, layerVisibility, isActive, anomalyScore]);

  // Update particle positions
  useEffect(() => {
    if (flowParticles.length === 0) return;

    const animationInterval = setInterval(() => {
      setFlowParticles(prev =>
        prev.map(particle => ({
          ...particle,
          progress: Math.min(1, particle.progress + particle.speed)
        })).filter(particle => particle.progress < 1)
      );
    }, 16); // ~60fps

    return () => clearInterval(animationInterval);
  }, [flowParticles.length]);

  return (
    <div className="absolute inset-0 pointer-events-none">
      <svg className="w-full h-full" style={{ zIndex: 10 }}>
        {flowParticles.map(particle => {
          // Calculate current position along the path
          const currentX = particle.fromX + (particle.toX - particle.fromX) * particle.progress;
          const currentY = particle.fromY + (particle.toY - particle.fromY) * particle.progress;
          
          // Add slight curve to the path
          const midX = (particle.fromX + particle.toX) / 2;
          const curveOffset = Math.sin(particle.progress * Math.PI) * 5;
          const adjustedY = currentY + curveOffset;

          return (
            <g key={particle.id}>
              {/* Particle trail effect */}
              <circle
                cx={`${currentX}%`}
                cy={`${adjustedY}%`}
                r="2"
                fill={particle.color}
                opacity={particle.intensity * 0.3}
              />
              {/* Main particle */}
              <circle
                cx={`${currentX}%`}
                cy={`${adjustedY}%`}
                r="1.5"
                fill={particle.color}
                opacity={particle.intensity * 0.8}
              >
                <animate
                  attributeName="r"
                  values="1.5;2.5;1.5"
                  dur="0.5s"
                  repeatCount="indefinite"
                />
              </circle>
              {/* Glow effect for high intensity particles */}
              {particle.intensity > 0.7 && (
                <circle
                  cx={`${currentX}%`}
                  cy={`${adjustedY}%`}
                  r="4"
                  fill={particle.color}
                  opacity={particle.intensity * 0.1}
                />
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}
