
import React, { useEffect, useState } from 'react';

interface NeuralActivityIndicatorProps {
  isActive: boolean;
  anomalyScore: number;
}

interface Neuron {
  x: number;
  y: number;
  size: number;
  active: boolean;
  intensity: number;
  type: 'normal' | 'suspicious' | 'anomaly';
  lastFired: number;
  connections: number[];
}

interface SynapticConnection {
  from: number;
  to: number;
  strength: number;
  active: boolean;
}

export function NeuralActivityIndicator({ isActive, anomalyScore }: NeuralActivityIndicatorProps) {
  const [neurons, setNeurons] = useState<Neuron[]>([]);
  const [plasticityParticles, setPlasticityParticles] = useState<Array<{ x: number; y: number; vx: number; vy: number; life: number }>>([]);
  
  // Generate simplified neural network
  useEffect(() => {
    const generatedNeurons = Array.from({ length: 15 }, (_, i) => ({ // Reduced from 25
      x: Math.random() * 100,
      y: Math.random() * 100,
      size: Math.random() * 2 + 1, // Reduced size
      active: false,
      intensity: Math.random(),
      type: 'normal' as const,
      lastFired: 0,
      connections: []
    }));
    
    setNeurons(generatedNeurons);
  }, []);
  
  // Optimized real-time activation with reduced frequency
  useEffect(() => {
    const interval = setInterval(() => {
      const currentTime = Date.now();
      
      setNeurons(prev => 
        prev.map((neuron, index) => {
          const baseActivationRate = isActive ? 0.2 : 0.05; // Reduced rates
          const anomalyBoost = anomalyScore * 0.4; // Reduced boost
          const shouldActivate = Math.random() < (baseActivationRate + anomalyBoost);
          
          if (shouldActivate) {
            let type: 'normal' | 'suspicious' | 'anomaly' = 'normal';
            const random = Math.random();
            
            if (anomalyScore > 0.7 && random < 0.3) {
              type = 'anomaly';
            } else if (anomalyScore > 0.4 && random < 0.2) {
              type = 'suspicious';
            }
            
            return {
              ...neuron,
              active: true,
              type,
              intensity: Math.min(1, (anomalyScore * 0.6) + (Math.random() * 0.3)),
              lastFired: currentTime
            };
          }
          
          const timeSinceLastFire = currentTime - neuron.lastFired;
          const isRecentlyActive = timeSinceLastFire < 2000; // Reduced duration
          
          return {
            ...neuron,
            active: isRecentlyActive,
            intensity: isRecentlyActive ? neuron.intensity * 0.95 : neuron.intensity * 0.9
          };
        })
      );
      
      // Reduced plasticity particles
      if (anomalyScore > 0.5 || isActive) {
        const particleCount = Math.floor((anomalyScore + (isActive ? 0.2 : 0)) * 3); // Reduced count
        setPlasticityParticles(prev => [
          ...prev.filter(p => p.life > 0).slice(0, 20), // Limit particle count
          ...Array.from({ length: particleCount }, () => ({
            x: Math.random() * 100,
            y: Math.random() * 100,
            vx: (Math.random() - 0.5) * 2, // Reduced velocity
            vy: (Math.random() - 0.5) * 2,
            life: 1.0
          }))
        ]);
      }
      
    }, 200); // Reduced frequency from 100ms to 200ms
    
    return () => clearInterval(interval);
  }, [isActive, anomalyScore, neurons]);
  
  // Optimized particle updates
  useEffect(() => {
    if (plasticityParticles.length === 0) return;
    
    const interval = setInterval(() => {
      setPlasticityParticles(prev =>
        prev.map(particle => ({
          ...particle,
          x: particle.x + particle.vx,
          y: particle.y + particle.vy,
          life: particle.life - 0.03 // Faster decay
        })).filter(particle => particle.life > 0 && particle.x >= 0 && particle.x <= 100 && particle.y >= 0 && particle.y <= 100)
      );
    }, 100); // Reduced frequency
    
    return () => clearInterval(interval);
  }, [plasticityParticles]);
  
  return (
    <div className="absolute inset-0 pointer-events-none">
      <div className="relative w-full h-full">
        {/* Simplified Neurons - Removed all connections/lines */}
        {neurons.map((neuron, i) => {
          const currentTime = Date.now();
          const timeSinceLastFire = currentTime - neuron.lastFired;
          const isRecentlyFired = timeSinceLastFire < 800;
          
          let backgroundColor = '';
          
          switch (neuron.type) {
            case 'anomaly':
              backgroundColor = '#FF4444';
              break;
            case 'suspicious':
              backgroundColor = '#FFAA44';
              break;
            default:
              backgroundColor = '#33C3F0';
          }
          
          const baseSize = neuron.size;
          const intensityMultiplier = 1 + (neuron.intensity * 0.3);
          const activityMultiplier = neuron.active ? 1.2 : 1;
          const finalSize = baseSize * intensityMultiplier * activityMultiplier;
          
          return (
            <div 
              key={i}
              className={`rounded-full transition-all duration-200 ${
                isRecentlyFired ? 'neural-glow' : ''
              }`}
              style={{
                left: `${neuron.x}%`,
                top: `${neuron.y}%`,
                width: `${finalSize}px`,
                height: `${finalSize}px`,
                backgroundColor,
                opacity: neuron.active ? 0.8 : 0.3,
                zIndex: neuron.active ? 3 : 2,
                transform: `translate(-50%, -50%) scale(${neuron.active ? 1.1 : 1})`,
                boxShadow: neuron.active ? 
                  `0 0 ${finalSize * 1.5}px ${backgroundColor}` : 
                  `0 0 ${finalSize * 0.5}px ${backgroundColor}`
              }}
            />
          );
        })}
        
        {/* Simplified Neural Plasticity Particles */}
        {plasticityParticles.map((particle, i) => (
          <div
            key={i}
            className="absolute rounded-full"
            style={{
              left: `${particle.x}%`,
              top: `${particle.y}%`,
              width: '1.5px',
              height: '1.5px',
              backgroundColor: '#E5DEFF',
              opacity: particle.life * 0.6,
              transform: 'translate(-50%, -50%)',
              zIndex: 4
            }}
          />
        ))}
        
        {/* Simplified Synaptic Storm Effect */}
        {anomalyScore > 0.8 && (
          <div className="absolute inset-0">
            <div 
              className="absolute inset-0 rounded-full"
              style={{
                background: 'radial-gradient(circle, rgba(255, 68, 68, 0.05), transparent)',
                animation: 'pulse 2s ease-in-out infinite'
              }}
            />
          </div>
        )}
      </div>
    </div>
  );
}
