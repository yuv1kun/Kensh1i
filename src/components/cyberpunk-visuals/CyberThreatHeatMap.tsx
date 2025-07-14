import React, { useEffect, useRef } from 'react';

interface ThreatPoint {
  x: number;
  y: number;
  intensity: number;
  type: 'malware' | 'ddos' | 'intrusion' | 'anomaly';
}

interface CyberThreatHeatMapProps {
  threats: number;
  anomalyScore: number;
  className?: string;
}

export const CyberThreatHeatMap: React.FC<CyberThreatHeatMapProps> = ({
  threats,
  anomalyScore,
  className
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();
  const threatPointsRef = useRef<ThreatPoint[]>([]);

  useEffect(() => {
    // Generate threat points based on current threats
    const newThreatPoints: ThreatPoint[] = [];
    const threatCount = Math.min(threats * 3, 20);
    
    for (let i = 0; i < threatCount; i++) {
      newThreatPoints.push({
        x: Math.random() * 100,
        y: Math.random() * 100,
        intensity: Math.random() * anomalyScore + 0.3,
        type: ['malware', 'ddos', 'intrusion', 'anomaly'][Math.floor(Math.random() * 4)] as ThreatPoint['type']
      });
    }
    
    threatPointsRef.current = newThreatPoints;
  }, [threats, anomalyScore]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const animate = () => {
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width;
      canvas.height = rect.height;

      // Clear canvas
      ctx.fillStyle = 'rgba(0, 0, 0, 0.1)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Draw threat points
      threatPointsRef.current.forEach((point, index) => {
        const x = (point.x / 100) * canvas.width;
        const y = (point.y / 100) * canvas.height;
        const radius = point.intensity * 30;

        // Create gradient based on threat type
        const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius);
        
        switch (point.type) {
          case 'malware':
            gradient.addColorStop(0, `rgba(239, 68, 68, ${point.intensity})`);
            gradient.addColorStop(1, 'rgba(239, 68, 68, 0)');
            break;
          case 'ddos':
            gradient.addColorStop(0, `rgba(249, 115, 22, ${point.intensity})`);
            gradient.addColorStop(1, 'rgba(249, 115, 22, 0)');
            break;
          case 'intrusion':
            gradient.addColorStop(0, `rgba(168, 85, 247, ${point.intensity})`);
            gradient.addColorStop(1, 'rgba(168, 85, 247, 0)');
            break;
          case 'anomaly':
            gradient.addColorStop(0, `rgba(59, 130, 246, ${point.intensity})`);
            gradient.addColorStop(1, 'rgba(59, 130, 246, 0)');
            break;
        }

        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fill();

        // Animate point movement
        point.x += (Math.sin(Date.now() / 1000 + index) * 0.1);
        point.y += (Math.cos(Date.now() / 1000 + index) * 0.1);
        
        // Keep points within bounds
        point.x = Math.max(0, Math.min(100, point.x));
        point.y = Math.max(0, Math.min(100, point.y));
      });

      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  return (
    <div className={`relative overflow-hidden rounded-lg ${className}`}>
      <canvas
        ref={canvasRef}
        className="w-full h-full"
        style={{ background: 'linear-gradient(45deg, rgba(0,0,0,0.8), rgba(20,20,40,0.8))' }}
      />
      <div className="absolute top-2 left-2 text-xs text-white/70">
        Threat Intensity Map
      </div>
    </div>
  );
};
