
import React, { useEffect, useRef } from 'react';

interface AttackVector {
  angle: number;
  distance: number;
  intensity: number;
  type: string;
}

interface AttackVectorRadarProps {
  threats: number;
  anomalyScore: number;
  className?: string;
}

export const AttackVectorRadar: React.FC<AttackVectorRadarProps> = ({
  threats,
  anomalyScore,
  className
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sweepAngleRef = useRef<number>(0);
  const attackVectorsRef = useRef<AttackVector[]>([]);

  useEffect(() => {
    // Generate attack vectors based on threats
    const vectors: AttackVector[] = [];
    const vectorCount = Math.min(threats * 2, 12);
    
    for (let i = 0; i < vectorCount; i++) {
      vectors.push({
        angle: (Math.PI * 2 * i) / vectorCount + (Math.random() * 0.5),
        distance: Math.random() * 0.8 + 0.2,
        intensity: Math.random() * anomalyScore + 0.3,
        type: ['Malware', 'DDoS', 'Phishing', 'Intrusion', 'Anomaly'][Math.floor(Math.random() * 5)]
      });
    }
    
    attackVectorsRef.current = vectors;
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

      const centerX = canvas.width / 2;
      const centerY = canvas.height / 2;
      const radius = Math.min(centerX, centerY) * 0.9;

      // Clear canvas
      ctx.fillStyle = 'rgba(0, 0, 0, 0.9)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Draw radar circles
      ctx.strokeStyle = 'rgba(0, 255, 255, 0.3)';
      ctx.lineWidth = 1;
      for (let i = 1; i <= 4; i++) {
        ctx.beginPath();
        ctx.arc(centerX, centerY, (radius * i) / 4, 0, Math.PI * 2);
        ctx.stroke();
      }

      // Draw radar lines
      for (let i = 0; i < 8; i++) {
        const angle = (Math.PI * 2 * i) / 8;
        ctx.beginPath();
        ctx.moveTo(centerX, centerY);
        ctx.lineTo(
          centerX + Math.cos(angle) * radius,
          centerY + Math.sin(angle) * radius
        );
        ctx.stroke();
      }

      // Draw sweep
      sweepAngleRef.current += 0.02;
      const sweepGradient = ctx.createConicGradient(sweepAngleRef.current, centerX, centerY);
      sweepGradient.addColorStop(0, 'rgba(0, 255, 255, 0)');
      sweepGradient.addColorStop(0.1, 'rgba(0, 255, 255, 0.3)');
      sweepGradient.addColorStop(0.2, 'rgba(0, 255, 255, 0)');
      
      ctx.fillStyle = sweepGradient;
      ctx.beginPath();
      ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
      ctx.fill();

      // Draw attack vectors
      attackVectorsRef.current.forEach((vector, index) => {
        const x = centerX + Math.cos(vector.angle) * radius * vector.distance;
        const y = centerY + Math.sin(vector.angle) * radius * vector.distance;

        // Pulsing effect
        const pulse = Math.sin(Date.now() / 300 + index) * 0.5 + 0.5;
        const size = 3 + pulse * 2 * vector.intensity;

        ctx.save();
        ctx.fillStyle = vector.intensity > 0.7 ? '#ff0040' : vector.intensity > 0.4 ? '#ff8000' : '#00ff80';
        ctx.shadowBlur = 10;
        ctx.shadowColor = ctx.fillStyle;
        ctx.beginPath();
        ctx.arc(x, y, size, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();

        // Draw vector type label
        ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
        ctx.font = '8px monospace';
        ctx.fillText(vector.type, x + 8, y - 8);
      });

      requestAnimationFrame(animate);
    };

    animate();
  }, []);

  return (
    <div className={`relative ${className}`}>
      <canvas ref={canvasRef} className="w-full h-full rounded-lg" />
      <div className="absolute top-2 left-2 text-xs text-cyan-400 font-mono">
        ATTACK VECTOR RADAR
      </div>
      <div className="absolute bottom-2 right-2 text-xs text-cyan-400 font-mono">
        {threats} ACTIVE THREATS
      </div>
    </div>
  );
};
