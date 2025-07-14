import React, { useEffect, useRef } from 'react';

interface DataParticle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  size: number;
  color: string;
  life: number;
}

interface DataFlowRiversProps {
  isActive: boolean;
  intensity: number;
  className?: string;
}

export const DataFlowRivers: React.FC<DataFlowRiversProps> = ({
  isActive,
  intensity,
  className
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const particlesRef = useRef<DataParticle[]>([]);
  const animationRef = useRef<number>();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size properly
    const resizeCanvas = () => {
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width;
      canvas.height = rect.height;
    };

    // Initial resize
    resizeCanvas();
    
    // Add resize listener
    window.addEventListener('resize', resizeCanvas);

    const createParticle = (): DataParticle => {
      const colors = ['#00f5ff', '#ff0080', '#00ff41', '#ff8c00', '#8a2be2'];
      return {
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 4,
        vy: (Math.random() - 0.5) * 4,
        size: Math.random() * 3 + 1,
        color: colors[Math.floor(Math.random() * colors.length)],
        life: 1.0
      };
    };

    // Initialize some particles immediately for visual feedback
    if (particlesRef.current.length === 0) {
      for (let i = 0; i < 20; i++) {
        particlesRef.current.push(createParticle());
      }
    }

    const animate = () => {
      // Ensure canvas dimensions are current
      const rect = canvas.getBoundingClientRect();
      if (canvas.width !== rect.width || canvas.height !== rect.height) {
        canvas.width = rect.width;
        canvas.height = rect.height;
      }

      // Create new particles if active
      if (isActive && particlesRef.current.length < intensity * 100) {
        for (let i = 0; i < Math.floor(intensity * 5); i++) {
          particlesRef.current.push(createParticle());
        }
      }

      // Clear canvas with trailing effect
      ctx.fillStyle = 'rgba(0, 0, 0, 0.05)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Update and draw particles
      particlesRef.current = particlesRef.current.filter(particle => {
        // Update position
        particle.x += particle.vx;
        particle.y += particle.vy;
        particle.life -= isActive ? 0.005 : 0.002; // Slower decay when not active

        // Bounce off edges
        if (particle.x <= 0 || particle.x >= canvas.width) particle.vx *= -1;
        if (particle.y <= 0 || particle.y >= canvas.height) particle.vy *= -1;

        // Keep particles in bounds
        particle.x = Math.max(0, Math.min(canvas.width, particle.x));
        particle.y = Math.max(0, Math.min(canvas.height, particle.y));

        // Draw particle
        ctx.save();
        ctx.globalAlpha = particle.life;
        ctx.fillStyle = particle.color;
        ctx.shadowBlur = 10;
        ctx.shadowColor = particle.color;
        ctx.beginPath();
        ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();

        // Draw connections to nearby particles
        particlesRef.current.forEach(other => {
          if (other === particle) return;
          
          const dx = particle.x - other.x;
          const dy = particle.y - other.y;
          const distance = Math.sqrt(dx * dx + dy * dy);

          if (distance < 80) {
            ctx.save();
            ctx.globalAlpha = (1 - distance / 80) * 0.3 * particle.life;
            ctx.strokeStyle = particle.color;
            ctx.lineWidth = 0.5;
            ctx.beginPath();
            ctx.moveTo(particle.x, particle.y);
            ctx.lineTo(other.x, other.y);
            ctx.stroke();
            ctx.restore();
          }
        });

        return particle.life > 0;
      });

      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener('resize', resizeCanvas);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isActive, intensity]);

  return (
    <canvas
      ref={canvasRef}
      className={`w-full h-full ${className}`}
      style={{ background: 'radial-gradient(circle, rgba(0,0,0,0.9), rgba(0,20,40,0.9))' }}
    />
  );
};
