
import React from 'react';
import { cn } from '@/lib/utils';

interface HolographicDisplayProps {
  title: string;
  children: React.ReactNode;
  className?: string;
  glowColor?: 'blue' | 'red' | 'green' | 'purple';
}

export const HolographicDisplay: React.FC<HolographicDisplayProps> = ({
  title,
  children,
  className,
  glowColor = 'blue'
}) => {
  const glowColorMap = {
    blue: 'shadow-[0_0_20px_rgba(59,130,246,0.5)] border-blue-500/50',
    red: 'shadow-[0_0_20px_rgba(239,68,68,0.5)] border-red-500/50',
    green: 'shadow-[0_0_20px_rgba(34,197,94,0.5)] border-green-500/50',
    purple: 'shadow-[0_0_20px_rgba(168,85,247,0.5)] border-purple-500/50'
  };

  return (
    <div className={cn(
      "relative bg-background/20 backdrop-blur-md border rounded-lg p-4",
      "before:absolute before:inset-0 before:bg-gradient-to-r before:from-transparent before:via-white/10 before:to-transparent",
      "before:animate-pulse before:rounded-lg",
      glowColorMap[glowColor],
      className
    )}>
      <div className="relative z-10">
        <div className="flex items-center gap-2 mb-3">
          <div className={cn(
            "h-2 w-2 rounded-full animate-pulse",
            glowColor === 'blue' && "bg-blue-500",
            glowColor === 'red' && "bg-red-500",
            glowColor === 'green' && "bg-green-500",
            glowColor === 'purple' && "bg-purple-500"
          )}></div>
          <h3 className="text-sm font-medium text-white/90 tracking-wider uppercase">
            {title}
          </h3>
        </div>
        {children}
      </div>
      
      {/* Scanning line effect */}
      <div className="absolute inset-0 overflow-hidden rounded-lg">
        <div className={cn(
          "absolute top-0 left-0 w-full h-0.5 opacity-60 animate-[scanline_3s_ease-in-out_infinite]",
          glowColor === 'blue' && "bg-blue-400",
          glowColor === 'red' && "bg-red-400",
          glowColor === 'green' && "bg-green-400",
          glowColor === 'purple' && "bg-purple-400"
        )}></div>
      </div>
    </div>
  );
};
