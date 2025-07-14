
import React, { useState } from 'react';

interface SplineViewerProps {
  scene: string;
  className?: string;
}

export function SplineViewer({ scene, className = "" }: SplineViewerProps) {
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const handleIframeLoad = () => {
    console.log('Spline iframe loaded successfully');
    setIsLoading(false);
  };

  const handleIframeError = () => {
    console.error('Failed to load Spline iframe');
    setError('Failed to load brain visualization');
    setIsLoading(false);
  };

  return (
    <div className={`w-full h-full overflow-hidden rounded-lg relative ${className}`}>
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-background/50 backdrop-blur-sm z-10">
          <div className="flex flex-col items-center">
            <div className="h-8 w-8 border-4 border-t-kenshi-blue border-kenshi-red rounded-full animate-spin"></div>
            <p className="mt-2 text-sm">Loading Neural Network...</p>
          </div>
        </div>
      )}
      {error && (
        <div className="absolute inset-0 flex items-center justify-center bg-background/50 backdrop-blur-sm z-10">
          <div className="text-center p-4">
            <p className="text-kenshi-red mb-2">{error}</p>
            <p className="text-xs text-muted-foreground">The neuromorphic visualization is temporarily unavailable</p>
            <p className="text-xs mt-2">Try checking your network connection or refreshing the page</p>
          </div>
        </div>
      )}
      <iframe 
        src={scene} 
        frameBorder="0" 
        width="100%" 
        height="100%" 
        onLoad={handleIframeLoad}
        onError={handleIframeError}
        title="Kenshi AI Brain Visualization"
        className="w-full h-full"
      />
    </div>
  );
}
