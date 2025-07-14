
import { NeuronData } from './types';

export const createCanvasInteractions = (
  canvas: HTMLCanvasElement,
  neurons: NeuronData[],
  layerVisibility: Record<string, boolean>,
  setHoverNeuron: (neuron: NeuronData | null) => void,
  scaleRef: React.MutableRefObject<number>,
  offsetRef: React.MutableRefObject<{ x: number, y: number }>
) => {
  const isDraggingRef = { current: false };
  const previousPositionRef = { current: { x: 0, y: 0 } };

  const handleMouseDown = (event: MouseEvent) => {
    isDraggingRef.current = true;
    previousPositionRef.current = { x: event.clientX, y: event.clientY };
    canvas.style.cursor = 'grabbing';
  };
  
  const handleMouseMove = (event: MouseEvent) => {
    const rect = canvas.getBoundingClientRect();
    
    if (isDraggingRef.current) {
      const dx = event.clientX - previousPositionRef.current.x;
      const dy = event.clientY - previousPositionRef.current.y;
      
      offsetRef.current.x += dx;
      offsetRef.current.y += dy;
      
      previousPositionRef.current = { x: event.clientX, y: event.clientY };
    } else {
      const transformedX = (event.clientX - rect.left - offsetRef.current.x) / scaleRef.current;
      const transformedY = (event.clientY - rect.top - offsetRef.current.y) / scaleRef.current;
      
      let hoveredNeuron = null;
      
      for (const neuron of neurons) {
        if (!neuron.graphPosition || !layerVisibility[neuron.layer]) continue;
        
        const { x, y, radius } = neuron.graphPosition;
        const distance = Math.sqrt(Math.pow(transformedX - x, 2) + Math.pow(transformedY - y, 2));
        
        if (distance <= radius * 1.5) {
          hoveredNeuron = neuron;
          break;
        }
      }
      
      if (hoveredNeuron) {
        setHoverNeuron(hoveredNeuron);
        canvas.style.cursor = 'pointer';
      } else {
        setHoverNeuron(null);
        canvas.style.cursor = 'grab';
      }
    }
  };
  
  const handleMouseUp = () => {
    isDraggingRef.current = false;
    canvas.style.cursor = 'grab';
  };
  
  const handleWheel = (event: WheelEvent) => {
    event.preventDefault();
    
    const rect = canvas.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;
    
    const delta = event.deltaY > 0 ? 0.9 : 1.1;
    const newScale = scaleRef.current * delta;
    
    if (newScale < 0.5 || newScale > 5) return;
    
    const scaleRatio = newScale / scaleRef.current;
    
    offsetRef.current.x = mouseX - (mouseX - offsetRef.current.x) * scaleRatio;
    offsetRef.current.y = mouseY - (mouseY - offsetRef.current.y) * scaleRatio;
    
    scaleRef.current = newScale;
  };

  // Touch handlers
  const handleTouchStart = (event: TouchEvent) => {
    if (event.touches.length === 1) {
      isDraggingRef.current = true;
      previousPositionRef.current = {
        x: event.touches[0].clientX,
        y: event.touches[0].clientY
      };
    }
  };
  
  const handleTouchMove = (event: TouchEvent) => {
    event.preventDefault();
    
    if (!isDraggingRef.current || event.touches.length !== 1) return;
    
    const touch = event.touches[0];
    
    const dx = touch.clientX - previousPositionRef.current.x;
    const dy = touch.clientY - previousPositionRef.current.y;
    
    offsetRef.current.x += dx;
    offsetRef.current.y += dy;
    
    previousPositionRef.current = { x: touch.clientX, y: touch.clientY };
  };
  
  const handleTouchEnd = () => {
    isDraggingRef.current = false;
  };

  // Add event listeners
  canvas.addEventListener('mousedown', handleMouseDown);
  canvas.addEventListener('mousemove', handleMouseMove);
  canvas.addEventListener('mouseup', handleMouseUp);
  canvas.addEventListener('mouseleave', handleMouseUp);
  canvas.addEventListener('wheel', handleWheel);
  
  canvas.addEventListener('touchstart', handleTouchStart);
  canvas.addEventListener('touchmove', handleTouchMove, { passive: false });
  canvas.addEventListener('touchend', handleTouchEnd);
  
  // Return cleanup function
  return () => {
    canvas.removeEventListener('mousedown', handleMouseDown);
    canvas.removeEventListener('mousemove', handleMouseMove);
    canvas.removeEventListener('mouseup', handleMouseUp);
    canvas.removeEventListener('mouseleave', handleMouseUp);
    canvas.removeEventListener('wheel', handleWheel);
    
    canvas.removeEventListener('touchstart', handleTouchStart);
    canvas.removeEventListener('touchmove', handleTouchMove);
    canvas.removeEventListener('touchend', handleTouchEnd);
  };
};
