
import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';

interface NeuronData {
  id: string;
  position: number[];
  layer: string;
  connections: string[];
  lastFired: number | null;
  spikeCount: number;
  spikeHistory: number[];
  trafficState?: 'normal' | 'suspicious' | 'anomaly';
  intensity?: number;
}

interface ThreeJsViewProps {
  neurons: NeuronData[];
  layerVisibility: Record<string, boolean>;
  setHoverNeuron: (neuron: NeuronData | null) => void;
  zoomLevel: number;
}

const ThreeJsView: React.FC<ThreeJsViewProps> = ({ 
  neurons, 
  layerVisibility, 
  setHoverNeuron,
  zoomLevel
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const isDraggingRef = useRef(false);
  const previousMousePositionRef = useRef({ x: 0, y: 0 });
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const neuronMeshesRef = useRef<THREE.Mesh[]>([]);
  const raycasterRef = useRef<THREE.Raycaster | null>(null);
  const mouseRef = useRef<THREE.Vector2>(new THREE.Vector2());
  const animationIdRef = useRef<number | null>(null);

  // Initialize Three.js scene with simplified setup
  useEffect(() => {
    if (!canvasRef.current) return;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, canvasRef.current.clientWidth / canvasRef.current.clientHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ 
      canvas: canvasRef.current, 
      antialias: false, // Disable antialiasing for performance
      alpha: true,
      powerPreference: "high-performance"
    });
    
    renderer.setSize(canvasRef.current.clientWidth, canvasRef.current.clientHeight);
    renderer.setClearColor(0x000000, 0);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2)); // Limit pixel ratio for performance
    
    // Simplified lighting setup
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.3);
    scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.7);
    directionalLight.position.set(5, 5, 5);
    scene.add(directionalLight);
    
    camera.position.z = 15 - (zoomLevel - 10) * 0.75;
    
    const raycaster = new THREE.Raycaster();
    
    sceneRef.current = scene;
    cameraRef.current = camera;
    rendererRef.current = renderer;
    raycasterRef.current = raycaster;
    
    // Simplified animation loop
    const animate = () => {
      if (!rendererRef.current || !sceneRef.current || !cameraRef.current) return;
      
      animationIdRef.current = requestAnimationFrame(animate);
      
      // Simplified neuron pulsation
      neuronMeshesRef.current.forEach(mesh => {
        const neuron = mesh.userData.neuronData;
        if (neuron.lastFired) {
          const timeSinceLastFire = Date.now() / 1000 - neuron.lastFired;
          if (timeSinceLastFire < 2) { // Reduced duration
            const pulseScale = 1 + Math.sin(timeSinceLastFire * Math.PI * 2) * 0.3;
            mesh.scale.setScalar(pulseScale);
            
            if (mesh.material instanceof THREE.MeshStandardMaterial) {
              mesh.material.emissiveIntensity = 0.5 + Math.sin(timeSinceLastFire * Math.PI * 2) * 0.3;
            }
          } else {
            mesh.scale.setScalar(1);
            if (mesh.material instanceof THREE.MeshStandardMaterial) {
              mesh.material.emissiveIntensity = 0.1;
            }
          }
        }
      });
      
      // Gentle auto-rotation
      if (!isDraggingRef.current) {
        cameraRef.current.position.x = cameraRef.current.position.x * Math.cos(0.001) - cameraRef.current.position.z * Math.sin(0.001);
        cameraRef.current.position.z = cameraRef.current.position.x * Math.sin(0.001) + cameraRef.current.position.z * Math.cos(0.001);
        cameraRef.current.lookAt(0, 0, 0);
      }
      
      rendererRef.current.render(sceneRef.current, cameraRef.current);
    };
    
    animate();
    
    const handleResize = () => {
      if (!canvasRef.current || !cameraRef.current || !rendererRef.current) return;
      
      const width = canvasRef.current.clientWidth;
      const height = canvasRef.current.clientHeight;
      
      cameraRef.current.aspect = width / height;
      cameraRef.current.updateProjectionMatrix();
      
      rendererRef.current.setSize(width, height);
    };
    
    window.addEventListener('resize', handleResize);
    
    return () => {
      if (animationIdRef.current) {
        cancelAnimationFrame(animationIdRef.current);
      }
      window.removeEventListener('resize', handleResize);
      
      if (sceneRef.current) {
        sceneRef.current.children.forEach(child => {
          if (child instanceof THREE.Mesh) {
            if (child.geometry) child.geometry.dispose();
            if (child.material) {
              if (Array.isArray(child.material)) {
                child.material.forEach(material => material.dispose());
              } else {
                child.material.dispose();
              }
            }
          }
        });
      }
      
      if (rendererRef.current) {
        rendererRef.current.dispose();
      }
    };
  }, [zoomLevel]);

  // Simplified neuron mesh updates
  useEffect(() => {
    if (!sceneRef.current) return;
    
    const handleMouseMove = (event: MouseEvent) => {
      if (!canvasRef.current || !cameraRef.current || !raycasterRef.current) return;
      
      const rect = canvasRef.current.getBoundingClientRect();
      mouseRef.current.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      mouseRef.current.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      
      raycasterRef.current.setFromCamera(mouseRef.current, cameraRef.current);
      const intersects = raycasterRef.current.intersectObjects(neuronMeshesRef.current);
      
      if (intersects.length > 0) {
        const intersectedNeuron = intersects[0].object;
        if (intersectedNeuron.userData.neuronData) {
          setHoverNeuron(intersectedNeuron.userData.neuronData);
          canvasRef.current.style.cursor = 'pointer';
        }
      } else {
        setHoverNeuron(null);
        canvasRef.current.style.cursor = isDraggingRef.current ? 'grabbing' : 'grab';
      }
      
      if (isDraggingRef.current && cameraRef.current) {
        const deltaX = event.clientX - previousMousePositionRef.current.x;
        const deltaY = event.clientY - previousMousePositionRef.current.y;
        
        cameraRef.current.position.x = 
          cameraRef.current.position.x * Math.cos(deltaX * 0.01) - 
          cameraRef.current.position.z * Math.sin(deltaX * 0.01);
          
        cameraRef.current.position.z = 
          cameraRef.current.position.x * Math.sin(deltaX * 0.01) + 
          cameraRef.current.position.z * Math.cos(deltaX * 0.01);
          
        cameraRef.current.position.y += deltaY * 0.05;
        cameraRef.current.position.y = Math.max(-10, Math.min(10, cameraRef.current.position.y));
        
        cameraRef.current.lookAt(0, 0, 0);
      }
      
      previousMousePositionRef.current = { x: event.clientX, y: event.clientY };
    };
    
    const handleMouseDown = () => {
      isDraggingRef.current = true;
      if (canvasRef.current) canvasRef.current.style.cursor = 'grabbing';
    };
    
    const handleMouseUp = () => {
      isDraggingRef.current = false;
      if (canvasRef.current) canvasRef.current.style.cursor = 'grab';
    };
    
    const handleWheel = (event: WheelEvent) => {
      event.preventDefault();
      
      if (!cameraRef.current) return;
      
      cameraRef.current.position.z += event.deltaY * 0.01;
      cameraRef.current.position.z = Math.max(2, Math.min(30, cameraRef.current.position.z));
    };
    
    // Simplified connection material
    const connectionMaterial = new THREE.LineBasicMaterial({
      color: 0xD3E4FD,
      opacity: 0.3,
      transparent: true
    });
    
    // Clear existing meshes
    neuronMeshesRef.current.forEach(mesh => {
      sceneRef.current?.remove(mesh);
    });
    neuronMeshesRef.current = [];
    
    // Clear existing connections
    sceneRef.current.children.forEach(child => {
      if (child instanceof THREE.Line) {
        sceneRef.current?.remove(child);
      }
    });
    
    // Create simplified neurons
    const neuronGeometry = new THREE.SphereGeometry(0.2, 16, 16); // Reduced geometry complexity
    
    neurons.forEach(neuron => {
      if (!layerVisibility[neuron.layer]) return;
      
      let color, emissiveColor;
      switch (neuron.layer) {
        case 'input': 
          color = 0x33C3F0; 
          emissiveColor = 0x1a5f80;
          break;
        case 'hidden': 
          color = 0xE5DEFF; 
          emissiveColor = 0x8a7fb8;
          break;
        case 'output': 
          color = 0xD3E4FD; 
          emissiveColor = 0x6a7290;
          break;
        default: 
          color = 0xF1F0FB;
          emissiveColor = 0x787590;
      }
      
      // Simplified material
      const material = new THREE.MeshBasicMaterial({ 
        color,
        transparent: true,
        opacity: 0.8
      });
      
      const mesh = new THREE.Mesh(neuronGeometry, material);
      const position = new THREE.Vector3(neuron.position[0], neuron.position[1], neuron.position[2]);
      mesh.position.copy(position);
      mesh.userData.neuronId = neuron.id;
      mesh.userData.neuronData = neuron;
      neuronMeshesRef.current.push(mesh);
      sceneRef.current?.add(mesh);
    });
    
    // Simplified connections (only show some to reduce complexity)
    neurons.forEach((fromNeuron, index) => {
      if (!layerVisibility[fromNeuron.layer] || index % 2 !== 0) return; // Show only every other neuron's connections
      
      const fromPos = new THREE.Vector3(...fromNeuron.position);
      
      fromNeuron.connections.slice(0, 2).forEach(toId => { // Limit connections per neuron
        const toNeuron = neurons.find(n => n.id === toId);
        if (!toNeuron || !layerVisibility[toNeuron.layer]) return;
        
        const toPos = new THREE.Vector3(...toNeuron.position);
        const points = [fromPos, toPos];
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        const line = new THREE.Line(geometry, connectionMaterial);
        sceneRef.current?.add(line);
      });
    });
    
    if (canvasRef.current) {
      canvasRef.current.addEventListener('mousemove', handleMouseMove);
      canvasRef.current.addEventListener('mousedown', handleMouseDown);
      canvasRef.current.addEventListener('wheel', handleWheel);
    }
    
    window.addEventListener('mouseup', handleMouseUp);
    
    return () => {
      if (canvasRef.current) {
        canvasRef.current.removeEventListener('mousemove', handleMouseMove);
        canvasRef.current.removeEventListener('mousedown', handleMouseDown);
        canvasRef.current.removeEventListener('wheel', handleWheel);
      }
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [neurons, layerVisibility, setHoverNeuron]);
  
  // Simplified zoom update
  useEffect(() => {
    if (!cameraRef.current) return;
    
    const targetZ = 15 - (zoomLevel - 5) * 0.6;
    cameraRef.current.position.z = targetZ;
  }, [zoomLevel]);

  return (
    <canvas 
      ref={canvasRef}
      className="w-full h-full cursor-grab active:cursor-grabbing" 
    />
  );
};

export default ThreeJsView;
