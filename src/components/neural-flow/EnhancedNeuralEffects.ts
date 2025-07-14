
import * as THREE from 'three';
import { NeuronData } from './types';

export interface SynapticStorm {
  id: string;
  epicenter: THREE.Vector3;
  intensity: number;
  radius: number;
  duration: number;
  startTime: number;
  particles: THREE.Mesh[];
}

export interface NeuralPathway {
  id: string;
  points: THREE.Vector3[];
  intensity: number;
  color: string;
  width: number;
  pulseSpeed: number;
  lastPulse: number;
}

export interface MemoryEngram {
  id: string;
  position: THREE.Vector3;
  strength: number;
  pattern: number[];
  connectedNeurons: string[];
  glowIntensity: number;
  size: number;
}

export class EnhancedNeuralEffects {
  private scene: THREE.Scene;
  private synapticStorms: Map<string, SynapticStorm> = new Map();
  private neuralPathways: Map<string, NeuralPathway> = new Map();
  private memoryEngrams: Map<string, MemoryEngram> = new Map();
  private plasticityParticles: THREE.Points[] = [];

  constructor(scene: THREE.Scene) {
    this.scene = scene;
  }

  // Simplified synaptic storm effect
  createSynapticStorm(epicenter: THREE.Vector3, intensity: number): string {
    const stormId = `storm_${Date.now()}_${Math.random()}`;
    const particles: THREE.Mesh[] = [];
    
    // Reduced particle count for performance
    const particleGeometry = new THREE.SphereGeometry(0.02, 4, 4);
    const particleCount = Math.floor(intensity * 20); // Reduced from 50
    
    for (let i = 0; i < particleCount; i++) {
      const material = new THREE.MeshBasicMaterial({
        color: 0xff4444,
        transparent: true,
        opacity: 0.6
      });
      
      const particle = new THREE.Mesh(particleGeometry, material);
      
      const angle = Math.random() * Math.PI * 2;
      const radius = Math.random() * 2; // Reduced radius
      const height = (Math.random() - 0.5) * 1; // Reduced height
      
      particle.position.set(
        epicenter.x + Math.cos(angle) * radius,
        epicenter.y + height,
        epicenter.z + Math.sin(angle) * radius
      );
      
      particle.userData = {
        velocity: new THREE.Vector3(
          (Math.random() - 0.5) * 0.05, // Reduced velocity
          (Math.random() - 0.5) * 0.05,
          (Math.random() - 0.5) * 0.05
        ),
        life: 1.0
      };
      
      particles.push(particle);
      this.scene.add(particle);
    }
    
    const storm: SynapticStorm = {
      id: stormId,
      epicenter: epicenter.clone(),
      intensity,
      radius: 0,
      duration: 3000, // Reduced from 5000
      startTime: Date.now(),
      particles
    };
    
    this.synapticStorms.set(stormId, storm);
    return stormId;
  }

  // Simplified neural pathway
  createNeuralPathway(fromPos: THREE.Vector3, toPos: THREE.Vector3, intensity: number): string {
    const pathwayId = `pathway_${Date.now()}_${Math.random()}`;
    
    // Simplified straight line instead of curve
    const points = [fromPos, toPos];
    
    const pathway: NeuralPathway = {
      id: pathwayId,
      points,
      intensity,
      color: intensity > 0.7 ? '#FF4444' : intensity > 0.4 ? '#FFAA44' : '#44AAFF',
      width: 0.5 + intensity * 1.5,
      pulseSpeed: 0.02 + intensity * 0.03,
      lastPulse: Date.now()
    };
    
    this.neuralPathways.set(pathwayId, pathway);
    return pathwayId;
  }

  // Simplified memory engram
  createMemoryEngram(position: THREE.Vector3, pattern: number[], strength: number): string {
    const engramId = `engram_${Date.now()}_${Math.random()}`;
    
    const engram: MemoryEngram = {
      id: engramId,
      position: position.clone(),
      strength,
      pattern,
      connectedNeurons: [],
      glowIntensity: strength,
      size: 0.3 + strength * 0.7
    };
    
    // Simplified visual representation
    const geometry = new THREE.SphereGeometry(engram.size, 8, 8); // Reduced complexity
    const material = new THREE.MeshBasicMaterial({
      color: 0xaa77ff,
      transparent: true,
      opacity: 0.4
    });
    
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.copy(position);
    mesh.userData = { engramId, type: 'memory_engram' };
    
    this.scene.add(mesh);
    this.memoryEngrams.set(engramId, engram);
    
    return engramId;
  }

  // Simplified plasticity particles
  createPlasticityParticles(neuronPositions: THREE.Vector3[], plasticityIndex: number): void {
    // Remove old particles
    this.plasticityParticles.forEach(particles => {
      this.scene.remove(particles);
      if (particles.geometry) particles.geometry.dispose();
      if (particles.material) {
        if (Array.isArray(particles.material)) {
          particles.material.forEach(mat => mat.dispose());
        } else {
          particles.material.dispose();
        }
      }
    });
    this.plasticityParticles = [];
    
    if (plasticityIndex < 0.3) return;
    
    // Reduced particle count
    const particleCount = Math.floor(plasticityIndex * 50); // Reduced from 200
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);
    
    for (let i = 0; i < particleCount; i++) {
      const i3 = i * 3;
      
      // Simplified positioning
      positions[i3] = (Math.random() - 0.5) * 10; // Reduced range
      positions[i3 + 1] = (Math.random() - 0.5) * 10;
      positions[i3 + 2] = (Math.random() - 0.5) * 10;
      
      // Simplified coloring
      const color = new THREE.Color(0x55aaff);
      colors[i3] = color.r;
      colors[i3 + 1] = color.g;
      colors[i3 + 2] = color.b;
    }
    
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    
    const material = new THREE.PointsMaterial({
      size: 0.05,
      vertexColors: true,
      transparent: true,
      opacity: 0.4
    });
    
    const particles = new THREE.Points(geometry, material);
    this.plasticityParticles.push(particles);
    this.scene.add(particles);
  }

  // Simplified update loop
  updateEffects(): void {
    const currentTime = Date.now();
    
    // Update storms with simplified logic
    this.synapticStorms.forEach((storm, id) => {
      const elapsed = currentTime - storm.startTime;
      const progress = elapsed / storm.duration;
      
      if (progress >= 1) {
        storm.particles.forEach(particle => {
          this.scene.remove(particle);
        });
        this.synapticStorms.delete(id);
      } else {
        storm.particles.forEach(particle => {
          if (particle.userData.velocity) {
            particle.position.add(particle.userData.velocity);
            particle.userData.life -= 0.03; // Faster decay
            
            if (particle.material instanceof THREE.MeshBasicMaterial) {
              particle.material.opacity = Math.max(0, particle.userData.life * 0.6);
            }
          }
        });
      }
    });
    
    // Simplified engram updates
    this.memoryEngrams.forEach((engram, id) => {
      const engramMesh = this.scene.children.find(child => 
        child.userData.engramId === id
      ) as THREE.Mesh;
      
      if (engramMesh) {
        const pulse = Math.sin(currentTime * 0.002) * 0.1 + 1;
        engramMesh.scale.setScalar(pulse);
        engramMesh.rotation.y += 0.005; // Slower rotation
      }
    });
    
    // Simplified particle updates
    this.plasticityParticles.forEach(particles => {
      if (particles.geometry.attributes.position) {
        const positions = particles.geometry.attributes.position.array as Float32Array;
        
        for (let i = 0; i < positions.length; i += 3) {
          positions[i + 1] += Math.sin(currentTime * 0.0005 + i) * 0.005; // Slower movement
        }
        
        particles.geometry.attributes.position.needsUpdate = true;
      }
    });
  }

  // Clean up all effects
  dispose(): void {
    this.synapticStorms.forEach(storm => {
      storm.particles.forEach(particle => {
        this.scene.remove(particle);
      });
    });
    this.synapticStorms.clear();
    
    this.neuralPathways.clear();
    
    this.memoryEngrams.forEach((engram, id) => {
      const mesh = this.scene.children.find(child => 
        child.userData.engramId === id
      );
      if (mesh) {
        this.scene.remove(mesh);
      }
    });
    this.memoryEngrams.clear();
    
    this.plasticityParticles.forEach(particles => {
      this.scene.remove(particles);
      if (particles.geometry) particles.geometry.dispose();
      if (particles.material) {
        if (Array.isArray(particles.material)) {
          particles.material.forEach(mat => mat.dispose());
        } else {
          particles.material.dispose();
        }
      }
    });
    this.plasticityParticles = [];
  }
}
