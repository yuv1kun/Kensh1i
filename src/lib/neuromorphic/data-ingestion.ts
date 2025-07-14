import { nanoid, pcap, Subject } from './mock-dependencies';
import { NetworkPacket, PacketSource, PacketStatistics, FeatureVector } from './types';

/**
 * NetworkDataIngestion - Responsible for capturing and processing network traffic
 * to feed into the neuromorphic computing engine
 */
export class NetworkDataIngestion {
  private packetSource: PacketSource | null = null;
  private isCapturing: boolean = false;
  private captureSession: string | null = null;
  private captureStartTime: number = 0;
  private packetBuffer: NetworkPacket[] = [];
  private stats: PacketStatistics = this.initializeStatistics();
  private packetSubject = new Subject<NetworkPacket>();
  private featureSubject = new Subject<FeatureVector>();
  private usePcap: boolean = false; // Set to true to use actual pcap for real traffic

  /**
   * Start capturing network packets
   */
  startCapture(networkInterface?: string): string {
    if (this.isCapturing) {
      return this.captureSession || '';
    }

    this.captureSession = nanoid();
    this.captureStartTime = Date.now();
    this.isCapturing = true;
    this.stats = this.initializeStatistics();
    this.packetBuffer = [];

    if (this.usePcap) {
      try {
        // Setup real packet capture
        this.packetSource = pcap.createSession(); // Using the mock implementation which takes no args
        this.setupPacketCapture();
      } catch (error) {
        console.error('Failed to initialize packet capture:', error);
        this.simulateNetworkTraffic(); // Fallback to simulation
      }
    } else {
      // For browser environments, we simulate network traffic
      this.simulateNetworkTraffic();
    }

    return this.captureSession;
  }

  /**
   * Stop the current capture session
   */
  stopCapture(): void {
    if (!this.isCapturing) return;

    this.isCapturing = false;
    
    if (this.usePcap && this.packetSource) {
      try {
        // Close the packet capture session
        (this.packetSource as any).close();
      } catch (error) {
        console.error('Error closing packet capture:', error);
      }
    }
    
    this.packetSource = null;
    this.captureSession = null;
  }

  /**
   * Get the current packet statistics
   */
  getStatistics(): PacketStatistics {
    return { ...this.stats };
  }

  /**
   * Subscribe to raw packet events
   */
  subscribeToPackets(callback: (packet: NetworkPacket) => void): { unsubscribe: () => void } {
    const subscription = this.packetSubject.subscribe(callback);
    return { 
      unsubscribe: () => subscription.unsubscribe() 
    };
  }

  /**
   * Subscribe to processed feature vectors
   */
  subscribeToFeatures(callback: (features: FeatureVector) => void): { unsubscribe: () => void } {
    const subscription = this.featureSubject.subscribe(callback);
    return {
      unsubscribe: () => subscription.unsubscribe()
    };
  }

  /**
   * Set up the packet capture handlers
   */
  private setupPacketCapture(): void {
    if (this.packetSource) {
      this.packetSource.on('packet', (rawPacket: any) => {
        const packet = this.processRawPacket(rawPacket);
        this.handlePacket(packet);
      });
    }
  }

  /**
   * Process a raw packet into our NetworkPacket format
   */
  private processRawPacket(rawPacket: any): NetworkPacket {
    // Mocked packet data for simulation
    const packet: NetworkPacket = {
      id: nanoid(),
      timestamp: Date.now(),
      sourceIP: '192.168.' + Math.floor(Math.random() * 255) + '.' + Math.floor(Math.random() * 255),
      destinationIP: '10.0.' + Math.floor(Math.random() * 255) + '.' + Math.floor(Math.random() * 255),
      protocol: String(Math.floor(Math.random() * 4) + 1), // 1-TCP, 2-UDP, 3-ICMP, 4-Other
      size: Math.floor(Math.random() * 1460) + 40, // Random packet size between 40 and 1500 bytes
      sourcePort: Math.floor(Math.random() * 65535),
      destinationPort: Math.floor(Math.random() * 65535),
      flags: ['SYN', 'ACK', 'FIN'].slice(0, Math.floor(Math.random() * 3) + 1),
      ttl: Math.floor(Math.random() * 64) + 1,
      payload: new Uint8Array(0), // Empty payload for mock
      networkInterface: 'eth0',
      direction: Math.random() > 0.5 ? 'inbound' : 'outbound'
    };
    
    return packet;
  }

  /**
   * Handle a new packet (real or simulated)
   */
  private handlePacket(packet: NetworkPacket): void {
    // Update statistics
    this.updateStatistics(packet);
    
    // Store in buffer (limited size)
    this.packetBuffer.push(packet);
    if (this.packetBuffer.length > 1000) {
      this.packetBuffer.shift();
    }
    
    // Process packet into features
    const features = this.createFeatureVector(packet);
    
    // Emit events
    this.packetSubject.next(packet);
    this.featureSubject.next(features);
  }

  /**
   * Update packet statistics based on a new packet
   */
  private updateStatistics(packet: NetworkPacket): void {
    this.stats.totalPackets++;
    this.stats.totalBytes += packet.size;
    
    // Update protocol stats
    const protocolKey = this.getProtocolKey(packet.protocol);
    this.stats.protocolDistribution[protocolKey] = 
      (this.stats.protocolDistribution[protocolKey] || 0) + 1;
    
    // Update port stats
    if (packet.destinationPort) {
      const portKey = `${packet.destinationPort}`;
      this.stats.portDistribution[portKey] = 
        (this.stats.portDistribution[portKey] || 0) + 1;
    }
    
    // Simple rate calculation
    const currentTime = Date.now();
    const elapsedSeconds = (currentTime - this.captureStartTime) / 1000;
    if (elapsedSeconds > 0) {
      this.stats.packetsPerSecond = this.stats.totalPackets / elapsedSeconds;
      this.stats.bytesPerSecond = this.stats.totalBytes / elapsedSeconds;
    }
  }

  /**
   * Extract features from a network packet for analysis
   */
  private createFeatureVector(packet: NetworkPacket): FeatureVector {
    // Calculate features
    const protocol = parseInt(packet.protocol) || 0;
    const packetSize = packet.size;
    // This is where we'd implement sophisticated feature engineering
    const features: FeatureVector = {
      timestamp: packet.timestamp,
      protocol: protocol,          // Normalized protocol type
      packetSize: packetSize,         // Normalized packet size
      sourcePortCategory: this.categorizePort(packet.sourcePort),
      destPortCategory: this.categorizePort(packet.destinationPort),
      flagsVector: [1, 0, 0, 0, 0], // Simplified flag vector
      payloadEntropy: this.calculatePayloadEntropy(packet.payload),
      srcIpEntropy: this.calculateIpEntropy(packet.sourceIP),
      dstIpEntropy: this.calculateIpEntropy(packet.destinationIP),
      isIntranet: this.isIntranetTraffic(packet.sourceIP, packet.destinationIP),
      headerFields: this.extractHeaderFeatures(packet),
      interPacketTime: this.calculateInterPacketTime(packet),
      packetRatio: this.calculatePacketRatio()
    };
    
    return features;
  }

  /**
   * Normalize protocol to a feature vector element between 0 and 1
   */
  private normalizeProtocol(protocol: number): number {
    // Map common protocols to normalized values
    const protocolMap: Record<number, number> = {
      1: 0.1, // ICMP
      6: 0.3, // TCP
      17: 0.5, // UDP
      // Add more as needed
    };
    
    return protocolMap[protocol] || 0.9; // 0.9 for rare/unusual protocols
  }

  /**
   * Normalize packet size to a value between 0 and 1
   */
  private normalizePacketSize(size: number): number {
    // Typical packet sizes range from 40 to 1500 bytes
    // Normalize to 0-1 range with logarithmic scaling for better distribution
    const logSize = Math.log(size + 1);
    const maxLogSize = Math.log(1501); // Max packet size + 1
    
    return Math.min(1, logSize / maxLogSize);
  }

  /**
   * Categorize ports into meaningful groups
   */
  private categorizePort(port: number): number {
    if (port === 0) return 0;
    if (port < 1024) return 0.25; // Well-known ports
    if (port < 49152) return 0.5; // Registered ports
    return 0.75; // Dynamic/private ports
  }

  // The extractFlagFeatures function now only takes string array, removed numeric version

  /**
   * Calculate Shannon entropy of the payload (helps identify encrypted/compressed data)
   */
  private calculatePayloadEntropy(payload: Uint8Array): number {
    if (payload.length === 0) return 0;
    
    const frequencies: Record<number, number> = {};
    payload.forEach(byte => {
      frequencies[byte] = (frequencies[byte] || 0) + 1;
    });
    
    let entropy = 0;
    for (const byte in frequencies) {
      const p = frequencies[byte] / payload.length;
      entropy -= p * Math.log2(p);
    }
    
    // Normalize to 0-1
    return entropy / 8; // Max entropy for byte values is 8 bits
  }

  /**
   * Calculate entropy of IP address (useful for detecting scanning)
   */
  private calculateIpEntropy(ip: string): number {
    if (!ip || ip === '0.0.0.0') return 0;
    
    // Calculate entropy of the least significant octet
    // (Simplified - a full implementation would analyze all octets)
    const octets = ip.split('.');
    const lastOctet = parseInt(octets[3], 10);
    
    // Normalize to 0-1 range
    return lastOctet / 255;
  }

  /**
   * Detect if traffic is between internal addresses
   */
  private isIntranetTraffic(srcIp: string, dstIp: string): number {
    // Check if both IPs are in private ranges
    const isPrivate = (ip: string): boolean => {
      return ip.startsWith('10.') || 
             ip.startsWith('192.168.') || 
             (ip.startsWith('172.') && parseInt(ip.split('.')[1], 10) >= 16 && parseInt(ip.split('.')[1], 10) <= 31);
    };
    
    return (isPrivate(srcIp) && isPrivate(dstIp)) ? 1 : 0;
  }

  /**
   * Extract additional header features
   */
  private extractHeaderFeatures(packet: NetworkPacket): number[] {
    // In a real implementation, we'd extract more sophisticated header features
    // Simplified for this example
    return [
      packet.size > 0 ? 1 : 0,
      packet.protocol !== '0' ? 1 : 0,
      packet.sourcePort !== 0 ? 1 : 0,
      packet.destinationPort !== 0 ? 1 : 0,
    ];
  }

  /**
   * Calculate time since last packet (useful for timing analysis)
   */
  private calculateInterPacketTime(packet: NetworkPacket): number {
    const lastPacket = this.packetBuffer[this.packetBuffer.length - 1];
    if (!lastPacket) return 0;
    
    const timeDiff = packet.timestamp - lastPacket.timestamp;
    // Normalize to 0-1 with 1 second as maximum
    return Math.min(1, timeDiff / 1000);
  }

  /**
   * Calculate ratios of packet types (useful for detecting certain attacks)
   */
  private calculatePacketRatio(): number {
    if (this.stats.totalPackets < 10) return 0.5;
    
    const tcpCount = this.stats.protocolDistribution['TCP'] || 0;
    const udpCount = this.stats.protocolDistribution['UDP'] || 0;
    const icmpCount = this.stats.protocolDistribution['ICMP'] || 0;
    
    // TCP to UDP ratio normalized to 0-1
    if (tcpCount + udpCount === 0) return 0.5;
    return tcpCount / (tcpCount + udpCount);
  }

  /**
   * Get a string key for a protocol number
   */
  private getProtocolKey(protocol: string): string {
    // Protocol is already a string in our mock implementation
    return protocol || 'OTHER';
  }

  /**
   * Initialize packet statistics
   */
  private initializeStatistics(): PacketStatistics {
    return {
      totalPackets: 0,
      totalBytes: 0,
      packetsPerSecond: 0,
      bytesPerSecond: 0,
      protocolDistribution: {},
      portDistribution: {}
    };
  }

  /**
   * Simulate network traffic for testing and demonstration
   */
  private simulateNetworkTraffic(): void {
    if (!this.isCapturing) return;
    
    // Simulate normal traffic patterns
    const normalInterval = setInterval(() => {
      if (!this.isCapturing) {
        clearInterval(normalInterval);
        return;
      }
      
      this.generateSimulatedPacket('normal');
    }, 100); // ~10 packets per second
    
    // Simulate occasional suspicious traffic
    const suspiciousInterval = setInterval(() => {
      if (!this.isCapturing) {
        clearInterval(suspiciousInterval);
        return;
      }
      
      if (Math.random() < 0.1) { // 10% chance
        this.generateSimulatedPacket('suspicious');
      }
    }, 500);
    
    // Simulate rare anomalous traffic
    const anomalousInterval = setInterval(() => {
      if (!this.isCapturing) {
        clearInterval(anomalousInterval);
        return;
      }
      
      if (Math.random() < 0.05) { // 5% chance
        this.generateSimulatedPacket('anomalous');
      }
    }, 2000);
    
    // Simulate occasional DDoS-like bursts
    const burstInterval = setInterval(() => {
      if (!this.isCapturing) {
        clearInterval(burstInterval);
        return;
      }
      
      if (Math.random() < 0.02) { // 2% chance
        this.simulateTrafficBurst();
      }
    }, 10000);
  }

  /**
   * Extract binary features from TCP flags
   */
  private extractFlagFeatures(flags: string[]): number[] {
    if (flags.length === 0) return [0, 0, 0, 0, 0, 0];
    
    // Extract individual TCP flags (FIN, SYN, RST, PSH, ACK, URG)
    const allFlags = ['FIN', 'SYN', 'RST', 'PSH', 'ACK', 'URG'];
    return allFlags.map(flag => flags.includes(flag) ? 1 : 0);
  }

  /**
   * Simulate network traffic for testing and demonstration
   */
  /**
   * Generate a simulated network packet
   */

  /**
   * Generate a simulated network packet
   */
  private generateSimulatedPacket(type: 'normal' | 'suspicious' | 'anomalous'): NetworkPacket {
  const timestamp = Date.now();
  const protocols = ['ICMP', 'TCP', 'UDP'];
  
  // Base packet
  const packet: NetworkPacket = {
    id: nanoid(),
    timestamp,
    sourceIP: this.generateRandomIp(),
    destinationIP: this.generateRandomIp(),
    protocol: protocols[Math.floor(Math.random() * protocols.length)],
    size: this.generateRandomPacketSize(),
    sourcePort: this.generateRandomPort(),
    destinationPort: this.generateRandomPort(),
    flags: this.generateRandomFlags(),
    ttl: 64,
    payload: new Uint8Array(Math.floor(Math.random() * 100)),
    networkInterface: 'eth0',
    direction: Math.random() > 0.5 ? 'inbound' : 'outbound'
  };
  
  // Modify packet based on type
  switch (type) {
    case 'normal':
      // Normal packets - common ports, normal size
      packet.destinationPort = this.getCommonPort();
      packet.flags = ['ACK'];
      break;
    
    case 'suspicious':
      // Suspicious packets - less common ports, unusual patterns
      packet.destinationPort = Math.floor(Math.random() * 10000) + 50000; // High ports
      packet.flags = ['SYN'];
      break;
    
    case 'anomalous':
      // Anomalous packets - unusual characteristics
      packet.size = 1500; // Max size
      packet.protocol = 'OTHER';
      packet.flags = ['FIN', 'SYN', 'RST', 'PSH', 'ACK', 'URG'];
      packet.destinationPort = 0; // Unusual port value
      break;
  }
  
  // Handle the packet
  this.handlePacket(packet);
  
  return packet;
}

  /**
   * Simulate a burst of traffic (like DDoS)
   */
  private simulateTrafficBurst(): void {
  // Target IP and port for the burst
  const targetIp = this.generateRandomIp();
  const targetPort = this.getCommonPort();
  
  // Generate a burst of packets to the same destination
  const burstSize = Math.floor(Math.random() * 30) + 20; // 20-50 packets
  
  for (let i = 0; i < burstSize; i++) {
    setTimeout(() => {
      if (!this.isCapturing) return;
      
      const packet: NetworkPacket = {
        id: nanoid(),
        timestamp: Date.now(),
        sourceIP: this.generateRandomIp(),
        destinationIP: targetIp,
        protocol: 'TCP',
        size: 60, // Small packet size
        sourcePort: Math.floor(Math.random() * 65535),
        destinationPort: targetPort,
        flags: ['SYN'],
        ttl: 64,
        payload: new Uint8Array(0),
        networkInterface: 'eth0',
        direction: 'inbound'
      };
      
      this.handlePacket(packet);
    }, i * 10); // 10ms between packets in the burst
  }
}

  /**
   * Generate a random IP address
   */
  private generateRandomIp(): string {
  return `${Math.floor(Math.random() * 223) + 1}.${Math.floor(Math.random() * 256)}.${Math.floor(Math.random() * 256)}.${Math.floor(Math.random() * 256)}`;
}

  /**
   * Generate a random packet size
   */
  private generateRandomPacketSize(): number {
  return Math.floor(Math.random() * 1460) + 40; // 40-1500 bytes
}

  /**
   * Generate random flags
   */
  private generateRandomFlags(): string[] {
    const allFlags = ['FIN', 'SYN', 'RST', 'PSH', 'ACK', 'URG'];
    const numFlags = Math.floor(Math.random() * 3) + 1; // 1-3 random flags
    const selectedFlags = [];
    
    for (let i = 0; i < numFlags; i++) {
      const randomIndex = Math.floor(Math.random() * allFlags.length);
      selectedFlags.push(allFlags[randomIndex]);
    }
    
    return selectedFlags;
  }

  /**
   * Get a common port number for simulated traffic
   */
  private getCommonPort(): number {
    const commonPorts = [21, 22, 23, 25, 53, 80, 110, 143, 443, 465, 587, 993, 995, 3306, 3389, 5432, 8080];
    return commonPorts[Math.floor(Math.random() * commonPorts.length)];
  }
  
  /**
   * Generate a random port number for simulated traffic
   */
  private generateRandomPort(): number {
    return Math.floor(Math.random() * 65535) + 1;
  }
}
