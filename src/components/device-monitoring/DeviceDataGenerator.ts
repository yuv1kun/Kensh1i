
export interface DeviceData {
  id: string;
  ipAddress: string;
  deviceType: string;
  status: 'normal' | 'suspicious' | 'anomaly';
  activeConnections: number;
  dataTransferred: string;
  lastActivity: number;
  position: { x: number; y: number };
}

export interface CommunicationLog {
  id: string;
  timestamp: number;
  sourceDevice: string;
  destinationDevice: string;
  protocol: string;
  fileSize: string;
  status: 'normal' | 'suspicious' | 'anomaly';
  anomalyScore: number;
}

const DEVICE_TYPES = ['Server', 'Workstation', 'Router', 'IoT Device', 'Mobile', 'Printer', 'Database'];
const PROTOCOLS = ['TCP', 'UDP', 'HTTP', 'HTTPS', 'FTP', 'SSH', 'DNS'];

// Generate realistic IP addresses
const generateIPAddress = (): string => {
  const subnet = Math.random() > 0.5 ? '192.168' : '10.0';
  const third = Math.floor(Math.random() * 255);
  const fourth = Math.floor(Math.random() * 254) + 1;
  return `${subnet}.${third}.${fourth}`;
};

// Generate realistic file sizes
const generateFileSize = (): string => {
  const size = Math.random() * 1000;
  if (size < 1) {
    return `${Math.floor(size * 1000)} B`;
  } else if (size < 100) {
    return `${size.toFixed(1)} KB`;
  } else {
    return `${(size / 100).toFixed(1)} MB`;
  }
};

// Generate device status based on anomaly score
const generateDeviceStatus = (anomalyScore: number, activeThreats: number): 'normal' | 'suspicious' | 'anomaly' => {
  const random = Math.random();
  
  if (activeThreats > 5 && random < 0.3) {
    return 'anomaly';
  } else if (anomalyScore > 0.5 && random < 0.4) {
    return 'suspicious';
  } else if (anomalyScore > 0.3 && random < 0.2) {
    return 'suspicious';
  } else {
    return 'normal';
  }
};

let deviceCounter = 1;
let logCounter = 1;
const existingDevices: DeviceData[] = [];

export const generateDeviceData = (
  isActive: boolean, 
  anomalyScore: number, 
  activeThreats: number
): { devices: DeviceData[]; communications: CommunicationLog[] } => {
  
  // Generate devices if we don't have enough
  if (existingDevices.length < 15) {
    for (let i = existingDevices.length; i < 15; i++) {
      const device: DeviceData = {
        id: `DEV-${String(deviceCounter++).padStart(3, '0')}`,
        ipAddress: generateIPAddress(),
        deviceType: DEVICE_TYPES[Math.floor(Math.random() * DEVICE_TYPES.length)],
        status: generateDeviceStatus(anomalyScore, activeThreats),
        activeConnections: Math.floor(Math.random() * 10) + 1,
        dataTransferred: generateFileSize(),
        lastActivity: Date.now() - Math.floor(Math.random() * 300000), // Last 5 minutes
        position: {
          x: Math.random() * 800 + 100,
          y: Math.random() * 400 + 100
        }
      };
      existingDevices.push(device);
    }
  }

  // Update existing devices with new status and activity
  existingDevices.forEach(device => {
    if (isActive && Math.random() > 0.7) {
      device.status = generateDeviceStatus(anomalyScore, activeThreats);
      device.activeConnections = Math.floor(Math.random() * 15) + 1;
      device.dataTransferred = generateFileSize();
      device.lastActivity = Date.now() - Math.floor(Math.random() * 60000); // Last minute
    }
  });

  // Generate communication logs
  const communications: CommunicationLog[] = [];
  const numLogs = isActive ? Math.floor(Math.random() * 20) + 10 : 5;

  for (let i = 0; i < numLogs; i++) {
    const sourceDevice = existingDevices[Math.floor(Math.random() * existingDevices.length)];
    let destinationDevice = existingDevices[Math.floor(Math.random() * existingDevices.length)];
    
    // Ensure source and destination are different
    while (destinationDevice.id === sourceDevice.id) {
      destinationDevice = existingDevices[Math.floor(Math.random() * existingDevices.length)];
    }

    const protocol = PROTOCOLS[Math.floor(Math.random() * PROTOCOLS.length)];
    const anomalyScoreValue = Math.random() * (anomalyScore + 0.3);
    
    let status: 'normal' | 'suspicious' | 'anomaly' = 'normal';
    if (anomalyScoreValue > 0.7) {
      status = 'anomaly';
    } else if (anomalyScoreValue > 0.4) {
      status = 'suspicious';
    }

    const communication: CommunicationLog = {
      id: `LOG-${String(logCounter++).padStart(4, '0')}`,
      timestamp: Date.now() - Math.floor(Math.random() * 30000), // Last 30 seconds
      sourceDevice: sourceDevice.id,
      destinationDevice: destinationDevice.id,
      protocol,
      fileSize: generateFileSize(),
      status,
      anomalyScore: anomalyScoreValue
    };

    communications.push(communication);
  }

  return { devices: [...existingDevices], communications };
};
