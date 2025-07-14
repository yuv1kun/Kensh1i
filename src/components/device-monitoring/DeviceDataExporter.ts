
import { DeviceData, CommunicationLog } from './DeviceDataGenerator';

export const exportDeviceDataToCSV = (
  devices: DeviceData[], 
  communications: CommunicationLog[]
): void => {
  // Prepare device data for CSV
  const deviceRows = devices.map(device => ({
    device_id: device.id,
    ip_address: device.ipAddress,
    device_type: device.deviceType,
    status: device.status,
    active_connections: device.activeConnections,
    data_transferred: device.dataTransferred,
    last_activity: new Date(device.lastActivity).toISOString(),
    position_x: device.position.x,
    position_y: device.position.y
  }));

  // Prepare communication logs for CSV
  const communicationRows = communications.map(comm => ({
    log_id: comm.id,
    timestamp: new Date(comm.timestamp).toISOString(),
    source_device: comm.sourceDevice,
    destination_device: comm.destinationDevice,
    protocol: comm.protocol,
    file_size: comm.fileSize,
    status: comm.status,
    anomaly_score: comm.anomalyScore.toFixed(3),
    anomaly_percentage: `${(comm.anomalyScore * 100).toFixed(1)}%`
  }));

  // Convert to CSV format
  const deviceCSV = convertToCSV(deviceRows);
  const communicationCSV = convertToCSV(communicationRows);

  // Create and download files
  downloadCSV(deviceCSV, 'device_monitoring_devices.csv');
  downloadCSV(communicationCSV, 'device_monitoring_communications.csv');

  // Also create a summary report
  const summary = generateSummaryReport(devices, communications);
  const summaryCSV = convertToCSV([summary]);
  downloadCSV(summaryCSV, 'device_monitoring_summary.csv');
};

const convertToCSV = (data: any[]): string => {
  if (data.length === 0) return '';
  
  const headers = Object.keys(data[0]);
  const csvRows = [
    headers.join(','),
    ...data.map(row => 
      headers.map(header => {
        const value = row[header];
        // Escape commas and quotes in values
        if (typeof value === 'string' && (value.includes(',') || value.includes('"'))) {
          return `"${value.replace(/"/g, '""')}"`;
        }
        return value;
      }).join(',')
    )
  ];
  
  return csvRows.join('\n');
};

const downloadCSV = (csvContent: string, fileName: string): void => {
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const link = document.createElement('a');
  
  if (link.download !== undefined) {
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', fileName);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }
};

const generateSummaryReport = (devices: DeviceData[], communications: CommunicationLog[]) => {
  const totalDevices = devices.length;
  const normalDevices = devices.filter(d => d.status === 'normal').length;
  const suspiciousDevices = devices.filter(d => d.status === 'suspicious').length;
  const anomalyDevices = devices.filter(d => d.status === 'anomaly').length;
  
  const totalCommunications = communications.length;
  const normalCommunications = communications.filter(c => c.status === 'normal').length;
  const suspiciousCommunications = communications.filter(c => c.status === 'suspicious').length;
  const anomalyCommunications = communications.filter(c => c.status === 'anomaly').length;
  
  const avgAnomalyScore = communications.reduce((sum, c) => sum + c.anomalyScore, 0) / totalCommunications;
  
  const protocolDistribution = communications.reduce((acc, c) => {
    acc[c.protocol] = (acc[c.protocol] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  const deviceTypeDistribution = devices.reduce((acc, d) => {
    acc[d.deviceType] = (acc[d.deviceType] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  return {
    report_timestamp: new Date().toISOString(),
    total_devices: totalDevices,
    normal_devices: normalDevices,
    suspicious_devices: suspiciousDevices,
    anomaly_devices: anomalyDevices,
    device_health_percentage: `${((normalDevices / totalDevices) * 100).toFixed(1)}%`,
    total_communications: totalCommunications,
    normal_communications: normalCommunications,
    suspicious_communications: suspiciousCommunications,
    anomaly_communications: anomalyCommunications,
    communication_health_percentage: `${((normalCommunications / totalCommunications) * 100).toFixed(1)}%`,
    average_anomaly_score: avgAnomalyScore.toFixed(3),
    most_used_protocol: Object.keys(protocolDistribution).reduce((a, b) => 
      protocolDistribution[a] > protocolDistribution[b] ? a : b
    ),
    most_common_device_type: Object.keys(deviceTypeDistribution).reduce((a, b) => 
      deviceTypeDistribution[a] > deviceTypeDistribution[b] ? a : b
    ),
    protocol_distribution: JSON.stringify(protocolDistribution),
    device_type_distribution: JSON.stringify(deviceTypeDistribution)
  };
};
