
import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Download, Monitor, Network, AlertTriangle } from "lucide-react";
import { DeviceConnectionGraph } from './DeviceConnectionGraph';
import { generateDeviceData, DeviceData, CommunicationLog } from './DeviceDataGenerator';
import { exportDeviceDataToCSV } from './DeviceDataExporter';

interface DeviceMonitoringProps {
  isActive: boolean;
  anomalyScore: number;
  activeThreats: number;
}

export function DeviceMonitoringVisualization({ 
  isActive, 
  anomalyScore, 
  activeThreats 
}: DeviceMonitoringProps) {
  const [devices, setDevices] = useState<DeviceData[]>([]);
  const [communicationLogs, setCommunicationLogs] = useState<CommunicationLog[]>([]);
  const [selectedDevice, setSelectedDevice] = useState<string | null>(null);

  // Generate and update device data
  useEffect(() => {
    const updateDeviceData = () => {
      const { devices: newDevices, communications } = generateDeviceData(isActive, anomalyScore, activeThreats);
      setDevices(newDevices);
      setCommunicationLogs(communications);
    };

    // Initial data
    updateDeviceData();

    // Update data every 2 seconds when analysis is active
    let interval: NodeJS.Timeout | null = null;
    if (isActive) {
      interval = setInterval(updateDeviceData, 2000);
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isActive, anomalyScore, activeThreats]);

  const handleExportCSV = () => {
    exportDeviceDataToCSV(devices, communicationLogs);
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'normal':
        return <Badge variant="secondary" className="bg-green-100 text-green-800">Normal</Badge>;
      case 'suspicious':
        return <Badge variant="outline" className="bg-yellow-100 text-yellow-800">Suspicious</Badge>;
      case 'anomaly':
        return <Badge variant="destructive">Anomaly</Badge>;
      default:
        return <Badge variant="secondary">{status}</Badge>;
    }
  };

  const getProtocolColor = (protocol: string) => {
    const colors = {
      'TCP': 'bg-blue-500',
      'UDP': 'bg-green-500',
      'HTTP': 'bg-purple-500',
      'HTTPS': 'bg-indigo-500',
      'FTP': 'bg-orange-500',
      'SSH': 'bg-red-500',
      'DNS': 'bg-cyan-500'
    };
    return colors[protocol as keyof typeof colors] || 'bg-gray-500';
  };

  return (
    <section className="rounded-lg neumorph overflow-hidden">
      <div className="bg-background p-6">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <Monitor className="h-5 w-5 text-kenshi-blue" />
            Device Communication Monitor
          </h2>
          <Button onClick={handleExportCSV} variant="outline" className="flex items-center gap-2">
            <Download className="h-4 w-4" />
            Export CSV
          </Button>
        </div>

        <Tabs defaultValue="graph" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="graph">Network Graph</TabsTrigger>
            <TabsTrigger value="devices">Device List</TabsTrigger>
            <TabsTrigger value="communications">Communication Logs</TabsTrigger>
          </TabsList>

          <TabsContent value="graph" className="mt-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Network className="h-4 w-4" />
                  Device Network Topology
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-[400px]">
                  <DeviceConnectionGraph 
                    devices={devices}
                    communications={communicationLogs}
                    onDeviceSelect={setSelectedDevice}
                    selectedDevice={selectedDevice}
                  />
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="devices" className="mt-4">
            <Card>
              <CardHeader>
                <CardTitle>Active Devices ({devices.length})</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="max-h-[400px] overflow-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Device ID</TableHead>
                        <TableHead>IP Address</TableHead>
                        <TableHead>Type</TableHead>
                        <TableHead>Status</TableHead>
                        <TableHead>Connections</TableHead>
                        <TableHead>Data Transfer</TableHead>
                        <TableHead>Last Activity</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {devices.map((device) => (
                        <TableRow 
                          key={device.id}
                          className={`cursor-pointer hover:bg-muted/50 ${selectedDevice === device.id ? 'bg-muted' : ''}`}
                          onClick={() => setSelectedDevice(selectedDevice === device.id ? null : device.id)}
                        >
                          <TableCell className="font-mono text-sm">{device.id}</TableCell>
                          <TableCell className="font-mono text-sm">{device.ipAddress}</TableCell>
                          <TableCell>
                            <Badge variant="outline">{device.deviceType}</Badge>
                          </TableCell>
                          <TableCell>{getStatusBadge(device.status)}</TableCell>
                          <TableCell>{device.activeConnections}</TableCell>
                          <TableCell>{device.dataTransferred}</TableCell>
                          <TableCell className="text-sm text-muted-foreground">
                            {new Date(device.lastActivity).toLocaleTimeString()}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="communications" className="mt-4">
            <Card>
              <CardHeader>
                <CardTitle>Communication Logs ({communicationLogs.length})</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="max-h-[400px] overflow-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Timestamp</TableHead>
                        <TableHead>Source</TableHead>
                        <TableHead>Destination</TableHead>
                        <TableHead>Protocol</TableHead>
                        <TableHead>File Size</TableHead>
                        <TableHead>Status</TableHead>
                        <TableHead>Anomaly Score</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {communicationLogs.slice(-50).reverse().map((log) => (
                        <TableRow key={log.id}>
                          <TableCell className="text-sm font-mono">
                            {new Date(log.timestamp).toLocaleTimeString()}
                          </TableCell>
                          <TableCell className="font-mono text-sm">{log.sourceDevice}</TableCell>
                          <TableCell className="font-mono text-sm">{log.destinationDevice}</TableCell>
                          <TableCell>
                            <Badge 
                              variant="outline" 
                              className={`${getProtocolColor(log.protocol)} text-white border-none`}
                            >
                              {log.protocol}
                            </Badge>
                          </TableCell>
                          <TableCell>{log.fileSize}</TableCell>
                          <TableCell>{getStatusBadge(log.status)}</TableCell>
                          <TableCell>
                            <div className="flex items-center gap-2">
                              <span className={`text-sm ${log.anomalyScore > 0.5 ? 'text-red-600 font-medium' : 'text-muted-foreground'}`}>
                                {(log.anomalyScore * 100).toFixed(1)}%
                              </span>
                              {log.anomalyScore > 0.7 && (
                                <AlertTriangle className="h-3 w-3 text-red-500" />
                              )}
                            </div>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </section>
  );
}
