
import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { DeviceData, CommunicationLog } from './DeviceDataGenerator';

interface DeviceConnectionGraphProps {
  devices: DeviceData[];
  communications: CommunicationLog[];
  onDeviceSelect: (deviceId: string | null) => void;
  selectedDevice: string | null;
}

export function DeviceConnectionGraph({ 
  devices, 
  communications, 
  onDeviceSelect,
  selectedDevice 
}: DeviceConnectionGraphProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!svgRef.current || !containerRef.current || devices.length === 0) return;

    // Clear previous content
    d3.select(svgRef.current).selectAll("*").remove();

    const svg = d3.select(svgRef.current);
    const containerRect = containerRef.current.getBoundingClientRect();
    const width = containerRect.width;
    const height = containerRect.height;
    
    // Set proper dimensions to prevent overflow
    svg.attr("viewBox", `0 0 ${width} ${height}`)
       .attr("width", width)
       .attr("height", height)
       .style("max-width", "100%")
       .style("max-height", "100%");

    // Create connections data from communications
    const connections = communications.slice(-20).map(comm => ({
      source: devices.find(d => d.id === comm.sourceDevice),
      target: devices.find(d => d.id === comm.destinationDevice),
      protocol: comm.protocol,
      status: comm.status,
      anomalyScore: comm.anomalyScore
    })).filter(conn => conn.source && conn.target);

    // Color schemes
    const deviceColors = {
      'Server': '#3B82F6',
      'Workstation': '#10B981',
      'Router': '#F59E0B',
      'IoT Device': '#8B5CF6',
      'Mobile': '#EF4444',
      'Printer': '#6B7280',
      'Database': '#EC4899'
    };

    const statusColors = {
      'normal': '#22C55E',
      'suspicious': '#F59E0B',
      'anomaly': '#EF4444'
    };

    // Draw connections with proper bounds
    const connectionGroup = svg.append("g").attr("class", "connections");
    
    connections.forEach(conn => {
      if (!conn.source || !conn.target) return;
      
      // Ensure positions are within bounds
      const sourceX = Math.max(30, Math.min(width - 30, conn.source.position.x * (width / 800)));
      const sourceY = Math.max(30, Math.min(height - 30, conn.source.position.y * (height / 400)));
      const targetX = Math.max(30, Math.min(width - 30, conn.target.position.x * (width / 800)));
      const targetY = Math.max(30, Math.min(height - 30, conn.target.position.y * (height / 400)));
      
      const line = connectionGroup.append("line")
        .attr("x1", sourceX)
        .attr("y1", sourceY)
        .attr("x2", targetX)
        .attr("y2", targetY)
        .attr("stroke", statusColors[conn.status])
        .attr("stroke-width", 2)
        .attr("opacity", 0.6)
        .attr("stroke-dasharray", conn.status === 'anomaly' ? "5,5" : "none");

      // Add protocol label with bounds checking
      const midX = (sourceX + targetX) / 2;
      const midY = (sourceY + targetY) / 2;
      
      if (midX > 0 && midX < width && midY > 0 && midY < height) {
        connectionGroup.append("text")
          .attr("x", midX)
          .attr("y", midY)
          .attr("text-anchor", "middle")
          .attr("dy", "-5")
          .attr("font-size", "10px")
          .attr("fill", statusColors[conn.status])
          .text(conn.protocol);
      }
    });

    // Draw devices with proper positioning
    const deviceGroup = svg.append("g").attr("class", "devices");
    
    devices.forEach(device => {
      // Scale and constrain device positions
      const deviceX = Math.max(30, Math.min(width - 30, device.position.x * (width / 800)));
      const deviceY = Math.max(30, Math.min(height - 30, device.position.y * (height / 400)));
      
      const group = deviceGroup.append("g")
        .attr("transform", `translate(${deviceX}, ${deviceY})`)
        .attr("cursor", "pointer")
        .on("click", () => {
          onDeviceSelect(selectedDevice === device.id ? null : device.id);
        });

      // Device circle
      const circle = group.append("circle")
        .attr("r", 20)
        .attr("fill", deviceColors[device.deviceType as keyof typeof deviceColors] || '#6B7280')
        .attr("stroke", statusColors[device.status])
        .attr("stroke-width", selectedDevice === device.id ? 4 : 2)
        .attr("opacity", 0.8);

      // Pulsing animation for anomalies
      if (device.status === 'anomaly') {
        circle.append("animate")
          .attr("attributeName", "r")
          .attr("values", "20;25;20")
          .attr("dur", "2s")
          .attr("repeatCount", "indefinite");
      }

      // Device type icon (using first letter)
      group.append("text")
        .attr("text-anchor", "middle")
        .attr("dy", "0.3em")
        .attr("font-size", "12px")
        .attr("font-weight", "bold")
        .attr("fill", "white")
        .text(device.deviceType.charAt(0));

      // Device ID label with proper positioning
      group.append("text")
        .attr("text-anchor", "middle")
        .attr("dy", "35")
        .attr("font-size", "10px")
        .attr("fill", "#374151")
        .text(device.id);

      // Connection count badge
      if (device.activeConnections > 0) {
        const badge = group.append("g")
          .attr("transform", "translate(15, -15)");
        
        badge.append("circle")
          .attr("r", 8)
          .attr("fill", "#EF4444")
          .attr("stroke", "white")
          .attr("stroke-width", 2);
        
        badge.append("text")
          .attr("text-anchor", "middle")
          .attr("dy", "0.3em")
          .attr("font-size", "8px")
          .attr("font-weight", "bold")
          .attr("fill", "white")
          .text(device.activeConnections.toString());
      }
    });

    // Add legend with proper positioning
    const legend = svg.append("g")
      .attr("class", "legend")
      .attr("transform", "translate(20, 20)");

    const legendData = [
      { label: "Normal", color: statusColors.normal },
      { label: "Suspicious", color: statusColors.suspicious },
      { label: "Anomaly", color: statusColors.anomaly }
    ];

    legendData.forEach((item, i) => {
      const legendItem = legend.append("g")
        .attr("transform", `translate(0, ${i * 20})`);
      
      legendItem.append("circle")
        .attr("r", 6)
        .attr("fill", item.color);
      
      legendItem.append("text")
        .attr("x", 15)
        .attr("dy", "0.3em")
        .attr("font-size", "12px")
        .attr("fill", "#374151")
        .text(item.label);
    });

  }, [devices, communications, selectedDevice, onDeviceSelect]);

  return (
    <div ref={containerRef} className="w-full h-full bg-gray-50 rounded-lg border overflow-hidden">
      <svg ref={svgRef} className="w-full h-full block" />
    </div>
  );
}
