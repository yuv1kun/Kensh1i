
export const sampleNetworkData = {
  nodes: [
    { id: "Core-Router", group: 0, size: 18 },
    { id: "Firewall", group: 0, size: 16 },
    { id: "API-Gateway", group: 0, size: 14 },
    { id: "Auth-Server", group: 0 },
    { id: "Web-Server-1", group: 0 },
    { id: "Web-Server-2", group: 0 },
    { id: "Web-Server-3", group: 0 },
    { id: "Database-Primary", group: 0, size: 15 },
    { id: "Database-Replica", group: 0 },
    { id: "Cache-Server", group: 0 },
    { id: "Load-Balancer", group: 0, size: 15 },
    { id: "Monitoring", group: 0 },
    { id: "Cloud-Storage", group: 0 },
    { id: "User-1", group: 3 },
    { id: "User-2", group: 3 },
    { id: "User-3", group: 3 },
    { id: "Admin-1", group: 3, size: 12 },
    { id: "Unknown-IP-1", group: 1, size: 14 },
    { id: "Unknown-IP-2", group: 1 },
    { id: "IoT-Device-1", group: 2 },
    { id: "IoT-Device-2", group: 0 },
    { id: "IoT-Device-3", group: 2 }
  ],
  links: [
    // External users to Load Balancer
    { source: "User-1", target: "Load-Balancer", value: 1, status: "normal" as const },
    { source: "User-2", target: "Load-Balancer", value: 1, status: "normal" as const },
    { source: "User-3", target: "Load-Balancer", value: 2, status: "normal" as const },
    { source: "Admin-1", target: "Firewall", value: 1, status: "normal" as const },
    
    // Load Balancer to Web Servers
    { source: "Load-Balancer", target: "Web-Server-1", value: 2, status: "normal" as const },
    { source: "Load-Balancer", target: "Web-Server-2", value: 2, status: "normal" as const },
    { source: "Load-Balancer", target: "Web-Server-3", value: 1, status: "normal" as const },
    
    // Web Servers to API Gateway
    { source: "Web-Server-1", target: "API-Gateway", value: 3, status: "normal" as const },
    { source: "Web-Server-2", target: "API-Gateway", value: 2, status: "normal" as const },
    { source: "Web-Server-3", target: "API-Gateway", value: 1, status: "normal" as const },
    
    // API Gateway connections
    { source: "API-Gateway", target: "Auth-Server", value: 3, status: "normal" as const },
    { source: "API-Gateway", target: "Database-Primary", value: 4, status: "normal" as const },
    { source: "API-Gateway", target: "Cache-Server", value: 2, status: "normal" as const },
    
    // Database replication
    { source: "Database-Primary", target: "Database-Replica", value: 2, status: "normal" as const },
    
    // Monitoring connections
    { source: "Monitoring", target: "Web-Server-1", value: 1, status: "normal" as const },
    { source: "Monitoring", target: "Web-Server-2", value: 1, status: "normal" as const },
    { source: "Monitoring", target: "Web-Server-3", value: 1, status: "normal" as const },
    { source: "Monitoring", target: "Database-Primary", value: 1, status: "normal" as const },
    
    // Cloud storage connections
    { source: "Web-Server-1", target: "Cloud-Storage", value: 1, status: "normal" as const },
    { source: "Web-Server-2", target: "Cloud-Storage", value: 1, status: "normal" as const },
    
    // IoT devices
    { source: "IoT-Device-1", target: "API-Gateway", value: 1, status: "warning" as const },
    { source: "IoT-Device-2", target: "API-Gateway", value: 1, status: "normal" as const },
    { source: "IoT-Device-3", target: "API-Gateway", value: 1, status: "warning" as const },
    
    // Unknown/suspicious activities
    { source: "Unknown-IP-1", target: "Firewall", value: 4, status: "critical" as const },
    { source: "Unknown-IP-1", target: "Load-Balancer", value: 3, status: "critical" as const },
    { source: "Unknown-IP-2", target: "Web-Server-1", value: 2, status: "warning" as const },
    
    // Core router connections
    { source: "Core-Router", target: "Firewall", value: 5, status: "normal" as const },
    { source: "Firewall", target: "Load-Balancer", value: 4, status: "normal" as const },
    { source: "Core-Router", target: "Monitoring", value: 1, status: "normal" as const }
  ]
};

export function generateRandomNetworkData() {
  // Create a deep copy of the base data
  const baseData = JSON.parse(JSON.stringify(sampleNetworkData));
  
  // Add some randomness to simulate live data
  
  // Randomly modify node statuses
  baseData.nodes.forEach((node: any) => {
    if (Math.random() > 0.8) {
      // 20% chance to change a node's group (status)
      if (node.id.startsWith('Unknown') || node.id.startsWith('IoT')) {
        node.group = Math.random() > 0.6 ? 1 : (Math.random() > 0.5 ? 2 : 0);
      } else if (!node.id.startsWith('User') && !node.id.startsWith('Admin')) {
        // Don't change user/admin groups often
        node.group = Math.random() > 0.9 ? 1 : 0; // 10% chance for servers to show anomalous
      }
    }
  });
  
  // Randomly update link values and statuses
  baseData.links.forEach((link: any) => {
    if (Math.random() > 0.7) {
      // 30% chance to change value
      link.value = Math.max(1, Math.floor(Math.random() * 5));
      
      // Update status based on value and source/target
      if (link.source.includes('Unknown') || link.target.includes('Unknown')) {
        // Links from unknown sources are more likely to be critical
        link.status = Math.random() > 0.4 ? "critical" : "warning";
      } else if (link.value > 3) {
        // High traffic can be suspicious
        link.status = Math.random() > 0.6 ? "warning" : "normal";
      } else {
        // Normal traffic
        link.status = Math.random() > 0.9 ? "warning" : "normal";
      }
    }
  });
  
  // Occasionally add a new random connection
  if (Math.random() > 0.8) {
    const randomSourceIndex = Math.floor(Math.random() * baseData.nodes.length);
    const randomTargetIndex = Math.floor(Math.random() * baseData.nodes.length);
    if (randomSourceIndex !== randomTargetIndex) {
      baseData.links.push({
        source: baseData.nodes[randomSourceIndex].id,
        target: baseData.nodes[randomTargetIndex].id,
        value: Math.ceil(Math.random() * 3),
        status: Math.random() > 0.7 ? "warning" : "normal"
      });
    }
  }
  
  return baseData;
}
