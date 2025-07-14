
import { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';

interface Node {
  id: string;
  group: number;
  size?: number;
  icon?: string;
}

interface Link {
  source: string | { id: string };
  target: string | { id: string };
  value: number;
  status?: "normal" | "warning" | "critical";
}

interface NetworkGraphProps {
  width?: number;
  height?: number;
  data: {
    nodes: Node[];
    links: Link[];
  };
}

// Colors based on node status/group
const NODE_COLORS = [
  '#1A73E8', // Normal traffic
  '#EA4335', // Anomalous traffic
  '#FBBC05', // Warning
  '#34A853', // Safe
];

// Link colors based on status
const LINK_COLORS = {
  normal: "#B8B8B8",
  warning: "#FBBC05",
  critical: "#EA4335",
};

export function NetworkGraph({ width = 600, height = 400, data }: NetworkGraphProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);

  useEffect(() => {
    if (!svgRef.current || !data.nodes.length) return;

    // Clear previous graph
    d3.select(svgRef.current).selectAll("*").remove();

    const svg = d3.select(svgRef.current)
      .attr("viewBox", `0 0 ${width} ${height}`)
      .attr("width", "100%")
      .attr("height", "100%");
    
    // Add definitions for arrow markers and glows
    const defs = svg.append("defs");
    
    // Add arrow marker definitions
    defs.append("marker")
      .attr("id", "arrow")
      .attr("viewBox", "0 -5 10 10")
      .attr("refX", 20)
      .attr("refY", 0)
      .attr("markerWidth", 6)
      .attr("markerHeight", 6)
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M0,-5L10,0L0,5")
      .attr("fill", "#999");
      
    // Add glow effect
    const glow = defs.append("filter")
      .attr("id", "glow")
      .attr("x", "-40%")
      .attr("y", "-40%")
      .attr("width", "180%")
      .attr("height", "180%");
    
    glow.append("feGaussianBlur")
      .attr("stdDeviation", "2.5")
      .attr("result", "coloredBlur");
    
    const glowMerge = glow.append("feMerge");
    glowMerge.append("feMergeNode").attr("in", "coloredBlur");
    glowMerge.append("feMergeNode").attr("in", "SourceGraphic");

    // Create a group for the graph
    const graphGroup = svg.append("g");

    // Add a subtle grid background
    const gridSize = 30;
    const numHorizontalLines = Math.ceil(height / gridSize);
    const numVerticalLines = Math.ceil(width / gridSize);
    
    const gridGroup = graphGroup.append("g").attr("class", "grid");
    
    // Add horizontal grid lines
    for (let i = 0; i <= numHorizontalLines; i++) {
      gridGroup.append("line")
        .attr("x1", 0)
        .attr("y1", i * gridSize)
        .attr("x2", width)
        .attr("y2", i * gridSize)
        .attr("stroke", "#f0f0f0")
        .attr("stroke-width", 0.5);
    }
    
    // Add vertical grid lines
    for (let i = 0; i <= numVerticalLines; i++) {
      gridGroup.append("line")
        .attr("x1", i * gridSize)
        .attr("y1", 0)
        .attr("x2", i * gridSize)
        .attr("y2", height)
        .attr("stroke", "#f0f0f0")
        .attr("stroke-width", 0.5);
    }

    // Prepare the data by adding size to nodes if not set
    const nodes = data.nodes.map(node => ({
      ...node,
      size: node.size || (node.group === 0 ? 12 : node.group === 1 ? 15 : 10)
    }));
    
    // Add status to links if not set and ensure status is a valid type
    const links = data.links.map(link => {
      const validStatus = (link.status === "normal" || 
                          link.status === "warning" || 
                          link.status === "critical") 
                          ? link.status 
                          : (link.value > 3 ? "critical" : link.value > 1 ? "warning" : "normal");
      
      return {
        ...link,
        status: validStatus as "normal" | "warning" | "critical"
      };
    });
    
    // Create force simulation
    const simulation = d3.forceSimulation<any>(nodes)
      .force("link", d3.forceLink<any, any>(links).id(d => d.id).distance(80))
      .force("charge", d3.forceManyBody().strength(-150))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide().radius(d => (d.size || 10) * 1.5));

    // Draw links with animations
    const link = graphGroup.append("g")
      .attr("class", "links")
      .selectAll("line")
      .data(links)
      .join("line")
      .attr("stroke-width", d => Math.sqrt(d.value) * 1.5)
      .attr("stroke", d => LINK_COLORS[d.status || "normal"])
      .attr("stroke-opacity", 0.6)
      .attr("marker-end", "url(#arrow)")
      .attr("class", d => `link ${d.status}`);
      
    // Add traffic animation to links
    const packets = graphGroup.append("g")
      .attr("class", "packets")
      .selectAll("circle")
      .data(links.filter(l => l.value > 0))
      .join("circle")
      .attr("r", d => Math.min(Math.max(d.value * 0.8, 2), 4))
      .attr("fill", d => d.status === "critical" ? "#EA4335" : d.status === "warning" ? "#FBBC05" : "#1A73E8")
      .attr("class", "packet")
      .attr("opacity", 0.7);
    
    // Function to animate packets along links
    function animatePackets() {
      packets.each(function(d: any) {
        const packet = d3.select(this);
        
        function animatePacket() {
          packet
            .attr("cx", d.source.x)
            .attr("cy", d.source.y)
            .transition()
            .duration(2000 / d.value) // Faster for higher value links
            .attr("cx", d.target.x)
            .attr("cy", d.target.y)
            .transition()
            .duration(100)
            .attr("cx", d.source.x)
            .attr("cy", d.source.y)
            .on("end", animatePacket);
        }
        
        // Stagger the animation start
        setTimeout(animatePacket, Math.random() * 2000);
      });
    }
    
    // Draw nodes with icon support
    const nodeGroup = graphGroup.append("g")
      .attr("class", "nodes")
      .selectAll("g")
      .data(nodes)
      .join("g")
      .attr("class", "node-group")
      .call(drag(simulation) as any);
      
    // Node circles
    const node = nodeGroup.append("circle")
      .attr("r", d => d.size || 10)
      .attr("fill", d => NODE_COLORS[d.group % NODE_COLORS.length])
      .attr("stroke", "#fff")
      .attr("stroke-width", 1.5)
      .on("mouseover", (event, d) => {
        setHoveredNode(d.id);
        d3.select(event.currentTarget).attr("filter", "url(#glow)");
      })
      .on("mouseout", (event) => {
        setHoveredNode(null);
        d3.select(event.currentTarget).attr("filter", null);
      });
      
    // Node icons or labels
    nodeGroup.append("text")
      .attr("text-anchor", "middle")
      .attr("dy", ".3em")
      .attr("fill", "#ffffff")
      .attr("font-size", d => (d.size || 10) * 0.8)
      .text(d => {
        if (d.icon) return d.icon;
        // Use first letter of ID as fallback icon
        return d.id.charAt(0);
      });

    // Add node labels
    nodeGroup.append("text")
      .attr("dy", d => (d.size || 10) + 12)
      .attr("text-anchor", "middle")
      .attr("class", "node-label")
      .text(d => d.id)
      .attr("fill", "#505050")
      .attr("font-size", "9px")
      .attr("font-family", "monospace");
      
    // Add tooltips
    nodeGroup.append("title")
      .text(d => `${d.id}\nGroup: ${d.group}`);
      
    // Add hover effects for connected nodes
    node.on("mouseover", (event, d) => {
      // Highlight connected links and nodes
      link
        .attr("stroke-opacity", l => {
          // Fix: Handle when source and target are either string or object with id property
          const sourceId = typeof l.source === 'string' ? l.source : l.source.id;
          const targetId = typeof l.target === 'string' ? l.target : l.target.id;
          return (sourceId === d.id || targetId === d.id) ? 1 : 0.2;
        });
        
      node
        .attr("opacity", n => {
          if (n.id === d.id) return 1;
          
          // Fix: Check connections by comparing IDs correctly
          if (links.some(l => {
            const sourceId = typeof l.source === 'string' ? l.source : l.source.id;
            const targetId = typeof l.target === 'string' ? l.target : l.target.id;
            return (sourceId === d.id && targetId === n.id) || (targetId === d.id && sourceId === n.id);
          })) {
            return 1;
          }
          return 0.3;
        });
    }).on("mouseout", () => {
      link.attr("stroke-opacity", 0.6);
      node.attr("opacity", 1);
    });

    // Update positions on each "tick"
    simulation.on("tick", () => {
      link
        .attr("x1", d => {
          // Fix: Handle both string and object source
          const sourceId = typeof d.source === 'string' ? d.source : d.source.id;
          // Find the corresponding node by id
          const sourceNode = nodes.find(n => n.id === sourceId);
          return sourceNode ? (sourceNode as any).x : 0;
        })
        .attr("y1", d => {
          const sourceId = typeof d.source === 'string' ? d.source : d.source.id;
          const sourceNode = nodes.find(n => n.id === sourceId);
          return sourceNode ? (sourceNode as any).y : 0;
        })
        .attr("x2", d => {
          const targetId = typeof d.target === 'string' ? d.target : d.target.id;
          const targetNode = nodes.find(n => n.id === targetId);
          return targetNode ? (targetNode as any).x : 0;
        })
        .attr("y2", d => {
          const targetId = typeof d.target === 'string' ? d.target : d.target.id;
          const targetNode = nodes.find(n => n.id === targetId);
          return targetNode ? (targetNode as any).y : 0;
        });

      nodeGroup
        .attr("transform", d => `translate(${d.x}, ${d.y})`);
    });
    
    // Start packet animation after simulation stabilizes
    setTimeout(animatePackets, 1000);

    // Implement drag behavior
    function drag(simulation: d3.Simulation<d3.SimulationNodeDatum, undefined>) {
      function dragstarted(event: any) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        event.subject.fx = event.subject.x;
        event.subject.fy = event.subject.y;
      }
      
      function dragged(event: any) {
        event.subject.fx = event.x;
        event.subject.fy = event.y;
      }
      
      function dragended(event: any) {
        if (!event.active) simulation.alphaTarget(0);
        event.subject.fx = null;
        event.subject.fy = null;
      }
      
      return d3.drag()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended);
    }
    
    return () => {
      simulation.stop();
    };
  }, [data, width, height, hoveredNode]);

  return (
    <div className="w-full h-full overflow-hidden rounded-lg bg-white/5">
      <svg ref={svgRef} className="w-full h-full" />
      {hoveredNode && (
        <div className="absolute top-2 right-2 bg-background/80 backdrop-blur-sm p-3 rounded-md text-xs">
          <p className="font-medium">{hoveredNode}</p>
          <p className="text-muted-foreground">Hover over nodes to see connections</p>
        </div>
      )}
    </div>
  );
}
