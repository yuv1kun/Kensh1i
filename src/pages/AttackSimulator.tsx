import { useState } from "react";
import { Header } from "@/components/header";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { AspectRatio } from "@/components/ui/aspect-ratio";
import { SplineViewer } from "@/components/spline-viewer";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Play, FileHeart, Layers, Book, ChartLine, CircleChevronDown } from "lucide-react";
import { useToast } from "@/components/ui/use-toast";

// Mock data for attack scenarios
const scenarioPresets = [
  {
    id: "ransomware",
    name: "Zero-Day Ransomware",
    techniques: ["T1486", "T1490", "T1489"],
    complexity: "High",
    impact: 90,
    description: "Simulates a sophisticated ransomware attack that encrypts files and disrupts system backups."
  },
  {
    id: "phishing",
    name: "AI-Powered Phishing",
    techniques: ["T1566.002", "T1078", "T1539"],
    complexity: "Medium",
    impact: 70,
    description: "Emulates advanced phishing attacks using AI to generate convincing targeted messages."
  },
  {
    id: "supply-chain",
    name: "Supply Chain Compromise",
    techniques: ["T1195.002", "T1133", "T1505.001"],
    complexity: "Very High",
    impact: 95,
    description: "Simulates attack through compromised software distribution channels."
  },
  {
    id: "lateral",
    name: "Lateral Movement Campaign",
    techniques: ["T1021.001", "T1550.002", "T1098"],
    complexity: "High",
    impact: 85,
    description: "Emulates adversary moving through network after initial compromise to establish persistence."
  },
  {
    id: "data-exfiltration",
    name: "Data Exfiltration",
    techniques: ["T1567.002", "T1048.003", "T1071.001"],
    complexity: "Medium",
    impact: 75,
    description: "Simulates unauthorized data extraction using encrypted channels and DNS tunneling."
  },
  {
    id: "crypto-mining",
    name: "Cryptojacking Operation",
    techniques: ["T1496", "T1053.005", "T1070.004"],
    complexity: "Low",
    impact: 60,
    description: "Emulates attackers deploying crypto-mining malware that consumes system resources."
  }
];

// Mock data for past simulation runs
const simulationHistory = [
  {
    id: "sim-001",
    scenario: "Zero-Day Ransomware",
    timestamp: "2025-05-12T14:30:00",
    duration: "12:45",
    detectionRate: 87,
    falsePositives: 3,
    status: "completed"
  },
  {
    id: "sim-002",
    scenario: "AI-Powered Phishing",
    timestamp: "2025-05-10T09:15:00",
    duration: "08:22",
    detectionRate: 92,
    falsePositives: 5,
    status: "completed"
  },
  {
    id: "sim-003",
    scenario: "Supply Chain Compromise",
    timestamp: "2025-05-05T16:45:00",
    duration: "15:18",
    detectionRate: 79,
    falsePositives: 2,
    status: "completed"
  },
  {
    id: "sim-004",
    scenario: "Data Exfiltration",
    timestamp: "2025-05-02T11:20:00",
    duration: "10:05",
    detectionRate: 85,
    falsePositives: 4,
    status: "completed"
  }
];

// Details for a selected technique
const techniquesData = {
  "T1486": {
    name: "Data Encryption for Impact",
    tactic: "Impact",
    description: "Attackers encrypt data to disrupt business operations and demand ransom payments.",
    mitigations: ["Regular backups", "Least privilege", "Application control"]
  },
  "T1490": {
    name: "Inhibit System Recovery",
    tactic: "Impact",
    description: "Attackers delete or corrupt system recovery mechanisms to prevent data restoration.",
    mitigations: ["Protected backups", "Access controls", "Recovery planning"]
  },
  "T1489": {
    name: "Service Stop",
    tactic: "Impact",
    description: "Attackers stop or disable services to render systems inoperable or limit recovery.",
    mitigations: ["Service monitoring", "Privilege management", "Service hardening"]
  },
  "T1566.002": {
    name: "Spearphishing Link",
    tactic: "Initial Access",
    description: "Attackers send targeted emails with malicious links to gain access.",
    mitigations: ["User training", "Link filtering", "Email scanning"]
  },
  "T1078": {
    name: "Valid Accounts",
    tactic: "Defense Evasion",
    description: "Attackers use legitimate credentials to access systems and blend in with normal activity.",
    mitigations: ["MFA", "Account monitoring", "Privileged account management"]
  },
  "T1539": {
    name: "Steal Web Session Cookie",
    tactic: "Credential Access",
    description: "Attackers steal session cookies to gain unauthorized access to web applications.",
    mitigations: ["Secure cookie flags", "Session timeouts", "HTTPS everywhere"]
  },
  "T1195.002": {
    name: "Compromise Software Supply Chain",
    tactic: "Initial Access",
    description: "Attackers manipulate software prior to receipt by consumer to enable unauthorized access.",
    mitigations: ["Vendor assessment", "Code signing", "Software integrity verification"]
  }
};

// Spline scene URL for attack visualization
const attackVisualizationUrl = "https://my.spline.design/particleaibrain-fRP1lexDTOPNTrZqEBAsVvOA/";

const AttackSimulator = () => {
  const { toast } = useToast();
  const [selectedScenario, setSelectedScenario] = useState<string | null>(null);
  const [selectedTechnique, setSelectedTechnique] = useState<string | null>(null);
  const [isSimulating, setIsSimulating] = useState(false);
  const [simulationProgress, setSimulationProgress] = useState(0);
  const [activeSimulation, setActiveSimulation] = useState<any | null>(null);

  // Handle scenario selection
  const handleScenarioSelect = (scenarioId: string) => {
    if (isSimulating) return;
    
    setSelectedScenario(scenarioId === selectedScenario ? null : scenarioId);
    setSelectedTechnique(null);
  };

  // Handle technique selection
  const handleTechniqueSelect = (technique: string) => {
    setSelectedTechnique(technique === selectedTechnique ? null : technique);
  };

  // Run attack simulation
  const runSimulation = () => {
    if (!selectedScenario) return;
    
    const scenario = scenarioPresets.find(s => s.id === selectedScenario);
    if (!scenario) return;
    
    setIsSimulating(true);
    setSimulationProgress(0);
    
    toast({
      title: "Simulation Started",
      description: `Running "${scenario.name}" attack scenario`,
    });
    
    // Create active simulation object
    const simulation = {
      id: `sim-${Date.now()}`,
      scenario: scenario.name,
      timestamp: new Date().toISOString(),
      startTime: Date.now(),
      techniques: scenario.techniques,
      status: "running" as const,
      events: [],
      detectionRate: 0,
      falsePositives: 0
    };
    
    setActiveSimulation(simulation);
    
    // Simulate progress updates
    const interval = setInterval(() => {
      setSimulationProgress(prev => {
        const newProgress = prev + Math.random() * 10;
        
        // Add mock events at certain progress points
        if (prev < 25 && newProgress >= 25) {
          const updatedSim = { ...simulation };
          updatedSim.events = [...(updatedSim.events || []), {
            time: new Date().toISOString(),
            message: `Initial access achieved using ${scenario.techniques[0]}`,
            type: "alert"
          }];
          setActiveSimulation(updatedSim);
          
          toast({
            title: "Initial Access Detected",
            description: `Kenshi identified ${techniquesData[scenario.techniques[0] as keyof typeof techniquesData]?.name || scenario.techniques[0]}`,
            variant: "destructive",
          });
        }
        
        if (prev < 60 && newProgress >= 60) {
          const updatedSim = { ...simulation };
          updatedSim.events = [...(updatedSim.events || []), {
            time: new Date().toISOString(),
            message: `Lateral movement detected using ${scenario.techniques[1] || "unknown technique"}`,
            type: "alert"
          }];
          setActiveSimulation(updatedSim);
          
          toast({
            title: "Lateral Movement Detected",
            description: "Kenshi is tracking attacker movement through the network",
            variant: "destructive",
          });
        }
        
        // Simulation complete
        if (newProgress >= 100) {
          clearInterval(interval);
          
          // Calculate final simulation stats
          const updatedSim = { 
            ...simulation,
            status: "completed" as const,
            duration: `${Math.floor(Math.random() * 10 + 5)}:${Math.floor(Math.random() * 60).toString().padStart(2, '0')}`,
            detectionRate: Math.floor(Math.random() * 15 + 80),
            falsePositives: Math.floor(Math.random() * 7),
            events: [...(simulation.events || []), {
              time: new Date().toISOString(),
              message: "Simulation completed successfully",
              type: "info"
            }]
          };
          
          setActiveSimulation(updatedSim);
          setIsSimulating(false);
          
          toast({
            title: "Simulation Complete",
            description: `Detection rate: ${updatedSim.detectionRate}%, False positives: ${updatedSim.falsePositives}`,
          });
          
          return 100;
        }
        
        return newProgress;
      });
    }, 1200);
    
    // Cleanup function
    return () => clearInterval(interval);
  };

  // Format timestamp for display
  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
      hour12: true
    });
  };

  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      
      <main className="flex-1 container py-8 px-4 sm:px-8">
        <div className="flex flex-col gap-6">
          <div className="flex items-center justify-between">
            <h1 className="text-3xl font-bold bg-gradient-to-r from-kenshi-blue to-kenshi-red bg-clip-text text-transparent">
              Attack Simulator
            </h1>
            <Badge variant="outline" className="px-4 py-1">
              Kenshi Sandbox Environment
            </Badge>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Scenario Builder */}
            <div className="lg:col-span-1 space-y-6">
              <Card className="neumorph">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Book className="h-5 w-5 text-kenshi-blue" />
                    Scenario Presets
                  </CardTitle>
                  <CardDescription>Select an attack scenario to simulate</CardDescription>
                </CardHeader>
                <CardContent className="max-h-[400px] overflow-y-auto">
                  <div className="space-y-3">
                    {scenarioPresets.map(scenario => (
                      <Card 
                        key={scenario.id}
                        className={`cursor-pointer transition-colors hover:bg-accent ${
                          selectedScenario === scenario.id ? 'ring-2 ring-primary' : 'border'
                        }`}
                        onClick={() => handleScenarioSelect(scenario.id)}
                      >
                        <CardHeader className="p-4 pb-2">
                          <div className="flex justify-between items-center">
                            <CardTitle className="text-base">{scenario.name}</CardTitle>
                            <Badge variant={
                              scenario.complexity === "Very High" ? "destructive" :
                              scenario.complexity === "High" ? "default" :
                              scenario.complexity === "Medium" ? "secondary" : "outline"
                            }>
                              {scenario.complexity}
                            </Badge>
                          </div>
                        </CardHeader>
                        <CardContent className="p-4 pt-0">
                          <p className="text-sm text-muted-foreground mb-3">{scenario.description}</p>
                          <div className="flex flex-wrap gap-2">
                            {scenario.techniques.map(technique => (
                              <Badge 
                                key={technique} 
                                variant="outline" 
                                className={`bg-muted ${selectedTechnique === technique ? 'ring-1 ring-primary' : ''}`}
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleTechniqueSelect(technique);
                                }}
                              >
                                {technique}
                              </Badge>
                            ))}
                          </div>
                        </CardContent>
                        <CardFooter className="p-4 pt-0">
                          <div className="w-full">
                            <div className="flex justify-between text-xs text-muted-foreground mb-1">
                              <span>Impact</span>
                              <span>{scenario.impact}%</span>
                            </div>
                            <Progress value={scenario.impact} className="h-1" />
                          </div>
                        </CardFooter>
                      </Card>
                    ))}
                  </div>
                </CardContent>
              </Card>
              
              {/* Selected Technique Details */}
              {selectedTechnique && techniquesData[selectedTechnique as keyof typeof techniquesData] && (
                <Card className="neumorph">
                  <CardHeader className="pb-2">
                    <div className="flex justify-between items-start">
                      <CardTitle className="text-base">{techniquesData[selectedTechnique as keyof typeof techniquesData].name}</CardTitle>
                      <Badge variant="outline">{selectedTechnique}</Badge>
                    </div>
                    <CardDescription>
                      {techniquesData[selectedTechnique as keyof typeof techniquesData].tactic}
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm mb-4">{techniquesData[selectedTechnique as keyof typeof techniquesData].description}</p>
                    <div>
                      <h4 className="text-sm font-medium mb-2">Mitigations:</h4>
                      <ul className="text-sm text-muted-foreground list-disc list-inside">
                        {techniquesData[selectedTechnique as keyof typeof techniquesData].mitigations.map((mitigation, i) => (
                          <li key={i}>{mitigation}</li>
                        ))}
                      </ul>
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>
            
            {/* Visualization and Controls */}
            <div className="lg:col-span-2">
              <Card className="neumorph overflow-hidden">
                <CardHeader className="pb-2 border-b">
                  <div className="flex justify-between items-center">
                    <CardTitle className="flex items-center gap-2">
                      <Play className="h-5 w-5 text-kenshi-blue" />
                      Attack Visualization
                    </CardTitle>
                    {selectedScenario && (
                      <Button 
                        variant="default" 
                        size="sm"
                        disabled={isSimulating}
                        onClick={runSimulation}
                      >
                        {isSimulating ? 'Simulating...' : 'Run Simulation'}
                      </Button>
                    )}
                  </div>
                </CardHeader>
                
                <CardContent className="p-0">
                  <div className="relative">
                    <AspectRatio ratio={16/9}>
                      <SplineViewer scene={attackVisualizationUrl} />
                      
                      {isSimulating && (
                        <div className="absolute inset-0 bg-background/10 backdrop-blur-sm flex flex-col items-center justify-center z-10">
                          <div className="text-center mb-4">
                            <h3 className="text-lg font-bold mb-1">Attack Simulation in Progress</h3>
                            <p className="text-sm text-muted-foreground">
                              {simulationProgress < 25 ? "Establishing initial access..." :
                               simulationProgress < 60 ? "Executing payload..." :
                               simulationProgress < 100 ? "Attempting lateral movement..." :
                               "Executing final attack stage..."}
                            </p>
                          </div>
                          
                          <div className="w-64 mb-2">
                            <Progress value={simulationProgress} className="h-2" />
                          </div>
                          <p className="text-sm">{Math.floor(simulationProgress)}% Complete</p>
                        </div>
                      )}
                      
                      {!selectedScenario && !isSimulating && (
                        <div className="absolute inset-0 bg-background/20 backdrop-blur-sm flex items-center justify-center">
                          <div className="text-center max-w-md p-6 neumorph rounded-lg">
                            <CircleChevronDown className="h-10 w-10 mx-auto mb-4 text-muted-foreground" />
                            <h3 className="text-lg font-bold mb-2">Select an Attack Scenario</h3>
                            <p className="text-sm text-muted-foreground">
                              Choose from the scenario presets on the left to begin a simulated attack against your protected infrastructure.
                            </p>
                          </div>
                        </div>
                      )}
                    </AspectRatio>
                  </div>
                  
                  {/* Live attack metrics section */}
                  {isSimulating && (
                    <div className="p-4 border-t">
                      <h3 className="text-sm font-medium mb-2">Live Attack Telemetry</h3>
                      <div className="grid grid-cols-3 gap-4">
                        <div className="p-3 bg-muted/30 rounded-md">
                          <div className="text-xs text-muted-foreground mb-1">Detection Confidence</div>
                          <div className="font-mono text-lg">
                            {Math.min(99, Math.floor(simulationProgress * 0.9))}%
                          </div>
                        </div>
                        <div className="p-3 bg-muted/30 rounded-md">
                          <div className="text-xs text-muted-foreground mb-1">Response Time</div>
                          <div className="font-mono text-lg">
                            {Math.floor(simulationProgress / 10) + 100}ms
                          </div>
                        </div>
                        <div className="p-3 bg-muted/30 rounded-md">
                          <div className="text-xs text-muted-foreground mb-1">Events Detected</div>
                          <div className="font-mono text-lg">
                            {Math.floor(simulationProgress / 20) + 1}
                          </div>
                        </div>
                      </div>
                      
                      <div className="mt-3">
                        <div className="text-xs text-muted-foreground mb-1">Latest Events</div>
                        <div className="text-sm max-h-[100px] overflow-y-auto bg-black/50 p-2 rounded font-mono">
                          {activeSimulation?.events?.map((event: any, i: number) => (
                            <div key={i} className="mb-1">
                              <span className="text-kenshi-gray mr-2">[{new Date(event.time).toLocaleTimeString()}]</span>
                              <span className={event.type === "alert" ? "text-kenshi-red" : "text-muted-foreground"}>
                                {event.message}
                              </span>
                            </div>
                          )) || <span className="text-muted-foreground">No events recorded yet...</span>}
                        </div>
                      </div>
                    </div>
                  )}
                  
                  {/* Results section when simulation is complete */}
                  {activeSimulation && activeSimulation.status === "completed" && !isSimulating && (
                    <div className="p-4 border-t">
                      <div className="flex flex-col sm:flex-row gap-4 sm:items-center justify-between mb-4">
                        <div>
                          <h3 className="font-medium">Simulation Results</h3>
                          <p className="text-sm text-muted-foreground">
                            Completed {formatTimestamp(activeSimulation.timestamp)}
                          </p>
                        </div>
                        <div className="flex gap-2">
                          <TooltipProvider>
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <Button variant="outline" size="sm" className="h-8 w-8 p-0">
                                  <FileHeart className="h-4 w-4" />
                                </Button>
                              </TooltipTrigger>
                              <TooltipContent>
                                <p>Export Report</p>
                              </TooltipContent>
                            </Tooltip>
                          </TooltipProvider>
                          
                          <Button variant="default" size="sm" onClick={runSimulation}>
                            Run Again
                          </Button>
                        </div>
                      </div>
                      
                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                        <div className="p-3 bg-muted/30 rounded-md">
                          <div className="text-xs text-muted-foreground mb-1">Detection Rate</div>
                          <div className="font-mono text-lg">{activeSimulation.detectionRate}%</div>
                        </div>
                        <div className="p-3 bg-muted/30 rounded-md">
                          <div className="text-xs text-muted-foreground mb-1">False Positives</div>
                          <div className="font-mono text-lg">{activeSimulation.falsePositives}</div>
                        </div>
                        <div className="p-3 bg-muted/30 rounded-md">
                          <div className="text-xs text-muted-foreground mb-1">Duration</div>
                          <div className="font-mono text-lg">{activeSimulation.duration}</div>
                        </div>
                        <div className="p-3 bg-muted/30 rounded-md">
                          <div className="text-xs text-muted-foreground mb-1">Total Events</div>
                          <div className="font-mono text-lg">
                            {(activeSimulation.events?.length || 0) + Math.floor(Math.random() * 20 + 10)}
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </div>
          
          {/* Simulation History */}
          <div className="neumorph rounded-lg p-6">
            <div className="flex items-center gap-3 mb-6">
              <FileHeart className="h-5 w-5 text-kenshi-blue" />
              <h2 className="text-xl font-semibold">Simulation History</h2>
            </div>
            
            <div className="overflow-x-auto">
              <table className="w-full border-collapse">
                <thead>
                  <tr>
                    <th className="text-left p-2 text-muted-foreground text-sm">ID</th>
                    <th className="text-left p-2 text-muted-foreground text-sm">Scenario</th>
                    <th className="text-left p-2 text-muted-foreground text-sm">Timestamp</th>
                    <th className="text-left p-2 text-muted-foreground text-sm">Duration</th>
                    <th className="text-left p-2 text-muted-foreground text-sm">Detection Rate</th>
                    <th className="text-left p-2 text-muted-foreground text-sm">False Positives</th>
                    <th className="text-left p-2 text-muted-foreground text-sm">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {activeSimulation?.status === "completed" && (
                    <tr className="border-b">
                      <td className="p-2">{activeSimulation.id}</td>
                      <td className="p-2">{activeSimulation.scenario}</td>
                      <td className="p-2">{formatTimestamp(activeSimulation.timestamp)}</td>
                      <td className="p-2">{activeSimulation.duration}</td>
                      <td className="p-2">
                        <Badge variant={
                          activeSimulation.detectionRate >= 90 ? "default" :
                          activeSimulation.detectionRate >= 75 ? "secondary" : "destructive"
                        }>
                          {activeSimulation.detectionRate}%
                        </Badge>
                      </td>
                      <td className="p-2">{activeSimulation.falsePositives}</td>
                      <td className="p-2">
                        <Button variant="ghost" size="sm">Report</Button>
                      </td>
                    </tr>
                  )}
                  
                  {simulationHistory.map(sim => (
                    <tr key={sim.id} className="border-b">
                      <td className="p-2">{sim.id}</td>
                      <td className="p-2">{sim.scenario}</td>
                      <td className="p-2">{formatTimestamp(sim.timestamp)}</td>
                      <td className="p-2">{sim.duration}</td>
                      <td className="p-2">
                        <Badge variant={
                          sim.detectionRate >= 90 ? "default" :
                          sim.detectionRate >= 75 ? "secondary" : "destructive"
                        }>
                          {sim.detectionRate}%
                        </Badge>
                      </td>
                      <td className="p-2">{sim.falsePositives}</td>
                      <td className="p-2">
                        <Button variant="ghost" size="sm">Report</Button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default AttackSimulator;
