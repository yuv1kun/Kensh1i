
import React, { useState } from "react";
import { Header } from "@/components/header";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { AspectRatio } from "@/components/ui/aspect-ratio";
import { SplineViewer } from "@/components/spline-viewer";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import {
  NavigationMenu,
  NavigationMenuContent,
  NavigationMenuItem,
  NavigationMenuLink,
  NavigationMenuList,
  NavigationMenuTrigger,
  navigationMenuTriggerStyle,
} from "@/components/ui/navigation-menu";
import { Book, FileHeart, Layers, FileText, ChartLine, BookOpen, Search, Brain } from "lucide-react";
import { cn } from "@/lib/utils";

// Mock data for research papers
const researchPapers = [
  {
    id: "paper-001",
    title: "Neuromorphic Computing: From Materials to Systems Architecture",
    authors: "Irina Rish, et al.",
    year: "2024",
    journal: "Nature Reviews Materials",
    abstract: "This review discusses recent progress in neuromorphic computing, from novel materials to system architecture.",
    url: "https://example.com/paper1",
    tags: ["neuromorphic", "materials", "architecture"]
  },
  {
    id: "paper-002",
    title: "Spike-Timing-Dependent Plasticity in Artificial Neural Networks",
    authors: "David Silver, et al.",
    year: "2023",
    journal: "Neural Computation",
    abstract: "We demonstrate how STDP rules can be incorporated into modern deep learning architectures for more efficient learning.",
    url: "https://example.com/paper2",
    tags: ["STDP", "plasticity", "learning"]
  },
  {
    id: "paper-003",
    title: "Real-time Threat Detection Using Spiking Neural Networks",
    authors: "Grace Hopper, et al.",
    year: "2024",
    journal: "IEEE Transactions on Cybersecurity",
    abstract: "Novel approach to network threat detection using event-based spiking neural networks with reduced latency.",
    url: "https://example.com/paper3",
    tags: ["security", "threat-detection", "real-time"]
  },
  {
    id: "paper-004",
    title: "Energy Efficiency Comparison between SNN and CNN Architectures",
    authors: "Alan Turing, et al.",
    year: "2023",
    journal: "Journal of Machine Learning Research",
    abstract: "Comprehensive analysis of energy consumption patterns between spiking and conventional neural networks.",
    url: "https://example.com/paper4",
    tags: ["energy-efficiency", "comparison", "benchmarks"]
  }
];

// Mock data for benchmark comparisons
const benchmarkData = {
  technical: [
    { metric: "Detection Accuracy", traditional: 89, neuromorphic: 95, unit: "%" },
    { metric: "Inference Latency", traditional: 210, neuromorphic: 28, unit: "ms" },
    { metric: "Energy Efficiency", traditional: 112, neuromorphic: 8, unit: "mW" },
    { metric: "False Positive Rate", traditional: 6.2, neuromorphic: 2.8, unit: "%" },
    { metric: "Model Size", traditional: 420, neuromorphic: 85, unit: "MB" }
  ],
  userStudy: [
    { metric: "Setup Complexity", traditional: 7.2, neuromorphic: 5.8, unit: "/10" },
    { metric: "Response Time", traditional: 6.5, neuromorphic: 8.7, unit: "/10" },
    { metric: "Alert Quality", traditional: 7.4, neuromorphic: 8.9, unit: "/10" },
    { metric: "Overall UX", traditional: 6.8, neuromorphic: 8.3, unit: "/10" },
    { metric: "Learning Curve", traditional: 6.1, neuromorphic: 7.2, unit: "/10" }
  ]
};

// Glossary of terms
const glossaryTerms = [
  {
    term: "SNN",
    definition: "Spiking Neural Network: A type of artificial neural network that more closely mimics natural neural networks by transmitting information through spikes (discrete events) rather than continuous values.",
    category: "neural"
  },
  {
    term: "STDP",
    definition: "Spike-Timing-Dependent Plasticity: A biological process that adjusts the strength of connections between neurons based on the relative timing of a particular neuron's output and input action potentials (spikes).",
    category: "learning"
  },
  {
    term: "Neuromorphic",
    definition: "Computing architectures that mimic neuro-biological structures present in the nervous system, designed to process information similarly to the brain.",
    category: "architecture"
  },
  {
    term: "Membrane Potential",
    definition: "The difference in electrical potential between the interior of a neuron and its surroundings. In SNNs, when this potential exceeds a threshold, the neuron fires a spike.",
    category: "neural"
  },
  {
    term: "LIF Neuron",
    definition: "Leaky Integrate-and-Fire Neuron: A mathematical model that describes how neurons integrate incoming signals and generate output spikes when reaching a threshold.",
    category: "neural"
  },
  {
    term: "Event-Based Processing",
    definition: "Computing paradigm where operations are triggered by events (like spikes) rather than synchronized clock cycles, leading to energy efficiency.",
    category: "architecture"
  },
  {
    term: "Homeostasis",
    definition: "Regulatory mechanisms that maintain neural activity within a stable range, preventing excessive excitation or inhibition.",
    category: "learning"
  },
  {
    term: "Temporal Coding",
    definition: "Information encoding method where the timing of spikes carries meaningful data, as opposed to rate coding where the frequency of spikes matters.",
    category: "neural"
  }
];

// Mock data for the brain regions
const brainRegions = [
  {
    id: "region-001",
    name: "Input Layer",
    description: "Receives network traffic features and converts them to spike trains",
    neurons: 256,
    role: "Feature extraction"
  },
  {
    id: "region-002",
    name: "Hidden Layer 1",
    description: "First processing layer with recurrent connections",
    neurons: 512,
    role: "Pattern detection"
  },
  {
    id: "region-003",
    name: "Hidden Layer 2",
    description: "Second processing layer with lateral inhibition",
    neurons: 384,
    role: "Feature integration"
  },
  {
    id: "region-004",
    name: "Output Layer",
    description: "Classification layer with winner-take-all mechanism",
    neurons: 128,
    role: "Decision making"
  }
];

// Spline scene URL for brain visualization
const brainVisualizationUrl = "https://my.spline.design/particleaibrain-fRP1lexDTOPNTrZqEBAsVvOA/";

// Components for NavigationMenu content
const ListItem = React.forwardRef<
  React.ElementRef<"a">,
  React.ComponentPropsWithoutRef<"a">
>(({ className, title, children, ...props }, ref) => {
  return (
    <li>
      <NavigationMenuLink asChild>
        <a
          ref={ref}
          className={cn(
            "block select-none space-y-1 rounded-md p-3 leading-none no-underline outline-none transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground",
            className
          )}
          {...props}
        >
          <div className="text-sm font-medium leading-none">{title}</div>
          <p className="line-clamp-2 text-sm leading-snug text-muted-foreground">
            {children}
          </p>
        </a>
      </NavigationMenuLink>
    </li>
  );
});
ListItem.displayName = "ListItem";

const Documentation = () => {
  const [selectedComparisonType, setSelectedComparisonType] = useState<"technical" | "userStudy">("technical");
  const [selectedLetter, setSelectedLetter] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [activeRegion, setActiveRegion] = useState<string | null>(null);

  // Filter glossary terms by search or selected letter
  const filteredGlossaryTerms = glossaryTerms.filter(term => {
    if (searchTerm) {
      return term.term.toLowerCase().includes(searchTerm.toLowerCase()) || 
             term.definition.toLowerCase().includes(searchTerm.toLowerCase());
    }
    if (selectedLetter) {
      return term.term.charAt(0).toUpperCase() === selectedLetter;
    }
    return true;
  });

  // Generate alphabet for glossary navigation
  const alphabet = Array.from({ length: 26 }, (_, i) => String.fromCharCode(65 + i));
  const availableLetters = new Set(glossaryTerms.map(term => term.term.charAt(0).toUpperCase()));

  // Handle region selection
  const handleRegionSelect = (regionId: string) => {
    setActiveRegion(regionId === activeRegion ? null : regionId);
  };

  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      
      <main className="flex-1 container py-8 px-4 sm:px-8">
        <div className="flex flex-col gap-6">
          <div className="flex items-center justify-between">
            <h1 className="text-3xl font-bold bg-gradient-to-r from-kenshi-blue to-kenshi-red bg-clip-text text-transparent">
              Documentation
            </h1>
            <Badge variant="outline" className="px-4 py-1">
              Neuromorphic Security Platform
            </Badge>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 doc-grid">
            {/* Navigation Menu */}
            <div className="lg:col-span-1">
              <Card className="neumorph h-full">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Book className="h-5 w-5 text-kenshi-blue" />
                    Documentation Hub
                  </CardTitle>
                  <CardDescription>Explore neuromorphic security concepts</CardDescription>
                </CardHeader>
                <CardContent>
                  <NavigationMenu orientation="vertical" className="max-w-full w-full">
                    <NavigationMenuList className="flex flex-col space-y-2 w-full">
                      <NavigationMenuItem className="w-full">
                        <NavigationMenuTrigger className="w-full justify-start">
                          <ChartLine className="mr-2 h-4 w-4" />
                          Comparative Analysis
                        </NavigationMenuTrigger>
                        <NavigationMenuContent>
                          <ul className="grid w-[400px] gap-3 p-4">
                            <ListItem
                              title="SNN vs. Traditional ML"
                              href="#comparative-analysis"
                            >
                              Performance benchmarks and architectural differences
                            </ListItem>
                            <ListItem
                              title="Energy Efficiency"
                              href="#energy-efficiency"
                            >
                              Power consumption and environmental impact
                            </ListItem>
                            <ListItem
                              title="Security Applications"
                              href="#security-applications"
                            >
                              Case studies in network security and threat detection
                            </ListItem>
                          </ul>
                        </NavigationMenuContent>
                      </NavigationMenuItem>

                      <NavigationMenuItem className="w-full">
                        <NavigationMenuTrigger className="w-full justify-start">
                          <Layers className="mr-2 h-4 w-4" />
                          Neuromorphic Computing
                        </NavigationMenuTrigger>
                        <NavigationMenuContent>
                          <ul className="grid w-[400px] gap-3 p-4">
                            <ListItem
                              title="Architecture Overview"
                              href="#architecture"
                            >
                              Core components of the Kenshi neuromorphic platform
                            </ListItem>
                            <ListItem
                              title="Neural Models"
                              href="#neural-models"
                            >
                              Spiking neuron models and connection topologies
                            </ListItem>
                            <ListItem
                              title="Learning Rules"
                              href="#learning-rules"
                            >
                              STDP and other plasticity mechanisms
                            </ListItem>
                          </ul>
                        </NavigationMenuContent>
                      </NavigationMenuItem>

                      <NavigationMenuItem className="w-full">
                        <NavigationMenuTrigger className="w-full justify-start">
                          <FileText className="mr-2 h-4 w-4" />
                          Educational Resources
                        </NavigationMenuTrigger>
                        <NavigationMenuContent>
                          <ul className="grid w-[400px] gap-3 p-4">
                            <ListItem
                              title="Research Papers"
                              href="#research-papers"
                            >
                              Academic publications related to neuromorphic security
                            </ListItem>
                            <ListItem
                              title="Glossary"
                              href="#glossary"
                            >
                              Technical terminology and definitions
                            </ListItem>
                            <ListItem
                              title="External Links"
                              href="#external-links"
                            >
                              Additional learning resources and communities
                            </ListItem>
                          </ul>
                        </NavigationMenuContent>
                      </NavigationMenuItem>

                      <li className="w-full">
                        <a
                          href="#brain-visualization"
                          className={cn(navigationMenuTriggerStyle(), "w-full justify-start cursor-pointer")}
                        >
                          <BookOpen className="mr-2 h-4 w-4" />
                          Interactive Brain Model
                        </a>
                      </li>
                    </NavigationMenuList>
                  </NavigationMenu>
                </CardContent>
              </Card>
            </div>
            
            {/* Brain Visualization and Content */}
            <div className="lg:col-span-2">
              <Card className="neumorph overflow-hidden">
                <CardHeader className="pb-2 border-b">
                  <div className="flex justify-between items-center">
                    <CardTitle className="flex items-center gap-2">
                      <Brain className="h-5 w-5 text-kenshi-blue" />
                      Neuromorphic Computing Primer
                    </CardTitle>
                  </div>
                </CardHeader>
                
                <CardContent className="p-0">
                  <div className="relative">
                    <AspectRatio ratio={16/9}>
                      <SplineViewer scene={brainVisualizationUrl} />
                    </AspectRatio>
                    
                    <div className="absolute bottom-4 left-4 right-4 flex flex-wrap gap-2 z-10">
                      {brainRegions.map(region => (
                        <TooltipProvider key={region.id}>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Button 
                                variant={activeRegion === region.id ? "default" : "outline"} 
                                size="sm"
                                className={activeRegion === region.id ? "bg-kenshi-blue text-white" : "bg-background/80 backdrop-blur-sm"}
                                onClick={() => handleRegionSelect(region.id)}
                              >
                                {region.name}
                              </Button>
                            </TooltipTrigger>
                            <TooltipContent>
                              <p className="font-medium">{region.name}</p>
                              <p className="text-sm text-muted-foreground">{region.description}</p>
                              <p className="text-xs mt-1">{region.neurons} neurons • {region.role}</p>
                            </TooltipContent>
                          </Tooltip>
                        </TooltipProvider>
                      ))}
                    </div>
                  </div>
                  
                  {/* Region details */}
                  {activeRegion && (
                    <div className="p-4 border-t">
                      {brainRegions.filter(region => region.id === activeRegion).map(region => (
                        <div key={region.id}>
                          <h3 className="text-lg font-medium">{region.name}</h3>
                          <p className="text-muted-foreground mt-1">{region.description}</p>
                          <div className="grid grid-cols-2 gap-4 mt-3">
                            <div className="bg-muted/30 p-3 rounded-md">
                              <div className="text-xs text-muted-foreground">Neurons</div>
                              <div className="text-lg font-mono">{region.neurons}</div>
                            </div>
                            <div className="bg-muted/30 p-3 rounded-md">
                              <div className="text-xs text-muted-foreground">Role</div>
                              <div className="text-lg font-mono">{region.role}</div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </div>
          
          {/* Comparative Analysis Section */}
          <div id="comparative-analysis" className="neumorph rounded-lg p-6">
            <div className="flex items-center gap-3 mb-6">
              <ChartLine className="h-5 w-5 text-kenshi-blue" />
              <h2 className="text-xl font-semibold">Comparative Analysis: SNN vs. Traditional ML</h2>
            </div>
            
            <Tabs value={selectedComparisonType} onValueChange={(v) => setSelectedComparisonType(v as "technical" | "userStudy")}>
              <div className="flex justify-end mb-4">
                <TabsList>
                  <TabsTrigger value="technical">Technical Benchmarks (CIC-IDS2017)</TabsTrigger>
                  <TabsTrigger value="userStudy">User Studies</TabsTrigger>
                </TabsList>
              </div>
              
              <TabsContent value="technical" className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <Card className="col-span-1 md:col-span-3">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-lg">Performance Comparison</CardTitle>
                      <CardDescription>
                        Benchmark results using the CIC-IDS2017 network intrusion dataset
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="overflow-x-auto">
                        <table className="w-full">
                          <thead>
                            <tr className="border-b">
                              <th className="text-left p-2">Metric</th>
                              <th className="text-center p-2">Traditional ML</th>
                              <th className="text-center p-2">Neuromorphic (SNN)</th>
                              <th className="text-left p-2">Improvement</th>
                            </tr>
                          </thead>
                          <tbody>
                            {benchmarkData.technical.map((item, index) => {
                              const isImproved = item.metric.includes('False') || item.metric.includes('Latency') || item.metric.includes('Size') || item.metric.includes('Energy') 
                                ? item.neuromorphic < item.traditional
                                : item.neuromorphic > item.traditional;
                              
                              const improvement = item.metric.includes('False') || item.metric.includes('Latency') || item.metric.includes('Size') || item.metric.includes('Energy')
                                ? ((item.traditional - item.neuromorphic) / item.traditional * 100).toFixed(1)
                                : ((item.neuromorphic - item.traditional) / item.traditional * 100).toFixed(1);
                              
                              return (
                                <tr key={index} className="border-b">
                                  <td className="p-2 font-medium">{item.metric}</td>
                                  <td className="p-2 text-center">{item.traditional} {item.unit}</td>
                                  <td className="p-2 text-center">{item.neuromorphic} {item.unit}</td>
                                  <td className="p-2">
                                    <Badge variant={isImproved ? "default" : "destructive"}>
                                      {isImproved ? "+" : "-"}{Math.abs(parseFloat(improvement))}%
                                    </Badge>
                                  </td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </TabsContent>
              
              <TabsContent value="userStudy" className="space-y-4">
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-lg">User Experience Metrics</CardTitle>
                    <CardDescription>
                      Results from user studies comparing traditional ML and neuromorphic security systems
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-6">
                      {benchmarkData.userStudy.map((item, index) => (
                        <div key={index} className="space-y-2">
                          <div className="flex justify-between">
                            <span className="font-medium">{item.metric}</span>
                            <span className="text-sm text-muted-foreground">(Scale: 1-10)</span>
                          </div>
                          <div className="flex items-center gap-4">
                            <span className="w-28 text-sm text-right">Traditional: {item.traditional}</span>
                            <div className="flex-1 flex gap-2">
                              <div className="flex-1 h-2 rounded-full bg-muted overflow-hidden">
                                <div 
                                  className="h-full bg-muted-foreground rounded-full" 
                                  style={{ width: `${(item.traditional/10)*100}%` }}
                                ></div>
                              </div>
                              <div className="flex-1 h-2 rounded-full bg-muted overflow-hidden">
                                <div 
                                  className="h-full bg-kenshi-blue rounded-full" 
                                  style={{ width: `${(item.neuromorphic/10)*100}%` }}
                                ></div>
                              </div>
                            </div>
                            <span className="w-28 text-sm">Neuromorphic: {item.neuromorphic}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          </div>
          
          {/* Educational Resources */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Research Papers */}
            <div id="research-papers" className="neumorph rounded-lg p-6">
              <div className="flex items-center gap-3 mb-6">
                <FileHeart className="h-5 w-5 text-kenshi-blue" />
                <h2 className="text-xl font-semibold">Research Papers</h2>
              </div>
              
              <div className="space-y-4">
                {researchPapers.map(paper => (
                  <Card key={paper.id}>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-base">{paper.title}</CardTitle>
                      <CardDescription>
                        {paper.authors} • {paper.year} • {paper.journal}
                      </CardDescription>
                    </CardHeader>
                    <CardContent className="pb-2">
                      <p className="text-sm text-muted-foreground">{paper.abstract}</p>
                    </CardContent>
                    <CardFooter className="pt-0">
                      <div className="flex flex-wrap gap-2">
                        {paper.tags.map((tag, i) => (
                          <Badge key={i} variant="outline">{tag}</Badge>
                        ))}
                      </div>
                    </CardFooter>
                  </Card>
                ))}
              </div>
            </div>
            
            {/* Glossary */}
            <div id="glossary" className="neumorph rounded-lg p-6">
              <div className="flex items-center gap-3 mb-4">
                <BookOpen className="h-5 w-5 text-kenshi-blue" />
                <h2 className="text-xl font-semibold">Glossary</h2>
              </div>
              
              <div className="flex flex-wrap items-center gap-1 mb-4">
                <div className="flex items-center border rounded-md overflow-hidden">
                  <div className="flex items-center pl-3">
                    <Search className="h-4 w-4 text-muted-foreground" />
                  </div>
                  <input
                    type="text"
                    placeholder="Search terms..."
                    className="py-1 px-2 bg-transparent border-0 focus:outline-none focus:ring-0 text-sm"
                    value={searchTerm}
                    onChange={(e) => {
                      setSearchTerm(e.target.value);
                      setSelectedLetter(null);
                    }}
                  />
                </div>
                
                <div className="flex flex-wrap gap-1 ml-2">
                  {alphabet.map(letter => (
                    <button
                      key={letter}
                      className={cn(
                        "w-6 h-6 flex items-center justify-center rounded text-xs",
                        selectedLetter === letter 
                          ? "bg-primary text-primary-foreground" 
                          : availableLetters.has(letter) 
                            ? "hover:bg-muted cursor-pointer" 
                            : "text-muted-foreground cursor-not-allowed opacity-50"
                      )}
                      disabled={!availableLetters.has(letter)}
                      onClick={() => {
                        setSelectedLetter(letter === selectedLetter ? null : letter);
                        setSearchTerm("");
                      }}
                    >
                      {letter}
                    </button>
                  ))}
                </div>
              </div>
              
              <div className="space-y-4 max-h-[400px] overflow-y-auto pr-2">
                {filteredGlossaryTerms.length > 0 ? (
                  filteredGlossaryTerms.map(item => (
                    <div key={item.term} className="border-b pb-3">
                      <div className="flex justify-between items-start">
                        <h3 className="font-bold">{item.term}</h3>
                        <Badge variant="outline">{item.category}</Badge>
                      </div>
                      <p className="text-sm text-muted-foreground mt-1">{item.definition}</p>
                    </div>
                  ))
                ) : (
                  <div className="text-center py-8 text-muted-foreground">
                    No terms found matching your search.
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Documentation;
