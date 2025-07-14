
import { useState } from "react";
import { Header } from "@/components/header";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Separator } from "@/components/ui/separator";
import { Search, Grid2x2, FolderOpen, Book, CircleHelp, AlertTriangle, Shield, Target, ExternalLink } from "lucide-react";

// Enhanced threat actor data with detailed information
const threatActors = [
  {
    id: "apt28",
    name: "APT28",
    aliases: ["Fancy Bear", "Sofacy", "Strontium"],
    origin: "Russia",
    motivations: ["Espionage", "Information theft"],
    techniques: ["T1566.002", "T1486", "T1190"],
    description: "Russian state-sponsored threat actor targeting government, military, and security organizations.",
    detailedInfo: {
      firstSeen: "2008",
      targetSectors: ["Government", "Military", "Defense contractors", "Think tanks", "Media"],
      notableAttacks: ["2016 US Election interference", "DNC hack", "NATO member countries targeting"],
      infrastructure: "Uses compromised websites, legitimate web services, and custom domains for C2",
      tools: ["CHOPSTICK", "ADVSTORESHELL", "X-Tunnel", "Zebrocy"],
      tactics: "Spear-phishing emails with malicious attachments or links, watering hole attacks, credential harvesting",
      attribution: "Linked to GRU Unit 26165 and Unit 74455",
      riskLevel: "Critical"
    }
  },
  {
    id: "lazarus",
    name: "Lazarus Group",
    aliases: ["Hidden Cobra", "Guardians of Peace"],
    origin: "North Korea",
    motivations: ["Financial gain", "Regime interests"],
    techniques: ["T1566.001", "T1027", "T1059.003"],
    description: "North Korean state-sponsored group known for financial theft, WannaCry ransomware, and destructive attacks.",
    detailedInfo: {
      firstSeen: "2009",
      targetSectors: ["Financial institutions", "Cryptocurrency exchanges", "Entertainment", "Healthcare"],
      notableAttacks: ["Sony Pictures hack (2014)", "WannaCry ransomware (2017)", "SWIFT banking attacks"],
      infrastructure: "Extensive botnet infrastructure, compromised legitimate websites, bulletproof hosting",
      tools: ["WannaCry", "RATANKBA", "BADCALL", "Joanap"],
      tactics: "Spear-phishing, watering hole attacks, supply chain compromises, living-off-the-land techniques",
      attribution: "Linked to RGB (Reconnaissance General Bureau) Unit 180",
      riskLevel: "Critical"
    }
  },
  {
    id: "darkside",
    name: "DarkSide",
    aliases: ["Carbon Spider", "Ransomware-as-a-Service"],
    origin: "Eastern Europe",
    motivations: ["Financial gain"],
    techniques: ["T1566", "T1078", "T1486"],
    description: "Ransomware group responsible for the Colonial Pipeline attack, operating a ransomware-as-a-service model.",
    detailedInfo: {
      firstSeen: "2020",
      targetSectors: ["Energy", "Manufacturing", "Legal", "Technology", "Retail"],
      notableAttacks: ["Colonial Pipeline (2021)", "Multiple Fortune 500 companies"],
      infrastructure: "RaaS model with affiliate network, Tor-based payment sites, bulletproof hosting",
      tools: ["DarkSide ransomware", "Cobalt Strike", "PowerShell Empire"],
      tactics: "Double extortion, data exfiltration before encryption, professional negotiation tactics",
      attribution: "Russian-speaking cybercriminal group",
      riskLevel: "High"
    }
  },
  {
    id: "apt41",
    name: "APT41",
    aliases: ["Double Dragon", "Wicked Panda"],
    origin: "China",
    motivations: ["Espionage", "Financial gain"],
    techniques: ["T1190", "T1133", "T1559"],
    description: "Chinese threat actor conducting state-sponsored espionage and financially motivated operations.",
    detailedInfo: {
      firstSeen: "2012",
      targetSectors: ["Healthcare", "Telecommunications", "Technology", "Gaming", "Government"],
      notableAttacks: ["Supply chain attacks on software companies", "COVID-19 research theft"],
      infrastructure: "Global infrastructure including compromised websites and cloud services",
      tools: ["MESSAGETAP", "HIGHNOON", "LOWKEY", "WINNKIT"],
      tactics: "Supply chain compromises, SQL injection attacks, spear-phishing, watering hole attacks",
      attribution: "Linked to MSS-affiliated contractor groups",
      riskLevel: "Critical"
    }
  }
];

// Enhanced malware data with detailed information
const malwareSamples = [
  {
    id: "zloader",
    name: "ZLoader",
    type: "Banking Trojan",
    firstSeen: "2020-03",
    techniques: ["T1204", "T1055"],
    description: "Banking trojan that steals credentials and financial information through web injection techniques.",
    detailedInfo: {
      family: "Zeus variants",
      platform: "Windows",
      propagation: ["Malicious email attachments", "Exploit kits", "Malvertising"],
      capabilities: ["Credential harvesting", "Web injection", "Form grabbing", "Screenshot capture"],
      persistence: "Registry modifications, scheduled tasks, service installation",
      c2Communication: "HTTPS with domain generation algorithm (DGA)",
      variants: ["ZLoader v2.0", "ZLoader Silent Night"],
      yaraRule: `rule ZLoader_Banker {
  meta:
    description = "ZLoader Banking Trojan"
    author = "Threat Research Team"
  strings:
    $s1 = "ZeuS" ascii
    $s2 = "botnet" ascii
    $hex1 = { 8B 45 ?? 83 C0 ?? 89 45 ?? }
  condition:
    2 of them
}`,
      riskLevel: "High"
    }
  },
  {
    id: "emotet",
    name: "Emotet",
    type: "Banking Trojan / Dropper",
    firstSeen: "2014-06",
    techniques: ["T1566.001", "T1027"],
    description: "Sophisticated banking trojan that serves as a downloader for other malware families.",
    detailedInfo: {
      family: "Emotet",
      platform: "Windows",
      propagation: ["Phishing emails", "Malicious macros", "Network shares"],
      capabilities: ["Email harvesting", "Credential theft", "Module loading", "Lateral movement"],
      persistence: "Windows services, registry keys, scheduled tasks",
      c2Communication: "HTTP/HTTPS with RSA encryption",
      variants: ["Epoch 1", "Epoch 2", "Epoch 3"],
      yaraRule: `rule Emotet_Dropper {
  meta:
    description = "Emotet Malware Family"
    author = "Threat Research Team"
  strings:
    $s1 = "RegSvcs.exe" ascii
    $s2 = "rundll32" ascii
    $hex1 = { E8 ?? ?? ?? ?? 83 C4 ?? 5E 5D C3 }
  condition:
    all of them
}`,
      riskLevel: "Critical"
    }
  },
  {
    id: "ryuk",
    name: "Ryuk",
    type: "Ransomware",
    firstSeen: "2018-08",
    techniques: ["T1486", "T1489"],
    description: "Targeted ransomware used in high-profile attacks against enterprises and critical infrastructure.",
    detailedInfo: {
      family: "Ryuk",
      platform: "Windows",
      propagation: ["Manual deployment", "Lateral movement", "Remote access tools"],
      capabilities: ["File encryption", "Service termination", "Shadow copy deletion", "Network discovery"],
      persistence: "Typically manual deployment, no persistence mechanisms",
      c2Communication: "No traditional C2, operates standalone after deployment",
      variants: ["Ryuk v1", "Ryuk v2"],
      yaraRule: `rule Ryuk_Ransomware {
  meta:
    description = "Ryuk Ransomware"
    author = "Threat Research Team"
  strings:
    $s1 = "RyukReadMe.txt" ascii
    $s2 = "Ryuk" ascii
    $hex1 = { 48 8B C4 48 89 58 08 48 89 70 10 }
  condition:
    2 of them
}`,
      riskLevel: "Critical"
    }
  },
  {
    id: "solarwinds",
    name: "SUNBURST",
    type: "Supply Chain Backdoor",
    firstSeen: "2020-12",
    techniques: ["T1195.002", "T1059.003"],
    description: "Backdoor implanted in SolarWinds Orion software affecting thousands of organizations worldwide.",
    detailedInfo: {
      family: "SUNBURST/Solorigate",
      platform: "Windows",
      propagation: ["Supply chain compromise", "Software updates"],
      capabilities: ["Command execution", "File operations", "Process enumeration", "Registry manipulation"],
      persistence: "Legitimate software update mechanism",
      c2Communication: "DNS tunneling with subdomain encoding",
      variants: ["SUNBURST", "SUNSPOT", "TEARDROP"],
      yaraRule: `rule SUNBURST_Backdoor {
  meta:
    description = "SUNBURST Supply Chain Backdoor"
    author = "Threat Research Team"
  strings:
    $s1 = "SolarWinds.Orion.Core.BusinessLayer.dll" ascii
    $s2 = "avsvmcloud.com" ascii
    $hex1 = { 56 69 72 75 73 54 6F 74 61 6C }
  condition:
    2 of them
}`,
      riskLevel: "Critical"
    }
  }
];

// Enhanced MITRE ATT&CK techniques with detailed information
const techniques = [
  { 
    id: "T1566.002", 
    name: "Spearphishing Link", 
    tactic: "initial-access",
    detailedInfo: {
      description: "Adversaries may send spearphishing emails with a malicious link in an attempt to gain access to victim systems.",
      platforms: ["Windows", "macOS", "Linux"],
      dataComponents: ["Email gateway traffic", "Web traffic"],
      mitigations: ["User awareness training", "Email filtering", "URL filtering"],
      detection: "Monitor for suspicious email patterns, URL analysis, user behavior analytics",
      examples: ["APT28 campaigns", "Lazarus Group attacks", "Business email compromise"],
      severity: "Medium"
    }
  },
  { 
    id: "T1486", 
    name: "Data Encryption for Impact", 
    tactic: "impact",
    detailedInfo: {
      description: "Adversaries may encrypt data on target systems or on large numbers of systems in a network to interrupt availability to system and network resources.",
      platforms: ["Windows", "macOS", "Linux"],
      dataComponents: ["File system activity", "Process monitoring"],
      mitigations: ["Regular backups", "Application control", "Behavior prevention"],
      detection: "Monitor for rapid file modifications, encryption indicators, ransom notes",
      examples: ["Ryuk ransomware", "WannaCry", "NotPetya"],
      severity: "High"
    }
  },
  { 
    id: "T1190", 
    name: "Exploit Public-Facing Application", 
    tactic: "initial-access",
    detailedInfo: {
      description: "Adversaries may attempt to take advantage of a weakness in an Internet-facing computer or program using software, data, or commands.",
      platforms: ["Windows", "macOS", "Linux", "Network"],
      dataComponents: ["Network traffic", "Application logs"],
      mitigations: ["Vulnerability management", "Web application firewall", "Network segmentation"],
      detection: "Monitor for exploitation attempts, unusual network traffic, application errors",
      examples: ["Exchange server exploits", "VPN vulnerabilities", "Web application attacks"],
      severity: "High"
    }
  },
  { 
    id: "T1133", 
    name: "External Remote Services", 
    tactic: "initial-access",
    detailedInfo: {
      description: "Adversaries may leverage external-facing remote services to initially access and/or persist within a network.",
      platforms: ["Windows", "macOS", "Linux"],
      dataComponents: ["Authentication logs", "Network traffic"],
      mitigations: ["Multi-factor authentication", "Network segmentation", "Account use policies"],
      detection: "Monitor authentication logs, unusual login patterns, geographic anomalies",
      examples: ["RDP brute force", "VPN compromise", "Cloud service abuse"],
      severity: "Medium"
    }
  },
  { 
    id: "T1078", 
    name: "Valid Accounts", 
    tactic: "defense-evasion",
    detailedInfo: {
      description: "Adversaries may obtain and abuse credentials of existing accounts as a means of gaining Initial Access, Persistence, Privilege Escalation, or Defense Evasion.",
      platforms: ["Windows", "macOS", "Linux", "Cloud"],
      dataComponents: ["Authentication logs", "Account management"],
      mitigations: ["Privileged account management", "Account use policies", "Multi-factor authentication"],
      detection: "Monitor for account usage anomalies, privilege escalation, credential dumping",
      examples: ["Credential stuffing", "Password spraying", "Stolen credentials"],
      severity: "Medium"
    }
  },
  { 
    id: "T1027", 
    name: "Obfuscated Files or Information", 
    tactic: "defense-evasion",
    detailedInfo: {
      description: "Adversaries may attempt to make an executable or file difficult to discover or analyze by encrypting, encoding, or otherwise obfuscating its contents.",
      platforms: ["Windows", "macOS", "Linux"],
      dataComponents: ["File system activity", "Process monitoring"],
      mitigations: ["Antivirus/antimalware", "Code analysis", "Execution prevention"],
      detection: "Monitor for obfuscated files, encoding patterns, suspicious file operations",
      examples: ["Packed malware", "Encoded scripts", "Steganography"],
      severity: "Medium"
    }
  },
  { 
    id: "T1059.003", 
    name: "Windows Command Shell", 
    tactic: "execution",
    detailedInfo: {
      description: "Adversaries may abuse the Windows command shell for execution.",
      platforms: ["Windows"],
      dataComponents: ["Process monitoring", "Command execution"],
      mitigations: ["Application control", "Execution prevention", "Privileged access management"],
      detection: "Monitor command-line activity, process creation, script execution",
      examples: ["Living off the land", "Fileless attacks", "PowerShell abuse"],
      severity: "Medium"
    }
  },
  { 
    id: "T1204", 
    name: "User Execution", 
    tactic: "execution",
    detailedInfo: {
      description: "An adversary may rely upon specific actions by a user in order to gain execution.",
      platforms: ["Windows", "macOS", "Linux"],
      dataComponents: ["Process monitoring", "File system activity"],
      mitigations: ["User training", "Application control", "Execution prevention"],
      detection: "Monitor for user-initiated executions, suspicious file operations",
      examples: ["Malicious attachments", "Social engineering", "Drive-by downloads"],
      severity: "Low"
    }
  },
  { 
    id: "T1055", 
    name: "Process Injection", 
    tactic: "privilege-escalation",
    detailedInfo: {
      description: "Adversaries may inject code into processes in order to evade process-based defenses or elevate privileges.",
      platforms: ["Windows", "macOS", "Linux"],
      dataComponents: ["Process monitoring", "API monitoring"],
      mitigations: ["Behavior prevention", "Privilege access management", "Application control"],
      detection: "Monitor for process modifications, memory injections, API calls",
      examples: ["DLL injection", "Hollowing", "Atom bombing"],
      severity: "Medium"
    }
  },
  { 
    id: "T1489", 
    name: "Service Stop", 
    tactic: "impact",
    detailedInfo: {
      description: "Adversaries may stop or disable services on a system to render those services unavailable to legitimate users.",
      platforms: ["Windows", "macOS", "Linux"],
      dataComponents: ["Service monitoring", "Process monitoring"],
      mitigations: ["User account management", "Operating system configuration"],
      detection: "Monitor for service state changes, process terminations, system events",
      examples: ["Ransomware preparation", "Security tool disabling", "System disruption"],
      severity: "Medium"
    }
  },
  { 
    id: "T1195.002", 
    name: "Compromise Software Supply Chain", 
    tactic: "initial-access",
    detailedInfo: {
      description: "Adversaries may manipulate application software prior to receipt by a final consumer for the purpose of data or system compromise.",
      platforms: ["Windows", "macOS", "Linux"],
      dataComponents: ["File integrity monitoring", "Software inventory"],
      mitigations: ["Code signing", "Software integrity checks", "Supply chain security"],
      detection: "Monitor for unsigned software, integrity violations, unusual updates",
      examples: ["SolarWinds attack", "CCleaner compromise", "NotPetya"],
      severity: "High"
    }
  },
  { 
    id: "T1566.001", 
    name: "Spearphishing Attachment", 
    tactic: "initial-access",
    detailedInfo: {
      description: "Adversaries may send spearphishing emails with a malicious attachment in an attempt to gain access to victim systems.",
      platforms: ["Windows", "macOS", "Linux"],
      dataComponents: ["Email gateway traffic", "File system activity"],
      mitigations: ["User training", "Email filtering", "Attachment scanning"],
      detection: "Monitor for suspicious attachments, execution patterns, email analysis",
      examples: ["Macro-enabled documents", "Weaponized PDFs", "Archive files"],
      severity: "Medium"
    }
  },
  { 
    id: "T1559", 
    name: "Inter-Process Communication", 
    tactic: "execution",
    detailedInfo: {
      description: "Adversaries may abuse inter-process communication (IPC) mechanisms for local code or command execution.",
      platforms: ["Windows", "macOS", "Linux"],
      dataComponents: ["Process monitoring", "API monitoring"],
      mitigations: ["Application control", "Privilege access management"],
      detection: "Monitor for IPC mechanisms, process interactions, API calls",
      examples: ["COM objects", "Named pipes", "Shared memory"],
      severity: "Low"
    }
  }
];

// MITRE ATT&CK tactic categories
const tactics = [
  { id: "reconnaissance", name: "Reconnaissance", color: "bg-blue-500" },
  { id: "resource-development", name: "Resource Development", color: "bg-indigo-500" },
  { id: "initial-access", name: "Initial Access", color: "bg-purple-500" },
  { id: "execution", name: "Execution", color: "bg-pink-500" },
  { id: "persistence", name: "Persistence", color: "bg-red-500" },
  { id: "privilege-escalation", name: "Privilege Escalation", color: "bg-orange-500" },
  { id: "defense-evasion", name: "Defense Evasion", color: "bg-yellow-500" },
  { id: "credential-access", name: "Credential Access", color: "bg-green-500" },
  { id: "discovery", name: "Discovery", color: "bg-teal-500" },
  { id: "lateral-movement", name: "Lateral Movement", color: "bg-cyan-500" },
  { id: "collection", name: "Collection", color: "bg-blue-400" },
  { id: "command-and-control", name: "Command & Control", color: "bg-blue-600" },
  { id: "exfiltration", name: "Exfiltration", color: "bg-violet-500" },
  { id: "impact", name: "Impact", color: "bg-rose-500" }
];

const ThreatLibrary = () => {
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedTactic, setSelectedTactic] = useState<string | null>(null);
  const [selectedTechnique, setSelectedTechnique] = useState<string | null>(null);

  // Filter techniques based on search and selected tactic
  const filteredTechniques = techniques.filter(technique => 
    (selectedTactic ? technique.tactic === selectedTactic : true) &&
    (searchTerm ? 
      technique.name.toLowerCase().includes(searchTerm.toLowerCase()) || 
      technique.id.toLowerCase().includes(searchTerm.toLowerCase()) 
      : true)
  );

  // Filter threat actors based on selected technique
  const filteredActors = threatActors.filter(actor => 
    !selectedTechnique || actor.techniques.includes(selectedTechnique)
  );

  // Filter malware based on selected technique
  const filteredMalware = malwareSamples.filter(malware => 
    !selectedTechnique || malware.techniques.includes(selectedTechnique)
  );

  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      
      <main className="flex-1 container py-8 px-4 sm:px-8">
        <div className="flex flex-col gap-6">
          <div className="flex items-center justify-between">
            <h1 className="text-3xl font-bold bg-gradient-to-r from-kenshi-blue to-kenshi-red bg-clip-text text-transparent">
              Threat Library
            </h1>
            <div className="relative w-64">
              <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search techniques..."
                className="pl-8"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </div>
          </div>

          <Tabs defaultValue="matrix" className="w-full">
            <TabsList className="grid w-full grid-cols-3 mb-8">
              <TabsTrigger value="matrix" className="flex items-center gap-2">
                <Grid2x2 className="h-4 w-4" />
                <span>MITRE ATT&CK Matrix</span>
              </TabsTrigger>
              <TabsTrigger value="actors" className="flex items-center gap-2">
                <FolderOpen className="h-4 w-4" />
                <span>Threat Actor Profiles</span>
              </TabsTrigger>
              <TabsTrigger value="malware" className="flex items-center gap-2">
                <Book className="h-4 w-4" />
                <span>Malware Zoo</span>
              </TabsTrigger>
            </TabsList>
            
            <TabsContent value="matrix">
              <div className="neumorph p-6 rounded-lg">
                <div className="grid grid-cols-1 md:grid-cols-3 xl:grid-cols-4 gap-4 mb-6">
                  {tactics.map(tactic => (
                    <Button
                      key={tactic.id}
                      variant={selectedTactic === tactic.id ? "default" : "outline"}
                      className="justify-start h-auto py-2"
                      onClick={() => setSelectedTactic(selectedTactic === tactic.id ? null : tactic.id)}
                    >
                      <div className={`h-3 w-3 rounded-full mr-2 ${tactic.color}`}></div>
                      <span className="text-sm">{tactic.name}</span>
                    </Button>
                  ))}
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
                  {filteredTechniques.map(technique => (
                    <Dialog key={technique.id}>
                      <DialogTrigger asChild>
                        <Card 
                          className={`cursor-pointer border-l-4 hover:bg-accent transition-colors ${
                            selectedTechnique === technique.id ? 'ring-2 ring-primary' : ''
                          } ${
                            tactics.find(t => t.id === technique.tactic)?.color.replace('bg-', 'border-l-')
                          }`}
                          onClick={() => setSelectedTechnique(selectedTechnique === technique.id ? null : technique.id)}
                        >
                          <CardHeader className="py-3">
                            <div className="flex justify-between items-start">
                              <CardTitle className="text-base">{technique.name}</CardTitle>
                              <Badge variant="outline">{technique.id}</Badge>
                            </div>
                            <CardDescription className="text-xs">
                              {tactics.find(t => t.id === technique.tactic)?.name}
                            </CardDescription>
                          </CardHeader>
                        </Card>
                      </DialogTrigger>
                      <DialogContent className="max-w-4xl max-h-[80vh] overflow-y-auto">
                        <DialogHeader>
                          <DialogTitle className="flex items-center gap-3">
                            <Target className="h-5 w-5" />
                            {technique.name} ({technique.id})
                          </DialogTitle>
                          <DialogDescription>
                            MITRE ATT&CK Technique - {tactics.find(t => t.id === technique.tactic)?.name}
                          </DialogDescription>
                        </DialogHeader>
                        
                        {technique.detailedInfo && (
                          <div className="space-y-6 mt-4">
                            <div>
                              <h3 className="text-lg font-semibold mb-2">Description</h3>
                              <p className="text-muted-foreground">{technique.detailedInfo.description}</p>
                            </div>
                            
                            <Separator />
                            
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                              <div>
                                <h4 className="font-medium mb-3 flex items-center gap-2">
                                  <Shield className="h-4 w-4" />
                                  Platforms
                                </h4>
                                <div className="flex flex-wrap gap-2">
                                  {technique.detailedInfo.platforms.map(platform => (
                                    <Badge key={platform} variant="secondary">{platform}</Badge>
                                  ))}
                                </div>
                              </div>
                              
                              <div>
                                <h4 className="font-medium mb-3">Severity Level</h4>
                                <Badge 
                                  variant={technique.detailedInfo.severity === 'High' ? 'destructive' : 
                                          technique.detailedInfo.severity === 'Medium' ? 'default' : 'outline'}
                                >
                                  {technique.detailedInfo.severity}
                                </Badge>
                              </div>
                            </div>
                            
                            <Separator />
                            
                            <div>
                              <h4 className="font-medium mb-3">Data Components</h4>
                              <div className="flex flex-wrap gap-2">
                                {technique.detailedInfo.dataComponents.map(component => (
                                  <Badge key={component} variant="outline">{component}</Badge>
                                ))}
                              </div>
                            </div>
                            
                            <div>
                              <h4 className="font-medium mb-3">Mitigations</h4>
                              <div className="flex flex-wrap gap-2">
                                {technique.detailedInfo.mitigations.map(mitigation => (
                                  <Badge key={mitigation} variant="outline" className="bg-green-50 text-green-700 border-green-200">
                                    {mitigation}
                                  </Badge>
                                ))}
                              </div>
                            </div>
                            
                            <div>
                              <h4 className="font-medium mb-3">Detection Methods</h4>
                              <p className="text-sm text-muted-foreground bg-muted p-3 rounded-lg">
                                {technique.detailedInfo.detection}
                              </p>
                            </div>
                            
                            <div>
                              <h4 className="font-medium mb-3">Real-world Examples</h4>
                              <div className="space-y-2">
                                {technique.detailedInfo.examples.map(example => (
                                  <div key={example} className="flex items-center gap-2 text-sm">
                                    <AlertTriangle className="h-3 w-3 text-orange-500" />
                                    <span>{example}</span>
                                  </div>
                                ))}
                              </div>
                            </div>
                          </div>
                        )}
                      </DialogContent>
                    </Dialog>
                  ))}
                </div>
                
                {filteredTechniques.length === 0 && (
                  <div className="py-12 text-center">
                    <p className="text-muted-foreground">No techniques found matching your criteria.</p>
                    <Button variant="outline" className="mt-4" onClick={() => { setSearchTerm(""); setSelectedTactic(null); }}>
                      Clear filters
                    </Button>
                  </div>
                )}
              </div>
            </TabsContent>
            
            <TabsContent value="actors">
              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-2 gap-6">
                {filteredActors.map(actor => (
                  <Dialog key={actor.id}>
                    <DialogTrigger asChild>
                      <Card className="neumorph overflow-hidden cursor-pointer hover:shadow-lg transition-shadow">
                        <CardHeader className="bg-gradient-to-r from-gray-900 to-slate-800 text-white">
                          <div className="flex justify-between items-center">
                            <CardTitle>{actor.name}</CardTitle>
                            <Badge variant="secondary">{actor.origin}</Badge>
                          </div>
                          <CardDescription className="text-gray-300">
                            {actor.aliases.join(" / ")}
                          </CardDescription>
                        </CardHeader>
                        <CardContent className="pt-6">
                          <p className="mb-4">{actor.description}</p>
                          <div className="grid grid-cols-2 gap-4">
                            <div>
                              <h4 className="text-sm font-medium mb-2">Motivations</h4>
                              <div className="flex flex-wrap gap-2">
                                {actor.motivations.map(motivation => (
                                  <Badge key={motivation} variant="outline" className="bg-muted">
                                    {motivation}
                                  </Badge>
                                ))}
                              </div>
                            </div>
                            <div>
                              <h4 className="text-sm font-medium mb-2">Primary Techniques</h4>
                              <div className="flex flex-wrap gap-2">
                                {actor.techniques.map(technique => (
                                  <Badge 
                                    key={technique} 
                                    variant="outline" 
                                    className={`bg-muted ${selectedTechnique === technique ? 'ring-2 ring-primary' : ''}`}
                                  >
                                    {technique}
                                  </Badge>
                                ))}
                              </div>
                            </div>
                          </div>
                        </CardContent>
                        <CardFooter className="bg-muted/30 border-t">
                          <Button variant="ghost" size="sm" className="ml-auto">
                            View Complete Profile
                          </Button>
                        </CardFooter>
                      </Card>
                    </DialogTrigger>
                    <DialogContent className="max-w-4xl max-h-[80vh] overflow-y-auto">
                      <DialogHeader>
                        <DialogTitle className="flex items-center gap-3">
                          <Shield className="h-5 w-5 text-red-500" />
                          {actor.name} - Threat Actor Profile
                        </DialogTitle>
                        <DialogDescription>
                          {actor.aliases.join(" / ")} | Origin: {actor.origin}
                        </DialogDescription>
                      </DialogHeader>
                      
                      {actor.detailedInfo && (
                        <div className="space-y-6 mt-4">
                          <div>
                            <h3 className="text-lg font-semibold mb-2">Overview</h3>
                            <p className="text-muted-foreground">{actor.description}</p>
                          </div>
                          
                          <Separator />
                          
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <div>
                              <h4 className="font-medium mb-3">First Seen</h4>
                              <Badge variant="outline">{actor.detailedInfo.firstSeen}</Badge>
                            </div>
                            <div>
                              <h4 className="font-medium mb-3">Risk Level</h4>
                              <Badge 
                                variant={actor.detailedInfo.riskLevel === 'Critical' ? 'destructive' : 'default'}
                              >
                                {actor.detailedInfo.riskLevel}
                              </Badge>
                            </div>
                          </div>
                          
                          <div>
                            <h4 className="font-medium mb-3">Target Sectors</h4>
                            <div className="flex flex-wrap gap-2">
                              {actor.detailedInfo.targetSectors.map(sector => (
                                <Badge key={sector} variant="secondary">{sector}</Badge>
                              ))}
                            </div>
                          </div>
                          
                          <div>
                            <h4 className="font-medium mb-3">Notable Attacks</h4>
                            <div className="space-y-2">
                              {actor.detailedInfo.notableAttacks.map(attack => (
                                <div key={attack} className="flex items-center gap-2 text-sm">
                                  <AlertTriangle className="h-3 w-3 text-red-500" />
                                  <span>{attack}</span>
                                </div>
                              ))}
                            </div>
                          </div>
                          
                          <Separator />
                          
                          <div>
                            <h4 className="font-medium mb-3">Infrastructure</h4>
                            <p className="text-sm text-muted-foreground bg-muted p-3 rounded-lg">
                              {actor.detailedInfo.infrastructure}
                            </p>
                          </div>
                          
                          <div>
                            <h4 className="font-medium mb-3">Known Tools</h4>
                            <div className="flex flex-wrap gap-2">
                              {actor.detailedInfo.tools.map(tool => (
                                <Badge key={tool} variant="outline" className="bg-orange-50 text-orange-700 border-orange-200">
                                  {tool}
                                </Badge>
                              ))}
                            </div>
                          </div>
                          
                          <div>
                            <h4 className="font-medium mb-3">Tactics & Techniques</h4>
                            <p className="text-sm text-muted-foreground bg-muted p-3 rounded-lg">
                              {actor.detailedInfo.tactics}
                            </p>
                          </div>
                          
                          <div>
                            <h4 className="font-medium mb-3">Attribution</h4>
                            <p className="text-sm text-muted-foreground bg-blue-50 p-3 rounded-lg border border-blue-200">
                              {actor.detailedInfo.attribution}
                            </p>
                          </div>
                        </div>
                      )}
                    </DialogContent>
                  </Dialog>
                ))}
                
                {filteredActors.length === 0 && (
                  <div className="col-span-full py-12 text-center">
                    <p className="text-muted-foreground">No threat actors found matching the selected technique.</p>
                    <Button variant="outline" className="mt-4" onClick={() => setSelectedTechnique(null)}>
                      Clear selection
                    </Button>
                  </div>
                )}
              </div>
            </TabsContent>
            
            <TabsContent value="malware">
              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
                {filteredMalware.map(malware => (
                  <Dialog key={malware.id}>
                    <DialogTrigger asChild>
                      <Card className="neumorph cursor-pointer hover:shadow-lg transition-shadow">
                        <CardHeader>
                          <div className="flex justify-between items-center">
                            <CardTitle>{malware.name}</CardTitle>
                            <Badge variant="outline">{malware.type}</Badge>
                          </div>
                          <CardDescription>First seen: {malware.firstSeen}</CardDescription>
                        </CardHeader>
                        <CardContent>
                          <p className="mb-4">{malware.description}</p>
                          <div>
                            <h4 className="text-sm font-medium mb-2">Associated Techniques</h4>
                            <div className="flex flex-wrap gap-2">
                              {malware.techniques.map(technique => (
                                <Badge 
                                  key={technique} 
                                  variant="outline" 
                                  className={`bg-muted ${selectedTechnique === technique ? 'ring-2 ring-primary' : ''}`}
                                >
                                  {technique}
                                </Badge>
                              ))}
                            </div>
                          </div>
                        </CardContent>
                        <CardFooter className="flex justify-between border-t pt-4">
                          <Button variant="outline" size="sm">
                            YARA Rules
                          </Button>
                          <Button variant="default" size="sm">
                            View Analysis
                          </Button>
                        </CardFooter>
                      </Card>
                    </DialogTrigger>
                    <DialogContent className="max-w-4xl max-h-[80vh] overflow-y-auto">
                      <DialogHeader>
                        <DialogTitle className="flex items-center gap-3">
                          <AlertTriangle className="h-5 w-5 text-red-500" />
                          {malware.name} - Malware Analysis
                        </DialogTitle>
                        <DialogDescription>
                          {malware.type} | First seen: {malware.firstSeen}
                        </DialogDescription>
                      </DialogHeader>
                      
                      {malware.detailedInfo && (
                        <div className="space-y-6 mt-4">
                          <div>
                            <h3 className="text-lg font-semibold mb-2">Overview</h3>
                            <p className="text-muted-foreground">{malware.description}</p>
                          </div>
                          
                          <Separator />
                          
                          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                            <div>
                              <h4 className="font-medium mb-3">Family</h4>
                              <Badge variant="outline">{malware.detailedInfo.family}</Badge>
                            </div>
                            <div>
                              <h4 className="font-medium mb-3">Platform</h4>
                              <Badge variant="secondary">{malware.detailedInfo.platform}</Badge>
                            </div>
                            <div>
                              <h4 className="font-medium mb-3">Risk Level</h4>
                              <Badge 
                                variant={malware.detailedInfo.riskLevel === 'Critical' ? 'destructive' : 'default'}
                              >
                                {malware.detailedInfo.riskLevel}
                              </Badge>
                            </div>
                          </div>
                          
                          <div>
                            <h4 className="font-medium mb-3">Propagation Methods</h4>
                            <div className="flex flex-wrap gap-2">
                              {malware.detailedInfo.propagation.map(method => (
                                <Badge key={method} variant="outline" className="bg-red-50 text-red-700 border-red-200">
                                  {method}
                                </Badge>
                              ))}
                            </div>
                          </div>
                          
                          <div>
                            <h4 className="font-medium mb-3">Capabilities</h4>
                            <div className="flex flex-wrap gap-2">
                              {malware.detailedInfo.capabilities.map(capability => (
                                <Badge key={capability} variant="outline" className="bg-orange-50 text-orange-700 border-orange-200">
                                  {capability}
                                </Badge>
                              ))}
                            </div>
                          </div>
                          
                          <Separator />
                          
                          <div>
                            <h4 className="font-medium mb-3">Persistence Mechanisms</h4>
                            <p className="text-sm text-muted-foreground bg-muted p-3 rounded-lg">
                              {malware.detailedInfo.persistence}
                            </p>
                          </div>
                          
                          <div>
                            <h4 className="font-medium mb-3">Command & Control</h4>
                            <p className="text-sm text-muted-foreground bg-muted p-3 rounded-lg">
                              {malware.detailedInfo.c2Communication}
                            </p>
                          </div>
                          
                          <div>
                            <h4 className="font-medium mb-3">Known Variants</h4>
                            <div className="flex flex-wrap gap-2">
                              {malware.detailedInfo.variants.map(variant => (
                                <Badge key={variant} variant="secondary">{variant}</Badge>
                              ))}
                            </div>
                          </div>
                          
                          <Separator />
                          
                          <div>
                            <h4 className="font-medium mb-3 flex items-center gap-2">
                              <ExternalLink className="h-4 w-4" />
                              YARA Detection Rule
                            </h4>
                            <div className="bg-slate-900 text-green-400 p-4 rounded-lg font-mono text-sm overflow-x-auto">
                              <pre>{malware.detailedInfo.yaraRule}</pre>
                            </div>
                          </div>
                        </div>
                      )}
                    </DialogContent>
                  </Dialog>
                ))}
                
                {filteredMalware.length === 0 && (
                  <div className="col-span-full py-12 text-center">
                    <p className="text-muted-foreground">No malware samples found matching the selected technique.</p>
                    <Button variant="outline" className="mt-4" onClick={() => setSelectedTechnique(null)}>
                      Clear selection
                    </Button>
                  </div>
                )}
              </div>
            </TabsContent>
          </Tabs>

          {/* Documentation tooltip */}
          <div className="fixed bottom-4 right-4">
            <div className="bg-background/80 backdrop-blur-sm neumorph p-3 rounded-full flex items-center gap-2 shadow-lg">
              <CircleHelp className="h-4 w-4 text-muted-foreground" />
              <span className="text-xs text-muted-foreground">Hover over elements for more information</span>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default ThreatLibrary;
