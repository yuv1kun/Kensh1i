
import React, { useState, useEffect } from 'react';
import { Command, CommandDialog, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from '@/components/ui/command';
import { Badge } from '@/components/ui/badge';
import { Brain, Shield, Activity, Network, Settings, Zap, Globe, Heart } from 'lucide-react';

interface CommandPaletteProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onExecuteCommand: (command: string, args?: any) => void;
  analysisState: { isRunning: boolean };
  metrics: {
    activeThreats: number;
    anomalyScore: number;
    processedPackets: number;
  };
}

interface Command {
  id: string;
  label: string;
  category: string;
  icon: React.ComponentType<{ className?: string }>;
  description: string;
  shortcut?: string;
  action: string;
  args?: any;
}

export function CommandPalette({ open, onOpenChange, onExecuteCommand, analysisState, metrics }: CommandPaletteProps) {
  const [searchValue, setSearchValue] = useState('');

  const commands: Command[] = [
    // Neural Analysis Commands
    {
      id: 'start-analysis',
      label: 'Start Neural Analysis',
      category: 'Neural Operations',
      icon: Brain,
      description: 'Begin real-time network traffic analysis',
      shortcut: 'Ctrl+A',
      action: 'START_ANALYSIS'
    },
    {
      id: 'pause-analysis',
      label: 'Pause Neural Analysis',
      category: 'Neural Operations',
      icon: Brain,
      description: 'Pause the neural network processing',
      shortcut: 'Ctrl+P',
      action: 'PAUSE_ANALYSIS'
    },
    {
      id: 'neural-reset',
      label: 'Reset Neural Network',
      category: 'Neural Operations',
      icon: Zap,
      description: 'Reset neural network to initial state',
      action: 'RESET_NEURAL'
    },
    
    // Threat Management Commands
    {
      id: 'isolate-threats',
      label: 'Isolate Active Threats',
      category: 'Threat Management',
      icon: Shield,
      description: `Quarantine ${metrics.activeThreats} detected threats`,
      action: 'ISOLATE_THREATS',
      args: { count: metrics.activeThreats }
    },
    {
      id: 'threat-analysis',
      label: 'Deep Threat Analysis',
      category: 'Threat Management',
      icon: Activity,
      description: 'Perform comprehensive threat vector analysis',
      action: 'DEEP_ANALYSIS'
    },
    {
      id: 'auto-response',
      label: 'Enable Auto-Response',
      category: 'Threat Management',
      icon: Shield,
      description: 'Activate automated threat response protocols',
      action: 'AUTO_RESPONSE'
    },
    
    // Network Commands
    {
      id: 'network-scan',
      label: 'Full Network Scan',
      category: 'Network Operations',
      icon: Network,
      description: 'Initiate comprehensive network topology scan',
      action: 'NETWORK_SCAN'
    },
    {
      id: 'traffic-analysis',
      label: 'Traffic Pattern Analysis',
      category: 'Network Operations',
      icon: Activity,
      description: `Analyze ${metrics.processedPackets.toLocaleString()} processed packets`,
      action: 'TRAFFIC_ANALYSIS',
      args: { packets: metrics.processedPackets }
    },
    {
      id: 'bandwidth-optimization',
      label: 'Optimize Bandwidth',
      category: 'Network Operations',
      icon: Globe,
      description: 'Optimize network bandwidth allocation',
      action: 'OPTIMIZE_BANDWIDTH'
    },
    
    // System Commands
    {
      id: 'system-health',
      label: 'System Health Check',
      category: 'System Operations',
      icon: Heart,
      description: 'Perform comprehensive system diagnostics',
      action: 'HEALTH_CHECK'
    },
    {
      id: 'export-logs',
      label: 'Export System Logs',
      category: 'System Operations',
      icon: Settings,
      description: 'Export detailed analysis logs',
      action: 'EXPORT_LOGS'
    },
    {
      id: 'neural-config',
      label: 'Neural Configuration',
      category: 'System Operations',
      icon: Settings,
      description: 'Configure neural network parameters',
      action: 'NEURAL_CONFIG'
    }
  ];

  const filteredCommands = commands.filter(command =>
    command.label.toLowerCase().includes(searchValue.toLowerCase()) ||
    command.description.toLowerCase().includes(searchValue.toLowerCase()) ||
    command.category.toLowerCase().includes(searchValue.toLowerCase())
  );

  const groupedCommands = filteredCommands.reduce((groups, command) => {
    if (!groups[command.category]) {
      groups[command.category] = [];
    }
    groups[command.category].push(command);
    return groups;
  }, {} as Record<string, Command[]>);

  const handleSelect = (command: Command) => {
    onExecuteCommand(command.action, command.args);
    onOpenChange(false);
    setSearchValue('');
  };

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        onOpenChange(!open);
      }
      
      if ((e.ctrlKey || e.metaKey) && e.key === 'a' && !open) {
        e.preventDefault();
        onExecuteCommand('START_ANALYSIS');
      }
      
      if ((e.ctrlKey || e.metaKey) && e.key === 'p' && !open) {
        e.preventDefault();
        onExecuteCommand('PAUSE_ANALYSIS');
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [open, onOpenChange, onExecuteCommand]);

  return (
    <CommandDialog open={open} onOpenChange={onOpenChange}>
      <CommandInput
        placeholder="Type a command or search..."
        value={searchValue}
        onValueChange={setSearchValue}
      />
      <CommandList>
        <CommandEmpty>No commands found.</CommandEmpty>
        
        {Object.entries(groupedCommands).map(([category, categoryCommands]) => (
          <CommandGroup key={category} heading={category}>
            {categoryCommands.map((command) => (
              <CommandItem
                key={command.id}
                value={command.label}
                onSelect={() => handleSelect(command)}
                className="flex items-center justify-between py-3"
              >
                <div className="flex items-center gap-3">
                  <command.icon className="h-4 w-4 text-muted-foreground" />
                  <div className="flex flex-col">
                    <span className="font-medium">{command.label}</span>
                    <span className="text-xs text-muted-foreground">{command.description}</span>
                  </div>
                </div>
                
                {command.shortcut && (
                  <Badge variant="outline" className="text-xs">
                    {command.shortcut}
                  </Badge>
                )}
              </CommandItem>
            ))}
          </CommandGroup>
        ))}
        
        {searchValue && (
          <CommandGroup heading="AI Assistant">
            <CommandItem
              value={`Ask AI: ${searchValue}`}
              onSelect={() => {
                onExecuteCommand('ASK_AI', { query: searchValue });
                onOpenChange(false);
                setSearchValue('');
              }}
              className="flex items-center gap-3 py-3"
            >
              <Brain className="h-4 w-4 text-kenshi-blue" />
              <div className="flex flex-col">
                <span className="font-medium">Ask AI Assistant</span>
                <span className="text-xs text-muted-foreground">"{searchValue}"</span>
              </div>
            </CommandItem>
          </CommandGroup>
        )}
      </CommandList>
    </CommandDialog>
  );
}
