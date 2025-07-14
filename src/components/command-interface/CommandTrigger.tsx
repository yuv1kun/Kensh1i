
import React from 'react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Command, Brain } from 'lucide-react';

interface CommandTriggerProps {
  onOpenCommandPalette: () => void;
  onOpenAIAssistant: () => void;
}

export function CommandTrigger({ onOpenCommandPalette, onOpenAIAssistant }: CommandTriggerProps) {
  return (
    <div className="fixed bottom-4 left-4 flex flex-col gap-2 z-40">
      {/* Command Palette Trigger */}
      <Button
        variant="outline"
        size="sm"
        onClick={onOpenCommandPalette}
        className="bg-background/95 backdrop-blur-md border-kenshi-blue/30 hover:border-kenshi-blue/60 transition-colors group"
      >
        <Command className="h-4 w-4 mr-2 text-kenshi-blue" />
        Command Palette
        <Badge variant="outline" className="ml-2 text-xs">
          Ctrl+K
        </Badge>
      </Button>
      
      {/* AI Assistant Trigger */}
      <Button
        variant="outline"
        size="sm"
        onClick={onOpenAIAssistant}
        className="bg-background/95 backdrop-blur-md border-kenshi-purple/30 hover:border-kenshi-purple/60 transition-colors group"
      >
        <Brain className="h-4 w-4 mr-2 text-kenshi-purple" />
        AI Assistant
        <Badge variant="outline" className="ml-2 text-xs">
          Ask AI
        </Badge>
      </Button>
    </div>
  );
}
