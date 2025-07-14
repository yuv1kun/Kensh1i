
import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Brain, MessageCircle, X, Minimize2, Maximize2 } from 'lucide-react';

interface AIAssistantProps {
  isVisible: boolean;
  onClose: () => void;
  analysisState: { isRunning: boolean };
  metrics: {
    activeThreats: number;
    anomalyScore: number;
    processedPackets: number;
  };
}

interface AIMessage {
  id: string;
  type: 'user' | 'ai';
  content: string;
  timestamp: Date;
  context?: any;
}

export function AIAssistant({ isVisible, onClose, analysisState, metrics }: AIAssistantProps) {
  const [messages, setMessages] = useState<AIMessage[]>([]);
  const [isMinimized, setIsMinimized] = useState(false);
  const [isThinking, setIsThinking] = useState(false);

  // Generate AI insights based on current system state
  const generateAIInsight = (query?: string): string => {
    if (query) {
      // Handle specific queries
      if (query.toLowerCase().includes('threat')) {
        return `Based on current analysis, we have ${metrics.activeThreats} active threats with an anomaly score of ${metrics.anomalyScore}. The neural network is ${analysisState.isRunning ? 'actively processing' : 'in standby mode'}. I recommend ${metrics.anomalyScore > 0.5 ? 'immediate threat isolation' : 'continued monitoring'}.`;
      }
      if (query.toLowerCase().includes('network')) {
        return `Network analysis shows ${metrics.processedPackets.toLocaleString()} packets processed. Current anomaly detection is ${analysisState.isRunning ? 'active' : 'paused'}. Network health appears ${metrics.anomalyScore < 0.3 ? 'stable' : 'concerning'}.`;
      }
      if (query.toLowerCase().includes('optimize')) {
        return `System optimization suggestions: ${analysisState.isRunning ? 'Neural processing is active and performing well' : 'Consider starting neural analysis for real-time insights'}. With ${metrics.activeThreats} threats detected, I recommend ${metrics.activeThreats > 3 ? 'aggressive filtering' : 'standard monitoring protocols'}.`;
      }
      
      return `I've analyzed your query: "${query}". Based on current system metrics (${metrics.activeThreats} threats, ${metrics.anomalyScore} anomaly score), the neural network recommends ${metrics.anomalyScore > 0.4 ? 'heightened security measures' : 'maintaining current protocols'}.`;
    }
    
    // General system insights
    const insights = [
      `Neural analysis ${analysisState.isRunning ? 'is actively monitoring' : 'is in standby mode'}. Current threat level: ${metrics.activeThreats > 2 ? 'ELEVATED' : 'NORMAL'}.`,
      `Anomaly detection shows ${(metrics.anomalyScore * 100).toFixed(1)}% deviation from baseline. ${metrics.anomalyScore > 0.5 ? 'Immediate attention recommended.' : 'Within acceptable parameters.'}`,
      `Network processing efficiency is optimal with ${metrics.processedPackets.toLocaleString()} packets analyzed. ${analysisState.isRunning ? 'Real-time processing active.' : 'Batch processing mode.'}`,
      `System recommendation: ${metrics.activeThreats > 3 ? 'Implement threat isolation protocols' : 'Continue standard monitoring procedures'}.`
    ];
    
    return insights[Math.floor(Math.random() * insights.length)];
  };

  // Add initial AI message when assistant opens
  useEffect(() => {
    if (isVisible && messages.length === 0) {
      const initialMessage: AIMessage = {
        id: Date.now().toString(),
        type: 'ai',
        content: `Neural AI Assistant online. ${generateAIInsight()}`,
        timestamp: new Date()
      };
      setMessages([initialMessage]);
    }
  }, [isVisible, messages.length, analysisState, metrics]);

  const handleUserQuery = (query: string) => {
    const userMessage: AIMessage = {
      id: Date.now().toString(),
      type: 'user',
      content: query,
      timestamp: new Date()
    };
    
    setMessages(prev => [...prev, userMessage]);
    setIsThinking(true);
    
    // Simulate AI processing time
    setTimeout(() => {
      const aiResponse: AIMessage = {
        id: (Date.now() + 1).toString(),
        type: 'ai',
        content: generateAIInsight(query),
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, aiResponse]);
      setIsThinking(false);
    }, 1500);
  };

  const getQuickActions = () => [
    { label: 'System Status', query: 'What is the current system status?' },
    { label: 'Threat Analysis', query: 'Analyze current threats' },
    { label: 'Network Health', query: 'Check network performance' },
    { label: 'Optimize System', query: 'How can I optimize the system?' }
  ];

  if (!isVisible) return null;

  return (
    <Card className="fixed bottom-4 right-4 w-96 max-h-[600px] bg-background/95 backdrop-blur-md border border-kenshi-blue/30 shadow-lg z-50">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-sm">
            <Brain className="h-4 w-4 text-kenshi-blue" />
            Neural AI Assistant
            <Badge variant="outline" className="text-xs">
              {analysisState.isRunning ? 'Active' : 'Standby'}
            </Badge>
          </CardTitle>
          
          <div className="flex items-center gap-1">
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6"
              onClick={() => setIsMinimized(!isMinimized)}
            >
              {isMinimized ? <Maximize2 className="h-3 w-3" /> : <Minimize2 className="h-3 w-3" />}
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6"
              onClick={onClose}
            >
              <X className="h-3 w-3" />
            </Button>
          </div>
        </div>
      </CardHeader>
      
      {!isMinimized && (
        <CardContent className="pt-0">
          {/* Messages */}
          <div className="max-h-64 overflow-y-auto mb-4 space-y-2">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-[80%] rounded-lg px-3 py-2 text-sm ${
                    message.type === 'user'
                      ? 'bg-kenshi-blue text-white'
                      : 'bg-muted text-foreground'
                  }`}
                >
                  {message.content}
                </div>
              </div>
            ))}
            
            {isThinking && (
              <div className="flex justify-start">
                <div className="bg-muted text-foreground rounded-lg px-3 py-2 text-sm">
                  <div className="flex items-center gap-2">
                    <div className="flex space-x-1">
                      <div className="h-1 w-1 bg-kenshi-blue rounded-full animate-pulse"></div>
                      <div className="h-1 w-1 bg-kenshi-blue rounded-full animate-pulse" style={{ animationDelay: '0.2s' }}></div>
                      <div className="h-1 w-1 bg-kenshi-blue rounded-full animate-pulse" style={{ animationDelay: '0.4s' }}></div>
                    </div>
                    <span>AI thinking...</span>
                  </div>
                </div>
              </div>
            )}
          </div>
          
          {/* Quick Actions */}
          <div className="grid grid-cols-2 gap-2 mb-4">
            {getQuickActions().map((action, index) => (
              <Button
                key={index}
                variant="outline"
                size="sm"
                className="text-xs h-8"
                onClick={() => handleUserQuery(action.query)}
                disabled={isThinking}
              >
                {action.label}
              </Button>
            ))}
          </div>
          
          {/* Current Metrics Display */}
          <div className="text-xs text-muted-foreground bg-muted/50 rounded p-2">
            <div className="grid grid-cols-3 gap-2">
              <div>Threats: {metrics.activeThreats}</div>
              <div>Anomaly: {(metrics.anomalyScore * 100).toFixed(1)}%</div>
              <div>Status: {analysisState.isRunning ? 'Active' : 'Standby'}</div>
            </div>
          </div>
        </CardContent>
      )}
    </Card>
  );
}
