
import React, { useState } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ThreatIntelligencePanel } from './ThreatIntelligencePanel';
import { PredictiveAnalytics } from './PredictiveAnalytics';
import { Button } from "@/components/ui/button";
import { BarChart3, Brain, Shield, TrendingUp, Download, RefreshCw } from "lucide-react";
import { toast } from "@/hooks/use-toast";

interface AdvancedAnalyticsPanelProps {
  activeThreats: number;
  anomalyScore: number;
  isAnalysisActive: boolean;
  processedPackets: number;
}

export function AdvancedAnalyticsPanel({
  activeThreats,
  anomalyScore,
  isAnalysisActive,
  processedPackets
}: AdvancedAnalyticsPanelProps) {
  const [isRefreshing, setIsRefreshing] = useState(false);

  const handleRefresh = async () => {
    setIsRefreshing(true);
    // Simulate refresh delay
    await new Promise(resolve => setTimeout(resolve, 1000));
    setIsRefreshing(false);
    toast({
      title: "Analytics Refreshed",
      description: "Threat intelligence and predictions updated with latest data",
    });
  };

  const handleExportReport = () => {
    toast({
      title: "Analytics Report Exported",
      description: "Comprehensive threat analysis report has been generated",
    });
  };

  return (
    <section className="rounded-lg neumorph overflow-hidden">
      <div className="bg-background p-6 border-b border-border/50">
        <div className="flex justify-between items-center">
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <BarChart3 className="h-5 w-5 text-kenshi-purple" />
            Advanced Analytics Dashboard
          </h2>
          <div className="flex gap-2">
            <Button 
              variant="outline" 
              size="sm"
              onClick={handleRefresh}
              disabled={isRefreshing}
            >
              <RefreshCw className={`h-4 w-4 mr-2 ${isRefreshing ? 'animate-spin' : ''}`} />
              {isRefreshing ? 'Refreshing...' : 'Refresh'}
            </Button>
            <Button 
              variant="outline" 
              size="sm"
              onClick={handleExportReport}
            >
              <Download className="h-4 w-4 mr-2" />
              Export Report
            </Button>
          </div>
        </div>
        <p className="text-muted-foreground mt-1">
          Real-time threat intelligence and predictive analytics powered by neural networks
        </p>
      </div>
      
      <div className="p-6">
        <Tabs defaultValue="intelligence" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="intelligence" className="flex items-center gap-2">
              <Shield className="h-4 w-4" />
              Threat Intelligence
            </TabsTrigger>
            <TabsTrigger value="predictions" className="flex items-center gap-2">
              <Brain className="h-4 w-4" />
              Predictive Analytics
            </TabsTrigger>
          </TabsList>
          
          <TabsContent value="intelligence" className="mt-6">
            <ThreatIntelligencePanel 
              activeThreats={activeThreats}
              anomalyScore={anomalyScore}
            />
          </TabsContent>
          
          <TabsContent value="predictions" className="mt-6">
            <PredictiveAnalytics 
              anomalyScore={anomalyScore}
              activeThreats={activeThreats}
              isAnalysisActive={isAnalysisActive}
            />
          </TabsContent>
        </Tabs>
      </div>
    </section>
  );
}
