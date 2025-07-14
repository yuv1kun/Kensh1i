
import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { TrendingUp, Brain, Zap, AlertCircle } from "lucide-react";

interface PredictionData {
  time: string;
  threatProbability: number;
  anomalyForecast: number;
  confidenceInterval: number;
}

interface PredictiveAnalyticsProps {
  anomalyScore: number;
  activeThreats: number;
  isAnalysisActive: boolean;
}

export function PredictiveAnalytics({ anomalyScore, activeThreats, isAnalysisActive }: PredictiveAnalyticsProps) {
  const [predictionData, setPredictionData] = useState<PredictionData[]>([]);
  const [riskLevel, setRiskLevel] = useState<'low' | 'medium' | 'high'>('low');

  useEffect(() => {
    // Generate predictive analytics data
    const generatePredictions = () => {
      const now = new Date();
      const predictions: PredictionData[] = [];
      
      for (let i = 0; i < 24; i++) { // 24 hour forecast
        const futureTime = new Date(now.getTime() + i * 60 * 60 * 1000);
        const baseProb = Math.min(0.9, anomalyScore + (activeThreats * 0.1));
        const randomVariation = (Math.random() - 0.5) * 0.3;
        const timePattern = Math.sin((i / 24) * Math.PI * 2) * 0.2; // Daily pattern
        
        predictions.push({
          time: futureTime.getHours() + ':00',
          threatProbability: Math.max(0.1, Math.min(0.95, baseProb + randomVariation + timePattern)),
          anomalyForecast: Math.max(0.05, Math.min(0.9, anomalyScore + randomVariation * 0.5)),
          confidenceInterval: Math.random() * 0.2 + 0.7 // 70-90% confidence
        });
      }
      
      setPredictionData(predictions);
      
      // Calculate overall risk level
      const avgThreatProb = predictions.reduce((sum, p) => sum + p.threatProbability, 0) / predictions.length;
      if (avgThreatProb > 0.7) setRiskLevel('high');
      else if (avgThreatProb > 0.4) setRiskLevel('medium');
      else setRiskLevel('low');
    };

    generatePredictions();
    
    if (isAnalysisActive) {
      const interval = setInterval(generatePredictions, 30000); // Update every 30 seconds
      return () => clearInterval(interval);
    }
  }, [anomalyScore, activeThreats, isAnalysisActive]);

  const getRiskColor = () => {
    switch (riskLevel) {
      case 'high': return 'text-red-500';
      case 'medium': return 'text-yellow-500';
      case 'low': return 'text-green-500';
      default: return 'text-gray-500';
    }
  };

  const getRiskIcon = () => {
    switch (riskLevel) {
      case 'high': return <AlertCircle className="h-5 w-5 text-red-500" />;
      case 'medium': return <TrendingUp className="h-5 w-5 text-yellow-500" />;
      case 'low': return <Brain className="h-5 w-5 text-green-500" />;
      default: return <Zap className="h-5 w-5" />;
    }
  };

  return (
    <div className="space-y-4">
      {/* Risk Assessment Summary */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5 text-purple-500" />
            AI Threat Prediction
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              {getRiskIcon()}
              <span className={`font-semibold ${getRiskColor()}`}>
                {riskLevel.toUpperCase()} RISK
              </span>
            </div>
            <div className="text-sm text-muted-foreground">
              Next 24 hours forecast
            </div>
          </div>
          
          <div className="grid grid-cols-3 gap-4 text-center">
            <div className="p-2 bg-card/50 rounded-lg">
              <div className="text-2xl font-bold text-blue-500">
                {(predictionData[0]?.threatProbability * 100 || 0).toFixed(0)}%
              </div>
              <div className="text-xs text-muted-foreground">Threat Probability</div>
            </div>
            <div className="p-2 bg-card/50 rounded-lg">
              <div className="text-2xl font-bold text-orange-500">
                {(predictionData[0]?.anomalyForecast || 0).toFixed(2)}
              </div>
              <div className="text-xs text-muted-foreground">Anomaly Forecast</div>
            </div>
            <div className="p-2 bg-card/50 rounded-lg">
              <div className="text-2xl font-bold text-green-500">
                {(predictionData[0]?.confidenceInterval * 100 || 0).toFixed(0)}%
              </div>
              <div className="text-xs text-muted-foreground">Confidence</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Threat Probability Chart */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5 text-blue-500" />
            24-Hour Threat Forecast
          </CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={predictionData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis domain={[0, 1]} />
              <Tooltip 
                formatter={(value: number) => [`${(value * 100).toFixed(1)}%`, 'Threat Probability']}
                labelFormatter={(label) => `Time: ${label}`}
              />
              <Area 
                type="monotone" 
                dataKey="threatProbability" 
                stroke="#8884d8" 
                fillOpacity={0.6}
                fill="url(#colorThreat)"
              />
              <defs>
                <linearGradient id="colorThreat" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#8884d8" stopOpacity={0.8}/>
                  <stop offset="95%" stopColor="#8884d8" stopOpacity={0}/>
                </linearGradient>
              </defs>
            </AreaChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Anomaly Forecast Chart */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="h-5 w-5 text-yellow-500" />
            Anomaly Score Prediction
          </CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={predictionData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis domain={[0, 1]} />
              <Tooltip 
                formatter={(value: number) => [value.toFixed(3), 'Anomaly Score']}
                labelFormatter={(label) => `Time: ${label}`}
              />
              <Line 
                type="monotone" 
                dataKey="anomalyForecast" 
                stroke="#ff7300" 
                strokeWidth={2}
                dot={{ fill: '#ff7300', strokeWidth: 2, r: 3 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </div>
  );
}
