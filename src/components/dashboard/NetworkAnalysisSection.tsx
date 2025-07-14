
import React from 'react';
import { NetworkGraph } from "@/components/network-graph";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Network } from "lucide-react";

type NetworkNode = {
  id: string;
  group: number;
};

type NetworkLink = {
  source: string | NetworkNode;
  target: string | NetworkNode;
  value: number;
};

interface NetworkData {
  nodes: NetworkNode[];
  links: NetworkLink[];
}

interface NetworkAnalysisSectionProps {
  networkData: NetworkData;
}

export function NetworkAnalysisSection({ networkData }: NetworkAnalysisSectionProps) {
  return (
    <section className="rounded-lg neumorph overflow-hidden">
      <div className="bg-background p-6">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <Network className="h-5 w-5 text-kenshi-red" />
            Network Analysis
          </h2>
        </div>
        
        <Tabs defaultValue="graph">
          <TabsList>
            <TabsTrigger value="graph">Graph View</TabsTrigger>
            <TabsTrigger value="raw">Raw Data</TabsTrigger>
          </TabsList>
          
          <TabsContent value="graph" className="mt-2">
            <div className="h-[400px]">
              <NetworkGraph data={networkData} />
            </div>
          </TabsContent>
          
          <TabsContent value="raw" className="mt-2">
            <div className="h-[400px] overflow-auto">
              <div className="grid grid-cols-4 gap-2 font-mono text-xs">
                <div className="font-bold">Source</div>
                <div className="font-bold">Destination</div>
                <div className="font-bold">Packets</div>
                <div className="font-bold">Status</div>
                
                {networkData.links.map((link: NetworkLink, i) => (
                  <React.Fragment key={i}>
                    <div>{typeof link.source === 'string' ? link.source : link.source.id}</div>
                    <div>{typeof link.target === 'string' ? link.target : link.target.id}</div>
                    <div>{link.value * 137} pkts</div>
                    <div>
                      {Math.random() > 0.8 ? (
                        <span className="text-kenshi-red">Anomalous</span>
                      ) : (
                        <span className="text-green-600">Normal</span>
                      )}
                    </div>
                  </React.Fragment>
                ))}
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </section>
  );
}
