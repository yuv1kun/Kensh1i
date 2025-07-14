
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";

interface StatusCardProps {
  title: string;
  value: string | number;
  description?: string;
  icon?: React.ReactNode;
  trend?: "up" | "down" | "neutral";
  trendValue?: string;
  className?: string;
}

export function StatusCard({
  title,
  value,
  description,
  icon,
  trend,
  trendValue,
  className,
}: StatusCardProps) {
  return (
    <Card className={cn("neumorph overflow-hidden", className)}>
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
        {icon && <div className="opacity-70">{icon}</div>}
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">{value}</div>
        {description && (
          <CardDescription className="mt-1 text-xs">{description}</CardDescription>
        )}
        {trend && (
          <p
            className={cn(
              "mt-2 text-xs font-medium",
              trend === "up" && "text-kenshi-red",
              trend === "down" && "text-green-600",
              trend === "neutral" && "text-muted-foreground"
            )}
          >
            {trend === "up" && "▲"}
            {trend === "down" && "▼"}
            {trend === "neutral" && "●"} {trendValue}
          </p>
        )}
      </CardContent>
    </Card>
  );
}
