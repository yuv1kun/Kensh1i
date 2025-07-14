import { useState } from "react";
import { Link } from "react-router-dom";
import { Brain, Menu, X } from "lucide-react";
import { ThemeToggle } from "@/components/theme-toggle";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

type NavItem = {
  label: string;
  icon: string;
  href: string;
};

const navItems: NavItem[] = [
  { label: "Real-Time Monitor", icon: "🟢", href: "/" },
  { label: "Threat Library", icon: "📚", href: "/threat-library" },
  { label: "Neural Architecture", icon: "🧠", href: "/neural-architecture" },
  { label: "Documentation", icon: "📖", href: "/documentation" },
];

export function Header() {
  const [isOpen, setIsOpen] = useState(false);
  const [threatCount, setThreatCount] = useState(3);

  return (
    <header className="sticky top-0 z-40 w-full border-b bg-background/80 backdrop-blur">
      <div className="container flex h-16 items-center justify-between px-4 sm:px-8">
        <div className="flex items-center gap-2">
          <Link to="/" className="flex items-center gap-2 outline-none">
            <div className="relative h-8 w-8 overflow-hidden">
              <Brain className="h-8 w-8 text-kenshi-blue" />
            </div>
            <span className="hidden text-xl font-bold bg-gradient-to-r from-kenshi-blue to-kenshi-red bg-clip-text text-transparent sm:inline-block">
              Kenshi
            </span>
          </Link>
          <div className="ml-2 flex items-center">
            <span className="relative h-3 w-3">
              <span className="absolute h-3 w-3 rounded-full bg-kenshi-red animate-status-pulse"></span>
              <span className="absolute h-3 w-3 rounded-full bg-kenshi-red"></span>
            </span>
            <span className="ml-2 text-xs font-medium text-muted-foreground">
              {threatCount} active threats
            </span>
          </div>
        </div>
        
        <nav className="hidden md:flex items-center gap-6">
          {navItems.map((item) => (
            <Link
              key={item.href}
              to={item.href}
              className="flex items-center gap-1 text-sm font-medium text-muted-foreground transition-colors hover:text-primary"
            >
              <span>{item.icon}</span>
              <span>{item.label}</span>
            </Link>
          ))}
        </nav>
        
        <div className="flex items-center gap-4">
          <ThemeToggle />
          <Button
            variant="ghost"
            className="md:hidden"
            onClick={() => setIsOpen(!isOpen)}
          >
            <Menu className="h-6 w-6" />
            <span className="sr-only">Toggle menu</span>
          </Button>
        </div>
      </div>
      
      {/* Mobile Navigation */}
      <div
        className={cn(
          "fixed inset-0 top-16 z-50 grid h-[calc(100vh-4rem)] grid-flow-row auto-rows-max overflow-auto p-6 pb-32 shadow-md animate-in md:hidden",
          isOpen
            ? "fade-in-from-right slide-in-from-right duration-300"
            : "fade-out-to-right slide-out-to-right duration-300 hidden"
        )}
      >
        <div className="relative z-20 grid gap-6 rounded-md bg-background p-4 neumorph">
          <nav className="grid grid-flow-row auto-rows-max text-sm">
            {navItems.map((item, index) => (
              <Link
                key={index}
                to={item.href}
                className="flex w-full items-center rounded-md p-2 text-sm font-medium hover:bg-accent hover:text-accent-foreground"
                onClick={() => setIsOpen(false)}
              >
                <span className="mr-2">{item.icon}</span>
                {item.label}
              </Link>
            ))}
          </nav>
        </div>
      </div>
    </header>
  );
}
