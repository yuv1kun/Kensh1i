
import { useEffect, useState } from "react";

type Theme = "light" | "dark";

export function useTheme() {
  // Check if we're on the client side
  const isBrowser = typeof window !== "undefined";
  
  // Initialize state with user's preference or default to light
  const [theme, setTheme] = useState<Theme>(() => {
    if (!isBrowser) return "light";
    
    const storedTheme = localStorage.getItem("theme");
    if (storedTheme === "light" || storedTheme === "dark") {
      return storedTheme;
    }
    
    return window.matchMedia("(prefers-color-scheme: dark)").matches 
      ? "dark" 
      : "light";
  });

  useEffect(() => {
    if (!isBrowser) return;

    // Update HTML class and localStorage when theme changes
    const root = window.document.documentElement;
    root.classList.remove("light", "dark");
    root.classList.add(theme);
    localStorage.setItem("theme", theme);
    
    // Force a re-render by updating a CSS custom property
    root.style.setProperty('--theme-transition', 'all 0.3s ease');
  }, [theme, isBrowser]);

  // Listen for system theme changes
  useEffect(() => {
    if (!isBrowser) return;

    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
    const handleChange = (e: MediaQueryListEvent) => {
      const storedTheme = localStorage.getItem("theme");
      if (!storedTheme) {
        setTheme(e.matches ? "dark" : "light");
      }
    };

    mediaQuery.addEventListener("change", handleChange);
    return () => mediaQuery.removeEventListener("change", handleChange);
  }, [isBrowser]);

  return { theme, setTheme };
}
