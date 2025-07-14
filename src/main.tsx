
import { createRoot } from 'react-dom/client'
import App from './App.tsx'
import './index.css'

// Add Content Security Policy meta tag to allow iframe embedding
const meta = document.createElement('meta');
meta.httpEquiv = 'Content-Security-Policy';
meta.content = "default-src * 'unsafe-inline' 'unsafe-eval'; script-src * 'unsafe-inline' 'unsafe-eval'; connect-src * 'unsafe-inline'; img-src * data: blob: 'unsafe-inline'; frame-src *; style-src * 'unsafe-inline';";
document.head.appendChild(meta);

createRoot(document.getElementById("root")!).render(<App />);
