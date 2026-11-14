import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './theme/theme.js'; // applies the saved or system colour theme (index.html already did, before the first paint)
import App from './App.jsx';
import '@fontsource/barlow-condensed/500.css';
import '@fontsource/barlow-condensed/600.css';
import '@fontsource/inter/400.css';
import '@fontsource/inter/500.css';
import '@fontsource/inter/600.css';
import '@fontsource/ibm-plex-mono/500.css';
import './styles.css';

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
