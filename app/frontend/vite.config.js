import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// `npm run dev` proxies /api to the FastAPI backend (uvicorn aeronet.main:app --port 8000).
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { '/api': process.env.AERONET_API || 'http://localhost:8000' } },
  test: { environment: 'node' },
});
