import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/aws-api': {
        target: 'https://18-234-54-50.nip.io',
        changeOrigin: true,
        secure: true,
        rewrite: (path) => path.replace(/^\/aws-api/, ''),
      },
      '/local-api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/local-api/, ''),
      },
      '/characters': {
        target: 'https://18-234-54-50.nip.io',
        changeOrigin: true,
      },
      '/static': {
        target: 'https://18-234-54-50.nip.io',
        changeOrigin: true,
      },
    },
  },
});
