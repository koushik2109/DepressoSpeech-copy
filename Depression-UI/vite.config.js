import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],

  // Pre-bundle all heavy deps so the browser doesn't re-parse them on every reload
  optimizeDeps: {
    include: ['react', 'react-dom', 'react-router-dom'],
  },

  build: {
    // Smaller output → faster initial load
    target: 'es2020',
    // Split vendor chunks so React / router are cached separately from app code
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom', 'react-router-dom'],
        },
      },
    },
    // Enable minification
    minify: 'esbuild',
  },

  server: {
    // Proxy API calls so the browser avoids CORS preflight overhead in dev
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
