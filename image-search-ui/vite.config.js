import { defineConfig, loadEnv } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'

export default defineConfig(({ mode }) => {
  // VITE_BACKEND_URL example: http://localhost:8000
  const env = loadEnv(mode, process.cwd(), '')
  const backend = env.VITE_BACKEND_URL || 'http://localhost:8000'

  return {
    plugins: [svelte()],
    server: {
      proxy: {
        '/image-search': {
          target: backend,
          changeOrigin: true
        }
      }
    }
  }
})
