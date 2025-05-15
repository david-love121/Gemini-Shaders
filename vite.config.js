import { defineConfig } from 'vite'

// Replace 'ai-shaders' with the actual name of your GitHub repository if it's different
const repoName = 'ai-shaders';

export default defineConfig({
  // No specific plugins needed for this vanilla Three.js project
  base: process.env.NODE_ENV === 'production' ? `/${repoName}/` : '/',
})