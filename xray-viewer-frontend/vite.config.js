import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { nodePolyfills } from 'vite-plugin-node-polyfills'
import { viteCommonjs } from '@originjs/vite-plugin-commonjs'

export default defineConfig({
  plugins: [
    react(),
    nodePolyfills({
      // Polyfill Node globals/modules (needed by cornerstoneJS's dependencies)
      include: ['events'],
    }),
    // Official Cornerstone3D + Vite requirement (cornerstonejs.org/docs/
    // getting-started/vue-angular-react-etc) - dicom-image-loader's codec
    // sub-packages are genuine CommonJS; this plugin does the CJS->ESM
    // interop that excluding them from optimizeDeps alone doesn't provide.
    viteCommonjs(),
  ],
  optimizeDeps: {
    // Known Vite + cornerstone dicom-image-loader issue: pre-bundling breaks
    // web worker file resolution, causing loadImage() promises to hang forever
    // (never resolve, never reject). See cornerstonejs/cornerstone3D#1538.
    exclude: ['@cornerstonejs/dicom-image-loader'],
    include: ['dicom-parser'],
  },
  worker: {
    format: 'es',
  },
  server: {
    // Proxy DICOMweb requests to Orthanc through the dev server itself, so
    // the browser sees a same-origin request - avoids CORS entirely rather
    // than trying to configure CORS headers on Orthanc, which is what both
    // the Orthanc community and OHIF's own docs recommend (OHIF's own dev
    // server uses the equivalent PROXY_TARGET/PROXY_DOMAIN pattern).
    proxy: {
      '/dicom-web': {
        target: 'http://localhost:8042',
        changeOrigin: true,
      },
    },
  },
})