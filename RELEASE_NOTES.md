# 0.2.13 — Infernux 0.4.2

- Publish the exporter and matching precompiled Player payload for Infernux 0.4.2.
- Include the current portable assets, scene lifecycle, physics, rendering and compute fixes.
- Require exactly Infernux 0.4.2 so native payloads cannot be installed into a different engine version.

# 0.2.1 — Infernux 0.4.1

- Publish the current exporter and precompiled Player payload for Infernux 0.4.1.
- Include the 041 scene, asset, UI, compute, input and lifecycle fixes from the pinned engine sources.
- Require Infernux 0.4.1 so old native payloads cannot be installed into the new engine.

# Infernux Web Platform 0.2.0

Official platform package for Infernux 0.4.0.

- Ships a precompiled WebGPU/CPython WASM runtime and native Windows/Linux shader tools.
- Ordinary exports cook assets and assemble browser files without engine sources, CMake, WSL or Emscripten.
- Separates project content into an InxPackage loaded before Python starts; no project directory tree is published.
- Windowed/fullscreen presentation is selected at runtime without relinking.
- Includes English and Simplified Chinese documentation with a build workflow illustration.
- Ships an installable InxPackage and the engine's compatible-release manifest.

## Requirements

Infernux 0.4.0 on Windows x64 or Linux x64. A WebGPU-capable browser is required; WebGL is not a fallback.

Download the .inxpkg asset to install the plugin. The automatic source archives are for plugin development. See the README for setup, output and troubleshooting details.
