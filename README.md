# Infernux Web Platform

The official browser build plugin for [Infernux](https://github.com/ChenlizheMe/Infernux). It exports an Infernux game as WebAssembly, runs the engine's Python gameplay layer in the browser, and renders through WebGPU.

[简体中文](README.zh-CN.md) · [Infernux Engine](https://github.com/ChenlizheMe/Infernux) · [Plugin Template](https://github.com/InfernuxEngine/infernux_plugin_template) · [Releases](https://github.com/InfernuxEngine/infernux_web/releases)

![Infernux Web export workflow](package/plugin_pages/media/overview.png)

## What this plugin provides

- The `web-wasm32` build target
- Precompiled WebAssembly/JavaScript Player and CPython data
- Native glslang and Tint tools for Windows x64 and Linux x64 hosts
- WebGPU rendering, browser input bridges, and a customizable HTML shell

| Package | Version | Compatible engine | Build hosts | Target |
| --- | --- | --- | --- | --- |
| `infernux/platform-web` | 0.2.17 | Infernux >=0.4.3; payload 0.4.3 | Windows/Linux x64 | WebAssembly + WebGPU |

## Install and use

Open **Plugins** in Infernux, select **Infernux Web Platform** from the official catalog, then import and enable it. Official installs use the Infernux distribution service first and GitHub Releases as the network fallback. Manual installation is available through `infernux.platform-web.inxpkg` on the Releases page.

Choose `web-wasm32` in the build settings and export. Publish the entire generated directory and serve it over HTTP for local testing or HTTPS in production; opening the page with `file://` is unsupported. A WebGPU-capable browser and device are required. Users do not install Emscripten or compile the engine.

To customize the host page, copy `package/editor/infernux_web/templates/host/shell.html` to `ProjectSettings/WebTemplate/shell.html` in your project. Keep the runtime markers and canvas integration while changing the surrounding page.

## Repository guide

Only `package/` is distributed as the plugin. The outer native sources, build scripts, tests, and CI are maintainer material. Release engineering builds the Web Player and shader tools directly into `package/`; pushing a matching `v<version>` tag publishes the `.inxpkg` and release manifest automatically.

## License

[MIT](LICENSE). Bundled third-party components retain their own licenses.
