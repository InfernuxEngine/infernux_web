# Web Platform

![Build workflow](media/overview.png)

Build browser Players with CPython 3.13 compiled to WebAssembly and rendering through WebGPU. The package includes precompiled WASM/JavaScript, CPython data and native Windows/Linux shader tools, alongside the browser host, input bridges and project HTML templates.

## Before building

Infernux 0.4.3 on Windows x64 or Linux x64, and a WebGPU-capable browser. Ordinary exports require no engine sources, Git submodules, CMake, WSL or Emscripten.

## Included payload

`editor/infernux_web/player/` contains the precompiled runtime and CPython data. `tools/windows-x64/` and `tools/linux-x64/` contain glslang and Tint. Installing this plugin supplies the Web build payload; no separate Web SDK download is required.

Export cooks project content, translates GLSL to WGSL, writes an `.inxpkg` and assembles the web page. It does not rebuild the engine. Browser startup loads the binary package into memory instead of publishing Assets, Library or the project source tree as HTTP directories.

## Export and serve

Select web-wasm32 in the build settings. Publish the whole generated directory, including infernux-player.html, JavaScript, WebAssembly and game data. Serve it over HTTP on localhost for testing, or HTTPS for deployment; do not open the page through file://. Browser permissions and device capabilities must allow WebGPU.

## Project Web template

Create ProjectSettings/WebTemplate/shell.html from package/editor/infernux_web/templates/host/shell.html. Keep the runtime markers and canvas integration intact while changing CSS, metadata and the surrounding page. Other template files are exported under web-template/ with relative directories preserved.

Once the template directory exists, shell.html is required. Rebuilding replaces generated output: edit project template sources, not the exported files. Gameplay UI uses Infernux Screen UI; the HTML shell hosts the game rather than implementing its UI as DOM elements.

## Troubleshooting

If the payload is missing or incompatible, explicitly select a complete `.inxpkg` matching the engine version. Do not install GitHub's source archive as a runtime payload. If the browser cannot initialize WebGPU, use a supported browser/device and check its GPU settings; there is no alternate WebGL rendering path.
