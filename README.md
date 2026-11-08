# Infernux Web Platform

[简体中文](README.zh-CN.md) · [Engine releases](https://github.com/ChenlizheMe/Infernux/releases)

The official `web-wasm32` Player exporter for Infernux. Build hosts and targets: Windows/Linux x64 → WebAssembly.

![Infernux Web export workflow](package/plugin_pages/media/overview.png)

## Install

Install this platform from the engine's official plugin catalog. Platform packages are built and published with the engine release, using its exact hotfix, Python ABI and target. For example, engine `0.4.3-v8` owns platform package version `0.4.3.post8`. The source metadata in this repository is replaced by that generated identity during final packaging.

The distribution service stores each engine's platform packages under `platform-plugins/<engine_release>/`. The corresponding engine GitHub Release provides the same packages as a download alternative. The plugin repositories no longer independently publish platform binaries.

A matching precompiled Player is used directly. If an installed Player is incompatible, the engine export worker attempts a source rebuild for the current engine release, validates it, and replaces the installed plugin's Player payload. This repair requires the corresponding source toolchain. Successful repairs are reused; the engine SDK is not replaced.

## Maintain

From the engine repository, use `scripts/release/build_platform_plugins.py` to build or package this target. The engine's `engine-platform-plugins.yml` workflow collects all four platforms from the same engine commit, checks their identities and digests, and gates publication of the engine release. `package.py` and `release.py` remain available for standalone development checks, not independent platform distribution.

Only `package/` and the compiled target payload enter the installable archive. Tests, CMake, CI, source build tools and caches remain outside it. See the engine documentation [Platform Player releases](https://github.com/ChenlizheMe/Infernux/blob/main/docs/learn/platform-player-releases.md) for toolchain variables and batch commands.

## License

[MIT](LICENSE). Bundled dependencies retain their own licenses.
