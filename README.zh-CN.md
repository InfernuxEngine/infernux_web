# Infernux Web 平台插件

这是 [Infernux](https://github.com/ChenlizheMe/Infernux) 游戏引擎的官方网页构建插件。它把游戏导出为 WebAssembly，让引擎的 Python 玩法层直接在浏览器中运行，并通过 WebGPU 完成渲染。

[English](README.md) · [Infernux 引擎](https://github.com/ChenlizheMe/Infernux) · [插件模板](https://github.com/InfernuxEngine/infernux_plugin_template) · [发布制品](https://github.com/InfernuxEngine/infernux_web/releases)

![Infernux Web 导出流程](package/plugin_pages/media/overview.png)

## 插件提供什么

- 编辑器中的 `web-wasm32` 构建目标
- 预编译的 WebAssembly/JavaScript Player 与 CPython 数据
- 面向 Windows x64、Linux x64 构建机的 glslang 和 Tint 工具
- WebGPU 渲染、浏览器输入桥接与可定制的 HTML 外壳

| 包标识 | 版本 | 适配引擎 | 构建环境 | 目标平台 |
| --- | --- | --- | --- | --- |
| `infernux/platform-web` | 0.2.17 | Infernux >=0.4.3; payload 0.4.3 | Windows/Linux x64 | WebAssembly + WebGPU |

## 安装与导出

在 Infernux 中打开**插件**窗口，从官方列表选择 **Infernux Web Platform**，导入并启用即可。官方安装优先走 Infernux 分发服务，网络不可用时回退到 GitHub Releases；也可以手动导入 Releases 页面中的 `infernux.platform-web.inxpkg`。

在构建设置中选择 `web-wasm32` 后导出，并把整个输出目录发布到服务器。开发环境可以使用 localhost 的 HTTP 服务，正式部署应使用 HTTPS；不能直接通过 `file://` 打开页面。运行设备和浏览器必须支持 WebGPU。普通用户不需要安装 Emscripten，也不需要编译引擎。

如果需要修改网页外观，可把 `package/editor/infernux_web/templates/host/shell.html` 复制到项目的 `ProjectSettings/WebTemplate/shell.html`，保留运行时标记和 canvas 接入点，再调整页面结构与样式。

## 仓库说明

只有 `package/` 会作为插件发布。外层的原生源码、构建脚本、测试与 CI 供维护者使用。发布流程会把 Web Player 和 Shader 工具直接生成到 `package/`；推送与版本一致的 `v<version>` 标签后，GitHub Actions 自动发布 `.inxpkg` 和 release manifest。

## 许可证

[MIT](LICENSE)。随包提供的第三方组件继续遵守各自的许可证。
