# Web 平台

![构建流程](media/overview.png)

通过编译为 WebAssembly 的 CPython 3.13 和 WebGPU 渲染构建浏览器 Player。插件包含预编译 WASM/JavaScript、CPython 标准库和 Windows/Linux 原生着色器工具，并提供浏览器宿主、输入桥接和项目 HTML 模板。

## 构建前准备

Infernux 0.4.3，Windows x64 或 Linux x64 编辑器，以及支持 WebGPU 的浏览器。普通导出不需要引擎源码、Git 子模块、CMake、WSL 或 Emscripten。

## 插件载荷

`editor/infernux_web/player/` 存放预编译运行时和 CPython 数据；`tools/windows-x64/`、`tools/linux-x64/` 存放 glslang 和 Tint。安装插件即可使用，无需另外下载 Web SDK。

导出只处理项目内容、将 GLSL 转为 WGSL、封装 `.inxpkg` 并生成网页，不重新编译引擎。游戏内容进入二进制包，浏览器加载后在内存中挂载，HTTP 目录不暴露 Assets、Library 或项目源码树。

## 导出与访问

构建设置中选择 web-wasm32。发布整个输出目录，包括 infernux-player.html、JavaScript、WebAssembly 和游戏数据。本机测试使用 localhost HTTP，部署使用 HTTPS，不要通过 file:// 打开页面。浏览器权限和设备能力必须允许 WebGPU。

## 项目 Web 模板

将 package/editor/infernux_web/templates/host/shell.html 复制为项目 ProjectSettings/WebTemplate/shell.html。可以调整样式、元数据和外围页面，但须保留运行时标记与画布集成。其他模板资源导出到 web-template/，保持相对目录关系。

模板目录一旦存在就必须包含 shell.html。重新构建会替换生成内容，应编辑项目模板源文件而不是导出文件。游戏 UI 使用 Infernux Screen UI，HTML 是页面宿主，不是游戏的 DOM 渲染层。

## 排错

若提示插件载荷缺失或引擎版本不兼容，请显式选择匹配版本的完整 `.inxpkg`，不要安装 GitHub 源码归档。浏览器无法初始化 WebGPU 时，检查浏览器、设备支持和 GPU 设置；不存在 WebGL 渲染兜底。
