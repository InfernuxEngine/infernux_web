# Infernux Web 平台插件

[English](README.md) · [引擎版本制品](https://github.com/ChenlizheMe/Infernux/releases)

为 Infernux 提供 `web-wasm32` 构建目标。构建宿主与目标：Windows/Linux x64 → WebAssembly。

![Infernux Web 构建流程](package/plugin_pages/media/overview.png)

## 安装与使用

通过引擎的官方插件目录安装。平台插件随引擎版本统一构建，必须匹配引擎热修号、Python ABI 和目标平台。例如引擎 `0.4.3-v8` 对应平台安装包 `0.4.3.post8`；本仓库的源码包元数据会在最终打包时替换为自动生成的版本标识。

下载服务按 `platform-plugins/<engine_release>/` 保存制品，GitHub 备用下载来自同一个引擎 Release。四个平台子仓库不再各自独立发布二进制。

版本匹配时直接使用预编译 Player。已安装插件的载荷不匹配时，引擎构建器尝试从当前引擎版本的源码重新编译，验证后覆盖该插件的 Player 载荷，后续直接复用。重建需要相应工具链，不会替换引擎 SDK。

## 维护

在引擎仓库使用 `scripts/release/build_platform_plugins.py` 构建或打包此平台。引擎 CI 使用同一源码提交生成四个平台插件，统一核对版本与摘要，全部成功后才能发布引擎。平台仓库自身的 CI 负责源码与打包检查。`package.py`、`release.py` 保留为独立开发检查工具。

安装包只包含 `package/` 和编译后的平台载荷。测试、CMake、CI、构建工具和缓存不会进入安装包。具体环境变量与批处理命令见引擎的[平台 Player 发布说明](https://github.com/ChenlizheMe/Infernux/blob/main/docs/learn/platform-player-releases.md)。

## 许可证

[MIT](LICENSE)。捆绑依赖保留各自的许可证。
