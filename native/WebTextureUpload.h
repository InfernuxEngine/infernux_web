#pragma once

#include <function/resources/InxTexture/InxTexture.h>
#include <optional>
#include <webgpu/webgpu_cpp.h>

namespace infernux::web
{
wgpu::TextureFormat ToWebTextureFormat(TextureFormat format);
// The baseline WebGPU device has no optional BC feature. Scene and UI use
// the same one-time compatibility conversion before uploading cooked texels.
std::optional<TextureCpuData> DecodeBcTextureToRgba8(const TextureCpuData &source);
} // namespace infernux::web
