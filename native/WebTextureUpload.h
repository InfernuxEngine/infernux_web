#pragma once

#include <function/resources/InxTexture/TexturePixels.h>

#include <function/resources/InxTexture/InxTexture.h>
#include <optional>
#include <webgpu/webgpu_cpp.h>

namespace infernux::web
{
wgpu::TextureFormat ToWebTextureFormat(TextureFormat format);
// The baseline WebGPU device has no optional BC feature. Scene and UI use
// the same one-time compatibility conversion before uploading cooked texels.
using texture_pixels::DecodeBcTextureToRgba8;
} // namespace infernux::web
