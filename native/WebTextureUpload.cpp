#include "WebTextureUpload.h"

namespace infernux::web
{
wgpu::TextureFormat ToWebTextureFormat(TextureFormat format)
{
    switch (format) {
    case TextureFormat::Rgba8UNorm:
        return wgpu::TextureFormat::RGBA8Unorm;
    case TextureFormat::Rgba8Srgb:
        return wgpu::TextureFormat::RGBA8UnormSrgb;
    case TextureFormat::Rgba16Float:
        return wgpu::TextureFormat::RGBA16Float;
    case TextureFormat::Rgba32Float:
        return wgpu::TextureFormat::RGBA32Float;
    default:
        return wgpu::TextureFormat::Undefined;
    }
}

} // namespace infernux::web
