#include "WebTextureUpload.h"

#include <function/resources/InxTexture/TexturePixels.h>

#include <limits>
#include <stdexcept>
#include <vector>

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

std::optional<TextureCpuData> DecodeBcTextureToRgba8(const TextureCpuData &source)
{
    if (source.dimension != TextureDimension::Texture2D || !TextureFormatIsBlockCompressed(source.format) ||
        !source.IsValid())
        return std::nullopt;

    TextureCpuData decoded;
    decoded.semantic = source.semantic;
    decoded.format = TextureFormatIsSrgb(source.format) ? TextureFormat::Rgba8Srgb : TextureFormat::Rgba8UNorm;
    decoded.bakeBasis = source.bakeBasis;
    decoded.valueMin = source.valueMin;
    decoded.valueMax = source.valueMax;
    decoded.mipLevels.reserve(source.mipLevels.size());

    std::vector<uint8_t> subsequentMip;
    try {
        for (const auto &mip : source.mipLevels) {
            const auto offset = decoded.bytes.size();
            const auto rowPitch = static_cast<uint64_t>(mip.width) * 4u;
            if (mip.width == 0 || mip.height == 0 ||
                rowPitch > ((std::numeric_limits<size_t>::max)() - offset) / mip.height)
                return std::nullopt;
            const auto byteSize = static_cast<size_t>(rowPitch * mip.height);
            auto &pixels = offset == 0 ? decoded.bytes : subsequentMip;
            pixels.resize(byteSize);
            texture_pixels::detail::DecodeBcMip(source, mip, pixels);
            if (offset != 0)
                decoded.bytes.insert(decoded.bytes.end(), pixels.begin(), pixels.end());
            decoded.mipLevels.push_back({mip.width, mip.height, 1, offset, byteSize, rowPitch, byteSize});
        }
    } catch (const std::invalid_argument &) {
        return std::nullopt;
    }
    return decoded;
}

} // namespace infernux::web
