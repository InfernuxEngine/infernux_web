#include "WebTextureUpload.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>

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

namespace
{
std::array<uint8_t, 4> DecodeRgb565(uint16_t packed)
{
    const uint8_t red = static_cast<uint8_t>((packed >> 11u) & 0x1fu);
    const uint8_t green = static_cast<uint8_t>((packed >> 5u) & 0x3fu);
    const uint8_t blue = static_cast<uint8_t>(packed & 0x1fu);
    return {static_cast<uint8_t>((red << 3u) | (red >> 2u)), static_cast<uint8_t>((green << 2u) | (green >> 4u)),
            static_cast<uint8_t>((blue << 3u) | (blue >> 2u)), 255u};
}

std::array<std::array<uint8_t, 4>, 4> DecodeBcColorTable(const uint8_t *block, bool allowTransparent)
{
    const uint16_t endpoint0 = static_cast<uint16_t>(block[0]) | static_cast<uint16_t>(block[1]) << 8u;
    const uint16_t endpoint1 = static_cast<uint16_t>(block[2]) | static_cast<uint16_t>(block[3]) << 8u;
    std::array<std::array<uint8_t, 4>, 4> table{};
    table[0] = DecodeRgb565(endpoint0);
    table[1] = DecodeRgb565(endpoint1);
    if (!allowTransparent || endpoint0 > endpoint1) {
        for (uint32_t channel = 0; channel < 3; ++channel) {
            table[2][channel] = static_cast<uint8_t>((2u * table[0][channel] + table[1][channel]) / 3u);
            table[3][channel] = static_cast<uint8_t>((table[0][channel] + 2u * table[1][channel]) / 3u);
        }
        table[2][3] = 255u;
        table[3][3] = 255u;
    } else {
        for (uint32_t channel = 0; channel < 3; ++channel)
            table[2][channel] = static_cast<uint8_t>((table[0][channel] + table[1][channel]) / 2u);
        table[2][3] = 255u;
        table[3] = {0u, 0u, 0u, 0u};
    }
    return table;
}

std::array<uint8_t, 8> DecodeBc3AlphaTable(uint8_t endpoint0, uint8_t endpoint1)
{
    std::array<uint8_t, 8> table{endpoint0, endpoint1};
    if (endpoint0 > endpoint1) {
        for (uint32_t index = 1; index <= 6; ++index)
            table[index + 1] = static_cast<uint8_t>(((7u - index) * endpoint0 + index * endpoint1) / 7u);
    } else {
        for (uint32_t index = 1; index <= 4; ++index)
            table[index + 1] = static_cast<uint8_t>(((5u - index) * endpoint0 + index * endpoint1) / 5u);
        table[6] = 0u;
        table[7] = 255u;
    }
    return table;
}

uint64_t DecodeBcScalarIndices(const uint8_t *block)
{
    uint64_t indices = 0;
    for (uint32_t byte = 0; byte < 6; ++byte)
        indices |= static_cast<uint64_t>(block[2 + byte]) << (8u * byte);
    return indices;
}

} // namespace

std::optional<TextureCpuData> DecodeBcTextureToRgba8(const TextureCpuData &source)
{
    const bool bc1 = source.format == TextureFormat::BC1RgbaUNorm || source.format == TextureFormat::BC1RgbaSrgb;
    const bool bc3 = source.format == TextureFormat::BC3UNorm || source.format == TextureFormat::BC3Srgb;
    const bool bc4 = source.format == TextureFormat::BC4UNorm;
    const bool bc5 = source.format == TextureFormat::BC5UNorm;
    if ((!bc1 && !bc3 && !bc4 && !bc5) || source.dimension != TextureDimension::Texture2D || source.mipLevels.empty())
        return std::nullopt;

    TextureCpuData decoded;
    decoded.dimension = TextureDimension::Texture2D;
    decoded.semantic = source.semantic;
    decoded.format = TextureFormatIsSrgb(source.format) ? TextureFormat::Rgba8Srgb : TextureFormat::Rgba8UNorm;

    const uint32_t blockBytes = (bc1 || bc4) ? 8u : 16u;
    for (const TextureMipLevel &mip : source.mipLevels) {
        const uint32_t blockColumns = std::max(1u, (mip.width + 3u) / 4u);
        const uint32_t blockRows = std::max(1u, (mip.height + 3u) / 4u);
        const uint64_t requiredRowPitch = static_cast<uint64_t>(blockColumns) * blockBytes;
        if (mip.width == 0 || mip.height == 0 || mip.depth != 1 || mip.rowPitch < requiredRowPitch ||
            mip.byteOffset > source.bytes.size() || mip.byteSize > source.bytes.size() - mip.byteOffset)
            return std::nullopt;
        const uint64_t requiredBytes = static_cast<uint64_t>(blockRows - 1u) * mip.rowPitch + requiredRowPitch;
        if (mip.byteSize < requiredBytes)
            return std::nullopt;

        const uint64_t rowPitch = static_cast<uint64_t>(mip.width) * 4u;
        const uint64_t byteSize = rowPitch * mip.height;
        if (byteSize > std::numeric_limits<size_t>::max() - decoded.bytes.size())
            return std::nullopt;
        TextureMipLevel decodedMip;
        decodedMip.width = mip.width;
        decodedMip.height = mip.height;
        decodedMip.depth = 1;
        decodedMip.byteOffset = decoded.bytes.size();
        decodedMip.byteSize = byteSize;
        decodedMip.rowPitch = rowPitch;
        decodedMip.slicePitch = byteSize;
        decoded.mipLevels.push_back(decodedMip);
        decoded.bytes.resize(decoded.bytes.size() + static_cast<size_t>(byteSize));

        for (uint32_t blockY = 0; blockY < blockRows; ++blockY) {
            for (uint32_t blockX = 0; blockX < blockColumns; ++blockX) {
                const uint8_t *block = source.bytes.data() + mip.byteOffset +
                                       static_cast<uint64_t>(blockY) * mip.rowPitch +
                                       static_cast<uint64_t>(blockX) * blockBytes;
                std::array<std::array<uint8_t, 4>, 4> colors{};
                uint32_t colorIndices = 0;
                std::array<uint8_t, 8> alphaOrRed{};
                std::array<uint8_t, 8> green{};
                uint64_t alphaOrRedIndices = 0;
                uint64_t greenIndices = 0;
                if (bc1 || bc3) {
                    const uint8_t *colorBlock = block + (bc3 ? 8u : 0u);
                    colors = DecodeBcColorTable(colorBlock, bc1);
                    colorIndices = static_cast<uint32_t>(colorBlock[4]) | static_cast<uint32_t>(colorBlock[5]) << 8u |
                                   static_cast<uint32_t>(colorBlock[6]) << 16u |
                                   static_cast<uint32_t>(colorBlock[7]) << 24u;
                    if (bc3) {
                        alphaOrRed = DecodeBc3AlphaTable(block[0], block[1]);
                        alphaOrRedIndices = DecodeBcScalarIndices(block);
                    }
                } else {
                    alphaOrRed = DecodeBc3AlphaTable(block[0], block[1]);
                    alphaOrRedIndices = DecodeBcScalarIndices(block);
                    if (bc5) {
                        green = DecodeBc3AlphaTable(block[8], block[9]);
                        greenIndices = DecodeBcScalarIndices(block + 8u);
                    }
                }

                for (uint32_t pixel = 0; pixel < 16; ++pixel) {
                    const uint32_t x = blockX * 4u + pixel % 4u;
                    const uint32_t y = blockY * 4u + pixel / 4u;
                    if (x >= mip.width || y >= mip.height)
                        continue;
                    std::array<uint8_t, 4> rgba{};
                    if (bc1 || bc3) {
                        rgba = colors[(colorIndices >> (pixel * 2u)) & 0x3u];
                        if (bc3)
                            rgba[3] = alphaOrRed[(alphaOrRedIndices >> (pixel * 3u)) & 0x7u];
                    } else {
                        const uint8_t red = alphaOrRed[(alphaOrRedIndices >> (pixel * 3u)) & 0x7u];
                        const uint8_t greenValue = bc5 ? green[(greenIndices >> (pixel * 3u)) & 0x7u] : red;
                        rgba = {red, greenValue, static_cast<uint8_t>(bc5 ? 255u : red), 255u};
                    }
                    std::memcpy(decoded.bytes.data() + decodedMip.byteOffset + static_cast<uint64_t>(y) * rowPitch +
                                    x * 4u,
                                rgba.data(), rgba.size());
                }
            }
        }
    }
    return decoded;
}

} // namespace infernux::web
