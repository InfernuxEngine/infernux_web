#include "WebSceneRenderer.h"
#include "WebGpuRhiDevice.h"
#include "WebTextureUpload.h"

#include <core/types/ColorSpace.h>
#include <function/renderer/RendererParameterBlock.h>
#include <function/resources/AssetRegistry/AssetRegistry.h>
#include <function/resources/InxMaterial/InxMaterial.h>
#include <function/resources/InxTexture/InxTexture.h>
#include <function/resources/ShaderAsset/SlangSceneBoundsModule.h>
#include <function/resources/ShaderAsset/SlangSceneModule.h>
#include <function/scene/Camera.h>
#include <function/scene/GameObject.h>
#include <function/scene/Light.h>
#include <function/scene/Scene.h>
#include <function/scene/SceneManager.h>
#include <function/scene/Transform.h>

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <iterator>
#include <limits>
#include <stdexcept>

namespace infernux::web
{
namespace
{

constexpr uint32_t kLineVertexMarker = 0x4C494E45u;
constexpr uint32_t kMaxPunctualLights = 8u;

template <typename Cache> void RetireUnused(Cache &cache, uint64_t frame)
{
    for (auto entry = cache.begin(); entry != cache.end();)
        entry = entry->second.lastFrame == frame ? std::next(entry) : cache.erase(entry);
}

const MaterialProperty *ResolvedMaterialProperty(const std::shared_ptr<InxMaterial> &material,
                                                 const std::shared_ptr<const RendererParameterBlock> &parameters,
                                                 const char *name)
{
    if (parameters) {
        const auto found = parameters->properties.find(name);
        if (found != parameters->properties.end())
            return &found->second;
    }
    return material ? material->GetProperty(name) : nullptr;
}

glm::vec4 MaterialColor(const std::shared_ptr<InxMaterial> &material,
                        const std::shared_ptr<const RendererParameterBlock> &parameters)
{
    const MaterialProperty *property = ResolvedMaterialProperty(material, parameters, "baseColor");
    if (!property)
        return glm::vec4(1.0f);
    if (property->type == MaterialPropertyType::Color || property->type == MaterialPropertyType::Float4) {
        if (const auto *value = std::get_if<glm::vec4>(&property->value))
            return *value;
    }
    if (property->type == MaterialPropertyType::Float3) {
        if (const auto *value = std::get_if<glm::vec3>(&property->value))
            return glm::vec4(*value, 1.0f);
    }
    return glm::vec4(1.0f);
}

float MaterialFloat(const std::shared_ptr<InxMaterial> &material,
                    const std::shared_ptr<const RendererParameterBlock> &parameters, const char *name, float fallback)
{
    const MaterialProperty *property = ResolvedMaterialProperty(material, parameters, name);
    if (!property || property->type != MaterialPropertyType::Float)
        return fallback;
    if (const auto *value = std::get_if<float>(&property->value))
        return *value;
    return fallback;
}

glm::vec4 MaterialVector(const std::shared_ptr<InxMaterial> &material,
                         const std::shared_ptr<const RendererParameterBlock> &parameters, const char *name,
                         const glm::vec4 &fallback)
{
    const MaterialProperty *property = ResolvedMaterialProperty(material, parameters, name);
    if (!property)
        return fallback;
    if (const auto *value = std::get_if<glm::vec4>(&property->value))
        return *value;
    if (const auto *value = std::get_if<glm::vec3>(&property->value))
        return glm::vec4(*value, fallback.w);
    return fallback;
}

struct MaterialTextureBinding
{
    std::string guid;
    MaterialTextureSampler sampler;
};

template <size_t Size>
MaterialTextureBinding MaterialTexture(const std::shared_ptr<InxMaterial> &material,
                                       const std::shared_ptr<const RendererParameterBlock> &parameters,
                                       const std::array<const char *, Size> &names, const char *fallback)
{
    for (const char *name : names) {
        const MaterialProperty *property = ResolvedMaterialProperty(material, parameters, name);
        if (!property || property->type != MaterialPropertyType::Texture2D)
            continue;
        if (const auto *guid = std::get_if<std::string>(&property->value); guid && !guid->empty()) {
            const MaterialTextureSampler *sampler = material ? material->GetTextureSampler(name) : nullptr;
            return {*guid, sampler ? *sampler : MaterialTextureSampler{}};
        }
    }
    return {fallback, {}};
}

std::array<MaterialTextureBinding, 6> MaterialTextures(const std::shared_ptr<InxMaterial> &material,
                                                       const std::shared_ptr<const RendererParameterBlock> &parameters)
{
    constexpr std::array<const char *, 5> baseNames = {"baseColorTexture", "albedoMap", "albedoTexture", "mainTexture",
                                                       "texSampler"};
    constexpr std::array<const char *, 3> metallicNames = {"metallicMap", "metalnessMap", "metallicTexture"};
    constexpr std::array<const char *, 2> smoothnessNames = {"smoothnessMap", "smoothnessTexture"};
    constexpr std::array<const char *, 3> aoNames = {"aoMap", "occlusionMap", "ambientOcclusionMap"};
    constexpr std::array<const char *, 3> normalNames = {"normalMap", "normalTexture", "bumpMap"};
    constexpr std::array<const char *, 2> emissionNames = {"emissionMap", "emissiveMap"};
    return {MaterialTexture(material, parameters, baseNames, "white"),
            MaterialTexture(material, parameters, metallicNames, "white"),
            MaterialTexture(material, parameters, smoothnessNames, "white"),
            MaterialTexture(material, parameters, aoNames, "white"),
            MaterialTexture(material, parameters, normalNames, "normal"),
            MaterialTexture(material, parameters, emissionNames, "white")};
}

int MaterialInt(const std::shared_ptr<InxMaterial> &material,
                const std::shared_ptr<const RendererParameterBlock> &parameters, const char *name, int fallback)
{
    const MaterialProperty *property = ResolvedMaterialProperty(material, parameters, name);
    if (!property || property->type != MaterialPropertyType::Int)
        return fallback;
    if (const auto *value = std::get_if<int>(&property->value))
        return *value;
    return fallback;
}

std::string SamplerIdentity(const MaterialTextureSampler &sampler)
{
    return std::to_string(static_cast<uint32_t>(sampler.minFilter)) + ":" +
           std::to_string(static_cast<uint32_t>(sampler.magFilter)) + ":" +
           std::to_string(static_cast<uint32_t>(sampler.mipFilter)) + ":" +
           std::to_string(static_cast<uint32_t>(sampler.addressU)) + ":" +
           std::to_string(static_cast<uint32_t>(sampler.addressV)) + ":" +
           std::to_string(static_cast<uint32_t>(sampler.addressW));
}

bool MaterialIsUnlit(const std::shared_ptr<InxMaterial> &material)
{
    if (!material)
        return false;
    std::string shader = material->GetFragShaderName();
    std::transform(shader.begin(), shader.end(), shader.begin(),
                   [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    return shader.find("unlit") != std::string::npos;
}

bool MaterialIsToon(const std::shared_ptr<InxMaterial> &material)
{
    if (!material)
        return false;
    std::string shader = material->GetFragShaderName();
    std::transform(shader.begin(), shader.end(), shader.begin(),
                   [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    return shader.find("toon") != std::string::npos;
}

bool Finite(const glm::vec3 &value)
{
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

uint64_t GrowCapacity(uint64_t required)
{
    uint64_t capacity = 4096;
    while (capacity < required && capacity <= std::numeric_limits<uint64_t>::max() / 2)
        capacity *= 2;
    return std::max(capacity, required);
}

glm::mat4 ToWebClipSpace(const glm::mat4 &vulkanViewProjection)
{
    // Camera publishes the Vulkan projection used by the native renderer,
    // including its Y inversion. WebGPU performs its own framebuffer-space Y
    // mapping, so carrying that inversion across would present the whole game
    // upside down. Keep the correction at this backend boundary.
    glm::mat4 correction(1.0f);
    correction[1][1] = -1.0f;
    return correction * vulkanViewProjection;
}

} // namespace

bool WebSceneRenderer::Initialize(WebGpuRhiDevice &device, wgpu::TextureFormat colorFormat, uint32_t sceneSampleCount)
{
    static_assert(sizeof(Vertex) == 100);
    static_assert(sizeof(WebDrawData) == 160);
    static_assert(sizeof(WorldBounds) == 32);
    static_assert(sizeof(CameraData) == 880);
    m_shaders.clear();
    m_geometry.clear();
    m_geometryBindings.clear();
    m_drawRanges.clear();
    m_rhi = &device;
    m_device = device.NativeDevice();
    m_queue = device.NativeQueue();
    m_colorFormat = colorFormat;
    m_sceneSampleCount = sceneSampleCount;
    if (!m_device || !m_queue || colorFormat == wgpu::TextureFormat::Undefined ||
        (sceneSampleCount != 1 && sceneSampleCount != 4))
        return false;

    auto buffer = [&](uint64_t bytes, wgpu::BufferUsage usage) {
        wgpu::BufferDescriptor descriptor;
        descriptor.size = bytes;
        descriptor.usage = usage | wgpu::BufferUsage::CopyDst;
        return m_device.CreateBuffer(&descriptor);
    };
    m_cameraBuffer = buffer(sizeof(CameraData), wgpu::BufferUsage::Uniform);
    m_shadowMatrixBuffer = buffer(sizeof(glm::mat4), wgpu::BufferUsage::Storage);
    m_shadowFitParametersBuffer = buffer(sizeof(ShadowFitParameters), wgpu::BufferUsage::Uniform);
    const glm::mat4 identity(1.0f);
    m_queue.WriteBuffer(m_shadowMatrixBuffer, 0, &identity, sizeof(identity));
    m_drawDataCapacity = m_modelCapacity = m_boneCapacity = m_boundsCapacity = m_boundsTilesCapacity = 0;
    m_drawDataBuffer = {};
    m_modelBuffer = {};
    m_boneBuffer = {};
    m_boundsBuffer = {};
    m_boundsTilesBuffer = {};
    if (!m_cameraBuffer || !m_shadowMatrixBuffer || !m_shadowFitParametersBuffer ||
        !EnsureBuffer(m_drawDataBuffer, m_drawDataCapacity, sizeof(WebDrawData), wgpu::BufferUsage::Storage) ||
        !EnsureBuffer(m_modelBuffer, m_modelCapacity, sizeof(glm::mat4), wgpu::BufferUsage::Storage) ||
        !EnsureBuffer(m_boneBuffer, m_boneCapacity, sizeof(glm::mat4), wgpu::BufferUsage::Storage) ||
        !EnsureBuffer(m_boundsBuffer, m_boundsCapacity, sizeof(WorldBounds), wgpu::BufferUsage::Storage) ||
        !EnsureBuffer(m_boundsTilesBuffer, m_boundsTilesCapacity, sizeof(WorldBounds), wgpu::BufferUsage::Storage))
        return false;

    std::array<wgpu::BindGroupLayoutEntry, 6> cameraEntries{};
    cameraEntries[0].binding = 0;
    cameraEntries[0].visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment;
    cameraEntries[0].buffer.type = wgpu::BufferBindingType::Uniform;
    cameraEntries[0].buffer.minBindingSize = sizeof(CameraData);
    cameraEntries[1].binding = 1;
    cameraEntries[1].visibility = wgpu::ShaderStage::Fragment;
    cameraEntries[1].texture.sampleType = wgpu::TextureSampleType::Depth;
    cameraEntries[1].texture.viewDimension = wgpu::TextureViewDimension::e2D;
    cameraEntries[2].binding = 2;
    cameraEntries[2].visibility = wgpu::ShaderStage::Fragment;
    cameraEntries[2].sampler.type = wgpu::SamplerBindingType::Comparison;
    for (size_t i = 3; i < cameraEntries.size(); ++i) {
        cameraEntries[i].binding = static_cast<uint32_t>(i);
        cameraEntries[i].visibility = wgpu::ShaderStage::Vertex;
        cameraEntries[i].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
    }
    cameraEntries[3].visibility |= wgpu::ShaderStage::Fragment;
    cameraEntries[3].buffer.minBindingSize = sizeof(WebDrawData);
    cameraEntries[4].buffer.minBindingSize = sizeof(glm::mat4);
    cameraEntries[5].buffer.minBindingSize = sizeof(WorldBounds);
    wgpu::BindGroupLayoutDescriptor layoutDescriptor;
    layoutDescriptor.entryCount = cameraEntries.size();
    layoutDescriptor.entries = cameraEntries.data();
    m_cameraLayout = m_device.CreateBindGroupLayout(&layoutDescriptor);

    std::array<wgpu::BindGroupLayoutEntry, 4> shadowEntries{cameraEntries[0], cameraEntries[3], cameraEntries[4],
                                                            cameraEntries[5]};
    for (auto &entry : shadowEntries)
        entry.visibility = wgpu::ShaderStage::Vertex;
    layoutDescriptor.entryCount = shadowEntries.size();
    layoutDescriptor.entries = shadowEntries.data();
    m_shadowCameraLayout = m_device.CreateBindGroupLayout(&layoutDescriptor);

    std::array<wgpu::BindGroupLayoutEntry, 2> geometryEntries{};
    for (size_t i = 0; i < geometryEntries.size(); ++i) {
        geometryEntries[i].binding = static_cast<uint32_t>(i);
        geometryEntries[i].visibility = wgpu::ShaderStage::Vertex;
        geometryEntries[i].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
        geometryEntries[i].buffer.minBindingSize = sizeof(glm::mat4);
    }
    layoutDescriptor.entryCount = geometryEntries.size();
    layoutDescriptor.entries = geometryEntries.data();
    m_geometryLayout = m_device.CreateBindGroupLayout(&layoutDescriptor);
    return m_cameraLayout && m_shadowCameraLayout && m_geometryLayout && CreateShadowResources() &&
           CreateMaterialTextureResources() && CreateCameraBindGroups() && CreateShadowFitPipeline() &&
           CreatePipelines();
}

bool WebSceneRenderer::CreateCameraBindGroups()
{
    std::array<wgpu::BindGroupEntry, 6> bindings{};
    bindings[0].binding = 0;
    bindings[0].buffer = m_cameraBuffer;
    bindings[0].size = sizeof(CameraData);
    bindings[1].binding = 1;
    bindings[1].textureView = m_shadowView;
    bindings[2].binding = 2;
    bindings[2].sampler = m_shadowSampler;
    bindings[3].binding = 3;
    bindings[3].buffer = m_drawDataBuffer;
    bindings[3].size = m_drawDataCapacity;
    bindings[4].binding = 4;
    bindings[4].buffer = m_shadowMatrixBuffer;
    bindings[4].size = sizeof(glm::mat4);
    bindings[5].binding = 5;
    bindings[5].buffer = m_boundsBuffer;
    bindings[5].size = m_boundsCapacity;
    wgpu::BindGroupDescriptor descriptor;
    descriptor.layout = m_cameraLayout;
    descriptor.entryCount = bindings.size();
    descriptor.entries = bindings.data();
    m_cameraGroup = m_device.CreateBindGroup(&descriptor);
    const std::array<wgpu::BindGroupEntry, 4> shadowBindings{bindings[0], bindings[3], bindings[4], bindings[5]};
    descriptor.layout = m_shadowCameraLayout;
    descriptor.entryCount = shadowBindings.size();
    descriptor.entries = shadowBindings.data();
    m_shadowCameraGroup = m_device.CreateBindGroup(&descriptor);
    return m_cameraGroup && m_shadowCameraGroup;
}

bool WebSceneRenderer::CreateShadowResources()
{
    wgpu::TextureDescriptor textureDescriptor;
    textureDescriptor.dimension = wgpu::TextureDimension::e2D;
    textureDescriptor.size = {m_shadowResolution, m_shadowResolution, 1};
    textureDescriptor.format = wgpu::TextureFormat::Depth32Float;
    textureDescriptor.mipLevelCount = 1;
    textureDescriptor.sampleCount = 1;
    textureDescriptor.usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding;
    m_shadowTexture = m_device.CreateTexture(&textureDescriptor);
    m_shadowView = m_shadowTexture ? m_shadowTexture.CreateView() : wgpu::TextureView{};

    wgpu::SamplerDescriptor samplerDescriptor;
    samplerDescriptor.addressModeU = wgpu::AddressMode::ClampToEdge;
    samplerDescriptor.addressModeV = wgpu::AddressMode::ClampToEdge;
    samplerDescriptor.minFilter = wgpu::FilterMode::Linear;
    samplerDescriptor.magFilter = wgpu::FilterMode::Linear;
    samplerDescriptor.compare = wgpu::CompareFunction::GreaterEqual;
    m_shadowSampler = m_device.CreateSampler(&samplerDescriptor);
    return m_shadowView && m_shadowSampler;
}

bool WebSceneRenderer::CreateMaterialTextureResources()
{
    std::array<wgpu::BindGroupLayoutEntry, 12> entries{};
    for (uint32_t textureIndex = 0; textureIndex < 6; ++textureIndex) {
        const uint32_t samplerBinding = textureIndex * 2u;
        const uint32_t textureBinding = samplerBinding + 1u;
        entries[samplerBinding].binding = samplerBinding;
        entries[samplerBinding].visibility = wgpu::ShaderStage::Fragment;
        entries[samplerBinding].sampler.type = wgpu::SamplerBindingType::Filtering;
        entries[textureBinding].binding = textureBinding;
        entries[textureBinding].visibility = wgpu::ShaderStage::Fragment;
        entries[textureBinding].texture.sampleType = wgpu::TextureSampleType::Float;
        entries[textureBinding].texture.viewDimension = wgpu::TextureViewDimension::e2D;
    }
    wgpu::BindGroupLayoutDescriptor layoutDescriptor;
    layoutDescriptor.entryCount = entries.size();
    layoutDescriptor.entries = entries.data();
    m_materialTextureLayout = m_device.CreateBindGroupLayout(&layoutDescriptor);
    if (!m_materialTextureLayout)
        return false;

    const auto createSolid = [this](const std::array<uint8_t, 4> &pixel, bool srgb) {
        TextureCpuData data;
        data.dimension = TextureDimension::Texture2D;
        data.semantic = TextureSemantic::Color;
        data.format = srgb ? TextureFormat::Rgba8Srgb : TextureFormat::Rgba8UNorm;
        data.mipLevels.push_back(TextureMipLevel{1, 1, 1, 0, 4, 4, 4});
        data.bytes.assign(pixel.begin(), pixel.end());
        return UploadMaterialTexture(data, "linear", "repeat", 1);
    };
    m_whiteTexture = createSolid({255, 255, 255, 255}, true);
    m_blackTexture = createSolid({0, 0, 0, 255}, true);
    m_normalTexture = createSolid({128, 128, 255, 255}, false);
    const std::array<GPUTexture, 6> defaults = {m_whiteTexture, m_whiteTexture,  m_whiteTexture,
                                                m_whiteTexture, m_normalTexture, m_whiteTexture};
    m_defaultMaterialTextureGroup = CreateMaterialTextureGroup(defaults);
    return m_whiteTexture.view && m_blackTexture.view && m_normalTexture.view && m_defaultMaterialTextureGroup;
}

wgpu::BindGroup WebSceneRenderer::CreateMaterialTextureGroup(const std::array<GPUTexture, 6> &textures)
{
    std::array<wgpu::BindGroupEntry, 12> entries{};
    for (uint32_t textureIndex = 0; textureIndex < textures.size(); ++textureIndex) {
        if (!textures[textureIndex].view || !textures[textureIndex].sampler)
            return {};
        const uint32_t samplerBinding = textureIndex * 2u;
        const uint32_t textureBinding = samplerBinding + 1u;
        entries[samplerBinding].binding = samplerBinding;
        entries[samplerBinding].sampler = textures[textureIndex].sampler;
        entries[textureBinding].binding = textureBinding;
        entries[textureBinding].textureView = textures[textureIndex].view;
    }
    wgpu::BindGroupDescriptor descriptor;
    descriptor.layout = m_materialTextureLayout;
    descriptor.entryCount = entries.size();
    descriptor.entries = entries.data();
    return m_device.CreateBindGroup(&descriptor);
}

WebSceneRenderer::GPUTexture WebSceneRenderer::UploadMaterialTexture(const TextureCpuData &texture,
                                                                     const std::string &filterMode,
                                                                     const std::string &wrapMode, int anisoLevel,
                                                                     const MaterialTextureSampler &sampler)
{
    GPUTexture gpu;
    if (!m_device || !m_queue || texture.dimension != TextureDimension::Texture2D || !texture.IsValid())
        return gpu;
    std::optional<TextureCpuData> decoded;
    const TextureCpuData *upload = &texture;
    if (TextureFormatIsBlockCompressed(texture.format)) {
        decoded = DecodeBcTextureToRgba8(texture);
        if (!decoded)
            return gpu;
        upload = &*decoded;
    }
    const wgpu::TextureFormat format = ToWebTextureFormat(upload->format);
    const TextureMipLevel &baseMip = upload->mipLevels.front();
    if (format == wgpu::TextureFormat::Undefined || baseMip.width == 0 || baseMip.height == 0 ||
        upload->mipLevels.size() > std::numeric_limits<uint32_t>::max())
        return gpu;
    for (const TextureMipLevel &mip : upload->mipLevels) {
        if (mip.width == 0 || mip.height == 0 || mip.depth != 1 || mip.byteOffset > upload->bytes.size() ||
            mip.byteSize > upload->bytes.size() - mip.byteOffset || mip.rowPitch > std::numeric_limits<uint32_t>::max())
            return {};
    }

    wgpu::TextureDescriptor descriptor;
    descriptor.dimension = wgpu::TextureDimension::e2D;
    descriptor.size = {baseMip.width, baseMip.height, 1};
    descriptor.format = format;
    descriptor.mipLevelCount = static_cast<uint32_t>(upload->mipLevels.size());
    descriptor.sampleCount = 1;
    descriptor.usage = wgpu::TextureUsage::CopyDst | wgpu::TextureUsage::TextureBinding;
    gpu.texture = m_device.CreateTexture(&descriptor);
    gpu.view = gpu.texture ? gpu.texture.CreateView() : wgpu::TextureView{};
    wgpu::SamplerDescriptor samplerDescriptor;
    const wgpu::AddressMode addressMode = wrapMode == "clamp"    ? wgpu::AddressMode::ClampToEdge
                                          : wrapMode == "mirror" ? wgpu::AddressMode::MirrorRepeat
                                                                 : wgpu::AddressMode::Repeat;
    const auto address = [addressMode](MaterialSamplerAddress value) {
        switch (value) {
        case MaterialSamplerAddress::Repeat:
            return wgpu::AddressMode::Repeat;
        case MaterialSamplerAddress::Clamp:
            return wgpu::AddressMode::ClampToEdge;
        case MaterialSamplerAddress::Mirror:
            return wgpu::AddressMode::MirrorRepeat;
        case MaterialSamplerAddress::Inherit:
            return addressMode;
        }
        return addressMode;
    };
    samplerDescriptor.addressModeU = address(sampler.addressU);
    samplerDescriptor.addressModeV = address(sampler.addressV);
    samplerDescriptor.addressModeW = address(sampler.addressW);
    const bool point = filterMode == "point";
    const auto filter = [point](MaterialSamplerFilter value) {
        switch (value) {
        case MaterialSamplerFilter::Nearest:
            return wgpu::FilterMode::Nearest;
        case MaterialSamplerFilter::Linear:
            return wgpu::FilterMode::Linear;
        case MaterialSamplerFilter::Inherit:
            return point ? wgpu::FilterMode::Nearest : wgpu::FilterMode::Linear;
        }
        return wgpu::FilterMode::Nearest;
    };
    const uint16_t assetAnisotropy =
        point ? 1u : static_cast<uint16_t>(std::clamp(anisoLevel < 0 ? 16 : anisoLevel, 1, 16));
    samplerDescriptor.minFilter = filter(sampler.minFilter);
    samplerDescriptor.magFilter = filter(sampler.magFilter);
    const bool assetMipLinear = filterMode == "trilinear" || assetAnisotropy > 1u;
    switch (sampler.mipFilter) {
    case MaterialSamplerFilter::Nearest:
        samplerDescriptor.mipmapFilter = wgpu::MipmapFilterMode::Nearest;
        break;
    case MaterialSamplerFilter::Linear:
        samplerDescriptor.mipmapFilter = wgpu::MipmapFilterMode::Linear;
        break;
    case MaterialSamplerFilter::Inherit:
        samplerDescriptor.mipmapFilter =
            assetMipLinear ? wgpu::MipmapFilterMode::Linear : wgpu::MipmapFilterMode::Nearest;
        break;
    }
    const bool fullyLinear = samplerDescriptor.minFilter == wgpu::FilterMode::Linear &&
                             samplerDescriptor.magFilter == wgpu::FilterMode::Linear &&
                             samplerDescriptor.mipmapFilter == wgpu::MipmapFilterMode::Linear;
    const uint16_t anisotropy = fullyLinear ? assetAnisotropy : 1u;
    samplerDescriptor.lodMaxClamp = static_cast<float>(upload->mipLevels.size() - 1u);
    samplerDescriptor.maxAnisotropy = anisotropy;
    gpu.sampler = m_device.CreateSampler(&samplerDescriptor);
    if (!gpu.texture || !gpu.view || !gpu.sampler)
        return {};

    for (uint32_t mipLevel = 0; mipLevel < upload->mipLevels.size(); ++mipLevel) {
        const TextureMipLevel &mip = upload->mipLevels[mipLevel];
        wgpu::TexelCopyTextureInfo destination;
        destination.texture = gpu.texture;
        destination.mipLevel = mipLevel;
        wgpu::TexelCopyBufferLayout layout;
        layout.bytesPerRow = static_cast<uint32_t>(mip.rowPitch);
        layout.rowsPerImage = mip.height;
        const wgpu::Extent3D extent{mip.width, mip.height, 1};
        m_queue.WriteTexture(&destination, upload->bytes.data() + mip.byteOffset, static_cast<size_t>(mip.byteSize),
                             &layout, &extent);
    }
    gpu.mipLevels = static_cast<uint32_t>(upload->mipLevels.size());
    return gpu;
}

WebSceneRenderer::SampledTexture WebSceneRenderer::ResolveSampledTexture(const std::string &guid)
{
    const auto texture = ResolveMaterialTexture(guid, {}, {});
    return {texture.view, texture.sampler};
}

WebSceneRenderer::GPUTexture WebSceneRenderer::ResolveMaterialTexture(const std::string &guid,
                                                                      const MaterialTextureSampler &sampler,
                                                                      const GPUTexture &fallback)
{
    if (guid.empty() || guid == "white")
        return m_whiteTexture;
    if (guid == "normal")
        return m_normalTexture;
    if (guid == "black")
        return m_blackTexture;
    const std::string samplerIdentity = SamplerIdentity(sampler);
    const std::string textureIdentity = guid + '\x1e' + samplerIdentity;
    MaterialTextureState &state = m_materialTextures[textureIdentity];
    try {
        const auto status = state.load.Poll(guid);
        if (status == WebTextureAssetLoad::Result::Missing && state.gpu.view) {
            state.gpu = {};
            ++m_materialTextureGeneration;
        } else if (status == WebTextureAssetLoad::Result::Ready) {
            const auto data = state.load.TakeData();
            const auto &texture = data.texture;
            auto gpu = UploadMaterialTexture(texture, data.filterMode, data.wrapMode, data.anisoLevel, sampler);
            if (gpu.view) {
                state.gpu = std::move(gpu);
                ++m_materialTextureGeneration;
                std::printf("INFERNUX_WEB_MATERIAL_TEXTURE_READY guid=%s width=%u height=%u mips=%u filter=%s wrap=%s "
                            "sampler=%s\n",
                            guid.c_str(), texture.mipLevels.front().width, texture.mipLevels.front().height,
                            state.gpu.mipLevels, data.filterMode.c_str(), data.wrapMode.c_str(),
                            samplerIdentity.c_str());
            } else {
                std::fprintf(stderr, "INFERNUX_WEB_MATERIAL_TEXTURE_FAILED guid=%s format=%s\n", guid.c_str(),
                             TextureFormatName(texture.format));
            }
        }
    } catch (const std::exception &error) {
        std::fprintf(stderr, "INFERNUX_WEB_MATERIAL_TEXTURE_FAILED guid=%s error=%s\n", guid.c_str(), error.what());
    }
    return state.gpu.view ? state.gpu : fallback;
}

wgpu::BindGroup
WebSceneRenderer::ResolveMaterialTextureSet(const std::shared_ptr<InxMaterial> &material,
                                            const std::shared_ptr<const RendererParameterBlock> &parameters)
{
    const std::array<MaterialTextureBinding, 6> bindings = MaterialTextures(material, parameters);
    std::array<GPUTexture, 6> textures{};
    std::string key;
    for (size_t index = 0; index < bindings.size(); ++index) {
        if (index > 0)
            key.push_back('\x1f');
        key.append(bindings[index].guid);
        key.push_back('\x1e');
        key.append(SamplerIdentity(bindings[index].sampler));
        const GPUTexture &fallback = index == 4 ? m_normalTexture : m_whiteTexture;
        textures[index] = ResolveMaterialTexture(bindings[index].guid, bindings[index].sampler, fallback);
    }
    MaterialTextureSetState &state = m_materialTextureSets[key];
    if (!state.group || state.textureGeneration != m_materialTextureGeneration) {
        state.group = CreateMaterialTextureGroup(textures);
        state.textureGeneration = m_materialTextureGeneration;
    }
    return state.group ? state.group : m_defaultMaterialTextureGroup;
}

bool WebSceneRenderer::CreatePipelines()
{
    if (m_shaders.empty()) {
        try {
            auto compiled = SlangCompiler::Get().Compile(GetSlangSceneModule(), {}, {ShaderCodeTarget::Wgsl});
            for (const auto &entry : compiled.targets.front().entries) {
                wgpu::ShaderSourceWGSL source;
                source.code = wgpu::StringView(entry.code.data(), entry.code.size());
                wgpu::ShaderModuleDescriptor descriptor;
                descriptor.nextInChain = &source;
                m_shaders.emplace(entry.name, m_device.CreateShaderModule(&descriptor));
            }
        } catch (const std::exception &error) {
            std::fprintf(stderr, "INFERNUX_WEB_SCENE_SHADER_FAILED %s\n", error.what());
            return false;
        }
    }

    std::array<wgpu::VertexAttribute, 8> attributes{};
    const std::array<wgpu::VertexFormat, 8> formats = {wgpu::VertexFormat::Float32x3, wgpu::VertexFormat::Float32x3,
                                                       wgpu::VertexFormat::Float32x4, wgpu::VertexFormat::Float32x3,
                                                       wgpu::VertexFormat::Float32x2, wgpu::VertexFormat::Uint32x4,
                                                       wgpu::VertexFormat::Float32x4, wgpu::VertexFormat::Float32x2};
    const std::array<uint64_t, 8> offsets = {offsetof(Vertex, pos),         offsetof(Vertex, normal),
                                             offsetof(Vertex, tangent),     offsetof(Vertex, color),
                                             offsetof(Vertex, texCoord),    offsetof(Vertex, boneIndices),
                                             offsetof(Vertex, boneWeights), offsetof(Vertex, texCoord1)};
    for (size_t i = 0; i < attributes.size(); ++i) {
        attributes[i].format = formats[i];
        attributes[i].offset = offsets[i];
        attributes[i].shaderLocation = static_cast<uint32_t>(i);
    }
    wgpu::VertexBufferLayout vertexLayout;
    vertexLayout.arrayStride = sizeof(Vertex);
    vertexLayout.stepMode = wgpu::VertexStepMode::Vertex;
    vertexLayout.attributeCount = attributes.size();
    vertexLayout.attributes = attributes.data();

    wgpu::ColorTargetState colorTarget;
    colorTarget.format = m_colorFormat;
    colorTarget.writeMask = wgpu::ColorWriteMask::All;
    wgpu::FragmentState fragment;
    fragment.module = m_shaders.at("scene_fragment");
    fragment.entryPoint = "scene_fragment";
    fragment.targetCount = 1;
    fragment.targets = &colorTarget;

    wgpu::DepthStencilState depth;
    depth.format = wgpu::TextureFormat::Depth32Float;
    depth.depthWriteEnabled = wgpu::OptionalBool::True;
    depth.depthCompare = wgpu::CompareFunction::GreaterEqual;

    wgpu::PipelineLayoutDescriptor pipelineLayoutDescriptor;
    const std::array<wgpu::BindGroupLayout, 3> sceneLayouts = {m_cameraLayout, m_materialTextureLayout,
                                                               m_geometryLayout};
    pipelineLayoutDescriptor.bindGroupLayoutCount = sceneLayouts.size();
    pipelineLayoutDescriptor.bindGroupLayouts = sceneLayouts.data();

    wgpu::RenderPipelineDescriptor pipelineDescriptor;
    pipelineDescriptor.layout = m_device.CreatePipelineLayout(&pipelineLayoutDescriptor);
    pipelineDescriptor.vertex.module = m_shaders.at("scene_vertex");
    pipelineDescriptor.vertex.entryPoint = "scene_vertex";
    pipelineDescriptor.vertex.bufferCount = 1;
    pipelineDescriptor.vertex.buffers = &vertexLayout;
    pipelineDescriptor.fragment = &fragment;
    pipelineDescriptor.primitive.topology = wgpu::PrimitiveTopology::TriangleList;
    pipelineDescriptor.depthStencil = &depth;
    pipelineDescriptor.multisample.count = m_sceneSampleCount;
    wgpu::BlendState transparentBlend;
    transparentBlend.color.srcFactor = wgpu::BlendFactor::SrcAlpha;
    transparentBlend.color.dstFactor = wgpu::BlendFactor::OneMinusSrcAlpha;
    transparentBlend.alpha.srcFactor = wgpu::BlendFactor::One;
    transparentBlend.alpha.dstFactor = wgpu::BlendFactor::OneMinusSrcAlpha;
    const std::array<wgpu::FrontFace, 2> frontFaces = {wgpu::FrontFace::CCW, wgpu::FrontFace::CW};
    const std::array<wgpu::CullMode, 3> cullModes = {wgpu::CullMode::None, wgpu::CullMode::Front, wgpu::CullMode::Back};
    for (size_t face = 0; face < frontFaces.size(); ++face) {
        for (size_t cull = 0; cull < cullModes.size(); ++cull) {
            const size_t pipelineIndex = face * cullModes.size() + cull;
            pipelineDescriptor.primitive.frontFace = frontFaces[face];
            pipelineDescriptor.primitive.cullMode = cullModes[cull];
            m_opaquePipelines[pipelineIndex] = m_device.CreateRenderPipeline(&pipelineDescriptor);
            if (!m_opaquePipelines[pipelineIndex])
                return false;
            colorTarget.blend = &transparentBlend;
            depth.depthWriteEnabled = wgpu::OptionalBool::False;
            m_transparentPipelines[pipelineIndex] = m_device.CreateRenderPipeline(&pipelineDescriptor);
            colorTarget.blend = nullptr;
            depth.depthWriteEnabled = wgpu::OptionalBool::True;
            if (!m_transparentPipelines[pipelineIndex])
                return false;
        }
    }

    wgpu::FragmentState skyFragment;
    skyFragment.module = m_shaders.at("sky_fragment");
    skyFragment.entryPoint = "sky_fragment";
    skyFragment.targetCount = 1;
    skyFragment.targets = &colorTarget;
    wgpu::RenderPipelineDescriptor skyPipelineDescriptor;
    const std::array<wgpu::BindGroupLayout, 2> skyLayouts{m_cameraLayout, m_materialTextureLayout};
    wgpu::PipelineLayoutDescriptor skyLayoutDescriptor;
    skyLayoutDescriptor.bindGroupLayoutCount = skyLayouts.size();
    skyLayoutDescriptor.bindGroupLayouts = skyLayouts.data();
    skyPipelineDescriptor.layout = m_device.CreatePipelineLayout(&skyLayoutDescriptor);
    skyPipelineDescriptor.vertex.module = m_shaders.at("sky_vertex");
    skyPipelineDescriptor.vertex.entryPoint = "sky_vertex";
    skyPipelineDescriptor.fragment = &skyFragment;
    skyPipelineDescriptor.primitive.topology = wgpu::PrimitiveTopology::TriangleList;
    skyPipelineDescriptor.primitive.cullMode = wgpu::CullMode::None;
    skyPipelineDescriptor.multisample.count = m_sceneSampleCount;
    wgpu::DepthStencilState skyDepth;
    skyDepth.format = wgpu::TextureFormat::Depth32Float;
    skyDepth.depthWriteEnabled = wgpu::OptionalBool::False;
    skyDepth.depthCompare = wgpu::CompareFunction::Always;
    skyPipelineDescriptor.depthStencil = &skyDepth;
    m_skyPipeline = m_device.CreateRenderPipeline(&skyPipelineDescriptor);

    wgpu::DepthStencilState shadowDepth;
    shadowDepth.format = wgpu::TextureFormat::Depth32Float;
    shadowDepth.depthWriteEnabled = wgpu::OptionalBool::True;
    shadowDepth.depthCompare = wgpu::CompareFunction::GreaterEqual;
    shadowDepth.depthBias = -2;
    shadowDepth.depthBiasSlopeScale = -2.0f;
    wgpu::PipelineLayoutDescriptor shadowPipelineLayoutDescriptor;
    const wgpu::BindGroupLayoutDescriptor emptyLayoutDescriptor{};
    const std::array<wgpu::BindGroupLayout, 3> shadowLayouts = {
        m_shadowCameraLayout, m_device.CreateBindGroupLayout(&emptyLayoutDescriptor), m_geometryLayout};
    shadowPipelineLayoutDescriptor.bindGroupLayoutCount = shadowLayouts.size();
    shadowPipelineLayoutDescriptor.bindGroupLayouts = shadowLayouts.data();
    wgpu::RenderPipelineDescriptor shadowPipelineDescriptor;
    shadowPipelineDescriptor.layout = m_device.CreatePipelineLayout(&shadowPipelineLayoutDescriptor);
    shadowPipelineDescriptor.vertex.module = m_shaders.at("shadow_vertex");
    shadowPipelineDescriptor.vertex.entryPoint = "shadow_vertex";
    shadowPipelineDescriptor.vertex.bufferCount = 1;
    shadowPipelineDescriptor.vertex.buffers = &vertexLayout;
    shadowPipelineDescriptor.primitive.topology = wgpu::PrimitiveTopology::TriangleList;
    shadowPipelineDescriptor.depthStencil = &shadowDepth;
    for (size_t face = 0; face < frontFaces.size(); ++face) {
        for (size_t cull = 0; cull < cullModes.size(); ++cull) {
            const size_t pipelineIndex = face * cullModes.size() + cull;
            shadowPipelineDescriptor.primitive.frontFace = frontFaces[face];
            shadowPipelineDescriptor.primitive.cullMode = cullModes[cull];
            m_shadowPipelines[pipelineIndex] = m_device.CreateRenderPipeline(&shadowPipelineDescriptor);
            if (!m_shadowPipelines[pipelineIndex])
                return false;
        }
    }
    return static_cast<bool>(m_skyPipeline);
}

bool WebSceneRenderer::ConfigureSampleCount(uint32_t sampleCount)
{
    if (sampleCount != 1 && sampleCount != 4)
        return false;
    if (sampleCount == m_sceneSampleCount)
        return true;
    m_sceneSampleCount = sampleCount;
    if (!CreatePipelines())
        return false;
    m_depthTexture = {};
    m_depthView = {};
    if (m_depthWidth == 0 || m_depthHeight == 0)
        return true;
    Resize(m_depthWidth, m_depthHeight);
    return HasDepthTarget();
}

void WebSceneRenderer::Resize(uint32_t width, uint32_t height)
{
    width = std::max(1u, width);
    height = std::max(1u, height);
    if (m_depthTexture && width == m_depthWidth && height == m_depthHeight)
        return;

    wgpu::TextureDescriptor descriptor;
    descriptor.dimension = wgpu::TextureDimension::e2D;
    descriptor.size = {width, height, 1};
    descriptor.format = wgpu::TextureFormat::Depth32Float;
    descriptor.mipLevelCount = 1;
    descriptor.sampleCount = m_sceneSampleCount;
    descriptor.usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding;
    m_depthTexture = m_device.CreateTexture(&descriptor);
    m_depthView = m_depthTexture ? m_depthTexture.CreateView() : wgpu::TextureView{};
    m_depthWidth = width;
    m_depthHeight = height;
}

bool WebSceneRenderer::HasDepthTarget() const noexcept
{
    return static_cast<bool>(m_depthView);
}

wgpu::TextureView WebSceneRenderer::GetDepthView() const noexcept
{
    return m_depthView;
}

void WebSceneRenderer::SetSkyEnabledForDiagnostics(bool enabled) noexcept
{
    m_diagnosticSkyEnabled = enabled;
}

void WebSceneRenderer::SetShadowsEnabledForDiagnostics(bool enabled) noexcept
{
    m_diagnosticShadowsEnabled = enabled;
}

bool WebSceneRenderer::EnsureBuffer(wgpu::Buffer &buffer, uint64_t &capacity, uint64_t required,
                                    wgpu::BufferUsage usage)
{
    if (required == 0)
        return false;
    if (buffer && required <= capacity)
        return true;
    capacity = GrowCapacity(required);
    wgpu::BufferDescriptor descriptor;
    descriptor.size = capacity;
    descriptor.usage = usage | wgpu::BufferUsage::CopyDst;
    buffer = m_device.CreateBuffer(&descriptor);
    return static_cast<bool>(buffer);
}

bool WebSceneRenderer::CreateShadowFitPipeline()
{
    std::array<wgpu::BindGroupLayoutEntry, 4> entries{};
    for (size_t i = 0; i < entries.size(); ++i) {
        entries[i].binding = static_cast<uint32_t>(i);
        entries[i].visibility = wgpu::ShaderStage::Compute;
        entries[i].buffer.type = wgpu::BufferBindingType::Storage;
        entries[i].buffer.minBindingSize = sizeof(WorldBounds);
    }
    entries[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
    entries[2].buffer.minBindingSize = sizeof(glm::mat4);
    entries[3].buffer.type = wgpu::BufferBindingType::Uniform;
    entries[3].buffer.minBindingSize = sizeof(ShadowFitParameters);
    wgpu::BindGroupLayoutDescriptor layout;
    layout.entryCount = entries.size();
    layout.entries = entries.data();
    m_shadowFitLayout = m_device.CreateBindGroupLayout(&layout);
    m_shadowFitGroup = {};
    wgpu::PipelineLayoutDescriptor pipelineLayout;
    pipelineLayout.bindGroupLayoutCount = 1;
    pipelineLayout.bindGroupLayouts = &m_shadowFitLayout;
    const auto nativeLayout = m_device.CreatePipelineLayout(&pipelineLayout);
    auto compiled = SlangCompiler::Get().Compile(GetSlangSceneBoundsModule(), {}, {ShaderCodeTarget::Wgsl});
    for (const auto &entry : compiled.targets.front().entries) {
        wgpu::ShaderSourceWGSL source;
        source.code = wgpu::StringView(entry.code.data(), entry.code.size());
        wgpu::ShaderModuleDescriptor shader;
        shader.nextInChain = &source;
        wgpu::ComputePipelineDescriptor pipeline;
        pipeline.layout = nativeLayout;
        pipeline.compute.module = m_device.CreateShaderModule(&shader);
        pipeline.compute.entryPoint = entry.name.c_str();
        auto native = m_device.CreateComputePipeline(&pipeline);
        if (entry.name == "reduce_bounds")
            m_boundsPipeline = std::move(native);
        else
            m_shadowFitPipeline = std::move(native);
    }
    return m_boundsPipeline && m_shadowFitPipeline;
}

WebSceneRenderer::GeometryResource &WebSceneRenderer::ResolveGeometry(const DrawCall &draw)
{
    auto [found, inserted] = m_geometry.try_emplace(draw.meshVertices);
    auto &geometry = found->second;
    const bool changed =
        inserted || geometry.owner != draw.meshDataOwner || geometry.revision != draw.meshRuntimeVersion ||
        geometry.sourceIndices != draw.meshIndices || geometry.sourceFormat != draw.meshIndexFormat ||
        geometry.vertexCount != draw.meshVertices->size() || geometry.indexCount != draw.meshIndices->size() ||
        (draw.forceBufferUpdate && geometry.lastFrame != m_frame);
    if (changed) {
        WebPackedIndexStreams streams;
        const auto range = PackWebIndexRange(streams, draw.meshIndexFormat, 0, draw.meshVertices->size(),
                                             *draw.meshIndices, 0, draw.meshIndices->size());
        geometry = {};
        geometry.owner = draw.meshDataOwner;
        geometry.sourceIndices = draw.meshIndices;
        geometry.revision = draw.meshRuntimeVersion;
        geometry.format = range.format;
        geometry.sourceFormat = draw.meshIndexFormat;
        geometry.vertexCount = draw.meshVertices->size();
        geometry.indexCount = draw.meshIndices->size();
        geometry.line = std::any_of(draw.meshVertices->begin(), draw.meshVertices->end(),
                                    [](const Vertex &v) { return v.boneIndices.w == kLineVertexMarker; });
        wgpu::BufferDescriptor descriptor;
        descriptor.label = "Scene canonical indices";
        descriptor.usage = wgpu::BufferUsage::Index | wgpu::BufferUsage::CopyDst;
        descriptor.size = range.format == MeshIndexFormat::UInt16 ? streams.uint16Indices.size() * sizeof(uint16_t)
                                                                  : streams.uint32Indices.size() * sizeof(uint32_t);
        geometry.indices = m_device.CreateBuffer(&descriptor);
        const void *indices = range.format == MeshIndexFormat::UInt16
                                  ? static_cast<const void *>(streams.uint16Indices.data())
                                  : static_cast<const void *>(streams.uint32Indices.data());
        m_queue.WriteBuffer(geometry.indices, 0, indices, descriptor.size);
    }
    if (!draw.meshVertexBuffer && !geometry.vertices) {
        wgpu::BufferDescriptor descriptor;
        descriptor.label = "Scene canonical vertices";
        descriptor.usage = wgpu::BufferUsage::Vertex | wgpu::BufferUsage::CopyDst;
        descriptor.size = geometry.vertexCount * sizeof(Vertex);
        geometry.vertices = m_device.CreateBuffer(&descriptor);
        m_queue.WriteBuffer(geometry.vertices, 0, draw.meshVertices->data(), descriptor.size);
    }
    geometry.lastFrame = m_frame;
    return geometry;
}

wgpu::Buffer WebSceneRenderer::CaptureBuffer(const std::shared_ptr<rhi::ComputeBuffer> &value) const
{
    if (!value->GetHost().IsAlive() || value->GetHost().GetDeviceId() != m_rhi->GetDeviceId() ||
        !value->GetLastWriteSubmission().IsValid())
        throw std::invalid_argument("Scene compute input must be an accepted value on the rendering device");
    // The Web compute host and renderer submit to this exact native queue.
    // Accepted producers are already enqueued; there is no CPU completion wait.
    return m_rhi->GetNativeBuffer(value->GetBuffer());
}

wgpu::BindGroup WebSceneRenderer::ResolveGeometryBinding(wgpu::Buffer models)
{
    auto [found, inserted] = m_geometryBindings.try_emplace(models.Get());
    auto &binding = found->second;
    if (inserted) {
        std::array<wgpu::BindGroupEntry, 2> entries{};
        entries[0].binding = 0;
        entries[0].buffer = models;
        entries[0].size = models.GetSize();
        entries[1].binding = 1;
        entries[1].buffer = m_boneBuffer;
        entries[1].size = m_boneCapacity;
        wgpu::BindGroupDescriptor descriptor;
        descriptor.layout = m_geometryLayout;
        descriptor.entryCount = entries.size();
        descriptor.entries = entries.data();
        binding.group = m_device.CreateBindGroup(&descriptor);
    }
    binding.lastFrame = m_frame;
    return binding.group;
}

bool WebSceneRenderer::PrepareGeometry()
{
    const auto drawCapacity = m_drawDataCapacity;
    const auto boundsCapacity = m_boundsCapacity;
    const auto tilesCapacity = m_boundsTilesCapacity;
    const auto boneCapacity = m_boneCapacity;
    if (!EnsureBuffer(m_drawDataBuffer, m_drawDataCapacity,
                      std::max<size_t>(1, m_drawData.size()) * sizeof(WebDrawData), wgpu::BufferUsage::Storage) ||
        !EnsureBuffer(m_modelBuffer, m_modelCapacity, m_models.size() * sizeof(glm::mat4),
                      wgpu::BufferUsage::Storage) ||
        !EnsureBuffer(m_boneBuffer, m_boneCapacity, m_bones.size() * sizeof(glm::mat4), wgpu::BufferUsage::Storage) ||
        !EnsureBuffer(m_boundsBuffer, m_boundsCapacity, std::max<size_t>(1, m_bounds.size()) * sizeof(WorldBounds),
                      wgpu::BufferUsage::Storage) ||
        !EnsureBuffer(m_boundsTilesBuffer, m_boundsTilesCapacity,
                      std::max(1u, m_shadowFitParameters.counts.y) * sizeof(WorldBounds), wgpu::BufferUsage::Storage))
        return false;
    if ((drawCapacity != m_drawDataCapacity || boundsCapacity != m_boundsCapacity) && !CreateCameraBindGroups())
        return false;
    if (boneCapacity != m_boneCapacity)
        m_geometryBindings.clear();
    if (!m_shadowFitGroup || boundsCapacity != m_boundsCapacity || tilesCapacity != m_boundsTilesCapacity) {
        std::array<wgpu::BindGroupEntry, 4> entries{};
        const std::array<wgpu::Buffer, 4> buffers{m_boundsBuffer, m_boundsTilesBuffer, m_shadowMatrixBuffer,
                                                  m_shadowFitParametersBuffer};
        const std::array<uint64_t, 4> sizes{m_boundsCapacity, m_boundsTilesCapacity, sizeof(glm::mat4),
                                            sizeof(ShadowFitParameters)};
        for (size_t i = 0; i < entries.size(); ++i) {
            entries[i].binding = static_cast<uint32_t>(i);
            entries[i].buffer = buffers[i];
            entries[i].size = sizes[i];
        }
        wgpu::BindGroupDescriptor descriptor;
        descriptor.layout = m_shadowFitLayout;
        descriptor.entryCount = entries.size();
        descriptor.entries = entries.data();
        m_shadowFitGroup = m_device.CreateBindGroup(&descriptor);
    }
    for (auto &range : m_drawRanges)
        range.geometryGroup =
            ResolveGeometryBinding(range.poseValue ? CaptureBuffer(range.poseValue->models) : m_modelBuffer);
    RetireUnused(m_geometryBindings, m_frame);
    m_queue.WriteBuffer(m_modelBuffer, 0, m_models.data(), m_models.size() * sizeof(glm::mat4));
    m_queue.WriteBuffer(m_boneBuffer, 0, m_bones.data(), m_bones.size() * sizeof(glm::mat4));
    if (!m_drawData.empty()) {
        m_queue.WriteBuffer(m_drawDataBuffer, 0, m_drawData.data(), m_drawData.size() * sizeof(WebDrawData));
        m_queue.WriteBuffer(m_boundsBuffer, 0, m_bounds.data(), m_bounds.size() * sizeof(WorldBounds));
        m_queue.WriteBuffer(m_shadowFitParametersBuffer, 0, &m_shadowFitParameters, sizeof(m_shadowFitParameters));
    }
    return true;
}

void WebSceneRenderer::BindGeometry(wgpu::RenderPassEncoder pass, const WebDrawRange &range) const
{
    pass.SetVertexBuffer(0, range.vertexBuffer, 0, range.vertexBytes);
    pass.SetIndexBuffer(range.indexBuffer,
                        range.indices.format == MeshIndexFormat::UInt16 ? wgpu::IndexFormat::Uint16
                                                                        : wgpu::IndexFormat::Uint32,
                        0, range.indexBytes);
    pass.SetBindGroup(2, range.geometryGroup);
}

bool WebSceneRenderer::BuildFrame(uint32_t width, uint32_t height, Camera *camera)
{
    SceneManager &sceneManager = SceneManager::Instance();
    // The active Scene owns only the default camera, environment and authoring
    // policy. SceneRenderExtractor consumes SceneManager's process-wide
    // runtime renderer registry, which already contains every loaded additive
    // Scene plus the DontDestroyOnLoad Scene. Do not walk the active Scene or
    // build a Web-only scene list here.
    Scene *scene = sceneManager.GetActiveScene();
    if (!scene) {
        ReportFrameIssue("no-active-scene");
        return false;
    }
    if (!camera) {
        ReportFrameIssue("no-game-camera");
        return false;
    }

    const glm::vec4 background = inx::color::SrgbToLinear(camera->GetBackgroundColor());
    m_clearColor = {background.r, background.g, background.b, background.a};
    camera->SetAspectRatio(static_cast<float>(width) / static_cast<float>(std::max(1u, height)));
    (void)m_extractor.ExtractCameraFrame(m_world, camera);
    const auto frame = m_world.Acquire();
    if (!frame || !frame->PrimaryView().valid) {
        ReportFrameIssue("invalid-render-snapshot");
        return false;
    }

    const SceneEnvironmentSettings &environment = scene->GetEnvironment();
    m_cameraData = {};
    m_cameraData.viewProjection = ToWebClipSpace(frame->PrimaryView().viewProjection);
    m_cameraData.inverseViewProjection = glm::inverse(m_cameraData.viewProjection);
    m_cameraData.view = frame->PrimaryView().view;
    m_cameraData.cameraPosition = glm::vec4(frame->PrimaryView().position, 1.0f);
    m_cameraData.skyTopExposure = glm::vec4(inx::color::SrgbToLinear(environment.skyTopColor), environment.skyExposure);
    m_cameraData.skyHorizon = glm::vec4(inx::color::SrgbToLinear(environment.skyHorizonColor), 0.0f);
    m_cameraData.skyGround = glm::vec4(inx::color::SrgbToLinear(environment.skyGroundColor), 1.0f);
    m_drawSky = m_diagnosticSkyEnabled && camera->GetClearFlags() == CameraClearFlags::Skybox;
    m_skyTextureGroup = m_defaultMaterialTextureGroup;
    if (m_drawSky && !environment.skyboxMaterialGuid.empty()) {
        const auto sky = scene->ResolveSkyboxMaterial();
        if (!sky) {
            ReportFrameIssue("missing-skybox-material");
            return false;
        }
        const float exposure = MaterialFloat(sky, {}, "exposure", 1.0f);
        if (sky->GetShaderName() != "Skybox Procedural" && sky->HasProperty("texSampler")) {
            m_skyTextureGroup = ResolveMaterialTextureSet(sky, {});
            m_cameraData.skyTopExposure = glm::vec4(
                glm::vec3(inx::color::SrgbToLinear(MaterialVector(sky, {}, "tint", glm::vec4(1.0f)))), exposure);
            m_cameraData.skyHorizon.w = 1.0f;
            m_cameraData.skyGround.w = MaterialFloat(sky, {}, "rotation", 0.0f);
        } else {
            m_cameraData.skyTopExposure =
                glm::vec4(inx::color::SrgbToLinear(
                              glm::vec3(MaterialVector(sky, {}, "skyTopColor", glm::vec4(environment.skyTopColor, 1)))),
                          exposure);
            m_cameraData.skyHorizon =
                glm::vec4(inx::color::SrgbToLinear(glm::vec3(
                              MaterialVector(sky, {}, "skyHorizonColor", glm::vec4(environment.skyHorizonColor, 1)))),
                          0);
            m_cameraData.skyGround = glm::vec4(inx::color::SrgbToLinear(glm::vec3(MaterialVector(
                                                   sky, {}, "groundColor", glm::vec4(environment.skyGroundColor, 1)))),
                                               0);
        }
    }

    using AmbientSource = SceneEnvironmentSettings::AmbientSource;
    switch (static_cast<AmbientSource>(environment.ambientSource)) {
    case AmbientSource::Color:
        m_cameraData.ambient =
            glm::vec4(inx::color::SrgbToLinear(environment.ambientColor), environment.ambientIntensity);
        m_cameraData.ambientEquator.w = 0.0f;
        break;
    case AmbientSource::Gradient:
        m_cameraData.ambientSky =
            glm::vec4(inx::color::SrgbToLinear(environment.ambientSkyColor) * environment.ambientIntensity, 1.0f);
        m_cameraData.ambientEquator =
            glm::vec4(inx::color::SrgbToLinear(environment.ambientEquatorColor) * environment.ambientIntensity, 1.0f);
        m_cameraData.ambientGround =
            glm::vec4(inx::color::SrgbToLinear(environment.ambientGroundColor) * environment.ambientIntensity, 1.0f);
        break;
    case AmbientSource::Skybox:
    default:
        m_cameraData.ambientSky = glm::vec4(inx::color::SrgbToLinear(environment.skyTopColor) *
                                                (environment.skyExposure * environment.ambientIntensity),
                                            1.0f);
        m_cameraData.ambientEquator = glm::vec4(inx::color::SrgbToLinear(environment.skyHorizonColor) *
                                                    (environment.skyExposure * environment.ambientIntensity),
                                                1.0f);
        m_cameraData.ambientGround = glm::vec4(inx::color::SrgbToLinear(environment.skyGroundColor) *
                                                   (environment.skyExposure * environment.ambientIntensity),
                                               1.0f);
        break;
    }

    ++m_frame;
    m_drawRanges.clear();
    m_drawData.clear();
    m_models.assign(1, glm::mat4(1.0f));
    m_bones.assign(1, glm::mat4(1.0f));
    m_bounds.clear();
    m_vertexCount = 0;
    const auto &drawCalls = frame->DrawCalls().drawCalls;
    for (const DrawCall &draw : drawCalls) {
        if (!draw.frustumVisible || !(draw.layerMask & camera->GetCullingMask()) || !draw.meshVertices ||
            !draw.meshIndices || draw.indexCount == 0)
            continue;
        const auto &sourceVertices = *draw.meshVertices;
        const auto &sourceIndices = *draw.meshIndices;
        if (sourceVertices.empty() || sourceIndices.empty())
            continue;
        if (draw.indexStart >= sourceIndices.size()) {
            ReportFrameIssue("invalid-index-stream");
            return false;
        }
        const size_t indexCount = draw.indexCount;
        const glm::vec4 materialColor = inx::color::SrgbToLinear(MaterialColor(draw.material, draw.parameterBlock));
        const glm::vec4 emission = inx::color::SrgbToLinear(
            MaterialVector(draw.material, draw.parameterBlock, "emissionColor", glm::vec4(0.0f)));
        const glm::vec4 materialParameters(
            std::clamp(MaterialFloat(draw.material, draw.parameterBlock, "metallic", 0.0f), 0.0f, 1.0f),
            std::clamp(MaterialFloat(draw.material, draw.parameterBlock, "smoothness", 0.5f), 0.0f, 1.0f),
            std::clamp(MaterialFloat(draw.material, draw.parameterBlock, "ambientOcclusion", 1.0f), 0.0f, 1.0f),
            std::clamp(MaterialFloat(draw.material, draw.parameterBlock, "specularHighlights", 1.0f), 0.0f, 1.0f));
        const float normalScale =
            std::max(0.0f, MaterialFloat(draw.material, draw.parameterBlock, "normalScale", 1.0f));
        float shadingModel = 0.0f;
        if (MaterialIsUnlit(draw.material))
            shadingModel = 1.0f;
        else if (MaterialIsToon(draw.material))
            shadingModel = 2.0f;
        float alphaClipThreshold = -1.0f;
        if (draw.material && draw.material->GetRenderState().alphaClipEnabled)
            alphaClipThreshold = std::clamp(draw.material->GetRenderState().alphaClipThreshold, 0.0f, 1.0f);
        const glm::vec4 surfaceParameters(
            shadingModel,
            std::clamp(MaterialFloat(draw.material, draw.parameterBlock, "diffuseThreshold", 0.45f), 0.0f, 1.0f),
            std::clamp(MaterialFloat(draw.material, draw.parameterBlock, "bandSoftness", 0.04f), 0.0f, 0.25f),
            alphaClipThreshold);
        const std::array<int, 6> textureUvSets = {
            MaterialInt(draw.material, draw.parameterBlock, "baseColorUvSet", 0),
            MaterialInt(draw.material, draw.parameterBlock, "metallicUvSet", 0),
            MaterialInt(draw.material, draw.parameterBlock, "smoothnessUvSet", 0),
            MaterialInt(draw.material, draw.parameterBlock, "occlusionUvSet", 0),
            MaterialInt(draw.material, draw.parameterBlock, "normalUvSet", 0),
            MaterialInt(draw.material, draw.parameterBlock, "emissionUvSet", 0),
        };
        if (std::any_of(textureUvSets.begin(), textureUvSets.end(), [](int value) { return value < 0 || value > 1; })) {
            ReportFrameIssue("invalid-material-uv-set");
            return false;
        }
        const glm::vec4 textureUvSets0(textureUvSets[0], textureUvSets[1], textureUvSets[2], textureUvSets[3]);
        const glm::vec2 textureUvSets1(textureUvSets[4], textureUvSets[5]);
        const glm::vec4 metallicChannels =
            MaterialVector(draw.material, draw.parameterBlock, "metallicChannels", glm::vec4(1.0f, 0.0f, 0.0f, 0.0f));
        const glm::vec4 smoothnessChannels =
            MaterialVector(draw.material, draw.parameterBlock, "smoothnessChannels", glm::vec4(1.0f, 0.0f, 0.0f, 0.0f));
        const glm::vec2 materialSampling(
            std::clamp(MaterialFloat(draw.material, draw.parameterBlock, "smoothnessFromRoughness", 0.0f), 0.0f, 1.0f),
            std::clamp(MaterialFloat(draw.material, draw.parameterBlock, "occlusionStrength", 1.0f), 0.0f, 1.0f));
        WebDrawRange range;
        range.castsShadows = draw.castsShadows;
        range.materialTextureGroup = ResolveMaterialTextureSet(draw.material, draw.parameterBlock);
        if (draw.material) {
            const RenderState &state = draw.material->GetRenderState();
            range.transparent = state.blendEnable || state.renderQueue >= 3000 || materialColor.a < 0.999f;
            range.fullyCulled = state.cullMode == MaterialCullMode::FrontAndBack;
            const uint8_t frontFace = state.frontFace == MaterialFrontFace::Clockwise ? 1u : 0u;
            const uint8_t cullMode = static_cast<uint8_t>(state.cullMode);
            range.rasterPipeline = static_cast<uint8_t>(frontFace * 3u + std::min<uint8_t>(cullMode, 2u));
        } else {
            range.transparent = materialColor.a < 0.999f;
            range.rasterPipeline = 5;
        }
        auto &geometry = ResolveGeometry(draw);
        if (draw.indexCount > sourceIndices.size() - draw.indexStart)
            throw std::invalid_argument("Scene draw exceeds its published index range");
        range.indices = {geometry.format, draw.indexStart, draw.indexCount, draw.vertexStart};
        range.vertexBuffer = geometry.vertices;
        range.indexBuffer = geometry.indices;
        range.vertexBytes = geometry.vertexCount * sizeof(Vertex);
        range.indexBytes = geometry.format == MeshIndexFormat::UInt16 ? ((geometry.indexCount + 1) / 2) * 4
                                                                      : geometry.indexCount * sizeof(uint32_t);
        if (draw.meshVertexBuffer) {
            range.vertexValue = draw.meshVertexBuffer->Freeze();
            if (!range.vertexValue->HasInitializedRange(0, range.vertexBytes))
                throw std::invalid_argument(
                    "Scene resident vertex stream has no accepted value for its active vertices");
            range.vertexBuffer = CaptureBuffer(range.vertexValue);
            range.vertexBytes = range.vertexValue->GetByteSize();
        }
        range.line = geometry.line;
        range.transparent = range.transparent || range.line;
        range.drawIndex = static_cast<uint32_t>(m_drawData.size());
        WebDrawData objectData;
        objectData.color = materialColor;
        objectData.emission = emission;
        objectData.material = materialParameters;
        objectData.surface = surfaceParameters;
        objectData.textureUvSets0 = textureUvSets0;
        objectData.textureUvSets1 = glm::vec4(textureUvSets1, 0.0f, 0.0f);
        objectData.metallicChannels = metallicChannels;
        objectData.smoothnessChannels = smoothnessChannels;
        objectData.materialSampling = glm::vec4(materialSampling, normalScale, draw.receivesShadows ? 1.0f : 0.0f);
        objectData.geometry.y = range.drawIndex;
        if (draw.devicePose.value) {
            range.poseValue = draw.devicePose.value;
            objectData.geometry.x = range.poseValue->firstModel + draw.devicePose.element;
        } else {
            objectData.geometry.x = static_cast<uint32_t>(m_models.size());
            m_models.push_back(draw.worldMatrix);
        }
        if (draw.skinBoneMatrices && !draw.skinBoneMatrices->empty()) {
            objectData.geometry.z = static_cast<uint32_t>(m_bones.size());
            objectData.geometry.w = static_cast<uint32_t>(draw.skinBoneMatrices->size());
            m_bones.insert(m_bones.end(), draw.skinBoneMatrices->begin(), draw.skinBoneMatrices->end());
        }
        m_vertexCount += geometry.vertexCount;
        m_bounds.push_back({glm::vec4(draw.worldBounds.min, 0.0f), glm::vec4(draw.worldBounds.max, 0.0f)});
        m_drawData.push_back(objectData);
        m_drawRanges.push_back(std::move(range));
    }
    RetireUnused(m_geometry, m_frame);

    Light *directionalLight = nullptr;
    // Like desktop SceneLightCollector, Web lighting consumes the shared
    // runtime registry rather than limiting illumination to the active Scene.
    const auto &residentLights = sceneManager.GetActiveLights();
    for (Light *light : residentLights) {
        if (light && light->IsEnabled() && light->GetLightType() == LightType::Directional &&
            light->GetAffectGeometry()) {
            directionalLight = light;
            break;
        }
    }
    glm::vec3 rayDirection(-0.35f, -0.82f, -0.45f);
    glm::vec3 lightColor(0.0f);
    float lightIntensity = 0.0f;
    float shadowStrength = 0.0f;
    if (directionalLight) {
        if (Transform *transform = directionalLight->GetTransform()) {
            const glm::vec3 forward = transform->GetWorldForward();
            if (Finite(forward) && glm::dot(forward, forward) > 1.0e-8f)
                rayDirection = glm::normalize(forward);
        }
        lightColor = directionalLight->GetLinearColor();
        lightIntensity = directionalLight->GetIntensity();
        if (directionalLight->GetShadows() != LightShadows::None)
            shadowStrength = directionalLight->GetShadowStrength();
    }
    const glm::vec3 towardLight = -glm::normalize(rayDirection);
    m_cameraData.lightColorIntensity = glm::vec4(lightColor, lightIntensity);
    m_shadowEnabled = m_diagnosticShadowsEnabled && shadowStrength > 0.0f;
    m_cameraData.lightDirectionStrength = glm::vec4(towardLight, m_shadowEnabled ? shadowStrength : 0.0f);

    uint32_t additionalLightCount = 0;
    for (Light *light : residentLights) {
        if (!light || !light->IsEnabled() || !light->GetAffectGeometry() ||
            additionalLightCount >= kMaxPunctualLights || light == directionalLight)
            continue;
        const LightType type = light->GetLightType();
        if (type != LightType::Directional && type != LightType::Point && type != LightType::Spot)
            continue;
        Transform *transform = light->GetTransform();
        if (!transform)
            continue;
        CameraData::PunctualLightData &target = m_cameraData.punctualLights[additionalLightCount++];
        target.positionRange = glm::vec4(transform->GetWorldPosition(), light->GetRange());
        target.colorIntensity = glm::vec4(light->GetLinearColor(), light->GetIntensity());
        glm::vec3 forward = transform->GetWorldForward();
        if (!Finite(forward) || glm::dot(forward, forward) < 1.0e-8f)
            forward = glm::vec3(0.0f, -1.0f, 0.0f);
        target.directionOuterCos =
            glm::vec4(glm::normalize(forward), std::cos(glm::radians(light->GetOuterSpotAngle() * 0.5f)));
        target.parameters =
            glm::vec4(std::cos(glm::radians(light->GetSpotAngle() * 0.5f)), type == LightType::Spot ? 1.0f : 0.0f,
                      type == LightType::Directional ? 1.0f : 0.0f, 0.0f);
    }
    m_cameraData.lightCounts.x = additionalLightCount;

    m_shadowEnabled = m_shadowEnabled && !m_bounds.empty();
    m_shadowFitParameters.rayDirection = glm::vec4(rayDirection, 0.0f);
    m_shadowFitParameters.counts = glm::uvec4(m_bounds.size(), (m_bounds.size() + 63) / 64, 0, 0);
    m_residentSceneCount = sceneManager.GetSceneCount() + (sceneManager.GetRuntimePersistentScene() ? 1U : 0U);
    m_residentRendererCount = sceneManager.GetActiveMeshRenderers().size();
    m_residentLightCount = residentLights.size();
    m_activeWorldId = scene->GetWorldId();
    if (!m_lastFrameIssue.empty()) {
        std::printf("INFERNUX_WEB_SCENE_RENDER_RECOVERED previous=%s vertices=%zu draws=%zu\n",
                    m_lastFrameIssue.c_str(), m_vertexCount, m_drawRanges.size());
        m_lastFrameIssue.clear();
    }
    return true;
}

void WebSceneRenderer::ReportFrameIssue(const char *issue)
{
    if (m_lastFrameIssue == issue)
        return;
    m_lastFrameIssue = issue;
    std::fprintf(stderr, "INFERNUX_WEB_SCENE_RENDER_EMPTY reason=%s\n", issue);
}

bool WebSceneRenderer::Prepare(wgpu::CommandEncoder encoder, uint32_t width, uint32_t height, Camera *camera)
{
    m_framePrepared = false;
    if (!m_opaquePipelines[5] || !m_transparentPipelines[5] || !m_shadowPipelines[5] || !encoder ||
        !BuildFrame(width, height, camera) || !PrepareGeometry())
        return false;
    m_queue.WriteBuffer(m_cameraBuffer, 0, &m_cameraData, sizeof(m_cameraData));
    for (const auto &range : m_drawRanges) {
        if (range.poseValue) {
            const auto &pose = *range.poseValue;
            const uint64_t element = m_drawData[range.drawIndex].geometry.x - pose.firstModel;
            // Gather only the 32-byte bounds record required by this frame's
            // culling and shadow reduction. Models and vertices stay in place.
            encoder.CopyBufferToBuffer(CaptureBuffer(pose.bounds), (pose.firstBounds + element) * sizeof(WorldBounds),
                                       m_boundsBuffer, range.drawIndex * sizeof(WorldBounds), sizeof(WorldBounds));
        }
    }
    if (m_shadowEnabled) {
        auto fit = encoder.BeginComputePass();
        fit.SetBindGroup(0, m_shadowFitGroup);
        fit.SetPipeline(m_boundsPipeline);
        fit.DispatchWorkgroups(m_shadowFitParameters.counts.y);
        fit.SetPipeline(m_shadowFitPipeline);
        fit.DispatchWorkgroups(1);
        fit.End();
        wgpu::RenderPassDepthStencilAttachment depthAttachment;
        depthAttachment.view = m_shadowView;
        depthAttachment.depthLoadOp = wgpu::LoadOp::Clear;
        depthAttachment.depthStoreOp = wgpu::StoreOp::Store;
        depthAttachment.depthClearValue = 0.0f;
        wgpu::RenderPassDescriptor descriptor;
        descriptor.depthStencilAttachment = &depthAttachment;
        wgpu::RenderPassEncoder shadowPass = encoder.BeginRenderPass(&descriptor);
        shadowPass.SetBindGroup(0, m_shadowCameraGroup);
        for (const WebDrawRange &range : m_drawRanges) {
            if (range.castsShadows && !range.fullyCulled) {
                shadowPass.SetPipeline(m_shadowPipelines[range.rasterPipeline]);
                BindGeometry(shadowPass, range);
                shadowPass.DrawIndexed(range.indices.indexCount, 1, range.indices.firstIndex, range.indices.baseVertex,
                                       range.drawIndex);
            }
        }
        shadowPass.End();
    }
    m_framePrepared = true;
    return true;
}

bool WebSceneRenderer::RenderPrepared(wgpu::RenderPassEncoder pass)
{
    if (!m_framePrepared || !pass)
        return false;

    if (m_drawSky && m_skyPipeline) {
        pass.SetPipeline(m_skyPipeline);
        pass.SetBindGroup(0, m_cameraGroup);
        pass.SetBindGroup(1, m_skyTextureGroup);
        pass.Draw(3, 1, 0, 0);
    }

    // A camera plus World/Screen UI is a complete renderable scene.  Keep
    // the sky and UI pass alive even when the world has no mesh draw stream.
    if (m_drawRanges.empty())
        return true;

    pass.SetBindGroup(0, m_cameraGroup);
    size_t transparentCount = 0;
    size_t lineCount = 0;
    // Populate the complete opaque depth buffer first.  Interleaving opaque
    // and transparent ranges lets a later opaque draw cover a line that is
    // actually in front because transparent geometry intentionally does not
    // write depth.
    for (const WebDrawRange &range : m_drawRanges) {
        transparentCount += range.transparent ? 1u : 0u;
        lineCount += range.line ? 1u : 0u;
        if (range.transparent || range.fullyCulled)
            continue;
        pass.SetPipeline(m_opaquePipelines[range.rasterPipeline]);
        pass.SetBindGroup(1, range.materialTextureGroup ? range.materialTextureGroup : m_defaultMaterialTextureGroup);
        BindGeometry(pass, range);
        pass.DrawIndexed(range.indices.indexCount, 1, range.indices.firstIndex, range.indices.baseVertex,
                         range.drawIndex);
    }
    for (const WebDrawRange &range : m_drawRanges) {
        if (!range.transparent || range.fullyCulled)
            continue;
        pass.SetPipeline(m_transparentPipelines[range.rasterPipeline]);
        pass.SetBindGroup(1, range.materialTextureGroup ? range.materialTextureGroup : m_defaultMaterialTextureGroup);
        BindGeometry(pass, range);
        pass.DrawIndexed(range.indices.indexCount, 1, range.indices.firstIndex, range.indices.baseVertex,
                         range.drawIndex);
    }
    if (!m_reportedFirstFrame) {
        size_t indexCount16 = 0;
        size_t indexCount32 = 0;
        for (const WebDrawRange &range : m_drawRanges) {
            if (range.indices.format == MeshIndexFormat::UInt16)
                indexCount16 += range.indices.indexCount;
            else if (range.indices.format == MeshIndexFormat::UInt32)
                indexCount32 += range.indices.indexCount;
        }
        std::printf("INFERNUX_WEB_SCENE_RENDER_READY vertices=%zu indices=%zu index16=%zu index32=%zu draws=%zu "
                    "transparent=%zu scenes=%zu renderers=%zu lights=%zu active_world=%llu material=pbr\n",
                    m_vertexCount, indexCount16 + indexCount32, indexCount16, indexCount32, m_drawRanges.size(),
                    transparentCount, m_residentSceneCount, m_residentRendererCount, m_residentLightCount,
                    static_cast<unsigned long long>(m_activeWorldId));
        if (m_drawSky)
            std::printf("INFERNUX_WEB_SKY_READY mode=procedural\n");
        if (m_shadowEnabled)
            std::printf("INFERNUX_WEB_SHADOW_READY resolution=%u\n", m_shadowResolution);
        m_reportedFirstFrame = true;
    }
    if (lineCount > 0 && !m_reportedLineDraw) {
        std::printf("INFERNUX_WEB_LINE_DRAW_READY draws=%zu expansion=camera-facing alpha=vertex\n", lineCount);
        m_reportedLineDraw = true;
    }
    return true;
}

} // namespace infernux::web
