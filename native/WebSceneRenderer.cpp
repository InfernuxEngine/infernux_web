#include "WebSceneRenderer.h"
#include "WebTextureUpload.h"

#include <core/types/ColorSpace.h>
#include <function/renderer/RendererParameterBlock.h>
#include <function/resources/AssetRegistry/AssetRegistry.h>
#include <function/resources/InxMaterial/InxMaterial.h>
#include <function/resources/InxTexture/InxTexture.h>
#include <function/resources/ShaderAsset/SlangSceneModule.h>
#include <function/scene/Camera.h>
#include <function/scene/GameObject.h>
#include <function/scene/Light.h>
#include <function/scene/Scene.h>
#include <function/scene/SceneManager.h>
#include <function/scene/Transform.h>

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>

namespace infernux::web
{
namespace
{

constexpr uint32_t kLineVertexMarker = 0x4C494E45u;
constexpr uint32_t kMaxPunctualLights = 8u;

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

glm::mat4 SkinMatrix(const Vertex &vertex, const std::vector<glm::mat4> *palette)
{
    if (!palette)
        return glm::mat4(1.0f);
    glm::mat4 result(0.0f);
    float totalWeight = 0.0f;
    for (uint32_t slot = 0; slot < 4; ++slot) {
        const float weight = vertex.boneWeights[slot];
        const uint32_t bone = vertex.boneIndices[slot];
        if (!(weight > 0.0f) || bone >= palette->size())
            continue;
        result += (*palette)[bone] * weight;
        totalWeight += weight;
    }
    return totalWeight > 0.0f ? result : glm::mat4(1.0f);
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

bool WebSceneRenderer::Initialize(wgpu::Device device, wgpu::Queue queue, wgpu::TextureFormat colorFormat,
                                  uint32_t sceneSampleCount)
{
    m_shaders.clear();
    m_device = std::move(device);
    m_queue = std::move(queue);
    m_colorFormat = colorFormat;
    m_sceneSampleCount = sceneSampleCount;
    if (!m_device || !m_queue || colorFormat == wgpu::TextureFormat::Undefined ||
        (sceneSampleCount != 1 && sceneSampleCount != 4))
        return false;

    wgpu::BufferDescriptor cameraBufferDescriptor;
    cameraBufferDescriptor.size = sizeof(CameraData);
    cameraBufferDescriptor.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
    m_cameraBuffer = m_device.CreateBuffer(&cameraBufferDescriptor);
    if (!m_cameraBuffer)
        return false;

    m_drawDataCapacity = sizeof(WebDrawData);
    wgpu::BufferDescriptor drawDataBufferDescriptor;
    drawDataBufferDescriptor.size = m_drawDataCapacity;
    drawDataBufferDescriptor.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    m_drawDataBuffer = m_device.CreateBuffer(&drawDataBufferDescriptor);
    if (!m_drawDataBuffer)
        return false;

    std::array<wgpu::BindGroupLayoutEntry, 4> cameraEntries{};
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
    cameraEntries[3].binding = 3;
    cameraEntries[3].visibility = wgpu::ShaderStage::Vertex;
    cameraEntries[3].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
    cameraEntries[3].buffer.minBindingSize = sizeof(WebDrawData);
    wgpu::BindGroupLayoutDescriptor cameraLayoutDescriptor;
    cameraLayoutDescriptor.entryCount = cameraEntries.size();
    cameraLayoutDescriptor.entries = cameraEntries.data();
    m_cameraLayout = m_device.CreateBindGroupLayout(&cameraLayoutDescriptor);

    if (!m_cameraLayout || !CreateShadowResources() || !CreateMaterialTextureResources())
        return false;

    std::array<wgpu::BindGroupLayoutEntry, 2> shadowCameraEntries{};
    shadowCameraEntries[0].binding = 0;
    shadowCameraEntries[0].visibility = wgpu::ShaderStage::Vertex;
    shadowCameraEntries[0].buffer.type = wgpu::BufferBindingType::Uniform;
    shadowCameraEntries[0].buffer.minBindingSize = sizeof(CameraData);
    shadowCameraEntries[1].binding = 3;
    shadowCameraEntries[1].visibility = wgpu::ShaderStage::Vertex;
    shadowCameraEntries[1].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
    shadowCameraEntries[1].buffer.minBindingSize = sizeof(WebDrawData);
    wgpu::BindGroupLayoutDescriptor shadowCameraLayoutDescriptor;
    shadowCameraLayoutDescriptor.entryCount = shadowCameraEntries.size();
    shadowCameraLayoutDescriptor.entries = shadowCameraEntries.data();
    m_shadowCameraLayout = m_device.CreateBindGroupLayout(&shadowCameraLayoutDescriptor);
    return m_shadowCameraLayout && CreateCameraBindGroups() && CreatePipelines();
}

bool WebSceneRenderer::CreateCameraBindGroups()
{
    if (!m_cameraLayout || !m_shadowCameraLayout || !m_cameraBuffer || !m_drawDataBuffer || !m_shadowView ||
        !m_shadowSampler)
        return false;

    std::array<wgpu::BindGroupEntry, 4> cameraBindings{};
    cameraBindings[0].binding = 0;
    cameraBindings[0].buffer = m_cameraBuffer;
    cameraBindings[0].size = sizeof(CameraData);
    cameraBindings[1].binding = 1;
    cameraBindings[1].textureView = m_shadowView;
    cameraBindings[2].binding = 2;
    cameraBindings[2].sampler = m_shadowSampler;
    cameraBindings[3].binding = 3;
    cameraBindings[3].buffer = m_drawDataBuffer;
    cameraBindings[3].size = m_drawDataCapacity;
    wgpu::BindGroupDescriptor cameraGroupDescriptor;
    cameraGroupDescriptor.layout = m_cameraLayout;
    cameraGroupDescriptor.entryCount = cameraBindings.size();
    cameraGroupDescriptor.entries = cameraBindings.data();
    m_cameraGroup = m_device.CreateBindGroup(&cameraGroupDescriptor);

    std::array<wgpu::BindGroupEntry, 2> shadowCameraBindings{};
    shadowCameraBindings[0].binding = 0;
    shadowCameraBindings[0].buffer = m_cameraBuffer;
    shadowCameraBindings[0].size = sizeof(CameraData);
    shadowCameraBindings[1].binding = 3;
    shadowCameraBindings[1].buffer = m_drawDataBuffer;
    shadowCameraBindings[1].size = m_drawDataCapacity;
    wgpu::BindGroupDescriptor shadowCameraGroupDescriptor;
    shadowCameraGroupDescriptor.layout = m_shadowCameraLayout;
    shadowCameraGroupDescriptor.entryCount = shadowCameraBindings.size();
    shadowCameraGroupDescriptor.entries = shadowCameraBindings.data();
    m_shadowCameraGroup = m_device.CreateBindGroup(&shadowCameraGroupDescriptor);
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

    std::array<wgpu::VertexAttribute, 14> attributes{};
    attributes[0].format = wgpu::VertexFormat::Float32x3;
    attributes[0].offset = offsetof(WebVertex, position);
    attributes[0].shaderLocation = 0;
    attributes[1].format = wgpu::VertexFormat::Float32x3;
    attributes[1].offset = offsetof(WebVertex, normal);
    attributes[1].shaderLocation = 1;
    attributes[2].format = wgpu::VertexFormat::Float32x2;
    attributes[2].offset = offsetof(WebVertex, uv);
    attributes[2].shaderLocation = 2;
    attributes[3].format = wgpu::VertexFormat::Float32x4;
    attributes[3].offset = offsetof(WebVertex, color);
    attributes[3].shaderLocation = 3;
    attributes[4].format = wgpu::VertexFormat::Float32x4;
    attributes[4].offset = offsetof(WebVertex, emission);
    attributes[4].shaderLocation = 4;
    attributes[5].format = wgpu::VertexFormat::Float32x4;
    attributes[5].offset = offsetof(WebVertex, material);
    attributes[5].shaderLocation = 5;
    attributes[6].format = wgpu::VertexFormat::Float32x4;
    attributes[6].offset = offsetof(WebVertex, surface);
    attributes[6].shaderLocation = 6;
    attributes[7].format = wgpu::VertexFormat::Float32x4;
    attributes[7].offset = offsetof(WebVertex, tangent);
    attributes[7].shaderLocation = 7;
    attributes[8].format = wgpu::VertexFormat::Float32x2;
    attributes[8].offset = offsetof(WebVertex, uv1);
    attributes[8].shaderLocation = 8;
    attributes[9].format = wgpu::VertexFormat::Float32x4;
    attributes[9].offset = offsetof(WebVertex, textureUvSets0);
    attributes[9].shaderLocation = 9;
    attributes[10].format = wgpu::VertexFormat::Float32x2;
    attributes[10].offset = offsetof(WebVertex, textureUvSets1);
    attributes[10].shaderLocation = 10;
    attributes[11].format = wgpu::VertexFormat::Float32x4;
    attributes[11].offset = offsetof(WebVertex, metallicChannels);
    attributes[11].shaderLocation = 11;
    attributes[12].format = wgpu::VertexFormat::Float32x4;
    attributes[12].offset = offsetof(WebVertex, smoothnessChannels);
    attributes[12].shaderLocation = 12;
    attributes[13].format = wgpu::VertexFormat::Float32x2;
    attributes[13].offset = offsetof(WebVertex, materialSampling);
    attributes[13].shaderLocation = 13;
    wgpu::VertexBufferLayout vertexLayout;
    vertexLayout.arrayStride = sizeof(WebVertex);
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
    const std::array<wgpu::BindGroupLayout, 2> sceneLayouts = {m_cameraLayout, m_materialTextureLayout};
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
    skyPipelineDescriptor.layout = pipelineDescriptor.layout;
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
    shadowPipelineLayoutDescriptor.bindGroupLayoutCount = 1;
    shadowPipelineLayoutDescriptor.bindGroupLayouts = &m_shadowCameraLayout;
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

bool WebSceneRenderer::BindIndexStream(wgpu::RenderPassEncoder pass, MeshIndexFormat format) const
{
    if (format == MeshIndexFormat::UInt16) {
        if (!m_indexBuffer16)
            return false;
        pass.SetIndexBuffer(m_indexBuffer16, wgpu::IndexFormat::Uint16, 0,
                            m_indexStreams.uint16Indices.size() * sizeof(uint16_t));
        return true;
    }
    if (format == MeshIndexFormat::UInt32) {
        if (!m_indexBuffer32)
            return false;
        pass.SetIndexBuffer(m_indexBuffer32, wgpu::IndexFormat::Uint32, 0,
                            m_indexStreams.uint32Indices.size() * sizeof(uint32_t));
        return true;
    }
    return false;
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
    m_extractor.ExtractCameraFrame(m_world, camera);
    const auto frame = m_world.Acquire();
    if (!frame || !frame->PrimaryView().valid) {
        ReportFrameIssue("invalid-render-snapshot");
        return false;
    }

    const SceneEnvironmentSettings &environment = scene->GetEnvironment();
    m_cameraData = {};
    m_cameraData.viewProjection = ToWebClipSpace(frame->PrimaryView().viewProjection);
    m_cameraData.inverseViewProjection = glm::inverse(m_cameraData.viewProjection);
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

    std::vector<WebVertex> vertices;
    WebPackedIndexStreams indexStreams;
    std::vector<WebDrawRange> drawRanges;
    std::vector<WebDrawData> drawData;
    const glm::mat4 cameraToWorld = glm::inverse(camera->GetViewMatrix());
    const glm::vec3 cameraRight = glm::normalize(glm::vec3(cameraToWorld[0]));
    const glm::vec3 cameraUp = glm::normalize(glm::vec3(cameraToWorld[1]));
    const glm::vec3 viewFacing = glm::normalize(-glm::vec3(cameraToWorld[2]));
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
        const size_t vertexBase = vertices.size();
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
        try {
            range.indices = PackWebIndexRange(indexStreams, draw.meshIndexFormat, vertexBase, sourceVertices.size(),
                                              sourceIndices, draw.indexStart, indexCount);
        } catch (const std::invalid_argument &) {
            ReportFrameIssue("invalid-index-stream");
            return false;
        } catch (const std::overflow_error &) {
            ReportFrameIssue("index-stream-overflow");
            return false;
        }
        const glm::mat3 worldNormal = glm::inverseTranspose(glm::mat3(draw.worldMatrix));
        const bool lineDraw =
            std::any_of(range.indices.sourceVertices.begin(), range.indices.sourceVertices.end(),
                        [&](uint32_t index) { return sourceVertices[index].boneIndices.w == kLineVertexMarker; });
        range.line = lineDraw;

        vertices.reserve(vertices.size() + range.indices.sourceVertices.size());
        for (const uint32_t index : range.indices.sourceVertices) {
            const Vertex &source = sourceVertices[index];
            const bool lineVertex = source.boneIndices.w == kLineVertexMarker;
            glm::vec3 renderPosition;
            glm::vec3 renderNormalValue;
            glm::vec4 renderTangentValue(1.0f, 0.0f, 0.0f, normalScale);
            float vertexAlpha = 1.0f;
            if (lineVertex) {
                const glm::vec3 center = glm::vec3(draw.worldMatrix * glm::vec4(source.pos, 1.0f));
                glm::vec3 tangent = glm::mat3(draw.worldMatrix) * glm::vec3(source.tangent);
                tangent = Finite(tangent) && glm::dot(tangent, tangent) > 1.0e-10f ? glm::normalize(tangent)
                                                                                   : glm::vec3(1.0f, 0.0f, 0.0f);
                glm::vec3 facing = source.boneIndices.z == 0u ? viewFacing : worldNormal * source.normal;
                facing = Finite(facing) && glm::dot(facing, facing) > 1.0e-10f ? glm::normalize(facing)
                                                                               : glm::vec3(0.0f, 0.0f, 1.0f);
                glm::vec3 fallbackSide = cameraRight - tangent * glm::dot(cameraRight, tangent);
                if (glm::dot(fallbackSide, fallbackSide) < 1.0e-8f)
                    fallbackSide = cameraUp - tangent * glm::dot(cameraUp, tangent);
                if (glm::dot(fallbackSide, fallbackSide) < 1.0e-10f)
                    fallbackSide = std::abs(tangent.x) < 0.9f ? glm::cross(tangent, glm::vec3(1.0f, 0.0f, 0.0f))
                                                              : glm::cross(tangent, glm::vec3(0.0f, 1.0f, 0.0f));
                fallbackSide = glm::normalize(fallbackSide);
                glm::vec3 geometricSide = glm::cross(facing, tangent);
                const float geometricLength = glm::length(geometricSide);
                glm::vec3 side;
                if (geometricLength > 0.20f) {
                    side = geometricSide / geometricLength;
                } else if (geometricLength > 1.0e-6f) {
                    geometricSide /= geometricLength;
                    if (glm::dot(fallbackSide, geometricSide) < 0.0f)
                        fallbackSide = -fallbackSide;
                    const float normalizedWeight = std::clamp((geometricLength - 0.025f) / 0.175f, 0.0f, 1.0f);
                    const float geometricWeight =
                        normalizedWeight * normalizedWeight * (3.0f - 2.0f * normalizedWeight);
                    side = glm::normalize(glm::mix(fallbackSide, geometricSide, geometricWeight));
                } else {
                    side = fallbackSide;
                }
                renderPosition = center + side * source.boneWeights.x;
                renderNormalValue = source.boneIndices.y != 0u ? facing : worldNormal * source.normal;
                renderTangentValue = glm::vec4(tangent, normalScale);
                vertexAlpha = std::clamp(source.boneWeights.y, 0.0f, 1.0f);
            } else {
                const glm::mat4 skin = SkinMatrix(source, draw.skinBoneMatrices.get());
                const glm::vec3 localPosition = glm::vec3(skin * glm::vec4(source.pos, 1.0f));
                renderPosition = localPosition;
                const glm::vec3 localNormal = glm::mat3(skin) * source.normal;
                renderNormalValue = localNormal;
                const glm::mat3 tangentTransform = glm::mat3(skin);
                glm::vec3 tangent = tangentTransform * glm::vec3(source.tangent);
                const float skinSign = glm::determinant(tangentTransform) < 0.0f ? -1.0f : 1.0f;
                const float worldSign = glm::determinant(glm::mat3(draw.worldMatrix)) < 0.0f ? -1.0f : 1.0f;
                const float handedness = source.tangent.w < 0.0f ? -1.0f : 1.0f;
                renderTangentValue = glm::vec4(tangent, handedness * skinSign * worldSign * normalScale);
            }
            if (!Finite(renderNormalValue) || glm::dot(renderNormalValue, renderNormalValue) < 1.0e-8f)
                renderNormalValue = glm::vec3(0.0f, 1.0f, 0.0f);
            else
                renderNormalValue = glm::normalize(renderNormalValue);
            glm::vec3 tangent = glm::vec3(renderTangentValue);
            tangent -= renderNormalValue * glm::dot(renderNormalValue, tangent);
            if (!Finite(tangent) || glm::dot(tangent, tangent) < 1.0e-8f) {
                const glm::vec3 axis =
                    std::abs(renderNormalValue.y) > 0.999f ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::vec3(0.0f, 1.0f, 0.0f);
                tangent = glm::normalize(glm::cross(axis, renderNormalValue));
            } else {
                tangent = glm::normalize(tangent);
            }
            renderTangentValue = glm::vec4(tangent, renderTangentValue.w);
            const glm::vec4 color = glm::vec4(source.color, vertexAlpha) * materialColor;
            range.transparent = range.transparent || color.a < 0.999f;
            WebVertex vertex{};
            std::memcpy(vertex.position, &renderPosition, sizeof(vertex.position));
            std::memcpy(vertex.normal, &renderNormalValue, sizeof(vertex.normal));
            std::memcpy(vertex.tangent, &renderTangentValue, sizeof(vertex.tangent));
            std::memcpy(vertex.uv, &source.texCoord, sizeof(vertex.uv));
            std::memcpy(vertex.uv1, &source.texCoord1, sizeof(vertex.uv1));
            std::memcpy(vertex.color, &color, sizeof(vertex.color));
            std::memcpy(vertex.emission, &emission, sizeof(vertex.emission));
            std::memcpy(vertex.material, &materialParameters, sizeof(vertex.material));
            std::memcpy(vertex.surface, &surfaceParameters, sizeof(vertex.surface));
            std::memcpy(vertex.textureUvSets0, &textureUvSets0, sizeof(vertex.textureUvSets0));
            std::memcpy(vertex.textureUvSets1, &textureUvSets1, sizeof(vertex.textureUvSets1));
            std::memcpy(vertex.metallicChannels, &metallicChannels, sizeof(vertex.metallicChannels));
            std::memcpy(vertex.smoothnessChannels, &smoothnessChannels, sizeof(vertex.smoothnessChannels));
            std::memcpy(vertex.materialSampling, &materialSampling, sizeof(vertex.materialSampling));
            vertices.push_back(vertex);
        }
        // LineRenderer alpha is carried per vertex rather than in the
        // material base color.  It must therefore use the transparent pass
        // even when the material itself is nominally opaque.
        if (range.line)
            range.transparent = true;

        if (range.indices.indexCount > 0) {
            WebDrawData objectData;
            if (!range.line) {
                objectData.model = draw.worldMatrix;
                objectData.normal = glm::mat4(glm::inverseTranspose(glm::mat3(draw.worldMatrix)));
            }
            range.drawIndex = static_cast<uint32_t>(drawData.size());
            drawData.push_back(objectData);
            drawRanges.push_back(std::move(range));
        }
    }

    size_t totalIndexCount = 0;
    for (const WebDrawRange &range : drawRanges)
        totalIndexCount += range.indices.indexCount;
    if (vertices.empty() || totalIndexCount == 0) {
        m_shadowEnabled = false;
        m_vertices.clear();
        m_indexStreams.uint16Indices.clear();
        m_indexStreams.uint32Indices.clear();
        m_drawRanges.clear();
        m_drawData.clear();
        m_residentSceneCount = sceneManager.GetSceneCount() + (sceneManager.GetRuntimePersistentScene() ? 1U : 0U);
        m_residentRendererCount = sceneManager.GetActiveMeshRenderers().size();
        m_residentLightCount = sceneManager.GetActiveLights().size();
        m_activeWorldId = scene->GetWorldId();
        return true;
    }

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

    glm::vec3 boundsMin(std::numeric_limits<float>::max());
    glm::vec3 boundsMax(std::numeric_limits<float>::lowest());
    for (const DrawCall &draw : drawCalls) {
        if (!draw.frustumVisible || !draw.meshVertices || !draw.meshIndices || draw.indexCount == 0)
            continue;
        boundsMin = glm::min(boundsMin, draw.worldBounds.min);
        boundsMax = glm::max(boundsMax, draw.worldBounds.max);
    }
    const glm::vec3 center = (boundsMin + boundsMax) * 0.5f;
    const float radius = std::max(2.0f, glm::length(boundsMax - boundsMin) * 0.6f);
    const glm::vec3 lightUp = std::abs(glm::dot(rayDirection, glm::vec3(0.0f, 1.0f, 0.0f))) > 0.95f
                                  ? glm::vec3(1.0f, 0.0f, 0.0f)
                                  : glm::vec3(0.0f, 1.0f, 0.0f);
    const glm::vec3 lightPosition = center - rayDirection * (radius * 2.0f);
    const glm::mat4 lightView = glm::lookAtRH(lightPosition, center, lightUp);
    const glm::mat4 lightProjection = glm::orthoRH_ZO(-radius, radius, -radius, radius, radius * 4.5f, 0.1f);
    m_cameraData.lightViewProjection = lightProjection * lightView;

    m_vertices.swap(vertices);
    m_indexStreams.uint16Indices.swap(indexStreams.uint16Indices);
    m_indexStreams.uint32Indices.swap(indexStreams.uint32Indices);
    m_drawRanges.swap(drawRanges);
    m_drawData.swap(drawData);
    m_residentSceneCount = sceneManager.GetSceneCount() + (sceneManager.GetRuntimePersistentScene() ? 1U : 0U);
    m_residentRendererCount = sceneManager.GetActiveMeshRenderers().size();
    m_residentLightCount = residentLights.size();
    m_activeWorldId = scene->GetWorldId();
    if (!m_lastFrameIssue.empty()) {
        std::printf("INFERNUX_WEB_SCENE_RENDER_RECOVERED previous=%s vertices=%zu indices=%zu\n",
                    m_lastFrameIssue.c_str(), m_vertices.size(), totalIndexCount);
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
        !BuildFrame(width, height, camera))
        return false;
    const uint64_t vertexBytes = m_vertices.size() * sizeof(WebVertex);
    const uint64_t indexBytes16 = m_indexStreams.uint16Indices.size() * sizeof(uint16_t);
    const uint64_t indexBytes32 = m_indexStreams.uint32Indices.size() * sizeof(uint32_t);
    const uint64_t drawDataBytes = m_drawData.size() * sizeof(WebDrawData);
    const uint64_t previousDrawDataCapacity = m_drawDataCapacity;
    if ((vertexBytes > 0 && !EnsureBuffer(m_vertexBuffer, m_vertexCapacity, vertexBytes, wgpu::BufferUsage::Vertex)) ||
        (indexBytes16 > 0 &&
         !EnsureBuffer(m_indexBuffer16, m_indexCapacity16, indexBytes16, wgpu::BufferUsage::Index)) ||
        (indexBytes32 > 0 &&
         !EnsureBuffer(m_indexBuffer32, m_indexCapacity32, indexBytes32, wgpu::BufferUsage::Index)) ||
        (drawDataBytes > 0 &&
         !EnsureBuffer(m_drawDataBuffer, m_drawDataCapacity, drawDataBytes, wgpu::BufferUsage::Storage)))
        return false;
    if (m_drawDataCapacity != previousDrawDataCapacity && !CreateCameraBindGroups())
        return false;
    if (vertexBytes > 0)
        m_queue.WriteBuffer(m_vertexBuffer, 0, m_vertices.data(), vertexBytes);
    if (indexBytes16 > 0)
        m_queue.WriteBuffer(m_indexBuffer16, 0, m_indexStreams.uint16Indices.data(), indexBytes16);
    if (indexBytes32 > 0)
        m_queue.WriteBuffer(m_indexBuffer32, 0, m_indexStreams.uint32Indices.data(), indexBytes32);
    if (drawDataBytes > 0)
        m_queue.WriteBuffer(m_drawDataBuffer, 0, m_drawData.data(), drawDataBytes);
    m_queue.WriteBuffer(m_cameraBuffer, 0, &m_cameraData, sizeof(m_cameraData));

    if (m_shadowEnabled) {
        wgpu::RenderPassDepthStencilAttachment depthAttachment;
        depthAttachment.view = m_shadowView;
        depthAttachment.depthLoadOp = wgpu::LoadOp::Clear;
        depthAttachment.depthStoreOp = wgpu::StoreOp::Store;
        depthAttachment.depthClearValue = 0.0f;
        wgpu::RenderPassDescriptor descriptor;
        descriptor.colorAttachmentCount = 0;
        descriptor.colorAttachments = nullptr;
        descriptor.depthStencilAttachment = &depthAttachment;
        wgpu::RenderPassEncoder shadowPass = encoder.BeginRenderPass(&descriptor);
        shadowPass.SetBindGroup(0, m_shadowCameraGroup);
        shadowPass.SetVertexBuffer(0, m_vertexBuffer, 0, vertexBytes);
        for (const WebDrawRange &range : m_drawRanges) {
            if (range.castsShadows && !range.fullyCulled) {
                shadowPass.SetPipeline(m_shadowPipelines[range.rasterPipeline]);
                if (!BindIndexStream(shadowPass, range.indices.format)) {
                    shadowPass.End();
                    return false;
                }
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
    const uint64_t vertexBytes = m_vertices.size() * sizeof(WebVertex);

    if (m_drawSky && m_skyPipeline) {
        pass.SetPipeline(m_skyPipeline);
        pass.SetBindGroup(0, m_cameraGroup);
        pass.SetBindGroup(1, m_skyTextureGroup);
        pass.Draw(3, 1, 0, 0);
    }

    // A camera plus World/Screen UI is a complete renderable scene.  Keep
    // the sky and UI pass alive even when the world has no mesh draw stream.
    if (m_vertices.empty() || m_drawRanges.empty())
        return true;

    pass.SetBindGroup(0, m_cameraGroup);
    pass.SetVertexBuffer(0, m_vertexBuffer, 0, vertexBytes);
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
        if (!BindIndexStream(pass, range.indices.format))
            return false;
        pass.DrawIndexed(range.indices.indexCount, 1, range.indices.firstIndex, range.indices.baseVertex,
                         range.drawIndex);
    }
    for (const WebDrawRange &range : m_drawRanges) {
        if (!range.transparent || range.fullyCulled)
            continue;
        pass.SetPipeline(m_transparentPipelines[range.rasterPipeline]);
        pass.SetBindGroup(1, range.materialTextureGroup ? range.materialTextureGroup : m_defaultMaterialTextureGroup);
        if (!BindIndexStream(pass, range.indices.format))
            return false;
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
                    m_vertices.size(), indexCount16 + indexCount32, indexCount16, indexCount32, m_drawRanges.size(),
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
