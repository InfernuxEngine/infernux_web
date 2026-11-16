#include "WebSceneRenderer.h"
#include "WebGpuRhiDevice.h"
#include "WebIndexPacking.h"
#include "WebTextureAssetLoad.h"
#include "WebTextureUpload.h"

#include <core/math/DepthConvention.h>
#include <core/types/ColorSpace.h>
#include <function/renderer/MaterialSamplerState.h>
#include <function/renderer/RenderInstanceHistory.h>
#include <function/renderer/RenderWorld.h>
#include <function/renderer/SceneMaterialPass.h>
#include <function/renderer/TextureUploadBuilder.h>
#include <function/renderer/lighting/CanonicalLightGpuBuffer.h>
#include <function/renderer/rhi/RhiBuffer.h>
#include <function/renderer/rhi/RhiComputeHost.h>
#include <function/renderer/rhi/RhiRenderTexture.h>
#include <function/resources/InxTexture/InxTexture.h>
#include <function/resources/ShaderAsset/ShaderLoader.h>
#include <function/scene/Camera.h>
#include <function/scene/LightingData.h>
#include <function/scene/PrimitiveMeshes.h>
#include <function/scene/SceneRenderExtractor.h>

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <unordered_map>

namespace infernux::web
{
namespace
{
using Buffer = std::shared_ptr<rhi::BufferResource>;
using Texture = std::shared_ptr<rhi::TextureResource>;

std::string SamplerIdentity(const MaterialTextureSampler &sampler)
{
    return std::to_string(static_cast<uint32_t>(sampler.minFilter)) + ":" +
           std::to_string(static_cast<uint32_t>(sampler.magFilter)) + ":" +
           std::to_string(static_cast<uint32_t>(sampler.mipFilter)) + ":" +
           std::to_string(static_cast<uint32_t>(sampler.addressU)) + ":" +
           std::to_string(static_cast<uint32_t>(sampler.addressV)) + ":" +
           std::to_string(static_cast<uint32_t>(sampler.addressW));
}

UniformBufferObject CameraUniform(const RenderViewData &view)
{
    UniformBufferObject value{};
    value.model = glm::mat4(1);
    value.view = view.view;
    value.proj = view.projection;
    value.previousViewProj = view.viewProjection;
    value.inverseViewProj = glm::inverse(view.viewProjection);
    value.projectionParams = {view.nearClip, view.farClip, 1.0f / view.farClip, view.nearClip / view.farClip};
    value.zBufferParams = depth::LinearizationParams(view.nearClip, view.farClip);
    return value;
}
} // namespace

struct WebSceneRenderer::Impl
{
    WebGpuRhiDevice *device = nullptr;
    rhi::PixelFormat colorFormat = rhi::PixelFormat::Undefined;
    uint32_t samples = 1, width = 0, height = 0;
    static constexpr uint32_t shadowResolution = 2048;
    Texture depth;
    Texture white, black, normal;
    wgpu::Color clear{};
    bool skyEnabled = true, shadowsEnabled = true, prepared = false, reported = false;
    uint64_t frameIndex = 0;
    EngineGlobalsUBO frameGlobals;
    SceneRenderExtractor extractor;
    RenderWorldSnapshot world;
    std::shared_ptr<const RenderWorldFrame> frame;
    SceneLightCollector lights;
    Buffer dummy;
    std::vector<glm::mat4> modelValues, boneValues;
    std::vector<GPUSkinInstanceData> skinValues;
    std::vector<GPUInstanceAuxData> instanceValues;

    struct TextureEntry
    {
        WebTextureAssetLoad load;
        Texture image;
    };
    std::unordered_map<std::string, TextureEntry> textures;
    struct Geometry
    {
        std::shared_ptr<const void> owner;
        const std::vector<uint32_t> *sourceIndices = nullptr;
        uint64_t revision = 0, lastFrame = 0;
        size_t vertexCount = 0, indexCount = 0;
        MeshIndexFormat sourceFormat = MeshIndexFormat::Auto;
        MeshIndexFormat format = MeshIndexFormat::Auto;
        Buffer vertices, indices;
    };
    std::unordered_map<const std::vector<Vertex> *, Geometry> geometry;
    struct Material
    {
        std::shared_ptr<const ShaderProgramArtifact> artifact;
        RenderState state;
        bool invertCulling = false;
        uint64_t lastFrame = 0;
        std::unique_ptr<SceneMaterialPass> forward, shadow;
    };
    struct ViewResources
    {
        uint64_t lastFrame = 0;
        Buffer camera, globals, lighting, canonicalLights, models, skins, bones, instances;
        std::vector<Buffer> shadowCameras, shadowParameters;
        Texture shadow;
        std::unordered_map<RenderDrawIdentity, Material, RenderDrawIdentityHash> materials;
    };
    std::unordered_map<uint64_t, ViewResources> views;
    ViewResources *current = nullptr;
    struct Draw
    {
        DrawCall source;
        Buffer vertices, indices;
        std::shared_ptr<rhi::ComputeBuffer> residentVertices;
        rhi::BufferHandle vertexBuffer;
        uint64_t vertexBytes = 0;
        MeshIndexFormat indexFormat;
        Material *material = nullptr;
        uint32_t instance = 0;
        bool sky = false, ready = false;
    };
    std::vector<Draw> draws;
    std::vector<size_t> transparentDraws;

    void Upload(Buffer &buffer, const void *data, uint64_t bytes, rhi::BufferUsageFlags usage)
    {
        if (!buffer || buffer->GetByteSize() < bytes) {
            rhi::BufferDesc description;
            description.byteSize = std::max<uint64_t>(256, bytes);
            description.usage = usage | rhi::BufferUsageFlags::TransferDestination;
            description.memory = rhi::BufferMemory::Upload;
            buffer =
                std::make_shared<rhi::BufferResource>(*device, device->CreateBuffer(description), description.byteSize);
        }
        if (!device->WriteBuffer(buffer->GetBuffer(), 0, data, bytes))
            throw std::runtime_error("Scene resource upload failed");
    }
    template <typename T> void Uniform(Buffer &buffer, const T &value)
    {
        Upload(buffer, &value, sizeof(value), rhi::BufferUsageFlags::Uniform);
    }
    template <typename T> void Storage(Buffer &buffer, const std::vector<T> &values)
    {
        Upload(buffer, values.data(), values.size() * sizeof(T), rhi::BufferUsageFlags::Storage);
    }
    Texture Attachment(uint32_t w, uint32_t h, uint32_t count, bool sampled)
    {
        rhi::TextureDesc description;
        description.width = w;
        description.height = h;
        description.format = rhi::PixelFormat::D32SFloat;
        description.samples = static_cast<rhi::SampleCount>(count);
        description.usage = rhi::TextureUsageFlags::DepthStencilAttachment | rhi::TextureUsageFlags::Sampled;
        const auto image = device->CreateTexture(description);
        rhi::TextureViewDesc view;
        view.texture = image;
        view.format = description.format;
        view.aspect = rhi::TextureAspect::Depth;
        rhi::SamplerHandle sampler;
        if (sampled) {
            rhi::SamplerDesc settings;
            settings.minFilter = settings.magFilter = settings.mipFilter = rhi::FilterMode::Nearest;
            settings.addressU = settings.addressV = settings.addressW = rhi::AddressMode::ClampToEdge;
            sampler = device->CreateSampler(settings);
        }
        return std::make_shared<rhi::TextureResource>(*device, image, device->CreateTextureView(view), sampler,
                                                      uint64_t(w) * h * 4 * count, description.format, view);
    }
    Texture UploadTexture(const TextureCpuData &data, const std::string &filter, const std::string &wrap,
                          int anisotropy, const MaterialTextureSampler &override = {})
    {
        std::optional<TextureCpuData> decoded;
        const auto *source = &data;
        if (TextureFormatIsBlockCompressed(data.format)) {
            decoded = DecodeBcTextureToRgba8(data);
            if (!decoded)
                throw std::runtime_error("Cannot decode imported scene texture");
            source = &*decoded;
        }
        auto sampler = MaterialSamplerDescription(filter, wrap, anisotropy, source->mipLevels.size(),
                                                  device->GetCapabilities(), override);
        TextureUploadBatch upload(*source, sampler);
        return device->UploadTexture(upload.GetRequest());
    }
    Texture ResolveTexture(const std::string &guid, const MaterialTextureSampler &sampler)
    {
        if (guid.empty() || guid == "white" || guid == "White")
            return white;
        if (guid == "black" || guid == "Black")
            return black;
        if (guid == "normal" || guid == "Normal")
            return normal;
        auto &entry = textures[guid + '\x1e' + SamplerIdentity(sampler)];
        switch (entry.load.Poll(guid)) {
        case WebTextureAssetLoad::Result::Ready: {
            auto data = entry.load.TakeData();
            entry.image = UploadTexture(data.texture, data.filterMode, data.wrapMode, data.anisoLevel, sampler);
            break;
        }
        case WebTextureAssetLoad::Result::Missing:
            entry.image.reset();
            break;
        default:
            break;
        }
        return entry.image;
    }
    rhi::BufferHandle AcceptedBuffer(const std::shared_ptr<rhi::ComputeBuffer> &value) const
    {
        if (value->GetHost().GetDeviceId() != device->GetDeviceId() || !value->GetLastWriteSubmission().IsValid())
            throw std::invalid_argument("Scene input must be an accepted value on the rendering device");
        return value->GetBuffer();
    }
    Geometry &ResolveGeometry(const DrawCall &draw)
    {
        auto [iterator, inserted] = geometry.try_emplace(draw.meshVertices);
        auto &result = iterator->second;
        if (inserted || result.owner != draw.meshDataOwner || result.revision != draw.meshRuntimeVersion ||
            result.sourceIndices != draw.meshIndices || result.sourceFormat != draw.meshIndexFormat ||
            result.vertexCount != draw.meshVertices->size() || result.indexCount != draw.meshIndices->size() ||
            (draw.forceBufferUpdate && result.lastFrame != frameIndex)) {
            result = {};
            result.owner = draw.meshDataOwner;
            result.sourceIndices = draw.meshIndices;
            result.revision = draw.meshRuntimeVersion;
            result.sourceFormat = draw.meshIndexFormat;
            result.vertexCount = draw.meshVertices->size();
            result.indexCount = draw.meshIndices->size();
            WebPackedIndexStreams packed;
            auto range = PackWebIndexRange(packed, draw.meshIndexFormat, 0, result.vertexCount, *draw.meshIndices, 0,
                                           result.indexCount);
            result.format = range.format;
            if (range.format == MeshIndexFormat::UInt16)
                Upload(result.indices, packed.uint16Indices.data(), packed.uint16Indices.size() * sizeof(uint16_t),
                       rhi::BufferUsageFlags::Index);
            else
                Upload(result.indices, packed.uint32Indices.data(), packed.uint32Indices.size() * sizeof(uint32_t),
                       rhi::BufferUsageFlags::Index);
        }
        if (!draw.meshVertexBuffer && !result.vertices)
            Upload(result.vertices, draw.meshVertices->data(), result.vertexCount * sizeof(Vertex),
                   rhi::BufferUsageFlags::Vertex);
        result.lastFrame = frameIndex;
        return result;
    }
    Material &ResolveMaterial(const DrawCall &draw, bool sky)
    {
        auto &result = current->materials[draw.identity];
        const auto &material = *draw.material;
        const bool invertCulling = frame->PrimaryView().invertCulling;
        if (result.artifact != material.GetSelectedShaderArtifact() || !(result.state == material.GetRenderState()) ||
            result.invertCulling != invertCulling) {
            result = {};
            result.artifact = material.GetSelectedShaderArtifact();
            result.state = material.GetRenderState();
            result.invertCulling = invertCulling;
            MaterialPassPipelineDescriptor pass;
            pass.colorFormats = {colorFormat};
            pass.depthFormat = rhi::PixelFormat::D32SFloat;
            pass.samples = static_cast<rhi::SampleCount>(samples);
            pass.invertCulling = invertCulling;
            result.forward = std::make_unique<SceneMaterialPass>(*device, result.artifact, result.state, pass);
            const bool hasShadow =
                std::any_of(result.artifact->variants.begin(), result.artifact->variants.end(),
                            [](const auto &variant) { return variant.target == ShaderCompileTarget::Shadow; });
            if (!sky && hasShadow) {
                pass.target = ShaderCompileTarget::Shadow;
                pass.colorFormats.clear();
                pass.samples = rhi::SampleCount::One;
                result.shadow = std::make_unique<SceneMaterialPass>(*device, result.artifact, result.state, pass);
            }
        }
        result.lastFrame = frameIndex;
        return result;
    }
    void AddDraw(const DrawCall &source, bool sky = false)
    {
        if (!source.meshVertices || !source.meshIndices || !source.indexCount || !source.material)
            return;
        if (source.material->GetRenderState().cullMode == MaterialCullMode::FrontAndBack)
            return;
        auto &mesh = ResolveGeometry(source);
        Draw draw;
        draw.source = source;
        draw.sky = sky;
        draw.material = &ResolveMaterial(source, sky);
        draw.vertices = mesh.vertices;
        draw.indices = mesh.indices;
        draw.indexFormat = mesh.format;
        draw.vertexBytes = mesh.vertexCount * sizeof(Vertex);
        if (source.meshVertexBuffer) {
            draw.residentVertices = source.meshVertexBuffer->Freeze();
            draw.vertexBuffer = AcceptedBuffer(draw.residentVertices);
        } else
            draw.vertexBuffer = mesh.vertices->GetBuffer();
        draw.instance = static_cast<uint32_t>(modelValues.size());
        modelValues.push_back(source.worldMatrix);
        GPUSkinInstanceData skin;
        if (source.skinBoneMatrices && !source.skinBoneMatrices->empty()) {
            skin.boneOffset = skin.previousBoneOffset = boneValues.size();
            skin.boneCount = source.skinBoneMatrices->size();
            skin.flags = kGPUSkinFlagEnabled;
            boneValues.insert(boneValues.end(), source.skinBoneMatrices->begin(), source.skinBoneMatrices->end());
        }
        skinValues.push_back(skin);
        GPUInstanceAuxData auxiliary;
        auxiliary.previousModel = source.worldMatrix;
        auxiliary.objectId = PackGPUObjectId(source.objectId);
        auxiliary.layerMask = source.layerMask;
        auxiliary.flags = source.receivesShadows ? 0 : kGPUInstanceAuxFlagNoReceiveShadows;
        ApplyDevicePose(auxiliary, source.devicePose);
        instanceValues.push_back(auxiliary);
        draws.push_back(std::move(draw));
    }
    bool BindResources(const Draw &draw, uint32_t shadowView, const ShaderResourceLayout &resource,
                       rhi::BindGroupDesc &group)
    {
        const auto &binding = resource.binding;
        const auto addBuffer = [&](rhi::BufferHandle handle, uint64_t bytes) {
            group.buffers[group.bufferCount++] = {binding.binding, binding.type, handle, 0, bytes};
        };
        const auto addOwned = [&](const Buffer &buffer) { addBuffer(buffer->GetBuffer(), buffer->GetByteSize()); };
        if (resource.set == 0) {
            const auto &material = *draw.source.material;
            const MaterialProperty *property = material.GetProperty(resource.name);
            if (draw.source.parameterBlock) {
                const auto found = draw.source.parameterBlock->properties.find(resource.name);
                if (found != draw.source.parameterBlock->properties.end())
                    property = &found->second;
            }
            if (binding.type == rhi::BindingType::StorageBuffer) {
                auto buffer = material.GetBuffer(resource.name);
                if (!buffer)
                    throw std::invalid_argument("Scene material buffer is not assigned: " + resource.name);
                addBuffer(AcceptedBuffer(buffer), buffer->GetByteSize());
                return true;
            }
            const auto &guid = std::get<std::string>(property->value);
            const auto *sampler = material.GetTextureSampler(resource.name);
            Texture texture;
            if (auto target = material.GetRenderTextureGeneration(resource.name))
                texture = target->color;
            else
                texture = ResolveTexture(guid.empty() ? std::string(material.GetTextureDefault(resource.name)) : guid,
                                         sampler ? *sampler : MaterialTextureSampler{});
            // Imported textures publish asynchronously. The caller skips a draw
            // until all its declared resources are available.
            if (!texture)
                return false;
            group.textures[group.textureCount++] = {binding.binding, binding.type, texture->GetView(),
                                                    texture->GetSampler(), binding.depthRead};
            return true;
        }
        if (resource.set == 1) {
            switch (binding.binding) {
            case 0:
            case 6:
                group.textures[group.textureCount++] = {binding.binding, binding.type, current->shadow->GetView(),
                                                        current->shadow->GetSampler(), binding.depthRead};
                return true;
            case 1:
                addOwned(current->canonicalLights);
                return true;
            case 2:
            case 3:
                addOwned(dummy);
                return true;
            case 4:
                addOwned(current->lighting);
                return true;
            case 5:
                addOwned(shadowView == UINT32_MAX ? current->camera : current->shadowCameras[shadowView]);
                return true;
            case 7:
                addOwned(current->shadowParameters[shadowView]);
                return true;
            }
        }
        if (resource.set == 2) {
            const auto &pose = draw.source.devicePose.value;
            switch (binding.binding) {
            case 0:
                addOwned(current->globals);
                return true;
            case 1:
            case 5:
                if (pose)
                    addBuffer(AcceptedBuffer(pose->models), pose->models->GetByteSize());
                else
                    addOwned(current->models);
                return true;
            case 2:
                addOwned(current->skins);
                return true;
            case 3:
                addOwned(current->bones);
                return true;
            case 4:
                addOwned(current->instances);
                return true;
            case 6:
                if (pose)
                    addBuffer(AcceptedBuffer(pose->bounds), pose->bounds->GetByteSize());
                else
                    addOwned(dummy);
                return true;
            }
        }
        throw std::invalid_argument("Unsupported scene resource: " + resource.name);
    }
    void RecordDraw(const rhi::GraphicsCommandEncoder &commands, const Draw &draw, uint32_t shadowView = UINT32_MAX)
    {
        auto &program = shadowView == UINT32_MAX ? draw.material->forward : draw.material->shadow;
        program->Bind(commands, shadowView == UINT32_MAX ? 0 : shadowView);
        commands.BindVertexBuffer(0, draw.vertexBuffer, 0, draw.vertexBytes);
        commands.BindIndexBuffer(draw.indices->GetBuffer(), draw.indexFormat == MeshIndexFormat::UInt16
                                                                ? rhi::IndexFormat::Uint16
                                                                : rhi::IndexFormat::Uint32);
        commands.DrawIndexed(draw.source.indexCount, 1, draw.source.indexStart, draw.source.vertexStart, draw.instance);
    }
};

WebSceneRenderer::WebSceneRenderer() : m_impl(std::make_unique<Impl>())
{
}
WebSceneRenderer::~WebSceneRenderer() = default;

bool WebSceneRenderer::Initialize(WebGpuRhiDevice &device, rhi::PixelFormat colorFormat, uint32_t samples)
{
    auto &s = *m_impl;
    s.device = &device;
    s.colorFormat = colorFormat;
    s.samples = samples;
    s.extractor.SetMaterialPreparation(
        [](const std::shared_ptr<InxMaterial> &material) { ShaderLoader::PrepareMaterial(*material); });
    const auto solid = [&](std::array<uint8_t, 4> pixel, bool srgb) {
        TextureCpuData data;
        data.dimension = TextureDimension::Texture2D;
        data.semantic = TextureSemantic::Color;
        data.format = srgb ? TextureFormat::Rgba8Srgb : TextureFormat::Rgba8UNorm;
        data.mipLevels.push_back({1, 1, 1, 0, 4, 4, 4});
        data.bytes.assign(pixel.begin(), pixel.end());
        return s.UploadTexture(data, "linear", "repeat", 1);
    };
    s.white = solid({255, 255, 255, 255}, true);
    s.black = solid({0, 0, 0, 255}, true);
    s.normal = solid({128, 128, 255, 255}, false);
    std::array<uint32_t, 16> zero{};
    s.Upload(s.dummy, zero.data(), sizeof(zero), rhi::BufferUsageFlags::Storage);
    return true;
}
void WebSceneRenderer::Resize(uint32_t width, uint32_t height)
{
    auto &s = *m_impl;
    if (s.depth && s.width == width && s.height == height)
        return;
    s.depth = s.Attachment(width, height, s.samples, false);
    s.width = width;
    s.height = height;
}
bool WebSceneRenderer::ConfigureSampleCount(uint32_t samples)
{
    auto &s = *m_impl;
    if (samples == s.samples)
        return true;
    s.samples = samples;
    for (auto &[id, view] : s.views)
        view.materials.clear();
    s.depth.reset();
    if (s.width && s.height)
        Resize(s.width, s.height);
    return true;
}
bool WebSceneRenderer::HasDepthTarget() const noexcept
{
    return bool(m_impl->depth);
}
wgpu::TextureView WebSceneRenderer::GetDepthView() const noexcept
{
    return m_impl->depth ? m_impl->device->GetNativeTextureView(m_impl->depth->GetView()) : wgpu::TextureView{};
}
wgpu::Color WebSceneRenderer::GetClearColor() const noexcept
{
    return m_impl->clear;
}
void WebSceneRenderer::StageFrame(const EngineGlobalsUBO &frame)
{
    auto &s = *m_impl;
    s.frameGlobals = frame;
    ++s.frameIndex;
    // Retire only after every view of the previous frame has been seen. A
    // camera must not evict another camera's geometry during its own Prepare.
    const auto retireUnused = [&](auto &values) {
        for (auto it = values.begin(); it != values.end();)
            it = it->second.lastFrame + 1 < s.frameIndex ? values.erase(it) : std::next(it);
    };
    retireUnused(s.geometry);
    retireUnused(s.views);
    for (auto &[id, view] : s.views)
        retireUnused(view.materials);
    s.current = nullptr;
    s.prepared = false;
}
void WebSceneRenderer::SetSkyEnabledForDiagnostics(bool enabled) noexcept
{
    m_impl->skyEnabled = enabled;
}
void WebSceneRenderer::SetShadowsEnabledForDiagnostics(bool enabled) noexcept
{
    m_impl->shadowsEnabled = enabled;
}
WebSceneRenderer::SampledTexture WebSceneRenderer::ResolveSampledTexture(const std::string &guid,
                                                                         const MaterialTextureSampler *sampler)
{
    auto &s = *m_impl;
    auto value = s.ResolveTexture(guid, sampler ? *sampler : MaterialTextureSampler{});
    return value ? SampledTexture{s.device->GetNativeTextureView(value->GetView()),
                                  s.device->GetNativeSampler(value->GetSampler())}
                 : SampledTexture{};
}

bool WebSceneRenderer::Prepare(wgpu::CommandEncoder encoder, uint32_t width, uint32_t height, Camera *camera)
{
    auto &s = *m_impl;
    s.prepared = false;
    if (!camera)
        return false;
    camera->SetAspectRatio(float(width) / float(height));
    (void)s.extractor.ExtractCameraFrame(s.world, camera);
    s.frame = s.world.Acquire();
    const auto &view = s.frame->PrimaryView();
    if (!view.valid)
        return false;
    const auto background = inx::color::SrgbToLinear(view.backgroundColor);
    s.clear = {background.r, background.g, background.b, background.a};
    s.current = &s.views[view.cameraId];
    s.current->lastFrame = s.frameIndex;
    if (!s.current->shadow)
        s.current->shadow = s.Attachment(Impl::shadowResolution, Impl::shadowResolution, 1, true);
    s.draws.clear();
    s.modelValues.clear();
    s.skinValues.clear();
    s.instanceValues.clear();
    s.boneValues.assign(1, glm::mat4(1));
    for (const auto &draw : s.frame->DrawCalls().drawCalls)
        if (draw.layerMask & view.cullingMask)
            s.AddDraw(draw);
    if (s.skyEnabled && view.clearFlags == CameraClearFlags::Skybox && s.frame->Lighting()->skyboxMaterial) {
        DrawCall sky;
        sky.material = s.frame->Lighting()->skyboxMaterial;
        sky.objectId = 0xFFFFFFFFFFFFFF00ULL;
        sky.identity = RenderProxyHandle::Synthetic(RenderDomain::Skybox, sky.objectId).MakeDrawIdentity();
        sky.meshVertices = &PrimitiveMeshes::GetSkyboxCubeVertices();
        sky.meshIndices = &PrimitiveMeshes::GetSkyboxCubeIndices();
        sky.indexCount = sky.meshIndices->size();
        sky.castsShadows = false;
        s.AddDraw(sky, true);
    }
    // Empty scenes still have a defined storage contract for the sky and view.
    if (s.modelValues.empty()) {
        s.modelValues.emplace_back(1);
        s.skinValues.emplace_back();
        s.instanceValues.emplace_back();
    }
    s.Storage(s.current->models, s.modelValues);
    s.Storage(s.current->skins, s.skinValues);
    s.Storage(s.current->bones, s.boneValues);
    s.Storage(s.current->instances, s.instanceValues);
    auto viewport = CameraUniform(view);
    s.Uniform(s.current->camera, viewport);
    auto globals = s.frameGlobals;
    globals.screenParams = {float(width), float(height), 1.0f / width, 1.0f / height};
    globals.projectionParams = viewport.projectionParams;
    s.Uniform(s.current->globals, globals);
    s.lights.CollectLights(*s.frame->Lighting(), view.position);
    s.lights.ComputeShadowVP(*s.frame->Lighting(), view, s.shadowsEnabled ? Impl::shadowResolution : 0);
    s.lights.BuildShaderLightingUBO();
    s.Uniform(s.current->lighting, s.lights.GetShaderLightingUBO());
    const auto canonical = lighting::BuildCanonicalLightUpload(s.lights.GetCanonicalLightSnapshot());
    s.Upload(s.current->canonicalLights, canonical.bytes.data(), canonical.bytes.size(),
             rhi::BufferUsageFlags::Storage);
    const auto &shadows = s.lights.GetShadowFrame();
    s.current->shadowCameras.resize(shadows.views.size());
    s.current->shadowParameters.resize(shadows.views.size());
    for (size_t i = 0; i < shadows.views.size(); ++i) {
        const auto &shadow = shadows.views[i];
        UniformBufferObject lightView{};
        lightView.model = lightView.view = glm::mat4(1);
        lightView.proj = shadow.viewProjection;
        lightView.previousViewProj = shadow.viewProjection;
        lightView.inverseViewProj = glm::inverse(shadow.viewProjection);
        s.Uniform(s.current->shadowCameras[i], lightView);
        lighting::ShadowPassParameters parameters;
        parameters.lightVector = glm::vec4(shadow.lightVector, shadow.lightVectorIsPosition ? 1.0f : 0.0f);
        parameters.bias = {shadow.depthBiasTexels, shadow.normalBiasTexels, shadow.worldUnitsPerTexel, shadow.farPlane};
        s.Uniform(s.current->shadowParameters[i], parameters);
    }
    for (auto &draw : s.draws) {
        draw.ready = draw.material->forward->Prepare(
            draw.source.material, draw.source.parameterBlock,
            [&](const auto &resource, auto &group) { return s.BindResources(draw, UINT32_MAX, resource, group); });
        if (draw.ready && draw.material->shadow)
            for (uint32_t i = 0; i < shadows.views.size(); ++i)
                draw.ready = draw.material->shadow->Prepare(
                    draw.source.material, draw.source.parameterBlock,
                    [&](const auto &resource, auto &group) { return s.BindResources(draw, i, resource, group); }, i);
    }
    s.transparentDraws.clear();
    for (size_t i = 0; i < s.draws.size(); ++i) {
        const auto &draw = s.draws[i];
        if (draw.ready && !draw.sky && draw.source.frustumVisible && draw.material->state.renderQueue >= 3000) {
            if (draw.source.devicePose.value)
                throw std::invalid_argument("Device poses require a device-side transparent sort");
            s.transparentDraws.push_back(i);
        }
    }
    // The extractor groups materials. Transparency additionally needs the
    // same left-handed view-depth order used by the native renderer.
    std::stable_sort(s.transparentDraws.begin(), s.transparentDraws.end(), [&](size_t a, size_t b) {
        return (view.view * s.draws[a].source.worldMatrix[3]).z > (view.view * s.draws[b].source.worldMatrix[3]).z;
    });
    // Every shadow type uses the engine's atlas assignment and authored shadow
    // pass. Alpha clipping, deformation and skinning therefore match the color pass.
    wgpu::RenderPassDepthStencilAttachment attachment{};
    attachment.view = s.device->GetNativeTextureView(s.current->shadow->GetView());
    attachment.depthLoadOp = wgpu::LoadOp::Clear;
    attachment.depthStoreOp = wgpu::StoreOp::Store;
    attachment.depthClearValue = 0.0;
    wgpu::RenderPassDescriptor descriptor{};
    descriptor.depthStencilAttachment = &attachment;
    auto pass = encoder.BeginRenderPass(&descriptor);
    WebGpuGraphicsCommandContext context;
    auto commands = s.device->MakeGraphicsCommandEncoder(context, pass);
    for (uint32_t i = 0; i < shadows.views.size(); ++i) {
        const auto &shadow = shadows.views[i];
        const auto &rect = shadow.atlas;
        pass.SetViewport(rect.x + rect.guard, rect.y + rect.guard, rect.InnerSize(), rect.InnerSize(), 0, 1);
        pass.SetScissorRect(rect.x + rect.guard, rect.y + rect.guard, rect.InnerSize(), rect.InnerSize());
        for (const auto &draw : s.draws)
            if (draw.ready && draw.source.castsShadows && draw.material->shadow &&
                (draw.source.layerMask & shadow.cullingMask))
                s.RecordDraw(commands, draw, i);
    }
    pass.End();
    s.prepared = true;
    return true;
}

bool WebSceneRenderer::RenderPrepared(wgpu::RenderPassEncoder pass)
{
    auto &s = *m_impl;
    if (!s.prepared)
        return false;
    WebGpuGraphicsCommandContext context;
    const auto commands = s.device->MakeGraphicsCommandEncoder(context, pass);
    // The sky writes far depth and is drawn after opaque geometry, before alpha.
    for (const auto &draw : s.draws)
        if (draw.ready && !draw.sky && draw.source.frustumVisible && draw.material->state.renderQueue < 3000)
            s.RecordDraw(commands, draw);
    for (const auto &draw : s.draws)
        if (draw.ready && draw.sky)
            s.RecordDraw(commands, draw);
    for (const auto index : s.transparentDraws)
        s.RecordDraw(commands, s.draws[index]);
    if (!s.reported) {
        std::printf("INFERNUX_WEB_SCENE_RENDER_READY draws=%zu material=authored executor=rhi shadows=%zu\n",
                    s.draws.size(), s.lights.GetShadowFrame().views.size());
        s.reported = true;
    }
    return true;
}
} // namespace infernux::web
