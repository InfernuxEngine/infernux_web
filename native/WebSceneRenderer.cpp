#include "WebSceneRenderer.h"
#include "WebGpuRhiDevice.h"
#include <function/renderer/lighting/ForwardPlusLightGrid.h>
#include <function/resources/ShaderAsset/SlangCompiler.h>
#include "WebRenderGraph.h"
#include <function/renderer/RendererSelection.h>
#include <function/resources/ShaderAsset/FullscreenProgramAsset.h>
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
#include <unordered_set>

namespace infernux::web
{
namespace
{
using Buffer = std::shared_ptr<rhi::BufferResource>;
using Texture = std::shared_ptr<const rhi::TextureResource>;

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
    rhi::BindingLayoutHandle viewLayout;
    uint32_t width = 0, height = 0;
    uint32_t shadowResolution = 0;
    Texture depth;
    Texture white, black, normal;
    bool skyEnabled = true, shadowsEnabled = true;
    uint64_t frameIndex = 0;
    EngineGlobalsUBO frameGlobals;
    SceneRenderExtractor extractor;
    SceneRenderer culler;
    RenderWorldSnapshot world;
    std::shared_ptr<const RenderWorldFrame> frame;
    SceneLightCollector lights;
    std::shared_ptr<const lighting::ForwardPlusGridPipeline> gridProgram;
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
        struct Program {
            std::shared_ptr<const ShaderProgramArtifact> artifact;
            RenderState state;
            MaterialPassPipelineDescriptor pass;
            std::unique_ptr<SceneMaterialPass> pipeline;
        };
        std::unordered_map<const GraphCommandDesc *, Program> programs;
    };
    struct ViewResources
    {
        uint64_t lastFrame = 0;
        Buffer camera, globals, lighting, canonicalLights, models, skins, bones, instances;
        std::vector<Buffer> shadowCameras, shadowParameters;
        Texture shadow;
        rhi::BindGroupHandle fullscreenGroup;
        std::array<rhi::BufferHandle, 6> fullscreenBuffers{};
        std::unique_ptr<lighting::ForwardPlusLightGrid> grid;
        bool usesGrid = false;
        rhi::TextureViewHandle fullscreenShadow;
        std::shared_ptr<const RenderGraphDescription> graph;
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
    std::vector<size_t> orderedDraws;

    ~Impl()
    {
        if (!device)
            return;
        for (auto &[id, view] : views)
            device->Release(view.fullscreenGroup);
        device->Release(viewLayout);
    }

    void PrepareFullscreenView()
    {
        const std::array<Buffer, 6> buffers{current->canonicalLights, dummy, dummy,
                                           current->lighting, current->camera, current->camera};
        std::array<rhi::BufferHandle, 6> handles;
        for (size_t index = 0; index < buffers.size(); ++index)
            handles[index] = buffers[index]->GetBuffer();
        if (current->usesGrid) {
            handles[1] = current->grid->Frame(0).headers;
            handles[2] = current->grid->Frame(0).lightMasks;
        }
        if (handles == current->fullscreenBuffers && current->fullscreenShadow == current->shadow->GetView())
            return;
        device->Release(current->fullscreenGroup);
        rhi::BindGroupDesc group;
        group.layout = viewLayout;
        group.textureCount = 2;
        group.textures[0] = {0, rhi::BindingType::SampledTexture, current->shadow->GetView(), {}, true};
        group.textures[1] = {6, rhi::BindingType::Sampler, {}, current->shadow->GetSampler(), false};
        const std::array<uint32_t, 6> bindings{1, 2, 3, 4, 5, 7};
        for (size_t index = 0; index < buffers.size(); ++index) {
            const auto &buffer = buffers[index];
            group.buffers[group.bufferCount++] = {bindings[index], index < 3 ? rhi::BindingType::StorageBuffer
                                                                           : rhi::BindingType::UniformBuffer,
                                                  handles[index], 0, buffer->GetByteSize()};
            if (current->usesGrid && (index == 1 || index == 2))
                group.buffers[group.bufferCount - 1].byteSize = index == 1
                    ? current->grid->Frame(0).headerCapacityBytes : current->grid->Frame(0).maskCapacityBytes;
        }
        current->fullscreenGroup = device->CreateBindGroup(group);
        current->fullscreenBuffers = handles;
        current->fullscreenShadow = current->shadow->GetView();
    }

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
                if (current->usesGrid) {
                    const auto &grid = current->grid->Frame(0);
                    addBuffer(binding.binding == 2 ? grid.headers : grid.lightMasks,
                              binding.binding == 2 ? grid.headerCapacityBytes : grid.maskCapacityBytes);
                } else
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
    void RecordDraw(const rhi::GraphicsCommandEncoder &commands, const Draw &draw,
                    const GraphCommandDesc &command, MaterialPassPipelineDescriptor descriptor,
                    uint32_t shadowView = UINT32_MAX)
    {
        const auto material = command.rendererSelection ? command.rendererSelection->Material()
                              : command.overrideMaterialValue ? command.overrideMaterialValue : draw.source.material;
        auto parameters = command.overrideMaterialValue ? nullptr : draw.source.parameterBlock;
        if (command.rendererSelection)
            parameters = *command.rendererSelection->Find(draw.source.identity);
        descriptor.target = command.shaderTarget;
        if (command.type == GraphCommandType::DrawShadowCasters)
            descriptor.target = ShaderCompileTarget::Shadow;
        else
            descriptor.invertCulling = frame->PrimaryView().invertCulling;
        auto &program = draw.material->programs[&command];
        if (!program.pipeline || program.artifact != material->GetSelectedShaderArtifact() ||
            !(program.state == material->GetRenderState()) || program.pass != descriptor) {
            program.artifact = material->GetSelectedShaderArtifact();
            program.state = material->GetRenderState();
            program.pass = descriptor;
            program.pipeline = std::make_unique<SceneMaterialPass>(*device, program.artifact, program.state, descriptor);
        }
        const uint32_t slot = shadowView == UINT32_MAX ? 0 : shadowView;
        Draw bound = draw;
        bound.source.material = material;
        bound.source.parameterBlock = parameters;
        if (!program.pipeline->Prepare(material, parameters,
                [&](const auto &resource, auto &group) { return BindResources(bound, shadowView, resource, group); }, slot))
            return;
        program.pipeline->Bind(commands, slot);
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

bool WebSceneRenderer::Initialize(WebGpuRhiDevice &device)
{
    auto &s = *m_impl;
    s.device = &device;
    rhi::BindingLayoutDesc viewLayout;
    viewLayout.entryCount = 8;
    for (uint32_t binding = 0; binding < 8; ++binding) {
        auto &entry = viewLayout.entries[binding];
        entry.binding = binding;
        entry.visibility = rhi::ShaderStage::Fragment;
        entry.type = binding == 0 ? rhi::BindingType::SampledTexture
                                 : binding <= 3 ? rhi::BindingType::StorageBuffer
                                 : binding == 6 ? rhi::BindingType::Sampler : rhi::BindingType::UniformBuffer;
        entry.depthRead = binding == 0;
        entry.readOnlyStorage = binding >= 1 && binding <= 3;
    }
    viewLayout.entries[5].visibility = rhi::ShaderStage::Vertex | rhi::ShaderStage::Fragment;
    viewLayout.entries[6].samplerType = rhi::SamplerBindingType::NonFiltering;
    viewLayout.entries[7].visibility = rhi::ShaderStage::Vertex;
    s.viewLayout = device.CreateBindingLayout(viewLayout);
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
    for (auto it = s.views.begin(); it != s.views.end();) {
        if (it->second.lastFrame + 1 < s.frameIndex) {
            s.device->Release(it->second.fullscreenGroup);
            it = s.views.erase(it);
        } else
            ++it;
    }
    for (auto &[id, view] : s.views)
        retireUnused(view.materials);
    s.current = nullptr;
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

std::shared_ptr<const RenderViewPlan> WebSceneRenderer::Capture(uint32_t width, uint32_t height,
                                                               Camera &camera, RenderPipelineCallback &pipeline)
{
    auto &s = *m_impl;
    camera.SetScreenDimensions(width, height);
    s.width = camera.GetPixelWidth();
    s.height = camera.GetPixelHeight();
    (void)s.extractor.ExtractCameraFrame(s.world, &camera);
    s.frame = s.world.Acquire();
    const auto &view = s.frame->PrimaryView();
    s.current = &s.views[view.cameraId];
    s.current->lastFrame = s.frameIndex;
    RenderViewCaptureServices services;
    services.culler = &s.culler;
    services.includeShadowCasters = s.shadowsEnabled;
    services.prepareMaterial = [](const auto &material) { ShaderLoader::PrepareMaterial(*material); };
    services.acquireProgram = [](const auto &name) { return LoadFullscreenProgram(name, ShaderCodeTarget::Wgsl); };
    ScriptableRenderContext context(s.frame, view, std::move(services));
    pipeline.RenderSelectedView(context, context.GetView());
    return context.CapturePlan();
}

void WebSceneRenderer::Prepare(const RenderViewPlan &plan, const WebRenderGraph &graph)
{
    auto &s = *m_impl;
    const auto &view = s.frame->PrimaryView();
    const uint32_t width = s.width, height = s.height;
    if (s.current->graph != plan.graph) {
        s.current->materials.clear();
        s.current->graph = plan.graph;
    }
    s.shadowResolution = 0;
    for (const auto &texture : plan.graph->textures)
        if (texture.IsViewDepth())
            s.depth = graph.GetTexture(texture.name);
    for (const auto &pass : plan.graph->passes)
        for (const auto &command : pass.commands)
            if (command.type == GraphCommandType::DrawShadowCasters) {
                s.current->shadow = graph.GetTexture(pass.writeDepth);
                s.shadowResolution = ResolveGraphPassExtent(*plan.graph, pass, width, height).first;
            }
    if (!s.current->shadow)
        s.current->shadow = s.Attachment(1, 1, 1, true);
    s.draws.clear();
    s.modelValues.clear();
    s.skinValues.clear();
    s.instanceValues.clear();
    s.boneValues.assign(1, glm::mat4(1));
    const auto &visible = plan.culling.visibleRenderers.DrawCalls();
    const auto &casters = plan.culling.shadowCasters.DrawCalls();
    std::unordered_set<RenderDrawIdentity, RenderDrawIdentityHash> visibleDraws;
    visibleDraws.reserve(visible.size());
    for (const auto &draw : visible) {
        visibleDraws.insert(draw.identity);
        s.AddDraw(draw);
    }
    // Camera visibility and camera layers do not remove a light's casters.
    for (const auto &draw : casters) {
        if (draw.castsShadows && !visibleDraws.count(draw.identity)) {
            auto shadowOnly = draw;
            shadowOnly.frustumVisible = false;
            s.AddDraw(shadowOnly);
        }
    }
    for (const auto &draw : plan.draws)
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
    globals.viewOptions = {view.dithering ? 1u : 0u, view.stopNaNs ? 1u : 0u, 1, 0};
    s.Uniform(s.current->globals, globals);
    s.lights.CollectLights(*s.frame->Lighting(), view.position);
    s.lights.ComputeShadowVP(*s.frame->Lighting(), view, s.shadowsEnabled ? s.shadowResolution : 0);
    s.lights.BuildShaderLightingUBO();
    s.Uniform(s.current->lighting, s.lights.GetShaderLightingUBO());
    const auto canonical = lighting::BuildCanonicalLightUpload(s.lights.GetCanonicalLightSnapshot());
    s.Upload(s.current->canonicalLights, canonical.bytes.data(), canonical.bytes.size(),
             rhi::BufferUsageFlags::Storage);
    s.current->usesGrid = GraphUsesForwardPlus(*plan.graph);
    if (s.current->usesGrid) {
        if (!s.gridProgram) {
            const auto code = SlangCompiler::Get().Compile(
                {"inx_forward_plus_grid", "Infernux/ForwardPlus.compute.py",
                 std::string(lighting::ForwardPlusLightGrid::ShaderSource())}, {}, {ShaderCodeTarget::Wgsl});
            s.gridProgram = lighting::ForwardPlusLightGrid::CreateProgram(*s.device, code.targets[0]);
        }
        if (!s.current->grid) {
            s.current->grid = std::make_unique<lighting::ForwardPlusLightGrid>();
            if (!s.current->grid->Initialize(s.gridProgram, 1))
                throw std::runtime_error("Forward+ view initialization failed");
        }
        if (!s.current->grid->PrepareFrame(0, width, height, canonical.header.countsAndGeneration.y,
                                           CanonicalLightAffectsGeometry, s.current->canonicalLights->GetBuffer()))
            throw std::runtime_error("Forward+ view preparation failed");
        for (const auto &retired : s.current->grid->TakeRetiredResources()) {
            s.device->Release(retired.bindGroup);
            s.device->Release(retired.consumerBindGroup);
            s.device->Release(retired.headers);
            s.device->Release(retired.lightMasks);
        }
    }
    s.PrepareFullscreenView();
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
}

void WebSceneRenderer::RecordPreparation(wgpu::CommandEncoder encoder)
{
    auto &s = *m_impl;
    if (!s.current->usesGrid)
        return;
    const auto &view = s.frame->PrimaryView();
    lighting::ForwardPlusGridConstants constants{};
    std::memcpy(constants.viewProjection, &view.viewProjection[0][0], sizeof(constants.viewProjection));
    constants.viewportAndProjectionScale[2] = std::abs(view.projection[0][0]);
    constants.viewportAndProjectionScale[3] = std::abs(view.projection[1][1]);
    auto pass = encoder.BeginComputePass();
    WebGpuComputeCommandContext context;
    s.current->grid->Record(0, s.device->MakeComputeCommandEncoder(context, pass), constants);
    pass.End();
}

void WebSceneRenderer::Record(wgpu::RenderPassEncoder pass, const GraphCommandDesc &command,
                               const MaterialPassPipelineDescriptor &pipeline)
{
    auto &s = *m_impl;
    const bool shadow = command.type == GraphCommandType::DrawShadowCasters;
    const bool sky = command.type == GraphCommandType::DrawSkybox;
    s.orderedDraws.clear();
    for (size_t index = 0; index < s.draws.size(); ++index) {
        const auto &draw = s.draws[index];
        if (sky != draw.sky || (shadow ? !draw.source.castsShadows : !draw.source.frustumVisible))
            continue;
        const auto &source = draw.source.material;
        if (!sky && (source->GetRenderQueue() < command.queueMin || source->GetRenderQueue() > command.queueMax))
            continue;
        if (command.rendererSelection && !command.rendererSelection->Find(draw.source.identity))
            continue;
        const auto &tag = source->GetPassTag();
        if (!sky && !command.passTag.empty() &&
            ((command.overrideMaterialValue && tag != command.passTag) ||
             (!command.overrideMaterialValue && !tag.empty() && tag != command.passTag)))
            continue;
        const auto &material = command.rendererSelection ? command.rendererSelection->Material()
                               : command.overrideMaterialValue ? command.overrideMaterialValue : source;
        const auto &artifact = material->GetSelectedShaderArtifact();
        if (shadow && !artifact->FindVariant(ShaderCompileTarget::Shadow))
            continue;
        const bool deferred = artifact->FindVariant(ShaderCompileTarget::GBuffer) != nullptr;
        if ((command.materialFilter == GraphMaterialFilter::DeferredCompatible && !deferred) ||
            (command.materialFilter == GraphMaterialFilter::DeferredUnsupported && deferred))
            continue;
        s.orderedDraws.push_back(index);
    }
    if (command.sortMode == "back_to_front") {
        const auto &view = s.frame->PrimaryView();
        for (const auto index : s.orderedDraws)
            if (s.draws[index].source.devicePose.value)
                throw std::invalid_argument("Device poses require a device-side transparent sort");
        std::stable_sort(s.orderedDraws.begin(), s.orderedDraws.end(), [&](size_t a, size_t b) {
            return (view.view * s.draws[a].source.worldMatrix[3]).z > (view.view * s.draws[b].source.worldMatrix[3]).z;
        });
    }
    WebGpuGraphicsCommandContext context;
    const auto commands = s.device->MakeGraphicsCommandEncoder(context, pass);
    if (shadow) {
        const auto &shadows = s.lights.GetShadowFrame();
        for (uint32_t index = 0; index < shadows.views.size(); ++index) {
            const auto &light = shadows.views[index];
            const auto &rect = light.atlas;
            pass.SetViewport(rect.x + rect.guard, rect.y + rect.guard, rect.InnerSize(), rect.InnerSize(), 0, 1);
            pass.SetScissorRect(rect.x + rect.guard, rect.y + rect.guard, rect.InnerSize(), rect.InnerSize());
            for (const auto draw : s.orderedDraws)
                if (s.draws[draw].source.layerMask & light.cullingMask)
                    s.RecordDraw(commands, s.draws[draw], command, pipeline, index);
        }
    } else {
        for (const auto index : s.orderedDraws)
            s.RecordDraw(commands, s.draws[index], command, pipeline);
    }
}

const RenderViewData &WebSceneRenderer::View() const noexcept
{
    return m_impl->frame->PrimaryView();
}

rhi::BindingLayoutHandle WebSceneRenderer::ViewLayout() const noexcept
{
    return m_impl->viewLayout;
}

FullscreenViewResources WebSceneRenderer::FullscreenView() const noexcept
{
    const auto &view = *m_impl->current;
    return {{0, rhi::BindingType::UniformBuffer, view.globals->GetBuffer(), 0, sizeof(EngineGlobalsUBO)},
            view.fullscreenGroup,
            {1, rhi::BindingType::StorageBuffer, view.canonicalLights->GetBuffer(), 0,
             view.canonicalLights->GetByteSize()}};
}

wgpu::TextureView WebSceneRenderer::GetDepthView() const noexcept
{
    return m_impl->depth ? m_impl->device->GetNativeTextureView(m_impl->depth->GetView()) : wgpu::TextureView{};
}
} // namespace infernux::web
