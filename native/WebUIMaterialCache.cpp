#include "WebUIMaterialCache.h"

#include <function/renderer/InxRenderStruct.h>
#include <function/renderer/shader/DeviceShaderProgram.h>
#include <function/renderer/shader/MaterialUniformLayout.h>
#include <function/resources/AssetDatabase/AssetDatabase.h>
#include <function/resources/AssetRegistry/AssetRegistry.h>
#include <function/resources/InxMaterial/InxMaterial.h>
#include <function/resources/ShaderAsset/ShaderAsset.h>
#include <function/resources/ShaderAsset/SlangNativeModule.h>
#include <function/resources/ShaderAsset/SlangUIModule.h>
#include <function/scene/Camera.h>
#include <glm/gtc/matrix_inverse.hpp>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace infernux::web
{
namespace
{
wgpu::Buffer UniformBuffer(wgpu::Device device, uint64_t size)
{
    wgpu::BufferDescriptor descriptor;
    descriptor.size = size;
    descriptor.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
    return device.CreateBuffer(&descriptor);
}

wgpu::BindGroupLayout UniformLayout(wgpu::Device device, uint32_t binding, uint64_t size)
{
    wgpu::BindGroupLayoutEntry entry;
    entry.binding = binding;
    entry.visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment;
    entry.buffer.type = wgpu::BufferBindingType::Uniform;
    entry.buffer.minBindingSize = size;
    wgpu::BindGroupLayoutDescriptor descriptor;
    descriptor.entryCount = 1;
    descriptor.entries = &entry;
    return device.CreateBindGroupLayout(&descriptor);
}

wgpu::BindGroup UniformGroup(wgpu::Device device, wgpu::BindGroupLayout layout, uint32_t binding, wgpu::Buffer buffer,
                             uint64_t size)
{
    wgpu::BindGroupEntry entry;
    entry.binding = binding;
    entry.buffer = buffer;
    entry.size = size;
    wgpu::BindGroupDescriptor descriptor;
    descriptor.layout = layout;
    descriptor.entryCount = 1;
    descriptor.entries = &entry;
    return device.CreateBindGroup(&descriptor);
}
} // namespace

struct WebUIMaterialCache::Prepared
{
    struct TextureField
    {
        std::string name, guid;
        uint32_t slot = 0;
        Texture texture;
    };
    struct Root
    {
        wgpu::Buffer buffer;
        wgpu::BindGroup group;
    };
    uint64_t generation = 0, shaderVersion = 0, checkedFrame = 0;
    std::string shaderGuid;
    std::shared_ptr<const ShaderTargetCode> code;
    std::shared_ptr<const InxMaterial> values;
    std::vector<ShaderResourceLayout> resources;
    std::vector<TextureField> textures;
    wgpu::Buffer uniform;
    uint64_t uniformBytes = 0, rootBytes = 0;
    wgpu::BindGroupLayout layout;
    std::array<wgpu::RenderPipeline, 2> pipelines;
    std::array<Root, 3> roots;
    bool texturesReady = false, world = false;
};

WebUIMaterialCache::WebUIMaterialCache() = default;
WebUIMaterialCache::~WebUIMaterialCache() = default;

void WebUIMaterialCache::Initialize(wgpu::Device device, wgpu::Queue queue, wgpu::TextureFormat screenFormat,
                                    wgpu::TextureFormat worldFormat, uint32_t worldSamples, TextureResolver textures)
{
    m_device = device;
    m_queue = queue;
    m_screenFormat = screenFormat;
    m_worldFormat = worldFormat;
    m_worldSamples = worldSamples;
    m_textures = std::move(textures);
    m_cameraBuffer = UniformBuffer(device, sizeof(UniformBufferObject));
    m_globalsBuffer = UniformBuffer(device, sizeof(EngineGlobalsUBO));
    m_cameraLayout = UniformLayout(device, 5, sizeof(UniformBufferObject));
    m_globalsLayout = UniformLayout(device, 0, sizeof(EngineGlobalsUBO));
    m_cameraGroup = UniformGroup(device, m_cameraLayout, 5, m_cameraBuffer, sizeof(UniformBufferObject));
    m_globalsGroup = UniformGroup(device, m_globalsLayout, 0, m_globalsBuffer, sizeof(EngineGlobalsUBO));
    std::array<wgpu::BindGroupLayoutEntry, 2> images{};
    images[0].binding = 0;
    images[0].visibility = wgpu::ShaderStage::Fragment;
    images[0].texture.sampleType = wgpu::TextureSampleType::Float;
    images[0].texture.viewDimension = wgpu::TextureViewDimension::e2D;
    images[1].binding = 1;
    images[1].visibility = wgpu::ShaderStage::Fragment;
    images[1].sampler.type = wgpu::SamplerBindingType::Filtering;
    wgpu::BindGroupLayoutDescriptor descriptor;
    descriptor.entryCount = images.size();
    descriptor.entries = images.data();
    m_textureLayout = device.CreateBindGroupLayout(&descriptor);
    for (size_t index = 0; index < m_defaults.size(); ++index) {
        const auto &source = index == 0 ? GetSlangScreenUIModule() : GetSlangWorldUIModule();
        auto code = std::make_shared<const ShaderTargetCode>(CompileSlangNativeModule(source, ShaderCodeTarget::Wgsl));
        auto prepared = std::make_unique<Prepared>();
        PrepareProgram(*prepared, code, index != 0);
        m_defaults[index] = std::move(prepared);
    }
}

void WebUIMaterialCache::ConfigureSampleCount(uint32_t samples)
{
    m_worldSamples = samples;
    m_defaults[1]->pipelines = {};
    for (auto &[key, prepared] : m_prepared)
        prepared->pipelines = {};
}

void WebUIMaterialCache::Prune(const std::unordered_set<std::string> &activeMaterials)
{
    for (auto entry = m_prepared.begin(); entry != m_prepared.end();) {
        if (activeMaterials.count(entry->first))
            ++entry;
        else
            entry = m_prepared.erase(entry);
    }
}

const wgpu::BindGroupLayout &WebUIMaterialCache::GetTextureLayout() const noexcept
{
    return m_textureLayout;
}

void WebUIMaterialCache::StageFrame(const EngineGlobalsUBO &frame, uint32_t width, uint32_t height)
{
    ++m_frame;
    m_frameGlobals = frame;
    m_frameGlobals.screenParams = glm::vec4(float(width), float(height), 1.0f / float(width), 1.0f / float(height));
    m_queue.WriteBuffer(m_globalsBuffer, 0, &m_frameGlobals, sizeof(m_frameGlobals));
    UniformBufferObject viewport{};
    viewport.model = viewport.view = viewport.proj = viewport.previousViewProj = viewport.inverseViewProj =
        glm::mat4(1);
    m_queue.WriteBuffer(m_cameraBuffer, 0, &viewport, sizeof(viewport));
}

void WebUIMaterialCache::StageCamera(Camera &camera)
{
    UniformBufferObject view{};
    view.model = glm::mat4(1);
    view.view = camera.GetViewMatrix();
    view.proj = camera.GetProjectionMatrix();
    view.previousViewProj = camera.GetViewProjectionMatrix();
    view.inverseViewProj = glm::inverse(view.previousViewProj);
    const float nearClip = camera.GetNearClip(), farClip = camera.GetFarClip();
    view.projectionParams = glm::vec4(nearClip, farClip, 1.0f / farClip, nearClip / farClip);
    view.zBufferParams = depth::LinearizationParams(nearClip, farClip);
    auto globals = m_frameGlobals;
    globals.worldSpaceCameraPos = glm::vec4(glm::vec3(glm::inverse(view.view)[3]), 1);
    globals.projectionParams = view.projectionParams;
    globals.zBufferParams = view.zBufferParams;
    globals.viewOptions.x = static_cast<uint32_t>(camera.GetDithering());
    globals.viewOptions.y = static_cast<uint32_t>(camera.GetStopNaNs());
    m_queue.WriteBuffer(m_cameraBuffer, 0, &view, sizeof(view));
    m_queue.WriteBuffer(m_globalsBuffer, 0, &globals, sizeof(globals));
}

void WebUIMaterialCache::PrepareProgram(Prepared &result, const std::shared_ptr<const ShaderTargetCode> &code,
                                        bool world)
{
    result.code = code;
    result.world = world;
    std::vector<ShaderStageSelection> stages;
    for (uint32_t index = 0; index < code->entries.size(); ++index)
        stages.push_back({*code, index});
    const auto linked = LinkShaderStages(stages);
    result.rootBytes = linked.rootConstants->byteSize;
    std::vector<wgpu::BindGroupLayoutEntry> entries;
    for (const auto &resource : linked.resources) {
        if (resource.set != 0 || resource.binding.binding == rhi::RootConstantBinding)
            continue;
        result.resources.push_back(resource);
        wgpu::BindGroupLayoutEntry entry;
        entry.binding = resource.binding.binding;
        entry.visibility = wgpu::ShaderStage::None;
        if (rhi::HasShaderStage(resource.binding.visibility, rhi::ShaderStage::Vertex))
            entry.visibility |= wgpu::ShaderStage::Vertex;
        if (rhi::HasShaderStage(resource.binding.visibility, rhi::ShaderStage::Fragment))
            entry.visibility |= wgpu::ShaderStage::Fragment;
        switch (resource.binding.type) {
        case rhi::BindingType::UniformBuffer:
            entry.buffer.type = wgpu::BufferBindingType::Uniform;
            entry.buffer.minBindingSize = resource.value.byteSize;
            result.uniformBytes = (resource.value.byteSize + 15u) & ~uint64_t{15u};
            break;
        case rhi::BindingType::SampledTexture:
            entry.texture.sampleType = wgpu::TextureSampleType::Float;
            entry.texture.viewDimension = wgpu::TextureViewDimension::e2D;
            break;
        case rhi::BindingType::Sampler:
            entry.sampler.type = wgpu::SamplerBindingType::Filtering;
            break;
        default:
            throw UIShaderMaterialError("Unsupported compiled UI material resource: " + resource.name);
        }
        entries.push_back(entry);
    }
    wgpu::BindGroupLayoutEntry root;
    root.binding = rhi::RootConstantBinding;
    root.visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment;
    root.buffer.type = wgpu::BufferBindingType::Uniform;
    root.buffer.hasDynamicOffset = true;
    root.buffer.minBindingSize = result.rootBytes;
    entries.push_back(root);
    wgpu::BindGroupLayoutDescriptor descriptor;
    descriptor.entryCount = entries.size();
    descriptor.entries = entries.data();
    result.layout = m_device.CreateBindGroupLayout(&descriptor);
}

WebUIMaterialCache::Prepared &WebUIMaterialCache::Resolve(const UIShaderMaterialBinding &binding, bool world)
{
    if (binding.materialGuid.empty())
        return *m_defaults[world ? 1 : 0];
    auto &registry = AssetRegistry::Instance();
    const auto material = registry.GetAsset<InxMaterial>(binding.materialGuid);
    if (!material || material->IsDeleted() || material->GetVersion() != binding.generation)
        throw UIShaderMaterialError("UI material generation changed: " + binding.materialGuid);
    const auto found = m_prepared.find(binding.materialGuid);
    if (found != m_prepared.end() && found->second->generation == binding.generation &&
        registry.GetAssetVersion(found->second->shaderGuid) == found->second->shaderVersion) {
        const auto &cached = found->second;
        if (cached->world != world)
            throw ShaderProgramDomainMismatch(world ? ShaderProgramDomain::WorldUI : ShaderProgramDomain::ScreenUI,
                                              cached->world ? ShaderProgramDomain::WorldUI
                                                            : ShaderProgramDomain::ScreenUI);
        return *cached;
    }
    const auto &reference = material->GetFragShaderReference();
    if (reference.guid.empty() && reference.shaderId.empty())
        return *m_defaults[world ? 1 : 0];
    auto *database = registry.GetAssetDatabase();
    const auto path = reference.guid.empty() ? database->FindShaderPathById(reference.shaderId, "fragment")
                                             : database->GetPathFromGuid(reference.guid);
    const auto shaderGuid = reference.guid.empty() ? database->GetGuidFromPath(path) : reference.guid;
    const auto shader = registry.LoadAsset<ShaderAsset>(shaderGuid, ResourceType::Shader);
    if (!shader)
        throw UIShaderMaterialError("UI fragment asset is unavailable: " + path);
    if (shader->programDomain != ShaderProgramDomain::ScreenUI && shader->programDomain != ShaderProgramDomain::WorldUI)
        return *m_defaults[world ? 1 : 0]; // Ordinary material supplies the shared UI tint/texture contract.
    const auto expected = world ? ShaderProgramDomain::WorldUI : ShaderProgramDomain::ScreenUI;
    if (shader->programDomain != expected)
        throw ShaderProgramDomainMismatch(expected, *shader->programDomain);
    if (!material->GetVertShaderReference().guid.empty() || !material->GetVertShaderReference().shaderId.empty())
        throw UIShaderMaterialError("UI geometry belongs to its producer; remove the material vertex selection");
    const auto stages =
        SelectMaterialShaderEntries(*shader->materialProgram, ShaderCompileTarget::Forward, ShaderCodeTarget::Wgsl);
    auto next = std::make_unique<Prepared>();
    PrepareProgram(*next, std::shared_ptr<const ShaderTargetCode>(shader->program, &stages.front().program), world);
    next->generation = binding.generation;
    next->shaderGuid = shaderGuid;
    next->shaderVersion = registry.GetAssetVersion(shaderGuid);
    auto values = std::make_shared<InxMaterial>(*material);
    values->SynchronizeShaderPropertyDefaults(*shader->materialProgram);
    next->values = values;
    if (next->uniformBytes) {
        std::vector<std::byte> bytes(next->uniformBytes);
        for (const auto &resource : next->resources) {
            if (resource.binding.type != rhi::BindingType::UniformBuffer)
                continue;
            MaterialUniformLayout(resource.value.fields).Write(bytes.data(), *values);
        }
        next->uniform = UniformBuffer(m_device, next->uniformBytes);
        m_queue.WriteBuffer(next->uniform, 0, bytes.data(), bytes.size());
    }
    for (const auto &property : shader->materialProgram->properties) {
        if (!property.textureSlot)
            continue;
        const auto slot = *property.textureSlot;
        if (std::none_of(next->resources.begin(), next->resources.end(), [slot](const auto &resource) {
                return resource.binding.type == rhi::BindingType::SampledTexture &&
                       resource.binding.binding == 1 + 2 * slot;
            }))
            continue;
        const auto &selected = std::get<std::string>(values->GetProperty(property.name)->value);
        next->textures.push_back({property.name,
                                  selected.empty() ? std::string(values->GetTextureDefault(property.name)) : selected,
                                  slot,
                                  {}});
    }
    const auto accepted = m_prepared.insert_or_assign(binding.materialGuid, std::move(next));
    return *accepted.first->second;
}

wgpu::RenderPipeline WebUIMaterialCache::CreatePipeline(const Prepared &prepared, bool world, bool alwaysOnTop)
{
    const auto entry = [&](rhi::ShaderStage stage) -> const ShaderEntryPointCode & {
        return *std::find_if(prepared.code->entries.begin(), prepared.code->entries.end(),
                             [stage](const auto &value) { return value.stage == stage; });
    };
    const auto module = [&](const ShaderEntryPointCode &code) {
        wgpu::ShaderSourceWGSL source;
        source.code = wgpu::StringView(code.code.data(), code.code.size());
        wgpu::ShaderModuleDescriptor descriptor;
        descriptor.nextInChain = &source;
        return m_device.CreateShaderModule(&descriptor);
    };
    const auto &vertex = entry(rhi::ShaderStage::Vertex), &fragment = entry(rhi::ShaderStage::Fragment);
    std::array<wgpu::VertexAttribute, 7> attributes{};
    const auto attribute = [&](size_t index, wgpu::VertexFormat format, uint64_t offset) {
        attributes[index].shaderLocation = static_cast<uint32_t>(index);
        attributes[index].format = format;
        attributes[index].offset = offset;
    };
    if (world) {
        attribute(0, wgpu::VertexFormat::Float32x3, offsetof(UIWorldVertex, position));
        attribute(1, wgpu::VertexFormat::Float32x2, offsetof(UIWorldVertex, uv));
        attribute(2, wgpu::VertexFormat::Float32x4, offsetof(UIWorldVertex, color));
        attribute(3, wgpu::VertexFormat::Float32x2, offsetof(UIWorldVertex, localPosition));
        attribute(4, wgpu::VertexFormat::Float32x3, offsetof(UIWorldVertex, anchor));
        attribute(5, wgpu::VertexFormat::Float32x3, offsetof(UIWorldVertex, localOffset));
        attribute(6, wgpu::VertexFormat::Float32, offsetof(UIWorldVertex, policy));
    } else {
        attribute(0, wgpu::VertexFormat::Float32x2, offsetof(UIQuadVertex, position));
        attribute(1, wgpu::VertexFormat::Float32x2, offsetof(UIQuadVertex, uv));
        attribute(2, wgpu::VertexFormat::Float32x4, offsetof(UIQuadVertex, color));
    }
    wgpu::VertexBufferLayout vertices;
    vertices.arrayStride = world ? sizeof(UIWorldVertex) : sizeof(UIQuadVertex);
    vertices.stepMode = wgpu::VertexStepMode::Vertex;
    vertices.attributeCount = world ? 7 : 3;
    vertices.attributes = attributes.data();
    wgpu::BlendState blend;
    blend.color.srcFactor = wgpu::BlendFactor::SrcAlpha;
    blend.color.dstFactor = wgpu::BlendFactor::OneMinusSrcAlpha;
    blend.alpha.srcFactor = wgpu::BlendFactor::One;
    blend.alpha.dstFactor = wgpu::BlendFactor::OneMinusSrcAlpha;
    wgpu::ColorTargetState color;
    color.format = world ? m_worldFormat : m_screenFormat;
    color.blend = &blend;
    color.writeMask = wgpu::ColorWriteMask::All;
    wgpu::FragmentState fragmentState;
    fragmentState.module = module(fragment);
    fragmentState.entryPoint = fragment.name.c_str();
    fragmentState.targetCount = 1;
    fragmentState.targets = &color;
    const std::array<wgpu::BindGroupLayout, 4> layouts{prepared.layout, m_cameraLayout, m_globalsLayout,
                                                       m_textureLayout};
    wgpu::PipelineLayoutDescriptor layout;
    layout.bindGroupLayoutCount = layouts.size();
    layout.bindGroupLayouts = layouts.data();
    wgpu::DepthStencilState depth;
    depth.format = wgpu::TextureFormat::Depth32Float;
    depth.depthWriteEnabled = wgpu::OptionalBool::False;
    depth.depthCompare = alwaysOnTop ? wgpu::CompareFunction::Always : wgpu::CompareFunction::GreaterEqual;
    wgpu::RenderPipelineDescriptor descriptor;
    descriptor.layout = m_device.CreatePipelineLayout(&layout);
    descriptor.vertex.module = module(vertex);
    descriptor.vertex.entryPoint = vertex.name.c_str();
    descriptor.vertex.bufferCount = 1;
    descriptor.vertex.buffers = &vertices;
    descriptor.fragment = &fragmentState;
    descriptor.primitive.topology = wgpu::PrimitiveTopology::TriangleList;
    descriptor.primitive.cullMode = wgpu::CullMode::None;
    descriptor.multisample.count = world ? m_worldSamples : 1;
    descriptor.depthStencil = world ? &depth : nullptr;
    return m_device.CreateRenderPipeline(&descriptor);
}

bool WebUIMaterialCache::RefreshTextures(Prepared &prepared)
{
    if (prepared.checkedFrame == m_frame)
        return prepared.texturesReady;
    prepared.checkedFrame = m_frame;
    prepared.texturesReady = true;
    for (auto &field : prepared.textures) {
        const auto texture = m_textures(field.guid, prepared.values->GetTextureSampler(field.name));
        if (!texture.view || !texture.sampler) {
            prepared.texturesReady = false;
            continue;
        }
        if (field.texture.view.Get() != texture.view.Get() || field.texture.sampler.Get() != texture.sampler.Get()) {
            field.texture = texture;
            prepared.roots = {};
        }
    }
    return prepared.texturesReady;
}

wgpu::BindGroup WebUIMaterialCache::RootGroup(Prepared &prepared, const wgpu::Buffer &constants, uint32_t rootOwner)
{
    auto &root = prepared.roots[rootOwner];
    if (root.buffer.Get() == constants.Get() && root.group)
        return root.group;
    std::vector<wgpu::BindGroupEntry> entries;
    for (const auto &resource : prepared.resources) {
        wgpu::BindGroupEntry entry;
        entry.binding = resource.binding.binding;
        if (resource.binding.type == rhi::BindingType::UniformBuffer) {
            entry.buffer = prepared.uniform;
            entry.size = prepared.uniformBytes;
        } else {
            const auto slot = (entry.binding - 1) / 2;
            const auto &texture =
                std::find_if(prepared.textures.begin(), prepared.textures.end(), [slot](const auto &value) {
                    return value.slot == slot;
                })->texture;
            if (resource.binding.type == rhi::BindingType::SampledTexture)
                entry.textureView = texture.view;
            else
                entry.sampler = texture.sampler;
        }
        entries.push_back(entry);
    }
    wgpu::BindGroupEntry values;
    values.binding = rhi::RootConstantBinding;
    values.buffer = constants;
    values.size = prepared.rootBytes;
    entries.push_back(values);
    wgpu::BindGroupDescriptor descriptor;
    descriptor.layout = prepared.layout;
    descriptor.entryCount = entries.size();
    descriptor.entries = entries.data();
    root.buffer = constants;
    root.group = m_device.CreateBindGroup(&descriptor);
    return root.group;
}

bool WebUIMaterialCache::Bind(wgpu::RenderPassEncoder pass, const UIShaderMaterialBinding &binding, bool world,
                              bool alwaysOnTop, const wgpu::Buffer &constants, uint32_t rootOwner,
                              uint32_t dynamicOffset)
{
    auto &prepared = Resolve(binding, world);
    if (!RefreshTextures(prepared))
        return false;
    auto &pipeline = prepared.pipelines[alwaysOnTop ? 1 : 0];
    if (!pipeline)
        pipeline = CreatePipeline(prepared, world, alwaysOnTop);
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, RootGroup(prepared, constants, rootOwner), 1, &dynamicOffset);
    pass.SetBindGroup(1, m_cameraGroup);
    pass.SetBindGroup(2, m_globalsGroup);
    return true;
}

} // namespace infernux::web
