#include "WebScreenUIRenderer.h"
#include "WebTextureUpload.h"

#include "InfernuxWebHostModule.h"
#include <function/renderer/gui/InxTextGeometry.h>
#include <function/renderer/gui/InxTextLayout.h>
#include <function/resources/AssetRegistry/AssetRegistry.h>
#include <function/resources/InxMaterial/InxMaterial.h>
#include <function/resources/InxTexture/InxTexture.h>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <imgui_internal.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

namespace infernux::web
{
namespace
{
constexpr uint64_t kFontTextureId = 1;
constexpr float kPi = 3.14159265358979323846f;

void CommandBoundary(const ImDrawList *, const ImDrawCmd *)
{
}

wgpu::SamplerDescriptor UISamplerDescriptor(const std::string &filterMode, const std::string &wrapMode, int anisoLevel,
                                            const MaterialTextureSampler &override)
{
    wgpu::SamplerDescriptor descriptor;
    const wgpu::AddressMode inheritedAddress = wrapMode == "repeat"   ? wgpu::AddressMode::Repeat
                                               : wrapMode == "mirror" ? wgpu::AddressMode::MirrorRepeat
                                                                      : wgpu::AddressMode::ClampToEdge;
    const auto address = [inheritedAddress](MaterialSamplerAddress value) {
        switch (value) {
        case MaterialSamplerAddress::Repeat:
            return wgpu::AddressMode::Repeat;
        case MaterialSamplerAddress::Clamp:
            return wgpu::AddressMode::ClampToEdge;
        case MaterialSamplerAddress::Mirror:
            return wgpu::AddressMode::MirrorRepeat;
        case MaterialSamplerAddress::Inherit:
            return inheritedAddress;
        }
        return inheritedAddress;
    };
    descriptor.addressModeU = address(override.addressU);
    descriptor.addressModeV = address(override.addressV);
    descriptor.addressModeW = address(override.addressW);
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
        return wgpu::FilterMode::Linear;
    };
    descriptor.minFilter = filter(override.minFilter);
    descriptor.magFilter = filter(override.magFilter);
    const uint16_t assetAnisotropy =
        point ? 1u : static_cast<uint16_t>(std::clamp(anisoLevel < 0 ? 16 : anisoLevel, 1, 16));
    const bool assetMipLinear = filterMode == "trilinear" || assetAnisotropy > 1u;
    switch (override.mipFilter) {
    case MaterialSamplerFilter::Nearest:
        descriptor.mipmapFilter = wgpu::MipmapFilterMode::Nearest;
        break;
    case MaterialSamplerFilter::Linear:
        descriptor.mipmapFilter = wgpu::MipmapFilterMode::Linear;
        break;
    case MaterialSamplerFilter::Inherit:
        descriptor.mipmapFilter = assetMipLinear ? wgpu::MipmapFilterMode::Linear : wgpu::MipmapFilterMode::Nearest;
        break;
    }
    const bool fullyLinear = descriptor.minFilter == wgpu::FilterMode::Linear &&
                             descriptor.magFilter == wgpu::FilterMode::Linear &&
                             descriptor.mipmapFilter == wgpu::MipmapFilterMode::Linear;
    descriptor.maxAnisotropy = fullyLinear ? assetAnisotropy : 1u;
    return descriptor;
}

constexpr const char *kScreenUIShader = R"(
struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
};
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
};
struct ScreenConstants {
    scale: vec2<f32>,
    translate: vec2<f32>,
    encode_sample: f32,
    padding0: f32,
    padding1: f32,
    padding2: f32,
    material_color: vec4<f32>,
    alpha_clip_threshold: f32,
    alpha_clip_enabled: f32,
    tail_padding: vec2<f32>,
};
@group(0) @binding(0) var ui_texture: texture_2d<f32>;
@group(0) @binding(500) var ui_sampler: sampler;
@group(0) @binding(999) var<uniform> pc: ScreenConstants;

@vertex fn vertex_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.position = vec4<f32>(input.position * pc.scale + pc.translate, 0.0, 1.0);
    output.uv = input.uv;
    output.color = input.color;
    return output;
}

@fragment fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let color = input.color * textureSample(ui_texture, ui_sampler, input.uv) * pc.material_color;
    if ((pc.alpha_clip_enabled > 0.5 && color.a < pc.alpha_clip_threshold) || color.a <= 0.0) {
        discard;
    }
    return color;
}
)";

constexpr const char *kWorldUIShader = R"(
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
    @location(3) local_position: vec2<f32>,
    @location(4) anchor: vec3<f32>,
    @location(5) local_offset: vec3<f32>,
    @location(6) policy: f32,
};
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) local_position: vec2<f32>,
};
struct WorldConstants {
    view_projection: mat4x4<f32>,
    material_color: vec4<f32>,
    alpha_clip_threshold: f32,
    alpha_clip_enabled: f32,
    screen_scale: vec2<f32>,
    camera_right: vec4<f32>,
    camera_up: vec4<f32>,
};
@group(0) @binding(0) var ui_texture: texture_2d<f32>;
@group(0) @binding(500) var ui_sampler: sampler;
@group(0) @binding(999) var<uniform> pc: WorldConstants;

@vertex fn vertex_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.color = input.color;
    output.uv = input.uv;
    output.local_position = input.local_position;
    var position = input.position;
    let policy = u32(input.policy + 0.5);
    if (policy != 0u) {
        var offset = input.position - input.anchor;
        if ((policy & 1u) != 0u) {
            offset = pc.camera_right.xyz * input.local_offset.x
                   + pc.camera_up.xyz * input.local_offset.y
                   + cross(pc.camera_right.xyz, pc.camera_up.xyz) * input.local_offset.z;
        }
        if ((policy & 2u) != 0u) {
            let clip_w = (pc.view_projection * vec4<f32>(input.anchor, 1.0)).w;
            offset *= max(clip_w, 0.0) * pc.screen_scale.x * 100.0;
        }
        position = input.anchor + offset;
    }
    output.position = pc.view_projection * vec4<f32>(position, 1.0);
    return output;
}
@fragment fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let color = input.color * textureSample(ui_texture, ui_sampler, input.uv)
                * pc.material_color;
    if ((pc.alpha_clip_enabled > 0.5 && color.a < pc.alpha_clip_threshold)
        || color.a <= 0.0) {
        discard;
    }
    return color;
}
)";

uint64_t GrowCapacity(uint64_t required)
{
    uint64_t capacity = 4096;
    while (capacity < required)
        capacity *= 2;
    return capacity;
}

void ResetDrawList(ImDrawList &drawList, uint32_t width, uint32_t height)
{
    drawList._ResetForNewFrame();
    drawList.PushTextureID(ImGui::GetIO().Fonts->TexRef);
    drawList.PushClipRect({0.0f, 0.0f}, {static_cast<float>(width), static_cast<float>(height)});
}

void TransformVertices(ImDrawList &drawList, int firstVertex, float minX, float minY, float maxX, float maxY,
                       float rotation, bool mirrorH, bool mirrorV)
{
    rotation = std::fmod(rotation, 360.0f);
    if (!mirrorH && !mirrorV && std::abs(rotation) < 0.001f)
        return;
    const float radians = rotation * kPi / 180.0f;
    const float cosine = std::cos(radians);
    const float sine = std::sin(radians);
    const ImVec2 pivot{(minX + maxX) * 0.5f, (minY + maxY) * 0.5f};
    for (int index = firstVertex; index < drawList.VtxBuffer.Size; ++index) {
        ImVec2 local{drawList.VtxBuffer[index].pos.x - pivot.x, drawList.VtxBuffer[index].pos.y - pivot.y};
        if (mirrorH)
            local.x = -local.x;
        if (mirrorV)
            local.y = -local.y;
        drawList.VtxBuffer[index].pos = {pivot.x + local.x * cosine - local.y * sine,
                                         pivot.y + local.x * sine + local.y * cosine};
    }
}
} // namespace

WebScreenUIRenderer::~WebScreenUIRenderer()
{
    if (m_context)
        ImGui::SetCurrentContext(m_context);
    if (m_camera)
        IM_DELETE(m_camera);
    if (m_overlay)
        IM_DELETE(m_overlay);
    if (m_world)
        IM_DELETE(m_world);
    if (m_ownsContext && m_context)
        ImGui::DestroyContext(m_context);
}

bool WebScreenUIRenderer::Initialize(wgpu::Device device, wgpu::Queue queue, wgpu::TextureFormat colorFormat,
                                     wgpu::TextureFormat worldColorFormat, uint32_t worldSampleCount)
{
    m_device = device;
    m_queue = queue;
    m_colorFormat = colorFormat;
    m_worldColorFormat = worldColorFormat;
    m_worldSampleCount = worldSampleCount;
    m_context = ImGui::GetCurrentContext();
    if (!m_context) {
        m_context = ImGui::CreateContext();
        m_ownsContext = true;
    }
    ImGui::SetCurrentContext(m_context);
    ImGuiIO &io = ImGui::GetIO();
    io.DisplaySize = {1280.0f, 720.0f};
    io.DeltaTime = 1.0f / 60.0f;
    unsigned char *pixels = nullptr;
    int atlasWidth = 0;
    int atlasHeight = 0;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &atlasWidth, &atlasHeight);
    if (!pixels || atlasWidth < 1 || atlasHeight < 1)
        return false;
    io.Fonts->SetTexID(static_cast<ImTextureID>(kFontTextureId));
    ImGui::NewFrame();
    ImGui::EndFrame();
    m_camera = IM_NEW(ImDrawList)(ImGui::GetDrawListSharedData());
    m_overlay = IM_NEW(ImDrawList)(ImGui::GetDrawListSharedData());
    m_world = IM_NEW(ImDrawList)(ImGui::GetDrawListSharedData());

    wgpu::TextureDescriptor textureDescriptor;
    textureDescriptor.dimension = wgpu::TextureDimension::e2D;
    textureDescriptor.size = {static_cast<uint32_t>(atlasWidth), static_cast<uint32_t>(atlasHeight), 1};
    textureDescriptor.format = wgpu::TextureFormat::RGBA8Unorm;
    textureDescriptor.mipLevelCount = 1;
    textureDescriptor.sampleCount = 1;
    textureDescriptor.usage = wgpu::TextureUsage::CopyDst | wgpu::TextureUsage::TextureBinding;
    m_fontTexture = m_device.CreateTexture(&textureDescriptor);
    m_fontWidth = static_cast<uint32_t>(atlasWidth);
    m_fontHeight = static_cast<uint32_t>(atlasHeight);
    m_fontView = m_fontTexture ? m_fontTexture.CreateView() : wgpu::TextureView{};
    wgpu::TexelCopyTextureInfo destination;
    destination.texture = m_fontTexture;
    wgpu::TexelCopyBufferLayout layout;
    layout.bytesPerRow = static_cast<uint32_t>(atlasWidth * 4);
    layout.rowsPerImage = static_cast<uint32_t>(atlasHeight);
    const wgpu::Extent3D extent{static_cast<uint32_t>(atlasWidth), static_cast<uint32_t>(atlasHeight), 1};
    m_queue.WriteTexture(&destination, pixels, static_cast<size_t>(atlasWidth * atlasHeight * 4), &layout, &extent);

    wgpu::SamplerDescriptor samplerDescriptor;
    samplerDescriptor.addressModeU = wgpu::AddressMode::ClampToEdge;
    samplerDescriptor.addressModeV = wgpu::AddressMode::ClampToEdge;
    samplerDescriptor.minFilter = wgpu::FilterMode::Linear;
    samplerDescriptor.magFilter = wgpu::FilterMode::Linear;
    m_fontSampler = m_device.CreateSampler(&samplerDescriptor);
    return m_fontTexture && m_fontView && m_fontSampler && CreatePipelineAndFontAtlas() && CreateWorldPipelines();
}

bool WebScreenUIRenderer::ConfigureSampleCount(uint32_t sampleCount)
{
    if (sampleCount != 1 && sampleCount != 4)
        return false;
    if (sampleCount == m_worldSampleCount)
        return true;
    m_worldSampleCount = sampleCount;
    m_customWorldPipelines.clear();
    return CreateWorldPipelines();
}

bool WebScreenUIRenderer::RefreshFontAtlas()
{
    if (!m_fontAtlasDirty || !m_context || !m_device || !m_queue || !m_textureLayout)
        return true;
    ImGui::SetCurrentContext(m_context);
    unsigned char *pixels = nullptr;
    int atlasWidth = 0;
    int atlasHeight = 0;
    ImGui::GetIO().Fonts->GetTexDataAsRGBA32(&pixels, &atlasWidth, &atlasHeight);
    if (!pixels || atlasWidth < 1 || atlasHeight < 1)
        return false;

    const uint32_t width = static_cast<uint32_t>(atlasWidth);
    const uint32_t height = static_cast<uint32_t>(atlasHeight);
    if (!m_fontTexture || width != m_fontWidth || height != m_fontHeight) {
        wgpu::TextureDescriptor textureDescriptor;
        textureDescriptor.dimension = wgpu::TextureDimension::e2D;
        textureDescriptor.size = {width, height, 1};
        textureDescriptor.format = wgpu::TextureFormat::RGBA8Unorm;
        textureDescriptor.mipLevelCount = 1;
        textureDescriptor.sampleCount = 1;
        textureDescriptor.usage = wgpu::TextureUsage::CopyDst | wgpu::TextureUsage::TextureBinding;
        m_fontTexture = m_device.CreateTexture(&textureDescriptor);
        m_fontView = m_fontTexture ? m_fontTexture.CreateView() : wgpu::TextureView{};
        m_fontWidth = width;
        m_fontHeight = height;
    }
    if (!m_fontTexture || !m_fontView)
        return false;

    wgpu::TexelCopyTextureInfo destination;
    destination.texture = m_fontTexture;
    wgpu::TexelCopyBufferLayout layout;
    layout.bytesPerRow = width * 4;
    layout.rowsPerImage = height;
    const wgpu::Extent3D extent{width, height, 1};
    m_queue.WriteTexture(&destination, pixels, static_cast<size_t>(width) * height * 4, &layout, &extent);

    std::array<wgpu::BindGroupEntry, 2> groupEntries{};
    groupEntries[0].binding = 0;
    groupEntries[0].sampler = m_fontSampler;
    groupEntries[1].binding = 1;
    groupEntries[1].textureView = m_fontView;
    wgpu::BindGroupDescriptor groupDescriptor;
    groupDescriptor.layout = m_textureLayout;
    groupDescriptor.entryCount = groupEntries.size();
    groupDescriptor.entries = groupEntries.data();
    m_fontGroup = m_device.CreateBindGroup(&groupDescriptor);
    ImGui::GetIO().Fonts->SetTexID(static_cast<ImTextureID>(kFontTextureId));
    m_fontAtlasDirty = false;
    return static_cast<bool>(m_fontGroup);
}

bool WebScreenUIRenderer::CreatePipelineAndFontAtlas()
{
    std::array<wgpu::BindGroupLayoutEntry, 2> layoutEntries{};
    layoutEntries[0].binding = 0;
    layoutEntries[0].visibility = wgpu::ShaderStage::Fragment;
    layoutEntries[0].sampler.type = wgpu::SamplerBindingType::Filtering;
    layoutEntries[1].binding = 1;
    layoutEntries[1].visibility = wgpu::ShaderStage::Fragment;
    layoutEntries[1].texture.sampleType = wgpu::TextureSampleType::Float;
    layoutEntries[1].texture.viewDimension = wgpu::TextureViewDimension::e2D;
    wgpu::BindGroupLayoutDescriptor layoutDescriptor;
    layoutDescriptor.entryCount = layoutEntries.size();
    layoutDescriptor.entries = layoutEntries.data();
    m_textureLayout = m_device.CreateBindGroupLayout(&layoutDescriptor);

    std::array<wgpu::BindGroupEntry, 2> groupEntries{};
    groupEntries[0].binding = 0;
    groupEntries[0].sampler = m_fontSampler;
    groupEntries[1].binding = 1;
    groupEntries[1].textureView = m_fontView;
    wgpu::BindGroupDescriptor groupDescriptor;
    groupDescriptor.layout = m_textureLayout;
    groupDescriptor.entryCount = groupEntries.size();
    groupDescriptor.entries = groupEntries.data();
    m_fontGroup = m_device.CreateBindGroup(&groupDescriptor);

    std::array<wgpu::BindGroupLayoutEntry, 3> screenBindings{};
    screenBindings[0].binding = 0;
    screenBindings[0].visibility = wgpu::ShaderStage::Fragment;
    screenBindings[0].texture.sampleType = wgpu::TextureSampleType::Float;
    screenBindings[0].texture.viewDimension = wgpu::TextureViewDimension::e2D;
    screenBindings[1].binding = 500;
    screenBindings[1].visibility = wgpu::ShaderStage::Fragment;
    screenBindings[1].sampler.type = wgpu::SamplerBindingType::Filtering;
    screenBindings[2].binding = 999;
    screenBindings[2].visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment;
    screenBindings[2].buffer.type = wgpu::BufferBindingType::Uniform;
    screenBindings[2].buffer.hasDynamicOffset = true;
    screenBindings[2].buffer.minBindingSize = sizeof(ScreenConstants);
    wgpu::BindGroupLayoutDescriptor screenLayoutDescriptor;
    screenLayoutDescriptor.entryCount = screenBindings.size();
    screenLayoutDescriptor.entries = screenBindings.data();
    m_screenLayout = m_device.CreateBindGroupLayout(&screenLayoutDescriptor);

    wgpu::ShaderSourceWGSL shaderSource;
    shaderSource.code = kScreenUIShader;
    wgpu::ShaderModuleDescriptor shaderDescriptor;
    shaderDescriptor.nextInChain = &shaderSource;
    const wgpu::ShaderModule shader = m_device.CreateShaderModule(&shaderDescriptor);

    std::array<wgpu::VertexAttribute, 3> attributes{};
    attributes[0].format = wgpu::VertexFormat::Float32x2;
    attributes[0].offset = offsetof(GPUVertex, position);
    attributes[0].shaderLocation = 0;
    attributes[1].format = wgpu::VertexFormat::Float32x2;
    attributes[1].offset = offsetof(GPUVertex, uv);
    attributes[1].shaderLocation = 1;
    attributes[2].format = wgpu::VertexFormat::Float32x4;
    attributes[2].offset = offsetof(GPUVertex, color);
    attributes[2].shaderLocation = 2;
    wgpu::VertexBufferLayout vertexLayout;
    vertexLayout.arrayStride = sizeof(GPUVertex);
    vertexLayout.stepMode = wgpu::VertexStepMode::Vertex;
    vertexLayout.attributeCount = attributes.size();
    vertexLayout.attributes = attributes.data();

    wgpu::BlendState blend;
    blend.color.srcFactor = wgpu::BlendFactor::SrcAlpha;
    blend.color.dstFactor = wgpu::BlendFactor::OneMinusSrcAlpha;
    blend.alpha.srcFactor = wgpu::BlendFactor::One;
    blend.alpha.dstFactor = wgpu::BlendFactor::OneMinusSrcAlpha;
    wgpu::ColorTargetState colorTarget;
    colorTarget.format = m_colorFormat;
    colorTarget.blend = &blend;
    colorTarget.writeMask = wgpu::ColorWriteMask::All;
    wgpu::FragmentState fragment;
    fragment.module = shader;
    fragment.entryPoint = "fragment_main";
    fragment.targetCount = 1;
    fragment.targets = &colorTarget;
    wgpu::PipelineLayoutDescriptor pipelineLayoutDescriptor;
    pipelineLayoutDescriptor.bindGroupLayoutCount = 1;
    pipelineLayoutDescriptor.bindGroupLayouts = &m_screenLayout;
    wgpu::RenderPipelineDescriptor descriptor;
    descriptor.layout = m_device.CreatePipelineLayout(&pipelineLayoutDescriptor);
    descriptor.vertex.module = shader;
    descriptor.vertex.entryPoint = "vertex_main";
    descriptor.vertex.bufferCount = 1;
    descriptor.vertex.buffers = &vertexLayout;
    descriptor.fragment = &fragment;
    descriptor.primitive.topology = wgpu::PrimitiveTopology::TriangleList;
    descriptor.primitive.cullMode = wgpu::CullMode::None;
    descriptor.multisample.count = 1;
    m_pipeline = m_device.CreateRenderPipeline(&descriptor);
    return m_textureLayout && m_screenLayout && m_fontGroup && m_pipeline;
}

bool WebScreenUIRenderer::CreateWorldPipelines()
{
    if (m_worldColorFormat == wgpu::TextureFormat::Undefined || (m_worldSampleCount != 1 && m_worldSampleCount != 4))
        return false;
    std::array<wgpu::BindGroupLayoutEntry, 3> bindings{};
    bindings[0].binding = 0;
    bindings[0].visibility = wgpu::ShaderStage::Fragment;
    bindings[0].texture.sampleType = wgpu::TextureSampleType::Float;
    bindings[0].texture.viewDimension = wgpu::TextureViewDimension::e2D;
    bindings[1].binding = 500;
    bindings[1].visibility = wgpu::ShaderStage::Fragment;
    bindings[1].sampler.type = wgpu::SamplerBindingType::Filtering;
    bindings[2].binding = 999;
    bindings[2].visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment;
    bindings[2].buffer.type = wgpu::BufferBindingType::Uniform;
    bindings[2].buffer.hasDynamicOffset = true;
    bindings[2].buffer.minBindingSize = sizeof(WorldConstants);
    wgpu::BindGroupLayoutDescriptor layoutDescriptor;
    layoutDescriptor.entryCount = bindings.size();
    layoutDescriptor.entries = bindings.data();
    m_worldLayout = m_device.CreateBindGroupLayout(&layoutDescriptor);

    wgpu::ShaderSourceWGSL shaderSource;
    shaderSource.code = kWorldUIShader;
    wgpu::ShaderModuleDescriptor shaderDescriptor;
    shaderDescriptor.nextInChain = &shaderSource;
    const wgpu::ShaderModule shader = m_device.CreateShaderModule(&shaderDescriptor);
    std::array<wgpu::VertexAttribute, 7> attributes{};
    const auto attribute = [&](size_t index, wgpu::VertexFormat format, uint64_t offset) {
        attributes[index].format = format;
        attributes[index].offset = offset;
        attributes[index].shaderLocation = static_cast<uint32_t>(index);
    };
    attribute(0, wgpu::VertexFormat::Float32x3, offsetof(WorldVertex, position));
    attribute(1, wgpu::VertexFormat::Float32x2, offsetof(WorldVertex, uv));
    attribute(2, wgpu::VertexFormat::Float32x4, offsetof(WorldVertex, color));
    attribute(3, wgpu::VertexFormat::Float32x2, offsetof(WorldVertex, localPosition));
    attribute(4, wgpu::VertexFormat::Float32x3, offsetof(WorldVertex, anchor));
    attribute(5, wgpu::VertexFormat::Float32x3, offsetof(WorldVertex, localOffset));
    attribute(6, wgpu::VertexFormat::Float32, offsetof(WorldVertex, policy));
    wgpu::VertexBufferLayout vertexLayout;
    vertexLayout.arrayStride = sizeof(WorldVertex);
    vertexLayout.stepMode = wgpu::VertexStepMode::Vertex;
    vertexLayout.attributeCount = attributes.size();
    vertexLayout.attributes = attributes.data();

    wgpu::BlendState blend;
    blend.color.srcFactor = wgpu::BlendFactor::SrcAlpha;
    blend.color.dstFactor = wgpu::BlendFactor::OneMinusSrcAlpha;
    blend.alpha.srcFactor = wgpu::BlendFactor::One;
    blend.alpha.dstFactor = wgpu::BlendFactor::OneMinusSrcAlpha;
    wgpu::ColorTargetState colorTarget;
    colorTarget.format = m_worldColorFormat;
    colorTarget.blend = &blend;
    colorTarget.writeMask = wgpu::ColorWriteMask::All;
    wgpu::FragmentState fragment;
    fragment.module = shader;
    fragment.entryPoint = "fragment_main";
    fragment.targetCount = 1;
    fragment.targets = &colorTarget;

    wgpu::PipelineLayoutDescriptor pipelineLayoutDescriptor;
    pipelineLayoutDescriptor.bindGroupLayoutCount = 1;
    pipelineLayoutDescriptor.bindGroupLayouts = &m_worldLayout;
    const wgpu::PipelineLayout pipelineLayout = m_device.CreatePipelineLayout(&pipelineLayoutDescriptor);
    wgpu::DepthStencilState depth;
    depth.format = wgpu::TextureFormat::Depth32Float;
    depth.depthWriteEnabled = wgpu::OptionalBool::False;
    depth.depthCompare = wgpu::CompareFunction::GreaterEqual;
    wgpu::RenderPipelineDescriptor descriptor;
    descriptor.layout = pipelineLayout;
    descriptor.vertex.module = shader;
    descriptor.vertex.entryPoint = "vertex_main";
    descriptor.vertex.bufferCount = 1;
    descriptor.vertex.buffers = &vertexLayout;
    descriptor.fragment = &fragment;
    descriptor.primitive.topology = wgpu::PrimitiveTopology::TriangleList;
    descriptor.primitive.cullMode = wgpu::CullMode::None;
    descriptor.multisample.count = m_worldSampleCount;
    descriptor.depthStencil = &depth;
    m_worldDepthPipeline = m_device.CreateRenderPipeline(&descriptor);
    depth.depthCompare = wgpu::CompareFunction::Always;
    m_worldTopPipeline = m_device.CreateRenderPipeline(&descriptor);
    return m_worldLayout && m_worldDepthPipeline && m_worldTopPipeline;
}

wgpu::RenderPipeline WebScreenUIRenderer::ResolveMaterialPipeline(const MaterialBinding &binding, bool world,
                                                                  bool alwaysOnTop)
{
    if (binding.guid.empty())
        return world ? (alwaysOnTop ? m_worldTopPipeline : m_worldDepthPipeline) : m_pipeline;
    const auto material = AssetRegistry::Instance().GetAsset<InxMaterial>(binding.guid);
    if (!material || material->IsDeleted() || material->GetVersion() != binding.generation)
        throw std::runtime_error("UI material GUID is missing or its generation changed: " + binding.guid);
    const auto &vertex = material->GetVertShaderReference();
    const auto &fragment = material->GetFragShaderReference();
    if (vertex.guid.empty() && fragment.guid.empty())
        return world ? (alwaysOnTop ? m_worldTopPipeline : m_worldDepthPipeline) : m_pipeline;
    if (vertex.guid.empty() || fragment.guid.empty())
        throw std::runtime_error("UI material has an incomplete shader GUID pair: " + binding.guid);
    const std::string key = binding.guid + ':' + std::to_string(binding.generation) + ':' +
                            std::to_string(binding.pipelineKey.size()) + ':' + binding.pipelineKey +
                            (alwaysOnTop ? ":top" : ":depth");
    auto &cache = world ? m_customWorldPipelines : m_customScreenPipelines;
    if (const auto cached = cache.find(key); cached != cache.end())
        return cached->second;
    std::string vertexSource;
    std::string fragmentSource;
    if (!InfernuxWebFindShaderSource(vertex.shaderId, "vertex", vertexSource) ||
        !InfernuxWebFindShaderSource(fragment.shaderId, "fragment", fragmentSource))
        throw std::runtime_error("UI material shader is absent from the Web catalog: " + binding.guid);
    wgpu::ShaderSourceWGSL vertexWGSL;
    vertexWGSL.code = vertexSource.c_str();
    wgpu::ShaderModuleDescriptor vertexDescriptor;
    vertexDescriptor.nextInChain = &vertexWGSL;
    const wgpu::ShaderModule vertexModule = m_device.CreateShaderModule(&vertexDescriptor);
    wgpu::ShaderSourceWGSL fragmentWGSL;
    fragmentWGSL.code = fragmentSource.c_str();
    wgpu::ShaderModuleDescriptor fragmentDescriptor;
    fragmentDescriptor.nextInChain = &fragmentWGSL;
    const wgpu::ShaderModule fragmentModule = m_device.CreateShaderModule(&fragmentDescriptor);
    if (!vertexModule || !fragmentModule)
        throw std::runtime_error("WebGPU rejected UI material shader modules: " + binding.guid);
    std::array<wgpu::VertexAttribute, 7> attributes{};
    const auto attribute = [&](size_t index, wgpu::VertexFormat format, uint64_t offset) {
        attributes[index].format = format;
        attributes[index].offset = offset;
        attributes[index].shaderLocation = static_cast<uint32_t>(index);
    };
    if (world) {
        attribute(0, wgpu::VertexFormat::Float32x3, offsetof(WorldVertex, position));
        attribute(1, wgpu::VertexFormat::Float32x2, offsetof(WorldVertex, uv));
        attribute(2, wgpu::VertexFormat::Float32x4, offsetof(WorldVertex, color));
        attribute(3, wgpu::VertexFormat::Float32x2, offsetof(WorldVertex, localPosition));
        attribute(4, wgpu::VertexFormat::Float32x3, offsetof(WorldVertex, anchor));
        attribute(5, wgpu::VertexFormat::Float32x3, offsetof(WorldVertex, localOffset));
        attribute(6, wgpu::VertexFormat::Float32, offsetof(WorldVertex, policy));
    } else {
        attribute(0, wgpu::VertexFormat::Float32x2, offsetof(GPUVertex, position));
        attribute(1, wgpu::VertexFormat::Float32x2, offsetof(GPUVertex, uv));
        attribute(2, wgpu::VertexFormat::Float32x4, offsetof(GPUVertex, color));
    }
    wgpu::VertexBufferLayout vertexLayout;
    vertexLayout.arrayStride = world ? sizeof(WorldVertex) : sizeof(GPUVertex);
    vertexLayout.stepMode = wgpu::VertexStepMode::Vertex;
    vertexLayout.attributeCount = world ? 7 : 3;
    vertexLayout.attributes = attributes.data();
    wgpu::BlendState blend;
    blend.color.srcFactor = wgpu::BlendFactor::SrcAlpha;
    blend.color.dstFactor = wgpu::BlendFactor::OneMinusSrcAlpha;
    blend.alpha.srcFactor = wgpu::BlendFactor::One;
    blend.alpha.dstFactor = wgpu::BlendFactor::OneMinusSrcAlpha;
    wgpu::ColorTargetState colorTarget;
    colorTarget.format = world ? m_worldColorFormat : m_colorFormat;
    colorTarget.blend = &blend;
    colorTarget.writeMask = wgpu::ColorWriteMask::All;
    wgpu::FragmentState fragmentState;
    fragmentState.module = fragmentModule;
    fragmentState.entryPoint = "main";
    fragmentState.targetCount = 1;
    fragmentState.targets = &colorTarget;
    const wgpu::BindGroupLayout &layout = world ? m_worldLayout : m_screenLayout;
    wgpu::PipelineLayoutDescriptor layoutDescriptor;
    layoutDescriptor.bindGroupLayoutCount = 1;
    layoutDescriptor.bindGroupLayouts = &layout;
    const wgpu::PipelineLayout pipelineLayout = m_device.CreatePipelineLayout(&layoutDescriptor);
    wgpu::DepthStencilState depth;
    depth.format = wgpu::TextureFormat::Depth32Float;
    depth.depthWriteEnabled = wgpu::OptionalBool::False;
    depth.depthCompare = alwaysOnTop ? wgpu::CompareFunction::Always : wgpu::CompareFunction::GreaterEqual;
    wgpu::RenderPipelineDescriptor descriptor;
    descriptor.layout = pipelineLayout;
    descriptor.vertex.module = vertexModule;
    descriptor.vertex.entryPoint = "main";
    descriptor.vertex.bufferCount = 1;
    descriptor.vertex.buffers = &vertexLayout;
    descriptor.fragment = &fragmentState;
    descriptor.primitive.topology = wgpu::PrimitiveTopology::TriangleList;
    descriptor.primitive.cullMode = wgpu::CullMode::None;
    descriptor.multisample.count = world ? m_worldSampleCount : 1;
    descriptor.depthStencil = world ? &depth : nullptr;
    const wgpu::RenderPipeline pipeline = m_device.CreateRenderPipeline(&descriptor);
    if (!pipeline)
        throw std::runtime_error("WebGPU rejected UI material pipeline: " + binding.guid);
    return cache.emplace(key, pipeline).first->second;
}

std::pair<wgpu::Sampler, std::string> WebScreenUIRenderer::ResolveMaterialSampler(const MaterialBinding &binding,
                                                                                  uint64_t textureId)
{
    const auto texture = m_textures.find(textureId);
    if (textureId != kFontTextureId && texture == m_textures.end())
        return {};
    const wgpu::Sampler inherited = textureId == kFontTextureId ? m_fontSampler : texture->second.sampler;
    std::string key = std::to_string(textureId);
    if (binding.guid.empty())
        return {inherited, key};
    const auto material = AssetRegistry::Instance().GetAsset<InxMaterial>(binding.guid);
    if (!material || material->IsDeleted() || material->GetVersion() != binding.generation)
        throw std::runtime_error("UI material GUID is missing or its generation changed: " + binding.guid);
    const MaterialTextureSampler *override = material->GetTextureSampler("texSampler");
    if (!override || !override->HasOverrides())
        return {inherited, key};
    key.push_back(':');
    for (const auto value : {static_cast<uint8_t>(override->minFilter), static_cast<uint8_t>(override->magFilter),
                             static_cast<uint8_t>(override->mipFilter), static_cast<uint8_t>(override->addressU),
                             static_cast<uint8_t>(override->addressV), static_cast<uint8_t>(override->addressW)})
        key.push_back(static_cast<char>('0' + value));
    if (const auto cached = m_materialSamplers.find(key); cached != m_materialSamplers.end())
        return {cached->second, key};
    const std::string filterMode = textureId == kFontTextureId ? "bilinear" : texture->second.filterMode;
    const std::string wrapMode = textureId == kFontTextureId ? "clamp" : texture->second.wrapMode;
    const int anisoLevel = textureId == kFontTextureId ? 1 : texture->second.anisoLevel;
    auto descriptor = UISamplerDescriptor(filterMode, wrapMode, anisoLevel, *override);
    descriptor.lodMaxClamp = static_cast<float>(textureId == kFontTextureId ? 0 : texture->second.mipLevels - 1);
    const auto sampler = m_device.CreateSampler(&descriptor);
    if (!sampler)
        throw std::runtime_error("WebGPU rejected UI material texture sampler: " + binding.guid);
    return {m_materialSamplers.emplace(key, sampler).first->second, key};
}

ImDrawList *WebScreenUIRenderer::DrawList(int list) const
{
    switch (list) {
    case 0:
        return m_camera;
    case 1:
        return m_overlay;
    case 2:
        return m_world;
    default:
        return nullptr;
    }
}

void WebScreenUIRenderer::BeginFrame(uint32_t width, uint32_t height)
{
    if (!m_context || !m_camera || !m_overlay || !m_world)
        return;
    ImGui::SetCurrentContext(m_context);
    if (!ImGui::GetIO().Fonts->IsBuilt()) {
        m_fontAtlasDirty = true;
        if (!RefreshFontAtlas())
            throw std::runtime_error("Web UI font atlas publication failed before geometry recording");
    }
    m_cacheValid = false;
    ResetDrawList(*m_camera, width, height);
    ResetDrawList(*m_overlay, width, height);
    ResetDrawList(*m_world, width, height);
    m_world->PopClipRect();
    m_world->PushClipRect({-1.0e9f, -1.0e9f}, {1.0e9f, 1.0e9f});
    for (auto &boundaries : m_bindingBoundaries)
        boundaries.clear();
    for (auto &binding : m_currentBindings)
        binding = {};
    m_worldElements.clear();
    m_pendingWorldElement = {};
    m_worldElementOpen = false;
}

bool WebScreenUIRenderer::BeginFrameCached(uint32_t width, uint32_t height, uint64_t contentRevision)
{
    if (m_cacheValid && width == m_cachedWidth && height == m_cachedHeight && contentRevision == m_cachedRevision &&
        ImGui::GetIO().Fonts->IsBuilt())
        return true;
    BeginFrame(width, height);
    m_cacheValid = true;
    m_cachedWidth = width;
    m_cachedHeight = height;
    m_cachedRevision = contentRevision;
    return false;
}

void WebScreenUIRenderer::PushClipRect(int list, float minX, float minY, float maxX, float maxY)
{
    if (ImDrawList *drawList = DrawList(list))
        drawList->PushClipRect(ImVec2(minX, minY), ImVec2(maxX, maxY), true);
}

void WebScreenUIRenderer::PopClipRect(int list)
{
    if (ImDrawList *drawList = DrawList(list))
        drawList->PopClipRect();
}

void WebScreenUIRenderer::SetMaterialBinding(int list, const std::string &guid, uint64_t generation,
                                             const std::string &pipelineKey, const std::array<float, 4> &baseColor,
                                             bool alphaClipEnabled, float alphaClipThreshold)
{
    ImDrawList *draw = DrawList(list);
    if (!draw)
        throw std::invalid_argument("UI material binding has an invalid draw list");
    const bool defaultBinding = guid.empty() && generation == 0 && pipelineKey.empty();
    if (!defaultBinding && (guid.empty() || generation == 0 || pipelineKey.empty()))
        throw std::invalid_argument("UI material binding requires GUID, generation and diagnostic key");
    auto &current = m_currentBindings[static_cast<size_t>(list)];
    MaterialBinding next{guid, generation, pipelineKey, baseColor, alphaClipEnabled, alphaClipThreshold};
    if (current.guid == next.guid && current.generation == next.generation && current.pipelineKey == next.pipelineKey &&
        current.baseColor == next.baseColor && current.alphaClipEnabled == next.alphaClipEnabled &&
        current.alphaClipThreshold == next.alphaClipThreshold)
        return;
    // ImDrawList otherwise merges consecutive commands with the same texture
    // and scissor, which would mix two material generations into one draw.
    draw->AddCallback(CommandBoundary, nullptr);
    // AddCallback has already appended the empty draw command that receives
    // the next geometry. Bind there, not at the command after that geometry.
    m_bindingBoundaries[static_cast<size_t>(list)].push_back({draw->CmdBuffer.Size - 1, next});
    current = std::move(next);
}

void WebScreenUIRenderer::BeginWorldElement(const std::array<float, 16> &localToWorld, float pivotX, float pivotY,
                                            uint32_t layer, bool alwaysOnTop, bool billboard, bool constantScreenSize,
                                            uint64_t ignoredOccluderId)
{
    if (m_worldElementOpen || !m_world)
        throw std::logic_error("World UI element begin requires an idle renderer");
    if (layer >= 32)
        throw std::invalid_argument("World UI layer is outside the camera mask");
    if (ignoredOccluderId != 0)
        throw std::runtime_error("Web world UI does not yet support ignored-occluder depth exclusion");
    m_world->AddCallback(CommandBoundary, nullptr);
    m_pendingWorldElement = {};
    m_pendingWorldElement.vertexStart = m_world->VtxBuffer.Size;
    // AddCallback always appends a fresh draw command after the callback.
    // That trailing command is where the element's geometry is recorded.
    // Starting at CmdBuffer.Size skipped it and made every populated world
    // element appear to have zero drawable commands.
    m_pendingWorldElement.commandStart = std::max(0, m_world->CmdBuffer.Size - 1);
    m_pendingWorldElement.localToWorld = glm::make_mat4(localToWorld.data());
    m_pendingWorldElement.pivotX = pivotX;
    m_pendingWorldElement.pivotY = pivotY;
    m_pendingWorldElement.layer = layer;
    m_pendingWorldElement.alwaysOnTop = alwaysOnTop;
    m_pendingWorldElement.billboard = billboard;
    m_pendingWorldElement.constantScreenSize = constantScreenSize;
    m_pendingWorldElement.ignoredOccluderId = ignoredOccluderId;
    m_worldElementOpen = true;
}

void WebScreenUIRenderer::EndWorldElement()
{
    if (!m_worldElementOpen || !m_world)
        throw std::logic_error("World UI element end has no matching begin");
    m_pendingWorldElement.vertexEnd = m_world->VtxBuffer.Size;
    m_pendingWorldElement.vertexDepths.resize(m_pendingWorldElement.vertexEnd - m_pendingWorldElement.vertexStart,
                                              0.0f);
    m_pendingWorldElement.commandEnd = m_world->CmdBuffer.Size;
    if (m_pendingWorldElement.vertexEnd > m_pendingWorldElement.vertexStart)
        m_worldElements.push_back(std::move(m_pendingWorldElement));
    m_world->AddCallback(CommandBoundary, nullptr);
    m_pendingWorldElement = {};
    m_worldElementOpen = false;
}

void WebScreenUIRenderer::AddFilledRect(int list, float minX, float minY, float maxX, float maxY, float r, float g,
                                        float b, float a, float rounding, float rotation, bool mirrorH, bool mirrorV)
{
    ImDrawList *draw = DrawList(list);
    if (!draw)
        return;
    const int firstVertex = draw->VtxBuffer.Size;
    draw->AddRectFilled({minX, minY}, {maxX, maxY}, ImGui::ColorConvertFloat4ToU32({r, g, b, a}), rounding);
    TransformVertices(*draw, firstVertex, minX, minY, maxX, maxY, rotation, mirrorH, mirrorV);
}

void WebScreenUIRenderer::AddImage(int list, uint64_t textureId, float minX, float minY, float maxX, float maxY,
                                   float uv0X, float uv0Y, float uv1X, float uv1Y, float r, float g, float b, float a,
                                   float rotation, bool mirrorH, bool mirrorV, float rounding)
{
    ImDrawList *draw = DrawList(list);
    if (!draw || m_textures.find(textureId) == m_textures.end())
        return;
    const int firstVertex = draw->VtxBuffer.Size;
    const ImU32 tint = ImGui::ColorConvertFloat4ToU32({r, g, b, a});
    if (rounding > 0.5f)
        draw->AddImageRounded(static_cast<ImTextureID>(textureId), {minX, minY}, {maxX, maxY}, {uv0X, uv0Y},
                              {uv1X, uv1Y}, tint, rounding);
    else
        draw->AddImage(static_cast<ImTextureID>(textureId), {minX, minY}, {maxX, maxY}, {uv0X, uv0Y}, {uv1X, uv1Y},
                       tint);
    TransformVertices(*draw, firstVertex, minX, minY, maxX, maxY, rotation, mirrorH, mirrorV);
}

void WebScreenUIRenderer::AddText(int list, float minX, float minY, float maxX, float maxY, const std::string &text,
                                  float r, float g, float b, float a, float alignX, float alignY, float fontSize,
                                  float wrapWidth, float rotation, bool mirrorH, bool mirrorV,
                                  const std::string &fontPath, float lineHeight, float letterSpacing, bool clip,
                                  const std::vector<std::string> &fallbackFontPaths)
{
    ImDrawList *draw = DrawList(list);
    if (!draw || text.empty())
        return;
    ImGui::SetCurrentContext(m_context);
    const textlayout::TextLayoutResult layout =
        textlayout::LayoutText({text, fontPath, textlayout::ResolveFontSize(fontSize), wrapWidth, lineHeight,
                                letterSpacing, fallbackFontPaths});
    // With the legacy atlas upload path used by this renderer, a font added by
    // LayoutText must be baked before we read glyph UVs into the draw list.
    // Waiting until Render() rebuilds the atlas leaves the already-recorded
    // vertices carrying UVs from the previous atlas layout, which produces
    // scrambled text as soon as the engine PingFang face is loaded.
    if (m_context && !ImGui::GetIO().Fonts->IsBuilt()) {
        m_fontAtlasDirty = true;
        if (!RefreshFontAtlas())
            return;
    }
    const int firstVertex = draw->VtxBuffer.Size;
    if (clip)
        draw->PushClipRect({minX, minY}, {maxX, maxY}, true);
    draw->PushTextureID(ImGui::GetIO().Fonts->TexRef);
    textlayout::RenderTextBox(draw, minX, minY, maxX, maxY, layout, ImGui::ColorConvertFloat4ToU32({r, g, b, a}),
                              alignX, alignY, letterSpacing);
    draw->PopTextureID();
    if (clip)
        draw->PopClipRect();
    TransformVertices(*draw, firstVertex, minX, minY, maxX, maxY, rotation, mirrorH, mirrorV);
}

void WebScreenUIRenderer::AddTextGeometry(int list, float minX, float minY, float maxX, float maxY,
                                          textlayout::TextGeometry &geometry, float r, float g, float b, float a,
                                          float alignX, float alignY, float scale, float rotation, bool mirrorH,
                                          bool mirrorV, bool clip, float opacity)
{
    ImDrawList *draw = DrawList(list);
    if (!draw)
        return;
    ImGui::SetCurrentContext(m_context);
    const bool world = list == 2;
    if (world && !m_worldElementOpen)
        throw std::logic_error("World text requires an active world element");
    geometry.Prepare(scale);
    if (!ImGui::GetIO().Fonts->IsBuilt()) {
        m_fontAtlasDirty = true;
        if (!RefreshFontAtlas())
            throw std::runtime_error("Web UI font atlas publication failed during text preparation");
    }
    draw->PushTextureID(ImGui::GetIO().Fonts->TexRef);
    geometry.Render(*draw, minX, minY, maxX, maxY, {r, g, b, a}, alignX, alignY, scale, rotation, mirrorH, mirrorV,
                    false, clip, world ? &m_pendingWorldElement.vertexDepths : nullptr,
                    world ? m_pendingWorldElement.vertexStart : 0, opacity);
    draw->PopTextureID();
}

std::pair<float, float> WebScreenUIRenderer::MeasureText(const std::string &text, float fontSize, float wrapWidth,
                                                         const std::string &fontPath, float lineHeight,
                                                         float letterSpacing,
                                                         const std::vector<std::string> &fallbackFontPaths)
{
    ImGui::SetCurrentContext(m_context);
    const textlayout::TextLayoutResult layout =
        textlayout::LayoutText({text, fontPath, textlayout::ResolveFontSize(fontSize), wrapWidth, lineHeight,
                                letterSpacing, fallbackFontPaths});
    // RuntimeScreenUISubmission measures every text element before it records
    // any geometry.  Honour that preparation boundary by publishing the final
    // atlas immediately.  Otherwise a later AddText rebuild changes the
    // atlas white-pixel UV after filled rectangles have already copied the old
    // coordinate, making those rectangles permanently transparent in a cached
    // first-frame command list.
    if (m_context && !ImGui::GetIO().Fonts->IsBuilt()) {
        m_fontAtlasDirty = true;
        if (!RefreshFontAtlas())
            throw std::runtime_error("Web UI font atlas publication failed during text preparation");
    }
    return {layout.totalWidth, layout.totalHeight};
}

uint64_t WebScreenUIRenderer::UploadTexture(const TextureCpuData &source, uint64_t replaceTextureId,
                                            const std::string &filterMode, const std::string &wrapMode, int anisoLevel)
{
    if (!m_device || !m_queue || source.dimension != TextureDimension::Texture2D || !source.IsValid())
        return 0;
    std::optional<TextureCpuData> decoded;
    if (TextureFormatIsBlockCompressed(source.format)) {
        decoded = DecodeBcTextureToRgba8(source);
        if (!decoded)
            return 0;
    }
    const TextureCpuData &texture = decoded ? *decoded : source;
    const wgpu::TextureFormat format = ToWebTextureFormat(texture.format);
    if (format == wgpu::TextureFormat::Undefined)
        return 0;
    if (texture.mipLevels.size() > std::numeric_limits<uint32_t>::max())
        return 0;
    const TextureMipLevel &baseMip = texture.mipLevels.front();
    for (const TextureMipLevel &mip : texture.mipLevels)
        if (mip.width == 0 || mip.height == 0 || mip.depth != 1 || mip.byteOffset > texture.bytes.size() ||
            mip.byteSize > texture.bytes.size() - mip.byteOffset || mip.rowPitch > std::numeric_limits<uint32_t>::max())
            return 0;

    wgpu::TextureDescriptor descriptor;
    descriptor.dimension = wgpu::TextureDimension::e2D;
    descriptor.size = {baseMip.width, baseMip.height, 1};
    descriptor.format = format;
    descriptor.mipLevelCount = static_cast<uint32_t>(texture.mipLevels.size());
    descriptor.sampleCount = 1;
    descriptor.usage = wgpu::TextureUsage::CopyDst | wgpu::TextureUsage::TextureBinding;
    GPUTexture gpu;
    gpu.filterMode = filterMode;
    gpu.wrapMode = wrapMode;
    gpu.anisoLevel = anisoLevel;
    gpu.mipLevels = descriptor.mipLevelCount;
    gpu.texture = m_device.CreateTexture(&descriptor);
    gpu.view = gpu.texture ? gpu.texture.CreateView() : wgpu::TextureView{};

    auto samplerDescriptor = UISamplerDescriptor(filterMode, wrapMode, anisoLevel, {});
    samplerDescriptor.lodMaxClamp = static_cast<float>(descriptor.mipLevelCount - 1);
    gpu.sampler = m_device.CreateSampler(&samplerDescriptor);
    if (!gpu.texture || !gpu.view || !gpu.sampler)
        return 0;

    for (uint32_t level = 0; level < descriptor.mipLevelCount; ++level) {
        const TextureMipLevel &mip = texture.mipLevels[level];
        wgpu::TexelCopyTextureInfo destination;
        destination.texture = gpu.texture;
        destination.mipLevel = level;
        wgpu::TexelCopyBufferLayout layout;
        layout.bytesPerRow = static_cast<uint32_t>(mip.rowPitch);
        layout.rowsPerImage = mip.height;
        const wgpu::Extent3D extent{mip.width, mip.height, 1};
        m_queue.WriteTexture(&destination, texture.bytes.data() + mip.byteOffset, static_cast<size_t>(mip.byteSize),
                             &layout, &extent);
    }

    std::array<wgpu::BindGroupEntry, 2> entries{};
    entries[0].binding = 0;
    entries[0].sampler = gpu.sampler;
    entries[1].binding = 1;
    entries[1].textureView = gpu.view;
    wgpu::BindGroupDescriptor groupDescriptor;
    groupDescriptor.layout = m_textureLayout;
    groupDescriptor.entryCount = entries.size();
    groupDescriptor.entries = entries.data();
    gpu.group = m_device.CreateBindGroup(&groupDescriptor);
    if (!gpu.group)
        return 0;

    const uint64_t id = replaceTextureId > kFontTextureId ? replaceTextureId : m_nextTextureId++;
    if (replaceTextureId > kFontTextureId)
        m_materialSamplers.clear();
    m_textures.insert_or_assign(id, std::move(gpu));
    m_cacheValid = false;
    return id;
}

void WebScreenUIRenderer::ReleaseTexture(uint64_t textureId)
{
    if (textureId <= kFontTextureId)
        return;
    if (m_textures.erase(textureId) != 0)
        m_cacheValid = false;
    m_materialSamplers.clear();
}

bool WebScreenUIRenderer::EnsureBuffer(wgpu::Buffer &buffer, uint64_t &capacity, uint64_t required,
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

bool WebScreenUIRenderer::Render(wgpu::RenderPassEncoder pass, int list, uint32_t width, uint32_t height)
{
    ImDrawList *draw = DrawList(list);
    if (!pass || !m_pipeline || !draw || draw->VtxBuffer.empty() || draw->IdxBuffer.empty() || width == 0 ||
        height == 0)
        return false;
    if (!RefreshFontAtlas())
        return false;
    // Queue uploads require four-byte alignment even for 16-bit triangle indices.
    if ((draw->IdxBuffer.Size * sizeof(ImDrawIdx)) % 4 != 0)
        draw->IdxBuffer.push_back(0);
    std::vector<GPUVertex> vertices(static_cast<size_t>(draw->VtxBuffer.Size));
    for (int index = 0; index < draw->VtxBuffer.Size; ++index) {
        const ImDrawVert &source = draw->VtxBuffer[index];
        GPUVertex &target = vertices[static_cast<size_t>(index)];
        target.position[0] = source.pos.x;
        target.position[1] = source.pos.y;
        target.uv[0] = source.uv.x;
        target.uv[1] = source.uv.y;
        const ImVec4 color = ImGui::ColorConvertU32ToFloat4(source.col);
        target.color[0] = color.x;
        target.color[1] = color.y;
        target.color[2] = color.z;
        target.color[3] = color.w;
    }
    const uint64_t vertexBytes = vertices.size() * sizeof(GPUVertex);
    const uint64_t indexBytes = static_cast<uint64_t>(draw->IdxBuffer.Size) * sizeof(ImDrawIdx);
    constexpr uint64_t constantStride = 256;
    static_assert(sizeof(ScreenConstants) <= constantStride);
    if (!EnsureBuffer(m_vertexBuffer, m_vertexCapacity, vertexBytes, wgpu::BufferUsage::Vertex) ||
        !EnsureBuffer(m_indexBuffer, m_indexCapacity, indexBytes, wgpu::BufferUsage::Index) ||
        !EnsureBuffer(m_screenConstantsBuffer, m_screenConstantsCapacity,
                      constantStride * static_cast<uint64_t>(draw->CmdBuffer.Size), wgpu::BufferUsage::Uniform))
        return false;
    m_queue.WriteBuffer(m_vertexBuffer, 0, vertices.data(), vertexBytes);
    m_queue.WriteBuffer(m_indexBuffer, 0, draw->IdxBuffer.Data, indexBytes);
    pass.SetVertexBuffer(0, m_vertexBuffer, 0, vertexBytes);
    pass.SetIndexBuffer(m_indexBuffer, sizeof(ImDrawIdx) == 2 ? wgpu::IndexFormat::Uint16 : wgpu::IndexFormat::Uint32,
                        0, indexBytes);
    std::unordered_map<std::string, wgpu::BindGroup> textureGroups;
    const auto &boundaries = m_bindingBoundaries[static_cast<size_t>(list)];
    size_t boundaryIndex = 0;
    MaterialBinding binding;
    for (int commandIndex = 0; commandIndex < draw->CmdBuffer.Size; ++commandIndex) {
        while (boundaryIndex < boundaries.size() && boundaries[boundaryIndex].commandStart <= commandIndex)
            binding = boundaries[boundaryIndex++].binding;
        const ImDrawCmd &command = draw->CmdBuffer[commandIndex];
        if (command.ElemCount == 0 || command.UserCallback)
            continue;
        const int left = std::max(0, static_cast<int>(std::floor(command.ClipRect.x)));
        const int top = std::max(0, static_cast<int>(std::floor(command.ClipRect.y)));
        const int right = std::min(static_cast<int>(width), static_cast<int>(std::ceil(command.ClipRect.z)));
        const int bottom = std::min(static_cast<int>(height), static_cast<int>(std::ceil(command.ClipRect.w)));
        if (right <= left || bottom <= top)
            continue;
        const uint64_t textureId = static_cast<uint64_t>(command.GetTexID());
        wgpu::TextureView textureView = m_fontView;
        if (textureId != kFontTextureId) {
            const auto texture = m_textures.find(textureId);
            if (texture == m_textures.end())
                continue;
            textureView = texture->second.view;
        }
        const auto [sampler, samplerKey] = ResolveMaterialSampler(binding, textureId);
        auto group = textureGroups.find(samplerKey);
        if (group == textureGroups.end()) {
            std::array<wgpu::BindGroupEntry, 3> entries{};
            entries[0].binding = 0;
            entries[0].textureView = textureView;
            entries[1].binding = 500;
            entries[1].sampler = sampler;
            entries[2].binding = 999;
            entries[2].buffer = m_screenConstantsBuffer;
            entries[2].offset = 0;
            entries[2].size = sizeof(ScreenConstants);
            wgpu::BindGroupDescriptor descriptor;
            descriptor.layout = m_screenLayout;
            descriptor.entryCount = entries.size();
            descriptor.entries = entries.data();
            group = textureGroups.emplace(samplerKey, m_device.CreateBindGroup(&descriptor)).first;
        }
        if (!group->second)
            return false;
        ScreenConstants constants;
        constants.scale[0] = 2.0f / static_cast<float>(width);
        constants.scale[1] = -2.0f / static_cast<float>(height);
        constants.translate[0] = -1.0f;
        constants.translate[1] = 1.0f;
        constants.materialColor = glm::make_vec4(binding.baseColor.data());
        constants.alphaClipEnabled = binding.alphaClipEnabled ? 1.0f : 0.0f;
        constants.alphaClipThreshold = binding.alphaClipThreshold;
        const uint64_t offset = static_cast<uint64_t>(commandIndex) * constantStride;
        m_queue.WriteBuffer(m_screenConstantsBuffer, offset, &constants, sizeof(constants));
        pass.SetPipeline(ResolveMaterialPipeline(binding, false, false));
        const uint32_t dynamicOffset = static_cast<uint32_t>(offset);
        pass.SetBindGroup(0, group->second, 1, &dynamicOffset);
        pass.SetScissorRect(static_cast<uint32_t>(left), static_cast<uint32_t>(top),
                            static_cast<uint32_t>(right - left), static_cast<uint32_t>(bottom - top));
        pass.DrawIndexed(command.ElemCount, 1, command.IdxOffset, static_cast<int32_t>(command.VtxOffset), 0);
    }
    if (!m_reportedFirstDraw) {
        std::printf("INFERNUX_WEB_SCREEN_UI_READY vertices=%d indices=%d\n", draw->VtxBuffer.Size,
                    draw->IdxBuffer.Size);
        m_reportedFirstDraw = true;
    }
    return true;
}

bool WebScreenUIRenderer::RenderWorld(wgpu::RenderPassEncoder pass, const glm::mat4 &viewProjection,
                                      const glm::mat4 &view, const glm::mat4 &projection, uint32_t cullingMask,
                                      uint32_t width, uint32_t height)
{
    if (!pass || !m_world || m_worldElements.empty() || !m_worldDepthPipeline || !m_worldTopPipeline || width == 0 ||
        height == 0 || m_worldElementOpen) {
        if (!m_reportedWorldCommandState && m_world && (m_world->VtxBuffer.Size > 0 || !m_worldElements.empty())) {
            std::printf("INFERNUX_WEB_WORLD_UI_COMMAND_STATE spans=%zu vertices=%d indices=%d open=%d depth=%d top=%d "
                        "size=%ux%u\n",
                        m_worldElements.size(), m_world->VtxBuffer.Size, m_world->IdxBuffer.Size,
                        m_worldElementOpen ? 1 : 0, m_worldDepthPipeline ? 1 : 0, m_worldTopPipeline ? 1 : 0, width,
                        height);
            m_reportedWorldCommandState = true;
        }
        return false;
    }
    if (!RefreshFontAtlas())
        return false;

    std::vector<WorldVertex> vertices(static_cast<size_t>(m_world->VtxBuffer.Size));
    const glm::mat4 cameraToWorld = glm::inverse(view);
    const glm::vec3 cameraRight = glm::normalize(glm::vec3(cameraToWorld[0]));
    const glm::vec3 cameraUp = glm::normalize(glm::vec3(cameraToWorld[1]));
    struct Draw
    {
        const ImDrawCmd *command = nullptr;
        MaterialBinding binding;
        bool top = false;
        float distance = 0.0f;
    };
    std::vector<Draw> draws;
    const auto &boundaries = m_bindingBoundaries[2];
    size_t boundaryIndex = 0;
    MaterialBinding binding;
    for (const WorldElementSpan &span : m_worldElements) {
        if ((cullingMask & (1u << span.layer)) == 0)
            continue;
        const glm::vec3 anchor = glm::vec3(span.localToWorld[3]);
        for (int index = span.vertexStart; index < span.vertexEnd; ++index) {
            const ImDrawVert &source = m_world->VtxBuffer[index];
            WorldVertex &target = vertices[static_cast<size_t>(index)];
            const float localX = (source.pos.x - span.pivotX) * 0.01f;
            const float localY = (span.pivotY - source.pos.y) * 0.01f;
            const float localZ = span.vertexDepths[static_cast<size_t>(index - span.vertexStart)] * 0.01f;
            const glm::vec3 position = glm::vec3(span.localToWorld * glm::vec4(localX, localY, localZ, 1.0f));
            const ImVec4 color = ImGui::ColorConvertU32ToFloat4(source.col);
            target.position[0] = position.x;
            target.position[1] = position.y;
            target.position[2] = position.z;
            target.uv[0] = source.uv.x;
            target.uv[1] = source.uv.y;
            target.color[0] = color.x;
            target.color[1] = color.y;
            target.color[2] = color.z;
            target.color[3] = color.w;
            target.localPosition[0] = source.pos.x;
            target.localPosition[1] = source.pos.y;
            target.anchor[0] = anchor.x;
            target.anchor[1] = anchor.y;
            target.anchor[2] = anchor.z;
            target.localOffset[0] = localX;
            target.localOffset[1] = localY;
            target.localOffset[2] = localZ;
            target.policy = static_cast<float>((span.billboard ? 1 : 0) | (span.constantScreenSize ? 2 : 0));
        }
        const glm::vec4 viewAnchor = view * glm::vec4(anchor, 1.0f);
        for (int index = span.commandStart; index < span.commandEnd; ++index) {
            while (boundaryIndex < boundaries.size() && boundaries[boundaryIndex].commandStart <= index)
                binding = boundaries[boundaryIndex++].binding;
            const ImDrawCmd &command = m_world->CmdBuffer[index];
            if (command.ElemCount > 0 && !command.UserCallback)
                draws.push_back({&command, binding, span.alwaysOnTop, -viewAnchor.z});
        }
    }
    if (draws.empty()) {
        if (!m_reportedWorldCommandState) {
            std::printf("INFERNUX_WEB_WORLD_UI_COMMAND_STATE spans=%zu vertices=%d indices=%d draws=0 mask=%u\n",
                        m_worldElements.size(), m_world->VtxBuffer.Size, m_world->IdxBuffer.Size, cullingMask);
            m_reportedWorldCommandState = true;
        }
        return false;
    }
    std::stable_sort(draws.begin(), draws.end(), [](const Draw &a, const Draw &b) {
        if (a.top != b.top)
            return !a.top;
        return a.distance > b.distance;
    });
    const uint64_t vertexBytes = vertices.size() * sizeof(WorldVertex);
    if ((m_world->IdxBuffer.Size * sizeof(ImDrawIdx)) % 4 != 0)
        m_world->IdxBuffer.push_back(0);
    const uint64_t indexBytes = static_cast<uint64_t>(m_world->IdxBuffer.Size) * sizeof(ImDrawIdx);
    constexpr uint64_t constantStride = 256;
    static_assert(sizeof(WorldConstants) <= constantStride);
    if (!EnsureBuffer(m_worldVertexBuffer, m_worldVertexCapacity, vertexBytes, wgpu::BufferUsage::Vertex) ||
        !EnsureBuffer(m_worldIndexBuffer, m_worldIndexCapacity, indexBytes, wgpu::BufferUsage::Index) ||
        !EnsureBuffer(m_worldConstantsBuffer, m_worldConstantsCapacity, constantStride * draws.size(),
                      wgpu::BufferUsage::Uniform))
        return false;
    m_queue.WriteBuffer(m_worldVertexBuffer, 0, vertices.data(), vertexBytes);
    m_queue.WriteBuffer(m_worldIndexBuffer, 0, m_world->IdxBuffer.Data, indexBytes);
    pass.SetVertexBuffer(0, m_worldVertexBuffer, 0, vertexBytes);
    pass.SetIndexBuffer(m_worldIndexBuffer,
                        sizeof(ImDrawIdx) == 2 ? wgpu::IndexFormat::Uint16 : wgpu::IndexFormat::Uint32, 0, indexBytes);
    pass.SetScissorRect(0, 0, width, height);
    std::unordered_map<std::string, wgpu::BindGroup> textureGroups;
    for (size_t index = 0; index < draws.size(); ++index) {
        const Draw &draw = draws[index];
        const uint64_t textureId = static_cast<uint64_t>(draw.command->GetTexID());
        wgpu::TextureView textureView = m_fontView;
        if (textureId != kFontTextureId) {
            const auto found = m_textures.find(textureId);
            if (found == m_textures.end())
                continue;
            textureView = found->second.view;
        }
        const auto [sampler, samplerKey] = ResolveMaterialSampler(draw.binding, textureId);
        auto group = textureGroups.find(samplerKey);
        if (group == textureGroups.end()) {
            std::array<wgpu::BindGroupEntry, 3> entries{};
            entries[0].binding = 0;
            entries[0].textureView = textureView;
            entries[1].binding = 500;
            entries[1].sampler = sampler;
            entries[2].binding = 999;
            entries[2].buffer = m_worldConstantsBuffer;
            entries[2].offset = 0;
            entries[2].size = sizeof(WorldConstants);
            wgpu::BindGroupDescriptor descriptor;
            descriptor.layout = m_worldLayout;
            descriptor.entryCount = entries.size();
            descriptor.entries = entries.data();
            group = textureGroups.emplace(samplerKey, m_device.CreateBindGroup(&descriptor)).first;
        }
        if (!group->second)
            return false;
        WorldConstants constants;
        constants.viewProjection = viewProjection;
        constants.materialColor = glm::make_vec4(draw.binding.baseColor.data());
        constants.alphaClipEnabled = draw.binding.alphaClipEnabled ? 1.0f : 0.0f;
        constants.alphaClipThreshold = draw.binding.alphaClipThreshold;
        constants.screenScale[0] = 2.0f / (std::max(std::abs(projection[1][1]), 1.0e-6f) * static_cast<float>(height));
        constants.screenScale[1] = 0.0f;
        constants.cameraRight = glm::vec4(cameraRight, 0.0f);
        constants.cameraUp = glm::vec4(cameraUp, 0.0f);
        const uint64_t offset = index * constantStride;
        m_queue.WriteBuffer(m_worldConstantsBuffer, offset, &constants, sizeof(constants));
        pass.SetPipeline(ResolveMaterialPipeline(draw.binding, true, draw.top));
        const uint32_t dynamicOffset = static_cast<uint32_t>(offset);
        pass.SetBindGroup(0, group->second, 1, &dynamicOffset);
        pass.DrawIndexed(draw.command->ElemCount, 1, draw.command->IdxOffset,
                         static_cast<int32_t>(draw.command->VtxOffset), 0);
    }
    if (!m_reportedFirstWorldDraw) {
        std::printf("INFERNUX_WEB_WORLD_UI_READY elements=%zu draws=%zu\n", m_worldElements.size(), draws.size());
        m_reportedFirstWorldDraw = true;
    }
    return true;
}

} // namespace infernux::web
