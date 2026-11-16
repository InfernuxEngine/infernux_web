#include "WebScreenUIRenderer.h"
#include "WebGpuRhiDevice.h"
#include "WebSceneRenderer.h"
#include "WebTextureUpload.h"
#include <function/renderer/MaterialSamplerState.h>
#include <function/renderer/TextureUploadBuilder.h>
#include <function/scene/Camera.h>

#include "InfernuxWebHostModule.h"
#include <function/renderer/gui/InxTextGeometry.h>
#include <function/renderer/gui/InxTextLayout.h>
#include <function/resources/AssetRegistry/AssetRegistry.h>
#include <function/resources/InxMaterial/InxMaterial.h>
#include <function/resources/InxTexture/InxTexture.h>
#include <function/resources/ShaderAsset/SlangNativeModule.h>
#include <function/resources/ShaderAsset/SlangUIModule.h>
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

WebScreenUIRenderer::WebScreenUIRenderer() = default;

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
    if (m_ownsContext && m_context) {
        textlayout::SetDefaultFont(nullptr);
        textlayout::ClearFontCache();
        ImGui::DestroyContext(m_context);
    }
}

bool WebScreenUIRenderer::Initialize(WebGpuRhiDevice &device, WebSceneRenderer &scene)
{
    m_rhi = &device;
    m_device = device.NativeDevice();
    m_queue = device.NativeQueue();
    m_materials = std::make_unique<WebUIMaterialCache>();
    m_materials->Initialize(m_device, m_queue,
                            [&scene](const std::string &guid, const MaterialTextureSampler *sampler) {
                                const auto texture = scene.ResolveSampledTexture(guid, sampler);
                                return WebUIMaterialCache::Texture{texture.view, texture.sampler};
                            });
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

    return UploadFontAtlas() && CreatePipelineAndFontAtlas();
}

bool WebScreenUIRenderer::UploadFontAtlas()
{
    unsigned char *pixels = nullptr;
    int width = 0, height = 0;
    ImGui::GetIO().Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
    if (!pixels || width < 1 || height < 1)
        return false;
    const size_t rowPitch = static_cast<size_t>(width) * 4;
    const size_t bytes = rowPitch * height;
    rhi::TextureSubresourceUpload mip{
        pixels, bytes, 0, 0, 1, static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1, rowPitch, bytes};
    rhi::TextureUploadRequest upload;
    upload.texture.width = static_cast<uint32_t>(width);
    upload.texture.height = static_cast<uint32_t>(height);
    upload.texture.format = rhi::PixelFormat::RGBA8UNorm;
    upload.texture.usage = rhi::TextureUsageFlags::Sampled | rhi::TextureUsageFlags::TransferDestination;
    upload.sampler = MaterialSamplerDescription("bilinear", "clamp", 1, 1, m_rhi->GetCapabilities());
    upload.subresources = &mip;
    upload.subresourceCount = 1;
    auto resource = m_rhi->UploadTexture(upload);
    if (!resource)
        return false;
    m_fontTexture =
        std::make_shared<const rhi::TextureGpuView>("ui-font-atlas", ++m_textureRevision, std::move(resource));
    m_fontView = m_rhi->GetNativeTextureView(m_fontTexture->GetView());
    m_fontSampler = m_rhi->GetNativeSampler(m_fontTexture->GetSampler());
    m_materialSamplers.clear();
    return true;
}

bool WebScreenUIRenderer::RefreshFontAtlas()
{
    if (!m_fontAtlasDirty || !m_context || !m_device || !m_queue || !m_textureLayout)
        return true;
    ImGui::SetCurrentContext(m_context);
    if (!UploadFontAtlas())
        return false;

    std::array<wgpu::BindGroupEntry, 2> groupEntries{};
    groupEntries[0].binding = 0;
    groupEntries[0].textureView = m_fontView;
    groupEntries[1].binding = 1;
    groupEntries[1].sampler = m_fontSampler;
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
    m_textureLayout = m_materials->GetTextureLayout();
    std::array<wgpu::BindGroupEntry, 2> entries{};
    entries[0].binding = 0;
    entries[0].textureView = m_fontView;
    entries[1].binding = 1;
    entries[1].sampler = m_fontSampler;
    wgpu::BindGroupDescriptor descriptor;
    descriptor.layout = m_textureLayout;
    descriptor.entryCount = entries.size();
    descriptor.entries = entries.data();
    m_fontGroup = m_device.CreateBindGroup(&descriptor);
    return static_cast<bool>(m_fontGroup);
}

void WebScreenUIRenderer::StageFrame(const EngineGlobalsUBO &frame, uint32_t width, uint32_t height)
{
    m_materials->StageFrame(frame, width, height);
}

std::pair<wgpu::Sampler, std::string> WebScreenUIRenderer::ResolveMaterialSampler(const MaterialBinding &binding,
                                                                                  uint64_t textureId)
{
    const auto texture = m_textures.find(textureId);
    if (textureId != kFontTextureId && texture == m_textures.end())
        return {};
    const wgpu::Sampler inherited = textureId == kFontTextureId ? m_fontSampler : texture->second.sampler;
    std::string key = std::to_string(textureId);
    if (binding.materialGuid.empty())
        return {inherited, key};
    const auto material = AssetRegistry::Instance().GetAsset<InxMaterial>(binding.materialGuid);
    if (!material || material->IsDeleted() || material->GetVersion() != binding.generation)
        throw std::runtime_error("UI material GUID is missing or its generation changed: " + binding.materialGuid);
    const MaterialTextureSampler *override = material->GetTextureSampler("albedo_map");
    if (!override || !override->HasOverrides())
        return {inherited, key};
    key.push_back(':');
    for (const auto value : {static_cast<uint8_t>(override->minFilter), static_cast<uint8_t>(override->magFilter),
                             static_cast<uint8_t>(override->mipFilter), static_cast<uint8_t>(override->addressU),
                             static_cast<uint8_t>(override->addressV), static_cast<uint8_t>(override->addressW)})
        key.push_back(static_cast<char>('0' + value));
    if (const auto cached = m_materialSamplers.find(key); cached != m_materialSamplers.end())
        return {m_rhi->GetNativeSampler(cached->second->GetSampler()), key};
    const std::string filterMode = textureId == kFontTextureId ? "bilinear" : texture->second.filterMode;
    const std::string wrapMode = textureId == kFontTextureId ? "clamp" : texture->second.wrapMode;
    const int anisoLevel = textureId == kFontTextureId ? 1 : texture->second.anisoLevel;
    const auto source = textureId == kFontTextureId ? m_fontTexture : texture->second.resource;
    const auto mipCount = textureId == kFontTextureId ? 1u : texture->second.mipLevels;
    const auto descriptor =
        MaterialSamplerDescription(filterMode, wrapMode, anisoLevel, mipCount, m_rhi->GetCapabilities(), *override);
    auto publication = rhi::WithTextureSampler(*m_rhi, source, descriptor);
    if (!publication)
        throw std::runtime_error("WebGPU rejected UI material texture sampler: " + binding.materialGuid);
    const auto sampler = m_rhi->GetNativeSampler(publication->GetSampler());
    m_materialSamplers.emplace(key, std::move(publication));
    return {sampler, key};
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
    m_materialUsageDirty = true;
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

void WebScreenUIRenderer::PruneUnusedMaterials()
{
    if (!m_materialUsageDirty)
        return;
    std::unordered_set<std::string> active;
    for (size_t list = 0; list < m_bindingBoundaries.size(); ++list) {
        const auto *draw = DrawList(static_cast<int>(list));
        const auto &boundaries = m_bindingBoundaries[list];
        size_t boundary = 0;
        const MaterialBinding *binding = nullptr;
        for (int index = 0; index < draw->CmdBuffer.Size; ++index) {
            while (boundary < boundaries.size() && boundaries[boundary].commandStart <= index)
                binding = &boundaries[boundary++].binding;
            const auto &command = draw->CmdBuffer[index];
            if (command.ElemCount && !command.UserCallback && binding && !binding->materialGuid.empty())
                active.insert(binding->materialGuid);
        }
    }
    m_materials->Prune(active);
    m_materialUsageDirty = false;
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
    MaterialBinding next{guid, generation, pipelineKey, baseColor, alphaClipThreshold, alphaClipEnabled};
    if (current.materialGuid == next.materialGuid && current.generation == next.generation &&
        current.pipelineKey == next.pipelineKey && current.baseColor == next.baseColor &&
        current.alphaClipEnabled == next.alphaClipEnabled && current.alphaClipThreshold == next.alphaClipThreshold)
        return;
    m_materialUsageDirty = true;
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
                              alignX, alignY);
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
    const auto sampler = MaterialSamplerDescription(
        filterMode, wrapMode, anisoLevel, static_cast<uint32_t>(texture.mipLevels.size()), m_rhi->GetCapabilities());
    TextureUploadBatch batch(texture, sampler);
    auto resource = m_rhi->UploadTexture(batch.GetRequest());
    if (!resource)
        return 0;
    const uint64_t id = replaceTextureId > kFontTextureId ? replaceTextureId : m_nextTextureId++;
    GPUTexture gpu;
    gpu.resource = std::make_shared<const rhi::TextureGpuView>("ui:" + std::to_string(id), ++m_textureRevision,
                                                               std::move(resource));
    gpu.view = m_rhi->GetNativeTextureView(gpu.resource->GetView());
    gpu.sampler = m_rhi->GetNativeSampler(gpu.resource->GetSampler());
    gpu.filterMode = filterMode;
    gpu.wrapMode = wrapMode;
    gpu.anisoLevel = anisoLevel;
    gpu.mipLevels = batch.GetRequest().texture.mipLevels;
    gpu.srgb = TextureFormatIsSrgb(texture.format);

    std::array<wgpu::BindGroupEntry, 2> entries{};
    entries[0].binding = 0;
    entries[0].textureView = gpu.view;
    entries[1].binding = 1;
    entries[1].sampler = gpu.sampler;
    wgpu::BindGroupDescriptor groupDescriptor;
    groupDescriptor.layout = m_textureLayout;
    groupDescriptor.entryCount = entries.size();
    groupDescriptor.entries = entries.data();
    gpu.group = m_device.CreateBindGroup(&groupDescriptor);
    if (!gpu.group)
        return 0;

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

bool WebScreenUIRenderer::Render(wgpu::RenderPassEncoder pass, int list, uint32_t width, uint32_t height,
                                  const MaterialPassPipelineDescriptor &target)
{
    ImDrawList *draw = DrawList(list);
    if (!pass || !draw || draw->VtxBuffer.empty() || draw->IdxBuffer.empty() || width == 0 || height == 0)
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
    auto &buffers = m_screenBuffers[static_cast<size_t>(list)];
    const uint64_t vertexBytes = vertices.size() * sizeof(GPUVertex);
    const uint64_t indexBytes = static_cast<uint64_t>(draw->IdxBuffer.Size) * sizeof(ImDrawIdx);
    constexpr uint64_t constantStride = 256;
    static_assert(sizeof(ScreenConstants) <= constantStride);
    if (!EnsureBuffer(buffers.vertex, buffers.vertexCapacity, vertexBytes, wgpu::BufferUsage::Vertex) ||
        !EnsureBuffer(buffers.index, buffers.indexCapacity, indexBytes, wgpu::BufferUsage::Index) ||
        !EnsureBuffer(buffers.constants, buffers.constantsCapacity,
                      constantStride * static_cast<uint64_t>(draw->CmdBuffer.Size), wgpu::BufferUsage::Uniform))
        return false;
    m_queue.WriteBuffer(buffers.vertex, 0, vertices.data(), vertexBytes);
    m_queue.WriteBuffer(buffers.index, 0, draw->IdxBuffer.Data, indexBytes);
    pass.SetVertexBuffer(0, buffers.vertex, 0, vertexBytes);
    pass.SetIndexBuffer(buffers.index, sizeof(ImDrawIdx) == 2 ? wgpu::IndexFormat::Uint16 : wgpu::IndexFormat::Uint32,
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
        float encodeSample = 0.0f;
        if (textureId != kFontTextureId) {
            const auto texture = m_textures.find(textureId);
            if (texture == m_textures.end())
                continue;
            textureView = texture->second.view;
            encodeSample = list == 1 && texture->second.srgb ? 1.0f : 0.0f;
        }
        const auto [sampler, samplerKey] = ResolveMaterialSampler(binding, textureId);
        auto group = textureGroups.find(samplerKey);
        if (group == textureGroups.end()) {
            std::array<wgpu::BindGroupEntry, 2> entries{};
            entries[0].binding = 0;
            entries[0].textureView = textureView;
            entries[1].binding = 1;
            entries[1].sampler = sampler;
            wgpu::BindGroupDescriptor descriptor;
            descriptor.layout = m_textureLayout;
            descriptor.entryCount = entries.size();
            descriptor.entries = entries.data();
            group = textureGroups.emplace(samplerKey, m_device.CreateBindGroup(&descriptor)).first;
        }
        if (!group->second)
            return false;
        ScreenConstants constants;
        constants.scale[0] = 2.0f / static_cast<float>(width);
        constants.scale[1] = 2.0f / static_cast<float>(height);
        constants.translate[0] = -1.0f;
        constants.translate[1] = -1.0f;
        constants.encodeSample = encodeSample;
        constants.materialColor = glm::make_vec4(binding.baseColor.data());
        constants.alphaClipEnabled = binding.alphaClipEnabled ? 1.0f : 0.0f;
        constants.alphaClipThreshold = binding.alphaClipThreshold;
        const uint64_t offset = static_cast<uint64_t>(commandIndex) * constantStride;
        m_queue.WriteBuffer(buffers.constants, offset, &constants, sizeof(constants));
        const uint32_t dynamicOffset = static_cast<uint32_t>(offset);
        if (!m_materials->Bind(pass, binding, false, false, buffers.constants, uint32_t(list), dynamicOffset, target))
            continue;
        pass.SetBindGroup(3, group->second);
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

bool WebScreenUIRenderer::RenderWorld(wgpu::RenderPassEncoder pass, Camera &camera, uint32_t width, uint32_t height,
                                       const MaterialPassPipelineDescriptor &target, uint32_t layerMask)
{
    m_materials->StageCamera(camera);
    const auto viewProjection = camera.GetViewProjectionMatrix();
    const auto view = camera.GetViewMatrix();
    const auto projection = camera.GetProjectionMatrix();
    const auto cullingMask = camera.GetCullingMask() & layerMask;
    if (!pass || !m_world || m_worldElements.empty() || width == 0 || height == 0 || m_worldElementOpen) {
        if (!m_reportedWorldCommandState && m_world && (m_world->VtxBuffer.Size > 0 || !m_worldElements.empty())) {
            std::printf("INFERNUX_WEB_WORLD_UI_COMMAND_STATE spans=%zu vertices=%d indices=%d open=%d "
                        "size=%ux%u\n",
                        m_worldElements.size(), m_world->VtxBuffer.Size, m_world->IdxBuffer.Size,
                        m_worldElementOpen ? 1 : 0, width, height);
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
            std::array<wgpu::BindGroupEntry, 2> entries{};
            entries[0].binding = 0;
            entries[0].textureView = textureView;
            entries[1].binding = 1;
            entries[1].sampler = sampler;
            wgpu::BindGroupDescriptor descriptor;
            descriptor.layout = m_textureLayout;
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
        const uint32_t dynamicOffset = static_cast<uint32_t>(offset);
        if (!m_materials->Bind(pass, draw.binding, true, draw.top, m_worldConstantsBuffer, 2, dynamicOffset, target))
            continue;
        pass.SetBindGroup(3, group->second);
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
