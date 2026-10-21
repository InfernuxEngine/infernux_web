#pragma once

#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>
#include <imgui.h>
#include <webgpu/webgpu_cpp.h>

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace infernux
{
namespace textlayout
{
class TextGeometry;
}
struct TextureCpuData;
struct MaterialTextureSampler;
} // namespace infernux

namespace infernux::web
{

/// Engine-owned Screen UI consumer for the WebGPU Player.
///
/// Python submits the same runtime UI packets used by the Vulkan Player. This
/// class owns only command accumulation, font-atlas upload, and WebGPU draw
/// recording; no visible browser DOM participates in presentation.
class WebScreenUIRenderer final
{
  public:
    WebScreenUIRenderer() = default;
    ~WebScreenUIRenderer();

    bool Initialize(wgpu::Device device, wgpu::Queue queue, wgpu::TextureFormat colorFormat,
                    wgpu::TextureFormat worldColorFormat, uint32_t worldSampleCount);
    bool ConfigureSampleCount(uint32_t sampleCount);
    void BeginFrame(uint32_t width, uint32_t height);
    bool BeginFrameCached(uint32_t width, uint32_t height, uint64_t contentRevision);
    void PushClipRect(int list, float minX, float minY, float maxX, float maxY);
    void PopClipRect(int list);
    void SetMaterialBinding(int list, const std::string &guid, uint64_t generation, const std::string &pipelineKey,
                            const std::array<float, 4> &baseColor, bool alphaClipEnabled, float alphaClipThreshold);
    void BeginWorldElement(const std::array<float, 16> &localToWorld, float pivotX, float pivotY, uint32_t layer,
                           bool alwaysOnTop, bool billboard, bool constantScreenSize, uint64_t ignoredOccluderId);
    void EndWorldElement();

    void AddFilledRect(int list, float minX, float minY, float maxX, float maxY, float r, float g, float b, float a,
                       float rounding);
    void AddImage(int list, uint64_t textureId, float minX, float minY, float maxX, float maxY, float uv0X, float uv0Y,
                  float uv1X, float uv1Y, float r, float g, float b, float a, float rotation, bool mirrorH,
                  bool mirrorV, float rounding);
    void AddText(int list, float minX, float minY, float maxX, float maxY, const std::string &text, float r, float g,
                 float b, float a, float alignX, float alignY, float fontSize, float wrapWidth, float rotation,
                 bool mirrorH, bool mirrorV, const std::string &fontPath, float lineHeight, float letterSpacing,
                 bool clip, const std::vector<std::string> &fallbackFontPaths = {});
    void AddTextGeometry(int list, float minX, float minY, float maxX, float maxY, textlayout::TextGeometry &geometry,
                         float r, float g, float b, float a, float alignX, float alignY, float scale, float rotation,
                         bool mirrorH, bool mirrorV, bool clip);
    [[nodiscard]] std::pair<float, float> MeasureText(const std::string &text, float fontSize, float wrapWidth,
                                                      const std::string &fontPath, float lineHeight,
                                                      float letterSpacing,
                                                      const std::vector<std::string> &fallbackFontPaths = {});
    [[nodiscard]] uint64_t UploadTexture(const TextureCpuData &texture, uint64_t replaceTextureId = 0,
                                         const std::string &filterMode = "bilinear",
                                         const std::string &wrapMode = "clamp", int anisoLevel = 1);
    void ReleaseTexture(uint64_t textureId);

    bool Render(wgpu::RenderPassEncoder pass, int list, uint32_t width, uint32_t height);
    bool RenderWorld(wgpu::RenderPassEncoder pass, const glm::mat4 &viewProjection, const glm::mat4 &view,
                     const glm::mat4 &projection, uint32_t cullingMask, uint32_t width, uint32_t height);

  private:
    struct MaterialBinding
    {
        std::string guid;
        uint64_t generation = 0;
        std::string pipelineKey;
        std::array<float, 4> baseColor{1.0f, 1.0f, 1.0f, 1.0f};
        bool alphaClipEnabled = false;
        float alphaClipThreshold = 0.0f;
    };

    struct BindingBoundary
    {
        int commandStart = 0;
        MaterialBinding binding;
    };

    struct WorldElementSpan
    {
        int vertexStart = 0;
        int vertexEnd = 0;
        int commandStart = 0;
        int commandEnd = 0;
        glm::mat4 localToWorld{1.0f};
        float pivotX = 0.0f;
        float pivotY = 0.0f;
        uint32_t layer = 0;
        bool alwaysOnTop = false;
        bool billboard = false;
        bool constantScreenSize = false;
        uint64_t ignoredOccluderId = 0;
    };

    struct GPUVertex
    {
        float position[2];
        float uv[2];
        float color[4];
    };

    struct WorldVertex
    {
        float position[3];
        float uv[2];
        float color[4];
        float localPosition[2];
        float anchor[3];
        float localOffset[2];
        float policy;
    };

    struct alignas(16) WorldConstants
    {
        glm::mat4 viewProjection{1.0f};
        glm::vec4 materialColor{1.0f};
        float alphaClipThreshold = 0.0f;
        float alphaClipEnabled = 0.0f;
        float screenScale[2]{};
        glm::vec4 cameraRight{1.0f, 0.0f, 0.0f, 0.0f};
        glm::vec4 cameraUp{0.0f, 1.0f, 0.0f, 0.0f};
    };

    struct alignas(16) ScreenConstants
    {
        float scale[2]{};
        float translate[2]{};
        float encodeSample = 0.0f;
        float padding[3]{};
        glm::vec4 materialColor{1.0f};
        float alphaClipThreshold = 0.0f;
        float alphaClipEnabled = 0.0f;
        float tailPadding[2]{};
    };

    struct GPUTexture
    {
        wgpu::Texture texture;
        wgpu::TextureView view;
        wgpu::Sampler sampler;
        wgpu::BindGroup group;
        std::string filterMode = "bilinear";
        std::string wrapMode = "clamp";
        int anisoLevel = 1;
        uint32_t mipLevels = 1;
    };

    ImDrawList *DrawList(int list) const;
    bool CreatePipelineAndFontAtlas();
    bool CreateWorldPipelines();
    wgpu::RenderPipeline ResolveMaterialPipeline(const MaterialBinding &binding, bool world, bool alwaysOnTop);
    [[nodiscard]] std::pair<wgpu::Sampler, std::string> ResolveMaterialSampler(const MaterialBinding &binding,
                                                                               uint64_t textureId);
    bool RefreshFontAtlas();
    bool EnsureBuffer(wgpu::Buffer &buffer, uint64_t &capacity, uint64_t required, wgpu::BufferUsage usage);

    wgpu::Device m_device;
    wgpu::Queue m_queue;
    wgpu::TextureFormat m_colorFormat = wgpu::TextureFormat::Undefined;
    wgpu::RenderPipeline m_pipeline;
    wgpu::RenderPipeline m_worldDepthPipeline;
    wgpu::RenderPipeline m_worldTopPipeline;
    std::unordered_map<std::string, wgpu::RenderPipeline> m_customScreenPipelines;
    std::unordered_map<std::string, wgpu::RenderPipeline> m_customWorldPipelines;
    wgpu::BindGroupLayout m_textureLayout;
    wgpu::BindGroupLayout m_screenLayout;
    wgpu::BindGroupLayout m_worldLayout;
    wgpu::Buffer m_worldConstantsBuffer;
    wgpu::Buffer m_screenConstantsBuffer;
    wgpu::Buffer m_worldVertexBuffer;
    wgpu::Buffer m_worldIndexBuffer;
    uint64_t m_worldConstantsCapacity = 0;
    uint64_t m_screenConstantsCapacity = 0;
    uint64_t m_worldVertexCapacity = 0;
    uint64_t m_worldIndexCapacity = 0;
    wgpu::TextureFormat m_worldColorFormat = wgpu::TextureFormat::Undefined;
    uint32_t m_worldSampleCount = 1;
    wgpu::BindGroup m_fontGroup;
    wgpu::Texture m_fontTexture;
    wgpu::TextureView m_fontView;
    wgpu::Sampler m_fontSampler;
    uint32_t m_fontWidth = 0;
    uint32_t m_fontHeight = 0;
    wgpu::Buffer m_vertexBuffer;
    wgpu::Buffer m_indexBuffer;
    uint64_t m_vertexCapacity = 0;
    uint64_t m_indexCapacity = 0;
    std::unordered_map<uint64_t, GPUTexture> m_textures;
    std::unordered_map<std::string, wgpu::Sampler> m_materialSamplers;
    uint64_t m_nextTextureId = 2;
    ImGuiContext *m_context = nullptr;
    ImDrawList *m_camera = nullptr;
    ImDrawList *m_overlay = nullptr;
    ImDrawList *m_world = nullptr;
    std::array<std::vector<BindingBoundary>, 3> m_bindingBoundaries;
    std::array<MaterialBinding, 3> m_currentBindings;
    std::vector<WorldElementSpan> m_worldElements;
    WorldElementSpan m_pendingWorldElement;
    bool m_worldElementOpen = false;
    bool m_ownsContext = false;
    bool m_cacheValid = false;
    uint32_t m_cachedWidth = 0;
    uint32_t m_cachedHeight = 0;
    uint64_t m_cachedRevision = 0;
    bool m_reportedFirstDraw = false;
    bool m_fontAtlasDirty = false;
    bool m_reportedFirstWorldDraw = false;
    bool m_reportedWorldCommandState = false;
};

} // namespace infernux::web
