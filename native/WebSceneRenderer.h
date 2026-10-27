#pragma once

#include "WebIndexPacking.h"

#include <function/renderer/RenderWorld.h>
#include <function/resources/InxMaterial/MaterialProperty.h>
#include <function/scene/SceneRenderExtractor.h>
#include <webgpu/webgpu_cpp.h>

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace infernux
{
class InxMaterial;
class Camera;
struct RendererParameterBlock;
class TextureUploadStagingTicket;
struct TextureCpuData;
} // namespace infernux

namespace infernux::web
{

/// WebGPU consumer for the engine's immutable scene publication.
///
/// Scene traversal remains in SceneRenderExtractor. This backend owns only
/// WebGPU resources and converts the published draw stream into one portable
/// lit mesh pass.
class WebSceneRenderer final
{
  public:
    bool Initialize(wgpu::Device device, wgpu::Queue queue, wgpu::TextureFormat colorFormat, uint32_t sceneSampleCount);
    void Resize(uint32_t width, uint32_t height);
    bool ConfigureSampleCount(uint32_t sampleCount);

    [[nodiscard]] bool HasDepthTarget() const noexcept;
    [[nodiscard]] wgpu::TextureView GetDepthView() const noexcept;

    /// Linear HDR clear color from the same camera used by Prepare().
    [[nodiscard]] wgpu::Color GetClearColor() const noexcept
    {
        return m_clearColor;
    }

    struct SampledTexture
    {
        wgpu::TextureView view;
        wgpu::Sampler sampler;
    };
    [[nodiscard]] SampledTexture ResolveSampledTexture(const std::string &guid);

    /// Extract, upload, and record the directional shadow map before the main
    /// render pass begins.
    bool Prepare(wgpu::CommandEncoder encoder, uint32_t width, uint32_t height, Camera *camera);

    /// Draw the prepared procedural sky and shadowed scene geometry.
    bool RenderPrepared(wgpu::RenderPassEncoder pass);

    /// Acceptance-only feature switches used to prove that sky and shadows
    /// materially affect the composed browser frame. Gameplay cannot access
    /// these switches through the Python runtime surface.
    void SetSkyEnabledForDiagnostics(bool enabled) noexcept;
    void SetShadowsEnabledForDiagnostics(bool enabled) noexcept;

  private:
    struct WebVertex
    {
        float position[3];
        float normal[3];
        float tangent[4];
        float uv[2];
        float uv1[2];
        float color[4];
        float emission[4];
        // x = metallic, y = smoothness, z = occlusion, w = specular highlights.
        float material[4];
        // x = 0 PBR / 1 unlit / 2 toon, y/z = toon threshold/softness,
        // w = alpha-clip threshold or -1 when clipping is disabled.
        float surface[4];
        // Per-texture UV set: base color, metallic, smoothness, occlusion.
        float textureUvSets0[4];
        // x = normal UV set, y = emission UV set.
        float textureUvSets1[2];
        float metallicChannels[4];
        float smoothnessChannels[4];
        // x = smoothness-from-roughness, y = occlusion strength.
        float materialSampling[2];
    };

    struct WebDrawRange
    {
        WebPackedIndexRange indices;
        uint32_t drawIndex = 0;
        bool transparent = false;
        bool castsShadows = true;
        bool line = false;
        bool fullyCulled = false;
        uint8_t rasterPipeline = 5;
        wgpu::BindGroup materialTextureGroup;
    };

    struct alignas(16) WebDrawData
    {
        glm::mat4 model{1.0f};
        glm::mat4 normal{1.0f};
    };

    struct GPUTexture
    {
        wgpu::Texture texture;
        wgpu::TextureView view;
        wgpu::Sampler sampler;
        uint32_t mipLevels = 0;
    };

    struct MaterialTextureState
    {
        std::shared_ptr<TextureUploadStagingTicket> ticket;
        GPUTexture gpu;
        std::string filterMode = "linear";
        std::string wrapMode = "repeat";
        int anisoLevel = 1;
        bool failed = false;
    };

    struct MaterialTextureSetState
    {
        wgpu::BindGroup group;
        uint64_t textureGeneration = 0;
    };

    struct alignas(16) CameraData
    {
        struct alignas(16) PunctualLightData
        {
            glm::vec4 positionRange{0.0f};
            glm::vec4 colorIntensity{0.0f};
            glm::vec4 directionOuterCos{0.0f};
            // x = inner cone cosine, y = 0 point / 1 spot.
            glm::vec4 parameters{0.0f};
        };

        glm::mat4 viewProjection{1.0f};
        glm::mat4 inverseViewProjection{1.0f};
        glm::mat4 lightViewProjection{1.0f};
        glm::vec4 cameraPosition{0.0f, 0.0f, 0.0f, 1.0f};
        glm::vec4 lightDirectionStrength{0.35f, 0.82f, 0.45f, 0.0f};
        glm::vec4 lightColorIntensity{1.0f};
        glm::vec4 skyTopExposure{0.431f, 0.494f, 0.612f, 1.0f};
        glm::vec4 skyHorizon{0.651f, 0.725f, 0.816f, 1.0f};
        glm::vec4 skyGround{0.345f, 0.345f, 0.345f, 1.0f};
        glm::vec4 ambient{0.16f, 0.17f, 0.20f, 1.0f};
        glm::vec4 ambientSky{0.16f, 0.17f, 0.20f, 1.0f};
        // xyz = equator radiance, w = 0 flat / 1 gradient.
        glm::vec4 ambientEquator{0.16f, 0.17f, 0.20f, 0.0f};
        glm::vec4 ambientGround{0.16f, 0.17f, 0.20f, 1.0f};
        glm::uvec4 lightCounts{0u};
        std::array<PunctualLightData, 8> punctualLights{};
    };

    bool CreatePipelines();
    bool CreateShadowResources();
    bool CreateMaterialTextureResources();
    bool CreateCameraBindGroups();
    [[nodiscard]] GPUTexture ResolveMaterialTexture(const std::string &guid, const MaterialTextureSampler &sampler,
                                                    const GPUTexture &fallback);
    [[nodiscard]] wgpu::BindGroup
    ResolveMaterialTextureSet(const std::shared_ptr<InxMaterial> &material,
                              const std::shared_ptr<const RendererParameterBlock> &parameters);
    [[nodiscard]] wgpu::BindGroup CreateMaterialTextureGroup(const std::array<GPUTexture, 6> &textures);
    [[nodiscard]] GPUTexture UploadMaterialTexture(const TextureCpuData &texture, const std::string &filterMode,
                                                   const std::string &wrapMode, int anisoLevel,
                                                   const MaterialTextureSampler &sampler = {});
    bool EnsureBuffer(wgpu::Buffer &buffer, uint64_t &capacity, uint64_t required, wgpu::BufferUsage usage);
    [[nodiscard]] bool BindIndexStream(wgpu::RenderPassEncoder pass, MeshIndexFormat format) const;
    bool BuildFrame(uint32_t width, uint32_t height, Camera *camera);
    void ReportFrameIssue(const char *issue);

    wgpu::Device m_device;
    wgpu::Queue m_queue;
    wgpu::TextureFormat m_colorFormat = wgpu::TextureFormat::Undefined;
    // One pipeline for each (front face, cull mode) pair. Material render
    // state is part of the shared RHI contract and cannot be flattened to a
    // Web-only double-sided pipeline without exposing back faces.
    std::array<wgpu::RenderPipeline, 6> m_opaquePipelines;
    std::array<wgpu::RenderPipeline, 6> m_transparentPipelines;
    wgpu::RenderPipeline m_skyPipeline;
    std::array<wgpu::RenderPipeline, 6> m_shadowPipelines;
    wgpu::BindGroupLayout m_cameraLayout;
    wgpu::BindGroup m_cameraGroup;
    wgpu::BindGroupLayout m_shadowCameraLayout;
    wgpu::BindGroup m_shadowCameraGroup;
    wgpu::BindGroupLayout m_materialTextureLayout;
    GPUTexture m_whiteTexture;
    GPUTexture m_blackTexture;
    GPUTexture m_normalTexture;
    wgpu::BindGroup m_defaultMaterialTextureGroup;
    wgpu::BindGroup m_skyTextureGroup;
    std::unordered_map<std::string, MaterialTextureState> m_materialTextures;
    std::unordered_map<std::string, MaterialTextureSetState> m_materialTextureSets;
    uint64_t m_materialTextureGeneration = 1;
    wgpu::Buffer m_cameraBuffer;
    wgpu::Buffer m_drawDataBuffer;
    wgpu::Buffer m_vertexBuffer;
    wgpu::Buffer m_indexBuffer16;
    wgpu::Buffer m_indexBuffer32;
    wgpu::Texture m_shadowTexture;
    wgpu::TextureView m_shadowView;
    wgpu::Sampler m_shadowSampler;
    uint64_t m_vertexCapacity = 0;
    uint64_t m_drawDataCapacity = 0;
    uint64_t m_indexCapacity16 = 0;
    uint64_t m_indexCapacity32 = 0;
    wgpu::Texture m_depthTexture;
    wgpu::TextureView m_depthView;
    uint32_t m_depthWidth = 0;
    uint32_t m_depthHeight = 0;
    uint32_t m_sceneSampleCount = 1;
    uint32_t m_shadowResolution = 2048;
    SceneRenderExtractor m_extractor;
    RenderWorldSnapshot m_world;
    std::vector<WebVertex> m_vertices;
    WebPackedIndexStreams m_indexStreams;
    std::vector<WebDrawRange> m_drawRanges;
    std::vector<WebDrawData> m_drawData;
    CameraData m_cameraData;
    wgpu::Color m_clearColor{0.0, 0.0, 0.0, 1.0};
    std::string m_lastFrameIssue;
    size_t m_residentSceneCount = 0;
    size_t m_residentRendererCount = 0;
    size_t m_residentLightCount = 0;
    uint64_t m_activeWorldId = 0;
    bool m_reportedFirstFrame = false;
    bool m_reportedLineDraw = false;
    bool m_framePrepared = false;
    bool m_drawSky = true;
    bool m_shadowEnabled = false;
    bool m_diagnosticSkyEnabled = true;
    bool m_diagnosticShadowsEnabled = true;
};

} // namespace infernux::web
