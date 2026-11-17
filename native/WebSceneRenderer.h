#pragma once

#include <function/renderer/EngineGlobals.h>
#include <function/renderer/FullscreenRenderer.h>
#include <function/renderer/MaterialPassPipeline.h>
#include <function/renderer/ScriptableRenderContext.h>
#include <function/renderer/raytracing/RayLitMaterials.h>
#include <function/renderer/rhi/RhiDescriptors.h>
#include <function/renderer/rhi/RhiRenderTexture.h>
#include <function/renderer/shader/ShaderViewBindings.h>
#include <function/resources/InxMaterial/MaterialProperty.h>
#include <memory>
#include <string>
#include <webgpu/webgpu_cpp.h>

namespace infernux
{
class Camera;
}
namespace infernux::web
{
class WebGpuRhiDevice;
class WebRenderGraph;

/// Browser attachments around the shared scene publication, material programs,
/// lighting policy and RHI draw commands.
class WebSceneRenderer final
{
  public:
    WebSceneRenderer();
    ~WebSceneRenderer();
    bool Initialize(WebGpuRhiDevice &device);
    void SetOutputSize(uint32_t width, uint32_t height);
    std::shared_ptr<rhi::RenderTexture> CreateRenderTexture(const rhi::RenderTextureDesc &description);
    std::shared_ptr<rhi::RenderTexture> LoadRenderTexture(const std::string &guid);
    std::shared_ptr<const rhi::TextureResource> ResolveSampledTexture(const std::string &guid,
                                                                      const MaterialTextureSampler *sampler = nullptr);
    std::shared_ptr<const rhi::TextureResource> ResolveGraphTextureAsset(const std::string &guid);
    std::optional<raytracing::RayLitTexture>
    ResolveRayLitTexture(const InxMaterial &material, const RendererParameterBlock *parameters, std::string_view name);
    void StageFrame(const EngineGlobalsUBO &frame);
    std::shared_ptr<const RenderViewPlan> Capture(uint32_t width, uint32_t height, Camera &camera,
                                                  RenderPipelineCallback &pipeline,
                                                  const FullscreenPipelineCache &programs);
    [[nodiscard]] std::vector<MaterialTextureRead> CollectMaterialTextureReads(const RenderViewPlan &plan,
                                                                               const WebRenderGraph &graph) const;
    void Prepare(const RenderViewPlan &plan, WebRenderGraph &graph);
    void AcceptView();
    void ReleaseView(uint64_t cameraId);
    [[nodiscard]] bool HasPreviousViewSample() const noexcept;
    [[nodiscard]] std::shared_ptr<GraphEventViewSample> CaptureViewSample() const;
    void RecordPreparation(wgpu::CommandEncoder encoder);
    void Record(wgpu::RenderPassEncoder pass, const GraphCommandDesc &command,
                const MaterialPassPipelineDescriptor &pipeline);
    [[nodiscard]] const RenderViewData &View() const noexcept;
    [[nodiscard]] rhi::BindingLayoutHandle ViewLayout() const noexcept;
    [[nodiscard]] ShaderViewBindings ParticleView(ShaderCompileTarget target) const noexcept;
    [[nodiscard]] FullscreenViewResources FullscreenView() const noexcept;
    [[nodiscard]] wgpu::TextureView GetDepthView() const noexcept;
    void SetSkyEnabledForDiagnostics(bool enabled) noexcept;
    void SetShadowsEnabledForDiagnostics(bool enabled) noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};
} // namespace infernux::web
