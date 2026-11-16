#pragma once

#include <function/renderer/EngineGlobals.h>
#include <function/renderer/ScriptableRenderContext.h>
#include <function/renderer/MaterialPassPipeline.h>
#include <function/renderer/FullscreenRenderer.h>
#include <function/renderer/rhi/RhiDescriptors.h>
#include <function/renderer/rhi/RhiRenderTexture.h>
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
    struct SampledTexture
    {
        wgpu::TextureView view;
        wgpu::Sampler sampler;
    };
    SampledTexture ResolveSampledTexture(const std::string &guid, const MaterialTextureSampler *sampler = nullptr);
    void StageFrame(const EngineGlobalsUBO &frame);
    std::shared_ptr<const RenderViewPlan> Capture(uint32_t width, uint32_t height, Camera &camera,
                                                   RenderPipelineCallback &pipeline);
    void Prepare(const RenderViewPlan &plan, const WebRenderGraph &graph);
    void RecordPreparation(wgpu::CommandEncoder encoder);
    void Record(wgpu::RenderPassEncoder pass, const GraphCommandDesc &command,
                   const MaterialPassPipelineDescriptor &pipeline);
    [[nodiscard]] const RenderViewData &View() const noexcept;
    [[nodiscard]] rhi::BindingLayoutHandle ViewLayout() const noexcept;
    [[nodiscard]] FullscreenViewResources FullscreenView() const noexcept;
    [[nodiscard]] wgpu::TextureView GetDepthView() const noexcept;
    void SetSkyEnabledForDiagnostics(bool enabled) noexcept;
    void SetShadowsEnabledForDiagnostics(bool enabled) noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};
} // namespace infernux::web
