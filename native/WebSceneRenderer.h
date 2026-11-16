#pragma once

#include <function/renderer/EngineGlobals.h>
#include <function/renderer/rhi/RhiDescriptors.h>
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

/// Browser attachments around the shared scene publication, material programs,
/// lighting policy and RHI draw commands.
class WebSceneRenderer final
{
  public:
    WebSceneRenderer();
    ~WebSceneRenderer();
    bool Initialize(WebGpuRhiDevice &device, rhi::PixelFormat colorFormat, uint32_t samples);
    void Resize(uint32_t width, uint32_t height);
    bool ConfigureSampleCount(uint32_t samples);
    bool HasDepthTarget() const noexcept;
    wgpu::TextureView GetDepthView() const noexcept;
    wgpu::Color GetClearColor() const noexcept;
    struct SampledTexture
    {
        wgpu::TextureView view;
        wgpu::Sampler sampler;
    };
    SampledTexture ResolveSampledTexture(const std::string &guid, const MaterialTextureSampler *sampler = nullptr);
    void StageFrame(const EngineGlobalsUBO &frame);
    bool Prepare(wgpu::CommandEncoder encoder, uint32_t width, uint32_t height, Camera *camera);
    bool RenderPrepared(wgpu::RenderPassEncoder pass);
    void SetSkyEnabledForDiagnostics(bool enabled) noexcept;
    void SetShadowsEnabledForDiagnostics(bool enabled) noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};
} // namespace infernux::web
