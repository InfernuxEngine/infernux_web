#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <tuple>
#include <webgpu/webgpu_cpp.h>

namespace infernux
{
struct EngineGlobalsUBO;
}

namespace infernux::web
{
class WebGpuRhiDevice;

/// Browser attachment adapter for shared fullscreen programs and RHI targets.
class WebPostProcessRenderer final
{
  public:
    struct Settings
    {
        bool bloomEnabled = false;
        float bloomThreshold = 1.0f;
        float bloomIntensity = 0.8f;
        float bloomScatter = 0.7f;
        float bloomClamp = 65472.0f;
        std::array<float, 3> bloomTint = {1.0f, 1.0f, 1.0f};
        uint32_t bloomIterations = 5;
        uint32_t toneMappingMode = 0;
        float exposure = 1.0f;
        bool operator==(const Settings &other) const noexcept
        {
            return std::tie(bloomEnabled, bloomThreshold, bloomIntensity, bloomScatter, bloomClamp, bloomTint,
                            bloomIterations, toneMappingMode,
                            exposure) == std::tie(other.bloomEnabled, other.bloomThreshold, other.bloomIntensity,
                                                  other.bloomScatter, other.bloomClamp, other.bloomTint,
                                                  other.bloomIterations, other.toneMappingMode, other.exposure);
        }
    };

    WebPostProcessRenderer();
    ~WebPostProcessRenderer();
    bool Initialize(WebGpuRhiDevice &device, wgpu::TextureFormat surfaceFormat, uint32_t sceneSampleCount);
    bool Resize(uint32_t width, uint32_t height);
    bool ConfigureSampleCount(uint32_t sampleCount);
    bool Configure(const Settings &settings);
    void StageFrame(const EngineGlobalsUBO &frame);
    bool SetBloomEnabledForDiagnostics(bool enabled);

    [[nodiscard]] wgpu::TextureFormat SceneColorFormat() const noexcept;
    [[nodiscard]] uint32_t SceneSampleCount() const noexcept;
    [[nodiscard]] wgpu::TextureView SceneColorAttachmentView() const noexcept;
    [[nodiscard]] wgpu::TextureView SceneColorView() const noexcept;
    [[nodiscard]] bool Prepare(wgpu::CommandEncoder encoder);
    [[nodiscard]] bool Render(wgpu::RenderPassEncoder pass, bool dithering = false, bool stopNaNs = false);

  private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};
} // namespace infernux::web
