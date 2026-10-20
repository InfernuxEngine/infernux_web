#pragma once

#include <Python.h>
#include <webgpu/webgpu_cpp.h>

#include <function/renderer/rhi/RhiTypes.h>

#include <cstdint>
#include <memory>
#include <string>

namespace infernux
{
class Camera;
}

namespace infernux::web
{

class WebGpuRhiDevice;
class WebSceneRenderer;

/// WebGPU execution and rendering host for the portable ParticleGraph subset.
/// Python owns scheduling; this class owns only browser GPU residency.
class WebParticleRuntime final
{
  public:
    WebParticleRuntime();
    ~WebParticleRuntime();

    WebParticleRuntime(const WebParticleRuntime &) = delete;
    WebParticleRuntime &operator=(const WebParticleRuntime &) = delete;

    [[nodiscard]] bool Initialize(WebGpuRhiDevice &device, WebSceneRenderer &scene, rhi::PixelFormat colorFormat,
                                  rhi::SampleCount sceneSampleCount);
    [[nodiscard]] bool ConfigureSampleCount(rhi::SampleCount sceneSampleCount);
    void Shutdown() noexcept;

    [[nodiscard]] std::string ReplaceGraph(uint64_t graphInstanceId, PyObject *programs, PyObject *removeIds);
    [[nodiscard]] std::string UpdateParameters(uint64_t graphInstanceId, PyObject *words);
    [[nodiscard]] bool BeginBatch(uint64_t graphInstanceId, PyObject *items);
    [[nodiscard]] bool SetPlaying(uint64_t emitterId, bool playing);
    [[nodiscard]] bool Reset(uint64_t emitterId);
    [[nodiscard]] uint64_t ArtifactRevision(uint64_t emitterId) const noexcept;
    [[nodiscard]] bool StateWasPreserved(uint64_t emitterId) const noexcept;

    [[nodiscard]] bool RecordCompute(wgpu::CommandEncoder encoder);
    void NotifySubmission() noexcept;
    [[nodiscard]] bool HasRenderWork(uint32_t cameraMask) const noexcept;
    [[nodiscard]] bool Render(wgpu::RenderPassEncoder pass, Camera &camera, uint32_t width, uint32_t height);

    [[nodiscard]] const std::string &LastError() const noexcept;

  private:
    struct State;
    std::unique_ptr<State> m_state;
};

} // namespace infernux::web
