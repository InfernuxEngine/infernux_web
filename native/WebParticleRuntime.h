#pragma once

#include <Python.h>
#include <webgpu/webgpu_cpp.h>

#include <function/renderer/rhi/RhiHandles.h>

#include <cstdint>
#include <memory>
#include <string>

namespace infernux
{
struct MaterialPassPipelineDescriptor;
struct RenderViewData;
struct ShaderViewBindings;
} // namespace infernux

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

    [[nodiscard]] bool Initialize(WebGpuRhiDevice &device, WebSceneRenderer &scene);
    void Shutdown() noexcept;

    [[nodiscard]] std::string ReplaceGraph(uint64_t graphInstanceId, PyObject *programs, PyObject *removeIds);
    [[nodiscard]] std::string ReplaceGraphs(PyObject *graphs);
    [[nodiscard]] std::string UpdateParameters(uint64_t graphInstanceId, PyObject *updates);
    [[nodiscard]] bool BeginBatch(uint64_t graphInstanceId, PyObject *items);
    [[nodiscard]] bool SetPlaying(uint64_t emitterId, bool playing);
    [[nodiscard]] bool Reset(uint64_t emitterId);
    [[nodiscard]] uint64_t ArtifactRevision(uint64_t emitterId) const noexcept;
    [[nodiscard]] bool StateWasPreserved(uint64_t emitterId) const noexcept;

    [[nodiscard]] bool RecordCompute(wgpu::CommandEncoder encoder);
    void NotifySubmission() noexcept;
    [[nodiscard]] bool RequiresSceneDepth(uint32_t cameraMask, int32_t queueMin, int32_t queueMax) const noexcept;
    [[nodiscard]] bool Render(wgpu::RenderPassEncoder pass, const RenderViewData &view,
                              const ShaderViewBindings &bindings, const MaterialPassPipelineDescriptor &target,
                              int32_t queueMin, int32_t queueMax, rhi::TextureViewHandle sceneDepth = {},
                              bool sceneDepthIsDepth = true);
    void ReleaseView(rhi::BufferHandle globals) noexcept;

    [[nodiscard]] const std::string &LastError() const noexcept;

  private:
    struct State;
    std::unique_ptr<State> m_state;
};

} // namespace infernux::web
