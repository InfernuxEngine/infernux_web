#pragma once

#include <function/renderer/FullscreenRenderer.h>
#include <function/renderer/MaterialPassPipeline.h>
#include <function/renderer/RenderGraphDescription.h>
#include <function/renderer/RenderViewSchedule.h>
#include <function/renderer/rhi/RenderViewContext.h>
#include <function/renderer/rhi/RhiRasterPass.h>
#include <functional>
#include <memory>
#include <webgpu/webgpu_cpp.h>

namespace infernux::rhi
{
class BufferResource;
class ComputeHost;
} // namespace infernux::rhi

namespace infernux
{
class GraphFramePublication;
class RenderWorldFrame;
namespace raytracing
{
struct RayWorldGeometryBinding;
struct RayWorldMaterialPublication;
struct RayTailCacheBinding;
} // namespace raytracing
} // namespace infernux

namespace infernux::web
{
class WebGpuRhiDevice;

/// WebGPU recording adapter for the engine's published graph. Pipeline authors,
/// effects, parameter blocks and dependency semantics are shared with Vulkan.
class WebRenderGraph final
{
  public:
    using Texture = std::shared_ptr<const rhi::TextureResource>;
    using ImportTexture = std::function<Texture(const GraphTextureDesc &)>;
    using RecordRaster =
        std::function<void(wgpu::RenderPassEncoder, const GraphCommandDesc &, const rhi::RasterPassDescription &,
                           const MaterialPassPipelineDescriptor &, const rhi::TextureResource *)>;
    explicit WebRenderGraph(WebGpuRhiDevice &device, std::shared_ptr<FullscreenRendererHost> fullscreenHost,
                            rhi::ComputeHost *computeHost = nullptr,
                            std::shared_ptr<FullscreenPipelineCache> pipelines = {});
    ~WebRenderGraph();
    /// Compile only when topology or view extent changes. External texture
    /// generations are part of that publication, never looked up while drawing.
    bool Apply(std::shared_ptr<const RenderGraphDescription> graph, const rhi::RenderViewContext &view,
               const ImportTexture &imports = {}, const std::optional<GraphPassDesc> &cameraClear = {},
               std::shared_ptr<const rhi::RenderTextureGeneration> output = {}, const glm::vec4 &rect = {0, 0, 1, 1},
               bool sampleParticleDepth = false);
    void UpdateParameters(const std::vector<GraphParameterBlockUpdate> &updates);
    void Record(wgpu::CommandEncoder encoder, const FullscreenViewResources &view,
                const raytracing::RayWorldGeometryBinding &rays,
                const raytracing::RayWorldMaterialPublication *worldRayMaterials,
                const raytracing::RayTailCacheBinding &tail, const RecordRaster &raster);
    [[nodiscard]] std::shared_ptr<void> RetainWorkStorage() const;
    [[nodiscard]] std::shared_ptr<GraphBufferCapture> CaptureBuffer(const std::string &name);
    [[nodiscard]] std::shared_ptr<GraphFramePublication>
    PreparePublication(const GraphEventInterests &interests,
                       const std::function<std::shared_ptr<GraphEventViewSample>()> &sample);
    void AcceptHistory();
    void InvalidateHistory() noexcept;
    void UpdateRayHistory(const RenderWorldFrame &world, const raytracing::RayWorldGeometryBinding &rays,
                          const std::shared_ptr<const raytracing::RayWorldMaterialPublication> &materials);
    [[nodiscard]] uint32_t AcceptedSampleIndex() const noexcept;
    [[nodiscard]] uint32_t AcceptedRayGeometry() const noexcept;
    [[nodiscard]] const RenderGraphDescription &Description() const noexcept;
    [[nodiscard]] Texture GetTexture(const std::string &name) const;
    [[nodiscard]] Texture TransparentLayer() const noexcept;
    [[nodiscard]] Texture Output() const;
    [[nodiscard]] const std::vector<uint32_t> &ExecutionOrder() const noexcept;
    [[nodiscard]] RenderViewAccess GetResourceAccess(const std::vector<MaterialTextureRead> &materialReads = {}) const;

  private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};
} // namespace infernux::web
