#pragma once

#include <function/renderer/EngineGlobals.h>
#include <function/renderer/MaterialPassPipeline.h>
#include <function/renderer/gui/UIRenderTypes.h>
#include <webgpu/webgpu_cpp.h>

#include <array>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace infernux
{
class Camera;
struct MaterialTextureSampler;
struct ShaderTargetCode;
} // namespace infernux

namespace infernux::web
{

struct UIQuadVertex
{
    float position[2];
    float uv[2];
    float color[4];
};

struct UIWorldVertex
{
    float position[3];
    float uv[2];
    float color[4];
    float localPosition[2];
    float anchor[3];
    float localOffset[3];
    float policy;
};

/// Typed UI programs consume the same asset, material values and engine view
/// ABI as native graphics. WebGPU owns only target objects and their bindings.
class WebUIMaterialCache final
{
  public:
    struct Texture
    {
        wgpu::TextureView view;
        wgpu::Sampler sampler;
    };
    using TextureResolver = std::function<Texture(const std::string &, const MaterialTextureSampler *)>;

    WebUIMaterialCache();
    ~WebUIMaterialCache();
    void Initialize(wgpu::Device device, wgpu::Queue queue, TextureResolver textures);
    void Prune(const std::unordered_set<std::string> &activeMaterials);
    void StageFrame(const EngineGlobalsUBO &frame, uint32_t width, uint32_t height);
    void StageCamera(Camera &camera);
    [[nodiscard]] const wgpu::BindGroupLayout &GetTextureLayout() const noexcept;
    bool Bind(wgpu::RenderPassEncoder pass, const UIShaderMaterialBinding &binding, bool world, bool alwaysOnTop,
              const wgpu::Buffer &constants, uint32_t rootOwner, uint32_t dynamicOffset,
              const MaterialPassPipelineDescriptor &target);

  private:
    struct Prepared;
    Prepared &Resolve(const UIShaderMaterialBinding &binding, bool world);
    void PrepareProgram(Prepared &result, const std::shared_ptr<const ShaderTargetCode> &code, bool world);
    [[nodiscard]] wgpu::RenderPipeline CreatePipeline(const Prepared &prepared, bool world, bool alwaysOnTop,
                                                     const MaterialPassPipelineDescriptor &target);
    [[nodiscard]] bool RefreshTextures(Prepared &prepared);
    [[nodiscard]] wgpu::BindGroup RootGroup(Prepared &prepared, const wgpu::Buffer &constants, uint32_t rootOwner);

    wgpu::Device m_device;
    wgpu::Queue m_queue;
    uint64_t m_frame = 0;
    TextureResolver m_textures;
    EngineGlobalsUBO m_frameGlobals;
    wgpu::Buffer m_cameraBuffer, m_globalsBuffer;
    wgpu::BindGroupLayout m_textureLayout, m_cameraLayout, m_globalsLayout;
    wgpu::BindGroup m_cameraGroup, m_globalsGroup;
    std::array<std::unique_ptr<Prepared>, 2> m_defaults;
    std::unordered_map<std::string, std::unique_ptr<Prepared>> m_prepared;
};

} // namespace infernux::web
