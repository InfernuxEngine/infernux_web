#include "WebPostProcessRenderer.h"
#include "WebGpuRhiDevice.h"

#include <function/renderer/EngineGlobals.h>
#include <function/renderer/FullscreenRenderer.h>
#include <function/renderer/RenderGraphSchedule.h>
#include <function/renderer/rhi/RhiRenderTexture.h>
#include <function/resources/ShaderAsset/FullscreenProgramAsset.h>

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace infernux::web
{
namespace
{
constexpr auto HdrFormat = rhi::PixelFormat::RGBA16SFloat;
enum class Effect
{
    Prefilter,
    Downsample,
    Upsample,
    Composite,
    ToneMapping,
    Display
};
constexpr const char *ShaderIds[] = {"infernux/post/bloom_prefilter", "infernux/post/bloom_downsample",
                                     "infernux/post/bloom_upsample",  "infernux/post/bloom_composite",
                                     "infernux/post/tonemapping",     "infernux/internal/display_encode"};

rhi::PixelFormat SurfaceFormat(wgpu::TextureFormat format)
{
    switch (format) {
    case wgpu::TextureFormat::RGBA8Unorm:
        return rhi::PixelFormat::RGBA8UNorm;
    case wgpu::TextureFormat::RGBA8UnormSrgb:
        return rhi::PixelFormat::RGBA8Srgb;
    case wgpu::TextureFormat::BGRA8Unorm:
        return rhi::PixelFormat::BGRA8UNorm;
    case wgpu::TextureFormat::BGRA8UnormSrgb:
        return rhi::PixelFormat::BGRA8Srgb;
    default:
        throw std::invalid_argument("Unsupported Web display attachment format");
    }
}

class WebFullscreenHost final : public FullscreenRendererHost
{
  public:
    explicit WebFullscreenHost(WebGpuRhiDevice &device) : m_device(device)
    {
    }
    rhi::Device &GetRhiDevice() noexcept override
    {
        return m_device;
    }
    uint32_t GetFrameCount() const noexcept override
    {
        return 1;
    }
    uint32_t GetCurrentFrame() const noexcept override
    {
        return 0;
    }
    rhi::BindingLayoutHandle GetPerViewLayout() const noexcept override
    {
        return {};
    }
    std::shared_ptr<const FullscreenProgram> AcquireProgram(const std::string &name) override
    {
        return LoadFullscreenProgram(name, ShaderCodeTarget::Wgsl);
    }

  private:
    WebGpuRhiDevice &m_device;
};
} // namespace

struct WebPostProcessRenderer::Impl
{
    using Texture = std::shared_ptr<const rhi::RenderTextureGeneration>;
    using Parameters = std::vector<std::pair<std::string, ShaderParameterValue>>;
    using Inputs = std::vector<std::pair<std::string, Texture>>;
    struct Pass
    {
        Effect effect;
        FullscreenPipelineEntry pipeline;
        FullscreenParameters parameters;
        std::vector<FullscreenResourceInput> inputs;
        std::vector<Texture> sources;
        Texture target;
        WebGpuRasterPass raster;
    };

    explicit Impl(WebGpuRhiDevice &rhi, rhi::PixelFormat format, uint32_t samples)
        : device(rhi), surfaceFormat(format), sceneSamples(samples)
    {
        renderer.Initialize(std::make_shared<WebFullscreenHost>(device));
        rhi::BufferDesc description;
        description.byteSize = sizeof(EngineGlobalsUBO);
        description.usage = rhi::BufferUsageFlags::Uniform;
        description.memory = rhi::BufferMemory::Upload;
        globals = device.CreateBuffer(description);
        if (!globals.IsValid())
            throw std::runtime_error("Web view globals allocation failed");
    }
    ~Impl()
    {
        renderer.Destroy();
        device.Release(globals);
    }

    WebGpuRhiDevice &device;
    rhi::PixelFormat surfaceFormat;
    uint32_t sceneSamples;
    uint32_t width = 0, height = 0;
    Settings settings;
    EngineGlobalsUBO frameGlobals;
    bool diagnosticBloom = true;
    bool reportedReady = false;
    FullscreenRenderer renderer;
    rhi::BufferHandle globals;
    std::unique_ptr<rhi::RenderTexture> scene;
    std::vector<std::unique_ptr<rhi::RenderTexture>> targets;
    std::vector<Pass> passes;
    std::vector<uint32_t> executionOrder;
    Pass display;

    bool BloomEnabled() const noexcept
    {
        return settings.bloomEnabled && diagnosticBloom;
    }

    Texture Target(const std::string &name, uint32_t targetWidth, uint32_t targetHeight)
    {
        rhi::RenderTextureDesc description;
        description.width = targetWidth;
        description.height = targetHeight;
        description.colorFormat = HdrFormat;
        auto target = std::make_unique<rhi::RenderTexture>(device, name, description);
        auto generation = target->Acquire();
        targets.push_back(std::move(target));
        return generation;
    }

    Pass Program(Effect effect, Texture target, const Inputs &inputs, const Parameters &parameters = {})
    {
        const std::string id = ShaderIds[static_cast<size_t>(effect)];
        const auto program = renderer.AcquireProgram(id);
        if (!program)
            throw std::runtime_error("Fullscreen shader asset is unavailable: " + id);
        FullscreenPipelineKey key;
        key.shaderName = id;
        key.program = program;
        key.colorFormat = target ? HdrFormat : surfaceFormat;
        key.useDynamicRendering = true;
        key.depth.testEnabled = false;
        key.depth.writeEnabled = false;
        Pass pass;
        pass.effect = effect;
        pass.pipeline = renderer.EnsurePipeline(key);
        pass.parameters = FullscreenRenderer::PrepareParameters(*program, parameters);
        pass.target = std::move(target);
        for (const auto &name : pass.pipeline.inputNames) {
            const auto found =
                std::find_if(inputs.begin(), inputs.end(), [&](const auto &input) { return input.first == name; });
            if (found == inputs.end())
                throw std::logic_error("Fullscreen input is not connected: " + id + '/' + name);
            const auto &texture = *found->second->color;
            FullscreenResourceInput input;
            input.view = texture.GetView();
            input.format = texture.GetFormat();
            input.sampler = texture.GetSampler();
            pass.inputs.push_back(input);
            pass.sources.push_back(found->second);
        }
        return pass;
    }

    void Rebuild()
    {
        // Output/settings changes replace the attachment plan, while the
        // shared renderer retains programs and graphics pipelines.
        passes.clear();
        display = {};
        targets.clear();
        rhi::RenderTextureDesc description;
        description.width = width;
        description.height = height;
        description.colorFormat = HdrFormat;
        description.samples = static_cast<rhi::SampleCount>(sceneSamples);
        if (scene)
            scene->Reconfigure(description);
        else
            scene = std::make_unique<rhi::RenderTexture>(device, "web/scene", description);
        auto color = scene->Acquire();

        if (BloomEnabled()) {
            std::vector<Texture> levels;
            levels.reserve(settings.bloomIterations);
            for (uint32_t index = 0; index < settings.bloomIterations; ++index) {
                auto level = Target("web/bloom/down/" + std::to_string(index), std::max(1u, width >> (index + 1u)),
                                    std::max(1u, height >> (index + 1u)));
                if (index == 0)
                    passes.push_back(Program(
                        Effect::Prefilter, level, {{"source_tex", color}},
                        {{"threshold", settings.bloomThreshold}, {"knee", 0.5f}, {"clamp_max", settings.bloomClamp}}));
                else
                    passes.push_back(Program(Effect::Downsample, level, {{"source_tex", levels.back()}}));
                levels.push_back(std::move(level));
            }
            auto bloom = levels.back();
            for (size_t index = levels.size() - 1; index > 0; --index) {
                const auto &high = levels[index - 1];
                auto level = Target("web/bloom/up/" + std::to_string(index - 1), high->width, high->height);
                passes.push_back(Program(Effect::Upsample, level, {{"source_tex", bloom}, {"destination_tex", high}},
                                         {{"scatter", settings.bloomScatter}}));
                bloom = std::move(level);
            }
            auto composite = Target("web/bloom/composite", width, height);
            const auto &tint = settings.bloomTint;
            passes.push_back(Program(Effect::Composite, composite, {{"bloom_tex", bloom}, {"scene_color", color}},
                                     {{"intensity", settings.bloomIntensity},
                                      {"tint", std::array<float, 4>{tint[0], tint[1], tint[2], 1.0f}}}));
            color = std::move(composite);
        }
        if (settings.toneMappingMode != 0 || settings.exposure != 1.0f) {
            auto mapped = Target("web/tonemapping", width, height);
            passes.push_back(
                Program(Effect::ToneMapping, mapped, {{"source_tex", color}},
                        {{"mode", static_cast<int32_t>(settings.toneMappingMode)}, {"exposure", settings.exposure}}));
            color = std::move(mapped);
        }
        display = Program(Effect::Display, {}, {{"source_tex", color}},
                          {{"target_is_srgb", rhi::IsSrgbFormat(surfaceFormat)}});
        CompileSchedule();
        // Prepare only after the vector has its final storage: native
        // attachment pointers refer to each pass's own cached arrays.
        for (auto &pass : passes) {
            rhi::RasterPassDescription raster;
            raster.width = pass.target->width;
            raster.height = pass.target->height;
            raster.colorCount = 1;
            auto &color = raster.colors[0];
            color.view = pass.target->color->GetView();
            color.format = pass.target->color->GetFormat();
            color.load = rhi::AttachmentLoad::Clear;
            color.clear = {0, 0, 0, 1};
            device.PrepareRasterPass(raster, pass.raster);
        }
    }

    void CompileSchedule()
    {
        RenderGraphIdentitySource identity;
        std::unordered_map<const rhi::RenderTextureGeneration *, GraphResourceHandle> resources;
        const auto resource = [&](const Texture &texture) {
            return resources
                .try_emplace(texture.get(),
                             GraphResourceHandle{identity.Current(), static_cast<uint32_t>(resources.size()), 1})
                .first->second;
        };
        std::vector<GraphSchedulePass> declarations(passes.size() + 1);
        for (size_t index = 0; index < declarations.size(); ++index) {
            const auto &pass = index == passes.size() ? display : passes[index];
            auto &declaration = declarations[index];
            for (const auto &source : pass.sources)
                declaration.reads.push_back({resource(source)});
            declaration.writes.push_back(resource(pass.target));
        }
        auto schedule = CompileRenderGraphSchedule(declarations, resource(display.target));
        if (!schedule.unresolved.empty())
            throw std::logic_error("Fullscreen graph contains a dependency cycle");
        executionOrder = std::move(schedule.order);
        // The display pass is encoded into the host's acquired surface below.
        executionOrder.pop_back();
    }

    void UpdateParameters()
    {
        // Parameter animation changes only prepared bytes, without rebuilding
        // targets, reparsing schemas or compiling shaders/pipelines.
        for (auto &pass : passes) {
            auto &parameters = pass.parameters;
            auto &values = parameters.values;
            switch (pass.effect) {
            case Effect::Prefilter:
                values[0] = settings.bloomThreshold;
                values[2] = settings.bloomClamp;
                break;
            case Effect::Upsample:
                values[0] = settings.bloomScatter;
                break;
            case Effect::Composite: {
                const auto &tint = settings.bloomTint;
                values[0] = settings.bloomIntensity;
                values[1] = std::array<float, 4>{tint[0], tint[1], tint[2], 1.0f};
                break;
            }
            case Effect::ToneMapping:
                values[0] = static_cast<int32_t>(settings.toneMappingMode);
                values[1] = settings.exposure;
                break;
            case Effect::Downsample:
            case Effect::Display:
                break;
            }
            for (size_t index = 0; index < values.size(); ++index)
                parameters.copies[index].Write(parameters.defaults.data(), values[index].Data());
        }
    }

    void Draw(wgpu::RenderPassEncoder pass, const Pass &prepared)
    {
        WebGpuGraphicsCommandContext context;
        auto commands = device.MakeGraphicsCommandEncoder(context, std::move(pass));
        FullscreenViewResources view;
        view.globals = {0, rhi::BindingType::UniformBuffer, globals, 0, sizeof(EngineGlobalsUBO)};
        renderer.DrawProgram(commands, prepared.pipeline, prepared.inputs.data(), prepared.parameters, nullptr, view);
    }
};

WebPostProcessRenderer::WebPostProcessRenderer() = default;
WebPostProcessRenderer::~WebPostProcessRenderer() = default;

bool WebPostProcessRenderer::Initialize(WebGpuRhiDevice &device, wgpu::TextureFormat surfaceFormat,
                                        uint32_t sceneSampleCount)
{
    m_impl = std::make_unique<Impl>(device, SurfaceFormat(surfaceFormat), sceneSampleCount);
    return true;
}

bool WebPostProcessRenderer::Resize(uint32_t width, uint32_t height)
{
    auto &impl = *m_impl;
    width = std::max(1u, width);
    height = std::max(1u, height);
    if (impl.width == width && impl.height == height)
        return true;
    impl.width = width;
    impl.height = height;
    impl.Rebuild();
    return true;
}

bool WebPostProcessRenderer::ConfigureSampleCount(uint32_t sampleCount)
{
    auto &impl = *m_impl;
    if (impl.sceneSamples == sampleCount)
        return true;
    impl.sceneSamples = sampleCount;
    if (impl.width != 0)
        impl.Rebuild();
    return true;
}

bool WebPostProcessRenderer::Configure(const Settings &settings)
{
    auto &impl = *m_impl;
    Settings normalized = settings;
    normalized.bloomThreshold = std::max(0.0f, settings.bloomThreshold);
    normalized.bloomIntensity = std::max(0.0f, settings.bloomIntensity);
    normalized.bloomScatter = std::clamp(settings.bloomScatter, 0.0f, 1.0f);
    normalized.bloomClamp = std::max(0.0f, settings.bloomClamp);
    normalized.bloomIterations = std::clamp(settings.bloomIterations, 1u, 8u);
    normalized.toneMappingMode = std::min(settings.toneMappingMode, 2u);
    normalized.exposure = std::max(0.0001f, settings.exposure);
    if (impl.settings == normalized)
        return true;
    const bool rebuild = impl.settings.bloomEnabled != normalized.bloomEnabled ||
                         impl.settings.bloomIterations != normalized.bloomIterations ||
                         (impl.settings.toneMappingMode != 0 || impl.settings.exposure != 1.0f) !=
                             (normalized.toneMappingMode != 0 || normalized.exposure != 1.0f);
    impl.settings = normalized;
    if (impl.width == 0)
        return true;
    if (rebuild)
        impl.Rebuild();
    else
        impl.UpdateParameters();
    return true;
}

bool WebPostProcessRenderer::SetBloomEnabledForDiagnostics(bool enabled)
{
    auto &impl = *m_impl;
    if (impl.diagnosticBloom == enabled)
        return true;
    impl.diagnosticBloom = enabled;
    if (impl.width != 0)
        impl.Rebuild();
    return true;
}

void WebPostProcessRenderer::StageFrame(const EngineGlobalsUBO &frame)
{
    m_impl->frameGlobals = frame;
}

wgpu::TextureFormat WebPostProcessRenderer::SceneColorFormat() const noexcept
{
    return wgpu::TextureFormat::RGBA16Float;
}

uint32_t WebPostProcessRenderer::SceneSampleCount() const noexcept
{
    return m_impl->sceneSamples;
}

wgpu::TextureView WebPostProcessRenderer::SceneColorAttachmentView() const noexcept
{
    return m_impl->device.GetNativeTextureView(m_impl->scene->Acquire()->ColorAttachment().GetView());
}

wgpu::TextureView WebPostProcessRenderer::SceneColorView() const noexcept
{
    return m_impl->device.GetNativeTextureView(m_impl->scene->Acquire()->color->GetView());
}

bool WebPostProcessRenderer::Prepare(wgpu::CommandEncoder encoder)
{
    auto &impl = *m_impl;
    // WriteBuffer is queue-ordered after the previous frame's submission.
    // Each draw gets a separate slot until this frame has been submitted.
    // Releasing RHI handles drops references; encoded WebGPU commands retain
    // their allocations, so pool reuse needs no browser-thread GPU wait.
    impl.renderer.ResetPool();
    for (const auto index : impl.executionOrder) {
        const auto &prepared = impl.passes[index];
        auto pass = encoder.BeginRenderPass(&prepared.raster.descriptor);
        impl.Draw(pass, prepared);
        pass.End();
    }
    return true;
}

bool WebPostProcessRenderer::Render(wgpu::RenderPassEncoder pass, bool dithering, bool stopNaNs)
{
    auto &impl = *m_impl;
    auto globals = impl.frameGlobals;
    globals.screenParams = {float(impl.width), float(impl.height), 1.0f / impl.width, 1.0f / impl.height};
    globals.viewOptions = {dithering ? 1u : 0u, stopNaNs ? 1u : 0u, 1u, 0u};
    if (!impl.device.WriteBuffer(impl.globals, 0, &globals, sizeof(globals)))
        throw std::runtime_error("Web display globals upload failed");
    impl.Draw(std::move(pass), impl.display);
    if (!impl.reportedReady) {
        std::printf("INFERNUX_WEB_POST_PROCESS_READY renderer=shared hdr=rgba16float msaa=%u bloom=%d iterations=%u "
                    "tonemap=%u\n",
                    impl.sceneSamples, impl.BloomEnabled() ? 1 : 0, impl.settings.bloomIterations,
                    impl.settings.toneMappingMode);
        impl.reportedReady = true;
    }
    return true;
}
} // namespace infernux::web
