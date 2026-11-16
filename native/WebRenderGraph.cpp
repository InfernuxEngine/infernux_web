#include "WebRenderGraph.h"
#include "WebGpuRhiDevice.h"

#include <function/renderer/RenderGraphSchedule.h>
#include <function/renderer/MaterialPassPipeline.h>
#include <function/renderer/SceneDepthResolver.h>
#include <function/resources/ShaderAsset/SlangCompiler.h>
#include <function/renderer/rhi/RenderViewContext.h>
#include <function/renderer/rhi/RhiRenderTexture.h>
#include <function/renderer/rhi/RhiComputeBuffer.h>
#include <algorithm>
#include <stdexcept>
#include <unordered_map>

namespace infernux::web
{
struct WebRenderGraph::Impl
{
    struct Fullscreen
    {
        FullscreenPipelineEntry pipeline;
        FullscreenParameters parameters;
        std::vector<FullscreenResourceInput> inputs;
        std::vector<uint32_t> viewLightInputs;
        std::vector<std::string> parameterNames;
        std::string parameterBlock;
        uint64_t parameterRevision = 0;
        bool parametersUploaded = false;
    };
    struct Pass
    {
        rhi::RasterPassDescription raster;
        MaterialPassPipelineDescriptor material;
        WebGpuRasterPass native;
        std::vector<Fullscreen> fullscreen;
        std::vector<std::string> resolvedInputs;
        std::vector<std::string> resolves;
    };

    WebGpuRhiDevice &device;
    FullscreenRenderer fullscreen;
    std::shared_ptr<const RenderGraphDescription> graph;
    std::unordered_map<std::string, Texture> textures;
    std::unordered_map<std::string, Texture> resolved;
    std::vector<Pass> passes;
    std::vector<uint32_t> order;
    std::unordered_map<std::string, WebGpuRasterPass> colorResolves;
    std::vector<std::string> finalResolves;
    std::shared_ptr<const SceneDepthResolveProgram> depthProgram;
    SceneDepthResolver depthResolver;
    rhi::RenderViewContext view;
    std::optional<GraphPassDesc> cameraClear;
    size_t cameraClearIndex = SIZE_MAX;
    uint32_t width = 0, height = 0, samples = 0;

    Impl(WebGpuRhiDevice &rhi, std::shared_ptr<FullscreenRendererHost> host) : device(rhi)
    {
        fullscreen.Initialize(std::move(host));
    }

    Texture Allocate(const GraphTextureDesc &source, const rhi::RenderViewContext &view, bool storage = false)
    {
        rhi::TextureDesc description;
        const auto extent = source.ResolveExtent(width, height);
        description.width = extent.first;
        description.height = extent.second;
        description.format = source.EffectiveFormat(view);
        description.samples = static_cast<rhi::SampleCount>(source.EffectiveSamples(samples));
        description.usage = rhi::TextureUsageFlags::Sampled |
                            (source.IsDepth() ? rhi::TextureUsageFlags::DepthStencilAttachment
                                              : rhi::TextureUsageFlags::ColorAttachment);
        if (storage)
            description.usage = description.usage | rhi::TextureUsageFlags::Storage;
        if (description.samples == rhi::SampleCount::One)
            description.usage = description.usage | rhi::TextureUsageFlags::TransferSource | rhi::TextureUsageFlags::TransferDestination;
        auto image = device.CreateTexture(description);
        if (!image.IsValid())
            throw std::runtime_error("Graph texture allocation failed: " + source.name + ": " + device.LastError());
        rhi::TextureViewDesc imageView;
        imageView.texture = image;
        imageView.format = description.format;
        imageView.aspect = source.IsDepth() ? rhi::TextureAspect::Depth : rhi::TextureAspect::Color;
        rhi::SamplerDesc sampler;
        sampler.addressU = sampler.addressV = sampler.addressW = rhi::AddressMode::ClampToEdge;
        sampler.minFilter = sampler.magFilter = sampler.mipFilter =
            storage || source.IsDepth() || rhi::IsIntegerFormat(description.format) ? rhi::FilterMode::Nearest : rhi::FilterMode::Linear;
        return std::make_shared<rhi::TextureResource>(device, image, device.CreateTextureView(imageView),
            device.CreateSampler(sampler), rhi::GetPixelFormatLayout(description.format).RowBytes(description.width) *
            description.height * uint32_t(description.samples), description.format, imageView);
    }

    Texture SampledTexture(const std::string &name, bool rawSamples, const rhi::RenderViewContext &view)
    {
        const auto &source = *std::find_if(graph->textures.begin(), graph->textures.end(),
            [&](const auto &texture) { return texture.name == name; });
        if (rawSamples || source.EffectiveSamples(samples) == 1)
            return textures.at(name);
        auto &texture = resolved[name];
        if (!texture) {
            if (rhi::IsIntegerFormat(source.EffectiveFormat(view)))
                throw std::invalid_argument("Integer multisample inputs must declare Texture2DMS: " + name);
            auto single = source;
            single.samples = 1;
            if (source.IsDepth()) {
                single.role = GraphTextureRole::Transient;
                single.format = rhi::PixelFormat::R32SFloat;
                if (!depthProgram) {
                    const auto program = SlangCompiler::Get().Compile(
                        {"inx_scene_depth_resolve", "Infernux/SceneDepthResolve.compute.py",
                         std::string(SceneDepthResolver::ShaderSource())}, {}, {ShaderCodeTarget::Wgsl});
                    depthProgram = SceneDepthResolver::CreateProgram(device, program.targets[0]);
                }
                if (!depthResolver.IsValid())
                    depthResolver.Initialize(depthProgram);
            }
            texture = Allocate(single, view, source.IsDepth());
        }
        return texture;
    }

    void CompileSchedule()
    {
        RenderGraphIdentitySource identity;
        std::unordered_map<std::string, GraphResourceHandle> resources;
        for (const auto &texture : graph->textures)
            resources.emplace(texture.name, GraphResourceHandle{identity.Current(), uint32_t(resources.size()), 0});
        for (const auto &buffer : graph->buffers)
            resources.emplace(buffer.name, GraphResourceHandle{identity.Current(), uint32_t(resources.size()), 0});
        std::vector<GraphSchedulePass> declarations(graph->passes.size());
        const auto shadow = ResolveGraphShadowTexture(*graph);
        for (size_t index = 0; index < graph->passes.size(); ++index) {
            const auto &source = cameraClear && cameraClear->name == graph->passes[index].name
                                     ? *cameraClear : graph->passes[index];
            auto &pass = declarations[index];
            pass.sideEffect = source.sideEffect;
            for (const auto &name : source.readTextures)
                pass.reads.push_back({resources.at(name)});
            if (!shadow.empty())
                for (const auto &command : source.commands)
                    if (command.fullscreenProgram && command.fullscreenProgram->UsesViewBinding(0)) {
                        pass.reads.push_back({resources.at(shadow)});
                        break;
                    }
            const auto write = [&](const std::string &name, bool retain) {
                auto &resource = resources.at(name);
                pass.reads.push_back({resource, retain});
                ++resource.version;
                pass.writes.push_back(resource);
                const auto texture = std::find_if(graph->textures.begin(), graph->textures.end(),
                    [&](const auto &item) { return item.name == name; });
                if (texture != graph->textures.end())
                    pass.externalWrite |= texture->role == GraphTextureRole::Persistent ||
                                          texture->role == GraphTextureRole::TemporalWrite;
            };
            for (const auto &[slot, name] : source.writeColors)
                write(name, !source.clearColor);
            if (!source.writeDepth.empty())
                write(source.writeDepth, !source.clearDepth);
            if (!source.resolveColor.empty())
                write(source.resolveColor, false);
            for (const auto &access : source.bufferAccesses) {
                if (access.type == GraphBufferAccessType::StorageWrite ||
                    access.type == GraphBufferAccessType::TransferWrite)
                    write(access.resource, true);
                else
                    pass.reads.push_back({resources.at(access.resource)});
            }
            for (const auto &command : source.commands) {
                if (!command.sourceResource.empty() && resources.count(command.sourceResource))
                    pass.reads.push_back({resources.at(command.sourceResource)});
                if (command.CopiesToTexture())
                    write(command.destinationResource, false);
            }
        }
        auto schedule = CompileRenderGraphSchedule(declarations, resources.at(graph->outputTexture));
        if (!schedule.unresolved.empty())
            throw std::invalid_argument("Render graph contains a dependency cycle: " + graph->name);
        order = std::move(schedule.order);
    }

    void CompileRaster(size_t index, const rhi::RenderViewContext &view)
    {
        const auto &source = cameraClear && cameraClear->name == graph->passes[index].name
                                 ? *cameraClear : graph->passes[index];
        if (cameraClear && cameraClear->name == source.name)
            cameraClearIndex = index;
        auto &pass = passes[index];
        const auto extent = ResolveGraphPassExtent(*graph, source, width, height);
        pass.raster.width = extent.first;
        pass.raster.height = extent.second;
        for (const auto &[slot, name] : source.writeColors) {
            const auto &texture = textures.at(name);
            auto &attachment = pass.raster.colors[slot];
            attachment.view = texture->GetView();
            attachment.format = texture->GetFormat();
            attachment.load = source.clearColor ? rhi::AttachmentLoad::Clear : rhi::AttachmentLoad::Load;
            attachment.clear = {source.clearColorR, source.clearColorG, source.clearColorB, source.clearColorA};
            pass.raster.colorCount = std::max(pass.raster.colorCount, uint32_t(slot + 1));
        }
        if (!source.resolveColor.empty())
            pass.raster.colors[0].resolve = textures.at(source.resolveColor)->GetView();
        std::string depth = source.writeDepth;
        if (depth.empty())
            for (const auto &name : source.readTextures)
                if (rhi::IsDepthFormat(textures.at(name)->GetFormat())) {
                    const auto sampled = std::any_of(source.commands.begin(), source.commands.end(),
                        [&](const auto &command) { return command.BindsResource(name); });
                    if (sampled)
                        continue;
                    depth = name;
                    pass.raster.depth.readOnly = true;
                    break;
                }
        if (!depth.empty()) {
            const auto &texture = textures.at(depth);
            auto &attachment = pass.raster.depth;
            attachment.view = texture->GetView();
            attachment.format = texture->GetFormat();
            attachment.load = source.clearDepth ? rhi::AttachmentLoad::Clear : rhi::AttachmentLoad::Load;
            attachment.clear = source.clearDepthValue;
        }
        const auto &materialPass = pass.material = ResolveMaterialPass(*graph, source, view);
        pass.fullscreen.resize(source.commands.size());
        for (size_t commandIndex = 0; commandIndex < source.commands.size(); ++commandIndex) {
            const auto &command = source.commands[commandIndex];
            if (command.type != GraphCommandType::FullscreenQuad)
                continue;
            auto &prepared = pass.fullscreen[commandIndex];
            auto program = command.fullscreenProgram ? command.fullscreenProgram : fullscreen.AcquireProgram(command.shaderName);
            if (!program)
                throw std::invalid_argument("Fullscreen asset is unavailable: " + command.shaderName);
            FullscreenPipelineKey key;
            key.program = program;
            key.shaderName = command.shaderName;
            key.samples = materialPass.samples;
            key.colorFormat = pass.raster.colorCount ? pass.raster.colors[0].format : rhi::PixelFormat::Undefined;
            key.depthFormat = pass.raster.depth.format;
            key.depth.testEnabled = command.depthTest;
            key.depth.writeEnabled = command.depthWrite;
            key.depth.compare = rhi::ReverseDepthCompare(command.depthCompare);
            key.alphaBlend = command.alphaBlend;
            key.useDynamicRendering = true;
            prepared.pipeline = fullscreen.EnsurePipeline(key);
            prepared.parameters = FullscreenRenderer::PrepareParameters(*program, command.parameters);
            prepared.parameterBlock = command.parameterBlock;
            for (const auto &[name, value] : command.parameters)
                prepared.parameterNames.push_back(name);
            for (const auto &name : prepared.pipeline.inputNames) {
                const auto binding = std::find_if(command.inputBindings.begin(), command.inputBindings.end(),
                    [&](const auto &input) { return input.first == name; });
                if (binding == command.inputBindings.end())
                    throw std::invalid_argument("Unbound graph input: " + source.name + "/" + name);
                const auto buffer = std::find_if(graph->buffers.begin(), graph->buffers.end(),
                    [&](const auto &item) { return item.name == binding->second; });
                if (buffer != graph->buffers.end()) {
                    FullscreenResourceInput input;
                    if (buffer->viewLightList)
                        prepared.viewLightInputs.push_back(uint32_t(prepared.inputs.size()));
                    else {
                        if (!buffer->computeBuffer)
                            throw std::invalid_argument("Graph buffer has no execution storage: " + buffer->name);
                        input.buffer = buffer->computeBuffer->GetBuffer();
                        input.byteOffset = buffer->byteOffset;
                        input.byteSize = buffer->byteSize;
                    }
                    prepared.inputs.push_back(input);
                    continue;
                }
                const bool rawSamples = std::any_of(program->code->resources.begin(), program->code->resources.end(),
                    [&](const auto &resource) { return resource.name == name && resource.binding.multisampled; });
                const auto texture = SampledTexture(binding->second, rawSamples, view);
                if (!rawSamples && resolved.count(binding->second))
                    pass.resolvedInputs.push_back(binding->second);
                FullscreenResourceInput input;
                input.view = texture->GetView();
                input.format = texture->GetFormat();
                input.depthRead = rhi::IsDepthFormat(input.format);
                input.sampler = texture->GetSampler();
                prepared.inputs.push_back(input);
            }
        }
    }

    void CompileResolves()
    {
        std::unordered_map<std::string, bool> dirty;
        for (const auto &entry : resolved) {
            const auto &name = entry.first;
            const auto &target = entry.second;
            dirty[name] = true;
            const auto &source = textures.at(name);
            if (rhi::IsDepthFormat(source->GetFormat()))
                continue;
            const auto description = std::find_if(graph->textures.begin(), graph->textures.end(),
                [&](const auto &item) { return item.name == name; });
            const auto [w, h] = description->ResolveExtent(width, height);
            rhi::RasterPassDescription pass;
            pass.width = w; pass.height = h; pass.colorCount = 1;
            pass.colors[0].view = source->GetView();
            pass.colors[0].format = source->GetFormat();
            pass.colors[0].resolve = target->GetView();
            pass.colors[0].load = rhi::AttachmentLoad::Load;
            device.PrepareRasterPass(pass, colorResolves[name]);
        }
        for (const auto index : order) {
            auto &pass = passes[index];
            for (const auto &name : pass.resolvedInputs)
                if (dirty[name]) {
                    pass.resolves.push_back(name);
                    dirty[name] = false;
                }
            const auto &source = graph->passes[index];
            for (const auto &[slot, name] : source.writeColors)
                dirty[name] = true;
            if (!source.writeDepth.empty())
                dirty[source.writeDepth] = true;
        }
        if (resolved.count(graph->outputTexture) && dirty[graph->outputTexture])
            finalResolves.push_back(graph->outputTexture);
    }

    void RecordResolves(wgpu::CommandEncoder encoder, const std::vector<std::string> &names)
    {
        for (const auto &name : names) {
            const auto color = colorResolves.find(name);
            if (color != colorResolves.end()) {
                auto pass = encoder.BeginRenderPass(&color->second.descriptor);
                pass.End();
            } else {
                const auto description = std::find_if(graph->textures.begin(), graph->textures.end(),
                    [&](const auto &item) { return item.name == name; });
                const auto [w, h] = description->ResolveExtent(width, height);
                auto pass = encoder.BeginComputePass();
                WebGpuComputeCommandContext context;
                depthResolver.Record(device.MakeComputeCommandEncoder(context, pass), textures.at(name)->GetView(),
                                     resolved.at(name)->GetView(), w, h, description->EffectiveSamples(samples));
                pass.End();
            }
        }
    }
};

WebRenderGraph::WebRenderGraph(WebGpuRhiDevice &device, std::shared_ptr<FullscreenRendererHost> host)
    : m_impl(std::make_unique<Impl>(device, std::move(host)))
{
}
WebRenderGraph::~WebRenderGraph() = default;

bool WebRenderGraph::Apply(std::shared_ptr<const RenderGraphDescription> graph, const rhi::RenderViewContext &view,
                             const ImportTexture &imports, const std::optional<GraphPassDesc> &cameraClear)
{
    auto &s = *m_impl;
    const bool sameClear = s.cameraClear.has_value() == cameraClear.has_value() &&
        (!cameraClear || (s.cameraClear->name == cameraClear->name &&
                         s.cameraClear->clearColor == cameraClear->clearColor &&
                         s.cameraClear->clearDepth == cameraClear->clearDepth));
    if (s.graph == graph && s.width == view.width && s.height == view.height && s.view.samples == view.samples &&
        s.view.colorFormat == view.colorFormat && s.view.depthFormat == view.depthFormat &&
        s.view.revision == view.revision && sameClear) {
        if (cameraClear && s.cameraClearIndex != SIZE_MAX) {
            const std::array<float, 4> color{cameraClear->clearColorR, cameraClear->clearColorG,
                                               cameraClear->clearColorB, cameraClear->clearColorA};
            auto &pass = s.passes[s.cameraClearIndex];
            if (pass.raster.colors[0].clear != color || pass.raster.depth.clear != cameraClear->clearDepthValue) {
                for (uint32_t slot = 0; slot < pass.raster.colorCount; ++slot)
                    pass.raster.colors[slot].clear = color;
                pass.raster.depth.clear = cameraClear->clearDepthValue;
                s.device.PrepareRasterPass(pass.raster, pass.native);
                pass.native.descriptor.label = s.graph->passes[s.cameraClearIndex].name.c_str();
            }
        }
        return false;
    }
    if (!ValidateRenderGraphDescription(*graph, view))
        throw std::invalid_argument("Invalid render graph publication: " + graph->name);
    s.graph = std::move(graph);
    s.width = view.width;
    s.height = view.height;
    s.samples = uint32_t(view.samples);
    s.view = view;
    s.cameraClear = cameraClear;
    s.cameraClearIndex = SIZE_MAX;
    s.CompileSchedule();
    s.passes.clear();
    s.textures.clear();
    s.resolved.clear();
    s.colorResolves.clear();
    s.finalResolves.clear();
    s.depthResolver.Destroy();
    for (const auto &description : s.graph->textures) {
        Texture texture = imports ? imports(description) : Texture{};
        if (!texture && description.renderTextureGeneration) {
            const auto *attachment = ResolveGraphTextureAttachment(*description.renderTextureGeneration, description.attachment);
            texture = Texture(description.renderTextureGeneration, attachment);
        }
        if (!texture) {
            if (description.role != GraphTextureRole::Transient && description.role != GraphTextureRole::Camera)
                throw std::invalid_argument("Missing imported graph texture: " + description.name);
            texture = s.Allocate(description, view);
        }
        s.textures.emplace(description.name, std::move(texture));
    }
    s.passes.resize(s.graph->passes.size());
    for (const auto index : s.order)
        if (s.graph->passes[index].type == GraphPassType::Raster)
            s.CompileRaster(index, view);
    (void)s.SampledTexture(s.graph->outputTexture, false, view);
    for (const auto index : s.order) {
        const auto &source = s.graph->passes[index];
        if (source.type != GraphPassType::Raster)
            continue;
        auto &pass = s.passes[index];
        s.device.PrepareRasterPass(pass.raster, pass.native);
        pass.native.descriptor.label = source.name.c_str();
    }
    s.CompileResolves();
    return true;
}

void WebRenderGraph::UpdateParameters(const std::vector<GraphParameterBlockUpdate> &updates)
{
    auto &s = *m_impl;
    for (const auto &update : updates)
        for (auto &pass : s.passes)
            for (auto &command : pass.fullscreen) {
                if (command.parameterBlock != update.id ||
                    (command.parametersUploaded && command.parameterRevision == update.revision))
                    continue;
                for (size_t index = 0; index < command.parameterNames.size(); ++index) {
                    const auto found = std::find_if(update.values.begin(), update.values.end(),
                        [&](const auto &value) { return value.first == command.parameterNames[index]; });
                    if (found != update.values.end()) {
                        const auto value = found->second.ConvertTo(command.parameters.values[index].Kind());
                        command.parameters.copies[index].Write(command.parameters.defaults.data(), value.Data());
                    }
                }
                command.parameterRevision = update.revision;
                command.parametersUploaded = true;
            }
}

void WebRenderGraph::Record(wgpu::CommandEncoder encoder, const FullscreenViewResources &view,
                              const RecordRaster &raster, const RecordWork &work)
{
    auto &s = *m_impl;
    s.fullscreen.ResetPool();
    for (const auto index : s.order) {
        const auto &source = s.graph->passes[index];
        auto &prepared = s.passes[index];
        s.RecordResolves(encoder, prepared.resolves);
        if (source.type == GraphPassType::Raster) {
            auto pass = encoder.BeginRenderPass(&prepared.native.descriptor);
            for (size_t commandIndex = 0; commandIndex < source.commands.size(); ++commandIndex) {
                const auto &command = source.commands[commandIndex];
                if (command.type == GraphCommandType::FullscreenQuad) {
                    WebGpuGraphicsCommandContext context;
                    auto commands = s.device.MakeGraphicsCommandEncoder(context, pass);
                    auto &program = prepared.fullscreen[commandIndex];
                    for (const auto inputIndex : program.viewLightInputs) {
                        auto &input = program.inputs[inputIndex];
                        input.buffer = view.canonicalLights.buffer;
                        input.byteOffset = view.canonicalLights.offset;
                        input.byteSize = view.canonicalLights.byteSize;
                    }
                    s.fullscreen.DrawProgram(commands, program.pipeline, program.inputs.data(), program.parameters, nullptr, view);
                } else
                    raster(pass, command, prepared.raster, prepared.material);
            }
            pass.End();
        } else {
            for (const auto &command : source.commands) {
                if (command.type == GraphCommandType::CopyTexture) {
                    const auto &from = s.textures.at(command.sourceResource);
                    const auto &to = s.textures.at(command.destinationResource);
                    rhi::TextureCopyRegion region;
                    const auto extent = ResolveGraphPassExtent(*s.graph, source, s.width, s.height);
                    region.width = command.copyWidth ? command.copyWidth : extent.first;
                    region.height = command.copyHeight ? command.copyHeight : extent.second;
                    region.depth = 1;
                    WebGpuTransferCommandContext context;
                    auto commands = s.device.MakeTransferCommandEncoder(context, encoder);
                    commands.CopyTexture(from->GetTexture(), to->GetTexture(), region);
                } else if (command.type != GraphCommandType::Present)
                    work(encoder, command);
            }
        }
    }
    s.RecordResolves(encoder, s.finalResolves);
}

WebRenderGraph::Texture WebRenderGraph::GetTexture(const std::string &name) const
{
    return m_impl->textures.at(name);
}
WebRenderGraph::Texture WebRenderGraph::Output() const
{
    const auto found = m_impl->resolved.find(m_impl->graph->outputTexture);
    if (found != m_impl->resolved.end())
        return found->second;
    return GetTexture(m_impl->graph->outputTexture);
}
const std::vector<uint32_t> &WebRenderGraph::ExecutionOrder() const noexcept
{
    return m_impl->order;
}
} // namespace infernux::web
