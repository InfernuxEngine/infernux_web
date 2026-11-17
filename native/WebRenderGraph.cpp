#include "WebRenderGraph.h"
#include "WebGpuRhiDevice.h"

#include <function/renderer/RenderGraphSchedule.h>
#include <function/renderer/RenderGraphTemporalHistory.h>
#include <function/renderer/RenderGraphTransfer.h>
#include <function/renderer/RenderGraphWork.h>
#include <function/renderer/raytracing/RayWorldMaterials.h>
#include <function/renderer/rhi/RhiBuffer.h>
#include <function/renderer/rhi/RhiComputeKernel.h>
#include <function/renderer/MaterialPassPipeline.h>
#include <function/renderer/SceneDepthResolver.h>
#include <function/renderer/TextureRegionRenderer.h>
#include <function/resources/ShaderAsset/SlangCompiler.h>
#include <function/renderer/rhi/RenderViewContext.h>
#include <function/renderer/rhi/RhiRenderTexture.h>
#include <function/renderer/rhi/RhiComputeBuffer.h>
#include <function/renderer/rhi/CameraViewportTarget.h>
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
        std::vector<std::pair<uint32_t, GraphFrameBuffer>> frameBufferInputs;
        std::vector<std::pair<uint32_t, raytracing::RayTexturePageClass>> worldRayPageInputs;
        struct HistoryInput
        {
            uint32_t index;
            const GraphTextureDesc *texture;
        };
        std::vector<HistoryInput> historyInputs;
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
        std::vector<std::pair<int, std::string>> historyColors;
        std::string historyResolve;
        Texture particleDepth;
    };

    WebGpuRhiDevice &device;
    rhi::ComputeHost *computeHost;
    FullscreenRenderer fullscreen;
    std::shared_ptr<const RenderGraphDescription> graph;
    std::unordered_map<std::string, Texture> textures;
    std::unordered_map<std::string, Texture> resolved;
    std::unordered_map<std::string, GraphTemporalHistory> histories;
    std::shared_ptr<RenderGraphWorkStorage> workStorage;
    RenderGraphResults results;
    uint64_t historyEpoch = 0;
    bool historyReset = true;
    std::vector<uint32_t> eventPasses;
    std::vector<const GraphBufferDesc *> liveBufferInputs;
    GraphBufferContents bufferContents;
    std::unordered_set<std::string> bufferWrites;
    std::unordered_map<std::string, std::shared_ptr<rhi::BufferResource>> transientBuffers;
    std::vector<Pass> passes;
    std::vector<uint32_t> order;
    std::unordered_map<std::string, WebGpuRasterPass> colorResolves;
    std::vector<std::string> finalResolves;
    std::shared_ptr<const SceneDepthResolveProgram> depthProgram;
    SceneDepthResolver depthResolver;
    rhi::RenderViewContext view;
    std::optional<GraphPassDesc> cameraClear;
    rhi::CameraViewportTarget viewport;
    std::shared_ptr<const rhi::RenderTextureGeneration> cameraTarget;
    std::unordered_map<uint32_t, std::unique_ptr<TextureRegionRenderer>> regionRenderers;
    glm::vec4 cameraRect{0, 0, 1, 1};
    bool sampleParticleDepth = false;
    size_t cameraClearIndex = SIZE_MAX;
    uint32_t width = 0, height = 0, samples = 0;

    Impl(WebGpuRhiDevice &rhi, std::shared_ptr<FullscreenRendererHost> host, rhi::ComputeHost *compute,
         std::shared_ptr<FullscreenPipelineCache> pipelines)
        : device(rhi), computeHost(compute)
    {
        fullscreen.Initialize(std::move(host), std::move(pipelines));
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
            texture = source.IsViewColor() && cameraTarget ? cameraTarget->color
                                                           : Allocate(single, view, source.IsDepth());
        }
        return texture;
    }

    const GraphTextureDesc &TextureDescription(const std::string &name) const
    {
        return *std::find_if(graph->textures.begin(), graph->textures.end(),
                             [&](const auto &item) { return item.name == name; });
    }

    static bool IsHistory(const GraphTextureDesc &texture)
    {
        return texture.role == GraphTextureRole::TemporalRead || texture.role == GraphTextureRole::TemporalWrite;
    }

    void RecordHistoryBindings(wgpu::CommandEncoder encoder)
    {
        if (histories.empty())
            return;
        for (const auto &[key, history] : histories) {
            auto read = history.Read();
            textures.at(history.readName) = read->color;
            textures.at(history.writeName) = history.Write()->color;
            if (history.valid)
                continue;
            rhi::RasterPassDescription init;
            init.width = history.width;
            init.height = history.height;
            init.colorCount = 1;
            init.colors[0].view = read->color->GetView();
            init.colors[0].format = read->color->GetFormat();
            init.colors[0].load = rhi::AttachmentLoad::Clear;
            WebGpuRasterPass native;
            device.PrepareRasterPass(init, native);
            auto pass = encoder.BeginRenderPass(&native.descriptor);
            pass.End();
        }
        for (const auto index : order) {
            auto &pass = passes[index];
            if (pass.historyColors.empty() && pass.historyResolve.empty())
                continue;
            for (const auto &[slot, name] : pass.historyColors)
                pass.raster.colors[slot].view = textures.at(name)->GetView();
            if (!pass.historyResolve.empty())
                pass.raster.colors[0].resolve = textures.at(pass.historyResolve)->GetView();
            device.PrepareRasterPass(pass.raster, pass.native);
            pass.native.descriptor.label = graph->passes[index].name.c_str();
        }
    }

    rhi::BufferHandle GetBuffer(const std::string &name) const
    {
        const auto owner = workStorage->buffers.find(name);
        return owner != workStorage->buffers.end() ? owner->second->GetBuffer()
                                                  : transientBuffers.at(name)->GetBuffer();
    }

    void PrepareBuffers()
    {
        if (std::all_of(graph->buffers.begin(), graph->buffers.end(),
                        [](const auto &buffer) { return buffer.HasFrameBuffer(); })) {
            workStorage.reset();
            transientBuffers.clear();
            bufferWrites.clear();
            return;
        }
        if (!computeHost)
            throw std::logic_error("A graph with buffers requires the renderer's compute host");
        const RenderGraphWorkStorage empty;
        auto next = PrepareRenderGraphWork(*computeHost, *graph, workStorage ? *workStorage : empty);
        if (!next)
            throw std::runtime_error("Could not allocate graph work storage");
        workStorage = std::make_shared<RenderGraphWorkStorage>(std::move(*next));
        workStorage->Publish();
        bufferWrites = FindGraphBufferWrites(*graph);
        transientBuffers.clear();
        for (const auto &buffer : graph->buffers) {
            if (buffer.HasFrameBuffer() || workStorage->buffers.count(buffer.name))
                continue;
            rhi::BufferDesc desc;
            desc.byteSize = buffer.byteSize;
            desc.usage = static_cast<rhi::BufferUsageFlags>(buffer.usage);
            auto handle = device.CreateBuffer(desc);
            if (!handle.IsValid())
                throw std::runtime_error("Could not allocate graph buffer: " + buffer.name);
            transientBuffers.emplace(buffer.name, std::make_shared<rhi::BufferResource>(device, handle, desc.byteSize));
        }
    }

    const GraphPassDesc &EffectivePass(size_t index) const
    {
        const auto &pass = graph->passes[index];
        return cameraClear && cameraClear->name == pass.name ? *cameraClear : pass;
    }

    void CompileSchedule()
    {
        RenderGraphIdentitySource identity;
        std::unordered_map<std::string, uint32_t> resources;
        std::vector<GraphResourceHandle> versions;
        const auto newResource = [&]() {
            const auto id = static_cast<uint32_t>(versions.size());
            versions.push_back({identity.Current(), id, 0});
            return id;
        };
        std::unordered_map<rhi::TextureHandle, uint32_t, rhi::HandleHash<rhi::TextureTag>> imageIds;
        std::unordered_map<rhi::BufferHandle, uint32_t, rhi::HandleHash<rhi::BufferTag>> bufferIds;
        std::unordered_set<uint32_t> exports;
        std::unordered_map<std::string_view, const GraphBufferDesc *> inputs;
        const auto resourceId = [&](auto &ids, auto handle) {
            const auto existing = ids.find(handle);
            if (existing != ids.end())
                return existing->second;
            return ids.emplace(handle, newResource()).first->second;
        };
        for (const auto &texture : graph->textures) {
            const auto id = texture.IsWorldRayPage() ? newResource()
                                                    : resourceId(imageIds, textures.at(texture.name)->GetTexture());
            resources.emplace(texture.name, id);
            if (texture.role == GraphTextureRole::Persistent || texture.role == GraphTextureRole::TemporalWrite ||
                texture.IsViewAttachment())
                exports.insert(id);
        }
        for (const auto &buffer : graph->buffers) {
            const auto id = buffer.HasFrameBuffer() ? newResource() : resourceId(bufferIds, GetBuffer(buffer.name));
            resources.emplace(buffer.name, id);
            if (buffer.exportLayout)
                exports.insert(id);
            const auto &storage = ResolveGraphBufferStorage(*graph, buffer);
            if (storage.input)
                inputs.emplace(buffer.name, &storage);
        }

        std::vector<GraphSchedulePass> declarations(graph->passes.size());
        std::vector<std::vector<const GraphBufferDesc *>> inputReads(graph->passes.size());
        liveBufferInputs.clear();
        const auto addInput = [&](const std::string &name, auto &reads) {
            if (const auto found = inputs.find(name); found != inputs.end())
                reads.push_back(found->second);
        };
        for (const auto &buffer : graph->buffers)
            if (buffer.exportLayout)
                addInput(buffer.name, liveBufferInputs);
        const auto shadow = ResolveGraphShadowTexture(*graph);
        for (size_t index = 0; index < graph->passes.size(); ++index) {
            const auto &source = EffectivePass(index);
            auto &pass = declarations[index];
            pass.sideEffect = source.sideEffect;
            std::unordered_map<uint32_t, bool> writes;
            const auto read = [&](const std::string &name) {
                pass.reads.push_back({versions[resources.at(name)]});
                addInput(name, inputReads[index]);
            };
            const auto write = [&](const std::string &name, bool retain) {
                const auto id = resources.at(name);
                auto [entry, inserted] = writes.try_emplace(id, retain);
                entry->second |= retain;
            };
            for (const auto &name : source.readTextures)
                read(name);
            for (const auto &[slot, name] : source.writeColors)
                write(name, !source.clearColor);
            if (!source.writeDepth.empty())
                write(source.writeDepth, !source.clearDepth);
            if (!source.resolveColor.empty())
                write(source.resolveColor, false);
            for (const auto &access : source.bufferAccesses) {
                if (access.type == GraphBufferAccessType::StorageWrite || access.type == GraphBufferAccessType::TransferWrite)
                    write(access.resource, true);
                else
                    read(access.resource);
            }
            if (const auto *command = source.PrimaryCommand()) {
                if (!shadow.empty() && command->fullscreenProgram && command->fullscreenProgram->UsesViewBinding(0))
                    read(shadow);
                for (const auto &[name, resource] : command->inputBindings)
                    read(resource);
                if (!command->sourceResource.empty())
                    read(command->sourceResource);
                if (!command->destinationResource.empty())
                    write(command->destinationResource, true);
                if (command->type == GraphCommandType::ComputeWork) {
                    const auto &work = *FindGraphWorkBinding(*graph, command->workBinding);
                    const auto names = GraphWorkSlotNames(*graph, work);
                    for (const auto &dispatch : work.recipe->GetDispatches()) {
                        for (const auto &binding : dispatch.bindings) {
                            const auto &name = names[binding.range.slot];
                            const auto bits = static_cast<uint8_t>(binding.access);
                            if (bits & static_cast<uint8_t>(rhi::ComputeBufferAccess::Read))
                                read(name);
                            if (bits & static_cast<uint8_t>(rhi::ComputeBufferAccess::Write))
                                write(name, true);
                        }
                        if (dispatch.indirectArguments)
                            read(names[dispatch.indirectArguments->slot]);
                    }
                }
            }
            for (const auto &[id, retain] : writes) {
                auto &version = versions[id];
                pass.reads.push_back({version, retain});
                ++version.version;
                pass.writes.push_back(version);
                pass.externalWrite |= exports.count(id) != 0;
            }
        }
        auto schedule = CompileRenderGraphSchedule(declarations, versions[resources.at(graph->outputTexture)]);
        if (!schedule.unresolved.empty())
            throw std::invalid_argument("Render graph contains a dependency cycle: " + graph->name);
        order = std::move(schedule.order);
        for (const auto index : order)
            for (const auto *input : inputReads[index])
                if (std::find(liveBufferInputs.begin(), liveBufferInputs.end(), input) == liveBufferInputs.end())
                    liveBufferInputs.push_back(input);
    }

    void CompileRaster(size_t index, const rhi::RenderViewContext &view)
    {
        const auto &source = EffectivePass(index);
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
            if (IsHistory(TextureDescription(name)))
                pass.historyColors.emplace_back(slot, name);
        }
        if (!source.resolveColor.empty()) {
            pass.raster.colors[0].resolve = textures.at(source.resolveColor)->GetView();
            if (IsHistory(TextureDescription(source.resolveColor)))
                pass.historyResolve = source.resolveColor;
        }
        const auto *readOnlyDepth = ResolveGraphReadOnlyDepth(*graph, source);
        const auto &depth = readOnlyDepth ? readOnlyDepth->name : source.writeDepth;
        pass.raster.depth.readOnly = readOnlyDepth != nullptr;
        if (!depth.empty()) {
            const auto &texture = textures.at(depth);
            auto &attachment = pass.raster.depth;
            attachment.view = texture->GetView();
            attachment.format = texture->GetFormat();
            attachment.load = source.clearDepth ? rhi::AttachmentLoad::Clear : rhi::AttachmentLoad::Load;
            attachment.clear = source.clearDepthValue;
        }
        const auto &materialPass = pass.material = ResolveMaterialPass(*graph, source, view);
        const auto *draw = source.PrimaryCommand();
        if (sampleParticleDepth && readOnlyDepth && draw && draw->type == GraphCommandType::DrawRenderers &&
            (draw->shaderTarget == ShaderCompileTarget::Forward || draw->shaderTarget == ShaderCompileTarget::ForwardPlus)) {
            pass.particleDepth = SampledTexture(depth, false, view);
            if (resolved.count(depth))
                pass.resolvedInputs.push_back(depth);
        }
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
                    if (buffer->HasFrameBuffer())
                        prepared.frameBufferInputs.emplace_back(uint32_t(prepared.inputs.size()), buffer->frameBuffer);
                    else {
                        input.buffer = GetBuffer(buffer->name);
                        input.byteOffset = buffer->byteOffset;
                        input.byteSize = buffer->byteSize;
                    }
                    prepared.inputs.push_back(input);
                    continue;
                }
                const auto &textureDesc = TextureDescription(binding->second);
                if (textureDesc.IsWorldRayPage()) {
                    const auto pageClass = textureDesc.role == GraphTextureRole::WorldRayColorPages
                                               ? raytracing::RayTexturePageClass::Color
                                               : raytracing::RayTexturePageClass::Data;
                    prepared.worldRayPageInputs.emplace_back(uint32_t(prepared.inputs.size()), pageClass);
                    prepared.inputs.emplace_back();
                    continue;
                }
                const bool rawSamples = std::any_of(program->code->resources.begin(), program->code->resources.end(),
                    [&](const auto &resource) { return resource.name == name && resource.binding.multisampled; });
                const auto texture = SampledTexture(binding->second, rawSamples, view);
                if (IsHistory(textureDesc))
                    prepared.historyInputs.push_back({static_cast<uint32_t>(prepared.inputs.size()), &textureDesc});
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
        // The shared camera attachment stays linear, independently of the
        // display-encoded graph output. A subsequent camera loads this domain.
        if (cameraTarget)
            for (const auto &texture : graph->textures)
                if (texture.IsViewColor() && resolved.count(texture.name) && dirty[texture.name] &&
                    texture.name != graph->outputTexture)
                    finalResolves.push_back(texture.name);
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

    void CopyRegion(wgpu::CommandEncoder encoder, Texture source, Texture destination,
                      uint32_t sourceX, uint32_t sourceY, uint32_t destinationX, uint32_t destinationY,
                      uint32_t copyWidth, uint32_t copyHeight, rhi::SampleCount count)
    {
        if (source == destination)
            return;
        const bool depth = rhi::IsDepthFormat(destination->GetFormat());
        if (!depth && count == rhi::SampleCount::One && source->GetFormat() == destination->GetFormat()) {
            rhi::TextureCopyRegion region;
            region.sourceX = sourceX; region.sourceY = sourceY;
            region.destinationX = destinationX; region.destinationY = destinationY;
            region.width = copyWidth; region.height = copyHeight;
            WebGpuTransferCommandContext context;
            device.MakeTransferCommandEncoder(context, encoder).CopyTexture(source->GetTexture(), destination->GetTexture(), region);
            return;
        }
        const auto key = (uint32_t(destination->GetFormat()) << 8) | uint32_t(count);
        auto &renderer = regionRenderers[key];
        if (!renderer) {
            const auto program = SlangCompiler::Get().Compile(
                {"inx_texture_region_" + std::to_string(key), "Infernux/TextureRegion.slang",
                 TextureRegionRenderer::ShaderSource(depth, count != rhi::SampleCount::One)}, {}, {ShaderCodeTarget::Wgsl});
            renderer = std::make_unique<TextureRegionRenderer>(device, program.targets[0], destination->GetFormat(), count);
        }
        rhi::RasterPassDescription description;
        description.width = destinationX + copyWidth;
        description.height = destinationY + copyHeight;
        if (depth) {
            description.depth.view = destination->GetView();
            description.depth.format = destination->GetFormat();
        } else {
            description.colorCount = 1;
            description.colors[0].view = destination->GetView();
            description.colors[0].format = destination->GetFormat();
        }
        WebGpuRasterPass native;
        device.PrepareRasterPass(description, native);
        auto pass = encoder.BeginRenderPass(&native.descriptor);
        pass.SetViewport(destinationX, destinationY, copyWidth, copyHeight, 0, 1);
        pass.SetScissorRect(destinationX, destinationY, copyWidth, copyHeight);
        WebGpuGraphicsCommandContext context;
        renderer->Record(device.MakeGraphicsCommandEncoder(context, pass), std::move(source),
                         sourceX, sourceY, destinationX, destinationY);
        pass.End();
    }

    Texture CameraAttachment(const GraphTextureDesc &description) const
    {
        if (!cameraTarget || !description.IsViewAttachment())
            return {};
        const auto *attachment = description.IsViewDepth() ? cameraTarget->depth.get()
                                                          : &cameraTarget->ColorAttachment();
        return attachment ? Texture(cameraTarget, attachment) : Texture{};
    }

    void RestoreViewport(wgpu::CommandEncoder encoder)
    {
        if (!viewport.IsPartial())
            return;
        const auto &destination = *cameraTarget;
        const auto &source = *viewport.Output();
        if (!cameraClear || !cameraClear->clearColor)
            CopyRegion(encoder, Texture(viewport.Output(), &source.ColorAttachment()),
                       Texture(cameraTarget, &destination.ColorAttachment()), viewport.X(), viewport.Y(), 0, 0,
                       width, height, destination.description.samples);
        if ((!cameraClear || !cameraClear->clearDepth) && destination.depth)
            CopyRegion(encoder, source.depth, destination.depth, viewport.X(), viewport.Y(), 0, 0,
                       width, height, destination.description.samples);
    }

    void PublishOutput(wgpu::CommandEncoder encoder, Texture finalColor)
    {
        const auto &output = viewport.Output();
        if (!output)
            return;
        CopyRegion(encoder, graph->outputDisplayEncoded ? cameraTarget->color : finalColor, output->color,
                   0, 0, viewport.X(), viewport.Y(), width, height, rhi::SampleCount::One);
        if (!viewport.IsPartial())
            return;
        if (output->multisampleColor)
            CopyRegion(encoder, cameraTarget->multisampleColor, output->multisampleColor, 0, 0, viewport.X(), viewport.Y(),
                       width, height, output->description.samples);
        if (output->depth)
            CopyRegion(encoder, cameraTarget->depth, output->depth, 0, 0, viewport.X(), viewport.Y(),
                       width, height, output->description.samples);
    }
};

WebRenderGraph::WebRenderGraph(WebGpuRhiDevice &device, std::shared_ptr<FullscreenRendererHost> host,
                               rhi::ComputeHost *computeHost, std::shared_ptr<FullscreenPipelineCache> pipelines)
    : m_impl(std::make_unique<Impl>(device, std::move(host), computeHost, std::move(pipelines)))
{
}
WebRenderGraph::~WebRenderGraph()
{
    m_impl->results.Cancel("render view was destroyed");
}

RenderViewAccess WebRenderGraph::GetResourceAccess(const std::vector<MaterialTextureRead> &materialReads) const
{
    const auto &s = *m_impl;
    RenderViewAccess access;
    access.viewId = s.view.id;
    access.admitted = !AppendPendingGraphBufferResults(access, s.liveBufferInputs);
    const auto addRead = [&](const rhi::TextureResource *resource, bool requireWrite) {
        const auto found = std::find_if(access.reads.begin(), access.reads.end(),
                                        [&](const auto &read) { return read.texture == resource; });
        if (found == access.reads.end())
            access.reads.emplace_back(resource, requireWrite);
        else
            found->requiresProducer |= requireWrite;
    };
    if (const auto &output = s.viewport.Output()) {
        access.writes.push_back(output->color.get());
        if (output->multisampleColor)
            access.writes.push_back(output->multisampleColor.get());
        if (output->depth && std::any_of(s.order.begin(), s.order.end(), [&](uint32_t index) {
                const auto &name = s.graph->passes[index].writeDepth;
                return !name.empty() && s.TextureDescription(name).IsViewDepth();
            }))
            access.writes.push_back(output->depth.get());
    }
    for (const auto &texture : s.graph->textures) {
        if (texture.role != GraphTextureRole::Persistent)
            continue;
        const auto *resource = s.textures.at(texture.name).get();
        const auto writes = std::any_of(s.order.begin(), s.order.end(), [&](uint32_t index) {
            return s.graph->passes[index].WritesTexture(texture.name);
        });
        if (writes) {
            access.writes.push_back(resource);
            continue;
        }
        const auto reads = texture.name == s.graph->outputTexture ||
            std::any_of(s.order.begin(), s.order.end(), [&](uint32_t index) {
                return s.graph->passes[index].ReadsTexture(texture.name);
            });
        if (reads)
            addRead(resource, texture.requireCurrentWrite);
    }
    for (const auto &read : materialReads)
        if (!HasLocalGraphTextureProducer(*s.graph, read))
            addRead(read.generation->color.get(), false);
    return access;
}

bool WebRenderGraph::Apply(std::shared_ptr<const RenderGraphDescription> graph, const rhi::RenderViewContext &view,
                             const ImportTexture &imports, const std::optional<GraphPassDesc> &cameraClear,
                             std::shared_ptr<const rhi::RenderTextureGeneration> output, const glm::vec4 &rect, bool sampleParticleDepth)
{
    auto &s = *m_impl;
    const bool sameAssets = !imports || std::all_of(graph->textures.begin(), graph->textures.end(), [&](const auto &texture) {
        if (texture.role != GraphTextureRole::Asset)
            return true;
        const auto current = s.textures.find(texture.name);
        return current != s.textures.end() && current->second == imports(texture);
    });
    const bool sameClear = s.cameraClear.has_value() == cameraClear.has_value() &&
        (!cameraClear || (s.cameraClear->name == cameraClear->name &&
                         s.cameraClear->clearColor == cameraClear->clearColor &&
                         s.cameraClear->clearDepth == cameraClear->clearDepth));
    const bool protectedWrites = s.workStorage && std::any_of(s.bufferWrites.begin(), s.bufferWrites.end(),
        [&](const auto &name) {
            const auto found = s.workStorage->buffers.find(name);
            return found != s.workStorage->buffers.end() && found->second->GetAllocation()->IsReadProtected();
        });
    if (s.graph == graph && s.width == view.width && s.height == view.height && s.view.samples == view.samples &&
        s.view.colorFormat == view.colorFormat && s.view.depthFormat == view.depthFormat &&
        s.view.revision == view.revision && s.viewport.Output() == output && s.cameraRect == rect && sameClear && sameAssets && s.sampleParticleDepth == sampleParticleDepth && !protectedWrites) {
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
    s.bufferContents.clear();
    if (cameraClear && RequiresGraphBufferProof(*graph)) {
        auto effective = *graph;
        const auto pass = std::find_if(effective.passes.begin(), effective.passes.end(),
                                       [&](const auto &item) { return item.name == cameraClear->name; });
        *pass = *cameraClear;
        if (!ValidateRenderGraphDescription(effective, view, &s.bufferContents))
            throw std::invalid_argument("Invalid render graph publication: " + graph->name);
    } else if (!ValidateRenderGraphDescription(*graph, view, &s.bufferContents)) {
        throw std::invalid_argument("Invalid render graph publication: " + graph->name);
    }
    const bool topologyChanged = !s.graph || (s.graph != graph && !RenderGraphTopologyEquals(*s.graph, *graph));
    const bool resetHistory = s.width != view.width || s.height != view.height || s.viewport.Output() != output ||
                              s.cameraRect != rect || topologyChanged;
    if (topologyChanged || (graph->sourceRevision && graph->sourceRevision != s.graph->sourceRevision))
        s.results.Reset("render graph definition changed");
    s.graph = std::move(graph);
    s.width = view.width;
    s.height = view.height;
    s.samples = uint32_t(view.samples);
    s.view = view;
    s.cameraClear = cameraClear;
    s.cameraRect = rect;
    s.sampleParticleDepth = sampleParticleDepth;
    if (output && FindGraphOutputTextureAlias(*s.graph, *output))
        throw std::invalid_argument("A camera target attachment cannot also be imported as a persistent graph resource");
    s.viewport.Prepare(s.device, "camera/viewport/" + std::to_string(view.id), std::move(output), rect);
    s.cameraTarget = s.viewport.IsPartial() ? s.viewport.Work()->Acquire() : s.viewport.Output();
    for (auto &[key, renderer] : s.regionRenderers)
        renderer->ClearBindings();
    s.cameraClearIndex = SIZE_MAX;
    s.passes.clear();
    s.textures.clear();
    s.resolved.clear();
    s.colorResolves.clear();
    s.finalResolves.clear();
    s.depthResolver.Destroy();
    PrepareGraphTemporalHistory(s.histories, s.device, *s.graph, s.width, s.height);
    if (resetHistory)
        InvalidateHistory();
    for (const auto &description : s.graph->textures) {
        if (description.IsWorldRayPage())
            continue;
        Texture texture = s.CameraAttachment(description);
        if (Impl::IsHistory(description)) {
            const auto &history = s.histories.at(description.temporalKey);
            texture = (description.role == GraphTextureRole::TemporalRead ? history.Read() : history.Write())->color;
        }
        if (!texture && imports)
            texture = imports(description);
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
    s.PrepareBuffers();
    s.CompileSchedule();
    s.eventPasses.clear();
    for (const auto index : s.order)
        if (!s.graph->passes[index].events.empty())
            s.eventPasses.push_back(index);
    s.passes.resize(s.graph->passes.size());
    for (const auto index : s.order)
        if (s.graph->passes[index].type == GraphPassType::Raster)
            s.CompileRaster(index, view);
    (void)s.SampledTexture(s.graph->outputTexture, false, view);
    if (s.cameraTarget)
        for (const auto &texture : s.graph->textures)
            if (texture.IsViewColor())
                (void)s.SampledTexture(texture.name, false, view);
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
                            const rhi::BufferResource *worldRayGeometry,
                            const raytracing::RayWorldMaterialPublication *worldRayMaterials, const RecordRaster &raster)
{
    auto &s = *m_impl;
    std::array<rhi::BufferBinding, static_cast<size_t>(GraphFrameBuffer::Count)> frameBuffers{};
    frameBuffers[static_cast<size_t>(GraphFrameBuffer::ViewLights)] = view.canonicalLights;
    frameBuffers[static_cast<size_t>(GraphFrameBuffer::WorldSurfaces)] = view.worldSurfaces;
    const auto bindBuffer = [&](GraphFrameBuffer source, const rhi::BufferResource &buffer) {
        frameBuffers[static_cast<size_t>(source)] =
            {0, rhi::BindingType::StorageBuffer, buffer.GetBuffer(), 0, buffer.GetByteSize()};
    };
    if (worldRayGeometry)
        bindBuffer(GraphFrameBuffer::WorldRayGeometry, *worldRayGeometry);
    if (worldRayMaterials) {
        bindBuffer(GraphFrameBuffer::WorldRayLitMaterials, *worldRayMaterials->materials);
        bindBuffer(GraphFrameBuffer::WorldRayTextureDescriptors, *worldRayMaterials->descriptors);
    }
    s.fullscreen.ResetPool();
    s.RestoreViewport(encoder);
    s.RecordHistoryBindings(encoder);
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
                    for (const auto &[inputIndex, source] : program.frameBufferInputs) {
                        auto &input = program.inputs[inputIndex];
                        const auto &binding = frameBuffers[static_cast<size_t>(source)];
                        if (!binding.buffer.IsValid())
                            throw std::logic_error("A fullscreen view frame buffer is unavailable");
                        input.buffer = binding.buffer;
                        input.byteOffset = binding.offset;
                        input.byteSize = binding.byteSize;
                    }
                    for (const auto &[inputIndex, pageClass] : program.worldRayPageInputs) {
                        if (!worldRayMaterials)
                            throw std::logic_error("A world ray Lit view requires its acquired texture pages");
                        const auto &texture = worldRayMaterials->pages[static_cast<size_t>(pageClass)];
                        auto &input = program.inputs[inputIndex];
                        input.view = texture->GetView();
                        input.format = texture->GetFormat();
                        input.sampler = texture->GetSampler();
                    }
                    for (const auto &binding : program.historyInputs) {
                        auto &input = program.inputs[binding.index];
                        const auto &texture = s.textures.at(binding.texture->name);
                        input.view = texture->GetView();
                        input.sampler = texture->GetSampler();
                        input.historyValid = binding.texture->role == GraphTextureRole::TemporalRead &&
                                             s.histories.at(binding.texture->temporalKey).valid;
                    }
                    s.fullscreen.DrawProgram(commands, program.pipeline, program.inputs.data(), program.parameters, nullptr, view);
                } else
                    raster(pass, command, prepared.raster, prepared.material, prepared.particleDepth.get());
            }
            pass.End();
        } else if (const auto *command = source.PrimaryCommand()) {
            if (command->type == GraphCommandType::ComputeWork) {
                auto pass = encoder.BeginComputePass();
                WebGpuComputeCommandContext context;
                auto commands = s.device.MakeComputeCommandEncoder(context, pass);
                const auto &instance = *s.workStorage->instances.at(command->workBinding);
                for (const auto &dispatch : instance.GetDispatches())
                    if (!dispatch.RecordOne(commands))
                        throw std::runtime_error("Prepared graph work could not record: " + source.name);
                pass.End();
            } else if (source.type == GraphPassType::Copy) {
                WebGpuTransferCommandContext context;
                auto commands = s.device.MakeTransferCommandEncoder(context, encoder);
                const auto buffer = [&](const std::string &name) -> const GraphBufferDesc & {
                    return *std::find_if(s.graph->buffers.begin(), s.graph->buffers.end(),
                                         [&](const auto &item) { return item.name == name; });
                };
                switch (command->type) {
                case GraphCommandType::CopyBuffer: {
                    const auto &from = buffer(command->sourceResource);
                    const auto &to = buffer(command->destinationResource);
                    commands.CopyBuffer(s.GetBuffer(from.name), s.GetBuffer(to.name),
                                        GraphBufferCopyRegion(*command, from, to));
                    break;
                }
                case GraphCommandType::CopyTexture: {
                    const auto &from = s.textures.at(command->sourceResource);
                    const auto &to = s.textures.at(command->destinationResource);
                    const auto &fromDesc = s.TextureDescription(command->sourceResource);
                    const auto &toDesc = s.TextureDescription(command->destinationResource);
                    const auto region = GraphTextureCopyRegion(fromDesc, toDesc, s.view);
                    s.CopyRegion(encoder, from, to, 0, 0, 0, 0, region.width, region.height,
                                 static_cast<rhi::SampleCount>(fromDesc.EffectiveSamples(s.samples)));
                    break;
                }
                case GraphCommandType::CopyTextureToBuffer:
                case GraphCommandType::CopyBufferToTexture: {
                    const bool toBuffer = command->type == GraphCommandType::CopyTextureToBuffer;
                    const auto &imageName = toBuffer ? command->sourceResource : command->destinationResource;
                    const auto &value = buffer(toBuffer ? command->destinationResource : command->sourceResource);
                    const auto &texture = s.textures.at(imageName);
                    const auto region = GraphTextureBufferCopyRegion(*command, s.TextureDescription(imageName), value, s.view);
                    const auto handle = s.GetBuffer(value.name);
                    const bool copied = toBuffer ? commands.CopyTextureToBuffer(texture->GetTexture(), handle, region)
                                                 : commands.CopyBufferToTexture(handle, texture->GetTexture(), region);
                    if (!copied)
                        throw std::runtime_error("Graph texture-buffer copy failed: " + s.device.LastError());
                    break;
                }
                default:
                    throw std::logic_error("Unexpected command in a graph copy pass");
                }
            }
        }
    }
    s.RecordResolves(encoder, s.finalResolves);
    s.PublishOutput(encoder, Output());
}

std::shared_ptr<void> WebRenderGraph::RetainWorkStorage() const
{
    return m_impl->workStorage;
}

std::shared_ptr<GraphBufferCapture> WebRenderGraph::CaptureBuffer(const std::string &name)
{
    if (!m_impl->graph)
        throw std::invalid_argument("The selected view has not prepared a render graph");
    return m_impl->results.CaptureBuffer(*m_impl->graph, m_impl->view.id, name);
}

std::shared_ptr<GraphFramePublication>
WebRenderGraph::PreparePublication(const GraphEventInterests &interests,
                                  const std::function<std::shared_ptr<GraphEventViewSample>()> &sample)
{
    auto &s = *m_impl;
    if (!s.workStorage && s.eventPasses.empty())
        return {};
    const RenderGraphWorkStorage empty;
    return s.results.Prepare(*s.graph, s.workStorage ? *s.workStorage : empty, s.bufferContents, s.bufferWrites,
                             s.eventPasses, interests, s.view.id, s.historyEpoch, sample);
}

void WebRenderGraph::AcceptHistory()
{
    m_impl->historyReset = false;
    for (auto &[key, history] : m_impl->histories)
        history.Accept();
}

void WebRenderGraph::InvalidateHistory() noexcept
{
    if (!m_impl->historyReset) {
        ++m_impl->historyEpoch;
        m_impl->historyReset = true;
    }
    for (auto &[key, history] : m_impl->histories)
        history.Invalidate();
}

const RenderGraphDescription &WebRenderGraph::Description() const noexcept
{
    return *m_impl->graph;
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
