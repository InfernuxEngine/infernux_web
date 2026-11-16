#include "WebParticleRuntime.h"

#include "WebGpuRhiDevice.h"
#include "WebSceneRenderer.h"

#include <function/renderer/RenderWorld.h>
#include <function/renderer/particle/ParticleGpuRenderer.h>
#include <function/renderer/particle/ParticleGpuRuntime.h>
#include <function/resources/InxMaterial/InxMaterial.h>
#include <function/resources/ShaderAsset/ShaderLoader.h>
#include <function/resources/ShaderAsset/SlangNativeModule.h>
#include <function/resources/ShaderAsset/SlangParticleModule.h>

#include <glm/glm.hpp>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace infernux::web
{
namespace
{

constexpr std::array<const char *, static_cast<size_t>(particle::GpuKernelStage::Count)> kStageNames = {
    "bootstrap",        "init",         "update",    "update_rendering_fused", "contact_prepare", "contact_solve",
    "contact_dispatch", "render_reset", "rendering",
};

template <typename T> bool ReadUnsigned(PyObject *value, T &result)
{
    if (!value || !PyLong_Check(value))
        return false;
    const unsigned long long decoded = PyLong_AsUnsignedLongLong(value);
    if (PyErr_Occurred() || decoded > std::numeric_limits<T>::max()) {
        PyErr_Clear();
        return false;
    }
    result = static_cast<T>(decoded);
    return true;
}

PyObject *Field(PyObject *mapping, const char *name)
{
    return mapping && PyDict_Check(mapping) ? PyDict_GetItemString(mapping, name) : nullptr;
}

bool ReadBool(PyObject *value, bool &result)
{
    if (!value || !PyBool_Check(value))
        return false;
    result = value == Py_True;
    return true;
}

bool ReadFloat(PyObject *value, float &result)
{
    if (!value)
        return false;
    const double decoded = PyFloat_AsDouble(value);
    if (PyErr_Occurred() || !std::isfinite(decoded)) {
        PyErr_Clear();
        return false;
    }
    result = static_cast<float>(decoded);
    return true;
}

bool ReadWordVector(PyObject *value, std::vector<uint32_t> &result)
{
    PyObject *sequence = value ? PySequence_Fast(value, "particle parameter words must be a sequence") : nullptr;
    if (!sequence) {
        PyErr_Clear();
        return false;
    }
    const Py_ssize_t count = PySequence_Fast_GET_SIZE(sequence);
    result.clear();
    result.reserve(static_cast<size_t>(std::max<Py_ssize_t>(0, count)));
    bool valid = true;
    for (Py_ssize_t index = 0; valid && index < count; ++index) {
        uint32_t word = 0;
        valid = ReadUnsigned(PySequence_Fast_GET_ITEM(sequence, index), word);
        if (valid)
            result.push_back(word);
    }
    Py_DECREF(sequence);
    return valid;
}

bool ReadTransforms(PyObject *value, particle::GpuParticleTransforms &result)
{
    Py_buffer view{};
    if (value && PyObject_GetBuffer(value, &view, PyBUF_CONTIG_RO | PyBUF_FORMAT) == 0) {
        const bool valid = view.buf && view.len == static_cast<Py_ssize_t>(sizeof(result)) && view.itemsize == 4 &&
                           (!view.format || std::strcmp(view.format, "f") == 0);
        if (valid)
            std::memcpy(&result, view.buf, sizeof(result));
        PyBuffer_Release(&view);
        return valid;
    }
    if (PyErr_Occurred())
        PyErr_Clear();
    PyObject *sequence = PySequence_Fast(value, "particle transforms must be a sequence");
    if (!sequence) {
        PyErr_Clear();
        return false;
    }
    constexpr Py_ssize_t FloatCount = static_cast<Py_ssize_t>(sizeof(result) / sizeof(float));
    bool valid = PySequence_Fast_GET_SIZE(sequence) == FloatCount;
    auto *destination = reinterpret_cast<float *>(&result);
    for (Py_ssize_t index = 0; valid && index < FloatCount; ++index)
        valid = ReadFloat(PySequence_Fast_GET_ITEM(sequence, index), destination[index]);
    Py_DECREF(sequence);
    return valid;
}

bool EmptySequence(PyObject *value)
{
    return value && PySequence_Check(value) && PySequence_Size(value) == 0;
}

bool ProgramUsesPortableSubset(PyObject *program, std::string &error)
{
    bool collision = false;
    if (!ReadBool(Field(program, "collision_enabled"), collision) || collision) {
        error = "WebGPU particles currently require collision_enabled=false";
        return false;
    }
    PyObject *continuation = Field(program, "continuation");
    if (!continuation || continuation != Py_None) {
        error = "WebGPU particles currently require continuation=null";
        return false;
    }
    PyObject *layout = Field(program, "data_interface_layout");
    if (!layout || !PyDict_Check(layout) || !EmptySequence(Field(layout, "mesh_interfaces")) ||
        !EmptySequence(Field(layout, "texture2d_parameters")) || !EmptySequence(Field(layout, "volume_interfaces"))) {
        error = "WebGPU particles currently require an empty data-interface layout";
        return false;
    }
    PyObject *fusion = Field(program, "update_render_fusion");
    bool eligible = false;
    if (!fusion || !PyDict_Check(fusion) || !ReadBool(Field(fusion, "eligible"), eligible) || !eligible) {
        error = "WebGPU particles currently require fused update/rendering";
        return false;
    }
    PyObject *outputs = Field(program, "outputs");
    PyObject *sequence = outputs ? PySequence_Fast(outputs, "particle outputs must be a sequence") : nullptr;
    if (!sequence) {
        PyErr_Clear();
        error = "WebGPU particle outputs are invalid";
        return false;
    }
    bool valid = PySequence_Fast_GET_SIZE(sequence) > 0;
    for (Py_ssize_t index = 0; valid && index < PySequence_Fast_GET_SIZE(sequence); ++index) {
        PyObject *output = PySequence_Fast_GET_ITEM(sequence, index);
        PyObject *type = Field(output, "output_type");
        valid = type && PyUnicode_Check(type) && std::strcmp(PyUnicode_AsUTF8(type), "sprite") == 0;
    }
    Py_DECREF(sequence);
    if (!valid)
        error = "WebGPU particles currently render sprite outputs only";
    return valid;
}

particle::GpuParticleRendererDesc DecodeSpriteOutput(PyObject *output)
{
    particle::GpuParticleRendererDesc desc;
    auto &semantics = desc.semantics;
    if (!ReadBool(Field(output, "receive_scene_lighting"), semantics.receiveSceneLighting) ||
        !ReadBool(Field(output, "receive_shadows"), semantics.receiveShadows) ||
        !ReadBool(Field(output, "soft_particles"), semantics.softParticles) ||
        !ReadFloat(Field(output, "soft_distance"), semantics.softDistance) ||
        !ReadUnsigned(Field(output, "flipbook_columns"), desc.flipbookColumns) ||
        !ReadUnsigned(Field(output, "flipbook_rows"), desc.flipbookRows) || desc.flipbookColumns == 0 ||
        desc.flipbookRows == 0 || static_cast<uint64_t>(desc.flipbookColumns) * desc.flipbookRows > 65536u)
        throw std::invalid_argument("WebGPU particle Sprite output parameters are invalid");

    // The portable Web simulation exports an unsorted index stream.
    semantics.sortMode = particle::ParticleSortMode::None;
    PyObject *alignmentValue = Field(output, "sprite_alignment");
    const char *alignment =
        alignmentValue && PyUnicode_Check(alignmentValue) ? PyUnicode_AsUTF8(alignmentValue) : nullptr;
    constexpr std::array<std::string_view, 4> names{"camera_plane", "camera_position", "axis", "velocity"};
    const auto match = alignment ? std::find(names.begin(), names.end(), alignment) : names.end();
    if (match == names.end()) {
        PyErr_Clear();
        throw std::invalid_argument("WebGPU particle Sprite alignment is invalid");
    }
    semantics.spriteAlignment = static_cast<particle::ParticleSpriteAlignment>(match - names.begin());
    PyObject *axis = Field(output, "alignment_axis");
    const std::unique_ptr<PyObject, decltype(&Py_DecRef)> alignmentAxis(
        axis ? PySequence_Fast(axis, "Sprite alignment axis requires three values") : nullptr, &Py_DecRef);
    if (!alignmentAxis || PySequence_Fast_GET_SIZE(alignmentAxis.get()) != 3) {
        PyErr_Clear();
        throw std::invalid_argument("WebGPU particle Sprite alignment axis requires three values");
    }
    for (size_t index = 0; index < semantics.alignmentAxis.size(); ++index)
        if (!ReadFloat(PySequence_Fast_GET_ITEM(alignmentAxis.get(), index), semantics.alignmentAxis[index]))
            throw std::invalid_argument("WebGPU particle Sprite alignment axis is invalid");
    if (!semantics.IsValid())
        throw std::invalid_argument("WebGPU particle Sprite output semantics are invalid");

    PyObject *material = Field(Field(output, "material"), "native");
    if (!material || material == Py_None)
        throw std::invalid_argument("WebGPU particle Sprite output requires a native material");
    desc.material = pybind11::cast<std::shared_ptr<InxMaterial>>(pybind11::handle(material));
    if (!desc.material || desc.material->IsDeleted())
        throw std::invalid_argument("WebGPU particle Sprite material is unavailable");
    ShaderLoader::PrepareMaterial(*desc.material);
    desc.surface = desc.material->GetSelectedShaderArtifact();
    if (!desc.surface || desc.surface->domain != ShaderProgramDomain::ParticleSprite)
        throw std::invalid_argument("WebGPU particle Sprite output requires a particle Surface shader");
    return desc;
}

struct FrameRequest
{
    uint32_t spawnCount = 0;
    uint32_t spawnBaseId = 0;
    uint32_t spawnGeneration = 0;
    uint32_t systemSeed = 0;
    uint32_t simulationStep = 0;
    float deltaTime = 0.0f;
    bool simulate = true;
    bool render = true;
};

struct alignas(16) WebSpawnMetadata
{
    uint32_t words[8]{};
};
static_assert(sizeof(WebSpawnMetadata) == 32);
// Each emitter binds a storage-buffer view at WebGPU's 256-byte offset alignment.
constexpr uint64_t kSpawnMetadataStride = 256;

bool DecodeRequest(PyObject *mapping, FrameRequest &request, bool preroll)
{
    if (!mapping || !PyDict_Check(mapping) || !ReadUnsigned(Field(mapping, "spawn_count"), request.spawnCount) ||
        !ReadUnsigned(Field(mapping, "spawn_base_id"), request.spawnBaseId) ||
        !ReadUnsigned(Field(mapping, "spawn_generation"), request.spawnGeneration) ||
        !ReadUnsigned(Field(mapping, "system_seed"), request.systemSeed) ||
        !ReadUnsigned(Field(mapping, "simulation_step"), request.simulationStep) ||
        !ReadFloat(Field(mapping, "delta_time"), request.deltaTime))
        return false;
    if (preroll) {
        request.simulate = true;
        request.render = false;
        return true;
    }
    return ReadBool(Field(mapping, "simulate"), request.simulate) && ReadBool(Field(mapping, "render"), request.render);
}

rhi::BufferHandle CreateStorageBuffer(WebGpuRhiDevice &device, uint64_t bytes, bool indirect = false,
                                      const void *initialData = nullptr)
{
    rhi::BufferDesc desc;
    desc.byteSize = std::max<uint64_t>(bytes, 16);
    desc.usage = rhi::BufferUsageFlags::Storage;
    if (indirect)
        desc.usage = desc.usage | rhi::BufferUsageFlags::Indirect;
    desc.memory = rhi::BufferMemory::Upload;
    desc.queueAccess = rhi::QueueAccessFlags::Graphics | rhi::QueueAccessFlags::Compute;
    desc.initialData = initialData;
    desc.initialDataBytes = initialData ? bytes : 0;
    return device.CreateBuffer(desc);
}

} // namespace

struct WebParticleRuntime::State
{
    struct Graph
    {
        explicit Graph(WebGpuRhiDevice &owner) : device(owner)
        {
        }
        ~Graph()
        {
            device.Release(playingStates);
            device.Release(playingRequests);
            device.Release(parameters);
            device.Release(spawnMetadata);
            device.Release(burstRequests);
        }

        WebGpuRhiDevice &device;
        std::vector<uint32_t> parameterWords;
        rhi::BufferHandle burstRequests;
        rhi::BufferHandle spawnMetadata;
        rhi::BufferHandle parameters;
        rhi::BufferHandle playingRequests;
        rhi::BufferHandle playingStates;
    };

    struct Emitter
    {
        [[nodiscard]] bool Visible(uint32_t cameraMask) const noexcept
        {
            return drawReady && playing && (ownerLayerMask & cameraMask) != 0;
        }

        uint64_t id = 0;
        uint64_t graphId = 0;
        uint64_t artifactRevision = 0;
        bool stateWasPreserved = false;
        uint32_t slot = 0;
        uint32_t ownerLayerMask = 0;
        bool playing = true;
        bool resetPending = false;
        bool drawReady = false;
        particle::GpuParticleTransforms transforms{};
        std::vector<FrameRequest> pending;
        std::unique_ptr<particle::ParticleGpuRuntime> runtime;
        std::vector<std::unique_ptr<particle::ParticleGpuRenderer>> outputs;
    };

    WebGpuRhiDevice *rhi = nullptr;
    WebSceneRenderer *scene = nullptr;
    std::shared_ptr<const ShaderCompilation> spriteGeometry;
    std::unordered_map<uint64_t, std::unique_ptr<Graph>> graphs;
    std::unordered_map<uint64_t, std::unique_ptr<Emitter>> emitters;
    std::unordered_set<uint64_t> capacityLimitedGraphs;
    std::unordered_map<uint64_t, uint64_t> capacityLimitedEmitterGraphs;
    std::unordered_map<uint64_t, uint64_t> capacityLimitedEmitterRevisions;
    std::string error;
    bool emissionSupported = true;
    bool reportedDraw = false;

    [[nodiscard]] uint64_t EmitterGraph(uint64_t id) const noexcept
    {
        if (const auto found = emitters.find(id); found != emitters.end())
            return found->second->graphId;
        const auto found = capacityLimitedEmitterGraphs.find(id);
        return found == capacityLimitedEmitterGraphs.end() ? 0 : found->second;
    }

    [[nodiscard]] std::string PrepareGraph(uint64_t graphId, PyObject *programs, PyObject *removeIds, State &staged,
                                           std::unordered_set<uint64_t> &batchEmitterIds);

    void RemoveGraph(uint64_t id) noexcept
    {
        for (auto it = emitters.begin(); it != emitters.end();)
            it = it->second->graphId == id ? emitters.erase(it) : std::next(it);
        graphs.erase(id);
        capacityLimitedGraphs.erase(id);
        for (auto it = capacityLimitedEmitterGraphs.begin(); it != capacityLimitedEmitterGraphs.end();) {
            if (it->second == id) {
                capacityLimitedEmitterRevisions.erase(it->first);
                it = capacityLimitedEmitterGraphs.erase(it);
            } else {
                ++it;
            }
        }
    }

    void Clear() noexcept
    {
        emitters.clear();
        graphs.clear();
        capacityLimitedGraphs.clear();
        capacityLimitedEmitterGraphs.clear();
        capacityLimitedEmitterRevisions.clear();
        spriteGeometry.reset();
        rhi = nullptr;
        scene = nullptr;
        emissionSupported = true;
        reportedDraw = false;
    }
};

WebParticleRuntime::WebParticleRuntime() : m_state(std::make_unique<State>())
{
}

WebParticleRuntime::~WebParticleRuntime()
{
    Shutdown();
}

bool WebParticleRuntime::Initialize(WebGpuRhiDevice &device, WebSceneRenderer &scene)
{
    Shutdown();
    m_state->error.clear();
    if (!device.NativeDevice() || !device.NativeQueue()) {
        m_state->error = "Web particle runtime requires a ready WebGPU device";
        return false;
    }
    m_state->rhi = &device;
    m_state->scene = &scene;
    constexpr uint32_t kRequiredParticleStorageBuffersPerStage = 12;
    const uint32_t availableStorageBuffers = device.GetCapabilities().limits.maxStorageBuffersPerStage;
    if (availableStorageBuffers < kRequiredParticleStorageBuffersPerStage) {
        // Keep the controls resident with zero emission capacity on smaller devices.
        m_state->emissionSupported = false;
        std::printf("INFERNUX_WEBGPU_PARTICLE_CAPACITY_LIMIT available=%u required=%u emission=paused\n",
                    availableStorageBuffers, kRequiredParticleStorageBuffersPerStage);
    } else {
        std::printf("INFERNUX_WEBGPU_PARTICLE_RUNTIME_READY\n");
    }
    return true;
}

void WebParticleRuntime::Shutdown() noexcept
{
    if (m_state)
        m_state->Clear();
}

std::string WebParticleRuntime::ReplaceGraph(uint64_t graphInstanceId, PyObject *programs, PyObject *removeIds)
{
    if (!programs || !removeIds)
        return "WebGPU particle graph publication contains an invalid sequence";
    const std::unique_ptr<PyObject, decltype(&Py_DecRef)> graphs(
        Py_BuildValue("[{s:K,s:O,s:O}]", "graph_instance_id", static_cast<unsigned long long>(graphInstanceId),
                      "programs", programs, "remove_ids", removeIds),
        &Py_DecRef);
    if (!graphs) {
        PyErr_Clear();
        return "WebGPU particle graph publication request could not be prepared";
    }
    return ReplaceGraphs(graphs.get());
}

std::string WebParticleRuntime::ReplaceGraphs(PyObject *graphs)
{
    if (!m_state->rhi)
        return "WebGPU particle runtime is not initialized";
    const std::unique_ptr<PyObject, decltype(&Py_DecRef)> sequence(
        graphs ? PySequence_Fast(graphs, "particle graphs must be a sequence") : nullptr, &Py_DecRef);
    if (!sequence || PySequence_Fast_GET_SIZE(sequence.get()) == 0) {
        PyErr_Clear();
        return "WebGPU particle graph publication batch requires graph requests";
    }

    State staged;
    std::unordered_set<uint64_t> graphIds;
    std::unordered_set<uint64_t> emitterIds;
    try {
        for (Py_ssize_t index = 0; index < PySequence_Fast_GET_SIZE(sequence.get()); ++index) {
            PyObject *graph = PySequence_Fast_GET_ITEM(sequence.get(), index);
            uint64_t graphId = 0;
            if (!ReadUnsigned(Field(graph, "graph_instance_id"), graphId) || graphId == 0 ||
                !graphIds.insert(graphId).second)
                return "WebGPU particle publication batch contains an invalid or duplicate graph id";
            if (auto error = m_state->PrepareGraph(graphId, Field(graph, "programs"), Field(graph, "remove_ids"),
                                                   staged, emitterIds);
                !error.empty())
                return error;
        }
        m_state->graphs.reserve(m_state->graphs.size() + staged.graphs.size());
        m_state->emitters.reserve(m_state->emitters.size() + staged.emitters.size());
        m_state->capacityLimitedGraphs.reserve(m_state->capacityLimitedGraphs.size() +
                                               staged.capacityLimitedGraphs.size());
        m_state->capacityLimitedEmitterGraphs.reserve(m_state->capacityLimitedEmitterGraphs.size() +
                                                      staged.capacityLimitedEmitterGraphs.size());
        m_state->capacityLimitedEmitterRevisions.reserve(m_state->capacityLimitedEmitterRevisions.size() +
                                                         staged.capacityLimitedEmitterRevisions.size());
    } catch (const std::exception &error) {
        PyErr_Clear();
        return error.what();
    }

    // Every candidate owns its buffers and surfaces before the accepted graphs retire.
    for (const auto graphId : graphIds)
        m_state->RemoveGraph(graphId);
    const size_t publishedEmitters = staged.emitters.size();
    m_state->graphs.merge(staged.graphs);
    m_state->emitters.merge(staged.emitters);
    m_state->capacityLimitedGraphs.merge(staged.capacityLimitedGraphs);
    m_state->capacityLimitedEmitterGraphs.merge(staged.capacityLimitedEmitterGraphs);
    m_state->capacityLimitedEmitterRevisions.merge(staged.capacityLimitedEmitterRevisions);
    if (publishedEmitters != 0)
        std::printf("INFERNUX_WEBGPU_PARTICLE_OUTPUT_READY geometry=slang emitters=%zu\n", publishedEmitters);
    return {};
}

std::string WebParticleRuntime::State::PrepareGraph(uint64_t graphInstanceId, PyObject *programs, PyObject *removeIds,
                                                    State &staged, std::unordered_set<uint64_t> &batchEmitterIds)
{
    const std::unique_ptr<PyObject, decltype(&Py_DecRef)> programOwner(
        programs ? PySequence_Fast(programs, "particle programs must be a sequence") : nullptr, &Py_DecRef);
    const std::unique_ptr<PyObject, decltype(&Py_DecRef)> removeOwner(
        removeIds ? PySequence_Fast(removeIds, "particle remove_ids must be a sequence") : nullptr, &Py_DecRef);
    if (!programOwner || !removeOwner) {
        PyErr_Clear();
        return "WebGPU particle graph publication contains an invalid sequence";
    }

    for (Py_ssize_t index = 0; index < PySequence_Fast_GET_SIZE(removeOwner.get()); ++index) {
        uint64_t id = 0;
        if (!ReadUnsigned(PySequence_Fast_GET_ITEM(removeOwner.get(), index), id) || id == 0 ||
            !batchEmitterIds.insert(id).second)
            return "WebGPU particle remove_ids contains an invalid or duplicate emitter id";
        if (const auto owner = EmitterGraph(id); owner != 0 && owner != graphInstanceId)
            return "WebGPU particle graph cannot remove another graph's emitter";
    }
    const Py_ssize_t programCount = PySequence_Fast_GET_SIZE(programOwner.get());
    if (programCount == 0)
        return PySequence_Fast_GET_SIZE(removeOwner.get()) == 0 ? "WebGPU particle graph update cannot be empty" : "";

    auto &device = *rhi;
    std::unique_ptr<Graph> graph;
    std::unordered_map<uint64_t, std::unique_ptr<Emitter>> candidates;
    std::unordered_set<uint32_t> slots;
    uint32_t slotCount = 0;
    std::vector<uint32_t> parameters;
    for (Py_ssize_t index = 0; index < programCount; ++index) {
        PyObject *program = PySequence_Fast_GET_ITEM(programOwner.get(), index);
        std::string error;
        uint64_t encodedGraph = 0;
        auto emitter = std::make_unique<Emitter>();
        emitter->graphId = graphInstanceId;
        if (!ProgramUsesPortableSubset(program, error) || !ReadUnsigned(Field(program, "id"), emitter->id) ||
            emitter->id == 0 || !ReadUnsigned(Field(program, "graph_instance_id"), encodedGraph) ||
            encodedGraph != graphInstanceId || !ReadUnsigned(Field(program, "graph_emitter_index"), emitter->slot) ||
            emitter->slot >= 1024 || !slots.insert(emitter->slot).second ||
            !ReadUnsigned(Field(program, "artifact_revision"), emitter->artifactRevision) ||
            emitter->artifactRevision == 0)
            return error.empty() ? "WebGPU particle program identity is invalid" : error;
        if (!batchEmitterIds.insert(emitter->id).second)
            return "WebGPU particle publication batch contains duplicate or conflicting emitter ids";
        if (const auto owner = EmitterGraph(emitter->id); owner != 0 && owner != graphInstanceId)
            return "WebGPU particle emitter id already belongs to another graph";
        std::vector<uint32_t> words;
        if (!ReadWordVector(Field(program, "parameter_words"), words) || words.size() % 4 != 0)
            return "WebGPU particle parameter block is invalid";
        if (index == 0)
            parameters = std::move(words);
        else if (parameters != words)
            return "WebGPU particle emitters in one graph disagree on parameters";
        slotCount = std::max(slotCount, emitter->slot + 1);
        const auto id = emitter->id;
        candidates.emplace(id, std::move(emitter));
    }
    if (!emissionSupported) {
        for (const auto &[id, emitter] : candidates) {
            staged.capacityLimitedEmitterGraphs.emplace(id, graphInstanceId);
            staged.capacityLimitedEmitterRevisions.emplace(id, emitter->artifactRevision);
        }
        staged.capacityLimitedGraphs.insert(graphInstanceId);
        return {};
    }

    graph = std::make_unique<Graph>(device);
    if (parameters.empty())
        parameters.assign(4, 0);
    graph->parameterWords = parameters;
    const std::vector<uint32_t> zeros(slotCount, 0);
    const std::vector<uint32_t> playing(slotCount, 1);
    graph->burstRequests = CreateStorageBuffer(device, slotCount * sizeof(uint32_t), false, zeros.data());
    graph->spawnMetadata = CreateStorageBuffer(device, slotCount * kSpawnMetadataStride, true);
    graph->parameters = CreateStorageBuffer(device, parameters.size() * sizeof(uint32_t), false, parameters.data());
    graph->playingRequests = CreateStorageBuffer(device, slotCount * sizeof(uint32_t), false, zeros.data());
    graph->playingStates = CreateStorageBuffer(device, slotCount * sizeof(uint32_t), false, playing.data());
    if (!graph->burstRequests.IsValid() || !graph->spawnMetadata.IsValid() || !graph->parameters.IsValid() ||
        !graph->playingRequests.IsValid() || !graph->playingStates.IsValid())
        return "WebGPU particle graph buffers could not be created";

    try {
        if (!spriteGeometry) {
            const auto &geometry = GetSlangParticleGraphicsModules().front().module;
            spriteGeometry = std::make_shared<const ShaderCompilation>(
                SlangCompiler::Get().Compile(geometry, {}, {ShaderCodeTarget::Wgsl}));
        }
        for (Py_ssize_t index = 0; index < programCount; ++index) {
            PyObject *program = PySequence_Fast_GET_ITEM(programOwner.get(), index);
            uint64_t id = 0;
            ReadUnsigned(Field(program, "id"), id);
            auto &emitter = *candidates.at(id);
            particle::GpuEmitterDesc desc;
            if (!ReadUnsigned(Field(program, "owner_layer_mask"), emitter.ownerLayerMask) ||
                !ReadUnsigned(Field(program, "capacity"), desc.capacity) || desc.capacity == 0 ||
                desc.capacity > (1u << 24u) || !ReadUnsigned(Field(program, "state_stride"), desc.stateStride) ||
                desc.stateStride == 0 || desc.stateStride % 16 != 0 ||
                !ReadUnsigned(Field(program, "event_type_count"), desc.eventTypeCount))
                return "WebGPU particle program layout is invalid";
            PyObject *hash = Field(program, "kernel_hash");
            const char *kernelHash = hash && PyUnicode_Check(hash) ? PyUnicode_AsUTF8(hash) : nullptr;
            if (!kernelHash || std::strlen(kernelHash) != 64) {
                PyErr_Clear();
                return "WebGPU particle kernel hash is invalid";
            }

            // Web simulation records the update and render export separately.
            desc.supportsFusedUpdateRendering = false;
            PyObject *sources = Field(program, "kernel_sources");
            std::array<ShaderTargetCode, kStageNames.size()> compiled;
            for (size_t stage = 0; stage < kStageNames.size(); ++stage) {
                if (static_cast<particle::GpuKernelStage>(stage) == particle::GpuKernelStage::UpdateRenderingFused)
                    continue;
                PyObject *value = Field(sources, kStageNames[stage]);
                const char *text = value && PyUnicode_Check(value) ? PyUnicode_AsUTF8(value) : nullptr;
                if (!text) {
                    PyErr_Clear();
                    return std::string("WebGPU particle stage source is missing: ") + kStageNames[stage];
                }
                const std::string sourcePath = std::string("Particle/") + kStageNames[stage] + ".slang";
                compiled[stage] = CompileSlangNativeModule(
                    {"particle_" + std::to_string(index) + "_" + kStageNames[stage], sourcePath, text},
                    ShaderCodeTarget::Wgsl);
            }
            desc.reflectedKernels = &compiled;
            desc.reflectedSpawnBindings = {
                {0, rhi::BindingType::StorageBuffer, graph->burstRequests},
                {1, rhi::BindingType::StorageBuffer, graph->spawnMetadata,
                 static_cast<uint64_t>(emitter.slot) * kSpawnMetadataStride, sizeof(WebSpawnMetadata)},
                {2, rhi::BindingType::StorageBuffer, graph->parameters},
                {3, rhi::BindingType::StorageBuffer, graph->playingRequests},
                {4, rhi::BindingType::StorageBuffer, graph->playingStates},
            };
            emitter.runtime = std::make_unique<particle::ParticleGpuRuntime>();
            if (!emitter.runtime->Create(device, desc))
                return "WebGPU particle compute pipelines could not be created: " + device.LastError();
            particle::GpuParticleSimulationControl control{};
            control.anyViewVisible = 1;
            control.simulationAllowed = 1;
            if (!device.WriteBuffer(emitter.runtime->SimulationControlBuffer(), 0, &control, sizeof(control)))
                return "WebGPU particle simulation control could not be initialized";
            const std::unique_ptr<PyObject, decltype(&Py_DecRef)> outputs(
                PySequence_Fast(Field(program, "outputs"), "particle outputs must be a sequence"), &Py_DecRef);
            if (!outputs) {
                PyErr_Clear();
                return "WebGPU particle outputs are invalid";
            }
            for (Py_ssize_t outputIndex = 0; outputIndex < PySequence_Fast_GET_SIZE(outputs.get()); ++outputIndex) {
                auto output = DecodeSpriteOutput(PySequence_Fast_GET_ITEM(outputs.get(), outputIndex));
                output.geometry = spriteGeometry;
                output.instances = emitter.runtime->InstanceBuffer();
                output.renderIndices = emitter.runtime->RenderIndexBuffer();
                output.textureResolver =
                    [scene = this->scene, material = output.material,
                     textureViews = std::unordered_map<std::string, std::weak_ptr<const rhi::TextureGpuView>>{}](
                        const std::string &guid, const std::string &binding,
                        particle::GpuParticleTextureRequest) mutable -> particle::GpuBillboardTextureLease {
                    auto texture = scene->ResolveSampledTexture(guid, material->GetTextureSampler(binding));
                    if (!texture)
                        return {};
                    const auto handle = texture->GetView();
                    const auto sampler = texture->GetSampler();
                    auto &previous = textureViews[binding];
                    auto view = previous.lock();
                    if (!view || view->GetView() != handle || view->GetSampler() != sampler) {
                        const uint64_t revision = (static_cast<uint64_t>(handle.generation) << 32u) | handle.index;
                        view = std::make_shared<const rhi::TextureGpuView>(
                            guid.empty() ? "white" : guid, revision, texture->GetTexture(), handle, sampler,
                            texture->GetResidentBytes(), texture, texture->GetFormat(), texture->GetViewDesc(),
                            texture->GetUploadSubmission());
                        previous = view;
                    }
                    return {particle::GpuBillboardTextureStatus::Ready, handle, sampler, {}, std::move(view)};
                };
                auto renderer = std::make_unique<particle::ParticleGpuRenderer>();
                if (!renderer->Create(device, output))
                    return "WebGPU particle Slang surface bindings could not be created: " + device.LastError();
                emitter.outputs.push_back(std::move(renderer));
            }
        }
    } catch (const std::exception &error) {
        PyErr_Clear();
        return error.what();
    }
    staged.graphs.emplace(graphInstanceId, std::move(graph));
    staged.emitters.merge(candidates);
    return {};
}

std::string WebParticleRuntime::UpdateParameters(uint64_t graphInstanceId, PyObject *updates)
{
    if (!m_state->emissionSupported)
        return m_state->capacityLimitedGraphs.find(graphInstanceId) != m_state->capacityLimitedGraphs.end()
                   ? std::string{}
                   : "WebGPU particle graph is not resident";
    const auto found = m_state->graphs.find(graphInstanceId);
    if (found == m_state->graphs.end())
        return "WebGPU particle graph is not resident";
    const std::unique_ptr<PyObject, decltype(&Py_DecRef)> sequence(
        updates ? PySequence_Fast(updates, "particle parameter updates must be a sequence") : nullptr, &Py_DecRef);
    if (!sequence || PySequence_Fast_GET_SIZE(sequence.get()) == 0) {
        PyErr_Clear();
        return "WebGPU particle parameter update is invalid";
    }
    auto decoded = found->second->parameterWords;
    for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(sequence.get()); ++i) {
        const std::unique_ptr<PyObject, decltype(&Py_DecRef)> update(
            PySequence_Fast(PySequence_Fast_GET_ITEM(sequence.get(), i),
                            "particle parameter update needs offset and words"),
            &Py_DecRef);
        uint32_t offset = 0;
        std::vector<uint32_t> words;
        if (!update || PySequence_Fast_GET_SIZE(update.get()) != 2 ||
            !ReadUnsigned(PySequence_Fast_GET_ITEM(update.get(), 0), offset) ||
            !ReadWordVector(PySequence_Fast_GET_ITEM(update.get(), 1), words) || offset > decoded.size() ||
            words.size() > decoded.size() - offset) {
            PyErr_Clear();
            return "WebGPU particle parameter update does not match the graph's parameter layout";
        }
        std::copy(words.begin(), words.end(), decoded.begin() + offset);
    }
    if (!m_state->rhi->WriteBuffer(found->second->parameters, 0, decoded.data(), decoded.size() * sizeof(uint32_t)))
        return "WebGPU particle parameter upload failed";
    found->second->parameterWords = std::move(decoded);
    return {};
}

bool WebParticleRuntime::BeginBatch(uint64_t graphInstanceId, PyObject *items)
{
    if (!m_state->emissionSupported)
        return m_state->capacityLimitedGraphs.find(graphInstanceId) != m_state->capacityLimitedGraphs.end();
    if (!m_state->rhi || m_state->graphs.find(graphInstanceId) == m_state->graphs.end())
        return false;
    PyObject *sequence = PySequence_Fast(items, "particle frame items must be a sequence");
    if (!sequence) {
        PyErr_Clear();
        return false;
    }
    bool accepted = true;
    for (Py_ssize_t index = 0; accepted && index < PySequence_Fast_GET_SIZE(sequence); ++index) {
        PyObject *item = PySequence_Fast_GET_ITEM(sequence, index);
        uint64_t emitterId = 0;
        accepted = ReadUnsigned(Field(item, "emitter_id"), emitterId);
        auto found = m_state->emitters.find(emitterId);
        accepted = accepted && found != m_state->emitters.end() && found->second->graphId == graphInstanceId &&
                   ReadUnsigned(Field(item, "owner_layer_mask"), found->second->ownerLayerMask) &&
                   ReadTransforms(Field(item, "transforms"), found->second->transforms);
        if (!accepted)
            break;
        found->second->pending.clear();
        PyObject *preroll = PySequence_Fast(Field(item, "preroll_steps"), "particle preroll must be a sequence");
        if (!preroll || PySequence_Fast_GET_SIZE(preroll) > 4096) {
            PyErr_Clear();
            Py_XDECREF(preroll);
            accepted = false;
            break;
        }
        for (Py_ssize_t step = 0; accepted && step < PySequence_Fast_GET_SIZE(preroll); ++step) {
            FrameRequest request;
            accepted = DecodeRequest(PySequence_Fast_GET_ITEM(preroll, step), request, true);
            if (accepted)
                found->second->pending.push_back(request);
        }
        Py_DECREF(preroll);
        FrameRequest request;
        accepted = accepted && DecodeRequest(item, request, false);
        if (accepted)
            found->second->pending.push_back(request);
    }
    Py_DECREF(sequence);
    return accepted;
}

bool WebParticleRuntime::SetPlaying(uint64_t emitterId, bool playing)
{
    if (!m_state->emissionSupported)
        return m_state->capacityLimitedEmitterGraphs.find(emitterId) != m_state->capacityLimitedEmitterGraphs.end();
    auto found = m_state->emitters.find(emitterId);
    if (found == m_state->emitters.end())
        return false;
    auto graph = m_state->graphs.find(found->second->graphId);
    if (graph == m_state->graphs.end())
        return false;
    const uint32_t value = playing ? 1u : 0u;
    if (!m_state->rhi->WriteBuffer(graph->second->playingStates,
                                   static_cast<uint64_t>(found->second->slot) * sizeof(uint32_t), &value,
                                   sizeof(value)))
        return false;
    found->second->playing = playing;
    return true;
}

bool WebParticleRuntime::Reset(uint64_t emitterId)
{
    if (!m_state->emissionSupported)
        return m_state->capacityLimitedEmitterGraphs.find(emitterId) != m_state->capacityLimitedEmitterGraphs.end();
    auto found = m_state->emitters.find(emitterId);
    if (found == m_state->emitters.end())
        return false;
    found->second->resetPending = true;
    return true;
}

uint64_t WebParticleRuntime::ArtifactRevision(uint64_t emitterId) const noexcept
{
    if (!m_state->emissionSupported) {
        const auto found = m_state->capacityLimitedEmitterRevisions.find(emitterId);
        return found == m_state->capacityLimitedEmitterRevisions.end() ? 0 : found->second;
    }
    const auto found = m_state->emitters.find(emitterId);
    return found == m_state->emitters.end() ? 0 : found->second->artifactRevision;
}

bool WebParticleRuntime::StateWasPreserved(uint64_t emitterId) const noexcept
{
    const auto found = m_state->emitters.find(emitterId);
    return found != m_state->emitters.end() && found->second->stateWasPreserved;
}

bool WebParticleRuntime::RecordCompute(wgpu::CommandEncoder commandEncoder)
{
    if (!m_state->emissionSupported || !m_state->rhi || !commandEncoder)
        return !m_state->emissionSupported;
    bool hasWork = false;
    for (const auto &[_, emitter] : m_state->emitters)
        hasWork = hasWork || !emitter->pending.empty() || emitter->resetPending;
    if (!hasWork)
        return true;
    const auto recordStage = [&](const auto &record) {
        wgpu::ComputePassDescriptor passDesc;
        auto pass = commandEncoder.BeginComputePass(&passDesc);
        WebGpuComputeCommandContext context;
        const auto commands = m_state->rhi->MakeComputeCommandEncoder(context, pass);
        const bool recorded = record(commands);
        pass.End();
        return recorded;
    };
    for (auto &[_, emitter] : m_state->emitters) {
        State::Emitter *currentEmitter = emitter.get();
        if (currentEmitter->resetPending) {
            currentEmitter->runtime->RequestBootstrap();
            currentEmitter->resetPending = false;
            currentEmitter->drawReady = false;
        }
        if (currentEmitter->pending.empty())
            continue;
        if (!currentEmitter->runtime->UpdateTransforms(currentEmitter->transforms)) {
            m_state->error = "WebGPU particle transform upload failed";
            return false;
        }
        // Shared particle preparation records immutable upload snapshots.
        // Publish them on this encoder before dispatch, as the Vulkan host does.
        WebGpuTransferCommandContext transferContext;
        const auto transfer = m_state->rhi->MakeTransferCommandEncoder(transferContext, commandEncoder);
        if (!currentEmitter->runtime->RecordPendingUploads(transfer)) {
            m_state->error = "WebGPU particle upload recording failed";
            return false;
        }
        for (const FrameRequest &request : currentEmitter->pending) {
            if (currentEmitter->runtime->NeedsBootstrap() && !recordStage([&](const auto &commands) {
                    return currentEmitter->runtime->RecordBootstrap(commands, request.systemSeed, {});
                })) {
                m_state->error = "WebGPU particle bootstrap recording failed";
                return false;
            }
            if (request.simulate && request.spawnCount > 0) {
                if (!recordStage([&](const auto &commands) {
                        currentEmitter->runtime->RecordInitIndirect(
                            commands, request.spawnCount, request.spawnBaseId, request.spawnGeneration,
                            request.systemSeed, request.simulationStep, request.deltaTime, {}, {}, 0);
                        return true;
                    })) {
                    m_state->error = "WebGPU particle initialization recording failed";
                    return false;
                }
            }
            if (request.simulate) {
                const bool resetRecorded = recordStage([&](const auto &commands) {
                    return currentEmitter->runtime->RecordRenderReset(commands, {}, false, true);
                });
                const bool updateRecorded =
                    resetRecorded && recordStage([&](const auto &commands) {
                        return currentEmitter->runtime->RecordUpdate(commands, request.systemSeed,
                                                                     request.simulationStep, request.deltaTime, {});
                    });
                if (!updateRecorded) {
                    m_state->error = "WebGPU particle simulation recording failed";
                    return false;
                }
                currentEmitter->runtime->PublishAliveWrite();
            }
            if (request.render) {
                // Update may have published a different alive list. Prepare
                // its indirect dispatch arguments before recording render export.
                const bool resetRecorded = recordStage(
                    [&](const auto &commands) { return currentEmitter->runtime->RecordRenderReset(commands, {}); });
                const bool renderingRecorded = resetRecorded && recordStage([&](const auto &commands) {
                                                   return currentEmitter->runtime->RecordRendering(
                                                       commands, request.systemSeed, request.simulationStep, {});
                                               });
                if (!renderingRecorded) {
                    m_state->error = "WebGPU particle rendering export failed";
                    return false;
                }
            }
            currentEmitter->drawReady = request.render;
        }
        currentEmitter->pending.clear();
    }
    return true;
}

void WebParticleRuntime::NotifySubmission() noexcept
{
    for (const auto &[_, emitter] : m_state->emitters)
        emitter->runtime->NotifySubmission(true);
}

bool WebParticleRuntime::RequiresSceneDepth(uint32_t cameraMask, int32_t queueMin, int32_t queueMax) const noexcept
{
    for (const auto &[_, emitter] : m_state->emitters) {
        if (!emitter->Visible(cameraMask))
            continue;
        for (const auto &output : emitter->outputs) {
            const auto queue = output->RenderQueue();
            if (queueMin <= queue && queue <= queueMax && output->RequiresSceneDepth())
                return true;
        }
    }
    return false;
}

bool WebParticleRuntime::Render(wgpu::RenderPassEncoder pass, const RenderViewData &view,
                                const ShaderViewBindings &bindings, const MaterialPassPipelineDescriptor &target,
                                int32_t queueMin, int32_t queueMax, rhi::TextureViewHandle sceneDepth,
                                bool sceneDepthIsDepth)
{
    if (!m_state->rhi || !pass || target.colorFormats.size() != 1 ||
        (target.target != ShaderCompileTarget::Forward && target.target != ShaderCompileTarget::ForwardPlus))
        return false;

    const glm::mat4 cameraToWorld = glm::inverse(view.view);
    const glm::vec3 right = glm::normalize(glm::vec3(cameraToWorld[0]));
    const glm::vec3 up = glm::normalize(glm::vec3(cameraToWorld[1]));
    particle::GpuParticleViewConstants constants;
    constants.cameraRight = {right.x, right.y, right.z, 0.0f};
    constants.cameraUp = {up.x, up.y, up.z, 0.0f};
    const auto &projection = view.projection;
    constants.depthReconstruct = {projection[2][2], projection[3][2], projection[2][3], projection[3][3]};
    constants.alignmentReference = {view.position.x, view.position.y, view.position.z, 0.0f};

    WebGpuGraphicsCommandContext context;
    const auto encoder = m_state->rhi->MakeGraphicsCommandEncoder(context, pass);
    bool drew = false;
    for (const auto &[_, emitter] : m_state->emitters) {
        if (!emitter->Visible(view.cullingMask))
            continue;
        constants.objectData[2] = emitter->ownerLayerMask;
        for (const auto &output : emitter->outputs) {
            const auto queue = output->RenderQueue();
            if (queue < queueMin || queue > queueMax)
                continue;
            const auto depth = output->RequiresSceneDepth() ? sceneDepth : rhi::TextureViewHandle{};
            if (output->RecordDraw(encoder, target, emitter->runtime->IndirectBuffer(), constants, bindings, {}, depth,
                                   sceneDepthIsDepth))
                drew = true;
        }
    }
    if (drew && !m_state->reportedDraw) {
        std::printf("INFERNUX_WEBGPU_PARTICLE_DRAW_READY emitters=%zu\n", m_state->emitters.size());
        m_state->reportedDraw = true;
    }
    return drew;
}

void WebParticleRuntime::ReleaseView(rhi::BufferHandle globals) noexcept
{
    for (const auto &[_, emitter] : m_state->emitters)
        for (auto &output : emitter->outputs)
            output->ReleaseView(globals);
}

const std::string &WebParticleRuntime::LastError() const noexcept
{
    return m_state->error;
}

} // namespace infernux::web
