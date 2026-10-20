#include "WebParticleRuntime.h"

#include "InfernuxWebHostModule.h"
#include "WebGpuRhiDevice.h"
#include "WebSceneRenderer.h"

#include <core/types/ColorSpace.h>
#include <function/renderer/particle/ParticleGpuRuntime.h>
#include <function/scene/Camera.h>

#include <glm/glm.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
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

constexpr char kDepthShader[] = R"wgsl(
@group(2) @binding(0) var scene_depth: texture_depth_2d;
fn scene_depth_at(pixel: vec2<i32>) -> f32 {
    return textureLoad(scene_depth, pixel, 0);
}
)wgsl";

constexpr char kMultisampleDepthShader[] = R"wgsl(
@group(2) @binding(0) var scene_depth: texture_depth_multisampled_2d;
fn scene_depth_at(pixel: vec2<i32>) -> f32 {
    var depth = textureLoad(scene_depth, pixel, 0);
    for (var sample = 1u; sample < textureNumSamples(scene_depth); sample++) {
        depth = min(depth, textureLoad(scene_depth, pixel, sample));
    }
    return depth;
}
)wgsl";

constexpr char kParticleShader[] = R"wgsl(
struct CameraData {
    view_projection: mat4x4<f32>,
    camera_right: vec4<f32>,
    camera_up: vec4<f32>,
    depth_reconstruct: vec4<f32>,
};

struct ParticleInstance {
    position_size: vec4<f32>,
    color: vec4<f32>,
    rotation_custom: vec4<f32>,
    scale_custom: vec4<f32>,
    ribbon_data: vec4<u32>,
    custom_data: vec4<f32>,
    previous_position_history: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraData;
@group(1) @binding(0) var<storage, read> instances: array<ParticleInstance>;
@group(1) @binding(1) var<storage, read> render_indices: array<u32>;

struct OutputData {
    material_tint: vec4<f32>,
    flipbook_grid: vec4<f32>,
};
@group(1) @binding(2) var<uniform> output_data: OutputData;
@group(1) @binding(3) var sprite_texture: texture_2d<f32>;
@group(1) @binding(4) var sprite_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) next_uv: vec2<f32>,
    @location(3) frame_blend: f32,
};

@vertex
fn vertex_main(@builtin(vertex_index) vertex_index: u32,
               @builtin(instance_index) instance_index: u32) -> VertexOutput {
    let corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0));
    let particle = instances[render_indices[instance_index]];
    let corner = corners[vertex_index];
    let angle = particle.rotation_custom.x;
    let rotated = vec2<f32>(
        corner.x * cos(angle) - corner.y * sin(angle),
        corner.x * sin(angle) + corner.y * cos(angle));
    let extent = max(abs(particle.position_size.w), 0.001) *
                 max(abs(particle.scale_custom.xy), vec2<f32>(0.001));
    let world = particle.position_size.xyz +
                camera.camera_right.xyz * rotated.x * extent.x +
                camera.camera_up.xyz * rotated.y * extent.y;
    var output: VertexOutput;
    output.position = camera.view_projection * vec4<f32>(world, 1.0);
    output.color = particle.color;
    let grid = max(output_data.flipbook_grid.xy, vec2<f32>(1.0));
    let frame_position = max(particle.custom_data.x, 0.0) % (grid.x * grid.y);
    let frame = floor(frame_position);
    let next_frame = (frame + 1.0) % (grid.x * grid.y);
    let uv = vec2<f32>(corner.x * 0.5 + 0.5, 0.5 - corner.y * 0.5);
    output.uv = (uv + vec2<f32>(frame % grid.x, floor(frame / grid.x))) / grid;
    output.next_uv = (uv + vec2<f32>(next_frame % grid.x, floor(next_frame / grid.x))) / grid;
    output.frame_blend = fract(frame_position);
    return output;
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let current = textureSample(sprite_texture, sprite_sampler, input.uv);
    let next = textureSample(sprite_texture, sprite_sampler, input.next_uv);
    var color = mix(current, next, input.frame_blend) * input.color * output_data.material_tint;
    if (output_data.flipbook_grid.w > 0.5) {
        let scene = eye_depth(scene_depth_at(vec2<i32>(input.position.xy)));
        let particle = eye_depth(input.position.z);
        color.a *= smoothstep(0.0, 1.0, clamp((scene - particle) / output_data.flipbook_grid.z, 0.0, 1.0));
    }
    return color;
}

fn eye_depth(depth: f32) -> f32 {
    let p = camera.depth_reconstruct;
    let denominator = depth * p.z - p.x;
    return abs((p.y - depth * p.w) / select(1e-7, denominator, abs(denominator) > 1e-7));
}
)wgsl";

struct ParticleOutputData
{
    glm::vec4 tint{1.0f};
    glm::vec4 flipbookGrid{1.0f, 1.0f, 0.0f, 0.0f};
};

glm::mat4 ToWebClipSpace(const glm::mat4 &vulkanViewProjection)
{
    glm::mat4 correction(1.0f);
    correction[1][1] = -1.0f;
    return correction * vulkanViewProjection;
}

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

bool ReadVector4(PyObject *value, glm::vec4 &result)
{
    PyObject *sequence = PySequence_Fast(value, "particle color must contain four numbers");
    if (sequence) {
        bool valid = PySequence_Fast_GET_SIZE(sequence) == 4;
        for (Py_ssize_t index = 0; valid && index < 4; ++index)
            valid = ReadFloat(PySequence_Fast_GET_ITEM(sequence, index), result[static_cast<glm::length_t>(index)]);
        Py_DECREF(sequence);
        return valid;
    }
    PyErr_Clear();
    constexpr std::array<const char *, 4> names = {"x", "y", "z", "w"};
    for (size_t index = 0; index < names.size(); ++index) {
        PyObject *component = PyObject_GetAttrString(value, names[index]);
        if (!component || !ReadFloat(component, result[static_cast<glm::length_t>(index)])) {
            Py_XDECREF(component);
            PyErr_Clear();
            return false;
        }
        Py_DECREF(component);
    }
    return true;
}

glm::vec4 ReadOutputMaterialTint(PyObject *program)
{
    glm::vec4 tint(1.0f);
    PyObject *outputs = PySequence_Fast(Field(program, "outputs"), "particle outputs must be a sequence");
    if (!outputs || PySequence_Fast_GET_SIZE(outputs) < 1) {
        Py_XDECREF(outputs);
        PyErr_Clear();
        return tint;
    }
    PyObject *material = Field(PySequence_Fast_GET_ITEM(outputs, 0), "material");
    PyObject *native = Field(material, "native");
    PyObject *color = native ? PyObject_CallMethod(native, "get_color", "s", "baseColor") : nullptr;
    glm::vec4 authored(1.0f);
    if (color && ReadVector4(color, authored))
        tint = inx::color::SrgbToLinear(authored);
    else
        PyErr_Clear();
    Py_XDECREF(color);
    Py_DECREF(outputs);
    return tint;
}

bool ReadOutputData(PyObject *program, ParticleOutputData &data, std::string &textureGuid)
{
    data.tint = ReadOutputMaterialTint(program);
    textureGuid = "white";
    PyObject *outputs = PySequence_Fast(Field(program, "outputs"), "particle outputs must be a sequence");
    if (!outputs)
        return false;
    bool valid = true;
    if (PySequence_Fast_GET_SIZE(outputs) > 0) {
        PyObject *output = PySequence_Fast_GET_ITEM(outputs, 0);
        uint32_t columns = 0, rows = 0;
        valid = ReadUnsigned(Field(output, "flipbook_columns"), columns) && columns > 0 &&
                ReadUnsigned(Field(output, "flipbook_rows"), rows) && rows > 0;
        data.flipbookGrid = glm::vec4(columns, rows, 0.0f, 0.0f);
        bool softParticles = false;
        valid = valid && ReadBool(Field(output, "soft_particles"), softParticles) &&
                ReadFloat(Field(output, "soft_distance"), data.flipbookGrid.z) && data.flipbookGrid.z > 0.0f;
        data.flipbookGrid.w = softParticles ? 1.0f : 0.0f;
        PyObject *native = Field(Field(output, "material"), "native");
        PyObject *texture = native ? PyObject_CallMethod(native, "get_texture", "s", "texSampler") : nullptr;
        if (!texture || (texture != Py_None && !PyUnicode_Check(texture)))
            valid = false;
        else if (texture != Py_None)
            textureGuid = PyUnicode_AsUTF8(texture);
        Py_XDECREF(texture);
    }
    Py_DECREF(outputs);
    return valid;
}

bool ReadWordVector(PyObject *value, std::vector<uint32_t> &result)
{
    PyObject *sequence = PySequence_Fast(value, "particle parameter words must be a sequence");
    if (!sequence) {
        PyErr_Clear();
        return false;
    }
    const Py_ssize_t count = PySequence_Fast_GET_SIZE(sequence);
    result.clear();
    result.reserve(static_cast<size_t>(std::max<Py_ssize_t>(0, count)));
    bool valid = count >= 0 && count % 4 == 0;
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
    PyObject *sequence = PySequence_Fast(outputs, "particle outputs must be a sequence");
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
        uint64_t id = 0;
        uint32_t slotCount = 0;
        std::vector<uint32_t> parameterWords;
        rhi::BufferHandle burstRequests;
        rhi::BufferHandle spawnMetadata;
        rhi::BufferHandle parameters;
        rhi::BufferHandle playingRequests;
        rhi::BufferHandle playingStates;
    };

    struct Emitter
    {
        uint64_t id = 0;
        uint64_t graphId = 0;
        uint64_t artifactRevision = 0;
        bool stateWasPreserved = false;
        uint32_t slot = 0;
        uint32_t capacity = 0;
        uint32_t ownerLayerMask = 0;
        bool playing = true;
        bool resetPending = false;
        bool drawReady = false;
        particle::GpuParticleTransforms transforms{};
        std::vector<FrameRequest> pending;
        std::unique_ptr<particle::ParticleGpuRuntime> runtime;
        rhi::BindGroupHandle spawnGroup;
        wgpu::Buffer outputBuffer;
        wgpu::BindGroup renderGroup;
        std::string textureGuid;
        wgpu::TextureView boundTexture;
        wgpu::Sampler boundSampler;
    };

    WebGpuRhiDevice *rhi = nullptr;
    WebSceneRenderer *scene = nullptr;
    wgpu::Device device;
    wgpu::Queue queue;
    rhi::PixelFormat colorFormat = rhi::PixelFormat::Undefined;
    wgpu::RenderPipeline renderPipeline;
    rhi::SampleCount renderSampleCount = rhi::SampleCount::One;
    wgpu::BindGroupLayout cameraLayout;
    wgpu::BindGroupLayout geometryLayout;
    wgpu::Buffer cameraBuffer;
    wgpu::BindGroup cameraGroup;
    wgpu::BindGroupLayout depthLayout;
    wgpu::BindGroup depthGroup;
    wgpu::TextureView depthView;
    std::unordered_map<uint64_t, Graph> graphs;
    std::unordered_map<uint64_t, std::unique_ptr<Emitter>> emitters;
    std::unordered_set<uint64_t> capacityLimitedGraphs;
    std::unordered_map<uint64_t, uint64_t> capacityLimitedEmitterGraphs;
    std::unordered_map<uint64_t, uint64_t> capacityLimitedEmitterRevisions;
    std::string error;
    bool emissionSupported = true;
    bool reportedReady = false;
    bool reportedDraw = false;

    void ReleaseGraph(Graph &graph) noexcept
    {
        if (!rhi)
            return;
        rhi->Release(graph.playingStates);
        rhi->Release(graph.playingRequests);
        rhi->Release(graph.parameters);
        rhi->Release(graph.spawnMetadata);
        rhi->Release(graph.burstRequests);
        graph = {};
    }

    void ReleaseEmitter(Emitter &emitter) noexcept
    {
        emitter.renderGroup = {};
        emitter.boundTexture = {};
        emitter.boundSampler = {};
        emitter.outputBuffer = {};
        if (rhi)
            rhi->Release(emitter.spawnGroup);
        emitter.spawnGroup = {};
        emitter.runtime.reset();
    }

    bool PrepareRenderGroup(Emitter &emitter)
    {
        const auto texture = scene->ResolveSampledTexture(emitter.textureGuid);
        if (!texture.view || !texture.sampler)
            return false;
        if (emitter.renderGroup && emitter.boundTexture.Get() == texture.view.Get() &&
            emitter.boundSampler.Get() == texture.sampler.Get())
            return true;
        std::array<wgpu::BindGroupEntry, 5> bindings{};
        bindings[0].binding = 0;
        bindings[0].buffer = rhi->GetNativeBuffer(emitter.runtime->InstanceBuffer());
        bindings[0].size = static_cast<uint64_t>(emitter.capacity) * particle::ParticleGpuRuntime::RenderInstanceStride;
        bindings[1].binding = 1;
        bindings[1].buffer = rhi->GetNativeBuffer(emitter.runtime->RenderIndexBuffer());
        bindings[1].size = static_cast<uint64_t>(emitter.capacity) * sizeof(uint32_t);
        bindings[2].binding = 2;
        bindings[2].buffer = emitter.outputBuffer;
        bindings[2].size = sizeof(ParticleOutputData);
        bindings[3].binding = 3;
        bindings[3].textureView = texture.view;
        bindings[4].binding = 4;
        bindings[4].sampler = texture.sampler;
        wgpu::BindGroupDescriptor descriptor;
        descriptor.layout = geometryLayout;
        descriptor.entryCount = bindings.size();
        descriptor.entries = bindings.data();
        emitter.renderGroup = device.CreateBindGroup(&descriptor);
        emitter.boundTexture = texture.view;
        emitter.boundSampler = texture.sampler;
        return static_cast<bool>(emitter.renderGroup);
    }

    void Clear() noexcept
    {
        for (auto &[_, emitter] : emitters)
            ReleaseEmitter(*emitter);
        emitters.clear();
        for (auto &[_, graph] : graphs)
            ReleaseGraph(graph);
        graphs.clear();
        capacityLimitedGraphs.clear();
        capacityLimitedEmitterGraphs.clear();
        capacityLimitedEmitterRevisions.clear();
        renderPipeline = {};
        cameraGroup = {};
        cameraBuffer = {};
        geometryLayout = {};
        cameraLayout = {};
        depthGroup = {};
        depthView = {};
        depthLayout = {};
        queue = {};
        device = {};
        rhi = nullptr;
        scene = nullptr;
        emissionSupported = true;
    }
};

WebParticleRuntime::WebParticleRuntime() : m_state(std::make_unique<State>())
{
}

WebParticleRuntime::~WebParticleRuntime()
{
    Shutdown();
}

bool WebParticleRuntime::Initialize(WebGpuRhiDevice &device, WebSceneRenderer &scene, rhi::PixelFormat colorFormat,
                                    rhi::SampleCount sceneSampleCount)
{
    Shutdown();
    m_state->rhi = &device;
    m_state->scene = &scene;
    m_state->device = device.NativeDevice();
    m_state->queue = device.NativeQueue();
    m_state->colorFormat = colorFormat;
    if (!m_state->device || !m_state->queue || colorFormat == rhi::PixelFormat::Undefined) {
        m_state->error = "Web particle runtime requires a ready WebGPU device";
        return false;
    }
    constexpr uint32_t kRequiredParticleStorageBuffersPerStage = 12;
    const uint32_t availableStorageBuffers = device.GetCapabilities().limits.maxStorageBuffersPerStage;
    if (availableStorageBuffers < kRequiredParticleStorageBuffersPerStage) {
        // Device limits are immutable for a WebGPU device. Keep the public
        // particle control plane resident, but apply capacity backpressure at
        // zero so scene startup and non-particle rendering remain available.
        // A supported device takes the only full GPU implementation below.
        m_state->emissionSupported = false;
        m_state->reportedReady = true;
        std::printf("INFERNUX_WEBGPU_PARTICLE_CAPACITY_LIMIT available=%u required=%u emission=paused\n",
                    availableStorageBuffers, kRequiredParticleStorageBuffersPerStage);
        return true;
    }

    wgpu::BufferDescriptor cameraBufferDesc;
    cameraBufferDesc.size = sizeof(glm::mat4) + sizeof(glm::vec4) * 3;
    cameraBufferDesc.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
    m_state->cameraBuffer = m_state->device.CreateBuffer(&cameraBufferDesc);

    wgpu::BindGroupLayoutEntry cameraEntry;
    cameraEntry.binding = 0;
    cameraEntry.visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment;
    cameraEntry.buffer.type = wgpu::BufferBindingType::Uniform;
    cameraEntry.buffer.minBindingSize = cameraBufferDesc.size;
    wgpu::BindGroupLayoutDescriptor cameraLayoutDesc;
    cameraLayoutDesc.entryCount = 1;
    cameraLayoutDesc.entries = &cameraEntry;
    m_state->cameraLayout = m_state->device.CreateBindGroupLayout(&cameraLayoutDesc);
    wgpu::BindGroupEntry cameraBinding;
    cameraBinding.binding = 0;
    cameraBinding.buffer = m_state->cameraBuffer;
    cameraBinding.size = cameraBufferDesc.size;
    wgpu::BindGroupDescriptor cameraGroupDesc;
    cameraGroupDesc.layout = m_state->cameraLayout;
    cameraGroupDesc.entryCount = 1;
    cameraGroupDesc.entries = &cameraBinding;
    m_state->cameraGroup = m_state->device.CreateBindGroup(&cameraGroupDesc);

    std::array<wgpu::BindGroupLayoutEntry, 5> geometryEntries{};
    for (uint32_t index = 0; index < 2; ++index) {
        geometryEntries[index].binding = index;
        geometryEntries[index].visibility = wgpu::ShaderStage::Vertex;
        geometryEntries[index].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
    }
    geometryEntries[2].binding = 2;
    geometryEntries[2].visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment;
    geometryEntries[2].buffer.type = wgpu::BufferBindingType::Uniform;
    geometryEntries[2].buffer.minBindingSize = sizeof(ParticleOutputData);
    geometryEntries[3].binding = 3;
    geometryEntries[3].visibility = wgpu::ShaderStage::Fragment;
    geometryEntries[3].texture.sampleType = wgpu::TextureSampleType::Float;
    geometryEntries[3].texture.viewDimension = wgpu::TextureViewDimension::e2D;
    geometryEntries[4].binding = 4;
    geometryEntries[4].visibility = wgpu::ShaderStage::Fragment;
    geometryEntries[4].sampler.type = wgpu::SamplerBindingType::Filtering;
    wgpu::BindGroupLayoutDescriptor geometryLayoutDesc;
    geometryLayoutDesc.entryCount = geometryEntries.size();
    geometryLayoutDesc.entries = geometryEntries.data();
    m_state->geometryLayout = m_state->device.CreateBindGroupLayout(&geometryLayoutDesc);

    if (!m_state->cameraBuffer || !m_state->cameraLayout || !m_state->cameraGroup || !m_state->geometryLayout ||
        !ConfigureSampleCount(sceneSampleCount)) {
        m_state->error = "WebGPU failed to create the particle render pipeline";
        Shutdown();
        return false;
    }
    std::printf("INFERNUX_WEBGPU_PARTICLE_RUNTIME_READY\n");
    m_state->reportedReady = true;
    return true;
}

bool WebParticleRuntime::ConfigureSampleCount(rhi::SampleCount sceneSampleCount)
{
    if (!m_state->device || (sceneSampleCount != rhi::SampleCount::One && sceneSampleCount != rhi::SampleCount::Four))
        return false;
    if (!m_state->emissionSupported)
        return true;
    if (m_state->renderPipeline && m_state->renderSampleCount == sceneSampleCount)
        return true;
    const auto colorFormat = m_state->colorFormat;

    wgpu::BindGroupLayoutEntry depthEntry;
    depthEntry.binding = 0;
    depthEntry.visibility = wgpu::ShaderStage::Fragment;
    depthEntry.texture.sampleType = wgpu::TextureSampleType::Depth;
    depthEntry.texture.viewDimension = wgpu::TextureViewDimension::e2D;
    depthEntry.texture.multisampled = sceneSampleCount == rhi::SampleCount::Four;
    wgpu::BindGroupLayoutDescriptor depthLayoutDesc;
    depthLayoutDesc.entryCount = 1;
    depthLayoutDesc.entries = &depthEntry;
    const auto depthLayout = m_state->device.CreateBindGroupLayout(&depthLayoutDesc);
    const std::string shaderText =
        std::string(depthEntry.texture.multisampled ? kMultisampleDepthShader : kDepthShader) + kParticleShader;
    wgpu::ShaderSourceWGSL shaderSource;
    shaderSource.code = shaderText.c_str();
    wgpu::ShaderModuleDescriptor shaderDesc;
    shaderDesc.nextInChain = &shaderSource;
    const auto shader = m_state->device.CreateShaderModule(&shaderDesc);
    const std::array<wgpu::BindGroupLayout, 3> layouts = {m_state->cameraLayout, m_state->geometryLayout, depthLayout};
    wgpu::PipelineLayoutDescriptor pipelineLayoutDesc;
    pipelineLayoutDesc.bindGroupLayoutCount = layouts.size();
    pipelineLayoutDesc.bindGroupLayouts = layouts.data();
    wgpu::ColorTargetState colorTarget;
    colorTarget.format = colorFormat == rhi::PixelFormat::BGRA8Srgb      ? wgpu::TextureFormat::BGRA8UnormSrgb
                         : colorFormat == rhi::PixelFormat::RGBA8Srgb    ? wgpu::TextureFormat::RGBA8UnormSrgb
                         : colorFormat == rhi::PixelFormat::RGBA8UNorm   ? wgpu::TextureFormat::RGBA8Unorm
                         : colorFormat == rhi::PixelFormat::RGBA16SFloat ? wgpu::TextureFormat::RGBA16Float
                                                                         : wgpu::TextureFormat::BGRA8Unorm;
    wgpu::BlendState blend;
    blend.color.srcFactor = wgpu::BlendFactor::SrcAlpha;
    blend.color.dstFactor = wgpu::BlendFactor::OneMinusSrcAlpha;
    blend.alpha.srcFactor = wgpu::BlendFactor::One;
    blend.alpha.dstFactor = wgpu::BlendFactor::OneMinusSrcAlpha;
    colorTarget.blend = &blend;
    wgpu::FragmentState fragment;
    fragment.module = shader;
    fragment.entryPoint = "fragment_main";
    fragment.targetCount = 1;
    fragment.targets = &colorTarget;
    wgpu::DepthStencilState depth;
    depth.format = wgpu::TextureFormat::Depth24Plus;
    depth.depthWriteEnabled = wgpu::OptionalBool::False;
    depth.depthCompare = wgpu::CompareFunction::LessEqual;
    wgpu::RenderPipelineDescriptor pipelineDesc;
    pipelineDesc.layout = m_state->device.CreatePipelineLayout(&pipelineLayoutDesc);
    pipelineDesc.vertex.module = shader;
    pipelineDesc.vertex.entryPoint = "vertex_main";
    pipelineDesc.fragment = &fragment;
    pipelineDesc.primitive.topology = wgpu::PrimitiveTopology::TriangleList;
    pipelineDesc.primitive.cullMode = wgpu::CullMode::None;
    pipelineDesc.depthStencil = &depth;
    pipelineDesc.multisample.count = static_cast<uint32_t>(sceneSampleCount);
    const auto candidate = m_state->device.CreateRenderPipeline(&pipelineDesc);
    if (!candidate) {
        m_state->error = "WebGPU failed to create the particle render pipeline";
        return false;
    }
    m_state->renderPipeline = candidate;
    m_state->renderSampleCount = sceneSampleCount;
    m_state->depthLayout = depthLayout;
    m_state->depthGroup = {};
    m_state->depthView = {};
    return true;
}

void WebParticleRuntime::Shutdown() noexcept
{
    if (m_state)
        m_state->Clear();
}

std::string WebParticleRuntime::ReplaceGraph(uint64_t graphInstanceId, PyObject *programs, PyObject *removeIds)
{
    if (!m_state->rhi || graphInstanceId == 0)
        return "WebGPU particle runtime is not initialized";
    PyObject *programSequence = PySequence_Fast(programs, "particle programs must be a sequence");
    PyObject *removeSequence = PySequence_Fast(removeIds, "particle remove_ids must be a sequence");
    if (!programSequence || !removeSequence) {
        PyErr_Clear();
        Py_XDECREF(programSequence);
        Py_XDECREF(removeSequence);
        return "WebGPU particle graph publication contains an invalid sequence";
    }
    for (Py_ssize_t index = 0; index < PySequence_Fast_GET_SIZE(removeSequence); ++index) {
        uint64_t id = 0;
        if (!ReadUnsigned(PySequence_Fast_GET_ITEM(removeSequence, index), id)) {
            Py_DECREF(programSequence);
            Py_DECREF(removeSequence);
            return "WebGPU particle remove_ids contains an invalid emitter id";
        }
        auto found = m_state->emitters.find(id);
        if (found != m_state->emitters.end()) {
            m_state->ReleaseEmitter(*found->second);
            m_state->emitters.erase(found);
        }
        m_state->capacityLimitedEmitterGraphs.erase(id);
        m_state->capacityLimitedEmitterRevisions.erase(id);
    }
    Py_DECREF(removeSequence);

    const Py_ssize_t programCount = PySequence_Fast_GET_SIZE(programSequence);
    if (programCount == 0) {
        Py_DECREF(programSequence);
        auto graph = m_state->graphs.find(graphInstanceId);
        if (graph != m_state->graphs.end()) {
            m_state->ReleaseGraph(graph->second);
            m_state->graphs.erase(graph);
        }
        m_state->capacityLimitedGraphs.erase(graphInstanceId);
        for (auto iterator = m_state->capacityLimitedEmitterGraphs.begin();
             iterator != m_state->capacityLimitedEmitterGraphs.end();) {
            if (iterator->second != graphInstanceId) {
                ++iterator;
                continue;
            }
            m_state->capacityLimitedEmitterRevisions.erase(iterator->first);
            iterator = m_state->capacityLimitedEmitterGraphs.erase(iterator);
        }
        return {};
    }

    uint32_t slotCount = 0;
    std::vector<uint32_t> parameters;
    std::vector<uint64_t> graphEmitterIds;
    for (Py_ssize_t index = 0; index < programCount; ++index) {
        PyObject *program = PySequence_Fast_GET_ITEM(programSequence, index);
        std::string error;
        uint64_t id = 0;
        uint64_t encodedGraph = 0;
        uint32_t slot = 0;
        if (!ProgramUsesPortableSubset(program, error) || !ReadUnsigned(Field(program, "id"), id) || id == 0 ||
            !ReadUnsigned(Field(program, "graph_instance_id"), encodedGraph) || encodedGraph != graphInstanceId ||
            !ReadUnsigned(Field(program, "graph_emitter_index"), slot) || slot >= 1024) {
            Py_DECREF(programSequence);
            return error.empty() ? "WebGPU particle program identity is invalid" : error;
        }
        std::vector<uint32_t> words;
        if (!ReadWordVector(Field(program, "parameter_words"), words)) {
            Py_DECREF(programSequence);
            return "WebGPU particle parameter block is invalid";
        }
        if (parameters.empty())
            parameters = std::move(words);
        else if (parameters != words) {
            Py_DECREF(programSequence);
            return "WebGPU particle emitters in one graph disagree on parameters";
        }
        slotCount = std::max(slotCount, slot + 1);
        graphEmitterIds.push_back(id);
    }
    if (parameters.empty())
        parameters.assign(4, 0);

    if (!m_state->emissionSupported) {
        for (auto iterator = m_state->capacityLimitedEmitterGraphs.begin();
             iterator != m_state->capacityLimitedEmitterGraphs.end();) {
            if (iterator->second != graphInstanceId) {
                ++iterator;
                continue;
            }
            m_state->capacityLimitedEmitterRevisions.erase(iterator->first);
            iterator = m_state->capacityLimitedEmitterGraphs.erase(iterator);
        }
        for (Py_ssize_t index = 0; index < programCount; ++index) {
            PyObject *program = PySequence_Fast_GET_ITEM(programSequence, index);
            uint64_t emitterId = 0;
            uint64_t artifactRevision = 0;
            if (!ReadUnsigned(Field(program, "id"), emitterId) || emitterId == 0 ||
                !ReadUnsigned(Field(program, "artifact_revision"), artifactRevision) || artifactRevision == 0) {
                Py_DECREF(programSequence);
                return "WebGPU capacity-limited particle program identity is invalid";
            }
            m_state->capacityLimitedEmitterGraphs[emitterId] = graphInstanceId;
            m_state->capacityLimitedEmitterRevisions[emitterId] = artifactRevision;
        }
        m_state->capacityLimitedGraphs.insert(graphInstanceId);
        Py_DECREF(programSequence);
        return {};
    }

    for (auto iterator = m_state->emitters.begin(); iterator != m_state->emitters.end();) {
        if (iterator->second->graphId != graphInstanceId) {
            ++iterator;
            continue;
        }
        m_state->ReleaseEmitter(*iterator->second);
        iterator = m_state->emitters.erase(iterator);
    }
    auto oldGraph = m_state->graphs.find(graphInstanceId);
    if (oldGraph != m_state->graphs.end()) {
        m_state->ReleaseGraph(oldGraph->second);
        m_state->graphs.erase(oldGraph);
    }

    State::Graph graph;
    graph.id = graphInstanceId;
    graph.slotCount = slotCount;
    graph.parameterWords = parameters;
    std::vector<uint32_t> zeros(slotCount, 0);
    std::vector<uint32_t> playing(slotCount, 1);
    graph.burstRequests = CreateStorageBuffer(*m_state->rhi, slotCount * sizeof(uint32_t), false, zeros.data());
    graph.spawnMetadata = CreateStorageBuffer(*m_state->rhi, slotCount * kSpawnMetadataStride, true);
    graph.parameters =
        CreateStorageBuffer(*m_state->rhi, parameters.size() * sizeof(uint32_t), false, parameters.data());
    graph.playingRequests = CreateStorageBuffer(*m_state->rhi, slotCount * sizeof(uint32_t), false, zeros.data());
    graph.playingStates = CreateStorageBuffer(*m_state->rhi, slotCount * sizeof(uint32_t), false, playing.data());
    if (!graph.burstRequests.IsValid() || !graph.spawnMetadata.IsValid() || !graph.parameters.IsValid() ||
        !graph.playingRequests.IsValid() || !graph.playingStates.IsValid()) {
        m_state->ReleaseGraph(graph);
        Py_DECREF(programSequence);
        return "WebGPU particle graph buffers could not be created";
    }
    m_state->graphs.emplace(graphInstanceId, std::move(graph));
    auto &publishedGraph = m_state->graphs.at(graphInstanceId);

    for (Py_ssize_t index = 0; index < programCount; ++index) {
        PyObject *program = PySequence_Fast_GET_ITEM(programSequence, index);
        auto emitter = std::make_unique<State::Emitter>();
        uint32_t capacity = 0;
        uint32_t stateStride = 0;
        uint32_t eventTypes = 0;
        uint32_t slot = 0;
        if (!ReadUnsigned(Field(program, "id"), emitter->id) ||
            !ReadUnsigned(Field(program, "graph_emitter_index"), slot) ||
            !ReadUnsigned(Field(program, "owner_layer_mask"), emitter->ownerLayerMask) ||
            !ReadUnsigned(Field(program, "artifact_revision"), emitter->artifactRevision) ||
            !ReadUnsigned(Field(program, "capacity"), capacity) || capacity == 0 || capacity > (1u << 24u) ||
            !ReadUnsigned(Field(program, "state_stride"), stateStride) || stateStride == 0 || stateStride % 16 != 0 ||
            !ReadUnsigned(Field(program, "event_type_count"), eventTypes)) {
            Py_DECREF(programSequence);
            return "WebGPU particle program layout is invalid";
        }
        PyObject *hashObject = Field(program, "kernel_hash");
        if (!hashObject || !PyUnicode_Check(hashObject)) {
            Py_DECREF(programSequence);
            return "WebGPU particle kernel hash is missing";
        }
        const char *kernelHash = PyUnicode_AsUTF8(hashObject);
        if (!kernelHash || std::strlen(kernelHash) != 64) {
            Py_DECREF(programSequence);
            return "WebGPU particle kernel hash is invalid";
        }
        particle::GpuEmitterDesc desc;
        desc.capacity = capacity;
        desc.stateStride = stateStride;
        desc.eventTypeCount = eventTypes;
        // The fused Vulkan kernel exceeds the portable WebGPU per-stage
        // storage-buffer budget. Keep simulation and render export as distinct
        // GPU stages so each pipeline publishes only the resources it uses.
        desc.supportsFusedUpdateRendering = false;
        std::array<std::string, static_cast<size_t>(particle::GpuKernelStage::Count)> sources;
        for (size_t stage = 0; stage < sources.size(); ++stage) {
            const std::string name = std::string("Particle/") + kernelHash + "/" + kStageNames[stage];
            if (!InfernuxWebFindShaderSource(name, "compute", sources[stage])) {
                Py_DECREF(programSequence);
                return "WebGPU particle shader catalog is missing " + name;
            }
            desc.kernels[stage] = particle::ShaderBytecode::FromWgsl(sources[stage].data(), sources[stage].size());
        }
        emitter->graphId = graphInstanceId;
        emitter->slot = slot;
        emitter->capacity = capacity;
        emitter->runtime = std::make_unique<particle::ParticleGpuRuntime>();
        if (!emitter->runtime->Create(*m_state->rhi, desc)) {
            Py_DECREF(programSequence);
            return "WebGPU particle compute pipelines could not be created: " + m_state->rhi->LastError();
        }
        const particle::GpuParticleSimulationControl simulationControl{};
        particle::GpuParticleSimulationControl enabledControl = simulationControl;
        enabledControl.anyViewVisible = 1;
        enabledControl.simulationAllowed = 1;
        if (!m_state->rhi->WriteBuffer(emitter->runtime->SimulationControlBuffer(), 0, &enabledControl,
                                       sizeof(enabledControl))) {
            Py_DECREF(programSequence);
            return "WebGPU particle simulation control could not be initialized";
        }
        rhi::BindGroupDesc spawnGroupDesc;
        spawnGroupDesc.layout = emitter->runtime->GraphSpawnLayout();
        spawnGroupDesc.buffers[spawnGroupDesc.bufferCount++] = {0, rhi::BindingType::StorageBuffer,
                                                                publishedGraph.burstRequests};
        spawnGroupDesc.buffers[spawnGroupDesc.bufferCount++] = {
            1, rhi::BindingType::StorageBuffer, publishedGraph.spawnMetadata,
            static_cast<uint64_t>(slot) * kSpawnMetadataStride, sizeof(WebSpawnMetadata)};
        spawnGroupDesc.buffers[spawnGroupDesc.bufferCount++] = {2, rhi::BindingType::StorageBuffer,
                                                                publishedGraph.parameters};
        spawnGroupDesc.buffers[spawnGroupDesc.bufferCount++] = {3, rhi::BindingType::StorageBuffer,
                                                                publishedGraph.playingRequests};
        spawnGroupDesc.buffers[spawnGroupDesc.bufferCount++] = {4, rhi::BindingType::StorageBuffer,
                                                                publishedGraph.playingStates};
        emitter->spawnGroup = m_state->rhi->CreateBindGroup(spawnGroupDesc);

        const auto instances = m_state->rhi->GetNativeBuffer(emitter->runtime->InstanceBuffer());
        const auto indices = m_state->rhi->GetNativeBuffer(emitter->runtime->RenderIndexBuffer());
        ParticleOutputData outputData;
        if (!ReadOutputData(program, outputData, emitter->textureGuid)) {
            PyErr_Clear();
            Py_DECREF(programSequence);
            return "WebGPU particle output material or flipbook layout is invalid";
        }
        wgpu::BufferDescriptor outputBufferDescriptor;
        outputBufferDescriptor.size = sizeof(outputData);
        outputBufferDescriptor.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
        emitter->outputBuffer = m_state->device.CreateBuffer(&outputBufferDescriptor);
        if (emitter->outputBuffer)
            m_state->queue.WriteBuffer(emitter->outputBuffer, 0, &outputData, sizeof(outputData));
        if (!emitter->spawnGroup.IsValid() || !emitter->outputBuffer || !instances || !indices) {
            Py_DECREF(programSequence);
            return "WebGPU particle bind groups could not be created";
        }
        std::printf("INFERNUX_WEBGPU_PARTICLE_OUTPUT_READY tint=%.4f,%.4f,%.4f,%.4f hdr=1\n", outputData.tint.x,
                    outputData.tint.y, outputData.tint.z, outputData.tint.w);
        m_state->emitters.emplace(emitter->id, std::move(emitter));
    }
    Py_DECREF(programSequence);
    return {};
}

std::string WebParticleRuntime::UpdateParameters(uint64_t graphInstanceId, PyObject *words)
{
    if (!m_state->emissionSupported)
        return m_state->capacityLimitedGraphs.find(graphInstanceId) != m_state->capacityLimitedGraphs.end()
                   ? std::string{}
                   : "WebGPU particle graph is not resident";
    auto found = m_state->graphs.find(graphInstanceId);
    std::vector<uint32_t> decoded;
    if (found == m_state->graphs.end() || !ReadWordVector(words, decoded) ||
        decoded.size() != found->second.parameterWords.size())
        return "WebGPU particle parameter update does not match a live graph";
    if (!m_state->rhi->WriteBuffer(found->second.parameters, 0, decoded.data(), decoded.size() * sizeof(uint32_t)))
        return "WebGPU particle parameter upload failed";
    found->second.parameterWords = std::move(decoded);
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
    if (!m_state->rhi->WriteBuffer(graph->second.playingStates,
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
                    return currentEmitter->runtime->RecordBootstrap(commands, request.systemSeed,
                                                                    currentEmitter->spawnGroup);
                })) {
                m_state->error = "WebGPU particle bootstrap recording failed";
                return false;
            }
            if (request.simulate && request.spawnCount > 0) {
                if (!recordStage([&](const auto &commands) {
                        currentEmitter->runtime->RecordInitIndirect(commands, request.spawnCount, request.spawnBaseId,
                                                                    request.spawnGeneration, request.systemSeed,
                                                                    request.simulationStep, request.deltaTime,
                                                                    currentEmitter->spawnGroup, {}, 0);
                        return true;
                    })) {
                    m_state->error = "WebGPU particle initialization recording failed";
                    return false;
                }
            }
            if (request.simulate) {
                const bool resetRecorded = recordStage([&](const auto &commands) {
                    return currentEmitter->runtime->RecordRenderReset(commands, currentEmitter->spawnGroup, false,
                                                                      true);
                });
                const bool updateRecorded = resetRecorded && recordStage([&](const auto &commands) {
                                                return currentEmitter->runtime->RecordUpdate(
                                                    commands, request.systemSeed, request.simulationStep,
                                                    request.deltaTime, currentEmitter->spawnGroup);
                                            });
                if (!updateRecorded) {
                    m_state->error = "WebGPU particle simulation recording failed";
                    return false;
                }
                currentEmitter->runtime->PublishAliveWrite();
                if (request.render && !recordStage([&](const auto &commands) {
                        return currentEmitter->runtime->RecordRendering(
                            commands, request.systemSeed, request.simulationStep, currentEmitter->spawnGroup);
                    })) {
                    m_state->error = "WebGPU particle rendering export failed";
                    return false;
                }
            } else if (request.render) {
                const bool resetRecorded = recordStage([&](const auto &commands) {
                    return currentEmitter->runtime->RecordRenderReset(commands, currentEmitter->spawnGroup);
                });
                const bool renderingRecorded =
                    resetRecorded && recordStage([&](const auto &commands) {
                        return currentEmitter->runtime->RecordRendering(
                            commands, request.systemSeed, request.simulationStep, currentEmitter->spawnGroup);
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

bool WebParticleRuntime::HasRenderWork(uint32_t cameraMask) const noexcept
{
    return m_state->emissionSupported && m_state->renderPipeline &&
           std::any_of(m_state->emitters.begin(), m_state->emitters.end(), [cameraMask](const auto &entry) {
               const auto &emitter = *entry.second;
               return emitter.drawReady && emitter.playing && (emitter.ownerLayerMask & cameraMask) != 0;
           });
}

bool WebParticleRuntime::Render(wgpu::RenderPassEncoder pass, Camera &camera, uint32_t width, uint32_t height)
{
    if (!m_state->emissionSupported || !m_state->renderPipeline || !pass)
        return false;
    camera.SetAspectRatio(static_cast<float>(width) / static_cast<float>(std::max(1u, height)));
    struct CameraData
    {
        glm::mat4 viewProjection{1.0f};
        glm::vec4 right{1.0f, 0.0f, 0.0f, 0.0f};
        glm::vec4 up{0.0f, 1.0f, 0.0f, 0.0f};
        glm::vec4 depthReconstruct{};
    } data;
    data.viewProjection = ToWebClipSpace(camera.GetViewProjectionMatrix());
    const glm::mat4 cameraToWorld = glm::inverse(camera.GetViewMatrix());
    data.right = glm::vec4(glm::normalize(glm::vec3(cameraToWorld[0])), 0.0f);
    data.up = glm::vec4(glm::normalize(glm::vec3(cameraToWorld[1])), 0.0f);
    const auto projection = camera.GetProjectionMatrix();
    data.depthReconstruct = {projection[2][2], projection[3][2], projection[2][3], projection[3][3]};
    const auto depthView = m_state->scene->GetDepthView();
    if (!depthView)
        return false;
    if (!m_state->depthGroup || m_state->depthView.Get() != depthView.Get()) {
        wgpu::BindGroupEntry entry;
        entry.binding = 0;
        entry.textureView = depthView;
        wgpu::BindGroupDescriptor descriptor;
        descriptor.layout = m_state->depthLayout;
        descriptor.entryCount = 1;
        descriptor.entries = &entry;
        m_state->depthGroup = m_state->device.CreateBindGroup(&descriptor);
        m_state->depthView = depthView;
    }
    m_state->queue.WriteBuffer(m_state->cameraBuffer, 0, &data, sizeof(data));
    pass.SetPipeline(m_state->renderPipeline);
    pass.SetBindGroup(0, m_state->cameraGroup);
    pass.SetBindGroup(2, m_state->depthGroup);
    bool drew = false;
    for (const auto &[_, emitter] : m_state->emitters) {
        if (!emitter->drawReady || !emitter->playing ||
            (emitter->ownerLayerMask & camera.GetCullingMask()) == 0)
            continue;
        if (!m_state->PrepareRenderGroup(*emitter))
            continue;
        const auto indirect = m_state->rhi->GetNativeBuffer(emitter->runtime->IndirectBuffer());
        if (!indirect)
            continue;
        pass.SetBindGroup(1, emitter->renderGroup);
        pass.DrawIndirect(indirect, 0);
        drew = true;
    }
    if (drew && !m_state->reportedDraw) {
        std::printf("INFERNUX_WEBGPU_PARTICLE_DRAW_READY emitters=%zu\n", m_state->emitters.size());
        m_state->reportedDraw = true;
    }
    return drew;
}

const std::string &WebParticleRuntime::LastError() const noexcept
{
    return m_state->error;
}

} // namespace infernux::web
