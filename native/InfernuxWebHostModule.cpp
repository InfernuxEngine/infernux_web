#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "InfernuxWebHostModule.h"
#include "WebParticleRuntime.h"
#include "WebScreenUIRenderer.h"
#include "WebSceneRenderer.h"

#include <core/threading/FrameWorkOwner.h>
#include <function/renderer/gui/InxTextLayout.h>
#include <function/renderer/rhi/RhiComputeHost.h>
#include <function/resources/InxTexture/InxTexture.h>
#include <platform/filesystem/InxPack.h>
#include <platform/input/InputManager.h>

#include <emscripten.h>
#include <stb_image.h>

#if defined(INFERNUX_WEB_ENGINE_RUNTIME)
#include "WebTextureAssetLoad.h"

#include <core/config/EngineConfig.h>
#include <function/audio/AudioClipLoader.h>
#include <function/renderer/gui/InxTextGeometry.h>
#include <function/renderer/particle/ParticleGpuPublicationStatus.h>
#include <function/resources/AssetDatabase/AssetDatabase.h>
#include <function/resources/AssetRegistry/AssetRegistry.h>
#include <function/resources/InxFileLoader/InxDefaultLoader.hpp>
#include <function/resources/InxFileLoader/InxPythonScriptLoader.hpp>
#include <function/resources/InxMaterial/MaterialLoader.h>
#include <function/resources/InxMesh/MeshLoader.h>
#include <function/resources/InxTexture/InxTexture.h>
#include <function/resources/InxTexture/TextureLoader.h>
#include <function/resources/PhysicMaterial/PhysicMaterialLoader.h>
#include <function/resources/RenderTexture/RenderTextureLoader.h>
#include <function/resources/ShaderAsset/ShaderLoader.h>
#include <function/scene/SceneManager.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#endif

#include <array>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace
{

// clang-format off
EM_JS(int, RequestCanvasCapture, (const char *path), {
    const captures = Module.infernuxCaptures ||= {next: 1, records: new Map(), pending: new Set(), encoding: 0};
    if (captures.records.size >= 256) {
        for (const [id, row] of captures.records) {
            if (["completed", "failed", "cancelled"].includes(row.status)) captures.records.delete(id);
            if (captures.records.size < 256) break;
        }
    }
    if (captures.records.size >= 256) return 0;
    const id = captures.next++;
    captures.records.set(id, {capture_id: id, source: "game", pixel_origin: "webgpu_canvas",
        status: "queued", output_path: UTF8ToString(path), error: ""});
    captures.pending.add(id);
    return id;
});

EM_JS(char *, QueryCanvasCapture, (int id), {
    const row = Module.infernuxCaptures?.records.get(id);
    const text = JSON.stringify(row || {capture_id: id, status: "failed", error: "Unknown capture id"});
    const result = _malloc(lengthBytesUTF8(text) + 1);
    stringToUTF8(text, result, lengthBytesUTF8(text) + 1);
    return result;
});

EM_JS(int, CancelCanvasCapture, (int id), {
    const row = Module.infernuxCaptures?.records.get(id);
    if (!row || !["queued", "capturing"].includes(row.status)) return 0;
    row.status = "cancelled";
    Module.infernuxCaptures.pending.delete(id);
    return 1;
});

EM_JS(void, PublishCanvasCaptures, (int gameRendered, double frame, double device), {
    const captures = Module.infernuxCaptures;
    if (!gameRendered || !captures?.pending.size || captures.encoding >= 4) return;
    const canvas = Module.canvas || document.getElementById("canvas");
    for (const id of captures.pending) {
        if (captures.encoding >= 4) break;
        captures.pending.delete(id);
        const row = captures.records.get(id);
        ++captures.encoding;
        row.status = "capturing";
        row.engine_frame = frame;
        row.device_id = device;
        row.width = canvas.width;
        row.height = canvas.height;
        try {
            // toBlob snapshots the just-submitted WebGPU drawing buffer here,
            // before the browser ends this turn and clears its canvas image.
            canvas.toBlob(async blob => {
                try {
                    if (row.status !== "capturing") return;
                    if (!blob) throw new Error("WebGPU canvas PNG encoding failed");
                    const bytes = new Uint8Array(await blob.arrayBuffer());
                    if (row.status !== "capturing") return;
                    FS.writeFile(row.output_path, bytes);
                    row.status = "completed";
                } catch (error) {
                    if (row.status === "capturing") {
                        row.error = String(error);
                        row.status = "failed";
                    }
                } finally {
                    --captures.encoding;
                }
            }, "image/png");
        } catch (error) {
            --captures.encoding;
            row.error = String(error);
            row.status = "failed";
        }
    }
});
// clang-format on

PyObject *RequestCapture(PyObject *, PyObject *arguments)
{
    const char *source;
    const char *path;
    unsigned long long camera = 0;
    if (!PyArg_ParseTuple(arguments, "ss|K:request_capture", &source, &path, &camera))
        return nullptr;
    if (std::string_view(source) != "game" || camera != 0) {
        PyErr_SetString(PyExc_ValueError, "Web Player capture supports the composited Game canvas only");
        return nullptr;
    }
    const int id = RequestCanvasCapture(path);
    if (!id) {
        PyErr_SetString(PyExc_RuntimeError, "Web Player capture request limit reached");
        return nullptr;
    }
    return PyLong_FromLong(id);
}

PyObject *QueryCapture(PyObject *, PyObject *value)
{
    const long id = PyLong_AsLong(value);
    if (PyErr_Occurred()) return nullptr;
    char *json = QueryCanvasCapture(static_cast<int>(id));
    PyObject *module = PyImport_ImportModule("json");
    PyObject *result = module ? PyObject_CallMethod(module, "loads", "s", json) : nullptr;
    std::free(json);
    Py_XDECREF(module);
    return result;
}

PyObject *CancelCapture(PyObject *, PyObject *value)
{
    const long id = PyLong_AsLong(value);
    if (PyErr_Occurred()) return nullptr;
    return PyBool_FromLong(CancelCanvasCapture(static_cast<int>(id)));
}

infernux::web::WebParticleRuntime *g_particleRuntime = nullptr;
infernux::web::WebScreenUIRenderer *g_screenUIRenderer = nullptr;
infernux::web::WebSceneRenderer *g_sceneRenderer = nullptr;
infernux::web::WebGraphServices *g_graphServices = nullptr;
infernux::rhi::ComputeHost *g_computeHost = nullptr;
infernux::FrameWorkOwner *g_frameWork = nullptr;
bool g_textInputActive = false;

#if defined(INFERNUX_WEB_ENGINE_RUNTIME)
PyObject *CreateRenderTexture(PyObject *, PyObject *description)
{
    try {
        return pybind11::cast(g_sceneRenderer->CreateRenderTexture(
            pybind11::cast<const infernux::rhi::RenderTextureDesc &>(pybind11::handle(description)))).release().ptr();
    } catch (const std::exception &error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
        return nullptr;
    }
}

PyObject *LoadRenderTexture(PyObject *, PyObject *guid)
{
    try {
        return pybind11::cast(g_sceneRenderer->LoadRenderTexture(
            pybind11::cast<std::string>(pybind11::handle(guid)))).release().ptr();
    } catch (const std::exception &error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
        return nullptr;
    }
}

PyObject *AcquireComputeHost(PyObject *, PyObject *)
{
    try {
        if (g_computeHost == nullptr)
            throw std::runtime_error("WebGPU compute is not initialized");
        g_computeHost->RequireAlive();
        return pybind11::cast(std::make_unique<infernux::rhi::ComputeHost>(*g_computeHost)).release().ptr();
    } catch (const std::exception &error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
        return nullptr;
    }
}

PyObject *GetFrameWork(PyObject *, PyObject *)
{
    return pybind11::cast(g_frameWork, pybind11::return_value_policy::reference).release().ptr();
}

PyObject *CaptureGraphBuffer(PyObject *, PyObject *arguments)
{
    unsigned long long camera;
    const char *name;
    if (!PyArg_ParseTuple(arguments, "Ks:_capture_graph_buffer", &camera, &name))
        return nullptr;
    try {
        return pybind11::cast(g_graphServices->capture(camera, name)).release().ptr();
    } catch (const std::exception &error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
        return nullptr;
    }
}

PyObject *WatchGraphEvents(PyObject *, PyObject *value)
{
    try {
        const auto label = pybind11::cast<std::string>(pybind11::handle(value));
        auto &interest = g_graphServices->interests[label];
        auto active = interest.lock();
        if (!active) {
            active = std::make_shared<infernux::GraphEventInterest>();
            interest = active;
        }
        return pybind11::cast(std::move(active)).release().ptr();
    } catch (const std::exception &error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
        return nullptr;
    }
}

PyObject *TakeGraphEvents(PyObject *, PyObject *)
{
    infernux::GraphEventQueue events;
    events.splice(events.end(), g_graphServices->events);
    return pybind11::cast(std::move(events)).release().ptr();
}

PyObject *IsGraphSubmissionCompleted(PyObject *, PyObject *value)
{
    try {
        const auto [device, queue, serial] = pybind11::cast<
            std::tuple<infernux::rhi::DeviceId, uint8_t, infernux::rhi::SubmissionSerial>>(pybind11::handle(value));
        return PyBool_FromLong(g_computeHost->Queue().IsComplete(
            {device, static_cast<infernux::rhi::QueueRole>(queue), serial}));
    } catch (const std::exception &error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
        return nullptr;
    }
}

struct ScreenUITextureLoad
{
    infernux::web::WebTextureAssetLoad load;
    uint64_t textureId = 0;
};
std::unordered_map<std::string, ScreenUITextureLoad> g_screenUITextures;
#endif

PyObject *ReadPackageEntry(PyObject *, PyObject *arguments)
{
    const char *packagePath = nullptr;
    const char *entryPath = nullptr;
    if (!PyArg_ParseTuple(arguments, "ss:read_entry", &packagePath, &entryPath))
        return nullptr;

    try {
        const auto bytes = infernux::inxpack::ReadEntry(std::filesystem::path(packagePath), entryPath);
        return PyBytes_FromStringAndSize(reinterpret_cast<const char *>(bytes.data()),
                                         static_cast<Py_ssize_t>(bytes.size()));
    } catch (const std::exception &error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
        return nullptr;
    }
}

PyObject *ExtractPackage(PyObject *, PyObject *arguments)
{
    const char *packagePath = nullptr;
    const char *destinationPath = nullptr;
    if (!PyArg_ParseTuple(arguments, "ss:extract_package", &packagePath, &destinationPath))
        return nullptr;

    try {
        const auto manifest =
            infernux::inxpack::Extract(std::filesystem::path(packagePath), std::filesystem::path(destinationPath));
        PyObject *summary = PyDict_New();
        if (summary == nullptr)
            return nullptr;
        const auto setInteger = [summary](const char *name, uint64_t value) {
            PyObject *item = PyLong_FromUnsignedLongLong(value);
            if (item == nullptr)
                return false;
            const int result = PyDict_SetItemString(summary, name, item);
            Py_DECREF(item);
            return result == 0;
        };
        const bool complete = setInteger("entries", static_cast<uint64_t>(manifest.entries.size())) &&
                              setInteger("raw_bytes", manifest.rawBytes) &&
                              setInteger("stored_bytes", manifest.storedBytes) &&
                              setInteger("archive_bytes", manifest.archiveBytes);
        if (!complete) {
            Py_DECREF(summary);
            return nullptr;
        }
        return summary;
    } catch (const std::exception &error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
        return nullptr;
    }
}

PyObject *InitializeRuntimeAssets(PyObject *, PyObject *arguments)
{
#if !defined(INFERNUX_WEB_ENGINE_RUNTIME)
    PyErr_SetString(PyExc_RuntimeError, "the Web engine runtime is not linked");
    return nullptr;
#else
    const char *projectRoot = nullptr;
    const char *recordsPath = nullptr;
    if (!PyArg_ParseTuple(arguments, "ss:initialize_runtime_assets", &projectRoot, &recordsPath))
        return nullptr;

    try {
        auto &registry = infernux::AssetRegistry::Instance();
        if (registry.IsInitialized())
            throw std::logic_error("the Web runtime asset registry is already initialized");

        auto database = std::make_unique<infernux::AssetDatabase>();
        database->InitializeRuntime(projectRoot);
        database->InstallRuntimeAssetCatalog(recordsPath, true);
        const auto assetCount = database->GetAssetCount();
        registry.Initialize(std::move(database));
        registry.RegisterLoader(infernux::ResourceType::Shader, std::make_unique<infernux::ShaderLoader>());
        registry.RegisterLoader(infernux::ResourceType::Material, std::make_unique<infernux::MaterialLoader>());
        registry.RegisterLoader(infernux::ResourceType::PhysicMaterial,
                                std::make_unique<infernux::PhysicMaterialLoader>());
        registry.RegisterLoader(infernux::ResourceType::Texture, std::make_unique<infernux::TextureLoader>());
        registry.RegisterLoader(infernux::ResourceType::RenderTexture, std::make_unique<infernux::RenderTextureLoader>());
        registry.RegisterLoader(infernux::ResourceType::Mesh, std::make_unique<infernux::MeshLoader>());
        registry.RegisterLoader(infernux::ResourceType::Audio, std::make_unique<infernux::AudioClipLoader>());
        registry.RegisterLoader(infernux::ResourceType::Script, std::make_unique<infernux::InxPythonScriptLoader>());
        for (const auto type : {infernux::ResourceType::DefaultText, infernux::ResourceType::DefaultBinary,
                                infernux::ResourceType::RenderEffect, infernux::ResourceType::ParticleGraph,
                                infernux::ResourceType::DataAsset})
            registry.RegisterLoader(type, std::make_unique<infernux::InxDefaultLoader>(type));
        registry.PopulateAssetDatabaseLoaders();
        registry.InitializeBuiltinMaterials();
        return PyLong_FromSize_t(assetCount);
    } catch (const std::exception &error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
        return nullptr;
    }
#endif
}

PyObject *ConfigurePhysics(PyObject *, PyObject *arguments)
{
#if !defined(INFERNUX_WEB_ENGINE_RUNTIME)
    PyErr_SetString(PyExc_RuntimeError, "the Web engine runtime is not linked");
    return nullptr;
#else
    const char *documentText = nullptr;
    Py_ssize_t documentSize = 0;
    if (!PyArg_ParseTuple(arguments, "s#:configure_physics", &documentText, &documentSize))
        return nullptr;

    try {
        const auto document = nlohmann::json::parse(documentText, documentText + documentSize);
        const auto &gravity = document.at("gravity");
        if (!document.is_object() || !gravity.is_array() || gravity.size() != 3)
            throw std::invalid_argument("physics configuration has an invalid shape");

        const uint64_t temporaryBytes = document.at("temp_allocator_mb").get<uint64_t>() * 1024ULL * 1024ULL;
        if (temporaryBytes > std::numeric_limits<size_t>::max())
            throw std::out_of_range("physics temporary allocator exceeds the Web address space");

        auto &config = infernux::EngineConfig::Get();
        config.physicsTempAllocatorSize = static_cast<size_t>(temporaryBytes);
        config.physicsMaxJobs = document.at("max_jobs").get<uint32_t>();
        config.physicsMaxBarriers = document.at("max_barriers").get<uint32_t>();
        config.physicsMaxBodies = document.at("max_bodies").get<uint32_t>();
        config.physicsMaxBodyPairs = document.at("max_body_pairs").get<uint32_t>();
        config.physicsMaxContactConstraints = document.at("max_contact_constraints").get<uint32_t>();
        config.physicsMaxConcurrency = document.at("max_concurrency").get<uint32_t>();
        config.physicsCollisionSteps = document.at("collision_steps").get<int>();
        config.physicsVelocitySteps = document.at("velocity_steps").get<int>();
        config.physicsPositionSteps = document.at("position_steps").get<int>();
        config.physicsPenetrationSlop = document.at("penetration_slop").get<float>();
        config.physicsSpeculativeContactDistance = document.at("speculative_contact_distance").get<float>();
        config.physicsLinearCastMaxPenetration = document.at("linear_cast_max_penetration").get<float>();
        config.physicsBaumgarte = document.at("baumgarte").get<float>();
        config.physicsMaxPenetrationDistance = document.at("max_penetration_distance").get<float>();
        config.physicsLinearCastThreshold = document.at("linear_cast_threshold").get<float>();
        config.physicsMinVelocityForRestitution = document.at("min_velocity_for_restitution").get<float>();
        config.physicsTimeBeforeSleep = document.at("time_before_sleep").get<float>();
        config.physicsPointVelocitySleepThreshold = document.at("point_velocity_sleep_threshold").get<float>();
        config.physicsGravity = {gravity.at(0).get<float>(), gravity.at(1).get<float>(), gravity.at(2).get<float>()};

        auto &sceneManager = infernux::SceneManager::Instance();
        sceneManager.SetFixedTimeStep(document.at("fixed_delta_time").get<float>());
        sceneManager.SetMaxFixedDeltaTime(document.at("max_fixed_delta_time").get<float>());
        Py_RETURN_NONE;
    } catch (const std::exception &error) {
        PyErr_SetString(PyExc_ValueError, error.what());
        return nullptr;
    }
#endif
}

#if defined(INFERNUX_WEB_ENGINE_RUNTIME)
PyObject *ParticlePublicationResult(std::string error)
{
    using infernux::particle::GpuParticlePublicationStatus;
    try {
        const auto status =
            error.empty() ? GpuParticlePublicationStatus::Published : GpuParticlePublicationStatus::Failed;
        return pybind11::make_tuple(status, error).release().ptr();
    } catch (pybind11::error_already_set &failure) {
        failure.restore();
        return nullptr;
    } catch (const std::exception &failure) {
        PyErr_SetString(PyExc_RuntimeError, failure.what());
        return nullptr;
    }
}

PyObject *ReplaceGpuParticleGraph(PyObject *, PyObject *arguments)
{
    unsigned long long graphId = 0;
    PyObject *programs = nullptr;
    PyObject *removeIds = nullptr;
    if (!PyArg_ParseTuple(arguments, "KOO:_replace_gpu_particle_graph", &graphId, &programs, &removeIds))
        return nullptr;
    return ParticlePublicationResult(
        g_particleRuntime ? g_particleRuntime->ReplaceGraph(static_cast<uint64_t>(graphId), programs, removeIds)
                          : "WebGPU particle runtime is not initialized");
}

PyObject *ReplaceGpuParticleGraphs(PyObject *, PyObject *arguments)
{
    PyObject *graphs = nullptr;
    if (!PyArg_ParseTuple(arguments, "O:_replace_gpu_particle_graphs", &graphs))
        return nullptr;
    return ParticlePublicationResult(g_particleRuntime ? g_particleRuntime->ReplaceGraphs(graphs)
                                                       : "WebGPU particle runtime is not initialized");
}
#endif

PyObject *UpdateGpuParticleParameters(PyObject *, PyObject *arguments)
{
    unsigned long long graphId = 0;
    PyObject *updates = nullptr;
    if (!PyArg_ParseTuple(arguments, "KO:_update_gpu_particle_parameters", &graphId, &updates))
        return nullptr;
    const std::string error = g_particleRuntime
                                  ? g_particleRuntime->UpdateParameters(static_cast<uint64_t>(graphId), updates)
                                  : "WebGPU particle runtime is not initialized";
    return PyUnicode_FromString(error.c_str());
}

PyObject *BeginGpuParticleBatch(PyObject *, PyObject *arguments)
{
    unsigned long long graphId = 0;
    PyObject *items = nullptr;
    if (!PyArg_ParseTuple(arguments, "KO:_begin_gpu_particle_batch", &graphId, &items))
        return nullptr;
    return PyBool_FromLong(g_particleRuntime && g_particleRuntime->BeginBatch(static_cast<uint64_t>(graphId), items));
}

PyObject *SetGpuParticleEmitterPlaying(PyObject *, PyObject *arguments)
{
    unsigned long long emitterId = 0;
    int playing = 0;
    if (!PyArg_ParseTuple(arguments, "Kp:_set_gpu_particle_emitter_playing", &emitterId, &playing))
        return nullptr;
    return PyBool_FromLong(g_particleRuntime &&
                           g_particleRuntime->SetPlaying(static_cast<uint64_t>(emitterId), playing != 0));
}

PyObject *ResetGpuParticleEmitter(PyObject *, PyObject *arguments)
{
    unsigned long long emitterId = 0;
    if (!PyArg_ParseTuple(arguments, "K:_reset_gpu_particle_emitter", &emitterId))
        return nullptr;
    return PyBool_FromLong(g_particleRuntime && g_particleRuntime->Reset(static_cast<uint64_t>(emitterId)));
}

PyObject *GpuParticleArtifactRevision(PyObject *, PyObject *arguments)
{
    unsigned long long emitterId = 0;
    if (!PyArg_ParseTuple(arguments, "K:_gpu_particle_artifact_revision", &emitterId))
        return nullptr;
    return PyLong_FromUnsignedLongLong(
        g_particleRuntime ? g_particleRuntime->ArtifactRevision(static_cast<uint64_t>(emitterId)) : 0);
}

PyObject *GpuParticleStateWasPreserved(PyObject *, PyObject *arguments)
{
    unsigned long long emitterId = 0;
    if (!PyArg_ParseTuple(arguments, "K:_gpu_particle_state_was_preserved", &emitterId))
        return nullptr;
    return PyBool_FromLong(g_particleRuntime && g_particleRuntime->StateWasPreserved(static_cast<uint64_t>(emitterId)));
}

PyObject *BeginTextInput(PyObject *, PyObject *arguments)
{
    const char *initialValue = nullptr;
    const char *inputType = nullptr;
    if (!PyArg_ParseTuple(arguments, "ss:begin_text_input", &initialValue, &inputType))
        return nullptr;
    const bool started = infernux::InputManager::Instance().StartTextInput();
    if (!started)
        Py_RETURN_FALSE;
    EM_ASM(
        {
            if (Module.infernuxBeginTextInput)
                Module.infernuxBeginTextInput(UTF8ToString($0), UTF8ToString($1));
        },
        initialValue, inputType);
    g_textInputActive = true;
    Py_RETURN_TRUE;
}

PyObject *EndTextInput(PyObject *, PyObject *)
{
    InfernuxWebEndTextInput();
    Py_RETURN_NONE;
}

PyObject *IsTextInputActive(PyObject *, PyObject *)
{
    return PyBool_FromLong(g_textInputActive ? 1 : 0);
}

PyObject *ScreenUIBeginFrame(PyObject *, PyObject *arguments)
{
    unsigned int width = 0;
    unsigned int height = 0;
    if (!PyArg_ParseTuple(arguments, "II:screen_ui_begin_frame", &width, &height))
        return nullptr;
    if (g_screenUIRenderer)
        g_screenUIRenderer->BeginFrame(width, height);
    Py_RETURN_NONE;
}

PyObject *ScreenUIBeginFrameCached(PyObject *, PyObject *arguments)
{
    unsigned int width = 0;
    unsigned int height = 0;
    unsigned long long revision = 0;
    if (!PyArg_ParseTuple(arguments, "IIK:screen_ui_begin_frame_cached", &width, &height, &revision))
        return nullptr;
    return PyBool_FromLong(g_screenUIRenderer && g_screenUIRenderer->BeginFrameCached(width, height, revision));
}

PyObject *ScreenUISetMaterialBinding(PyObject *, PyObject *arguments)
{
    int list = 0;
    const char *guid = nullptr;
    unsigned long long generation = 0;
    const char *pipelineKey = nullptr;
    PyObject *colorObject = nullptr;
    int alphaClipEnabled = 0;
    double alphaClipThreshold = 0.0;
    if (!PyArg_ParseTuple(arguments, "isKsOpd:screen_ui_set_material_binding", &list, &guid, &generation, &pipelineKey,
                          &colorObject, &alphaClipEnabled, &alphaClipThreshold))
        return nullptr;
    PyObject *colors = PySequence_Fast(colorObject, "UI material color must have four components");
    if (!colors)
        return nullptr;
    if (PySequence_Fast_GET_SIZE(colors) != 4) {
        Py_DECREF(colors);
        PyErr_SetString(PyExc_ValueError, "UI material color must have four components");
        return nullptr;
    }
    std::array<float, 4> color{};
    for (int index = 0; index < 4; ++index) {
        color[static_cast<size_t>(index)] =
            static_cast<float>(PyFloat_AsDouble(PySequence_Fast_GET_ITEM(colors, index)));
    }
    Py_DECREF(colors);
    if (PyErr_Occurred())
        return nullptr;
    try {
        if (!g_screenUIRenderer)
            throw std::logic_error("Web UI material renderer is unavailable");
        g_screenUIRenderer->SetMaterialBinding(list, guid, generation, pipelineKey, color, alphaClipEnabled != 0,
                                               static_cast<float>(alphaClipThreshold));
        Py_RETURN_NONE;
    } catch (const std::exception &error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
        return nullptr;
    }
}

PyObject *ScreenUIBeginWorldElement(PyObject *, PyObject *arguments)
{
    PyObject *matrixObject = nullptr;
    double pivotX = 0.0;
    double pivotY = 0.0;
    unsigned int layer = 0;
    int alwaysOnTop = 0;
    int billboard = 0;
    int constantScreenSize = 0;
    unsigned long long ignoredOccluderId = 0;
    if (!PyArg_ParseTuple(arguments, "OddIpppK:screen_ui_begin_world_element", &matrixObject, &pivotX, &pivotY, &layer,
                          &alwaysOnTop, &billboard, &constantScreenSize, &ignoredOccluderId))
        return nullptr;
    PyObject *values = PySequence_Fast(matrixObject, "World UI transform must have 16 floats");
    if (!values)
        return nullptr;
    if (PySequence_Fast_GET_SIZE(values) != 16) {
        Py_DECREF(values);
        PyErr_SetString(PyExc_ValueError, "World UI transform must have 16 floats");
        return nullptr;
    }
    std::array<float, 16> matrix{};
    for (int index = 0; index < 16; ++index)
        matrix[static_cast<size_t>(index)] =
            static_cast<float>(PyFloat_AsDouble(PySequence_Fast_GET_ITEM(values, index)));
    Py_DECREF(values);
    if (PyErr_Occurred())
        return nullptr;
    for (float value : matrix) {
        if (!std::isfinite(value)) {
            PyErr_SetString(PyExc_ValueError, "World UI transform must be finite");
            return nullptr;
        }
    }
    try {
        if (!g_screenUIRenderer)
            throw std::logic_error("Web world UI renderer is unavailable");
        g_screenUIRenderer->BeginWorldElement(matrix, static_cast<float>(pivotX), static_cast<float>(pivotY), layer,
                                              alwaysOnTop != 0, billboard != 0, constantScreenSize != 0,
                                              ignoredOccluderId);
        Py_RETURN_NONE;
    } catch (const std::exception &error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
        return nullptr;
    }
}

PyObject *ScreenUIEndWorldElement(PyObject *, PyObject *)
{
    try {
        if (!g_screenUIRenderer)
            throw std::logic_error("Web world UI renderer is unavailable");
        g_screenUIRenderer->EndWorldElement();
        Py_RETURN_NONE;
    } catch (const std::exception &error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
        return nullptr;
    }
}

PyObject *ScreenUIPushClipRect(PyObject *, PyObject *arguments)
{
    int list = 0;
    double minX = 0.0, minY = 0.0, maxX = 0.0, maxY = 0.0;
    if (!PyArg_ParseTuple(arguments, "idddd:screen_ui_push_clip_rect", &list, &minX, &minY, &maxX, &maxY))
        return nullptr;
    if (g_screenUIRenderer)
        g_screenUIRenderer->PushClipRect(list, static_cast<float>(minX), static_cast<float>(minY),
                                         static_cast<float>(maxX), static_cast<float>(maxY));
    Py_RETURN_NONE;
}

PyObject *ScreenUIPopClipRect(PyObject *, PyObject *arguments)
{
    int list = 0;
    if (!PyArg_ParseTuple(arguments, "i:screen_ui_pop_clip_rect", &list))
        return nullptr;
    if (g_screenUIRenderer)
        g_screenUIRenderer->PopClipRect(list);
    Py_RETURN_NONE;
}

PyObject *ScreenUIAddFilledRect(PyObject *, PyObject *arguments)
{
    int list = 0;
    double values[10]{};
    int mirrorH = 0, mirrorV = 0;
    if (!PyArg_ParseTuple(arguments, "iddddddddddpp:screen_ui_add_filled_rect", &list, &values[0], &values[1],
                          &values[2], &values[3], &values[4], &values[5], &values[6], &values[7], &values[8],
                          &values[9], &mirrorH, &mirrorV))
        return nullptr;
    if (g_screenUIRenderer)
        g_screenUIRenderer->AddFilledRect(
            list, static_cast<float>(values[0]), static_cast<float>(values[1]), static_cast<float>(values[2]),
            static_cast<float>(values[3]), static_cast<float>(values[4]), static_cast<float>(values[5]),
            static_cast<float>(values[6]), static_cast<float>(values[7]), static_cast<float>(values[8]),
            static_cast<float>(values[9]), mirrorH != 0, mirrorV != 0);
    Py_RETURN_NONE;
}

double TupleNumber(PyObject *arguments, Py_ssize_t index)
{
    return PyFloat_AsDouble(PyTuple_GetItem(arguments, index));
}

bool ParseFallbackFontPaths(PyObject *value, std::vector<std::string> &paths)
{
    if (value == nullptr)
        return true;
    if (PyUnicode_Check(value) || PyBytes_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "fallback font paths must be a sequence of strings");
        return false;
    }
    PyObject *sequence = PySequence_Fast(value, "fallback font paths must be a sequence");
    if (sequence == nullptr)
        return false;
    const Py_ssize_t count = PySequence_Fast_GET_SIZE(sequence);
    if (count == 0) {
        Py_DECREF(sequence);
        return true;
    }
    paths.reserve(static_cast<size_t>(count));
    for (Py_ssize_t index = 0; index < count; ++index) {
        PyObject *item = PySequence_Fast_GET_ITEM(sequence, index);
        Py_ssize_t size = 0;
        const char *path = PyUnicode_AsUTF8AndSize(item, &size);
        if (path == nullptr) {
            Py_DECREF(sequence);
            return false;
        }
        if (size <= 0) {
            Py_DECREF(sequence);
            PyErr_SetString(PyExc_ValueError, "fallback font paths must not contain an empty entry");
            return false;
        }
        paths.emplace_back(path, static_cast<size_t>(size));
    }
    Py_DECREF(sequence);
    return true;
}

PyObject *ScreenUIAddImage(PyObject *, PyObject *arguments)
{
    if (PyTuple_Size(arguments) != 18) {
        PyErr_SetString(PyExc_TypeError, "screen_ui_add_image expects 18 arguments");
        return nullptr;
    }
    const int list = static_cast<int>(PyLong_AsLong(PyTuple_GetItem(arguments, 0)));
    const uint64_t texture = PyLong_AsUnsignedLongLong(PyTuple_GetItem(arguments, 1));
    double values[13]{};
    for (int index = 0; index < 13; ++index)
        values[index] = TupleNumber(arguments, index + 2);
    const bool mirrorH = PyObject_IsTrue(PyTuple_GetItem(arguments, 15)) != 0;
    const bool mirrorV = PyObject_IsTrue(PyTuple_GetItem(arguments, 16)) != 0;
    const double rounding = TupleNumber(arguments, 17);
    if (PyErr_Occurred())
        return nullptr;
    if (g_screenUIRenderer)
        g_screenUIRenderer->AddImage(
            list, texture, static_cast<float>(values[0]), static_cast<float>(values[1]), static_cast<float>(values[2]),
            static_cast<float>(values[3]), static_cast<float>(values[4]), static_cast<float>(values[5]),
            static_cast<float>(values[6]), static_cast<float>(values[7]), static_cast<float>(values[8]),
            static_cast<float>(values[9]), static_cast<float>(values[10]), static_cast<float>(values[11]),
            static_cast<float>(values[12]), mirrorH, mirrorV, static_cast<float>(rounding));
    Py_RETURN_NONE;
}

PyObject *ScreenUIAddText(PyObject *, PyObject *arguments)
{
    const Py_ssize_t argumentCount = PyTuple_Size(arguments);
    if (argumentCount < 20 || argumentCount > 22) {
        PyErr_SetString(PyExc_TypeError, "screen_ui_add_text expects 20-22 arguments");
        return nullptr;
    }
    const int list = static_cast<int>(PyLong_AsLong(PyTuple_GetItem(arguments, 0)));
    const char *text = PyUnicode_AsUTF8(PyTuple_GetItem(arguments, 5));
    const bool mirrorH = PyObject_IsTrue(PyTuple_GetItem(arguments, 15)) != 0;
    const bool mirrorV = PyObject_IsTrue(PyTuple_GetItem(arguments, 16)) != 0;
    const char *fontPath = PyUnicode_AsUTF8(PyTuple_GetItem(arguments, 17));
    const bool clip = argumentCount >= 21 && PyObject_IsTrue(PyTuple_GetItem(arguments, 20)) != 0;
    std::vector<std::string> fallbackFontPaths;
    double values[15]{};
    for (int index = 0; index < 4; ++index)
        values[index] = TupleNumber(arguments, index + 1);
    for (int index = 4; index < 13; ++index)
        values[index] = TupleNumber(arguments, index + 2);
    values[13] = TupleNumber(arguments, 18);
    values[14] = TupleNumber(arguments, 19);
    if (PyErr_Occurred() || !text || !fontPath ||
        (argumentCount == 22 && !ParseFallbackFontPaths(PyTuple_GetItem(arguments, 21), fallbackFontPaths)))
        return nullptr;
    if (g_screenUIRenderer)
        g_screenUIRenderer->AddText(
            list, static_cast<float>(values[0]), static_cast<float>(values[1]), static_cast<float>(values[2]),
            static_cast<float>(values[3]), text, static_cast<float>(values[4]), static_cast<float>(values[5]),
            static_cast<float>(values[6]), static_cast<float>(values[7]), static_cast<float>(values[8]),
            static_cast<float>(values[9]), static_cast<float>(values[10]), static_cast<float>(values[11]),
            static_cast<float>(values[12]), mirrorH, mirrorV, fontPath, static_cast<float>(values[13]),
            static_cast<float>(values[14]), clip, fallbackFontPaths);
    Py_RETURN_NONE;
}

#if defined(INFERNUX_WEB_ENGINE_RUNTIME)
PyObject *ScreenUIAddTextGeometry(PyObject *, PyObject *arguments)
{
    int list, mirrorH, mirrorV, clip;
    double minX, minY, maxX, maxY, r, g, b, a, alignX, alignY, scale, rotation, opacity;
    PyObject *object = nullptr;
    if (!PyArg_ParseTuple(arguments, "iddddOddddddddpppd:screen_ui_add_text_geometry", &list, &minX, &minY, &maxX,
                          &maxY, &object, &r, &g, &b, &a, &alignX, &alignY, &scale, &rotation, &mirrorH, &mirrorV,
                          &clip, &opacity))
        return nullptr;
    if (!g_screenUIRenderer) {
        PyErr_SetString(PyExc_RuntimeError, "Web UI renderer is not initialized");
        return nullptr;
    }
    try {
        auto &geometry = pybind11::cast<infernux::textlayout::TextGeometry &>(pybind11::handle(object));
        g_screenUIRenderer->AddTextGeometry(
            list, static_cast<float>(minX), static_cast<float>(minY), static_cast<float>(maxX),
            static_cast<float>(maxY), geometry, static_cast<float>(r), static_cast<float>(g), static_cast<float>(b),
            static_cast<float>(a), static_cast<float>(alignX), static_cast<float>(alignY), static_cast<float>(scale),
            static_cast<float>(rotation), mirrorH != 0, mirrorV != 0, clip != 0, static_cast<float>(opacity));
        Py_RETURN_NONE;
    } catch (const pybind11::cast_error &error) {
        PyErr_SetString(PyExc_TypeError, error.what());
    } catch (pybind11::error_already_set &error) {
        error.restore();
    } catch (const std::exception &error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
    }
    return nullptr;
}
#endif

PyObject *ScreenUISetDefaultFont(PyObject *, PyObject *arguments)
{
    const char *path = nullptr;
    const char *cjkPath = nullptr;
    if (!PyArg_ParseTuple(arguments, "ss:screen_ui_set_default_font", &path, &cjkPath))
        return nullptr;
    ImFont *font = infernux::textlayout::LoadEngineFont(path, {cjkPath}, 18.0f);
    if (!font) {
        PyErr_SetString(PyExc_RuntimeError, "Web Player bundled font could not be loaded");
        return nullptr;
    }
    infernux::textlayout::SetDefaultFont(font);
    ImGui::GetIO().FontDefault = font;
    Py_RETURN_NONE;
}

PyObject *ScreenUIMeasureText(PyObject *, PyObject *arguments)
{
    const char *text = nullptr;
    const char *fontPath = nullptr;
    double fontSize = 0.0;
    double wrapWidth = 0.0;
    double lineHeight = 0.0;
    double letterSpacing = 0.0;
    PyObject *fallbackFontPathsObject = nullptr;
    if (!PyArg_ParseTuple(arguments, "sddsdd|O:screen_ui_measure_text", &text, &fontSize, &wrapWidth, &fontPath,
                          &lineHeight, &letterSpacing, &fallbackFontPathsObject))
        return nullptr;
    std::vector<std::string> fallbackFontPaths;
    if (!ParseFallbackFontPaths(fallbackFontPathsObject, fallbackFontPaths))
        return nullptr;
    try {
        const auto measured =
            g_screenUIRenderer
                ? g_screenUIRenderer->MeasureText(text, static_cast<float>(fontSize), static_cast<float>(wrapWidth),
                                                  fontPath, static_cast<float>(lineHeight),
                                                  static_cast<float>(letterSpacing), fallbackFontPaths)
                : std::pair<float, float>{0.0f, 0.0f};
        return Py_BuildValue("ff", measured.first, measured.second);
    } catch (const std::exception &error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
        return nullptr;
    }
}

PyObject *ScreenUIResolveTexture(PyObject *, PyObject *arguments)
{
#if !defined(INFERNUX_WEB_ENGINE_RUNTIME)
    return PyLong_FromLongLong(-1);
#else
    const char *guidText = nullptr;
    if (!PyArg_ParseTuple(arguments, "s:screen_ui_resolve_texture", &guidText))
        return nullptr;
    if (!g_screenUIRenderer || guidText == nullptr || *guidText == '\0')
        return PyLong_FromLongLong(-1);
    uint64_t publishedTexture = 0;
    try {
        const std::string guid(guidText);
        auto &state = g_screenUITextures[guid];
        publishedTexture = state.textureId;
        using Result = infernux::web::WebTextureAssetLoad::Result;
        const auto status = state.load.Poll(guid);
        if (status == Result::Missing) {
            if (state.textureId) {
                g_screenUIRenderer->ReleaseTexture(state.textureId);
                state.textureId = 0;
            }
            return PyLong_FromLongLong(-1);
        }
        if (status == Result::Ready) {
            const auto data = state.load.TakeData();
            const uint64_t replacement = g_screenUIRenderer->UploadTexture(
                data.texture, state.textureId, data.filterMode, data.wrapMode, data.anisoLevel);
            if (replacement) {
                state.textureId = replacement;
                std::printf("INFERNUX_WEB_SCREEN_UI_TEXTURE_READY guid=%s id=%llu\n", guid.c_str(),
                            static_cast<unsigned long long>(state.textureId));
            } else {
                std::fprintf(stderr, "INFERNUX_WEB_SCREEN_UI_TEXTURE_UNSUPPORTED guid=%s format=%s\n", guid.c_str(),
                             infernux::TextureFormatName(data.texture.format));
            }
        }
        if (state.textureId)
            return PyLong_FromUnsignedLongLong(state.textureId);
        return PyLong_FromLongLong(status == Result::Pending ? 0 : -1);
    } catch (const std::exception &error) {
        std::fprintf(stderr, "INFERNUX_WEB_SCREEN_UI_TEXTURE_FAILED guid=%s error=%s\n", guidText, error.what());
        return publishedTexture ? PyLong_FromUnsignedLongLong(publishedTexture) : PyLong_FromLongLong(-1);
    }
#endif
}

PyObject *ScreenUIUploadImage(PyObject *, PyObject *arguments)
{
    Py_buffer encoded{};
    unsigned long long replaceTextureId = 0;
    if (!PyArg_ParseTuple(arguments, "y*|K:screen_ui_upload_image", &encoded, &replaceTextureId))
        return nullptr;
    if (!g_screenUIRenderer || encoded.len <= 0 || encoded.len > std::numeric_limits<int>::max()) {
        PyBuffer_Release(&encoded);
        PyErr_SetString(PyExc_ValueError, "Screen UI encoded image payload is unavailable or invalid");
        return nullptr;
    }

    int width = 0;
    int height = 0;
    int channels = 0;
    stbi_uc *decoded = stbi_load_from_memory(static_cast<const stbi_uc *>(encoded.buf), static_cast<int>(encoded.len),
                                             &width, &height, &channels, STBI_rgb_alpha);
    PyBuffer_Release(&encoded);
    std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> release(decoded, &stbi_image_free);
    if (!decoded || width <= 0 || height <= 0 ||
        static_cast<uint64_t>(width) * static_cast<uint64_t>(height) > std::numeric_limits<size_t>::max() / 4ULL) {
        PyErr_SetString(PyExc_ValueError, "Screen UI encoded image could not be decoded as RGBA8");
        return nullptr;
    }

    infernux::TextureCpuData texture;
    texture.dimension = infernux::TextureDimension::Texture2D;
    texture.semantic = infernux::TextureSemantic::UserInterface;
    texture.format = infernux::TextureFormat::Rgba8Srgb;
    infernux::TextureMipLevel mip;
    mip.width = static_cast<uint32_t>(width);
    mip.height = static_cast<uint32_t>(height);
    mip.depth = 1;
    mip.byteOffset = 0;
    mip.byteSize = static_cast<uint64_t>(width) * static_cast<uint64_t>(height) * 4ULL;
    mip.rowPitch = static_cast<uint64_t>(width) * 4ULL;
    mip.slicePitch = mip.byteSize;
    texture.mipLevels.push_back(mip);
    texture.bytes.assign(decoded, decoded + static_cast<size_t>(mip.byteSize));
    const uint64_t textureId = g_screenUIRenderer->UploadTexture(texture, static_cast<uint64_t>(replaceTextureId));
    if (textureId == 0) {
        PyErr_SetString(PyExc_RuntimeError, "Screen UI decoded image could not be uploaded to WebGPU");
        return nullptr;
    }
    return Py_BuildValue("(KII)", static_cast<unsigned long long>(textureId), static_cast<unsigned int>(width),
                         static_cast<unsigned int>(height));
}

PyObject *ScreenUIReleaseTexture(PyObject *, PyObject *arguments)
{
    unsigned long long textureId = 0;
    if (!PyArg_ParseTuple(arguments, "K:screen_ui_release_texture", &textureId))
        return nullptr;
    if (g_screenUIRenderer)
        g_screenUIRenderer->ReleaseTexture(static_cast<uint64_t>(textureId));
    Py_RETURN_NONE;
}

PyMethodDef kMethods[] = {
    {"request_capture", RequestCapture, METH_VARARGS, "Capture the next submitted Game canvas as PNG."},
    {"query_capture", QueryCapture, METH_O, "Poll PNG publication in the Player filesystem."},
    {"cancel_capture", CancelCapture, METH_O, "Cancel an unfinished Game capture."},
#if defined(INFERNUX_WEB_ENGINE_RUNTIME)
    {"_create_render_texture", CreateRenderTexture, METH_O, "Create a target through the shared RHI registry."},
    {"_load_render_texture", LoadRenderTexture, METH_O, "Acquire the shared imported target owner."},
    {"acquire_compute_host", AcquireComputeHost, METH_NOARGS, "Borrow the renderer's WebGPU compute services."},
    {"frame_work", GetFrameWork, METH_NOARGS, "Borrow the browser frame's native work owner."},
    {"_capture_graph_buffer", CaptureGraphBuffer, METH_VARARGS, nullptr},
    {"_watch_graph_events", WatchGraphEvents, METH_O, nullptr},
    {"_take_graph_events", TakeGraphEvents, METH_NOARGS, nullptr},
    {"_is_graph_submission_completed", IsGraphSubmissionCompleted, METH_O, nullptr},
#endif
    {"read_entry", ReadPackageEntry, METH_VARARGS,
     "Read and validate one entry from the native Infernux Player container."},
    {"extract_package", ExtractPackage, METH_VARARGS, "Validate and extract one native Infernux Player container."},
    {"initialize_runtime_assets", InitializeRuntimeAssets, METH_VARARGS,
     "Install the immutable cooked GUID catalog and runtime asset loaders."},
    {"configure_physics", ConfigurePhysics, METH_VARARGS,
     "Apply validated project physics settings before the Web physics world is created."},
#if defined(INFERNUX_WEB_ENGINE_RUNTIME)
    {"_replace_gpu_particle_graph", ReplaceGpuParticleGraph, METH_VARARGS, nullptr},
    {"_replace_gpu_particle_graphs", ReplaceGpuParticleGraphs, METH_VARARGS, nullptr},
#endif
    {"_update_gpu_particle_parameters", UpdateGpuParticleParameters, METH_VARARGS, nullptr},
    {"_begin_gpu_particle_batch", BeginGpuParticleBatch, METH_VARARGS, nullptr},
    {"_set_gpu_particle_emitter_playing", SetGpuParticleEmitterPlaying, METH_VARARGS, nullptr},
    {"_reset_gpu_particle_emitter", ResetGpuParticleEmitter, METH_VARARGS, nullptr},
    {"_gpu_particle_artifact_revision", GpuParticleArtifactRevision, METH_VARARGS, nullptr},
    {"_gpu_particle_state_was_preserved", GpuParticleStateWasPreserved, METH_VARARGS, nullptr},
    {"begin_text_input", BeginTextInput, METH_VARARGS, "Focus the browser text bridge and begin committed text input."},
    {"end_text_input", EndTextInput, METH_NOARGS, "End browser text input and dismiss the software keyboard."},
    {"is_text_input_active", IsTextInputActive, METH_NOARGS, "Return whether browser text input is active."},
    {"screen_ui_begin_frame", ScreenUIBeginFrame, METH_VARARGS, nullptr},
    {"screen_ui_begin_frame_cached", ScreenUIBeginFrameCached, METH_VARARGS, nullptr},
    {"screen_ui_set_material_binding", ScreenUISetMaterialBinding, METH_VARARGS, nullptr},
    {"screen_ui_begin_world_element", ScreenUIBeginWorldElement, METH_VARARGS, nullptr},
    {"screen_ui_end_world_element", ScreenUIEndWorldElement, METH_NOARGS, nullptr},
    {"screen_ui_push_clip_rect", ScreenUIPushClipRect, METH_VARARGS, nullptr},
    {"screen_ui_pop_clip_rect", ScreenUIPopClipRect, METH_VARARGS, nullptr},
    {"screen_ui_add_filled_rect", ScreenUIAddFilledRect, METH_VARARGS, nullptr},
    {"screen_ui_add_image", ScreenUIAddImage, METH_VARARGS, nullptr},
    {"screen_ui_add_text", ScreenUIAddText, METH_VARARGS, nullptr},
#if defined(INFERNUX_WEB_ENGINE_RUNTIME)
    {"screen_ui_add_text_geometry", ScreenUIAddTextGeometry, METH_VARARGS, nullptr},
#endif
    {"screen_ui_set_default_font", ScreenUISetDefaultFont, METH_VARARGS, nullptr},
    {"screen_ui_measure_text", ScreenUIMeasureText, METH_VARARGS, nullptr},
    {"screen_ui_resolve_texture", ScreenUIResolveTexture, METH_VARARGS, nullptr},
    {"screen_ui_upload_image", ScreenUIUploadImage, METH_VARARGS, nullptr},
    {"screen_ui_release_texture", ScreenUIReleaseTexture, METH_VARARGS, nullptr},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef kModule = {
    PyModuleDef_HEAD_INIT,
    "_InfernuxWebHost",
    "Web Player host services that do not belong to the gameplay API.",
    -1,
    kMethods,
};

} // namespace

PyMODINIT_FUNC PyInit__InfernuxWebHost()
{
    return PyModule_Create(&kModule);
}

void InfernuxWebSetParticleRuntime(infernux::web::WebParticleRuntime *runtime) noexcept
{
    g_particleRuntime = runtime;
}

void InfernuxWebPublishCaptureFrame(bool gameRendered, uint64_t frame, uint64_t device)
{
    PublishCanvasCaptures(gameRendered, static_cast<double>(frame), static_cast<double>(device));
}

void InfernuxWebSetScreenUIRenderer(infernux::web::WebScreenUIRenderer *renderer) noexcept
{
    g_screenUIRenderer = renderer;
}

void InfernuxWebSetSceneRenderer(infernux::web::WebSceneRenderer *renderer) noexcept
{
    g_sceneRenderer = renderer;
}

void InfernuxWebSetGraphServices(infernux::web::WebGraphServices *services) noexcept
{
    g_graphServices = services;
}

void InfernuxWebSetComputeHost(infernux::rhi::ComputeHost *host) noexcept
{
    g_computeHost = host;
}

void InfernuxWebSetFrameWork(infernux::FrameWorkOwner *owner) noexcept
{
    g_frameWork = owner;
}

void InfernuxWebEndTextInput() noexcept
{
    EM_ASM({
        if (Module.infernuxEndTextInput)
            Module.infernuxEndTextInput();
    });
    infernux::InputManager::Instance().StopTextInput();
    g_textInputActive = false;
}
