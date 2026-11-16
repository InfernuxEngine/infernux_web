#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <webgpu/webgpu_cpp.h>

#include <function/renderer/rhi/RhiDescriptors.h>

#include <core/threading/FrameWorkOwner.h>
#include <core/threading/JobSystem.h>
#include <core/types/ColorSpace.h>
#include <core/math/ViewportRect.h>
#include <function/audio/AudioEngine.h>
#include <platform/filesystem/InxPack.h>
#include <platform/input/InputManager.h>
#if defined(INFERNUX_WEB_ENGINE_RUNTIME)
#include <function/resources/AssetRegistry/AssetRegistry.h>
#include <function/scene/Camera.h>
#include <function/scene/Scene.h>
#include <function/scene/SceneManager.h>
#include <function/scene/physics/PhysicsWorld.h>
#endif

#include "InfernuxWebHostModule.h"
#include "WebGpuComputeQueue.h"
#include "WebGpuRhiDevice.h"
#include "WebKeyboard.h"
#include "WebParticleRuntime.h"
#include "WebRenderGraph.h"
#include <pybind11/pybind11.h>
#include <function/resources/ShaderAsset/FullscreenProgramAsset.h>
#include <function/renderer/rhi/RhiBuffer.h>
#include <function/renderer/rhi/RhiRenderTexture.h>
#include "WebSceneRenderer.h"
#include "WebScreenUIRenderer.h"
#include <function/renderer/rhi/RhiComputeHost.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <exception>
#include <limits>
#include <memory>
#include <string_view>
#include <unordered_map>

#include <emscripten.h>
#include <emscripten/html5.h>

extern "C" int InfernuxRegisterNumPyBuiltins(void);

// JavaScript bodies are not C++: clang-format splits strict equality tokens.
// clang-format off
EM_JS(double, InfernuxWebAcceptanceFixedDelta, (), {
    const clock = Module.infernuxAcceptanceClock;
    return clock && clock.enabled ? Number(clock.fixedDeltaSeconds) : 0.0;
});

EM_JS(double, InfernuxWebAcceptancePauseFrame, (), {
    const clock = Module.infernuxAcceptanceClock;
    return clock && clock.enabled ? Number(clock.pauseAfterFrame) : 0.0;
});

EM_JS(int, InfernuxWebForceFallbackAdapter, (), {
    return new URLSearchParams(globalThis.location.search).get("infernuxWebGpuAdapter") === "fallback" ? 1 : 0;
});

EM_JS(int, InfernuxWebFixedCanvasWidth, (), {
    const presentation = Module.infernuxPresentation;
    if (presentation.render_output.width > 0) return presentation.render_output.width;
    return presentation.mode === "windowed" ? presentation.width : 0;
});

EM_JS(int, InfernuxWebFixedCanvasHeight, (), {
    const presentation = Module.infernuxPresentation;
    return presentation.render_output.height || presentation.height;
});

// Use the Emscripten terminal error path so the host's onAbort handler ends
// loading, and startup cannot continue after a failed runtime contract.
EM_JS(void, InfernuxWebAbortStartup, (const char *reason), { abort(UTF8ToString(reason)); });

EM_JS(void, InfernuxWebStartRuntime, (), {
    // Only the suspended frame owns gameplay/rendering work. Schedule its
    // successor after it returns, including an explicit GPU/IO wait.
    const fail = error => abort(error instanceof Error ? error.message : String(error));
    const frame = async () => {
        try {
            await Module.ccall("infernux_web_frame", null, [], [], {async: true});
            requestAnimationFrame(frame);
        } catch (error) { fail(error); }
    };
    Promise.resolve()
        .then(() => Module.ccall("infernux_web_start", null, [], [], {async: true}))
        .then(() => requestAnimationFrame(frame))
        .catch(fail);
});
// clang-format on

#if defined(INFERNUX_WEB_ENGINE_RUNTIME)
PyMODINIT_FUNC PyInit__Infernux();
#endif

namespace
{

wgpu::Instance g_instance = wgpuCreateInstance(nullptr);
wgpu::Adapter g_adapter;
wgpu::Device g_device;
wgpu::Queue g_queue;
wgpu::Surface g_surface;
std::unique_ptr<infernux::web::WebGpuRhiDevice> g_rhi;
std::shared_ptr<infernux::rhi::ComputeHostServices> g_computeServices;
std::unique_ptr<infernux::web::WebGpuComputeQueue> g_computeQueue;
std::unique_ptr<infernux::rhi::ComputeHost> g_computeHost;
infernux::FrameWorkOwner g_frameWork;
infernux::web::WebSceneRenderer g_sceneRenderer;
infernux::web::WebParticleRuntime g_particleRuntime;
class FullscreenHost final : public infernux::FullscreenRendererHost
{
  public:
    infernux::rhi::Device &GetRhiDevice() noexcept override { return *g_rhi; }
    uint32_t GetFrameCount() const noexcept override { return 1; }
    uint32_t GetCurrentFrame() const noexcept override { return 0; }
    infernux::rhi::BindingLayoutHandle GetPerViewLayout() const noexcept override { return g_sceneRenderer.ViewLayout(); }
    std::shared_ptr<const infernux::FullscreenProgram> AcquireProgram(const std::string &name) override
    { return infernux::LoadFullscreenProgram(name, infernux::ShaderCodeTarget::Wgsl); }
};
std::shared_ptr<FullscreenHost> g_fullscreenHost;
infernux::FullscreenRenderer g_presentRenderer;
struct CameraGraph
{
    std::unique_ptr<infernux::web::WebRenderGraph> graph;
    std::shared_ptr<const infernux::RenderGraphDescription> source;
    std::shared_ptr<const infernux::RenderGraphDescription> description;
    uint64_t outputRevision = 0;
    uint64_t lastFrame = 0;
};
std::unordered_map<uint64_t, CameraGraph> g_cameraGraphs;
PyObject *g_renderPipelineOwner = nullptr;
infernux::RenderPipelineCallback *g_renderPipeline = nullptr;
uint32_t g_lastSampleCount = 1;
infernux::web::WebScreenUIRenderer g_screenUIRenderer;
bool g_particleRuntimeReady = false;
bool g_particleRenderingEnabledForDiagnostics = true;
bool g_webGpuValidationFailed = false;
uint32_t g_webGpuStorageBufferLimit = 8;
wgpu::TextureFormat g_surfaceFormat = wgpu::TextureFormat::Undefined;
PyObject *g_tick = nullptr;
PyObject *g_input = nullptr;
PyObject *g_activate = nullptr;
PyObject *g_runtimeDiagnostic = nullptr;
uint32_t g_width = 1;
uint32_t g_height = 1;
double g_cssWidth = 1.0;
double g_cssHeight = 1.0;
std::array<double, 32> g_gamepadTimestamps{};
std::unordered_map<int, std::pair<float, float>> g_pointerPositions;
double g_lastFrameTimeMilliseconds = 0.0;
float g_shaderTimeSeconds = 0.0f;
uint64_t g_shaderFrameIndex = 0;
bool g_runtimeFrameFailed = false;
bool g_viewportDirty = false;
bool g_splashActive = false;
double g_safeAreaLeft = 0.0;
double g_safeAreaTop = 0.0;
double g_safeAreaRight = 0.0;
double g_safeAreaBottom = 0.0;
double g_keyboardInset = 0.0;
bool g_keyboardInsetKnown = false;
bool g_runtimeActivated = false;
double g_acceptanceFixedDeltaSeconds = 0.0;
uint64_t g_acceptancePauseAfterFrame = 0;
bool g_acceptancePaused = false;

bool ConfigureAcceptanceClock()
{
    const double fixedDelta = InfernuxWebAcceptanceFixedDelta();
    const double pauseFrame = InfernuxWebAcceptancePauseFrame();
    if (fixedDelta == 0.0 && pauseFrame == 0.0)
        return true;
    if (!std::isfinite(fixedDelta) || fixedDelta <= 0.0 || fixedDelta > 0.25 || !std::isfinite(pauseFrame) ||
        pauseFrame < 1.0 || std::floor(pauseFrame) != pauseFrame ||
        pauseFrame > static_cast<double>(std::numeric_limits<uint64_t>::max())) {
        std::fprintf(stderr, "INFERNUX_WEB_ACCEPTANCE_CLOCK_INVALID fixed_delta=%g pause_frame=%g\n", fixedDelta,
                     pauseFrame);
        return false;
    }
    g_acceptanceFixedDeltaSeconds = fixedDelta;
    g_acceptancePauseAfterFrame = static_cast<uint64_t>(pauseFrame);
    std::printf("INFERNUX_WEB_ACCEPTANCE_CLOCK_READY fixed_delta=%g pause_frame=%llu\n", fixedDelta,
                static_cast<unsigned long long>(g_acceptancePauseAfterFrame));
    return true;
}

int BrowserButtonToUnityButton(unsigned short button)
{
    switch (button) {
    case 0:
        return 0;
    case 2:
        return 1;
    case 1:
        return 2;
    case 3:
        return 3;
    case 4:
        return 4;
    default:
        return -1;
    }
}

void PrintPythonError(const char *context)
{
    std::fprintf(stderr, "INFERNUX_WEB_PYTHON_ERROR context=%s\n", context);
    PyErr_Print();
}

bool VerifyPreloadedRuntime()
{
    static constexpr char pythonArchive[] = "/usr/local/lib/python313.zip";
    std::FILE *stream = std::fopen(pythonArchive, "rb");
    if (stream == nullptr) {
        std::fprintf(stderr, "INFERNUX_WEB_PYTHON_ARCHIVE_MISSING path=%s\n", pythonArchive);
        return false;
    }
    const int seekResult = std::fseek(stream, 0, SEEK_END);
    const long byteSize = seekResult == 0 ? std::ftell(stream) : -1;
    std::array<unsigned char, 4> signature{};
    const bool hasSignature = std::fseek(stream, 0, SEEK_SET) == 0 &&
                              std::fread(signature.data(), 1, signature.size(), stream) == signature.size();
    std::fclose(stream);
    const bool isZip =
        hasSignature && signature[0] == 'P' && signature[1] == 'K' && signature[2] == 3 && signature[3] == 4;
    if (byteSize <= 0 || !isZip) {
        std::fprintf(stderr, "INFERNUX_WEB_PYTHON_ARCHIVE_INVALID bytes=%ld zip=%d\n", byteSize, isZip ? 1 : 0);
        return false;
    }
    std::printf("INFERNUX_WEB_PYTHON_ARCHIVE_READY bytes=%ld\n", byteSize);
    return true;
}

bool VerifyNumericalRuntime()
{
    static constexpr char contract[] = "import os, sys\n"
                                       "import numpy as _infernux_numpy\n"
                                       "assert _infernux_numpy.__version__ == '2.2.5'\n"
                                       "assert sys.platform == 'emscripten'\n"
                                       "assert tuple(sys._emscripten_info.emscripten_version) == (4, 0, 10)\n"
                                       "assert os.uname().machine == 'wasm32'\n";
    if (PyRun_SimpleString(contract) != 0) {
        PrintPythonError("numerical-runtime");
        return false;
    }
    std::printf("INFERNUX_WEB_NUMPY_READY version=2.2.5 abi=cp313 platform=emscripten-4.0.10-wasm32\n");
    return true;
}

void SetDictString(PyObject *dictionary, const char *key, const char *value)
{
    PyObject *item = PyUnicode_FromString(value != nullptr ? value : "");
    if (item != nullptr) {
        PyDict_SetItemString(dictionary, key, item);
        Py_DECREF(item);
    }
}

void SetDictNumber(PyObject *dictionary, const char *key, double value)
{
    PyObject *item = PyFloat_FromDouble(value);
    if (item != nullptr) {
        PyDict_SetItemString(dictionary, key, item);
        Py_DECREF(item);
    }
}

void SetDictInteger(PyObject *dictionary, const char *key, long value)
{
    PyObject *item = PyLong_FromLong(value);
    if (item != nullptr) {
        PyDict_SetItemString(dictionary, key, item);
        Py_DECREF(item);
    }
}

void SetDictBool(PyObject *dictionary, const char *key, bool value)
{
    PyObject *item = PyBool_FromLong(value ? 1 : 0);
    if (item != nullptr) {
        PyDict_SetItemString(dictionary, key, item);
        Py_DECREF(item);
    }
}

void DispatchInput(const char *kind, PyObject *payload)
{
    if (g_input == nullptr || payload == nullptr)
        return;
    PyObject *result = PyObject_CallFunction(g_input, "sO", kind, payload);
    if (result == nullptr)
        PrintPythonError("input");
    else
        Py_DECREF(result);
}

extern "C" EMSCRIPTEN_KEEPALIVE void InfernuxWebUserActivation()
{
    if (g_runtimeActivated || g_activate == nullptr)
        return;

    auto &audio = infernux::AudioEngine::Instance();
    const bool audioReady = audio.IsInitialized() || audio.Initialize();
    if (!audioReady) {
        std::fprintf(stderr, "INFERNUX_WEB_AUDIO_INITIALIZATION_FAILED\n");
        return;
    }

    PyObject *argument = PyBool_FromLong(1);
    PyObject *result = argument != nullptr ? PyObject_CallOneArg(g_activate, argument) : nullptr;
    Py_XDECREF(argument);
    if (result == nullptr) {
        PrintPythonError("activation");
        return;
    }
    const int activated = PyObject_IsTrue(result);
    Py_DECREF(result);
    if (activated < 0) {
        PrintPythonError("activation-result");
        return;
    }
    if (activated == 0)
        return;

    g_runtimeActivated = true;
    std::printf("INFERNUX_WEB_AUDIO_READY sample_rate=%d channels=%d\n",
                infernux::AudioEngine::Instance().GetSampleRate(), infernux::AudioEngine::Instance().GetChannelCount());
    std::printf("INFERNUX_WEB_AUDIO_ACTIVE_VOICES count=%zu\n",
                infernux::AudioEngine::Instance().GetActiveVoiceCount());
}

void ResizeCanvas()
{
    double cssWidth = 0.0;
    double cssHeight = 0.0;
    emscripten_get_element_css_size("#canvas", &cssWidth, &cssHeight);
    g_cssWidth = std::max(1.0, cssWidth);
    g_cssHeight = std::max(1.0, cssHeight);
    const int fixedWidth = InfernuxWebFixedCanvasWidth();
    const double scale = fixedWidth > 0 ? static_cast<double>(fixedWidth) / g_cssWidth
                                        : std::max(1.0, emscripten_get_device_pixel_ratio());
    g_width =
        fixedWidth > 0 ? static_cast<uint32_t>(fixedWidth) : std::max(1u, static_cast<uint32_t>(cssWidth * scale));
    g_height = fixedWidth > 0 ? static_cast<uint32_t>(InfernuxWebFixedCanvasHeight())
                              : std::max(1u, static_cast<uint32_t>(cssHeight * scale));
    emscripten_set_canvas_element_size("#canvas", g_width, g_height);

    if (g_surface && g_device) {
        wgpu::SurfaceConfiguration config;
        config.device = g_device;
        config.format = g_surfaceFormat;
        config.usage = wgpu::TextureUsage::RenderAttachment;
        config.width = g_width;
        config.height = g_height;
        // A standalone Player owns the complete canvas, so transparency is
        // not part of its presentation contract.  Keep that contract
        // explicit and identical to the opaque desktop swapchains.
        config.alphaMode = wgpu::CompositeAlphaMode::Opaque;
        config.presentMode = wgpu::PresentMode::Fifo;
        g_surface.Configure(&config);
    }

    PyObject *payload = PyDict_New();
    SetDictInteger(payload, "width", g_width);
    SetDictInteger(payload, "height", g_height);
    SetDictNumber(payload, "pixel_ratio", scale);
    SetDictNumber(payload, "safe_left", g_safeAreaLeft);
    SetDictNumber(payload, "safe_top", g_safeAreaTop);
    SetDictNumber(payload, "safe_right", g_safeAreaRight);
    SetDictNumber(payload, "safe_bottom", g_safeAreaBottom);
    SetDictBool(payload, "keyboard_inset_known", g_keyboardInsetKnown);
    SetDictNumber(payload, "keyboard_inset", g_keyboardInset);
    DispatchInput("viewport", payload);
    Py_XDECREF(payload);
    infernux::InputManager::Instance().ProcessScreenMetrics(
        static_cast<int>(g_cssWidth), static_cast<int>(g_cssHeight), static_cast<int>(g_width),
        static_cast<int>(g_height), static_cast<float>(scale), static_cast<int>(g_safeAreaLeft),
        static_cast<int>(g_safeAreaTop), std::max(0, static_cast<int>(g_cssWidth - g_safeAreaLeft - g_safeAreaRight)),
        std::max(0, static_cast<int>(g_cssHeight - g_safeAreaTop - g_safeAreaBottom)), g_keyboardInsetKnown,
        static_cast<int>(g_keyboardInset));
}

extern "C" EMSCRIPTEN_KEEPALIVE void InfernuxWebResizeCanvas()
{
    g_viewportDirty = true;
}

extern "C" EMSCRIPTEN_KEEPALIVE void InfernuxWebSetRenderDiagnostic(int feature, int enabled)
{
    switch (feature) {
    case 0:
        g_sceneRenderer.SetSkyEnabledForDiagnostics(enabled != 0);
        break;
    case 1:
        g_sceneRenderer.SetShadowsEnabledForDiagnostics(enabled != 0);
        break;
    case 2:
        g_particleRenderingEnabledForDiagnostics = enabled != 0;
        break;
    default:
        std::fprintf(stderr, "INFERNUX_WEB_RENDER_DIAGNOSTIC_INVALID feature=%d\n", feature);
        break;
    }
}

extern "C" EMSCRIPTEN_KEEPALIVE void InfernuxWebViewportChanged(double safeLeft, double safeTop, double safeRight,
                                                                double safeBottom, int keyboardInsetKnown,
                                                                double keyboardInset)
{
    g_safeAreaLeft = std::max(0.0, safeLeft);
    g_safeAreaTop = std::max(0.0, safeTop);
    g_safeAreaRight = std::max(0.0, safeRight);
    g_safeAreaBottom = std::max(0.0, safeBottom);
    g_keyboardInsetKnown = keyboardInsetKnown != 0;
    g_keyboardInset = g_keyboardInsetKnown ? std::max(0.0, keyboardInset) : 0.0;
    g_viewportDirty = true;
}

extern "C" EMSCRIPTEN_KEEPALIVE void InfernuxWebPageLifecycle(int active)
{
    auto &input = infernux::InputManager::Instance();
    input.ProcessFocusEvent(active != 0);
    if (!active) {
        g_pointerPositions.clear();
        InfernuxWebEndTextInput();
        infernux::AudioEngine::Instance().PauseAll();
    } else if (g_runtimeActivated) {
        infernux::AudioEngine::Instance().ResumeAll();
    }
    PyObject *payload = PyDict_New();
    SetDictBool(payload, "active", active != 0);
    DispatchInput(active != 0 ? "page_show" : "page_hide", payload);
    Py_DECREF(payload);
}

EM_BOOL OnResize(int, const EmscriptenUiEvent *, void *)
{
    g_viewportDirty = true;
    return EM_TRUE;
}

EM_BOOL OnFocus(int eventType, const EmscriptenFocusEvent *, void *)
{
    infernux::InputManager::Instance().ProcessFocusEvent(eventType == EMSCRIPTEN_EVENT_FOCUS);
    if (eventType != EMSCRIPTEN_EVENT_FOCUS) {
        g_pointerPositions.clear();
        InfernuxWebEndTextInput();
    }
    PyObject *payload = PyDict_New();
    SetDictBool(payload, "focused", eventType == EMSCRIPTEN_EVENT_FOCUS);
    DispatchInput(eventType == EMSCRIPTEN_EVENT_FOCUS ? "focus" : "blur", payload);
    Py_DECREF(payload);
    return EM_FALSE;
}

EM_BOOL OnVisibility(int, const EmscriptenVisibilityChangeEvent *event, void *)
{
    if (event->hidden) {
        infernux::InputManager::Instance().ProcessFocusEvent(false);
        g_pointerPositions.clear();
        InfernuxWebEndTextInput();
        infernux::AudioEngine::Instance().PauseAll();
    } else if (g_runtimeActivated) {
        infernux::AudioEngine::Instance().ResumeAll();
    }
    PyObject *payload = PyDict_New();
    SetDictBool(payload, "hidden", event->hidden != 0);
    SetDictInteger(payload, "state", event->visibilityState);
    DispatchInput("visibility", payload);
    Py_DECREF(payload);
    return EM_FALSE;
}

EM_BOOL OnKey(int eventType, const EmscriptenKeyboardEvent *event, void *)
{
    if (eventType == EMSCRIPTEN_EVENT_KEYDOWN && !event->repeat)
        InfernuxWebUserActivation();
    infernux::InputManager::Instance().ProcessKeyEvent(infernux::web::BrowserCodeToScancode(event->code),
                                                       eventType == EMSCRIPTEN_EVENT_KEYDOWN);
    PyObject *payload = PyDict_New();
    SetDictString(payload, "key", event->key);
    SetDictString(payload, "code", event->code);
    SetDictBool(payload, "repeat", event->repeat != 0);
    SetDictBool(payload, "ctrl", event->ctrlKey != 0);
    SetDictBool(payload, "shift", event->shiftKey != 0);
    SetDictBool(payload, "alt", event->altKey != 0);
    SetDictBool(payload, "meta", event->metaKey != 0);
    DispatchInput(eventType == EMSCRIPTEN_EVENT_KEYDOWN ? "key_down" : "key_up", payload);
    Py_XDECREF(payload);
    return EM_TRUE;
}

extern "C" EMSCRIPTEN_KEEPALIVE int InfernuxWebGetKeyState(int scancode)
{
    return infernux::InputManager::Instance().GetKey(scancode) ? 1 : 0;
}

extern "C" EMSCRIPTEN_KEEPALIVE double InfernuxWebGetObjectPositionAxis(const char *name, int axis)
{
#if defined(INFERNUX_WEB_ENGINE_RUNTIME)
    if (name == nullptr || axis < 0 || axis > 2)
        return std::numeric_limits<double>::quiet_NaN();

    auto readScene = [&](infernux::Scene *scene) -> double {
        if (scene == nullptr)
            return std::numeric_limits<double>::quiet_NaN();
        for (infernux::GameObject *object : scene->GetAllObjects()) {
            if (object == nullptr || object->GetName() != name)
                continue;
            const glm::vec3 position = object->GetTransform()->GetWorldPosition();
            return static_cast<double>(position[axis]);
        }
        return std::numeric_limits<double>::quiet_NaN();
    };

    auto &sceneManager = infernux::SceneManager::Instance();
    const double activeValue = readScene(sceneManager.GetActiveScene());
    if (!std::isnan(activeValue))
        return activeValue;
    return readScene(sceneManager.GetRuntimePersistentScene());
#else
    (void)name;
    (void)axis;
    return std::numeric_limits<double>::quiet_NaN();
#endif
}

extern "C" EMSCRIPTEN_KEEPALIVE double InfernuxWebGetRuntimeDiagnostic(int probe, int argument)
{
    if (probe == 15)
        return static_cast<double>(g_lastSampleCount);
    if (g_runtimeDiagnostic == nullptr)
        return std::numeric_limits<double>::quiet_NaN();
    PyObject *result = PyObject_CallFunction(g_runtimeDiagnostic, "ii", probe, argument);
    if (result == nullptr) {
        PrintPythonError("runtime-diagnostic");
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double value = PyFloat_AsDouble(result);
    Py_DECREF(result);
    if (PyErr_Occurred()) {
        PrintPythonError("runtime-diagnostic-value");
        return std::numeric_limits<double>::quiet_NaN();
    }
    return value;
}

extern "C" EMSCRIPTEN_KEEPALIVE double InfernuxWebGetAcceptanceFrame()
{
#if defined(INFERNUX_WEB_ENGINE_RUNTIME)
    return static_cast<double>(infernux::SceneManager::Instance().GetRuntimeFrameCount());
#else
    return 0.0;
#endif
}

extern "C" EMSCRIPTEN_KEEPALIVE int InfernuxWebGetAcceptancePaused()
{
    return g_acceptancePaused ? 1 : 0;
}

EM_BOOL OnWheel(int, const EmscriptenWheelEvent *event, void *)
{
    infernux::InputManager::Instance().ProcessScrollEvent(static_cast<float>(event->deltaX),
                                                          static_cast<float>(-event->deltaY));
    PyObject *payload = PyDict_New();
    SetDictNumber(payload, "delta_x", event->deltaX);
    SetDictNumber(payload, "delta_y", event->deltaY);
    SetDictInteger(payload, "delta_mode", event->deltaMode);
    DispatchInput("wheel", payload);
    Py_XDECREF(payload);
    return EM_TRUE;
}

extern "C" EMSCRIPTEN_KEEPALIVE void InfernuxWebPointerEvent(int eventKind, int pointerId, int pointerType,
                                                             double xPixels, double yPixels, double movementXPixels,
                                                             double movementYPixels, double pressure,
                                                             double contactWidthPixels, double contactHeightPixels,
                                                             int button, int buttons, int isPrimary)
{
    if (eventKind == 0)
        InfernuxWebUserActivation();
    auto &input = infernux::InputManager::Instance();
    const float normalizedX = static_cast<float>(xPixels / g_cssWidth);
    const float normalizedY = static_cast<float>(yPixels / g_cssHeight);
    const auto previous = g_pointerPositions.find(pointerId);
    const float deltaX = previous == g_pointerPositions.end() ? 0.0f : normalizedX - previous->second.first;
    const float deltaY = previous == g_pointerPositions.end() ? 0.0f : normalizedY - previous->second.second;
    const bool terminal = eventKind == 2 || eventKind == 3;

    if (pointerType == 0) {
        const double surfaceScaleX = static_cast<double>(g_width) / g_cssWidth;
        const double surfaceScaleY = static_cast<double>(g_height) / g_cssHeight;
        input.ProcessPointerMotionEvent(
            static_cast<float>(xPixels * surfaceScaleX), static_cast<float>(yPixels * surfaceScaleY),
            static_cast<float>(movementXPixels * surfaceScaleX), static_cast<float>(movementYPixels * surfaceScaleY));
        const int mappedButton = BrowserButtonToUnityButton(static_cast<unsigned short>(button));
        if (mappedButton >= 0 && (eventKind == 0 || terminal))
            input.ProcessPointerButtonEvent(mappedButton, eventKind == 0);
    } else {
        infernux::TouchPhase phase = infernux::TouchPhase::Moved;
        if (eventKind == 0)
            phase = infernux::TouchPhase::Began;
        else if (eventKind == 2)
            phase = infernux::TouchPhase::Ended;
        else if (eventKind == 3)
            phase = infernux::TouchPhase::Canceled;
        input.ProcessTouchEvent(static_cast<uint64_t>(pointerType), static_cast<uint64_t>(pointerId),
                                static_cast<uint64_t>(emscripten_get_now() * 1000000.0), 0, normalizedX, normalizedY,
                                deltaX, deltaY, static_cast<float>(std::clamp(pressure, 0.0, 1.0)), phase,
                                static_cast<float>(contactWidthPixels / g_cssWidth),
                                static_cast<float>(contactHeightPixels / g_cssHeight), isPrimary != 0);
        if (terminal) {
            const char *phaseName = eventKind == 3 ? "canceled" : "ended";
            std::printf("INFERNUX_WEB_TOUCH_END finger=%d phase=%s\n", pointerId, phaseName);
        }
    }

    if (terminal)
        g_pointerPositions.erase(pointerId);
    else
        g_pointerPositions[pointerId] = {normalizedX, normalizedY};

    PyObject *payload = PyDict_New();
    SetDictInteger(payload, "id", pointerId);
    SetDictInteger(payload, "pointer_type", pointerType);
    SetDictNumber(payload, "x", xPixels);
    SetDictNumber(payload, "y", yPixels);
    SetDictNumber(payload, "normalized_x", normalizedX);
    SetDictNumber(payload, "normalized_y", normalizedY);
    SetDictNumber(payload, "movement_x", movementXPixels);
    SetDictNumber(payload, "movement_y", movementYPixels);
    SetDictNumber(payload, "pressure", pressure);
    SetDictNumber(payload, "contact_width", contactWidthPixels);
    SetDictNumber(payload, "contact_height", contactHeightPixels);
    SetDictInteger(payload, "button", button);
    SetDictInteger(payload, "buttons", buttons);
    SetDictBool(payload, "primary", isPrimary != 0);
    static constexpr const char *eventNames[] = {"pointer_down", "pointer_move", "pointer_up", "pointer_cancel"};
    DispatchInput(eventKind >= 0 && eventKind < 4 ? eventNames[eventKind] : "pointer_move", payload);
    Py_DECREF(payload);
}

void PollGamepads()
{
    if (emscripten_sample_gamepad_data() != EMSCRIPTEN_RESULT_SUCCESS)
        return;
    const int count = std::min(emscripten_get_num_gamepads(), static_cast<int>(g_gamepadTimestamps.size()));
    for (int index = 0; index < count; ++index) {
        EmscriptenGamepadEvent event{};
        if (emscripten_get_gamepad_status(index, &event) != EMSCRIPTEN_RESULT_SUCCESS || !event.connected)
            continue;
        if (event.timestamp == g_gamepadTimestamps[static_cast<size_t>(index)])
            continue;
        g_gamepadTimestamps[static_cast<size_t>(index)] = event.timestamp;

        PyObject *payload = PyDict_New();
        SetDictInteger(payload, "index", event.index);
        SetDictString(payload, "id", event.id);
        SetDictString(payload, "mapping", event.mapping);
        PyObject *axes = PyList_New(event.numAxes);
        for (int axis = 0; axis < event.numAxes; ++axis)
            PyList_SetItem(axes, axis, PyFloat_FromDouble(event.axis[axis]));
        PyObject *buttons = PyList_New(event.numButtons);
        for (int button = 0; button < event.numButtons; ++button) {
            PyObject *state = PyDict_New();
            SetDictNumber(state, "value", event.analogButton[button]);
            SetDictBool(state, "pressed", event.digitalButton[button] != 0);
            PyList_SetItem(buttons, button, state);
        }
        PyDict_SetItemString(payload, "axes", axes);
        PyDict_SetItemString(payload, "buttons", buttons);
        Py_DECREF(axes);
        Py_DECREF(buttons);
        DispatchInput("gamepad", payload);
        Py_DECREF(payload);
    }
}

infernux::rhi::PixelFormat ToRhiFormat(wgpu::TextureFormat format)
{
    switch (format) {
    case wgpu::TextureFormat::RGBA8Unorm:
        return infernux::rhi::PixelFormat::RGBA8UNorm;
    case wgpu::TextureFormat::RGBA8UnormSrgb:
        return infernux::rhi::PixelFormat::RGBA8Srgb;
    case wgpu::TextureFormat::BGRA8Unorm:
        return infernux::rhi::PixelFormat::BGRA8UNorm;
    case wgpu::TextureFormat::BGRA8UnormSrgb:
        return infernux::rhi::PixelFormat::BGRA8Srgb;
    default:
        return infernux::rhi::PixelFormat::Undefined;
    }
}

bool CreateRhiServices()
{
    if (ToRhiFormat(g_surfaceFormat) == infernux::rhi::PixelFormat::Undefined)
        return false;

    g_rhi = std::make_unique<infernux::web::WebGpuRhiDevice>(g_device, g_queue, g_webGpuStorageBufferLimit);
    g_computeServices = std::make_shared<infernux::rhi::ComputeHostServices>();
    g_computeServices->alive = true;
    g_computeQueue = std::make_unique<infernux::web::WebGpuComputeQueue>(*g_rhi);
    g_computeHost = std::make_unique<infernux::rhi::ComputeHost>(*g_rhi, *g_computeQueue, g_computeServices);
    InfernuxWebSetComputeHost(g_computeHost.get());
    InfernuxWebSetFrameWork(&g_frameWork);
    auto &sceneManager = infernux::SceneManager::Instance();
    sceneManager.EnsureAuthoringWorld();
    sceneManager.BindRuntimeFrameWork(g_frameWork);
    auto &physics = infernux::PhysicsWorld::Instance();
    physics.BindGpuServices(std::make_unique<infernux::rhi::ComputeHost>(*g_computeHost), &g_frameWork);
    physics.Initialize();
    std::printf("INFERNUX_WEB_PHYSICS_READY\n");
    return true;
}

extern "C" EMSCRIPTEN_KEEPALIVE void InfernuxWebTextInput(const char *text)
{
    infernux::InputManager::Instance().ProcessTextInputEvent(text != nullptr ? text : "");
    PyObject *payload = PyDict_New();
    SetDictString(payload, "text", text);
    DispatchInput("text_input", payload);
    Py_DECREF(payload);
}

bool InitializePython()
{
    if (!VerifyPreloadedRuntime())
        return false;
#if defined(INFERNUX_WEB_ENGINE_RUNTIME)
    if (!infernux::JobSystem::IsAvailable())
        infernux::JobSystem::InitializeInline();
#endif
    if (InfernuxRegisterNumPyBuiltins() != 0) {
        std::fprintf(stderr, "INFERNUX_WEB_NUMPY_REGISTRATION_FAILED\n");
        return false;
    }
    if (PyImport_AppendInittab("_InfernuxWebHost", &PyInit__InfernuxWebHost) == -1) {
        std::fprintf(stderr, "INFERNUX_WEB_HOST_MODULE_REGISTRATION_FAILED\n");
        return false;
    }
#if defined(INFERNUX_WEB_ENGINE_RUNTIME)
    if (PyImport_AppendInittab("_Infernux", &PyInit__Infernux) == -1) {
        std::fprintf(stderr, "INFERNUX_WEB_NATIVE_MODULE_REGISTRATION_FAILED\n");
        return false;
    }
#endif
    Py_Initialize();
    if (!Py_IsInitialized())
        return false;
    if (!VerifyNumericalRuntime())
        return false;
#if defined(INFERNUX_WEB_ENGINE_RUNTIME)
    if (PyRun_SimpleString("import _Infernux\n"
                           "assert _Infernux.__runtime_profile__ == 'web-player'\n"
                           "print('INFERNUX_WEB_NATIVE_MODULE_READY profile=web-player')") != 0) {
        PrintPythonError("native-module");
        return false;
    }
#endif
    if (PyRun_SimpleString("exec(open('/infernux/bootstrap.py', encoding='utf-8').read(), globals())") != 0) {
        PrintPythonError("bootstrap");
        return false;
    }
    PyObject *mainModule = PyImport_AddModule("__main__");
#if defined(INFERNUX_WEB_ENGINE_RUNTIME)
    PyObject *configurePhysics = PyObject_GetAttrString(mainModule, "infernux_web_configure_physics");
    if (configurePhysics == nullptr || !PyCallable_Check(configurePhysics)) {
        Py_XDECREF(configurePhysics);
        PrintPythonError("physics-configuration-contract");
        return false;
    }
    PyObject *physicsConfiguration = PyObject_CallNoArgs(configurePhysics);
    Py_DECREF(configurePhysics);
    if (physicsConfiguration == nullptr) {
        PrintPythonError("physics-configuration");
        return false;
    }
    const int physicsConfigured = PyObject_IsTrue(physicsConfiguration);
    Py_DECREF(physicsConfiguration);
    if (physicsConfigured != 1) {
        std::fprintf(stderr, "INFERNUX_WEB_PHYSICS_CONFIGURATION_FAILED\n");
        return false;
    }
#endif
    g_tick = PyObject_GetAttrString(mainModule, "infernux_web_tick");
    g_input = PyObject_GetAttrString(mainModule, "infernux_web_input");
    g_activate = PyObject_GetAttrString(mainModule, "infernux_web_activate");
    g_runtimeDiagnostic = PyObject_GetAttrString(mainModule, "infernux_web_runtime_diagnostic");
    return g_tick != nullptr && PyCallable_Check(g_tick) && g_input != nullptr && PyCallable_Check(g_input) &&
           g_activate != nullptr && PyCallable_Check(g_activate) && g_runtimeDiagnostic != nullptr &&
           PyCallable_Check(g_runtimeDiagnostic);
}

void Frame()
{
    g_frameWork.BeginOpportunity();
    infernux::SceneManager::Instance().BindRuntimeFrameWork(g_frameWork);
    g_computeQueue->Collect();
    if (g_viewportDirty) {
        g_viewportDirty = false;
        ResizeCanvas();
    }
    const double frameTimeMilliseconds = emscripten_get_now();
    const double wallDeltaSeconds =
        g_lastFrameTimeMilliseconds > 0.0 ? (frameTimeMilliseconds - g_lastFrameTimeMilliseconds) / 1000.0 : 0.0;
    g_lastFrameTimeMilliseconds = frameTimeMilliseconds;
    const bool acceptanceClockEnabled = g_acceptancePauseAfterFrame > 0;
    const double deltaSeconds =
        acceptanceClockEnabled ? (g_acceptancePaused ? 0.0 : g_acceptanceFixedDeltaSeconds) : wallDeltaSeconds;
    bool pauseAcceptanceAfterRender = false;
    g_shaderTimeSeconds += float(deltaSeconds);
    const auto shaderFrame = infernux::BuildEngineFrameGlobals(g_shaderTimeSeconds, float(deltaSeconds),
                                                               float(deltaSeconds), g_shaderFrameIndex++);
    g_screenUIRenderer.StageFrame(shaderFrame, g_width, g_height);
    g_sceneRenderer.StageFrame(shaderFrame);
#if defined(INFERNUX_WEB_ENGINE_RUNTIME)
    auto &registry = infernux::AssetRegistry::Instance();
    if (registry.IsInitialized())
        registry.PollTextureReads();
    if (infernux::JobSystem::IsAvailable())
        infernux::JobSystem::Get().RunPendingJobs(64);
#endif
    PollGamepads();
    if (g_tick != nullptr && !g_runtimeFrameFailed && !g_acceptancePaused) {
        PyObject *result = PyObject_CallFunction(g_tick, "d", deltaSeconds);
        if (result == nullptr) {
            PrintPythonError("frame");
            g_runtimeFrameFailed = true;
        } else {
            const int splashActive = PyObject_IsTrue(result);
            if (splashActive < 0) {
                PrintPythonError("frame-result");
                g_runtimeFrameFailed = true;
            } else
                g_splashActive = splashActive == 1;
            Py_DECREF(result);
        }
    }
#if defined(INFERNUX_WEB_ENGINE_RUNTIME)
    g_frameWork.AdvanceEarly();
    if (!g_runtimeFrameFailed && !g_acceptancePaused && !g_splashActive) {
        auto &sceneManager = infernux::SceneManager::Instance();
        // SceneManager applies the project's maximum delta and time scale.
        // A second browser-specific cap would change authored simulation time.
        const float delta = static_cast<float>(deltaSeconds);
        sceneManager.ServiceRuntimeFrames(g_frameWork, delta, acceptanceClockEnabled);
        if (acceptanceClockEnabled && sceneManager.IsPlaying()) {
            const uint64_t frame = sceneManager.GetRuntimeFrameCount();
            if (frame > g_acceptancePauseAfterFrame) {
                std::fprintf(stderr, "INFERNUX_WEB_ACCEPTANCE_CLOCK_OVERRUN requested=%llu actual=%llu\n",
                             static_cast<unsigned long long>(g_acceptancePauseAfterFrame),
                             static_cast<unsigned long long>(frame));
                g_runtimeFrameFailed = true;
            } else if (frame == g_acceptancePauseAfterFrame) {
                pauseAcceptanceAfterRender = true;
            }
        }
    }
#endif
    wgpu::SurfaceTexture surfaceTexture;
    g_surface.GetCurrentTexture(&surfaceTexture);
    bool foregroundSubmitted = false;
    if (surfaceTexture.texture) {
        wgpu::TextureView view = surfaceTexture.texture.CreateView();
        auto encoder = g_device.CreateCommandEncoder();
        if (!g_splashActive && g_particleRuntimeReady && !g_particleRuntime.RecordCompute(encoder))
            throw std::runtime_error(g_particleRuntime.LastError());
        const auto present = [&](const infernux::web::WebRenderGraph &graph, bool displayEncoded,
                                 const infernux::FullscreenViewResources &resources, wgpu::TextureView destination,
                                 infernux::rhi::PixelFormat format, const glm::uvec4 &rect, bool clear) {
            wgpu::RenderPassColorAttachment attachment{};
            attachment.view = destination;
            attachment.loadOp = clear ? wgpu::LoadOp::Clear : wgpu::LoadOp::Load;
            attachment.storeOp = wgpu::StoreOp::Store;
            wgpu::RenderPassDescriptor description{};
            description.colorAttachmentCount = 1;
            description.colorAttachments = &attachment;
            auto pass = encoder.BeginRenderPass(&description);
            pass.SetViewport(rect.x, rect.y, rect.z, rect.w, 0, 1);
            pass.SetScissorRect(rect.x, rect.y, rect.z, rect.w);
            infernux::FullscreenPipelineKey key;
            key.shaderName = displayEncoded ? "infernux/internal/fullscreen_blit" : "infernux/internal/display_encode";
            key.program = g_presentRenderer.AcquireProgram(key.shaderName);
            key.colorFormat = format;
            key.useDynamicRendering = true;
            const auto &pipeline = g_presentRenderer.EnsurePipeline(key);
            const auto values = infernux::FullscreenRenderer::PrepareParameters(*key.program, displayEncoded
                ? std::vector<std::pair<std::string, infernux::ShaderParameterValue>>{}
                : std::vector<std::pair<std::string, infernux::ShaderParameterValue>>{
                    {"target_is_srgb", infernux::ShaderParameterValue(infernux::rhi::IsSrgbFormat(format))}});
            const auto output = graph.Output();
            infernux::FullscreenResourceInput input;
            input.view = output->GetView(); input.format = output->GetFormat(); input.sampler = output->GetSampler();
            infernux::web::WebGpuGraphicsCommandContext context;
            auto commands = g_rhi->MakeGraphicsCommandEncoder(context, pass);
            g_presentRenderer.DrawProgram(commands, pipeline, &input, values, nullptr, resources);
            pass.End();
        };
        auto *scene = infernux::SceneManager::Instance().GetActiveScene();
        const auto cameras = scene ? scene->GetActiveGameCameras(nullptr) : std::vector<infernux::Camera *>{};
        bool rendered = false;
        g_presentRenderer.ResetPool();
        struct CapturedCamera
        {
            infernux::Camera *camera;
            std::shared_ptr<const infernux::RenderViewPlan> plan;
            CameraGraph *state;
        };
        std::vector<CapturedCamera> captured;
        std::vector<infernux::RenderViewAccess> accesses;
        captured.reserve(cameras.size());
        accesses.reserve(cameras.size());
        for (auto *camera : cameras) {
            if (g_splashActive || g_webGpuValidationFailed)
                break;
            const auto plan = g_sceneRenderer.Capture(g_width, g_height, *camera, *g_renderPipeline);
            if (plan->state == infernux::RenderViewPlanState::Skipped)
                continue;
            if (!plan->graph)
                throw std::runtime_error("Render pipeline submitted a view without its graph");
            auto &state = g_cameraGraphs[plan->view->cameraId];
            state.lastFrame = g_shaderFrameIndex;
            if (!state.graph) {
                state.graph = std::make_unique<infernux::web::WebRenderGraph>(*g_rhi, g_fullscreenHost);
            }
            const auto &cameraView = *plan->view;
            const auto output = cameraView.outputGeneration;
            const auto outputRevision = output ? output->revision : 0;
            if (state.source != plan->graph || state.outputRevision != outputRevision) {
                state.source = plan->graph;
                state.outputRevision = outputRevision;
                state.description = output ? std::make_shared<infernux::RenderGraphDescription>(
                    infernux::NormalizeRenderGraphForOutput(*plan->graph, true, output->description.samples)) : plan->graph;
            }
            g_lastSampleCount = state.description->msaaSamples > 0 ? uint32_t(state.description->msaaSamples) : 1;
            infernux::rhi::RenderViewContext graphView;
            graphView.id = cameraView.cameraId;
            graphView.device = g_rhi->GetDeviceId();
            graphView.width = cameraView.pixelWidth;
            graphView.height = cameraView.pixelHeight;
            graphView.colorFormat = output ? output->description.colorFormat : infernux::rhi::PixelFormat::RGBA16SFloat;
            graphView.depthFormat = output ? output->description.depthFormat : infernux::rhi::PixelFormat::D32SFloat;
            graphView.samples = static_cast<infernux::rhi::SampleCount>(g_lastSampleCount);
            graphView.revision = outputRevision;
            state.graph->Apply(state.description, graphView, {},
                infernux::ResolveGraphCameraClear(*state.description, cameraView.clearFlags,
                    inx::color::SrgbToLinear(cameraView.backgroundColor)));
            state.graph->UpdateParameters(plan->parameters);
            auto access = state.graph->GetResourceAccess();
            access.writes.push_back(output ? static_cast<const void *>(output->color.get()) : &g_cameraGraphs);
            if (output && output->depth)
                access.writes.push_back(output->depth.get());
            accesses.push_back(std::move(access));
            captured.push_back({camera, plan, &state});
        }
        const auto schedule = infernux::RenderViewSchedule::Build(accesses);
        for (const auto index : schedule.order) {
            const auto &capture = captured[index];
            auto *camera = capture.camera;
            const auto &plan = capture.plan;
            auto &state = *capture.state;
            const auto &cameraView = *plan->view;
            const auto output = cameraView.outputGeneration;
            g_lastSampleCount = state.description->msaaSamples > 0 ? uint32_t(state.description->msaaSamples) : 1;
            g_sceneRenderer.Prepare(*plan, *state.graph);
            g_sceneRenderer.RecordPreparation(encoder);
            if (!g_particleRuntime.ConfigureSampleCount(static_cast<infernux::rhi::SampleCount>(g_lastSampleCount)))
                throw std::runtime_error(g_particleRuntime.LastError());
            const auto resources = g_sceneRenderer.FullscreenView();
            state.graph->Record(encoder, resources,
                [&](auto pass, const auto &command, const auto &raster, const auto &target) {
                    using Type = infernux::GraphCommandType;
                    switch (command.type) {
                    case Type::DrawRenderers:
                        g_sceneRenderer.Record(pass, command, target);
                        if (command.queueMin <= 3000 && command.queueMax >= 3000 && g_particleRenderingEnabledForDiagnostics)
                            (void)g_particleRuntime.Render(pass, *camera, raster.width, raster.height);
                        break;
                    case Type::DrawSkybox:
                    case Type::DrawShadowCasters:
                        g_sceneRenderer.Record(pass, command, target);
                        break;
                    case Type::DrawWorldUI:
                        (void)g_screenUIRenderer.RenderWorld(pass, *camera, raster.width, raster.height, target, command.worldUILayerMask);
                        break;
                    case Type::DrawScreenUI:
                        (void)g_screenUIRenderer.Render(pass, command.screenUIList, raster.width, raster.height, target);
                        break;
                    default:
                        throw std::invalid_argument("Unexpected non-raster command in a render pass");
                    }
                }, [](auto, const auto &) { throw std::invalid_argument("Graph work has no execution binding"); });
            if (output) {
                // Persistent camera outputs stay linear. Screen-only UI and
                // display encoding were removed by the common output policy.
                present(*state.graph, true, resources, g_rhi->GetNativeTextureView(output->color->GetView()),
                        output->description.colorFormat,
                        infernux::RasterViewportRect(cameraView.rect, output->width, output->height), false);
            } else {
                present(*state.graph, state.description->outputDisplayEncoded, resources, view, ToRhiFormat(g_surfaceFormat),
                        infernux::RasterViewportRect(cameraView.rect, g_width, g_height), !rendered);
                rendered = true;
            }
            // Per-camera UI uploads precede this submit. The next view can
            // then reuse its UI buffers without changing this recorded draw.
            auto commands = encoder.Finish();
            g_queue.Submit(1, &commands);
            encoder = g_device.CreateCommandEncoder();
        }
        if (!rendered) {
            wgpu::RenderPassColorAttachment attachment{};
            attachment.view = view;
            attachment.loadOp = wgpu::LoadOp::Clear;
            attachment.storeOp = wgpu::StoreOp::Store;
            attachment.clearValue = {0, 0, 0, 1};
            wgpu::RenderPassDescriptor descriptor{};
            descriptor.colorAttachmentCount = 1;
            descriptor.colorAttachments = &attachment;
            auto pass = encoder.BeginRenderPass(&descriptor);
            if (g_splashActive) {
                infernux::MaterialPassPipelineDescriptor target;
                target.colorFormats = {ToRhiFormat(g_surfaceFormat)};
                (void)g_screenUIRenderer.Render(pass, 0, g_width, g_height, target);
                (void)g_screenUIRenderer.Render(pass, 1, g_width, g_height, target);
            }
            pass.End();
        }
        for (auto entry = g_cameraGraphs.begin(); entry != g_cameraGraphs.end();) {
            if (entry->second.lastFrame + 1 < g_shaderFrameIndex)
                entry = g_cameraGraphs.erase(entry);
            else
                ++entry;
        }
        g_screenUIRenderer.PruneUnusedMaterials();
        wgpu::CommandBuffer commands = encoder.Finish();
        g_queue.Submit(1, &commands);
        foregroundSubmitted = true;
        if (g_particleRuntimeReady)
            g_particleRuntime.NotifySubmission();
    }
#if defined(INFERNUX_WEB_ENGINE_RUNTIME)
    g_frameWork.AdvanceLate([foregroundSubmitted] {
        if (!foregroundSubmitted)
            return infernux::FrameGpuAvailability::ForegroundDeferred;
        g_computeQueue->Collect();
        return g_computeQueue->GetPendingSubmissionCount() ? infernux::FrameGpuAvailability::BackgroundPending
                                                           : infernux::FrameGpuAvailability::Ready;
    });
    if (pauseAcceptanceAfterRender) {
        auto &sceneManager = infernux::SceneManager::Instance();
        sceneManager.Pause();
        g_acceptancePaused = true;
        std::printf("INFERNUX_WEB_ACCEPTANCE_PAUSED frame=%llu fixed_delta=%g\n",
                    static_cast<unsigned long long>(sceneManager.GetRuntimeFrameCount()),
                    g_acceptanceFixedDeltaSeconds);
    }
#endif
    infernux::InputManager::Instance().BeginFrame();
}

void StartSurface()
{
    if (!ConfigureAcceptanceClock()) {
        InfernuxWebAbortStartup("Invalid acceptance clock configuration");
        return;
    }
    g_queue = g_device.GetQueue();
    wgpu::EmscriptenSurfaceSourceCanvasHTMLSelector canvasSource;
    canvasSource.selector = "#canvas";
    wgpu::SurfaceDescriptor surfaceDescriptor;
    surfaceDescriptor.nextInChain = &canvasSource;
    g_surface = g_instance.CreateSurface(&surfaceDescriptor);

    wgpu::SurfaceCapabilities capabilities;
    g_surface.GetCapabilities(g_adapter, &capabilities);
    if (capabilities.formatCount == 0) {
        std::fprintf(stderr, "INFERNUX_WEBGPU_NO_SURFACE_FORMAT\n");
        InfernuxWebAbortStartup("WebGPU surface has no supported format");
        return;
    }
    g_surfaceFormat = capabilities.formats[0];
    if (!CreateRhiServices()) {
        std::fprintf(stderr, "INFERNUX_WEBGPU_RHI_PIPELINE_FAILED %s\n",
                     g_rhi ? g_rhi->LastError().c_str() : "unsupported surface format");
        InfernuxWebAbortStartup("WebGPU render pipeline initialization failed");
        return;
    }
    PyObject *mainModule = PyImport_AddModule("__main__");
    auto createPipeline = pybind11::reinterpret_borrow<pybind11::object>(mainModule).attr("infernux_web_create_render_pipeline");
    auto pipeline = createPipeline();
    g_renderPipeline = pipeline.cast<infernux::RenderPipelineCallback *>();
    g_renderPipelineOwner = pipeline.release().ptr();
    g_fullscreenHost = std::make_shared<FullscreenHost>();
    g_presentRenderer.Initialize(g_fullscreenHost);
    constexpr uint32_t sceneSampleCount = 1;
    if (!g_sceneRenderer.Initialize(*g_rhi)) {
        std::fprintf(stderr, "INFERNUX_WEBGPU_SCENE_PIPELINE_FAILED\n");
        InfernuxWebAbortStartup("Scene rendering initialization failed");
        return;
    }
    if (!g_screenUIRenderer.Initialize(*g_rhi, g_sceneRenderer)) {
        std::fprintf(stderr, "INFERNUX_WEB_SCREEN_UI_INITIALIZATION_FAILED\n");
        InfernuxWebAbortStartup("Screen UI initialization failed");
        return;
    }
    InfernuxWebSetScreenUIRenderer(&g_screenUIRenderer);
    const auto particleSampleCount =
        sceneSampleCount == 4 ? infernux::rhi::SampleCount::Four : infernux::rhi::SampleCount::One;
    if (!g_particleRuntime.Initialize(*g_rhi, g_sceneRenderer, infernux::rhi::PixelFormat::RGBA16SFloat,
                                      particleSampleCount)) {
        std::fprintf(stderr, "INFERNUX_WEBGPU_PARTICLE_RUNTIME_FAILED %s\n", g_particleRuntime.LastError().c_str());
        InfernuxWebAbortStartup("Particle runtime initialization failed");
        return;
    }
    g_particleRuntimeReady = true;
    InfernuxWebSetParticleRuntime(&g_particleRuntime);
    std::printf("INFERNUX_WEBGPU_FULLSCREEN_RHI_READY\n");
    ResizeCanvas();
    emscripten_set_resize_callback(EMSCRIPTEN_EVENT_TARGET_WINDOW, nullptr, false, OnResize);
    emscripten_set_focus_callback(EMSCRIPTEN_EVENT_TARGET_WINDOW, nullptr, true, OnFocus);
    emscripten_set_blur_callback(EMSCRIPTEN_EVENT_TARGET_WINDOW, nullptr, true, OnFocus);
    emscripten_set_visibilitychange_callback(nullptr, true, OnVisibility);
    emscripten_set_keydown_callback(EMSCRIPTEN_EVENT_TARGET_WINDOW, nullptr, true, OnKey);
    emscripten_set_keyup_callback(EMSCRIPTEN_EVENT_TARGET_WINDOW, nullptr, true, OnKey);
    emscripten_set_wheel_callback("#canvas", nullptr, true, OnWheel);
    PyObject *details = PyDict_New();
    SetDictString(details, "graphics_api", "webgpu");
    SetDictInteger(details, "width", g_width);
    SetDictInteger(details, "height", g_height);
    // Scene callbacks can create voices before the first gesture. SDL keeps
    // WebAudio suspended and resumes it on browser user activation.
    infernux::AudioEngine::Instance().Initialize();
    PyObject *ready = PyObject_GetAttrString(mainModule, "infernux_web_ready");
    if (ready == nullptr) {
        Py_DECREF(details);
        PrintPythonError("ready-contract");
        InfernuxWebAbortStartup("Player startup callback is unavailable");
        return;
    }
    // Scene preparation crosses the Python/C++ boundary and may invoke the
    // native scene loader.  Never let a C++ exception escape through the
    // Python C API: on WebAssembly that becomes an opaque `___cxa_throw`
    // pointer in the browser and leaves the page stuck at "Publishing the
    // first frame".  Convert it to a normal Python exception so the host
    // prints the actual failure and the page receives a useful abort reason.
    PyObject *result = nullptr;
    try {
        result = PyObject_CallOneArg(ready, details);
    } catch (const std::exception &error) {
        std::fprintf(stderr, "INFERNUX_WEB_READY_NATIVE_EXCEPTION %s\n", error.what());
        PyErr_SetString(PyExc_RuntimeError, error.what());
    } catch (...) {
        std::fprintf(stderr, "INFERNUX_WEB_READY_NATIVE_EXCEPTION unknown\n");
        PyErr_SetString(PyExc_RuntimeError, "unknown native exception during Web Player startup");
    }
    Py_DECREF(ready);
    if (result == nullptr) {
        Py_DECREF(details);
        PrintPythonError("ready");
        InfernuxWebAbortStartup("Player scene could not be loaded");
        return;
    }
    Py_DECREF(result);
    Py_DECREF(details);

    std::printf("INFERNUX_WEBGPU_DEVICE_READY format=%d\n", static_cast<int>(g_surfaceFormat));
    g_lastFrameTimeMilliseconds = emscripten_get_now();
    infernux::InputManager::Instance().BeginFrame();
}

} // namespace

extern "C" EMSCRIPTEN_KEEPALIVE void infernux_web_start()
{
    try {
        StartSurface();
    } catch (const std::exception &error) {
        std::fprintf(stderr, "INFERNUX_WEB_STARTUP_FAILED %s\n", error.what());
        InfernuxWebAbortStartup(error.what());
    }
}

extern "C" EMSCRIPTEN_KEEPALIVE void infernux_web_frame()
{
    try {
        Frame();
        g_frameWork.EndOpportunity();
    } catch (const std::exception &error) {
        std::fprintf(stderr, "INFERNUX_WEB_FRAME_FAILED %s\n", error.what());
        InfernuxWebAbortStartup(error.what());
    }
}

int main()
{
    std::printf("INFERNUX_WEB_PLAYER_START\n");
    try {
        infernux::inxpack::Extract("/infernux-project.inxpkg", "/infernux/player");
        std::remove("/infernux-project.inxpkg");
    } catch (const std::exception &error) {
        std::fprintf(stderr, "INFERNUX_WEB_CONTENT_LOAD_FAILED %s\n", error.what());
        InfernuxWebAbortStartup("Game content could not be loaded");
        return 1;
    }
    if (!InitializePython()) {
        std::fprintf(stderr, "INFERNUX_WEB_PYTHON_INITIALIZATION_FAILED\n");
        InfernuxWebAbortStartup("Python runtime initialization failed");
        return 1;
    }
    wgpu::RequestAdapterOptions adapterOptions;
    adapterOptions.forceFallbackAdapter = InfernuxWebForceFallbackAdapter() != 0;
    std::printf("INFERNUX_WEBGPU_ADAPTER_REQUEST mode=%s\n",
                adapterOptions.forceFallbackAdapter ? "fallback" : "preferred");
    g_instance.RequestAdapter(
        &adapterOptions, wgpu::CallbackMode::AllowSpontaneous,
        [](wgpu::RequestAdapterStatus status, wgpu::Adapter adapter, wgpu::StringView message) {
            if (status != wgpu::RequestAdapterStatus::Success) {
                std::fprintf(stderr, "INFERNUX_WEBGPU_ADAPTER_FAILED %.*s\n", static_cast<int>(message.length),
                             message.data);
                InfernuxWebAbortStartup("No compatible WebGPU adapter is available");
                return;
            }
            g_adapter = std::move(adapter);
            wgpu::DeviceDescriptor descriptor;
            wgpu::Limits adapterLimits;
            wgpu::Limits requiredLimits;
            // Ask for the strongest storage-buffer contract the adapter can
            // provide up to the complete particle ABI. WebGPU otherwise gives
            // the device only its baseline default (commonly eight), even when
            // the adapter exposes more. Individual pipelines still validate
            // against the negotiated limit; content that needs twelve does not
            // silently acquire a different implementation on a smaller device.
            constexpr uint32_t kCompleteParticleStorageBufferLimit = 12;
            if (g_adapter.GetLimits(&adapterLimits)) {
                std::printf("INFERNUX_WEBGPU_ADAPTER_LIMITS bind_groups=%u storage_buffers=%u bindings=%u\n",
                            adapterLimits.maxBindGroups, adapterLimits.maxStorageBuffersPerShaderStage,
                            adapterLimits.maxBindingsPerBindGroup);
                g_webGpuStorageBufferLimit =
                    std::min(kCompleteParticleStorageBufferLimit, adapterLimits.maxStorageBuffersPerShaderStage);
                requiredLimits.maxStorageBuffersPerShaderStage = g_webGpuStorageBufferLimit;
                requiredLimits.maxColorAttachmentBytesPerSample = adapterLimits.maxColorAttachmentBytesPerSample;
                descriptor.requiredLimits = &requiredLimits;
                std::printf("INFERNUX_WEBGPU_STORAGE_LIMIT negotiated=%u complete_particle=%u\n",
                            g_webGpuStorageBufferLimit, kCompleteParticleStorageBufferLimit);
            }
            descriptor.SetUncapturedErrorCallback(
                [](const wgpu::Device &, wgpu::ErrorType type, wgpu::StringView errorMessage) {
                    g_webGpuValidationFailed = true;
                    std::fprintf(stderr, "INFERNUX_WEBGPU_UNCAPTURED_ERROR type=%d message=%.*s\n",
                                 static_cast<int>(type), static_cast<int>(errorMessage.length), errorMessage.data);
                });
            descriptor.SetDeviceLostCallback(
                wgpu::CallbackMode::AllowSpontaneous,
                [](const wgpu::Device &, wgpu::DeviceLostReason reason, wgpu::StringView lostMessage) {
                    std::fprintf(stderr, "INFERNUX_WEBGPU_DEVICE_LOST reason=%d message=%.*s\n",
                                 static_cast<int>(reason), static_cast<int>(lostMessage.length), lostMessage.data);
                });
            g_adapter.RequestDevice(
                &descriptor, wgpu::CallbackMode::AllowSpontaneous,
                [](wgpu::RequestDeviceStatus status, wgpu::Device device, wgpu::StringView message) {
                    if (status != wgpu::RequestDeviceStatus::Success) {
                        std::fprintf(stderr, "INFERNUX_WEBGPU_DEVICE_FAILED %.*s\n", static_cast<int>(message.length),
                                     message.data);
                        InfernuxWebAbortStartup("WebGPU device initialization failed");
                        return;
                    }
                    g_device = std::move(device);
                    InfernuxWebStartRuntime();
                });
        });
    return 0;
}
