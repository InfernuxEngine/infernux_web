cmake_minimum_required(VERSION 3.25)

project(InfernuxWebPlayerHost LANGUAGES C CXX)

if(NOT EMSCRIPTEN)
    message(FATAL_ERROR "InfernuxWebPlayerHost must be configured with emcmake")
endif()

set(INFERNUX_WEB_CPYTHON_SOURCE "" CACHE PATH
    "CPython 3.13 source directory")
set(INFERNUX_WEB_CPYTHON_BUILD "" CACHE PATH
    "Configured CPython 3.13 emscripten-browser build directory")
set(INFERNUX_WEB_PYTHON_RUNTIME_MANIFEST "" CACHE FILEPATH
    "Audited CPython/Emscripten package-closure manifest")
set(INFERNUX_WEB_NUMPY_PAYLOAD_ROOT "" CACHE PATH
    "Exact NumPy static archives, registration source, and preload tree")
set(INFERNUX_WEB_HOST_PYTHON "" CACHE FILEPATH
    "Absolute host Python used by Web configure/build tools")
set(INFERNUX_WEB_PLAYER_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/publish/player" CACHE PATH
    "Build-tree destination for the precompiled Web Player payload")
set(INFERNUX_ENGINE_SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." CACHE PATH
    "Infernux source directory providing the backend-neutral RHI contract")
set(INFERNUX_EXTERNAL_ROOT "${INFERNUX_ENGINE_SOURCE_ROOT}/external")
option(INFERNUX_WEB_BUILD_NATIVE_MODULE_PROBE
    "Compile the existing Infernux Python binding surface for wasm without linking it"
    OFF
)
option(INFERNUX_WEB_LINK_NATIVE_MODULE_PROBE
    "Link the Web Player runtime binding and scene inventories into the host"
    OFF
)
option(INFERNUX_WEB_LINK_ENGINE_RUNTIME
    "Statically link the supported Infernux gameplay runtime into the Web Player"
    ON
)
if(INFERNUX_WEB_LINK_NATIVE_MODULE_PROBE AND
   NOT INFERNUX_WEB_BUILD_NATIVE_MODULE_PROBE)
    message(FATAL_ERROR
        "INFERNUX_WEB_LINK_NATIVE_MODULE_PROBE requires INFERNUX_WEB_BUILD_NATIVE_MODULE_PROBE")
endif()

foreach(_required
        INFERNUX_WEB_CPYTHON_SOURCE
        INFERNUX_WEB_CPYTHON_BUILD
        INFERNUX_ENGINE_SOURCE_ROOT
        INFERNUX_WEB_NUMPY_PAYLOAD_ROOT)
    if(NOT IS_DIRECTORY "${${_required}}")
        message(FATAL_ERROR "${_required} must name an existing directory")
    endif()
endforeach()
if(NOT IS_ABSOLUTE "${INFERNUX_WEB_HOST_PYTHON}" OR
   NOT EXISTS "${INFERNUX_WEB_HOST_PYTHON}")
    message(FATAL_ERROR
        "INFERNUX_WEB_HOST_PYTHON must name an existing absolute interpreter")
endif()
if(NOT IS_ABSOLUTE "${INFERNUX_WEB_PYTHON_RUNTIME_MANIFEST}" OR
   NOT EXISTS "${INFERNUX_WEB_PYTHON_RUNTIME_MANIFEST}")
    message(FATAL_ERROR
        "INFERNUX_WEB_PYTHON_RUNTIME_MANIFEST must name an existing absolute file")
endif()
file(READ "${INFERNUX_WEB_PYTHON_RUNTIME_MANIFEST}" _python_runtime_manifest)
string(JSON _python_runtime_manifest_type ERROR_VARIABLE _python_runtime_manifest_error
    TYPE "${_python_runtime_manifest}")
if(_python_runtime_manifest_error OR
   NOT _python_runtime_manifest_type STREQUAL "OBJECT")
    message(FATAL_ERROR
        "INFERNUX_WEB_PYTHON_RUNTIME_MANIFEST must contain one JSON object")
endif()
execute_process(
    COMMAND "${INFERNUX_WEB_HOST_PYTHON}"
        "${CMAKE_CURRENT_SOURCE_DIR}/../package/editor/infernux_web/native_payload.py"
        --verify-python-package-payload "${INFERNUX_WEB_NUMPY_PAYLOAD_ROOT}"
        --runtime-manifest "${INFERNUX_WEB_PYTHON_RUNTIME_MANIFEST}"
        --package numpy
    RESULT_VARIABLE _python_runtime_validation_result
    ERROR_VARIABLE _python_runtime_validation_error
)
if(NOT _python_runtime_validation_result EQUAL 0)
    message(FATAL_ERROR
        "Web Python runtime manifest validation failed: ${_python_runtime_validation_error}")
endif()

set(_numpy_static_manifest
    "${INFERNUX_WEB_NUMPY_PAYLOAD_ROOT}/numpy_static_link.json")
set(_numpy_static_registry
    "${INFERNUX_WEB_NUMPY_PAYLOAD_ROOT}/numpy_builtin_registry.c")
set(_numpy_site_packages
    "${INFERNUX_WEB_NUMPY_PAYLOAD_ROOT}/site-packages")
foreach(_required_file _numpy_static_manifest _numpy_static_registry)
    if(NOT EXISTS "${${_required_file}}")
        message(FATAL_ERROR "Missing NumPy static payload file: ${${_required_file}}")
    endif()
endforeach()
if(NOT IS_DIRECTORY "${_numpy_site_packages}/numpy")
    message(FATAL_ERROR "NumPy preload tree is missing from the static payload")
endif()
file(GLOB_RECURSE _numpy_dynamic_payload_files LIST_DIRECTORIES FALSE
    "${INFERNUX_WEB_NUMPY_PAYLOAD_ROOT}/*.so"
    "${INFERNUX_WEB_NUMPY_PAYLOAD_ROOT}/*.wasm")
if(_numpy_dynamic_payload_files)
    message(FATAL_ERROR
        "NumPy static payload contains forbidden dynamic files: ${_numpy_dynamic_payload_files}")
endif()
file(READ "${_numpy_static_manifest}" _numpy_static_document)
string(JSON _numpy_archive_count ERROR_VARIABLE _numpy_static_error
    LENGTH "${_numpy_static_document}" archives)
if(_numpy_static_error OR _numpy_archive_count LESS 13)
    message(FATAL_ERROR "NumPy static link manifest is invalid: ${_numpy_static_error}")
endif()
set(_numpy_static_archives)
math(EXPR _numpy_archive_last "${_numpy_archive_count} - 1")
foreach(_index RANGE 0 ${_numpy_archive_last})
    string(JSON _relative_archive GET "${_numpy_static_document}" archives ${_index})
    set(_archive "${INFERNUX_WEB_NUMPY_PAYLOAD_ROOT}/${_relative_archive}")
    if(NOT EXISTS "${_archive}")
        message(FATAL_ERROR "NumPy static archive is missing: ${_archive}")
    endif()
    list(APPEND _numpy_static_archives "${_archive}")
endforeach()

set(_python_library "${INFERNUX_WEB_CPYTHON_BUILD}/libpython3.13.a")
set(_python_assets "${INFERNUX_WEB_CPYTHON_BUILD}/usr/local")
set(_python_expat "${INFERNUX_WEB_CPYTHON_BUILD}/Modules/expat/libexpat.a")
set(_python_mpdecimal
    "${INFERNUX_WEB_CPYTHON_BUILD}/Modules/_decimal/libmpdec/libmpdec.a")
set(_python_hacl
    "${INFERNUX_WEB_CPYTHON_BUILD}/Modules/_hacl/libHacl_Hash_SHA2.a")
if(NOT EXISTS "${_python_library}")
    message(FATAL_ERROR "Missing CPython wasm library: ${_python_library}")
endif()
if(NOT IS_DIRECTORY "${_python_assets}")
    message(FATAL_ERROR "Missing CPython wasm standard library: ${_python_assets}")
endif()
foreach(_archive _python_expat _python_mpdecimal _python_hacl)
    if(NOT EXISTS "${${_archive}}")
        message(FATAL_ERROR "Missing CPython wasm dependency: ${${_archive}}")
    endif()
endforeach()

# The Web Player owns an audio-only SDL runtime. Browser graphics and input
# remain on the explicit WebGPU/DOM adapters above this layer.
set(SDL_SHARED OFF CACHE BOOL "" FORCE)
set(SDL_STATIC ON CACHE BOOL "" FORCE)
set(SDL_TEST_LIBRARY OFF CACHE BOOL "" FORCE)
set(SDL_TESTS OFF CACHE BOOL "" FORCE)
set(SDL_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SDL_INSTALL OFF CACHE BOOL "" FORCE)
set(SDL_VIDEO OFF CACHE BOOL "" FORCE)
set(SDL_RENDER OFF CACHE BOOL "" FORCE)
set(SDL_GPU OFF CACHE BOOL "" FORCE)
set(SDL_CAMERA OFF CACHE BOOL "" FORCE)
set(SDL_JOYSTICK OFF CACHE BOOL "" FORCE)
set(SDL_HAPTIC OFF CACHE BOOL "" FORCE)
set(SDL_HIDAPI OFF CACHE BOOL "" FORCE)
set(SDL_SENSOR OFF CACHE BOOL "" FORCE)
set(SDL_DIALOG OFF CACHE BOOL "" FORCE)
add_subdirectory(
    "${INFERNUX_ENGINE_SOURCE_ROOT}/external/sdl_for_infernux"
    "${CMAKE_BINARY_DIR}/_deps/infernux-web-sdl"
    EXCLUDE_FROM_ALL
)

set(ZSTD_BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_STATIC ON CACHE BOOL "" FORCE)
set(ZSTD_MULTITHREAD_SUPPORT OFF CACHE BOOL "" FORCE)
set(ZSTD_LEGACY_SUPPORT OFF CACHE BOOL "" FORCE)
set(_web_zstd_source "${INFERNUX_ENGINE_SOURCE_ROOT}/external/zstd_for_infernux/build/cmake")
if(NOT EXISTS "${_web_zstd_source}/CMakeLists.txt")
    message(FATAL_ERROR "Initialize the engine's external/zstd_for_infernux submodule before building Web Players")
endif()
add_subdirectory("${_web_zstd_source}" "${CMAKE_BINARY_DIR}/_deps/infernux-web-zstd" EXCLUDE_FROM_ALL)
add_subdirectory("${INFERNUX_ENGINE_SOURCE_ROOT}/external/unicode"
    "${CMAKE_BINARY_DIR}/_deps/infernux-web-unicode" EXCLUDE_FROM_ALL)

add_executable(InfernuxWebPlayerHost
    main.cpp
    "${_numpy_static_registry}"
    InfernuxWebHostModule.cpp
    WebGpuRhiDevice.cpp
    WebParticleRuntime.cpp
    WebPostProcessRenderer.cpp
    WebIndexPacking.cpp
    WebTextureUpload.cpp
    WebSceneRenderer.cpp
    WebScreenUIRenderer.cpp
    "${INFERNUX_ENGINE_SOURCE_ROOT}/cpp/infernux/function/renderer/gui/InxTextLayout.cpp"
    "${INFERNUX_ENGINE_SOURCE_ROOT}/cpp/infernux/function/renderer/gui/InxTextGeometry.cpp"
    "${INFERNUX_ENGINE_SOURCE_ROOT}/cpp/infernux/function/renderer/particle/ParticleGpuContactRuntime.cpp"
    "${INFERNUX_ENGINE_SOURCE_ROOT}/cpp/infernux/function/renderer/particle/ParticleGpuContinuationRuntime.cpp"
    "${INFERNUX_ENGINE_SOURCE_ROOT}/cpp/infernux/function/renderer/particle/ParticleGpuRuntime.cpp"
    "${INFERNUX_ENGINE_SOURCE_ROOT}/cpp/infernux/function/renderer/FullscreenRenderer.cpp"
    "${INFERNUX_ENGINE_SOURCE_ROOT}/cpp/infernux/function/renderer/rhi/RhiComputeBuffer.cpp"
    "${INFERNUX_ENGINE_SOURCE_ROOT}/cpp/infernux/function/renderer/rhi/RhiComputeKernel.cpp"
    "${INFERNUX_ENGINE_SOURCE_ROOT}/cpp/infernux/function/renderer/rhi/RhiRenderTexture.cpp"
    "${INFERNUX_ENGINE_SOURCE_ROOT}/cpp/infernux/function/renderer/rhi/RhiDeviceIdentity.cpp"
    "${INFERNUX_ENGINE_SOURCE_ROOT}/cpp/infernux/function/audio/AudioClip.cpp"
    "${INFERNUX_ENGINE_SOURCE_ROOT}/cpp/infernux/function/audio/AudioClipLoader.cpp"
    "${INFERNUX_ENGINE_SOURCE_ROOT}/cpp/infernux/function/audio/AudioDecoders.cpp"
    "${INFERNUX_ENGINE_SOURCE_ROOT}/cpp/infernux/function/audio/AudioEngine.cpp"
    "${INFERNUX_ENGINE_SOURCE_ROOT}/cpp/infernux/function/audio/AudioListener.cpp"
    "${INFERNUX_ENGINE_SOURCE_ROOT}/cpp/infernux/function/audio/AudioSource.cpp"
    "${INFERNUX_ENGINE_SOURCE_ROOT}/cpp/infernux/function/audio/AudioStreamBuffer.cpp"
    "${INFERNUX_ENGINE_SOURCE_ROOT}/cpp/infernux/function/audio/AudioStreamDecoder.cpp"
    "${INFERNUX_ENGINE_SOURCE_ROOT}/cpp/infernux/platform/filesystem/InxPack.cpp"
    "${INFERNUX_ENGINE_SOURCE_ROOT}/cpp/infernux/platform/input/InputManager.cpp"
    "${INFERNUX_ENGINE_SOURCE_ROOT}/external/stb/stb_vorbis.c"
)
set_target_properties(InfernuxWebPlayerHost PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    OUTPUT_NAME "infernux-runtime"
    SUFFIX ".js"
)
set_property(TARGET InfernuxWebPlayerHost APPEND PROPERTY LINK_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/bootstrap.py"
)
file(GLOB_RECURSE _numpy_payload_files CONFIGURE_DEPENDS LIST_DIRECTORIES FALSE
    "${INFERNUX_WEB_NUMPY_PAYLOAD_ROOT}/*")
set_property(TARGET InfernuxWebPlayerHost APPEND PROPERTY LINK_DEPENDS
    ${_numpy_payload_files}
)
add_custom_target(infernux_web_verify_python_runtime ALL
    COMMAND "${INFERNUX_WEB_HOST_PYTHON}"
        "${CMAKE_CURRENT_SOURCE_DIR}/../package/editor/infernux_web/native_payload.py"
        --verify-python-package-payload "${INFERNUX_WEB_NUMPY_PAYLOAD_ROOT}"
        --runtime-manifest "${INFERNUX_WEB_PYTHON_RUNTIME_MANIFEST}"
        --package numpy
    VERBATIM
)
add_dependencies(InfernuxWebPlayerHost infernux_web_verify_python_runtime)

target_include_directories(InfernuxWebPlayerHost PRIVATE
    "${INFERNUX_WEB_CPYTHON_SOURCE}/Include"
    "${INFERNUX_WEB_CPYTHON_BUILD}"
    "${INFERNUX_ENGINE_SOURCE_ROOT}/cpp"
    "${INFERNUX_ENGINE_SOURCE_ROOT}/cpp/infernux"
    "${INFERNUX_ENGINE_SOURCE_ROOT}/external"
    "${INFERNUX_ENGINE_SOURCE_ROOT}/external/dr_libs"
    "${INFERNUX_ENGINE_SOURCE_ROOT}/external/glm"
    "${INFERNUX_ENGINE_SOURCE_ROOT}/external/imgui_for_infernux"
    "${INFERNUX_ENGINE_SOURCE_ROOT}/external/nlohmann/single_include"
    "${INFERNUX_ENGINE_SOURCE_ROOT}/external/sdl_for_infernux/include"
    "${INFERNUX_ENGINE_SOURCE_ROOT}/external/stb"
)
target_sources(InfernuxWebPlayerHost PRIVATE
    "${INFERNUX_ENGINE_SOURCE_ROOT}/external/imgui_for_infernux/imgui.cpp"
    "${INFERNUX_ENGINE_SOURCE_ROOT}/external/imgui_for_infernux/imgui_draw.cpp"
    "${INFERNUX_ENGINE_SOURCE_ROOT}/external/imgui_for_infernux/imgui_tables.cpp"
    "${INFERNUX_ENGINE_SOURCE_ROOT}/external/imgui_for_infernux/imgui_widgets.cpp"
)
target_link_libraries(InfernuxWebPlayerHost PRIVATE
    "-Wl,--start-group"
    ${_numpy_static_archives}
    "-Wl,--end-group"
    "${_python_library}"
    "${_python_expat}"
    "${_python_mpdecimal}"
    "${_python_hacl}"
    libzstd_static
    infernux_utf8proc
    SDL3::SDL3-static
    dl
    pthread
    m
)
target_compile_definitions(InfernuxWebPlayerHost PRIVATE
    GLM_FORCE_DEPTH_ZERO_TO_ONE
    GLM_FORCE_LEFT_HANDED
    INFERNUX_DISABLE_VULKAN_MATERIAL_RUNTIME=1
    INFERNUX_INPUT_SEMANTIC_HOST=1
    INFERNUX_RUNTIME_MINIMAL_HOST=1
    INFERNUX_SINGLE_THREADED_RUNTIME=1
    INFERNUX_WEBGPU_RUNTIME=1
)
if(INFERNUX_WEB_BUILD_NATIVE_MODULE_PROBE OR
   INFERNUX_WEB_LINK_ENGINE_RUNTIME)
    execute_process(
        COMMAND "${INFERNUX_WEB_HOST_PYTHON}"
            -c "import pybind11; print(pybind11.get_include())"
        OUTPUT_VARIABLE _web_pybind11_include
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _web_pybind11_result
    )
    if(NOT _web_pybind11_result EQUAL 0 OR
       NOT EXISTS "${_web_pybind11_include}/pybind11/pybind11.h")
        message(FATAL_ERROR
            "The active Web build environment must provide pybind11 headers")
    endif()
    file(GLOB _web_binding_sources CONFIGURE_DEPENDS
        "${INFERNUX_ENGINE_SOURCE_ROOT}/cpp/bindings/python/*.cpp"
    )
    list(FILTER _web_binding_sources EXCLUDE REGEX "BindingInfernuxBootstrap\\.cpp$")
    set(_web_portable_binding_sources ${_web_binding_sources})
    list(FILTER _web_portable_binding_sources EXCLUDE REGEX "BindingInfernux\\.cpp$")
    list(FILTER _web_portable_binding_sources EXCLUDE REGEX "BindingGUI\\.cpp$")
    list(FILTER _web_portable_binding_sources EXCLUDE REGEX "BindingRenderGraph\\.cpp$")
    set(_web_scene_binding_sources ${_web_binding_sources})
    list(FILTER _web_scene_binding_sources INCLUDE REGEX "/BindingScene[^/]*\\.cpp$")
    set(_web_runtime_binding_sources
        "${INFERNUX_ENGINE_SOURCE_ROOT}/cpp/bindings/python/BindingModule.cpp"
        "${INFERNUX_ENGINE_SOURCE_ROOT}/cpp/bindings/python/BindingAssetRegistry.cpp"
        "${INFERNUX_ENGINE_SOURCE_ROOT}/cpp/bindings/python/BindingBatch.cpp"
        "${INFERNUX_ENGINE_SOURCE_ROOT}/cpp/bindings/python/BindingAudio.cpp"
        "${INFERNUX_ENGINE_SOURCE_ROOT}/cpp/bindings/python/BindingInput.cpp"
        "${INFERNUX_ENGINE_SOURCE_ROOT}/cpp/bindings/python/BindingPhysics.cpp"
        "${INFERNUX_ENGINE_SOURCE_ROOT}/cpp/bindings/python/BindingResources.cpp"
        ${_web_scene_binding_sources}
        "${INFERNUX_ENGINE_SOURCE_ROOT}/cpp/bindings/python/BindingSemanticCatalog.cpp"
        "${INFERNUX_ENGINE_SOURCE_ROOT}/cpp/bindings/python/BindingTagLayer.cpp"
        "${INFERNUX_ENGINE_SOURCE_ROOT}/cpp/bindings/python/BindingText.cpp"
        "${INFERNUX_ENGINE_SOURCE_ROOT}/cpp/bindings/python/BindingVector2f.cpp"
        "${INFERNUX_ENGINE_SOURCE_ROOT}/cpp/bindings/python/BindingVector3f.cpp"
        "${INFERNUX_ENGINE_SOURCE_ROOT}/cpp/bindings/python/BindingVector4f.cpp"
    )

    add_library(InfernuxWebNativeModuleProbe OBJECT EXCLUDE_FROM_ALL
        ${_web_binding_sources}
    )
    add_library(InfernuxWebPortableBindingsProbe OBJECT EXCLUDE_FROM_ALL
        ${_web_portable_binding_sources}
    )
    add_library(InfernuxWebRuntimeBindingsProbe OBJECT EXCLUDE_FROM_ALL
        ${_web_runtime_binding_sources}
    )
    file(GLOB _web_scene_runtime_sources CONFIGURE_DEPENDS
        "${INFERNUX_ENGINE_SOURCE_ROOT}/cpp/infernux/function/scene/*.cpp"
        "${INFERNUX_ENGINE_SOURCE_ROOT}/cpp/infernux/function/scene/physics/*.cpp"
    )
    list(FILTER _web_scene_runtime_sources EXCLUDE REGEX "EditorCameraController\\.cpp$")
    list(FILTER _web_scene_runtime_sources EXCLUDE REGEX "SceneRenderBridge\\.cpp$")
    list(FILTER _web_scene_runtime_sources EXCLUDE REGEX "SceneRenderer\\.cpp$")
    add_library(InfernuxWebSceneRuntimeProbe OBJECT EXCLUDE_FROM_ALL
        ${_web_scene_runtime_sources}
    )
    file(GLOB_RECURSE _web_resource_runtime_sources CONFIGURE_DEPENDS
        "${INFERNUX_ENGINE_SOURCE_ROOT}/cpp/infernux/core/*.cpp"
        "${INFERNUX_ENGINE_SOURCE_ROOT}/cpp/infernux/function/resources/*.cpp"
        "${INFERNUX_ENGINE_SOURCE_ROOT}/cpp/infernux/platform/filesystem/*.cpp"
    )
    list(FILTER _web_resource_runtime_sources EXCLUDE REGEX "/AssetImporter/")
    list(FILTER _web_resource_runtime_sources EXCLUDE REGEX "SkinnedModelImporter\\.cpp$")
    list(FILTER _web_resource_runtime_sources EXCLUDE REGEX "/InxMesh/MeshLoader\\.cpp$")
    list(FILTER _web_resource_runtime_sources EXCLUDE REGEX "/InxMesh/ModelVertexBasis\\.cpp$")
    list(FILTER _web_resource_runtime_sources EXCLUDE REGEX "TextureProcessor\\.cpp$")
    list(FILTER _web_resource_runtime_sources EXCLUDE REGEX "InxShaderLoader\\.cpp$")
    list(FILTER _web_resource_runtime_sources EXCLUDE REGEX "InxPack\\.cpp$")
    add_library(InfernuxWebResourceRuntimeProbe OBJECT EXCLUDE_FROM_ALL
        ${_web_resource_runtime_sources}
    )
    add_subdirectory("${INFERNUX_ENGINE_SOURCE_ROOT}/external/svg"
        "${CMAKE_CURRENT_BINARY_DIR}/svg" EXCLUDE_FROM_ALL)
    if(INFERNUX_WEB_LINK_ENGINE_RUNTIME)
        # The C rasterizer uses setjmp/longjmp. Its unwinder must match the
        # native Wasm exception ABI used by the final Player and LunaSVG.
        target_compile_options(plutovg PRIVATE "-sSUPPORT_LONGJMP=wasm")
    endif()
    target_link_libraries(InfernuxWebResourceRuntimeProbe PRIVATE lunasvg::lunasvg infernux_utf8proc)
    add_library(InfernuxWebPlatformRuntimeProbe OBJECT EXCLUDE_FROM_ALL
        "${INFERNUX_ENGINE_SOURCE_ROOT}/cpp/infernux/platform/input/InputManager.cpp"
    )
    foreach(_web_binding_probe
            InfernuxWebNativeModuleProbe
            InfernuxWebPortableBindingsProbe
            InfernuxWebRuntimeBindingsProbe
            InfernuxWebSceneRuntimeProbe
            InfernuxWebResourceRuntimeProbe
            InfernuxWebPlatformRuntimeProbe)
        target_include_directories(${_web_binding_probe} PRIVATE
            "${_web_pybind11_include}"
            "${INFERNUX_WEB_CPYTHON_SOURCE}/Include"
            "${INFERNUX_WEB_CPYTHON_BUILD}"
            "${INFERNUX_ENGINE_SOURCE_ROOT}/cpp"
            "${INFERNUX_ENGINE_SOURCE_ROOT}/cpp/infernux"
            "${INFERNUX_ENGINE_SOURCE_ROOT}/external"
            "${INFERNUX_ENGINE_SOURCE_ROOT}/external/assimp/include"
            "${INFERNUX_ENGINE_SOURCE_ROOT}/external/glm"
            "${INFERNUX_ENGINE_SOURCE_ROOT}/external/glslang"
            "${INFERNUX_ENGINE_SOURCE_ROOT}/external/imgui_for_infernux"
            "${INFERNUX_ENGINE_SOURCE_ROOT}/external/joltphysics_for_infernux"
            "${INFERNUX_ENGINE_SOURCE_ROOT}/external/nlohmann/single_include"
            "${INFERNUX_ENGINE_SOURCE_ROOT}/external/sdl_for_infernux/include"
            "${INFERNUX_ENGINE_SOURCE_ROOT}/external/stb"
        )
        target_compile_definitions(${_web_binding_probe} PRIVATE
            GLM_FORCE_DEPTH_ZERO_TO_ONE
            GLM_FORCE_LEFT_HANDED
            INFERNUX_DISABLE_VULKAN_MATERIAL_RUNTIME=1
            INFERNUX_INPUT_SEMANTIC_HOST=1
            INFERNUX_RUNTIME_MINIMAL_HOST=1
            INFERNUX_SINGLE_THREADED_RUNTIME=1
            JPH_USE_CPU_COMPUTE
            JPH_OBJECT_STREAM
        )
        if(INFERNUX_WEB_LINK_ENGINE_RUNTIME)
            # Every object that can throw across the embedded Python boundary
            # must use the same Wasm EH ABI as the host executable.  Applying
            # the flag only to the final target leaves scene/resource objects
            # built with a different unwinder and aborts in __cxa_begin_catch.
            target_compile_options(${_web_binding_probe} PRIVATE "-fwasm-exceptions")
        endif()
    endforeach()
    target_compile_definitions(InfernuxWebRuntimeBindingsProbe PRIVATE
        INFERNUX_PYBIND_WEB_PLAYER=1
    )
    if(INFERNUX_WEB_LINK_NATIVE_MODULE_PROBE OR
       INFERNUX_WEB_LINK_ENGINE_RUNTIME)
        set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
        set(USE_SSE4_1 OFF CACHE BOOL "" FORCE)
        set(USE_SSE4_2 OFF CACHE BOOL "" FORCE)
        set(USE_AVX OFF CACHE BOOL "" FORCE)
        set(USE_AVX2 OFF CACHE BOOL "" FORCE)
        set(USE_AVX512 OFF CACHE BOOL "" FORCE)
        set(USE_LZCNT OFF CACHE BOOL "" FORCE)
        set(USE_TZCNT OFF CACHE BOOL "" FORCE)
        set(USE_F16C OFF CACHE BOOL "" FORCE)
        set(USE_FMADD OFF CACHE BOOL "" FORCE)
        # Must match external/CMakeLists.txt so wasm physics stays trajectory-
        # compatible with desktop/Android builds.
        set(CROSS_PLATFORM_DETERMINISTIC ON CACHE BOOL "" FORCE)
        set(OVERRIDE_CXX_FLAGS OFF CACHE BOOL "" FORCE)
        set(INTERPROCEDURAL_OPTIMIZATION OFF CACHE BOOL "" FORCE)
        set(DOUBLE_PRECISION OFF CACHE BOOL "" FORCE)
        set(GENERATE_DEBUG_SYMBOLS OFF CACHE BOOL "" FORCE)
        set(ENABLE_ALL_WARNINGS OFF CACHE BOOL "" FORCE)
        set(ENABLE_OBJECT_STREAM ON CACHE BOOL "" FORCE)
        set(ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
        set(CPP_EXCEPTIONS_ENABLED ON CACHE BOOL "" FORCE)
        set(CPP_RTTI_ENABLED ON CACHE BOOL "" FORCE)
        set(DEBUG_RENDERER_IN_DEBUG_AND_RELEASE OFF CACHE BOOL "" FORCE)
        set(PROFILER_IN_DEBUG_AND_RELEASE OFF CACHE BOOL "" FORCE)
        set(JPH_USE_DX12 OFF CACHE BOOL "" FORCE)
        set(JPH_USE_VK OFF CACHE BOOL "" FORCE)
        set(JPH_USE_MTL OFF CACHE BOOL "" FORCE)
        set(JPH_USE_CPU_COMPUTE ON CACHE BOOL "" FORCE)
        add_subdirectory(
            "${INFERNUX_ENGINE_SOURCE_ROOT}/external/joltphysics_for_infernux/Build"
            "${CMAKE_CURRENT_BINARY_DIR}/_deps/infernux-web-jolt"
            EXCLUDE_FROM_ALL
        )
        if(TARGET Jolt)
            # Jolt is linked into the same Wasm image as the scene/runtime
            # objects.  Its archive must use the identical Wasm EH ABI or the
            # final link leaves __resumeException unresolved when a physics
            # path is pulled in by scene loading.
            target_compile_options(Jolt PRIVATE "-fwasm-exceptions")
        endif()
        # These object libraries compile engine translation units that include
        # Jolt headers. Consume Jolt's PUBLIC compile contract as well as its
        # archive so JPH_VERSION_ID cannot diverge from the linked library.
        foreach(_web_jolt_consumer
                InfernuxWebNativeModuleProbe
                InfernuxWebPortableBindingsProbe
                InfernuxWebRuntimeBindingsProbe
                InfernuxWebSceneRuntimeProbe
                InfernuxWebResourceRuntimeProbe)
            target_link_libraries(${_web_jolt_consumer} PRIVATE Jolt)
        endforeach()
        target_sources(InfernuxWebPlayerHost PRIVATE
            $<TARGET_OBJECTS:InfernuxWebRuntimeBindingsProbe>
            $<TARGET_OBJECTS:InfernuxWebSceneRuntimeProbe>
            $<TARGET_OBJECTS:InfernuxWebResourceRuntimeProbe>
        )
        target_compile_definitions(InfernuxWebPlayerHost PRIVATE
            INFERNUX_WEB_ENGINE_RUNTIME=1
        )
        target_include_directories(InfernuxWebPlayerHost PRIVATE "${_web_pybind11_include}")
        target_link_libraries(InfernuxWebPlayerHost PRIVATE Jolt lunasvg::lunasvg)
    endif()
endif()

target_compile_options(InfernuxWebPlayerHost PRIVATE
    "--use-port=emdawnwebgpu"
    "-sUSE_ZLIB=1"
    "-sUSE_BZIP2=1"
    "-sUSE_SQLITE3=1"
)
if(INFERNUX_WEB_LINK_ENGINE_RUNTIME)
    # Scene loading crosses the embedded Python C API and native scene
    # deserializers. Keep C++ exceptions enabled at this boundary so a
    # malformed cooked scene becomes a normal Web Player error with its
    # message, instead of an opaque Emscripten ___cxa_throw pointer.
    target_compile_options(InfernuxWebPlayerHost PRIVATE "-fwasm-exceptions")
endif()
target_link_options(InfernuxWebPlayerHost PRIVATE
    "--use-port=emdawnwebgpu"
    "-sUSE_ZLIB=1"
    "-sUSE_BZIP2=1"
    "-sUSE_SQLITE3=1"
    "-sALLOW_MEMORY_GROWTH=1"
    "-sINITIAL_MEMORY=134217728"
    "-sMAXIMUM_MEMORY=2147483648"
    "-sSTACK_SIZE=2097152"
    "-sFORCE_FILESYSTEM=1"
    "-sENVIRONMENT=web"
    "-sEXIT_RUNTIME=0"
    $<$<BOOL:${INFERNUX_WEB_LINK_ENGINE_RUNTIME}>:-fwasm-exceptions>
    "-sEXPORTED_RUNTIME_METHODS=['ccall','FS','addRunDependency','removeRunDependency','abort']"
    "--preload-file=${_python_assets}@/usr/local"
    "--preload-file=${_numpy_site_packages}@/usr/local/lib/python3.13/site-packages"
    "--preload-file=${CMAKE_CURRENT_SOURCE_DIR}/bootstrap.py@/infernux/bootstrap.py"
)

# Keep generated Player payloads outside the source tree. Release packaging
# consumes this directory explicitly instead of relying on source mutation.
if(NOT IS_ABSOLUTE "${INFERNUX_WEB_PLAYER_OUTPUT_DIR}")
    message(FATAL_ERROR "INFERNUX_WEB_PLAYER_OUTPUT_DIR must be absolute")
endif()
set(_web_player_destination "${INFERNUX_WEB_PLAYER_OUTPUT_DIR}")
set(_engine_version_source "${INFERNUX_ENGINE_SOURCE_ROOT}/python/infernux/version.py")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_engine_version_source}")
execute_process(
    COMMAND "${INFERNUX_WEB_HOST_PYTHON}" -c
        "import runpy,sys; print(runpy.run_path(sys.argv[1])['ENGINE_VERSION'])"
        "${_engine_version_source}"
    OUTPUT_VARIABLE _engine_version
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/Player.inxmanifest" CONTENT
"{\"$schema\":\"infernux.web_player\",\"engine\":\"${_engine_version}\",\"platform\":\"web\",\"architecture\":\"wasm32\",\"python_abi\":\"cp313\",\"configuration\":\"$<CONFIG>\",\"python_runtime\":${_python_runtime_manifest}}\n")
add_custom_target(prebuild_web_player
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${_web_player_destination}"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "$<TARGET_FILE:InfernuxWebPlayerHost>"
        "${CMAKE_CURRENT_BINARY_DIR}/infernux-runtime.wasm"
        "${CMAKE_CURRENT_BINARY_DIR}/infernux-runtime.data"
        "${CMAKE_CURRENT_BINARY_DIR}/Player.inxmanifest"
        "${_web_player_destination}"
    DEPENDS InfernuxWebPlayerHost
    VERBATIM
)
