#pragma once

#include <Python.h>

#include <function/renderer/GraphEvents.h>
#include <functional>
#include <string>

namespace infernux::rhi
{
class ComputeHost;
}
namespace infernux
{
class FrameWorkOwner;
}

namespace infernux::web
{
class WebParticleRuntime;
class WebScreenUIRenderer;
class WebSceneRenderer;
struct WebGraphServices
{
    GraphEventInterests interests;
    GraphEventQueue events;
    std::function<std::shared_ptr<GraphBufferCapture>(uint64_t, const std::string &)> capture;
};
} // namespace infernux::web

PyMODINIT_FUNC PyInit__InfernuxWebHost();

void InfernuxWebSetParticleRuntime(infernux::web::WebParticleRuntime *runtime) noexcept;
void InfernuxWebSetScreenUIRenderer(infernux::web::WebScreenUIRenderer *renderer) noexcept;
void InfernuxWebSetSceneRenderer(infernux::web::WebSceneRenderer *renderer) noexcept;
void InfernuxWebSetGraphServices(infernux::web::WebGraphServices *services) noexcept;
void InfernuxWebSetComputeHost(infernux::rhi::ComputeHost *host) noexcept;
void InfernuxWebSetFrameWork(infernux::FrameWorkOwner *owner) noexcept;
void InfernuxWebEndTextInput() noexcept;
