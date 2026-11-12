#pragma once

#include <Python.h>

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
} // namespace infernux::web

PyMODINIT_FUNC PyInit__InfernuxWebHost();

[[nodiscard]] bool InfernuxWebFindShaderSource(const std::string &name, const char *stage, std::string &source);
void InfernuxWebSetParticleRuntime(infernux::web::WebParticleRuntime *runtime) noexcept;
void InfernuxWebSetScreenUIRenderer(infernux::web::WebScreenUIRenderer *renderer) noexcept;
void InfernuxWebSetComputeHost(infernux::rhi::ComputeHost *host) noexcept;
void InfernuxWebSetFrameWork(infernux::FrameWorkOwner *owner) noexcept;
void InfernuxWebEndTextInput() noexcept;
