"""Exercise the native fullscreen WGSL input contract with the host C++ compiler."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_fullscreen_shader_contract_compiles_and_rejects_mismatched_bindings(tmp_path):
    source = ROOT / "tests" / "fullscreen_shader_contract.cpp"
    output = tmp_path / ("fullscreen_shader_contract.exe" if os.name == "nt" else "fullscreen_shader_contract")
    if os.name == "nt":
        compiler = shutil.which("cl")
        if compiler is None:
            pytest.skip("MSVC is not available in this test environment")
        vcvars = Path(compiler).parents[6] / "Auxiliary" / "Build" / "vcvars64.bat"
        if not vcvars.is_file():
            pytest.skip("The MSVC x64 environment script is unavailable")
        command = (
            f'call "{vcvars}" >nul && "{compiler}" /nologo /EHsc /std:c++17 '
            f'"{source}" "/Fe:{output}" "/Fo:{tmp_path / "fullscreen_shader_contract.obj"}"'
        )
    else:
        compiler = shutil.which("c++")
        if compiler is None:
            pytest.skip("A C++ compiler is not available in this test environment")
        command = [compiler, "-std=c++17", str(source), "-o", str(output)]
    compile_result = subprocess.run(command, capture_output=True, text=True, shell=os.name == "nt")
    assert compile_result.returncode == 0, compile_result.stdout + compile_result.stderr
    contract_result = subprocess.run([str(output)], capture_output=True, text=True)
    assert contract_result.returncode == 0, contract_result.stdout + contract_result.stderr


def test_web_fullscreen_host_uses_the_four_argument_contract():
    source = " ".join((ROOT / "native" / "main.cpp").read_text(encoding="utf-8").split())
    assert "uint32_t inputCount, uint32_t inputBufferMask) override" in source
    assert "ValidateFullscreenShaderInputs(source, inputCount, inputBufferMask, contractError)" in source


def test_storage_access_layout_and_explicit_compute_widening():
    engine_root = ROOT.parents[2]
    descriptors = (engine_root / "cpp/infernux/function/renderer/rhi/RhiDescriptors.h").read_text(encoding="utf-8")
    fullscreen = (engine_root / "cpp/infernux/function/renderer/FullscreenRenderer.cpp").read_text(encoding="utf-8")
    particles = (engine_root / "cpp/infernux/function/renderer/particle/ParticleGpuRuntime.cpp").read_text(encoding="utf-8")
    web_rhi = (ROOT / "native/WebGpuRhiDevice.cpp").read_text(encoding="utf-8")

    assert "bool readOnlyStorage = false;" in descriptors
    assert "bool widenReadOnlyStorage = false;" in descriptors
    assert "bool widenReadOnlyStorage = false) noexcept" in descriptors
    assert "binding.readOnlyStorage = binding.type == rhi::BindingType::StorageBuffer;" in fullscreen
    assert "source.readOnlyStorage ? wgpu::BufferBindingType::ReadOnlyStorage" in web_rhi
    assert "if (desc.widenReadOnlyStorage)" in web_rhi
    assert "ShaderModuleDesc::FromWgsl(kernel.wgsl, kernel.wgslByteSize, true)" in particles
