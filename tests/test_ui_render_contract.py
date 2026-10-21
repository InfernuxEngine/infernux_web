"""The Web UI backend must consume, not merely accept, runtime UI packets."""

import ast
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def test_python_bridge_submits_guid_generation_and_world_transform(monkeypatch):
    source = (ROOT / "native/bootstrap.py").read_text(encoding="utf-8")
    cls = next(
        node for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == "_WebScreenUIRenderer"
    )
    calls = []
    host = ModuleType("_InfernuxWebHost")
    host.screen_ui_set_material_binding = lambda *args: calls.append(("material", args))
    host.screen_ui_begin_world_element = lambda *args: calls.append(("world", args))
    host.screen_ui_end_world_element = lambda: calls.append(("end", ()))
    monkeypatch.setitem(__import__("sys").modules, "_InfernuxWebHost", host)
    namespace = {"Any": Any}
    exec(compile(ast.Module([cls], type_ignores=[]), "bootstrap.py", "exec"), namespace)
    renderer = namespace["_WebScreenUIRenderer"]()
    transform = SimpleNamespace(
        position=SimpleNamespace(x=1.0, y=2.0, z=3.0),
        rotation=SimpleNamespace(x=0.0, y=0.0, z=0.6, w=0.8),
        scale=SimpleNamespace(x=0.01, y=2.0, z=3.0),
    )
    obj = SimpleNamespace(transform=transform, layer=6)
    renderer.set_material_binding(2, "a" * 32, 11, "ui|shader=fixture", (0.2, 0.4, 0.6, 0.8), True, 0.35)
    renderer.begin_world_object(obj, 50.0, 25.0, True, False, True, 9)
    renderer.end_world_element()

    assert calls[0] == ("material", (2, "a" * 32, 11, "ui|shader=fixture",
                                      (0.2, 0.4, 0.6, 0.8), True, 0.35))
    matrix = calls[1][1][0]
    import pytest

    assert matrix == pytest.approx((0.28, 0.96, 0, 0, -0.96, 0.28, 0, 0, 0, 0, 1, 0, 1, 2, 3, 1))
    assert calls[1][1][1:] == (50.0, 25.0, 6, True, False, True, 9)
    assert calls[2] == ("end", ())


def test_native_world_draw_is_depth_aware_and_material_shader_is_consumed():
    renderer = (ROOT / "native/WebScreenUIRenderer.cpp").read_text(encoding="utf-8")
    frame = (ROOT / "native/main.cpp").read_text(encoding="utf-8")

    assert "m_worldDepthPipeline" in renderer
    assert "m_worldTopPipeline" in renderer
    assert "cullingMask & (1u << span.layer)" in renderer
    assert "std::stable_sort(draws.begin(), draws.end()" in renderer
    assert "ResolveMaterialPipeline(draw.binding, true, draw.top)" in renderer
    assert "ResolveMaterialPipeline(binding, false, false)" in renderer
    assert 'InfernuxWebFindShaderSource(vertex.shaderId, "vertex"' in renderer
    assert 'InfernuxWebFindShaderSource(fragment.shaderId, "fragment"' in renderer
    assert "std::to_string(binding.pipelineKey.size()) + ':' + binding.pipelineKey" in renderer
    assert 'if (ignoredOccluderId != 0)' in renderer
    assert 'does not yet support ignored-occluder depth exclusion' in renderer
    assert "g_screenUIRenderer.RenderWorld(pass" in frame
    assert frame.index("g_screenUIRenderer.RenderWorld(pass") < frame.index("pass.End();", frame.index("g_screenUIRenderer.RenderWorld(pass"))


def test_browser_ui_input_uses_shared_world_projection():
    bootstrap = (ROOT / "native/bootstrap.py").read_text(encoding="utf-8")
    event_section = bootstrap.split("def _process_screen_ui_events", 1)[1].split(
        "def infernux_web_tick", 1
    )[0]
    assert "runtime_ui_scenes(_player_scene_manager)" in event_section
    assert (
        "collect_runtime_ui_input_surfaces(*runtime_ui_scenes(_player_scene_manager))"
        in event_section
    )
    assert "map_runtime_ui_pointer(" in event_section
    assert "scene.effective_game_camera" in event_section
    assert "_screen_ui_event_processor.process_pointers(" in event_section


def test_text_preparation_publishes_font_atlas_before_ui_geometry():
    renderer = (ROOT / "native/WebScreenUIRenderer.cpp").read_text(encoding="utf-8")
    measure = renderer.split(
        "std::pair<float, float> WebScreenUIRenderer::MeasureText", 1
    )[1].split("uint64_t WebScreenUIRenderer::UploadTexture", 1)[0]

    assert measure.index("textlayout::LayoutText") < measure.index("RefreshFontAtlas()")
    assert "!ImGui::GetIO().Fonts->IsBuilt()" in measure
    assert "Web UI font atlas publication failed during text preparation" in measure


def test_text_preparation_remeasures_cached_layouts_before_ui_geometry():
    runtime = (
        ROOT.parents[2] / "python" / "infernux" / "engine"
        / "runtime_screen_ui.py"
    ).read_text(encoding="utf-8")
    preparation = runtime.split("def prepare_text_layouts", 1)[1].split(
        "def _submit_world_element", 1
    )[0]

    assert 'getattr(element, "prepare_text_layout", None)' in preparation
    assert preparation.count("prepare(renderer.measure_text") == 2


def test_button_labels_join_text_atlas_preparation():
    button = (
        ROOT.parents[2] / "python" / "infernux" / "ui" / "ui_button.py"
    ).read_text(encoding="utf-8")

    preparation = button.split("def prepare_text_layout", 1)[1].split(
        "# ── Events", 1
    )[0]
    assert "ui_font_paths(self)" in preparation
    assert "measure_text(*arguments" in preparation
