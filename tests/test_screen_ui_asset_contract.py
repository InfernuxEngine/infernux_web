"""Web screen UI resources keep GUID identity until native resolution."""

import ast
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def test_native_texture_bridge_rejects_path_identity_fallback() -> None:
    source = (ROOT / "native/InfernuxWebHostModule.cpp").read_text(encoding="utf-8")
    loader = (ROOT / "native/WebTextureAssetLoad.h").read_text(encoding="utf-8")
    function = source.split("PyObject *ScreenUIResolveTexture", 1)[1].split(
        "PyObject *ScreenUIUploadImage", 1
    )[0]

    assert "state.load.Poll(guid)" in function
    assert "GetGuidFromPath" not in function
    assert "GetGuidFromPath" not in loader
    guid_check = loader.index("if (!database->ContainsGuid(guid))")
    begin_load = loader.index("registry.BeginLoadAsset(guid, ResourceType::Texture)")
    assert guid_check < loader.index("return Result::Missing;", guid_check) < begin_load


def test_screen_and_world_ui_bind_groups_include_material_sampler_identity() -> None:
    renderer = (ROOT / "native/WebScreenUIRenderer.cpp").read_text(encoding="utf-8")
    host = (ROOT / "native/InfernuxWebHostModule.cpp").read_text(encoding="utf-8")
    loader = (ROOT / "native/WebTextureAssetLoad.h").read_text(encoding="utf-8")
    sampler = renderer.split("WebScreenUIRenderer::ResolveMaterialSampler", 1)[1].split(
        "ImDrawList *WebScreenUIRenderer::DrawList", 1
    )[0]
    screen = renderer.split("bool WebScreenUIRenderer::Render(", 1)[1].split(
        "bool WebScreenUIRenderer::RenderWorld(", 1
    )[0]
    world = renderer.split("bool WebScreenUIRenderer::RenderWorld(", 1)[1]

    assert 'material->GetTextureSampler("texSampler")' in sampler
    assert "key.push_back(':')" in sampler
    assert "m_materialSamplers.emplace(key, sampler)" in sampler
    for draw_path in (screen, world):
        assert "std::unordered_map<std::string, wgpu::BindGroup> textureGroups" in draw_path
        assert "ResolveMaterialSampler(" in draw_path
        assert "textureGroups.find(samplerKey)" in draw_path
        assert "textureGroups.emplace(samplerKey," in draw_path
    assert "m_data.filterMode = texture->GetFilterMode()" in loader
    assert "m_data.wrapMode = texture->GetWrapMode()" in loader
    assert "m_data.anisoLevel = texture->GetAnisoLevel()" in loader
    assert "data.texture, state.textureId, data.filterMode, data.wrapMode, data.anisoLevel" in host


def test_python_texture_cache_extracts_asset_reference_guid() -> None:
    source = (ROOT / "native/bootstrap.py").read_text(encoding="utf-8")
    texture_cache = source.split("class _WebScreenUITextureCache", 1)[1].split(
        "def _submit_screen_ui", 1
    )[0]

    assert 'getattr(source, "guid", source)' in texture_cache
    assert "screen_ui_resolve_texture(guid)" in texture_cache


def test_render_effect_recursion_identity_is_guid_only() -> None:
    source = (ROOT / "native/bootstrap.py").read_text(encoding="utf-8")
    function = source.split("def _iter_web_render_effects", 1)[1].split(
        "def _web_render_effect_path", 1
    )[0]

    assert "identity = guid.casefold()" in function
    assert "os.path.abspath(path)" not in function


def test_python_texture_cache_submits_only_the_reference_guid(monkeypatch) -> None:
    source = (ROOT / "native/bootstrap.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    cache_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "_WebScreenUITextureCache"
    )
    namespace = {"Any": Any}
    exec(compile(ast.Module([cache_class], type_ignores=[]), "bootstrap.py", "exec"), namespace)

    submitted = []
    host = ModuleType("_InfernuxWebHost")
    host.screen_ui_resolve_texture = lambda guid: submitted.append(guid) or 91
    monkeypatch.setitem(__import__("sys").modules, "_InfernuxWebHost", host)
    cache = namespace["_WebScreenUITextureCache"]()

    reference = SimpleNamespace(
        guid="8a51dcd72aa64cd5963731d9b4f2507f",
        path_hint="Assets/Textures/stale.png",
    )
    assert cache.get(reference) == 91
    assert submitted == [reference.guid]
    assert cache.get(reference) == 91
    assert submitted == [reference.guid]


def test_native_text_bridge_preserves_clip_and_fallback_font_contract() -> None:
    host = (ROOT / "native/InfernuxWebHostModule.cpp").read_text(encoding="utf-8")
    parser = host.split("bool ParseFallbackFontPaths", 1)[1].split(
        "PyObject *ScreenUIAddImage", 1
    )[0]
    add_text = host.split("PyObject *ScreenUIAddText", 1)[1].split(
        "PyObject *ScreenUIMeasureText", 1
    )[0]
    measure_text = host.split("PyObject *ScreenUIMeasureText", 1)[1].split(
        "PyObject *ScreenUIResolveTexture", 1
    )[0]
    renderer = (ROOT / "native/WebScreenUIRenderer.cpp").read_text(encoding="utf-8")

    assert parser.index("if (value == nullptr)") < parser.index("PySequence_Fast(")
    assert "PyUnicode_Check(value) || PyBytes_Check(value)" in parser
    assert parser.index("if (count == 0)") < parser.index("paths.reserve(")
    assert "paths.emplace_back(path, static_cast<size_t>(size))" in parser
    assert parser.index("paths.emplace_back(") < parser.index("Py_DECREF(sequence);\n    return true;")
    assert "argumentCount >= 21" in add_text
    assert "PyTuple_GetItem(arguments, 20)" in add_text
    assert "ParseFallbackFontPaths(PyTuple_GetItem(arguments, 21)" in add_text
    assert "clip, fallbackFontPaths" in add_text
    assert '"sddsdd|O:screen_ui_measure_text"' in measure_text
    assert "ParseFallbackFontPaths(fallbackFontPathsObject" in measure_text
    assert "fallbackFontPaths)" in measure_text
    assert "fallbackFontPaths});" in renderer
    assert "if (clip)" in renderer
    assert "draw->PushClipRect({minX, minY}, {maxX, maxY}, true);" in renderer
