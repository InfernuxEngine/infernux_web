"""The browser host must preserve the project's simulation clock."""

import ast
import copy
import json
import os
from pathlib import Path
import sys
import types

import pytest


BOOTSTRAP = Path(__file__).resolve().parents[1] / "native/bootstrap.py"


def _function(name, namespace):
    node = next(
        node for node in ast.parse(BOOTSTRAP.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    exec(compile(ast.Module([node], type_ignores=[]), str(BOOTSTRAP), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("authored", [False, True])
@pytest.mark.parametrize("fixed_step,maximum_delta", [(0.02, 1.0 / 3.0), (0.01, 0.5), (0.04, 0.04)])
def test_physics_configuration_preserves_simulation_clock(monkeypatch, tmp_path, authored, fixed_step, maximum_delta):
    configuration = {
        "fixed_delta_time": fixed_step, "max_fixed_delta_time": maximum_delta,
        "gravity": [0.0, -9.81, 0.0], "point_velocity_sleep_threshold": 0.005,
        "temp_allocator_mb": 256, "max_jobs": 4096, "max_barriers": 16,
        "max_bodies": 65536, "max_body_pairs": 65536,
        "max_contact_constraints": 65536, "max_concurrency": 0,
    }
    settings_file = tmp_path / "PhysicsSettings.json"
    if authored:
        settings_file.write_text(json.dumps(configuration), encoding="utf-8")
    settings = types.SimpleNamespace(
        settings_path=lambda root: str(settings_file),
        load=lambda root: copy.deepcopy(configuration),
    )
    physics = types.ModuleType("infernux.physics")
    physics.settings = settings
    package = types.ModuleType("infernux")
    package.__path__ = []
    timing = types.ModuleType("infernux.timing")
    timing.Time = types.SimpleNamespace()
    published = []
    host = types.ModuleType("_InfernuxWebHost")
    host.configure_physics = lambda document: published.append(json.loads(document))
    for name, module in {
        "infernux": package, "infernux.physics": physics, "infernux.timing": timing,
        "_Infernux": types.ModuleType("_Infernux"), "_InfernuxWebHost": host,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    configure = _function("infernux_web_configure_physics", {
        "os": os, "json": json, "_runtime_data_root": str(tmp_path),
        "_install_platform_runtime_api": lambda module: None,
    })

    assert configure() is True
    assert len(published) == 1
    actual = published[0]
    assert actual["fixed_delta_time"] == fixed_step
    assert actual["max_fixed_delta_time"] == maximum_delta
    assert timing.Time._fixed_delta_time == fixed_step
    assert timing.Time._maximum_delta_time == maximum_delta
    assert actual["max_concurrency"] == 1
    assert actual["gravity"] == configuration["gravity"]
    assert actual["point_velocity_sleep_threshold"] == configuration["point_velocity_sleep_threshold"]
    if authored:
        assert actual == {**configuration, "max_concurrency": 1}
        assert json.loads(settings_file.read_text(encoding="utf-8")) == configuration
    else:
        assert actual["temp_allocator_mb"] == 32
        assert actual["max_bodies"] == 8192
        assert not settings_file.exists()


def test_player_session_receives_unscaled_browser_elapsed_time(monkeypatch):
    compute = types.ModuleType("infernux.compute")
    collected = []
    compute._collect_commands = collected.append
    monkeypatch.setitem(sys.modules, "infernux.compute", compute)
    events = types.ModuleType("infernux.engine.runtime_event_queue")
    events.drain = lambda: None
    monkeypatch.setitem(sys.modules, "infernux.engine.runtime_event_queue", events)
    ticks = []
    tick = _function("infernux_web_tick", {
        "_player_session": types.SimpleNamespace(tick=ticks.append, _infernux_compute_host=17),
        "_web_splash": None, "_player_activated": True, "_frame_count": 0,
        "_complete_pending_audio_activation": lambda: None,
        "_process_screen_ui_events": lambda delta: None,
        "_submit_screen_ui": lambda: None,
    })
    assert tick(0.4) is False
    assert ticks == [0.4]
    assert collected == [17]
