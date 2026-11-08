"""Check the installed Web payload, without looking for a compiler or checkout."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

from infernux.engine.build import (
    BuildDiagnostic, BuildTargetId, CapabilityReport, DiagnosticSeverity,
)
from infernux.version import ENGINE_RELEASE, ENGINE_VERSION
from infernux.plugins.platform_releases import PLAYER_SOURCE_REVISION

from .native_payload import inspect_player_payload, inspect_shader_tools


def inspect_web_toolchain(target: BuildTargetId | str) -> CapabilityReport:
    root = Path(__file__).resolve().parent
    details: dict[str, object] = {"target": str(target), "player_root": str(root / "player")}
    try:
        if sys.platform not in {"win32", "linux"} or platform.machine().lower() not in {
            "amd64", "x86_64",
        }:
            raise ValueError("The Web plugin requires a Windows or Linux x64 editor")
        host = "windows-x64" if sys.platform == "win32" else "linux-x64"
        details.update(inspect_shader_tools(root / "tools", host=host))
    except (OSError, ValueError) as error:
        return CapabilityReport(False, (
            BuildDiagnostic(
                DiagnosticSeverity.ERROR, "web.player.payload", str(error),
                source="infernux/platform-web",
            ),
        ), details)
    try:
        details["player"] = inspect_player_payload(root / "player", engine=ENGINE_VERSION, engine_release=ENGINE_RELEASE,
                                                   source_revision=PLAYER_SOURCE_REVISION)
    except (OSError, ValueError) as error:
        from infernux.engine.player_runtime_repair import rebuild_diagnostic
        details["runtime_rebuild_required"] = True
        return CapabilityReport(True, (
            rebuild_diagnostic(error, source="infernux/platform-web"),
        ), details)
    return CapabilityReport(True, (), details)


__all__ = ["inspect_web_toolchain"]
