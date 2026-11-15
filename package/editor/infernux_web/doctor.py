"""Check the installed Web payload, without looking for a compiler or checkout."""

from __future__ import annotations

from pathlib import Path

from infernux.engine.build import (
    BuildDiagnostic, BuildTargetId, CapabilityReport, DiagnosticSeverity,
)
from infernux.version import ENGINE_RELEASE, ENGINE_VERSION
from infernux.plugins.platform_releases import PLAYER_SOURCE_REVISION

from .native_payload import inspect_player_payload


def inspect_web_toolchain(target: BuildTargetId | str) -> CapabilityReport:
    root = Path(__file__).resolve().parent
    details: dict[str, object] = {"target": str(target), "player_root": str(root / "player")}
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
