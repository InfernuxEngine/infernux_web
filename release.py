"""Build this repository's standalone InxPackage and GitHub release manifest."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from runpy import run_path

import package


def _require_native_payloads(
    root: Path,
    player_payload: Path,
    tools_payload: Path,
    engine_version: str,
) -> None:
    directory = root / "package/editor/infernux_web"
    contract = run_path(str(directory / "native_payload.py"))
    manifest = contract["inspect_player_payload"](player_payload, engine=engine_version)
    if manifest["configuration"] != "Release":
        raise ValueError("Published Web Player must use the Release configuration")
    with (player_payload / "infernux-runtime.wasm").open("rb") as stream:
        if stream.read(8) != b"\0asm\x01\0\0\0":
            raise ValueError("Web Player payload is not a WebAssembly module")
    for host in contract["HOSTS"]:
        tools = contract["inspect_shader_tools"](tools_payload, host=host)
        for path in tools.values():
            with Path(path).open("rb") as stream:
                header = stream.read(20)
            if host == "windows-x64":
                if header[:2] != b"MZ":
                    raise ValueError(f"Web shader tool is not a Windows executable: {path}")
            elif header[:6] != b"\x7fELF\x02\x01" or int.from_bytes(header[18:20], "little") != 62:
                raise ValueError(f"Web shader tool is not a Linux x64 executable: {path}")


def build_release(
    tag: str,
    output: Path | None = None,
    *,
    player_payload: Path,
    tools_payload: Path,
) -> tuple[Path, Path]:
    root = Path(__file__).resolve().parent
    metadata = json.loads((root / "package/inx_package.json").read_text(encoding="utf-8"))
    expected = f"v{metadata['version']}"
    if tag != expected:
        raise ValueError(f"Release tag must match package version: {expected}")
    player_payload = player_payload.expanduser().resolve()
    tools_payload = tools_payload.expanduser().resolve()
    _require_native_payloads(root, player_payload, tools_payload, metadata["engine"].removeprefix(">="))
    destination = output if output is not None else root / "dist"
    destination.mkdir(parents=True, exist_ok=True)
    stem = metadata["reference"].replace("/", ".")
    with tempfile.TemporaryDirectory(prefix="infernux-web-release-") as temporary:
        staged_package = Path(temporary) / "package"
        shutil.copytree(
            root / "package",
            staged_package,
            ignore=shutil.ignore_patterns("player", "tools", "__pycache__"),
        )
        shutil.copytree(
            player_payload,
            staged_package / "editor/infernux_web/player",
        )
        shutil.copytree(
            tools_payload,
            staged_package / "editor/infernux_web/tools",
        )
        artifact = package.build(
            destination / f"{stem}.inxpkg",
            package_root=staged_package,
        )
    manifest = destination / f"{stem}.release.json"
    document = {
        "$schema": "infernux.plugin_release",
        "reference": metadata["reference"],
        "version": metadata["version"],
        "engine": metadata["engine"],
        "artifact": {"name": artifact.name},
        "generator": "Infernux platform package release.py",
        "release_tag": tag,
    }
    manifest.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return artifact, manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag", help="v followed by package/inx_package.json version")
    parser.add_argument("--player-payload", required=True, type=Path)
    parser.add_argument("--tools-payload", required=True, type=Path)
    arguments = parser.parse_args()
    for path in build_release(
        arguments.tag,
        player_payload=arguments.player_payload,
        tools_payload=arguments.tools_payload,
    ):
        print(path)
