"""Packaging tests run without installing Infernux or a platform SDK."""

import json
import shutil
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import release


def _python_runtime_manifest():
    return {
        "implementation": "cpython",
        "version": "3.13.15",
        "abi": "cp313",
        "platform": "emscripten",
        "platform_version": "4.0.10",
        "architecture": "wasm32",
        "extension_linkage": "static-built-in",
        "required_imports": ["numpy"],
        "packages": [
            {
                "name": "numpy",
                "version": "2.2.5",
                "imports": ["numpy"],
                "python_abi": "cp313",
                "platform": "emscripten",
                "platform_version": "4.0.10",
                "architecture": "wasm32",
                "linkage": "static-built-in",
                "source_url": "https://files.pythonhosted.org/packages/source/n/numpy/numpy-2.2.5.tar.gz",
                "source_sha256": "a9c0d994680cd991b1cb772e8b297340085466a6fe964bc9d4e80f5e2f43c291",
                "payload_hash_algorithm": "infernux-tree-sha256-v1",
                "payload_sha256": "1" * 64,
                "payload_bytes": 1,
                "payload_files": 1,
                "license": "BSD-3-Clause",
            }
        ],
    }


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        source = Path(__file__).resolve().parents[1]
        shutil.copytree(source / "package", self.root / "package",
                        ignore=shutil.ignore_patterns("player", "tools", "__pycache__"))
        self.metadata = json.loads((self.root / "package/inx_package.json").read_text(encoding="utf-8"))
        self.tag = f"v{self.metadata['version']}"
        self.payload = self.root / "package/editor/infernux_web"
        player = self.root / "player-output"
        player.mkdir()
        self.player = player
        (player / "Player.inxmanifest").write_text(json.dumps({
            "$schema": "infernux.web_player", "engine": self.metadata["engine"].removeprefix(">="), "platform": "web",
            "architecture": "wasm32", "python_abi": "cp313", "configuration": "Release",
            "python_runtime": _python_runtime_manifest(),
        }), encoding="utf-8")
        (player / "infernux-runtime.wasm").write_bytes(b"\0asm\x01\0\0\0")
        (player / "infernux-runtime.js").write_bytes(b"// runtime fixture")
        (player / "infernux-runtime.data").write_bytes(b"python fixture")
        self.tools = self.root / "tools-output"
        for host, suffix in (("windows-x64", ".exe"), ("linux-x64", "")):
            directory = self.tools / host
            directory.mkdir(parents=True)
            header = b"MZ" if suffix else b"\x7fELF\x02\x01" + bytes(12) + (62).to_bytes(2, "little")
            for name in ("tint", "glslangValidator"):
                (directory / f"{name}{suffix}").write_bytes(header)
        for module in (release, release.package):
            override = patch.object(module, "__file__", str(self.root / Path(module.__file__).name))
            override.start()
            self.addCleanup(override.stop)

    def test_release_requires_both_host_tool_payloads(self):
        (self.tools / "linux-x64/tint").unlink()
        with self.assertRaises(FileNotFoundError):
            release.build_release(
                self.tag,
                player_payload=self.player,
                tools_payload=self.tools,
            )
        self.assertFalse((self.root / "dist").exists())

    def test_release_rejects_wrong_wasm(self):
        (self.player / "infernux-runtime.wasm").write_bytes(b"not wasm")
        with self.assertRaisesRegex(ValueError, "WebAssembly"):
            release.build_release(
                self.tag,
                player_payload=self.player,
                tools_payload=self.tools,
            )

    def test_package_and_manifest(self):
        root = Path(__file__).resolve().parents[1]
        source = self.metadata
        with tempfile.TemporaryDirectory() as temporary:
            artifact, manifest = release.build_release(
                self.tag,
                Path(temporary),
                player_payload=self.player,
                tools_payload=self.tools,
            )
            document = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(document["artifact"]["name"], artifact.name)
            self.assertEqual(document["reference"], source["reference"])
            self.assertEqual(document["engine"], source["engine"])
            self.assertEqual(artifact.read_bytes()[:8], b"INXPKG\0\0")
            self.assertGreater(artifact.stat().st_size, 1024)
        self.assertFalse((self.payload / "player").exists())
        self.assertFalse((self.payload / "tools").exists())
        self.assertTrue((root / "package/plugin_pages/media/overview.png").is_file())
        for name in ("README.md", "README.zh-CN.md"):
            self.assertIn("package/plugin_pages/media/overview.png", (root / name).read_text(encoding="utf-8"))

    def test_reject_mismatched_tag(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "Release tag must match"):
                release.build_release(
                    "v999.0.0",
                    Path(temporary),
                    player_payload=self.player,
                    tools_payload=self.tools,
                )
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_player_publication_contract_is_out_of_source(self):
        root = Path(__file__).resolve().parents[1]
        cmake = (root / "native/CMakeLists.txt").read_text(encoding="utf-8")
        build = (root / "native/build.py").read_text(encoding="utf-8")
        tools_cmake = (root / "native/tools/CMakeLists.txt").read_text(
            encoding="utf-8"
        )

        assert (
            'set(INFERNUX_WEB_PLAYER_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/publish/player" CACHE PATH'
            in cmake
        )
        assert "${CMAKE_CURRENT_SOURCE_DIR}/../package/editor/infernux_web/player" not in cmake
        assert "package_web_plugin" not in cmake
        assert "-DINFERNUX_WEB_PLAYER_OUTPUT_DIR=" in build
        assert 'build_root / "publish/player"' in build
        assert (
            'set(INFERNUX_WEB_TOOLS_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/publish/tools"'
            in tools_cmake
        )
        assert (
            'set(_tools "${INFERNUX_WEB_TOOLS_OUTPUT_DIR}/${_host}")'
            in tools_cmake
        )
        assert "${CMAKE_CURRENT_SOURCE_DIR}/../../package/editor/infernux_web/tools" not in tools_cmake

        workflow = (root / ".github/workflows/release.yml").read_text(encoding="utf-8")
        assert "engine/out/build/web-precompiled-player/publish/player/" in workflow
        assert "package/editor/infernux_web/player/" not in workflow
        assert "engine/out/build/web-shader-tools/publish/tools/" in workflow
        assert "package/editor/infernux_web/tools/" not in workflow
        assert "--player-payload artifacts/web-player" in workflow
        assert "--tools-payload artifacts/web-tools" in workflow


if __name__ == "__main__":
    unittest.main()
