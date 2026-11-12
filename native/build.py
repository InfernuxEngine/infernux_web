"""Build and publish the project-independent Web runtime (release engineering)."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from urllib.request import urlopen


WEBGPU_RELEASE = "v20260423.175430"


def _prepare_webgpu_port(build_root: Path) -> Path:
    """Use the same Dawn header/JS pair in local builds and release CI."""
    port = build_root / "toolchain" / f"emdawnwebgpu-{WEBGPU_RELEASE}.remoteport.py"
    if not port.is_file():
        port.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(
            f"https://github.com/google/dawn/releases/download/{WEBGPU_RELEASE}/{port.name}", timeout=30,
        ) as response:
            contents = response.read()
        temporary = port.with_suffix(".tmp")
        temporary.write_bytes(contents)
        temporary.replace(port)
    return port


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emsdk", required=True, type=Path)
    parser.add_argument("--cpython-source", required=True, type=Path)
    parser.add_argument("--python-runtime-manifest", required=True, type=Path)
    parser.add_argument("--numpy-payload-root", required=True, type=Path)
    parser.add_argument("--host-python", required=True, type=Path)
    parser.add_argument("--build-root", required=True, type=Path)
    parser.add_argument(
        "--player-output",
        type=Path,
        help="Player payload destination (default: BUILD_ROOT/publish/player)",
    )
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    build_root = args.build_root.resolve()
    webgpu_port = _prepare_webgpu_port(build_root)
    player_output = (
        args.player_output.resolve()
        if args.player_output is not None
        else build_root / "publish/player"
    )
    subprocess.run([
        str(args.emsdk.resolve() / "upstream/emscripten/emcmake"), "cmake",
        "-S", str(source), "-B", str(build_root), "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DINFERNUX_WEB_CPYTHON_SOURCE={args.cpython_source.resolve()}",
        f"-DINFERNUX_WEB_CPYTHON_BUILD={args.cpython_source.resolve() / 'builddir/emscripten-browser'}",
        f"-DINFERNUX_WEB_PYTHON_RUNTIME_MANIFEST={args.python_runtime_manifest.resolve()}",
        f"-DINFERNUX_WEB_NUMPY_PAYLOAD_ROOT={args.numpy_payload_root.resolve()}",
        f"-DINFERNUX_WEB_HOST_PYTHON={args.host_python.resolve()}",
        f"-DINFERNUX_WEBGPU_PORT={webgpu_port}",
        f"-DINFERNUX_WEB_PLAYER_OUTPUT_DIR={player_output}",
    ], check=True)
    subprocess.run(["cmake", "--build", str(build_root),
                    "--target", "prebuild_web_player", "--parallel", "2"], check=True)


if __name__ == "__main__":
    main()
