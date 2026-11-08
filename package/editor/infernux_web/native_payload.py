"""The platform plugin's precompiled browser and host-tool layout."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

PLAYER_FILES = ("infernux-runtime.js", "infernux-runtime.wasm", "infernux-runtime.data")
HOSTS = ("windows-x64", "linux-x64")
WEB_CPYTHON_VERSION = "3.13.15"
WEB_PYTHON_ABI = "cp313"
WEB_EMSCRIPTEN_VERSION = "4.0.10"
WEB_NUMPY_VERSION = "2.2.5"
WEB_NUMPY_SOURCE_URL = "https://files.pythonhosted.org/packages/source/n/numpy/numpy-2.2.5.tar.gz"
WEB_NUMPY_SOURCE_SHA256 = "a9c0d994680cd991b1cb772e8b297340085466a6fe964bc9d4e80f5e2f43c291"
WEB_REQUIRED_PYTHON_IMPORTS = ("numpy",)
WEB_PYTHON_PAYLOAD_HASH = "infernux-tree-sha256-v1"
WEB_NUMPY_BUILTIN_MODULES = (
    "numpy._core._multiarray_umath",
    "numpy.fft._pocketfft_umath",
    "numpy.linalg._umath_linalg",
    "numpy.linalg.lapack_lite",
    "numpy.random._bounded_integers",
    "numpy.random._common",
    "numpy.random._generator",
    "numpy.random._mt19937",
    "numpy.random._pcg64",
    "numpy.random._philox",
    "numpy.random._sfc64",
    "numpy.random.bit_generator",
    "numpy.random.mtrand",
)


def python_package_payload_identity(root: Path) -> dict[str, object]:
    """Hash a package's Python files and static archives without tar metadata."""

    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"Web Python package payload must be a real directory: {root}")
    files: list[tuple[str, Path]] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Web Python package payload contains a symlink: {path}")
        if path.is_file():
            files.append((path.relative_to(root).as_posix(), path))
    if not files:
        raise ValueError(f"Web Python package payload is empty: {root}")

    digest = hashlib.sha256()
    digest.update(b"infernux.web_python_payload.v1\0")
    payload_bytes = 0
    for relative, path in sorted(files):
        size = path.stat().st_size
        payload_bytes += size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return {
        "algorithm": WEB_PYTHON_PAYLOAD_HASH,
        "sha256": digest.hexdigest(),
        "bytes": payload_bytes,
        "files": len(files),
    }


def verify_python_package_payload(
    root: Path, runtime: object, *, package_name: str
) -> dict[str, object]:
    """Match one on-disk preload/link payload to its audited manifest entry."""

    document = inspect_python_runtime(runtime)
    packages = document["packages"]
    matches = [package for package in packages if package.get("name") == package_name]
    if len(matches) != 1:
        raise ValueError(
            f"Web Player Python runtime must contain exactly one package named {package_name}"
        )
    package = matches[0]
    if package_name == "numpy":
        inspect_numpy_static_payload(root)
    identity = python_package_payload_identity(root)
    expected = {
        "algorithm": package.get("payload_hash_algorithm"),
        "sha256": package.get("payload_sha256"),
        "bytes": package.get("payload_bytes"),
        "files": package.get("payload_files"),
    }
    mismatched = [field for field in identity if identity[field] != expected[field]]
    if mismatched:
        raise ValueError(
            f"Web Player Python package payload does not match manifest: {package_name} "
            f"({', '.join(mismatched)})"
        )
    return identity


def inspect_numpy_static_payload(root: Path) -> dict[str, object]:
    """Reject incomplete or dynamic NumPy payload layouts before hashing them."""

    layout_path = root / "numpy_static_link.json"
    if not layout_path.is_file():
        raise ValueError("Web NumPy static link manifest is missing")
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    if not isinstance(layout, dict) or layout.get("schema") != "infernux.numpy_static_payload.v1":
        raise ValueError("Web NumPy static link manifest is invalid")
    if layout.get("modules") != list(WEB_NUMPY_BUILTIN_MODULES):
        raise ValueError("Web NumPy built-in module closure is not exact")
    if layout.get("registry") != "numpy_builtin_registry.c":
        raise ValueError("Web NumPy built-in registry identity is invalid")
    if layout.get("preload") != "site-packages":
        raise ValueError("Web NumPy preload identity is invalid")
    archives = layout.get("archives")
    if (
        not isinstance(archives, list)
        or len(archives) < len(WEB_NUMPY_BUILTIN_MODULES)
        or archives != list(dict.fromkeys(archives))
    ):
        raise ValueError("Web NumPy static archive closure is invalid")
    for relative in archives:
        if not isinstance(relative, str) or not re.fullmatch(
            r"archives/[A-Za-z0-9_.-]+\.a", relative
        ):
            raise ValueError("Web NumPy static archive path is invalid")
        if not (root / relative).is_file():
            raise ValueError(f"Web NumPy static archive is missing: {relative}")
    for module in WEB_NUMPY_BUILTIN_MODULES:
        if f"archives/{module}.a" not in archives:
            raise ValueError(f"Web NumPy built-in archive is missing: {module}")
    if not (root / "numpy_builtin_registry.c").is_file():
        raise ValueError("Web NumPy built-in registry is missing")
    if not (root / "site-packages/numpy/__init__.py").is_file():
        raise ValueError("Web NumPy Python package is missing")
    forbidden = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.suffix in {".so", ".wasm"}
    ]
    if forbidden:
        raise ValueError("Web NumPy payload contains dynamic modules: " + ", ".join(forbidden))
    return layout


def inspect_python_runtime(document: object) -> dict[str, object]:
    """Validate the immutable Python/WASM dependency closure in a Player payload."""

    if not isinstance(document, dict):
        raise ValueError("Web Player Python runtime manifest is missing")
    expected = {
        "implementation": "cpython",
        "version": WEB_CPYTHON_VERSION,
        "abi": WEB_PYTHON_ABI,
        "platform": "emscripten",
        "platform_version": WEB_EMSCRIPTEN_VERSION,
        "architecture": "wasm32",
        "extension_linkage": "static-built-in",
    }
    mismatched = [name for name, value in expected.items() if document.get(name) != value]
    if mismatched:
        raise ValueError(
            "Web Player Python runtime ABI does not match this engine: "
            + ", ".join(mismatched)
        )

    required_imports = document.get("required_imports")
    if required_imports != list(WEB_REQUIRED_PYTHON_IMPORTS):
        raise ValueError(
            "Web Player Python runtime required-import closure must be exactly: "
            + ", ".join(WEB_REQUIRED_PYTHON_IMPORTS)
        )
    packages = document.get("packages")
    if not isinstance(packages, list) or not packages:
        raise ValueError("Web Player Python runtime package closure is missing")

    package_names: set[str] = set()
    provided_imports: set[str] = set()
    for package in packages:
        if not isinstance(package, dict):
            raise ValueError("Web Player Python runtime package entry is invalid")
        name = package.get("name")
        imports = package.get("imports")
        if not isinstance(name, str) or not name or name in package_names:
            raise ValueError("Web Player Python runtime package identity is invalid")
        if (
            not isinstance(imports, list)
            or not imports
            or any(not isinstance(item, str) or not item for item in imports)
            or imports != sorted(set(imports))
        ):
            raise ValueError(f"Web Player Python package import closure is invalid: {name}")
        package_names.add(name)
        overlap = provided_imports.intersection(imports)
        if overlap:
            raise ValueError(
                "Web Player Python import is provided by multiple packages: "
                + ", ".join(sorted(overlap))
            )
        provided_imports.update(imports)

        compatible = {
            "python_abi": WEB_PYTHON_ABI,
            "platform": "emscripten",
            "platform_version": WEB_EMSCRIPTEN_VERSION,
            "architecture": "wasm32",
            "linkage": "static-built-in",
        }
        if any(package.get(field) != value for field, value in compatible.items()):
            raise ValueError(f"Web Player Python package ABI is incompatible: {name}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(package.get("source_sha256", ""))):
            raise ValueError(f"Web Player Python package source hash is invalid: {name}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(package.get("payload_sha256", ""))):
            raise ValueError(f"Web Player Python package payload hash is invalid: {name}")
        if package.get("payload_hash_algorithm") != WEB_PYTHON_PAYLOAD_HASH:
            raise ValueError(f"Web Player Python package payload hash algorithm is invalid: {name}")
        if type(package.get("payload_bytes")) is not int or package["payload_bytes"] <= 0:
            raise ValueError(f"Web Player Python package payload size is invalid: {name}")
        if type(package.get("payload_files")) is not int or package["payload_files"] <= 0:
            raise ValueError(f"Web Player Python package payload file count is invalid: {name}")
        if not isinstance(package.get("license"), str) or not package["license"]:
            raise ValueError(f"Web Player Python package license is missing: {name}")

    missing = sorted(set(WEB_REQUIRED_PYTHON_IMPORTS).difference(provided_imports))
    if missing:
        raise ValueError(
            "Web Player Python runtime is missing required packages for: "
            + ", ".join(missing)
        )
    numpy = next((package for package in packages if package.get("name") == "numpy"), None)
    if not isinstance(numpy, dict) or numpy.get("imports") != ["numpy"]:
        raise ValueError("Web Player Python runtime is missing the NumPy package")
    if (
        numpy.get("version") != WEB_NUMPY_VERSION
        or numpy.get("source_url") != WEB_NUMPY_SOURCE_URL
        or numpy.get("source_sha256") != WEB_NUMPY_SOURCE_SHA256
        or numpy.get("license") != "BSD-3-Clause"
    ):
        raise ValueError("Web Player NumPy source provenance does not match this engine")
    return document


def inspect_player_payload(root: Path, *, engine: str, engine_release: str | None = None,
                           source_revision: str | None = None) -> dict[str, object]:
    manifest = json.loads((root / "Player.inxmanifest").read_text(encoding="utf-8"))
    expected = {"$schema": "infernux.web_player", "engine": engine,
                "platform": "web", "architecture": "wasm32", "python_abi": WEB_PYTHON_ABI}
    if engine_release is not None:
        expected["engine_release"] = engine_release
    if source_revision is not None:
        expected["engine_source_revision"] = source_revision
    if not isinstance(manifest, dict) or any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError("Web Player payload does not match this engine")
    if manifest.get("configuration") not in {"Release", "RelWithDebInfo"}:
        raise ValueError("Web Player payload must be an optimized native build")
    python_runtime = manifest.get("python_runtime")
    if not isinstance(python_runtime, dict):
        raise ValueError(
            "Web Player payload is stale/out-of-source: Player.inxmanifest lacks "
            "python_runtime; regenerate the precompiled Web Player and inject its "
            "publish/player payload into the platform plugin before building"
        )
    inspect_python_runtime(python_runtime)
    if not isinstance(manifest.get("engine_release"), str) or not (
        manifest["engine_release"] == engine or manifest["engine_release"].startswith(engine + "-v")
    ):
        raise ValueError("Web Player payload has no matching hotfix release identity")
    for name in PLAYER_FILES:
        if not (root / name).is_file():
            raise FileNotFoundError(f"Web platform plugin is missing {root / name}")
    return manifest


def inspect_shader_tools(root: Path, *, host: str) -> dict[str, str]:
    if host not in HOSTS:
        raise ValueError(f"Unsupported Web shader tool host: {host}")
    suffix = ".exe" if host == "windows-x64" else ""
    tools = {"glslang": root / host / f"glslangValidator{suffix}",
             "tint": root / host / f"tint{suffix}"}
    for path in tools.values():
        if not path.is_file():
            raise FileNotFoundError(f"Web platform plugin is missing {path}")
    return {name: str(path) for name, path in tools.items()}


def _main() -> None:
    parser = argparse.ArgumentParser(description="Validate an Infernux Web Python runtime manifest")
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--inspect-python-runtime", type=Path)
    operation.add_argument("--hash-python-package-payload", type=Path)
    operation.add_argument("--verify-python-package-payload", type=Path)
    parser.add_argument("--runtime-manifest", type=Path)
    parser.add_argument("--package")
    arguments = parser.parse_args()
    if arguments.inspect_python_runtime is not None:
        document = json.loads(arguments.inspect_python_runtime.read_text(encoding="utf-8"))
        inspect_python_runtime(document)
    elif arguments.hash_python_package_payload is not None:
        print(
            json.dumps(
                python_package_payload_identity(arguments.hash_python_package_payload),
                sort_keys=True,
            )
        )
    else:
        if arguments.runtime_manifest is None or not arguments.package:
            parser.error(
                "--verify-python-package-payload requires --runtime-manifest and --package"
            )
        document = json.loads(arguments.runtime_manifest.read_text(encoding="utf-8"))
        print(
            json.dumps(
                verify_python_package_payload(
                    arguments.verify_python_package_payload,
                    document,
                    package_name=arguments.package,
                ),
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    _main()
