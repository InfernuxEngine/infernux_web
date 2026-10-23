"""Web target and its staged exporter implementation."""

from __future__ import annotations

import compileall
import ast
import hashlib
import importlib.util
import json
import os
import py_compile
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

from infernux.engine.build import (
    BuildArtifact,
    BuildDiagnostic,
    BuildPlan,
    BuildRequest,
    BuildResult,
    BuildStep,
    BuildOption,
    BuildOptionChoice,
    BuildOptionKind,
    BuildTarget,
    CapabilityReport,
    DiagnosticSeverity,
    PlatformCapabilities,
    PlatformExporter,
)
from infernux._compiler.kernel_contract import (
    implicit_receiver_attribute,
    implicit_receiver_name,
    kernel_diagnostic,
    source_module_name,
)

from .doctor import inspect_web_toolchain
from .native_payload import inspect_python_runtime
from .capabilities import (
    WEBGPU_CAPABILITY_FILENAME,
    validate_webgpu_capability_inventory,
    webgpu_capability_inventory,
)


_WEB_CAPABILITIES = PlatformCapabilities(
    graphics_api="webgpu",
    threads=False,
    dynamic_loading=False,
    filesystem=True,
    network=False,
    audio=True,
    pointer_input=True,
    text_input=True,
    gamepad_input=True,
    python_native_modules=False,
    cpu_jit=False,
    persistent_storage=False,
    features=frozenset(
        {
            "browser-lifecycle",
            "css-pixel-viewport",
            "dom-text-input",
            "gesture-gated-webaudio",
            "multi-pointer",
            "safe-area",
            "webgpu-capability-inventory",
        }
    ),
)

_PROJECT_WEB_TEMPLATE = Path("ProjectSettings") / "WebTemplate"
_WEB_SHELL_REQUIRED_MARKERS = (
    "{{{ SCRIPT }}}",
    "@INFERNUX_WEB_ASSET_REVISION@",
    "@INFERNUX_WEB_DISPLAY_MODE@",
    "@INFERNUX_WEB_CANVAS_WIDTH@",
    "@INFERNUX_WEB_CANVAS_HEIGHT@",
)


class WebPlatformExporter(PlatformExporter):
    @property
    def exporter_id(self) -> str:
        return "infernux/platform-web"

    def targets(self):
        return (
            BuildTarget(
                "web-wasm32",
                "Web WebAssembly",
                "web",
                "wasm32",
                _WEB_CAPABILITIES,
            ),
        )

    def build_options(self, target):
        if target.id != "web-wasm32":
            raise ValueError(f"Unsupported Web build target: {target.id}")
        return (
            BuildOption(
                "display_mode", "build.web_canvas_mode", BuildOptionKind.ENUM,
                "fullscreen_borderless",
                choices=(
                    BuildOptionChoice("fullscreen_borderless", "build.web_fill_viewport"),
                    BuildOptionChoice("windowed", "build.web_fixed_canvas"),
                ),
            ),
            BuildOption(
                "window_width", "build.web_canvas_width", BuildOptionKind.INTEGER,
                1280, minimum=320, maximum=7680,
                visible_when={"display_mode": "windowed"},
            ),
            BuildOption(
                "window_height", "build.web_canvas_height", BuildOptionKind.INTEGER,
                720, minimum=240, maximum=4320,
                visible_when={"display_mode": "windowed"},
            ),
            BuildOption(
                "window_resizable", "build.window_resizable",
                BuildOptionKind.BOOLEAN, False, editor_visible=False,
            ),
            BuildOption(
                "build_cache_root", "Build cache root", BuildOptionKind.PATH, "",
                editor_visible=False,
            ),
        )

    def doctor(self, request: BuildRequest) -> CapabilityReport:
        return inspect_web_toolchain(request.target)

    def create_plan(self, request: BuildRequest) -> BuildPlan:
        return BuildPlan(
            request.target,
            (
                BuildStep("cook", "Cook project content", "cook"),
                BuildStep(
                    "shaders",
                    "Translate portable GLSL shaders to validated WGSL",
                    "compile",
                ),
                BuildStep("imports", "Analyze browser Python imports", "analyze"),
                BuildStep("runtime", "Stage precompiled WebGPU and CPython runtime", "package"),
                BuildStep(
                    "package",
                    "Publish HTML, JavaScript, WebAssembly, and content",
                    "package",
                ),
                BuildStep("audit", "Audit browser package and server contract", "audit"),
            ),
            {
                "architecture": "wasm32",
                "graphics_api": "webgpu",
                "python": "3.13",
            },
        )

    def execute(self, request: BuildRequest, plan: BuildPlan) -> BuildResult:
        started = time.perf_counter()
        capability_inventory: dict[str, object] = {}
        report = self.doctor(request)
        if not report.available:
            return BuildResult(
                request.target,
                False,
                diagnostics=report.diagnostics,
                manifest={"toolchain": dict(report.details)},
                elapsed_seconds=time.perf_counter() - started,
            )
        details = dict(report.details)
        import infernux

        source_package = Path(infernux.__file__).resolve().parent
        staging = _web_staging_directory(request)
        logs: tuple[str, ...] = ()
        try:
            from infernux.engine.platform_content_cook import (
                build_settings_for_request,
            )

            build_settings = build_settings_for_request(request)
            presentation = {
                "display_mode": str(build_settings["display_mode"]),
                "window_width": int(build_settings["window_width"]),
                "window_height": int(build_settings["window_height"]),
                "render_output": build_settings["render_output"],
            }
            default_shell = (
                Path(__file__).resolve().parent / "templates" / "host" / "shell.html"
            )
            web_shell, project_web_template = _resolve_web_template(
                request,
                default_shell,
            )
            request.report("prepare", 0, 1, "Preparing Web Player staging")
            _prepare_web_staging(staging)
            game_name, player_assets, python_sources = _cook_web_player_assets(
                request, staging
            )
            _reject_unshipped_web_dependencies(
                python_sources,
                details.get("player"),
            )
            _stage_web_branding(player_assets, source_package, game_name)
            _stage_engine_python_package(request, player_assets, source_package)
            _stage_web_shader_sources(request, staging, player_assets, source_package)
            capability_inventory = _stage_webgpu_capability_inventory(player_assets)
            logs = _assemble_web_host(
                request,
                staging,
                player_assets,
                details,
                source_package,
                presentation,
                web_shell,
                project_web_template,
            )
            request.report("prepare", 1, 1, "Cooked Web Player host ready")
        except (OSError, RuntimeError, ValueError) as error:
            return BuildResult(
                request.target,
                False,
                diagnostics=(
                    BuildDiagnostic(
                        DiagnosticSeverity.ERROR,
                        "web.player.stage-failed",
                        str(error),
                        source=self.exporter_id,
                    ),
                ),
                manifest={
                    "exporter": self.exporter_id,
                    "abi": "wasm32",
                    "graphics_api": "webgpu",
                    "python": "3.13",
                    "scope": "staging-failed",
                    "staging": str(staging),
                },
                logs=logs,
                elapsed_seconds=time.perf_counter() - started,
            )

        try:
            artifacts, asset_revision = _publish_web_player(
                request,
                staging / "host-build",
                project_web_template,
            )
        except (OSError, RuntimeError, ValueError) as error:
            return BuildResult(
                request.target,
                False,
                diagnostics=(
                    BuildDiagnostic(
                        DiagnosticSeverity.ERROR,
                        "web.player.publish-failed",
                        str(error),
                        source=self.exporter_id,
                    ),
                ),
                manifest={
                    "exporter": self.exporter_id,
                    "abi": "wasm32",
                    "graphics_api": "webgpu",
                    "python": "3.13",
                    "scope": "publish-failed",
                    "game": game_name,
                    "staging": str(staging),
                },
                logs=logs,
                elapsed_seconds=time.perf_counter() - started,
            )

        return BuildResult(
            request.target,
            True,
            artifacts=artifacts,
            manifest={
                "exporter": self.exporter_id,
                "abi": "wasm32",
                "graphics_api": "webgpu",
                "python": "3.13",
                "python_runtime": details["player"]["python_runtime"],
                "scope": "cooked-player",
                "game": game_name,
                "asset_revision": asset_revision,
                "entry_point": "infernux-player.html",
                "presentation": presentation,
                "web_template": (
                    "project" if project_web_template is not None else "default"
                ),
                "webgpu_capabilities": {
                    "path": WEBGPU_CAPABILITY_FILENAME,
                    "schema": capability_inventory["$schema"],
                    "open_parity_gates": sorted(
                        str(feature["id"])
                        for feature in capability_inventory["features"]
                        if feature["parity_gate"] == "open"
                    ),
                },
            },
            logs=logs,
            elapsed_seconds=time.perf_counter() - started,
        )


def _publish_web_player(
    request: BuildRequest,
    host_build: Path,
    project_web_template: Path | None,
) -> tuple[tuple[BuildArtifact, ...], str]:
    entry_point = host_build / "infernux-player.html"
    if not entry_point.is_file():
        raise RuntimeError("Web host build has no infernux-player.html entry point")
    html = entry_point.read_text(encoding="utf-8")
    match = re.search(r"assetRevision\s*=\s*['\"]([0-9a-f]{24})['\"]", html)
    if match is None:
        raise RuntimeError("Web host entry point has no valid asset revision")
    revision = match.group(1)
    from infernux.engine.player_dependencies import stage_engine_licenses

    notices = stage_engine_licenses(host_build / "licenses" / "infernux")
    names = (
        "infernux-player.html",
        f"infernux-player.{revision}.js",
        f"infernux-player.{revision}.wasm",
        f"infernux-player.{revision}.data",
        f"infernux-player.{revision}.inxpkg",
        "infernux-logo.png",
        "infernux-favicon.png",
        "infernux-icon-192.png",
        "infernux-icon-512.png",
        "infernux.webmanifest",
        "infernux-branding.js",
        WEBGPU_CAPABILITY_FILENAME,
        *(path.relative_to(host_build).as_posix() for path in notices),
    )
    missing = [name for name in names if not (host_build / name).is_file()]
    if missing:
        raise RuntimeError(
            "Web host publication is incomplete: " + ", ".join(missing)
        )
    capability_path = host_build / WEBGPU_CAPABILITY_FILENAME
    try:
        capability_document = json.loads(capability_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"WebGPU capability inventory is unreadable: {error}"
        ) from error
    if not isinstance(capability_document, dict):
        raise RuntimeError("WebGPU capability inventory must be a JSON object")
    try:
        validate_webgpu_capability_inventory(capability_document)
    except ValueError as error:
        raise RuntimeError(str(error)) from error
    if names[1] not in html:
        raise RuntimeError(f"Web host entry point does not reference {names[1]}")
    for suffix in ("wasm", "data"):
        logical_name = f"infernux-runtime.{suffix}"
        versioned_expression = f"infernux-player.${{assetRevision}}.{suffix}"
        if logical_name not in html or versioned_expression not in html:
            raise RuntimeError(
                f"Web host entry point does not version-map {logical_name}"
            )

    output_root = Path(request.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    staged: dict[str, Path] = {}
    template_output = output_root / "web-template"
    workspace = Path(tempfile.mkdtemp(prefix=".web-template.", dir=output_root))
    template_next = workspace / "next"
    template_previous = workspace / "previous"
    template_backed_up = False
    template_installed = False
    retain_workspace = False
    template_files: list[Path] = []
    try:
        if template_output.is_symlink() or os.path.isjunction(template_output):
            raise RuntimeError("Published Web template cannot be a link or junction")
        if template_output.exists() and not template_output.is_dir():
            raise RuntimeError("Published Web template must be a directory")
        if project_web_template is not None:
            template_files = _stage_project_web_template(project_web_template, template_next)
        # Copy the complete generation without touching the currently served
        # one. Versioned resources are committed first and HTML is the final
        # atomic switch; a failed copy/replace leaves the old entry point and
        # all resources it names intact.
        for name in names:
            source = host_build / name
            destination = output_root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.{os.getpid()}.{time.time_ns()}.tmp")
            staged[name] = temporary
            shutil.copy2(source, temporary)
        artifacts = [
            BuildArtifact(str(output_root / name), (output_root / name).suffix.lstrip(".") or "file",
                          size=staged[name].stat().st_size)
            for name in names
        ]
        artifacts.extend(
            BuildArtifact(str(template_output / relative), relative.suffix.lstrip(".") or "file",
                          size=(template_next / relative).stat().st_size)
            for relative in template_files
        )
        for name in names[1:]:
            os.replace(staged[name], output_root / name)
            del staged[name]
        # The fixed template URL belongs to the entry point. Keep its previous
        # directory until the HTML switch succeeds, including template removal.
        if template_output.exists():
            os.replace(template_output, template_previous)
            template_backed_up = True
            retain_workspace = True
        if project_web_template is not None:
            os.replace(template_next, template_output)
            template_installed = True
        os.replace(staged[names[0]], output_root / names[0])
        del staged[names[0]]
        retain_workspace = False
    except BaseException:
        try:
            if template_installed:
                shutil.rmtree(template_output)
            if template_backed_up:
                os.replace(template_previous, template_output)
        except OSError as restore_error:
            # A failed rollback must retain the previous complete template for
            # recovery. Never let temporary-directory cleanup destroy it.
            retain_workspace = True
            raise RuntimeError(
                f"Web publication rollback failed; retained template workspace: {workspace}"
            ) from restore_error
        retain_workspace = False
        raise
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
        if not retain_workspace:
            shutil.rmtree(workspace)

    # Retire old revisions only after the new HTML points at a complete
    # generation. Never delete rollback resources before the entry switch.
    for stale in output_root.glob("infernux-player.*"):
        if stale.is_file() and stale.name not in names:
            stale.unlink()

    request.report("package", 1, 1, "Web Player package published")
    request.report("audit", 1, 1, "Web Player browser contract verified")
    return tuple(artifacts), revision


def _stage_project_web_template(source_root: Path, destination_root: Path) -> list[Path]:
    if source_root.is_symlink() or os.path.isjunction(source_root):
        raise RuntimeError("Project Web template cannot be a link or junction")
    if not source_root.is_dir():
        raise RuntimeError("Project Web template must be a directory")
    destination_root.mkdir()
    files: list[Path] = []
    def reject_walk_error(error: OSError) -> None:
        raise error

    for directory, subdirectories, filenames in os.walk(
        source_root, followlinks=False, onerror=reject_walk_error,
    ):
        subdirectories.sort()
        for name in [*subdirectories, *sorted(filenames)]:
            source = Path(directory) / name
            if source.is_symlink() or os.path.isjunction(source):
                raise RuntimeError(f"Project Web template cannot contain links or junctions: {source}")
        for name in sorted(filenames):
            source = Path(directory) / name
            relative = source.relative_to(source_root)
            if relative == Path("shell.html"):
                continue
            if not source.is_file():
                raise RuntimeError(f"Project Web template source is not a regular file: {source}")
            destination = destination_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            files.append(relative)
    return sorted(files)


def _resolve_web_template(
    request: BuildRequest,
    default_shell: Path,
) -> tuple[Path, Path | None]:
    template_root = Path(request.project_root).resolve() / _PROJECT_WEB_TEMPLATE
    if template_root.is_symlink() or os.path.isjunction(template_root):
        raise ValueError("Project Web template cannot be a link or junction")
    if not template_root.exists():
        return default_shell.resolve(), None
    if not template_root.is_dir():
        raise ValueError(
            f"Project Web template must be a directory: {template_root}"
        )
    shell = template_root / "shell.html"
    if not shell.is_file():
        raise ValueError(
            f"Project Web template is missing shell.html: {template_root}"
        )
    if shell.is_symlink():
        raise ValueError("Project Web template shell.html cannot be a symbolic link")
    document = shell.read_text(encoding="utf-8")
    missing = [
        marker for marker in _WEB_SHELL_REQUIRED_MARKERS if marker not in document
    ]
    if not re.search(r"\bid\s*=\s*(['\"])canvas\1", document):
        missing.append('id="canvas"')
    if missing:
        raise ValueError(
            "Project Web template shell.html is missing required markers: "
            + ", ".join(missing)
        )
    return shell.resolve(), template_root.resolve()


def _stage_webgpu_capability_inventory(
    player_assets: Path,
) -> dict[str, object]:
    document = webgpu_capability_inventory()
    validate_webgpu_capability_inventory(document)
    destination = player_assets / WEBGPU_CAPABILITY_FILENAME
    destination.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return document


def _prepare_web_staging(staging: Path) -> None:
    """Refresh generated project content and assembly output."""

    staging.mkdir(parents=True, exist_ok=True)
    for name in (".infernux-player-cook", "player-assets", "shader-cook", "host-build"):
        shutil.rmtree(staging / name, ignore_errors=True)


def _web_build_cache_root(request: BuildRequest) -> Path:
    configured = str(
        request.profile.options.get("build_cache_root", "") or ""
    ).strip()
    if not configured:
        configured = os.environ.get("INFERNUX_BUILD_CACHE_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(request.project_root).resolve() / "Cache" / "Build"


def _web_staging_directory(request: BuildRequest) -> Path:
    return _web_build_cache_root(request) / "WebHost" / str(request.target)


def _cook_web_player_assets(
    request: BuildRequest,
    staging: Path,
) -> tuple[str, Path, tuple[Path, ...]]:
    from infernux.engine.platform_content_cook import cook_platform_content

    cook_root = staging / ".infernux-player-cook"
    cook_root.mkdir(parents=True)
    cooked = cook_platform_content(
        request,
        cook_root,
        platform_host={
            "identity": "webgpu-cpython-wasm-player-host",
            "entry_point": "infernux-player.html",
            "platform": "web",
            "architecture": "wasm32",
        },
    )
    player_assets = staging / "player-assets"
    player_assets.mkdir(parents=True)
    shutil.copytree(
        cooked.data_directory,
        player_assets / cooked.data_directory.name,
    )
    return cooked.game_name, player_assets, cooked.python_sources


def _reject_unshipped_web_dependencies(
    python_sources: object,
    player_manifest: object = None,
) -> None:
    """Fail the Web build before publishing a runtime that cannot import a script.

    Native numerical packages belong to the immutable Player runtime, not the
    project's Content.inxpkg.  Permit an import only when the inspected Player
    manifest proves that its complete package is built for the exact Web ABI.
    """
    provided_imports: set[str] = set()
    if isinstance(player_manifest, dict):
        runtime = inspect_python_runtime(player_manifest.get("python_runtime"))
        for package in runtime["packages"]:
            provided_imports.update(package["imports"])

    if not isinstance(python_sources, (tuple, list, set, frozenset)):
        raise TypeError("Web dependency scan requires the frozen Player Python source closure")

    missing: set[str] = set()
    unsupported: set[str] = set()
    for source_value in python_sources:
        source_path = Path(source_value).resolve()
        if not source_path.is_file() or source_path.suffix.casefold() != ".py":
            raise ValueError(
                f"Web Player Python source closure contains an invalid file: {source_path}"
            )
        try:
            tree = ast.parse(source_path.read_text(encoding="utf-8"), str(source_path))
        except (OSError, SyntaxError) as error:
            raise ValueError(f"Web dependency scan failed for {source_path}: {error}") from error

        declarations = _web_gpu_declaration_records(tree)
        if declarations:
            module_name = source_module_name(source_path)
            checked: list[dict[str, object]] = []
            for item in declarations:
                record = dict(item)
                record["module"] = module_name
                record["qualified"] = f"{module_name}.{item['qualified']}"
                receiver = implicit_receiver_name(
                    item["node"], in_class=bool(item.get("in_class", False))
                )
                if receiver is not None:
                    raise ValueError(kernel_diagnostic(
                        source_path,
                        str(record["qualified"]),
                        item["node"],
                        target="Web/Cook",
                        reason=(
                            "class-contained kernels cannot use an implicit "
                            f"instance receiver '{receiver}'"
                        ),
                        advice=(
                            "put @staticmethod above @inx.compute.kernel "
                            "(decorator order: @staticmethod, then "
                            "@inx.compute.kernel) or move the kernel to module scope"
                        ),
                    ))
                access = implicit_receiver_attribute(item["node"])
                if access is not None:
                    receiver_name, attribute = access
                    raise ValueError(kernel_diagnostic(
                        source_path,
                        str(record["qualified"]),
                        item["node"],
                        target="Web/Cook",
                        reason=(
                            "class-contained kernels cannot access unbound "
                            f"receiver field '{receiver_name}.{attribute.attr}'"
                        ),
                        advice=(
                            "pass the required scalar or inx.buffer explicitly, "
                            "or put @staticmethod above @inx.compute.kernel"
                        ),
                        line=attribute.lineno,
                        column=attribute.col_offset + 1,
                    ))
                record.pop("node", None)
                checked.append(record)
            from infernux.engine.build.compute_cpu import build_cpu_compute_source

            try:
                build_cpu_compute_source(source_path.read_text(encoding="utf-8"))
            except (SyntaxError, ValueError) as error:
                raise ValueError(
                    f"Web CPU compute lowering failed for {source_path}: {error}"
                ) from error
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = (alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = (node.module.split(".", 1)[0],)
            else:
                continue
            for name in names:
                if name in {"numba", "llvmlite"}:
                    unsupported.add(name)
                elif name == "numpy" and name not in provided_imports:
                    missing.add(name)
    if unsupported or missing:
        unavailable = sorted(unsupported | missing)
        names = ", ".join(unavailable)
        reason = (
            "unsupported on the Web runtime"
            if unsupported
            else "missing from the inspected Web runtime package closure"
        )
        raise ValueError(
            "Web Player cannot package native numerical dependencies: "
            f"{names} ({reason}). An exact CPython/Emscripten-compatible "
            "dependency payload is required; "
            "the build was stopped before publishing an unusable Player."
        )


def _web_gpu_declaration_records(tree: ast.Module) -> tuple[dict[str, object], ...]:
    """Resolve GPU aliases and retain qualified source locations."""

    gpu_decorators = {
        "infernux.compute.kernel",
        "infernux.compute.function",
    }
    declarations: set[str] = set()
    declaration_nodes: list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef]] = []

    qualified_paths: dict[int, str] = {}

    class PathCollector(ast.NodeVisitor):
        def __init__(self) -> None:
            self.classes: list[str] = []
            self.class_scopes: dict[int, bool] = {}

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.classes.append(node.name)
            self.generic_visit(node)
            self.classes.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            qualified_paths[id(node)] = ".".join((*self.classes, node.name))
            self.class_scopes[id(node)] = bool(self.classes)
            self.generic_visit(node)

        visit_AsyncFunctionDef = visit_FunctionDef

    path_collector = PathCollector()
    path_collector.visit(tree)
    class_scopes = path_collector.class_scopes

    class Scope:
        def __init__(self, parent: "Scope | None" = None) -> None:
            self.parent = parent
            self.aliases: dict[str, frozenset[str]] = {}
            self.observed: dict[str, frozenset[str]] = {}

        def clone(self) -> "Scope":
            result = Scope(self.parent)
            result.aliases = dict(self.aliases)
            return result

        def merge(self, branches: list["Scope"]) -> None:
            names = set().union(*(branch.aliases for branch in branches))
            self.aliases = {
                name: frozenset().union(
                    *(branch.lookup(name) for branch in branches)
                )
                for name in names
            }
            prior_observed = dict(self.observed)
            observed_names = set(prior_observed).union(
                *(branch.observed for branch in branches)
            )
            self.observed = {
                name: prior_observed.get(name, frozenset()) | frozenset().union(
                    *(branch.observed.get(name, frozenset()) for branch in branches)
                )
                for name in observed_names
            }

        def bind_name(self, name: str, candidates: frozenset[str]) -> None:
            resolved = candidates or frozenset({name})
            self.aliases[name] = resolved
            self.observed[name] = self.observed.get(name, frozenset()) | resolved

        def promote_observed(self) -> None:
            for name, candidates in self.observed.items():
                self.aliases[name] = self.lookup(name) | candidates

        def lookup(self, name: str) -> frozenset[str]:
            if name in self.aliases:
                return self.aliases[name]
            if self.parent is not None:
                return self.parent.lookup(name)
            return frozenset({name})

        def canonical(self, expression: ast.expr) -> frozenset[str]:
            if isinstance(expression, ast.Name):
                return self.lookup(expression.id)
            if isinstance(expression, ast.Attribute):
                return frozenset(
                    f"{owner}.{expression.attr}"
                    for owner in self.canonical(expression.value)
                    if owner
                )
            return frozenset()

        def bind(self, target: ast.expr, canonical: frozenset[str]) -> None:
            if isinstance(target, ast.Name):
                self.bind_name(target.id, canonical)

    def bind_assignment(scope: Scope, target: ast.expr, value: ast.expr) -> None:
        if isinstance(target, (ast.Tuple, ast.List)) and isinstance(
            value, (ast.Tuple, ast.List)
        ):
            for item, item_value in zip(target.elts, value.elts):
                bind_assignment(scope, item, item_value)
            return
        scope.bind(target, scope.canonical(value))

    def scan_expression_aliases(expression: ast.expr, scope: Scope) -> None:
        for node in ast.walk(expression):
            if isinstance(node, ast.NamedExpr):
                bind_assignment(scope, node.target, node.value)

    def scan_body(statements: list[ast.stmt], scope: Scope) -> None:
        for statement in statements:
            if isinstance(statement, ast.Import):
                for alias in statement.names:
                    local = alias.asname or alias.name.split(".", 1)[0]
                    scope.bind_name(local, frozenset(
                        {alias.name if alias.asname else local}
                    ))
                continue
            if (
                isinstance(statement, ast.ImportFrom)
                and statement.level == 0
                and statement.module
            ):
                for alias in statement.names:
                    if alias.name == "*":
                        module = statement.module.casefold()
                        if module == "infernux":
                            scope.bind_name("compute", frozenset(
                                {f"{statement.module}.compute"}
                            ))
                        elif module == "infernux.compute":
                            scope.bind_name("kernel", frozenset(
                                {f"{statement.module}.kernel"}
                            ))
                            scope.bind_name("function", frozenset(
                                {f"{statement.module}.function"}
                            ))
                        continue
                    scope.bind_name(alias.asname or alias.name, frozenset(
                        {f"{statement.module}.{alias.name}"}
                    ))
                continue
            if isinstance(statement, ast.Assign):
                for target in statement.targets:
                    bind_assignment(scope, target, statement.value)
                continue
            if isinstance(statement, ast.AnnAssign) and statement.value is not None:
                bind_assignment(scope, statement.target, statement.value)
                continue
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                for decorator in statement.decorator_list:
                    target = decorator.func if isinstance(decorator, ast.Call) else decorator
                    if any(
                        candidate.casefold() in gpu_decorators
                        for candidate in scope.canonical(target)
                    ):
                        declarations.add(statement.name)
                        declaration_nodes.append((statement.name, statement))
                scan_body(statement.body, Scope(scope))
                scope.bind_name(statement.name, frozenset({statement.name}))
                continue
            if isinstance(statement, ast.If):
                condition = scope.clone()
                scan_expression_aliases(statement.test, condition)
                body = condition.clone()
                otherwise = condition.clone()
                scan_body(statement.body, body)
                scan_body(statement.orelse, otherwise)
                scope.merge([body, otherwise])
                continue
            if isinstance(statement, (ast.For, ast.AsyncFor, ast.While)):
                entry = scope.clone()
                if isinstance(statement, (ast.For, ast.AsyncFor)):
                    scan_expression_aliases(statement.iter, entry)
                else:
                    scan_expression_aliases(statement.test, entry)
                body = entry.clone()
                scan_body(statement.body, body)
                # A break/continue or exception caught by an enclosing block
                # can expose any prefix of the loop body's assignments.
                body.promote_observed()
                without_iteration = entry.clone()
                scan_body(statement.orelse, without_iteration)
                after_iteration = body.clone()
                scan_body(statement.orelse, after_iteration)
                scope.merge([entry, body, without_iteration, after_iteration])
                continue
            if isinstance(statement, (ast.With, ast.AsyncWith)):
                scan_body(statement.body, scope)
                continue
            if isinstance(statement, (ast.Try, ast.TryStar)):
                successful = scope.clone()
                scan_body(statement.body, successful)
                scan_body(statement.orelse, successful)
                branches = [successful]
                for handler in statement.handlers:
                    handled = scope.clone()
                    # Exceptions may leave bindings made by any completed
                    # prefix of the try body visible to the handler.
                    for name, candidates in successful.observed.items():
                        handled.aliases[name] = handled.lookup(name) | candidates
                    scan_body(handler.body, handled)
                    branches.append(handled)
                scope.merge(branches)
                scan_body(statement.finalbody, scope)
                continue
            if isinstance(statement, ast.Match):
                branches = [scope.clone()]
                for case in statement.cases:
                    branch = scope.clone()
                    if case.guard is not None:
                        scan_expression_aliases(case.guard, branch)
                    scan_body(case.body, branch)
                    branches.append(branch)
                scope.merge(branches)

    scan_body(tree.body, Scope())
    return tuple(
        {
            "name": name,
            "qualified": qualified_paths.get(id(node), name),
            "line": int(node.lineno),
            "column": int(node.col_offset + 1),
            "in_class": class_scopes.get(id(node), False),
            "node": node,
        }
        for name, node in declaration_nodes
    )


def _web_gpu_declarations(tree: ast.Module) -> set[str]:
    """Return declaration names for callers that only need membership."""
    return {str(item["name"]) for item in _web_gpu_declaration_records(tree)}


def _stage_web_branding(
    player_assets: Path,
    source_package: Path,
    game_name: str,
) -> Path:
    """Create browser metadata from the shared cooked Player branding."""

    from infernux.engine.platform_content_cook import read_cooked_player_icon

    data_roots = sorted(path for path in player_assets.glob("*_Data") if path.is_dir())
    if len(data_roots) != 1:
        raise ValueError("Web Player branding requires exactly one cooked *_Data directory")
    source_icon = read_cooked_player_icon(
        data_roots[0],
        default_icon=(
            source_package
            / "resources"
            / "icons"
            / "icon.png"
        ),
    )
    try:
        from PIL import Image, ImageOps
    except ImportError as error:
        raise ValueError("Pillow is required to generate Web application icons") from error
    try:
        from io import BytesIO

        with Image.open(BytesIO(source_icon)) as opened:
            source = opened.convert("RGBA")
    except (OSError, ValueError) as error:
        raise ValueError("Cooked Web application icon is unreadable") from error

    branding = player_assets / "web-branding"
    shutil.rmtree(branding, ignore_errors=True)
    branding.mkdir(parents=True)
    for name, size in (
        ("infernux-favicon.png", 32),
        ("infernux-icon-192.png", 192),
        ("infernux-icon-512.png", 512),
    ):
        canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        contained = ImageOps.contain(
            source,
            (size, size),
            method=Image.Resampling.LANCZOS,
        )
        canvas.alpha_composite(
            contained,
            ((size - contained.width) // 2, (size - contained.height) // 2),
        )
        canvas.save(branding / name, format="PNG", optimize=True)

    manifest = {
        "name": game_name,
        "short_name": game_name,
        "display": "fullscreen",
        "background_color": "#111214",
        "theme_color": "#111214",
        "icons": [
            {
                "src": "infernux-icon-192.png",
                "sizes": "192x192",
                "type": "image/png",
            },
            {
                "src": "infernux-icon-512.png",
                "sizes": "512x512",
                "type": "image/png",
            },
        ],
    }
    (branding / "infernux.webmanifest").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    branding_document = json.dumps(
        {"game_name": game_name},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    (branding / "infernux-branding.js").write_text(
        "globalThis.InfernuxBranding = Object.freeze("
        + branding_document
        + ");\n"
        "document.title = globalThis.InfernuxBranding.game_name;\n",
        encoding="utf-8",
        newline="\n",
    )
    return branding


def _stage_engine_python_package(
    request: BuildRequest,
    player_assets: Path,
    source_package: Path,
) -> None:
    if not (source_package / "engine" / "platform_player_bootstrap.py").is_file():
        raise ValueError(f"Infernux Player Python sources are incomplete: {source_package}")
    site_packages = player_assets / "python" / "site-packages"
    destination = site_packages / "infernux"
    request.report("analyze", 0, 2, "Staging Infernux Web Player modules")
    shutil.copytree(
        source_package,
        destination,
        ignore=_web_engine_python_ignore,
    )
    packaging_spec = importlib.util.find_spec("packaging")
    packaging_source = (
        Path(str(packaging_spec.origin)).resolve().parent
        if packaging_spec is not None and packaging_spec.origin
        else None
    )
    if packaging_source is None or not (packaging_source / "__init__.py").is_file():
        raise ValueError("The Web Player staging environment has no packaging module")
    shutil.copytree(
        packaging_source,
        site_packages / "packaging",
        ignore=_web_engine_python_ignore,
    )
    if sys.version_info[:2] != (3, 13):
        raise RuntimeError(
            "Web Player Python staging requires the engine's Python 3.13 runtime"
        )
    if not compileall.compile_dir(
        site_packages,
        quiet=1,
        optimize=0,
        invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
    ):
        raise RuntimeError("Web Player engine Python bytecode compilation failed")
    _validate_web_python_payload(site_packages)
    request.report("analyze", 2, 2, "Web Player Python modules staged")


_WEB_ENGINE_EXCLUDED_NAMES = frozenset(
    {
        "__pycache__",
        "_runtime_modules",
        "_runtime_packs",
        "infernux.mcp.inxpkg",
        "official_packages",
        "player_runtime",
        "project_templates",
        "test",
        "tests",
    }
)
_WEB_ENGINE_EXCLUDED_SUFFIXES = frozenset(
    {".dll", ".dylib", ".exp", ".lib", ".pyd", ".pyc", ".pyi", ".so"}
)
_WEB_FORBIDDEN_MODEL_SOURCE_SUFFIXES = frozenset(
    {".blend", ".fbx", ".glb", ".gltf", ".obj"}
)


def _web_engine_python_ignore(directory: str, names: list[str]) -> set[str]:
    del directory
    return {
        name
        for name in names
        if name.casefold() in _WEB_ENGINE_EXCLUDED_NAMES
        or Path(name).suffix.casefold() in _WEB_ENGINE_EXCLUDED_SUFFIXES
    }


def _validate_web_python_payload(site_packages: Path) -> None:
    """Keep editor/test/native build inputs outside the browser preload tree."""

    forbidden: list[str] = []
    for path in site_packages.rglob("*"):
        relative = path.relative_to(site_packages).as_posix()
        folded_parts = {
            part.casefold() for part in path.relative_to(site_packages).parts
        }
        suffix = path.suffix.casefold()
        if folded_parts & {"test", "tests"}:
            forbidden.append(relative)
        elif path.name.casefold() == "infernux.mcp.inxpkg":
            forbidden.append(relative)
        elif path.is_file() and suffix in _WEB_FORBIDDEN_MODEL_SOURCE_SUFFIXES:
            forbidden.append(relative)
        elif path.is_file() and suffix in {
            ".dll", ".dylib", ".exp", ".lib", ".obj", ".pyd", ".pyi", ".so"
        }:
            forbidden.append(relative)
    if forbidden:
        raise RuntimeError(
            "Web Player Python payload contains non-runtime files: "
            + ", ".join(sorted(forbidden)[:12])
        )


def _stage_web_shader_sources(
    request: BuildRequest,
    staging: Path,
    player_assets: Path,
    source_package: Path,
) -> None:
    """Generate the shared renderer's GLSL inputs before WGSL translation."""

    from infernux.lib import _Infernux as native

    vertex_path = (
        source_package
        / "resources"
        / "shaders"
        / "fullscreen_triangle.vert"
    )
    if not vertex_path.is_file():
        raise ValueError(f"Infernux fullscreen shader is missing: {vertex_path}")
    request.report("shaders", 0, 2, "Preparing shared fullscreen shader")
    generated_vertex = native._prepare_authored_shader_glsl(
        vertex_path.read_text(encoding="utf-8"),
        str(vertex_path),
    )
    generated_fragment = """#version 450
layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;
void main() {
    outColor = vec4(inUV.x, inUV.y, 0.20 + 0.12 * inUV.x, 1.0);
}
"""
    shader_root = staging / "shader-cook"
    shader_root.mkdir(parents=True, exist_ok=True)
    (shader_root / "fullscreen_triangle.vert").write_text(
        generated_vertex,
        encoding="utf-8",
        newline="\n",
    )
    (shader_root / "web_host.frag").write_text(
        generated_fragment,
        encoding="utf-8",
        newline="\n",
    )
    shader_entries = [
        {
            "name": "Fullscreen Triangle",
            "stage": "vertex",
            "source": "fullscreen_triangle.vert",
        },
        {
            "name": "Web Host",
            "stage": "fragment",
            "source": "web_host.frag",
        },
    ]
    particle_kernels: dict[str, dict[str, object]] = {}
    data_roots = sorted(player_assets.glob("*_Data"))
    if len(data_roots) != 1:
        raise ValueError(
            "Web Player shader cook requires exactly one staged *_Data directory"
        )
    _stage_web_ui_shader_sources(data_roots[0], shader_root, shader_entries, native)
    # Particle AOT artifacts are build inputs, not loose Player files. The
    # generic cooker seals runtime content into Content.inxpkg before Web
    # shader translation runs, so consume the project's current artifact
    # registry and stage only validated WGSL plus its compact catalog.
    particle_root = (
        Path(request.project_root).resolve()
        / "Library"
        / "Artifacts"
        / "Particle"
    )
    if particle_root.is_dir():
        for artifact_path in sorted(particle_root.glob("*.inxparticle")):
            try:
                artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"Web Particle artifact is unreadable: {artifact_path.name}: {error}"
                ) from error
            emitters = artifact.get("gpu_glsl", {}).get("emitters")
            if not isinstance(emitters, list):
                raise ValueError(
                    f"Web Particle artifact has no GLSL emitters: {artifact_path.name}"
                )
            for emitter in emitters:
                if not isinstance(emitter, dict):
                    raise ValueError("Web Particle emitter metadata must be an object")
                kernel_hash = str(emitter.get("kernel_hash", ""))
                stable_id = str(emitter.get("stable_id", ""))
                stages = emitter.get("stages")
                data_layout = emitter.get("data_interface_layout")
                if (
                    not re.fullmatch(r"[0-9a-f]{64}", kernel_hash)
                    or not stable_id
                    or not isinstance(stages, dict)
                ):
                    raise ValueError("Web Particle emitter identity is invalid")
                if emitter.get("collision_enabled") or emitter.get("continuation") is not None:
                    raise ValueError(
                        f"WebGPU Particle emitter {stable_id} uses collision or continuation, "
                        "which is not portable yet"
                    )
                if not isinstance(data_layout, dict) or any(
                    data_layout.get(name)
                    for name in (
                        "mesh_interfaces",
                        "texture2d_parameters",
                        "volume_interfaces",
                    )
                ):
                    raise ValueError(
                        f"WebGPU Particle emitter {stable_id} uses a data interface "
                        "which is not portable yet"
                    )
                expected_stages = {
                    "bootstrap",
                    "init",
                    "update",
                    "update_rendering_fused",
                    "contact_prepare",
                    "contact_solve",
                    "contact_dispatch",
                    "render_reset",
                    "rendering",
                }
                if set(stages) != expected_stages or any(
                    not isinstance(stages[name], str) or not stages[name]
                    for name in expected_stages
                ):
                    raise ValueError(
                        f"WebGPU Particle emitter {stable_id} has incomplete compute stages"
                    )
                existing = particle_kernels.get(kernel_hash)
                kernel_record = {
                    "kernel_hash": kernel_hash,
                    "stable_ids": [stable_id],
                    "state_stride": int(emitter.get("state_stride", 0)),
                    "event_type_count": int(emitter.get("event_type_count", 0)),
                    "update_render_fusion": emitter.get("update_render_fusion"),
                    "stages": {},
                }
                if existing is not None:
                    comparable = {
                        key: value
                        for key, value in existing.items()
                        if key not in {"stable_ids", "stages"}
                    }
                    expected = {
                        key: value
                        for key, value in kernel_record.items()
                        if key not in {"stable_ids", "stages"}
                    }
                    if comparable != expected:
                        raise ValueError(
                            f"Conflicting Web Particle kernel identity: {kernel_hash}"
                        )
                    stable_ids = existing["stable_ids"]
                    if stable_id not in stable_ids:
                        stable_ids.append(stable_id)
                        stable_ids.sort()
                    continue
                for stage_name in sorted(expected_stages):
                    source_name = f"particle-{kernel_hash}-{stage_name}.comp"
                    (shader_root / source_name).write_text(
                        stages[stage_name], encoding="utf-8", newline="\n"
                    )
                    shader_name = f"Particle/{kernel_hash}/{stage_name}"
                    shader_entries.append(
                        {
                            "name": shader_name,
                            "stage": "compute",
                            "source": source_name,
                        }
                    )
                    kernel_record["stages"][stage_name] = shader_name
                particle_kernels[kernel_hash] = kernel_record

    manifest = {
        "$schema": "infernux.web_shader_cook",
        "shaders": shader_entries,
    }
    (shader_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    particle_catalog = {
        "$schema": "infernux.web_particle_catalog",
        "kernels": sorted(
            particle_kernels.values(), key=lambda item: str(item["kernel_hash"])
        ),
    }
    (player_assets / "web-particles").mkdir(parents=True, exist_ok=True)
    (player_assets / "web-particles" / "catalog.json").write_text(
        json.dumps(
            particle_catalog, ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    request.report("shaders", 1, 2, "Shared fullscreen GLSL prepared")


def _stage_web_ui_shader_sources(
    data_root: Path,
    shader_root: Path,
    shader_entries: list[dict[str, str]],
    native: object,
) -> None:
    """Translate exactly the UI shader/material closure sealed by content cook."""
    from infernux.engine.player_package_native import read_entry
    from infernux.engine.player_service_graph import PlayerRuntimeAssetCatalog

    catalog = json.loads(read_entry(data_root / "AssetCatalog.inxcat", "RuntimeAssetCatalog.json"))
    records_artifacts = [
        item for item in catalog["artifacts"]
        if item["runtime_path"] == "Library/RuntimeAssetRecords.json"
    ]
    if len(records_artifacts) != 1:
        raise ValueError("Web UI shader cook requires one sealed runtime asset index")
    records_artifact = records_artifacts[0]
    records = json.loads(read_entry(
        data_root.parent / records_artifact["package"], records_artifact["runtime_path"],
    ))
    runtime = PlayerRuntimeAssetCatalog.from_documents(str(data_root), catalog, records)
    entries = sorted(records["entries"], key=lambda item: item["guid"])

    def payload(entry):
        artifact = runtime.artifact(entry["primary_runtime_artifact_id"])
        return read_entry(data_root.parent / artifact["package"], artifact["runtime_path"])

    shader_assets = {}
    for entry in entries:
        source_path = Path(entry["runtime_path"])
        if source_path.suffix not in {".vert", ".frag"}:
            continue
        metadata = entry["metadata"]["metadata"]
        guid = entry["guid"]
        capabilities = json.loads(metadata["shader_capabilities"]["value"])
        domains = set(capabilities) & {"ScreenUI", "WorldUI"}
        if domains:
            if len(domains) != 1:
                raise ValueError(f"UI shader has ambiguous domains: {guid}")
            has_properties = bool(json.loads(metadata["properties"]["value"]))
            shader_assets[guid] = (
                entry, str(metadata["shader_id"]["value"]), domains.pop(),
                has_properties,
            )

    staged: dict[tuple[str, str], str] = {}
    for entry in entries:
        material_path = Path(entry["runtime_path"])
        if material_path.suffix != ".mat":
            continue
        material_artifact = runtime.artifact(entry["primary_runtime_artifact_id"])
        material_payload = payload(entry)
        # Imported project/package materials are cooked documents; built-in
        # resources retain their authored format in the platform resource pack.
        material = (
            native._decode_asset_document(material_payload)
            if Path(material_artifact["runtime_path"]).suffix == ".inxdoc"
            else json.loads(material_payload)
        )
        stages = material.get("shaders", {})
        if not isinstance(stages, dict):
            continue
        stage_refs = (stages.get("vertex"), stages.get("fragment"))
        if not all(isinstance(reference, dict) for reference in stage_refs):
            continue
        shader_guids = [str(reference.get("guid", "")) for reference in stage_refs]
        if not any(guid in shader_assets for guid in shader_guids):
            continue
        if not all(guid in shader_assets for guid in shader_guids):
            raise ValueError(f"UI material has an incomplete shader GUID pair: {material_path.name}")
        if shader_assets[shader_guids[0]][2] != shader_assets[shader_guids[1]][2]:
            raise ValueError(f"UI material shader domains disagree: {material_path.name}")
        if any(shader_assets[guid][3] for guid in shader_guids):
            raise ValueError(
                "Web UI material shader properties require the material descriptor ABI: "
                f"{material_path.name}"
            )
        for stage, reference, guid in zip(("vertex", "fragment"), stage_refs, shader_guids):
            shader_entry, shader_id, _domain, _has_properties = shader_assets[guid]
            source_path = Path(shader_entry["runtime_path"])
            if source_path.suffix != (".vert" if stage == "vertex" else ".frag"):
                raise ValueError(f"UI shader GUID has the wrong stage: {guid}")
            if shader_id != str(reference.get("shader_id", "")):
                raise ValueError(f"UI material shader identity changed: {guid}")
            identity = (shader_id, stage)
            previous = staged.get(identity)
            if previous is not None:
                if previous != guid:
                    raise ValueError(f"Duplicate UI shader identity: {shader_id} ({stage})")
                continue
            source_name = f"ui-{guid}{source_path.suffix}"
            prepared = native._prepare_authored_shader_glsl(
                payload(shader_entry).decode("utf-8"), source_path.as_posix()
            )
            (shader_root / source_name).write_text(
                prepared, encoding="utf-8", newline="\n"
            )
            shader_entries.append(
                {"name": shader_id, "stage": stage, "source": source_name}
            )
            staged[identity] = guid


def _assemble_web_host(
    request: BuildRequest,
    staging: Path,
    player_assets: Path,
    details: dict[str, object],
    source_package: Path,
    presentation: dict[str, object],
    web_shell: Path,
    project_web_template: Path | None,
) -> tuple[str, ...]:
    """Assemble project content around the plugin's immutable native payload."""
    from infernux.engine.player_package_native import write_pack
    from .shader_pipeline import compile_shader_manifest

    # InxPack carries bytes, not POSIX modes. Restore execute permission only
    # for the two declared native tools when consuming an installed package.
    if os.name != "nt":
        for name in ("glslang", "tint"):
            tool = Path(str(details[name]))
            tool.chmod(tool.stat().st_mode | 0o111)
    compile_shader_manifest(
        staging / "shader-cook" / "manifest.json",
        player_assets / "web-shaders",
        glslang=str(details["glslang"]),
        tint=str(details["tint"]),
    )
    request.report("shaders", 2, 2, "Validated WGSL shader catalog ready")
    runtime = Path(str(details["player_root"]))
    revision = _web_asset_revision(
        staging, player_assets, runtime, presentation=presentation,
        project_template_root=project_web_template,
    )
    build_root = staging / "host-build"
    build_root.mkdir(parents=True)
    for suffix in ("js", "wasm", "data"):
        shutil.copy2(runtime / f"infernux-runtime.{suffix}",
                     build_root / f"infernux-player.{revision}.{suffix}")
    write_pack(
        [(path.relative_to(player_assets).as_posix(), path)
         for path in sorted(player_assets.rglob("*")) if path.is_file()],
        build_root / f"infernux-player.{revision}.inxpkg",
        profile=request.profile.configuration.value,
    )
    for path in (player_assets / "web-branding").iterdir():
        if path.is_file():
            shutil.copy2(path, build_root / path.name)
    shutil.copy2(source_package / "resources/icons/icon.png",
                 build_root / "infernux-logo.png")
    shutil.copy2(player_assets / WEBGPU_CAPABILITY_FILENAME,
                 build_root / WEBGPU_CAPABILITY_FILENAME)
    document = web_shell.read_text(encoding="utf-8")
    replacements = {
        "@INFERNUX_WEB_ASSET_REVISION@": revision,
        "@INFERNUX_WEB_DISPLAY_MODE@": str(presentation["display_mode"]),
        "@INFERNUX_WEB_CANVAS_WIDTH@": str(presentation["window_width"]),
        "@INFERNUX_WEB_CANVAS_HEIGHT@": str(presentation["window_height"]),
        "@INFERNUX_WEB_RENDER_OUTPUT@": json.dumps(presentation["render_output"]),
    }
    for marker, value in replacements.items():
        document = document.replace(marker, value)
    # This loader uses the same InxPack format as desktop content. C++ unpacks
    # into browser memory before starting Python; the web server never exposes
    # the project's directory tree.
    loader = (Path(__file__).parent / "templates" / "load_content.js").read_text(
        encoding="utf-8"
    ).replace("@INFERNUX_WEB_ASSET_REVISION@", revision)
    loader = loader.replace("@INFERNUX_WEB_PRESENTATION@", json.dumps({
        "mode": presentation["display_mode"],
        "width": presentation["window_width"],
        "height": presentation["window_height"],
        "render_output": presentation["render_output"],
    }))
    script = (f"<script>{loader}</script>\n"
              f'<script src="infernux-player.{revision}.js"></script>')
    (build_root / "infernux-player.html").write_text(
        document.replace("{{{ SCRIPT }}}", script), encoding="utf-8", newline="\n"
    )
    request.report("runtime", 1, 1, "Precompiled Web Player assembled")
    return ()


def _web_asset_revision(
    staging: Path,
    player_assets: Path,
    host_template_source: Path,
    *,
    presentation: dict[str, object],
    project_template_root: Path | None = None,
) -> str:
    """Hash every staged byte that can change the browser asset bundle."""

    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            presentation,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    digest.update(b"\0")
    roots = [
        ("player", player_assets),
        ("runtime", host_template_source),
        ("templates", Path(__file__).parent / "templates"),
        ("shaders", staging / "shader-cook"),
    ]
    if project_template_root is not None:
        roots.append(("project-template", project_template_root))
    files: list[tuple[str, Path]] = []
    for label, root in roots:
        if not root.is_dir():
            raise RuntimeError(f"Web asset revision input is missing: {root}")
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            files.append((f"{label}/{path.relative_to(root).as_posix()}", path))
    for relative, path in sorted(files, key=lambda item: item[0]):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()[:24]


__all__ = ["WebPlatformExporter"]
