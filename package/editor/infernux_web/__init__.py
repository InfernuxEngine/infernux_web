"""Web build target for Infernux."""
from .doctor import inspect_web_toolchain
from .exporter import WebPlatformExporter

__all__ = ["WebPlatformExporter", "inspect_web_toolchain"]
