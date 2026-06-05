"""Tiny local build backend for OpenEvalLab.

This keeps `pip install -e .` working in restricted environments without fetching a
build backend from PyPI. It implements the minimal wheel hooks needed by pip.
"""

from __future__ import annotations

import base64
import csv
import hashlib
from pathlib import Path
import zipfile

NAME = "openevallab"
VERSION = "0.1.0"
DIST_INFO = f"{NAME}-{VERSION}.dist-info"
ROOT = Path(__file__).parent.resolve()


def _metadata() -> str:
    readme = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""
    return "\n".join(
        [
            "Metadata-Version: 2.3",
            f"Name: {NAME}",
            f"Version: {VERSION}",
            "Summary: A lightweight research toolkit for evaluating language models and turning failures into better benchmarks.",
            "Requires-Python: >=3.10",
            "License: MIT",
            "Author: OpenEvalLab Contributors",
            "Description-Content-Type: text/markdown",
            "",
            readme,
        ]
    )


def _wheel() -> str:
    return "\n".join(
        [
            "Wheel-Version: 1.0",
            "Generator: openevallab-local-backend",
            "Root-Is-Purelib: true",
            "Tag: py3-none-any",
            "",
        ]
    )


def _entry_points() -> str:
    return "[console_scripts]\nopenevallab = openevallab.cli:main\n"


def _hash(data: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return f"sha256={digest}"


def _write_wheel(wheel_directory: str, files: dict[str, bytes]) -> str:
    wheel_name = f"{NAME}-{VERSION}-py3-none-any.whl"
    wheel_path = Path(wheel_directory) / wheel_name
    records: list[tuple[str, str, str]] = []
    with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path, data in files.items():
            zf.writestr(path, data)
            records.append((path, _hash(data), str(len(data))))
        record_path = f"{DIST_INFO}/RECORD"
        rows = records + [(record_path, "", "")]
        record_text = ""
        from io import StringIO

        buf = StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerows(rows)
        record_text = buf.getvalue()
        zf.writestr(record_path, record_text.encode("utf-8"))
    return wheel_name


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    files = {
        f"{NAME}.pth": f"{ROOT / 'src'}\n".encode("utf-8"),
        f"{DIST_INFO}/METADATA": _metadata().encode("utf-8"),
        f"{DIST_INFO}/WHEEL": _wheel().encode("utf-8"),
        f"{DIST_INFO}/entry_points.txt": _entry_points().encode("utf-8"),
    }
    return _write_wheel(wheel_directory, files)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    files = {
        f"{DIST_INFO}/METADATA": _metadata().encode("utf-8"),
        f"{DIST_INFO}/WHEEL": _wheel().encode("utf-8"),
        f"{DIST_INFO}/entry_points.txt": _entry_points().encode("utf-8"),
    }
    for file_path in (ROOT / "src" / NAME).rglob("*.py"):
        arcname = str(file_path.relative_to(ROOT / "src"))
        files[arcname] = file_path.read_bytes()
    return _write_wheel(wheel_directory, files)


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    dist = Path(metadata_directory) / DIST_INFO
    dist.mkdir(parents=True, exist_ok=True)
    (dist / "METADATA").write_text(_metadata(), encoding="utf-8")
    (dist / "WHEEL").write_text(_wheel(), encoding="utf-8")
    (dist / "entry_points.txt").write_text(_entry_points(), encoding="utf-8")
    return DIST_INFO


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    return prepare_metadata_for_build_editable(metadata_directory, config_settings)
