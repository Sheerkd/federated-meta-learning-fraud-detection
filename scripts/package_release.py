"""Create a source handoff archive from an explicit safe allowlist."""

import hashlib
import json
from pathlib import Path
import zipfile


ROOT_FILES = ("README.md", "pyproject.toml", "requirements.lock", ".gitignore")
SOURCE_DIRS = ("configs", "docs", "scripts", "src", "tests")
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".pytest_tmp"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".pt", ".pth", ".parquet", ".csv", ".fab"}


def package_source(root: Path) -> Path:
    root = root.resolve()
    selected = [root / name for name in ROOT_FILES]
    for dirname in SOURCE_DIRS:
        selected.extend(path for path in (root / dirname).rglob("*") if path.is_file())
    paths = []
    for path in selected:
        relative = path.relative_to(root)
        if (path.is_symlink() or not path.resolve().is_relative_to(root)
                or any(part in EXCLUDED_PARTS or part.endswith(".egg-info") for part in relative.parts)
                or path.suffix in EXCLUDED_SUFFIXES):
            continue
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append(path)
    if len(paths) != len(set(paths)) or not paths:
        raise ValueError("Release source selection is empty or duplicated")
    output_dir = root / "dist"
    output_dir.mkdir(exist_ok=True)
    output = output_dir / "fraud_detection_source_0.1.0.zip"
    if output.exists():
        raise FileExistsError(f"Source archive already exists: {output}")
    manifest = {"schema_version": 1, "files": []}
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            relative = path.relative_to(root).as_posix()
            data = path.read_bytes()
            archive.writestr(relative, data)
            manifest["files"].append({"path": relative, "bytes": len(data),
                                      "sha256": hashlib.sha256(data).hexdigest()})
        archive.writestr("SOURCE_MANIFEST.json", json.dumps(manifest, indent=2))
    return output


if __name__ == "__main__":
    print(package_source(Path(__file__).resolve().parents[1]))
