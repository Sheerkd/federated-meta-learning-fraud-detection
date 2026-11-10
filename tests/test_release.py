import hashlib
import json
import zipfile

from scripts.package_release import ROOT_FILES, package_source


def test_source_archive_contains_code_and_excludes_local_data(tmp_path):
    for name in ROOT_FILES:
        (tmp_path / name).write_text(name, encoding="utf-8")
    for dirname in ("configs", "docs", "scripts", "src", "tests"):
        folder = tmp_path / dirname
        folder.mkdir()
        (folder / "included.txt").write_text(dirname, encoding="utf-8")
    (tmp_path / "data/raw").mkdir(parents=True)
    (tmp_path / "data/raw/creditcard.csv").write_text("sensitive", encoding="utf-8")
    (tmp_path / "clients/bank_a").mkdir(parents=True)
    (tmp_path / "clients/bank_a/record.json").write_text("sensitive", encoding="utf-8")
    (tmp_path / "src/__pycache__").mkdir()
    (tmp_path / "src/__pycache__/cached.pyc").write_bytes(b"compiled")
    output = package_source(tmp_path)
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        assert "README.md" in names and "src/included.txt" in names
        assert not any(name.startswith(("data/", "clients/", "runs/")) for name in names)
        assert not any("__pycache__" in name or name.endswith(".pyc") for name in names)
        manifest = json.loads(archive.read("SOURCE_MANIFEST.json"))
        for item in manifest["files"]:
            assert hashlib.sha256(archive.read(item["path"])).hexdigest() == item["sha256"]
