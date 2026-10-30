"""Install checksum-pinned kitchen/GR1 assets using the upstream layouts."""

import hashlib
import importlib
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zipfile import ZipFile


def verify_file(path, item):
    if path.stat().st_size != item["bytes"]:
        raise RuntimeError(f"Asset size mismatch: {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != item["sha256"]:
        raise RuntimeError(f"Asset SHA-256 mismatch: {path.name}")


def download_file(cache, name, item):
    target = cache / name
    if not target.exists():
        temporary = target.with_suffix(target.suffix + ".partial")
        subprocess.run([
            "curl", "--fail", "--location", "--silent", "--show-error",
            "--retry", "5", "--connect-timeout", "30", "--output",
            str(temporary), item["url"],
        ], check=True)
        verify_file(temporary, item)
        temporary.replace(target)
    verify_file(target, item)
    print(f"Asset verified: {name}", flush=True)
    return target


def extract_archive(archive, config):
    destination = Path(config["folder"]).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as handle:
        invalid = handle.testzip()
        if invalid is not None:
            raise RuntimeError(f"Invalid asset ZIP member: {invalid}")
        # The pinned upstream archives include their destination directory.
        handle.extractall(destination.parent)
    if not destination.is_dir():
        raise RuntimeError(f"Asset archive did not create {destination}")


def main():
    source = Path(sys.argv[1]).resolve()
    cache = Path(sys.argv[2]).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    inputs = json.loads(Path(__file__).with_name("asset-inputs.json").read_text())
    sys.path.insert(0, str(source / "robocasa/scripts"))
    kitchen = importlib.import_module("download_kitchen_assets")
    groot = importlib.import_module("download_groot_assets")
    registry = {name: kitchen.DOWNLOAD_ASSET_REGISTRY[name]
                for name in ("textures", "fixtures", "objaverse", "generative_textures")}
    gr1 = groot.get_groot_asset_registry(
        groot.DownloadConfig.from_args(), inputs["groot_revision"],
    )
    if set(gr1) != {"lightwheel", "sketchfab"}:
        raise RuntimeError("The pinned GR1 asset registries are incomplete")
    registry.update(gr1)
    for name, config in registry.items():
        if config["url"] != inputs["archives"][name]["url"]:
            raise RuntimeError(f"Upstream asset URL differs from the lock: {name}")
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {name: pool.submit(download_file, cache, name + ".zip", item)
                   for name, item in inputs["archives"].items()}
        archives = {name: future.result() for name, future in futures.items()}
    for name, archive in archives.items():
        extract_archive(archive, registry[name])
    license_root = source / "robocasa/models/assets/objects/groot-attributions"
    license_root.mkdir(exist_ok=True)
    for name, item in inputs["licenses"].items():
        shutil.copyfile(download_file(cache, name, item), license_root / name)
    output = source / "roboscope-assets-provenance.json"
    output.write_text(json.dumps(inputs, indent=2) + "\n")
    print("RoboCasa kitchen and GR1 asset preparation passed", flush=True)


if __name__ == "__main__":
    main()
