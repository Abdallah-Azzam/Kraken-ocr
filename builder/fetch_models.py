"""Pre-download Kraken models at image build time."""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

MODEL_ROOT = Path("/models")
XDG_DATA = MODEL_ROOT / "data"
MANIFEST_PATH = MODEL_ROOT / "models.json"

os.environ["XDG_DATA_HOME"] = str(XDG_DATA)
os.environ["HOME"] = "/root"

MODEL_SPECS: dict[str, dict[str, str]] = {
    "segmentation_general": {
        "doi": "10.5281/zenodo.14602569",
        "kind": "segmentation",
    },
    "segmentation_arabic_handwritten": {
        "doi": "10.5281/zenodo.14295555",
        "kind": "segmentation",
    },
    "recognition_printed_en": {
        "doi": "10.5281/zenodo.10592716",
        "kind": "recognition",
    },
    "recognition_printed_ar": {
        "doi": "10.5281/zenodo.7050296",
        "kind": "recognition",
    },
    "recognition_handwritten_en": {
        "doi": "10.5281/zenodo.13788177",
        "kind": "recognition",
    },
    "recognition_handwritten_ar": {
        "doi": "10.5281/zenodo.14295489",
        "kind": "recognition",
    },
}

MODEL_DIR_RE = re.compile(
    r"Model dir:\s*(.+?)\s*\(model files:\s*(.+?)\)",
    re.IGNORECASE,
)


def _find_weight_file(model_dir: Path, hinted_names: str = "") -> Path:
    hints = [name.strip() for name in hinted_names.split(",") if name.strip()]
    for hint in hints:
        candidate = model_dir / hint
        if candidate.is_file():
            return candidate
    for pattern in ("*.safetensors", "*.mlmodel"):
        matches = sorted(model_dir.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No model weights found in {model_dir}")


def _download_with_htrmopo(doi: str) -> dict[str, str]:
    from htrmopo import get_model

    print(f"Falling back to htrmopo download for {doi}...", flush=True)
    model_dir = Path(get_model(doi))
    weight_path = _find_weight_file(model_dir)
    return {
        "doi": doi,
        "path": str(weight_path),
        "dir": str(model_dir),
        "filename": weight_path.name,
    }


def _download_model(key: str, doi: str) -> dict[str, str]:
    print(f"Downloading {key} ({doi})...", flush=True)
    result = subprocess.run(
        ["kraken", "get", doi],
        capture_output=True,
        text=True,
        env={**os.environ},
    )
    output = (result.stdout or "") + (result.stderr or "")
    if output.strip():
        print(output, flush=True)

    if result.returncode == 0:
        match = MODEL_DIR_RE.search(output)
        if match:
            model_dir = Path(match.group(1).strip())
            weight_path = _find_weight_file(model_dir, match.group(2))
            return {
                "doi": doi,
                "path": str(weight_path),
                "dir": str(model_dir),
                "filename": weight_path.name,
            }

    return _download_with_htrmopo(doi)


def main() -> None:
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, str]] = {}
    for key, spec in MODEL_SPECS.items():
        manifest[key] = {
            "kind": spec["kind"],
            **_download_model(key, spec["doi"]),
        }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest to {MANIFEST_PATH}", flush=True)


if __name__ == "__main__":
    main()
