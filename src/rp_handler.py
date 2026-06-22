"""RunPod serverless handler for Kraken OCR."""

from __future__ import annotations

import base64
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import runpod
from predict import Predictor
from rp_schema import INPUT_VALIDATIONS
from runpod.serverless.utils import download_files_from_urls, rp_cleanup, rp_debugger
from runpod.serverless.utils.rp_validator import validate

MODEL = Predictor()
MODEL.setup()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in ("1", "true", "yes")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return int(raw)


def _apply_env_defaults(job_input: dict) -> dict:
    resolved = dict(job_input)
    if not resolved.get("language"):
        resolved["language"] = os.environ.get("DEFAULT_LANGUAGE", "en")
    if not resolved.get("document_type"):
        resolved["document_type"] = os.environ.get("DEFAULT_DOCUMENT_TYPE", "printed")
    if resolved.get("binarize") is None:
        resolved["binarize"] = _env_bool("DEFAULT_BINARIZE", False)
    if resolved.get("batch_size") is None:
        resolved["batch_size"] = _env_int("DEFAULT_BATCH_SIZE", 32)
    if resolved.get("include_lines") is None:
        resolved["include_lines"] = _env_bool("DEFAULT_INCLUDE_LINES", True)
    return resolved


def base64_to_tempfile(base64_file: str, filename: str | None = None) -> str:
    suffix = Path(filename or "image.png").suffix or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        temp_file.write(base64.b64decode(base64_file))
    return temp_file.name


@rp_debugger.FunctionTimer
def run_kraken_job(job: Dict[str, Any]) -> Dict[str, Any]:
    job_input = job["input"]

    with rp_debugger.LineTimer("validation_step"):
        input_validation = validate(job_input, INPUT_VALIDATIONS)
        if "errors" in input_validation:
            return {"error": str(input_validation["errors"])}
        job_input = _apply_env_defaults(input_validation["validated_input"])

    has_image_url = bool(job_input.get("image"))
    has_image_base64 = bool(job_input.get("image_base64"))
    if not has_image_url and not has_image_base64:
        return {"error": "Must provide either image or image_base64"}
    if has_image_url and has_image_base64:
        return {"error": "Must provide either image or image_base64, not both"}

    image_input: str | None = None
    if has_image_url:
        with rp_debugger.LineTimer("download_step"):
            downloaded = download_files_from_urls(job["id"], [job_input["image"]])
            image_input = downloaded[0] if downloaded else None
    else:
        image_input = base64_to_tempfile(
            job_input["image_base64"],
            job_input.get("image_filename"),
        )

    if not image_input:
        return {"error": f"Failed to download image from URL: {job_input.get('image')}"}

    try:
        with rp_debugger.LineTimer("prediction_step"):
            result = MODEL.predict(
                image_path=image_input,
                language=job_input["language"],
                document_type=job_input["document_type"],
                binarize=job_input["binarize"],
                batch_size=job_input["batch_size"],
                include_lines=job_input["include_lines"],
            )
    except Exception as exc:  # noqa: BLE001 - return error payload to client
        return {"error": str(exc)}
    finally:
        with rp_debugger.LineTimer("cleanup_step"):
            rp_cleanup.clean(["input_objects"])

    return result


if __name__ == "__main__":
    runpod.serverless.start({"handler": run_kraken_job})
