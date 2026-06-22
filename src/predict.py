"""Kraken OCR predictor with GPU segment + recognize pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from kraken import binarization
from kraken.configs import RecognitionInferenceConfig, SegmentationInferenceConfig
from kraken.tasks import RecognitionTaskModel, SegmentationTaskModel
from runpod.serverless.utils import rp_cuda

DEFAULT_MANIFEST = Path("/models/models.json")
DEFAULT_BATCH_SIZE = int(os.environ.get("DEFAULT_BATCH_SIZE", "32"))
DEFAULT_NUM_LINE_WORKERS = int(os.environ.get("DEFAULT_NUM_LINE_WORKERS", "4"))
DEFAULT_PRECISION = os.environ.get("KRAKEN_PRECISION", "bf16-mixed")

ROUTING: dict[tuple[str, str], dict[str, str]] = {
    ("printed", "en"): {
        "segmentation_key": "segmentation_general",
        "recognition_key": "recognition_printed_en",
    },
    ("printed", "ar"): {
        "segmentation_key": "segmentation_general",
        "recognition_key": "recognition_printed_ar",
    },
    ("handwritten", "en"): {
        "segmentation_key": "segmentation_general",
        "recognition_key": "recognition_handwritten_en",
    },
    ("handwritten", "ar"): {
        "segmentation_key": "segmentation_arabic_handwritten",
        "recognition_key": "recognition_handwritten_ar",
    },
}

VALID_LANGUAGES = {"en", "ar"}
VALID_DOCUMENT_TYPES = {"printed", "handwritten"}


def _boundary_to_bbox(boundary: Any) -> Optional[List[int]]:
    if not boundary:
        return None
    xs: List[float] = []
    ys: List[float] = []
    for point in boundary:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            xs.append(float(point[0]))
            ys.append(float(point[1]))
    if not xs or not ys:
        return None
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def _mean_confidence(confidences: Any) -> Optional[float]:
    if not confidences:
        return None
    try:
        values = [float(value) for value in confidences]
    except (TypeError, ValueError):
        return None
    if not values:
        return None
    return round(sum(values) / len(values), 4)


class Predictor:
    """Kraken OCR predictor with eager-loaded segmentation and recognition models."""

    def __init__(self, manifest_path: Path | str = DEFAULT_MANIFEST) -> None:
        self.manifest_path = Path(manifest_path)
        self.manifest: dict[str, dict[str, str]] = {}
        self.segmenters: dict[str, SegmentationTaskModel] = {}
        self.recognizers: dict[str, RecognitionTaskModel] = {}

    def setup(self) -> None:
        if not rp_cuda.is_available():
            raise RuntimeError("CUDA GPU is required but not available")

        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Model manifest not found: {self.manifest_path}")

        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        for key, entry in self.manifest.items():
            path = entry["path"]
            kind = entry.get("kind")
            print(f"Loading {kind} model '{key}' from {path}...", flush=True)
            if kind == "segmentation":
                self.segmenters[key] = SegmentationTaskModel.load_model(path)
            elif kind == "recognition":
                self.recognizers[key] = RecognitionTaskModel.load_model(path)
            else:
                raise ValueError(f"Unknown model kind for {key}: {kind}")
        print("All Kraken models loaded.", flush=True)

    def _resolve_models(
        self, document_type: str, language: str
    ) -> Tuple[SegmentationTaskModel, RecognitionTaskModel, dict[str, str]]:
        document_type = document_type.lower()
        language = language.lower()
        if document_type not in VALID_DOCUMENT_TYPES:
            raise ValueError(
                f"document_type must be one of {sorted(VALID_DOCUMENT_TYPES)}"
            )
        if language not in VALID_LANGUAGES:
            raise ValueError(f"language must be one of {sorted(VALID_LANGUAGES)}")

        route = ROUTING.get((document_type, language))
        if not route:
            raise ValueError(f"No model route for {document_type}/{language}")

        seg_key = route["segmentation_key"]
        rec_key = route["recognition_key"]
        segmenter = self.segmenters.get(seg_key)
        recognizer = self.recognizers.get(rec_key)
        if segmenter is None or recognizer is None:
            raise RuntimeError(f"Models not loaded for route {document_type}/{language}")

        return segmenter, recognizer, {
            "segmentation": self.manifest[seg_key]["doi"],
            "recognition": self.manifest[rec_key]["doi"],
        }

    def predict(
        self,
        image_path: str,
        language: str = "en",
        document_type: str = "printed",
        binarize: bool = False,
        batch_size: int = DEFAULT_BATCH_SIZE,
        include_lines: bool = True,
    ) -> Dict[str, Any]:
        segmenter, recognizer, model_meta = self._resolve_models(
            document_type=document_type,
            language=language,
        )

        with Image.open(image_path) as image:
            im = image.convert("RGB")

        if binarize:
            im = binarization.nlbin(im)

        seg_config = SegmentationInferenceConfig(
            accelerator="gpu",
            device=[0],
        )
        rec_config = RecognitionInferenceConfig(
            accelerator="gpu",
            device=[0],
            precision=DEFAULT_PRECISION,
            batch_size=batch_size,
            num_line_workers=DEFAULT_NUM_LINE_WORKERS,
        )

        segmentation = segmenter.predict(im=im, config=seg_config)

        lines: List[Dict[str, Any]] = []
        text_parts: List[str] = []
        for record in recognizer.predict(
            im=im,
            segmentation=segmentation,
            config=rec_config,
        ):
            prediction = getattr(record, "prediction", "") or ""
            if prediction:
                text_parts.append(prediction)
            if include_lines:
                line_entry: Dict[str, Any] = {"text": prediction}
                confidence = _mean_confidence(getattr(record, "confidences", None))
                if confidence is not None:
                    line_entry["confidence"] = confidence
                bbox = _boundary_to_bbox(getattr(record, "boundary", None))
                if bbox is not None:
                    line_entry["bbox"] = bbox
                lines.append(line_entry)

        return {
            "text": "\n".join(text_parts),
            "language": language.lower(),
            "document_type": document_type.lower(),
            "lines": lines if include_lines else [],
            "models": model_meta,
        }
