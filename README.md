# Kraken OCR Endpoint

RunPod Serverless worker for **Kraken 7** GPU OCR. Supports English and Arabic, printed and handwritten text, via image URL or base64.

## Deploy

1. Build and push the Docker image (linux/amd64 required for RunPod):

```bash
docker build --platform linux/amd64 -t kraken-ocr .
```

2. Create a RunPod Serverless endpoint from the image (or publish via RunPod Hub using `.runpod/hub.json`).

3. Use a **16 GB GPU** (e.g. RTX A4000 / L4) and **≥ 20 GB** container disk.

Five Kraken models are baked into the image at build time (~300–600 MB total weights + PyTorch).

## Features

- Full page OCR: segmentation + line recognition on GPU
- `language`: `en` or `ar`
- `document_type`: `printed` or `handwritten`
- Image input via URL or base64
- Optional per-line text, confidence, and bounding boxes

## Models

| document_type | language | Recognition model |
|---------------|----------|-------------------|
| `printed` | `en` | CATMuS-Print Large (`10.5281/zenodo.10592716`) |
| `printed` | `ar` | OpenITI Arabic printed (`10.5281/zenodo.7050296`) |
| `handwritten` | `en` | McCATMuS (`10.5281/zenodo.13788177`) |
| `handwritten` | `ar` | Muharaf Arabic HTR (`10.5281/zenodo.14295489`) |

Segmentation uses a general multiscript model for printed/EN handwritten, and a dedicated Muharaf segmentation model for Arabic handwritten.

## API

`POST https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync`

### Printed document (Arabic)

```json
{
  "input": {
    "image": "https://example.com/document.jpg",
    "language": "ar",
    "document_type": "printed",
    "binarize": false,
    "include_lines": true
  }
}
```

### Handwritten document (English)

```json
{
  "input": {
    "image_base64": "<base64>",
    "language": "en",
    "document_type": "handwritten",
    "binarize": true,
    "include_lines": true
  }
}
```

### Response

```json
{
  "status": "COMPLETED",
  "output": {
    "text": "full page transcription\nline two",
    "language": "en",
    "document_type": "handwritten",
    "lines": [
      {"text": "...", "confidence": 0.97, "bbox": [0, 10, 200, 40]}
    ],
    "models": {
      "segmentation": "10.5281/zenodo.14602569",
      "recognition": "10.5281/zenodo.13788177"
    }
  }
}
```

### Input parameters

| Field | Default | Description |
|-------|---------|-------------|
| `image` | — | Image URL (use one of `image` or `image_base64`) |
| `image_base64` | — | Base64-encoded image bytes |
| `language` | `en` | `en` or `ar` |
| `document_type` | `printed` | `printed` or `handwritten` |
| `binarize` | `false` | Apply nlbin preprocessing (try `true` for faint pen/pencil) |
| `batch_size` | `32` | Lines per GPU recognition batch |
| `include_lines` | `true` | Return per-line text, confidence, bbox |

### Endpoint environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `DEFAULT_LANGUAGE` | `en` | Fallback when request omits `language` |
| `DEFAULT_DOCUMENT_TYPE` | `printed` | Fallback when request omits `document_type` |
| `DEFAULT_BATCH_SIZE` | `32` | Default recognition batch size |
| `DEFAULT_BINARIZE` | `false` | Default binarization |
| `KRAKEN_PRECISION` | `bf16-mixed` | GPU inference precision |

## Example

```bash
curl -X POST "https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input":{"image":"https://example.com/document.jpg","language":"en","document_type":"printed"}}'
```

## Local test

```bash
docker build --platform linux/amd64 -t kraken-ocr .
docker run --rm --gpus all -v "%cd%/test_input.json:/test_input.json" kraken-ocr python -u /test_local.py
```

## Accuracy expectations

These are **general-purpose** Kraken models. They work well on clean scans of printed labels, forms, and notes. Expect lower accuracy on:

- Noisy fax scans, stamps, or watermarks
- Domain-specific jargon or abbreviations
- Very faint pencil or rushed shorthand
- Arabic handwriting (Muharaf is trained on archival manuscripts)

Validate on your own sample documents before relying on output in production.

## Project layout

```
Kraken-ocr/
├── Dockerfile
├── handler.py
├── src/rp_handler.py      # RunPod handler
├── src/predict.py         # Kraken GPU pipeline
├── builder/fetch_models.py
└── .runpod/hub.json
```
