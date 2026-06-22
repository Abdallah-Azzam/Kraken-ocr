# Kraken OCR Endpoint

RunPod Serverless worker for **Kraken 7** GPU OCR, built for MedScribe clinical and pharmacy documents. Supports English and Arabic, printed and handwritten text, via image URL or base64.

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

### Printed pharmacy label (Arabic)

```json
{
  "input": {
    "image": "https://example.com/pharmacy-label.jpg",
    "language": "ar",
    "document_type": "printed",
    "binarize": false,
    "include_lines": true
  }
}
```

### Handwritten clinical note (English)

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
  -d '{"input":{"image":"https://example.com/rx.jpg","language":"en","document_type":"printed"}}'
```

## Local test

```bash
docker build --platform linux/amd64 -t kraken-ocr .
docker run --rm --gpus all -v "%cd%/test_input.json:/test_input.json" kraken-ocr python -u /test_local.py
```

## Accuracy expectations

These are **general-purpose** Kraken models, not clinical-fine-tuned. They work well as a baseline for clean scans of printed labels, forms, and notes. Expect lower accuracy on:

- Noisy fax scans, stamps, or watermarks
- Medical abbreviations and domain-specific jargon
- Very faint pencil or rushed clinical shorthand
- Arabic clinical handwriting (Muharaf is trained on archival manuscripts)

Validate on real MedScribe samples before using with production PHI.

## Follow-up: clinical fine-tuning

The baked models are a starting point. For MedScribe production quality, plan a fine-tuning pass on your own annotated clinical/pharmacy data.

### When to fine-tune

- CER (character error rate) on your validation set stays above your target after trying `binarize`, better scans, or `document_type` routing
- Specific recurring failures: drug names, dosages, Arabic diacritics, clinic-specific abbreviations
- A single domain (e.g. UAE pharmacy labels only) where a smaller specialized model would outperform the general one

### Workflow (high level)

1. **Collect ground truth** — Page images plus line-level transcriptions. Prefer PageXML or ALTO (Kraken's native training formats). Tools like [eScriptorium](https://escriptorium.readthedocs.io/) help with annotation.

2. **Normalize transcriptions** — Match the normalization of your base model (CATMuS uses NFKD; Arabic OpenITI/Muharaf use NFD). Keep drug names and abbreviations consistent in ground truth.

3. **Fine-tune recognition** — Start from the closest base model for your script and document type:

   ```bash
   # Example: fine-tune English printed from CATMuS-Print
   ketos train -f path/to/train.txt \
     -t path/to/validation.txt \
     --load 10.5281/zenodo.10592716 \
     --epochs 30 -q early
   ```

   For Arabic printed, start from `10.5281/zenodo.7050296`. For handwritten English, start from McCATMuS (`10.5281/zenodo.13788177`).

4. **Evaluate** — Hold out 10–20% of lines/pages. Track CER per document type and language. Spot-check drug names and numeric dosages manually.

5. **Publish and bake** — Publish the new weights to Zenodo (`ketos publish`) or copy the `.safetensors` / `.mlmodel` into the Docker image. Update `builder/fetch_models.py` and `ROUTING` in `src/predict.py` to point at your fine-tuned DOI or local path.

6. **Redeploy** — Rebuild the image, push to your registry, and update the RunPod endpoint. Consider `min_workers=1` during evaluation to avoid cold-start noise.

### Segmentation fine-tuning (optional)

If line detection fails on your forms (multi-column prescriptions, stamped overlays), fine-tune a segmentation model on PageXML with your layout. Start from the general segmentation model (`10.5281/zenodo.14602569`) or the Muharaf segmentation model for Arabic handwritten.

### Data hygiene for PHI

- Fine-tune only on de-identified or consent-approved data
- Keep training artifacts out of public Zenodo deposits if they contain sensitive patterns
- Prefer baking private fine-tuned weights into a private Docker image rather than publishing to the public Kraken model repository

### Resources

- [Kraken training tutorial](https://kraken.re/6.0.0/tutorials/training.html)
- [Kraken model repository](https://kraken.re/6.0.0/advanced/repo.html)
- [CATMuS transcription guidelines](https://catmus-guidelines.github.io/)

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
