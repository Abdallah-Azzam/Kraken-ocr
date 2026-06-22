"""RunPod Hub handler (see .runpod/hub.json)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import runpod
from rp_handler import run_kraken_job

runpod.serverless.start({"handler": run_kraken_job})
