"""Run the handler locally without RunPod (requires GPU + baked models)."""
import json
from pathlib import Path

from rp_handler import run_kraken_job


def main() -> None:
    payload = json.loads(Path("/test_input.json").read_text(encoding="utf-8"))
    result = run_kraken_job({"id": "local-test", "input": payload["input"]})
    print(json.dumps(result, indent=2)[:4000])


if __name__ == "__main__":
    main()
