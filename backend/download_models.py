"""
Pre-download the VERDANT ONNX models so the first API request is instant.

Run once before starting the server (the Docker image runs it at build time):
    python download_models.py          (from inside backend/)

Weights come from the VERDANT_ONNX_REPO model repo (int8 ONNX exports of the
three pretrained HuggingFace models) into VERDANT_ONNX_DIR (default: onnx_models/).
"""

import sys
import time

from app import ml


def _banner(msg: str) -> None:
    print(f"\n{'─' * 56}\n  {msg}\n{'─' * 56}")


if __name__ == "__main__":
    print("VERDANT — model pre-download")
    print(f"From {ml.ONNX_REPO} into {ml.ONNX_DIR}\n")

    failed = False
    for name, loader in (
        ("Disease classifier", ml.get_disease),
        ("Species classifier", ml.get_species),
        ("Advisor embeddings", ml._kb_embeddings),
    ):
        _banner(f"Loading  {name}")
        t0 = time.perf_counter()
        try:
            loader()
            print(f"  ✓ done in {time.perf_counter() - t0:.1f}s")
        except Exception as exc:
            failed = True
            print(f"  ✗ FAILED: {exc}", file=sys.stderr)

    if failed:
        sys.exit(1)
    _banner("All models ready — start the server with:")
    print("  python -m uvicorn app.main:app --port 8010\n")
