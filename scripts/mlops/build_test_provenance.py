"""Generate Provenance Audit Record for Stage 3 Test Benchmark.

DS108 Rainfall Forecasting Project.
Strictly compliant with Protocol v1.0.21 Option A (PROP-04).
"""

import sys
import os
import json
import hashlib
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "data" / "processed" / "results"
REPORTS_DIR = PROJECT_ROOT / "reports" / "test_benchmark"
PROVENANCE_FILE = REPORTS_DIR / "provenance_test.json"


def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_provenance():
    artifacts = [
        "statistical_test_results.csv",
        "statistical_test_horizon_breakdown.csv",
        "statistical_test_forecasts.npz",
        "ml_direct_test_results.csv",
        "ml_direct_test_horizon_breakdown.csv",
        "ml_direct_test_forecasts.npz",
        "ml_tweedie_test_results.csv",
        "ml_tweedie_test_horizon_breakdown.csv",
        "ml_tweedie_test_forecasts.npz",
        "ml_hurdle_test_results.csv",
        "ml_hurdle_test_horizon_breakdown.csv",
        "ml_hurdle_test_forecasts.npz",
        "ml_hurdle_test_probabilities.npz",
        "ml_hurdle_threshold_selection.json",
        "neural_test_results.csv",
        "neural_test_horizon_breakdown.csv",
        "neural_test_forecasts.npz",
    ]

    file_hashes = {}
    for fname in artifacts:
        fpath = RESULTS_DIR / fname
        if fpath.exists():
            file_hashes[fname] = {
                "sha256": compute_sha256(fpath),
                "size_bytes": fpath.stat().st_size,
                "modified_utc": datetime.fromtimestamp(fpath.stat().st_mtime, timezone.utc).isoformat(),
            }
        else:
            file_hashes[fname] = "MISSING"

    # Git metadata
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT).decode().strip()
    except Exception:
        git_commit = "UNKNOWN"

    provenance_payload = {
        "benchmark_partition": "Final Test (2,184 origins, H=7, 15,288 points)",
        "protocol_version": "v1.0.21",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "environment": {
            "os": platform.platform(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
        },
        "artifacts_sha256": file_hashes,
        "provenance_statement": (
            "Final Test benchmark evaluation executed under Single-Fit Policy (Decision M2) "
            "with causal expanding lookback and frozen weights fit once on Dev Train. "
            "Hurdle Strategy B strictly uses locked validation thresholds tau_M*."
        )
    }

    with open(PROVENANCE_FILE, "w", encoding="utf-8") as f:
        json.dump(provenance_payload, f, indent=2)

    print(f"Provenance saved to {PROVENANCE_FILE}")
    return provenance_payload


if __name__ == "__main__":
    generate_provenance()

