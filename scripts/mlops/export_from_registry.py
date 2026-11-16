"""
Synchronize Serving & Shadow Models from DagsHub MLflow Model Registry (DS108)
=============================================================================
Uses official MLflow 3.x Alias APIs (`get_model_version_by_alias` and
`models:/RainfallForecastHCMC@<alias>`) to download, SHA-256 verify, and
materialize production `.joblib` artifacts into `artifacts/models/`.

Also generates/verifies the lightweight 35-day sample history file
(`artifacts/models/sample_recent_history_35d.csv`) so slim Docker containers
can serve fallback requests without bundling full raw historical datasets.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlflow
import pandas as pd

from src.analysis.loaders import compute_sha256
from src.config.mlops import (
    DEFAULT_REGISTERED_MODEL_NAME,
    init_dagshub_mlflow,
    retry_with_backoff,
)
from src.data.dataquality import check_continuity
from src.data.loader import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ExportFromRegistry")

DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "models"
SAMPLE_HISTORY_FILENAME = "sample_recent_history_35d.csv"

ALIAS_ARTIFACT_MAP: Dict[str, Tuple[str, str]] = {
    "serving": ("sc35_xgboost_tweedie_f3.joblib", "feature_builder_f3.joblib"),
    "shadow_val_champ": ("sc30_lightgbm_tweedie_f2.joblib", "feature_builder_f2.joblib"),
    "shadow_test_contender": ("sc31_lightgbm_tweedie_f3.joblib", "feature_builder_f3.joblib"),
}


def _load_expected_checksums(artifacts_dir: Path) -> Dict[str, str]:
    meta_path = artifacts_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Governance metadata.json not found at {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    checksums = metadata.get("sha256_checksums", {})
    if not checksums:
        raise ValueError(f"sha256_checksums missing or empty in {meta_path}")
    return checksums


def _find_downloaded_file(root_dir: Path, filename: str) -> Path:
    direct = root_dir / filename
    if direct.exists():
        return direct
    matches = list(root_dir.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Expected artifact '{filename}' not found inside {root_dir}")
    return matches[0]


def export_sample_history(
    artifacts_dir: Path,
    days: int = 35,
    verify_only: bool = False,
) -> Optional[Path]:
    """Export or verify the bundled 35-day recent weather snapshot."""
    sample_path = artifacts_dir / SAMPLE_HISTORY_FILENAME
    if verify_only:
        if sample_path.exists():
            df_sample = pd.read_csv(sample_path)
            if len(df_sample) < 30:
                raise ValueError(f"{sample_path} has only {len(df_sample)} rows (< 30).")
            check_continuity(df_sample, date_col="Ngày", raise_on_gap=True)
            logger.info("[PASS] Verified bundled sample history: %s (%d rows)", sample_path, len(df_sample))
            return sample_path
        logger.warning("Bundled sample history %s does not exist yet.", sample_path)
        return None

    try:
        df_raw = DataLoader().load_data()
    except Exception as exc:
        logger.warning("Could not load raw CSV to refresh sample history: %s", exc)
        df_raw = None

    if df_raw is None or df_raw.empty:
        if sample_path.exists():
            logger.info("Using existing bundled sample history at %s", sample_path)
            return sample_path
        raise FileNotFoundError(
            "Neither canonical raw CSV nor existing sample_recent_history_35d.csv is available."
        )

    df_sample = df_raw.tail(days).copy().reset_index(drop=True)
    if "Ngày" in df_sample.columns:
        df_sample["Ngày"] = pd.to_datetime(df_sample["Ngày"]).dt.strftime("%Y-%m-%d")
    check_continuity(df_sample, date_col="Ngày", raise_on_gap=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    df_sample.to_csv(sample_path, index=False, encoding="utf-8")
    logger.info(
        "[EXPORT] Wrote %d-day canonical sample history to %s (%s -> %s)",
        len(df_sample),
        sample_path,
        df_sample["Ngày"].iloc[0],
        df_sample["Ngày"].iloc[-1],
    )
    return sample_path


def verify_and_export_from_registry(
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
    model_name: str = DEFAULT_REGISTERED_MODEL_NAME,
    verify_only: bool = False,
) -> Dict[str, Any]:
    """Resolve official MLflow aliases (`models:/name@alias`), audit SHA-256, and optionally download."""
    expected_checksums = _load_expected_checksums(artifacts_dir)
    client = init_dagshub_mlflow()

    summary: Dict[str, Any] = {
        "registered_model": model_name,
        "verify_only": verify_only,
        "aliases": {},
    }

    for alias, (model_file, fb_file) in ALIAS_ARTIFACT_MAP.items():
        mv = retry_with_backoff(
            lambda a=alias: client.get_model_version_by_alias(model_name, a),
            operation_name=f"get_model_version_by_alias({model_name}@{alias})",
        )
        if mv is None:
            raise LookupError(f"Alias '@{alias}' not found in Model Registry '{model_name}'.")

        tags = dict(getattr(mv, "tags", {}) or {})
        expected_model_hash = expected_checksums.get(model_file)
        expected_fb_hash = expected_checksums.get(fb_file)
        if not expected_model_hash or not expected_fb_hash:
            raise KeyError(
                f"Missing expected checksum for {model_file} or {fb_file} in metadata.json."
            )

        if tags.get("sha256_model") and tags["sha256_model"] != expected_model_hash:
            raise RuntimeError(
                f"Registry tag sha256_model mismatch for @{alias} v{mv.version}: "
                f"{tags['sha256_model']} != {expected_model_hash}"
            )
        if tags.get("sha256_feature_builder") and tags["sha256_feature_builder"] != expected_fb_hash:
            raise RuntimeError(
                f"Registry tag sha256_feature_builder mismatch for @{alias} v{mv.version}: "
                f"{tags['sha256_feature_builder']} != {expected_fb_hash}"
            )

        if verify_only:
            local_model = artifacts_dir / model_file
            local_fb = artifacts_dir / fb_file
            for local_path, exp_hash in ((local_model, expected_model_hash), (local_fb, expected_fb_hash)):
                if not local_path.exists():
                    raise FileNotFoundError(f"Local artifact missing for verify-only check: {local_path}")
                actual = compute_sha256(local_path)
                if actual != exp_hash:
                    raise RuntimeError(
                        f"Local SHA-256 mismatch for {local_path.name}: {actual} != {exp_hash}"
                    )
            logger.info(
                "[PASS] Verified @%s -> v%s (run=%s) & local SHA-256 match.",
                alias,
                mv.version,
                mv.run_id,
            )
            summary["aliases"][alias] = {
                "version": str(mv.version),
                "run_id": mv.run_id,
                "uri": f"models:/{model_name}@{alias}",
                "sha256_model": expected_model_hash,
                "sha256_feature_builder": expected_fb_hash,
            }
            continue

        artifact_uri = f"models:/{model_name}@{alias}/artifacts"
        with tempfile.TemporaryDirectory(prefix=f"mlflow_export_{alias}_") as tmp_dir:
            logger.info("Downloading artifacts from %s ...", artifact_uri)
            downloaded_dir = Path(
                retry_with_backoff(
                    lambda uri=artifact_uri, dst=tmp_dir: mlflow.artifacts.download_artifacts(
                        artifact_uri=uri,
                        dst_path=dst,
                    ),
                    operation_name=f"download_artifacts({artifact_uri})",
                )
            )
            src_model = _find_downloaded_file(downloaded_dir, model_file)
            src_fb = _find_downloaded_file(downloaded_dir, fb_file)

            actual_model_hash = compute_sha256(src_model)
            actual_fb_hash = compute_sha256(src_fb)
            if actual_model_hash != expected_model_hash:
                raise RuntimeError(
                    f"Downloaded SHA-256 mismatch for {model_file} (@{alias} v{mv.version}): "
                    f"{actual_model_hash} != {expected_model_hash}"
                )
            if actual_fb_hash != expected_fb_hash:
                raise RuntimeError(
                    f"Downloaded SHA-256 mismatch for {fb_file} (@{alias} v{mv.version}): "
                    f"{actual_fb_hash} != {expected_fb_hash}"
                )

            artifacts_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_model, artifacts_dir / model_file)
            shutil.copy2(src_fb, artifacts_dir / fb_file)
            logger.info(
                "[SYNC] Exported @%s (v%s) -> %s, %s (SHA-256 100%% matched)",
                alias,
                mv.version,
                model_file,
                fb_file,
            )
            summary["aliases"][alias] = {
                "version": str(mv.version),
                "run_id": mv.run_id,
                "uri": f"models:/{model_name}@{alias}",
                "sha256_model": actual_model_hash,
                "sha256_feature_builder": actual_fb_hash,
            }

    sample_path = export_sample_history(artifacts_dir, days=35, verify_only=verify_only)
    summary["sample_history_path"] = str(sample_path) if sample_path else None
    return summary


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Sync and SHA-256 verify DS108 serving models from DagsHub MLflow Registry."
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Destination model artifact directory (default: artifacts/models)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_REGISTERED_MODEL_NAME,
        help=f"Registered model name (default: {DEFAULT_REGISTERED_MODEL_NAME})",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify Registry aliases and SHA-256 digests against local artifacts without overwriting.",
    )
    args = parser.parse_args(argv)

    result = verify_and_export_from_registry(
        artifacts_dir=args.artifacts_dir,
        model_name=args.model_name,
        verify_only=args.verify_only,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
