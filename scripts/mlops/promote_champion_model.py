"""
Script: promote_champion_model.py
=================================
Cloud Model Registry Governance CLI for DagsHub Hosted MLflow Server.

Usage:
  python scripts/mlops/promote_champion_model.py --status
  python scripts/mlops/promote_champion_model.py --promote-scenario SC-30 --dry-run
  python scripts/mlops/promote_champion_model.py --promote-scenario SC-30
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.config.mlops import (
    DEFAULT_REGISTERED_MODEL_NAME,
    ModelRegistryGovernance,
    init_dagshub_mlflow,
    retry_with_backoff,
)

SUPPORTED_SCENARIOS = ("SC-35", "SC-30", "SC-31")


def get_registry_versions(client: MlflowClient, model_name: str) -> List:
    versions = retry_with_backoff(
        lambda: client.search_model_versions(f"name = '{model_name}'"),
        operation_name=f"search_model_versions({model_name})",
    )
    versions.sort(key=lambda mv: int(mv.version))
    return versions


def get_alias_map(client: MlflowClient, model_name: str) -> Dict[str, str]:
    """Return mapping of version_str -> comma-separated aliases."""
    reg_model = retry_with_backoff(
        lambda: client.get_registered_model(model_name),
        operation_name=f"get_registered_model({model_name})",
    )
    ver_to_aliases: Dict[str, List[str]] = {}
    aliases = getattr(reg_model, "aliases", None) or {}
    for alias_name, ver in aliases.items():
        ver_to_aliases.setdefault(str(ver), []).append(alias_name)
    return {ver: ", ".join(sorted(a_list)) for ver, a_list in ver_to_aliases.items()}


def print_status(client: MlflowClient, model_name: str = DEFAULT_REGISTERED_MODEL_NAME) -> None:
    versions = get_registry_versions(client, model_name)
    alias_map = get_alias_map(client, model_name)

    print("=" * 100)
    print(f"DAGSHUB MODEL REGISTRY GOVERNANCE STATUS: {model_name}")
    print(f"Tracking URI: {client.tracking_uri}")
    print("=" * 100)

    if not versions:
        print("No model versions found in registry.")
        return

    rows = []
    for mv in versions:
        tags = getattr(mv, "tags", {}) or {}
        ver_str = str(mv.version)
        sha_m = tags.get("sha256_model", "")
        rows.append(
            {
                "version": ver_str,
                "scenario_id": tags.get("scenario_id", "N/A"),
                "feature_level": tags.get("feature_level", "N/A"),
                "alias": alias_map.get(ver_str) or tags.get("governance_alias", "N/A"),
                "stage": getattr(mv, "current_stage", "None"),
                "governance_role": tags.get("governance_role", "N/A"),
                "sha256_model": f"{sha_m[:12]}..." if sha_m else "N/A",
                "run_id": mv.run_id,
            }
        )

    df_status = pd.DataFrame(rows)
    print(df_status.to_string(index=False))
    print("=" * 100)


def promote_scenario(
    client: MlflowClient,
    target_scenario: str,
    model_name: str = DEFAULT_REGISTERED_MODEL_NAME,
    dry_run: bool = False,
) -> None:
    versions = get_registry_versions(client, model_name)
    if not versions:
        raise RuntimeError(f"No versions registered under '{model_name}'.")

    # Locate target scenario version (pick highest version if multiple)
    matching_target = [
        mv for mv in versions if (getattr(mv, "tags", {}) or {}).get("scenario_id") == target_scenario
    ]
    if not matching_target:
        raise LookupError(
            f"Scenario '{target_scenario}' not found in '{model_name}' registry versions."
        )
    target_mv = matching_target[-1]
    target_ver = str(target_mv.version)

    # Identify current serving version
    current_serving_mv = None
    try:
        current_serving_mv, _ = ModelRegistryGovernance.resolve_model_version(
            client, model_name, "serving"
        )
    except LookupError:
        pass

    print("=" * 80)
    print(f"PROMOTION PLAN FOR '{model_name}' (dry_run={dry_run})")
    print("=" * 80)
    if current_serving_mv is not None:
        curr_tags = getattr(current_serving_mv, "tags", {}) or {}
        print(
            f"Current @serving : v{current_serving_mv.version} "
            f"({curr_tags.get('scenario_id', 'unknown')}, stage={current_serving_mv.current_stage})"
        )
    else:
        print("Current @serving : <none>")

    target_tags = getattr(target_mv, "tags", {}) or {}
    print(
        f"Target  @serving : v{target_ver} "
        f"({target_scenario}, current_alias={target_tags.get('governance_alias', 'none')}, "
        f"stage={target_mv.current_stage})"
    )

    if current_serving_mv is not None and str(current_serving_mv.version) == target_ver:
        print("\nTarget scenario is ALREADY the active @serving model. No changes needed.")
        return

    if dry_run:
        print("\n[DRY RUN] Would execute:")
        if current_serving_mv is not None:
            print(
                f"  1. Demote v{current_serving_mv.version}: tag governance_alias='former_serving', "
                "stage -> 'Archived'"
            )
        print(
            f"  2. Promote v{target_ver} ({target_scenario}): alias 'serving' -> v{target_ver}, "
            "tag governance_alias='serving', stage -> 'Production'"
        )
        print("=" * 80)
        return

    timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # 1. Demote previous serving version tags & stage so tag fallback stays unambiguous
    if current_serving_mv is not None:
        old_ver = str(current_serving_mv.version)
        old_alias = (getattr(current_serving_mv, "tags", {}) or {}).get("governance_alias", "serving")
        demoted_alias = "former_serving" if old_alias == "serving" else old_alias
        retry_with_backoff(
            lambda: client.set_model_version_tag(model_name, old_ver, "governance_alias", demoted_alias),
            operation_name=f"tag former serving v{old_ver}",
        )
        retry_with_backoff(
            lambda: client.set_model_version_tag(model_name, old_ver, "demoted_at", timestamp_utc),
            operation_name=f"tag demoted_at v{old_ver}",
        )
        try:
            retry_with_backoff(
                lambda: client.transition_model_version_stage(
                    name=model_name,
                    version=old_ver,
                    stage="Archived",
                    archive_existing_versions=False,
                ),
                operation_name=f"archive old serving v{old_ver}",
            )
        except Exception as exc:
            print(f"Warning: stage transition to Archived skipped for v{old_ver}: {exc}")

    # 2. Promote target version to serving / Production
    prev_alias = target_tags.get("governance_alias", "")
    if prev_alias and prev_alias != "serving":
        # Remove old shadow alias pointer if supported
        try:
            retry_with_backoff(
                lambda: client.delete_registered_model_alias(model_name, prev_alias),
                operation_name=f"delete old alias {prev_alias}",
            )
        except MlflowException:
            pass
        retry_with_backoff(
            lambda: client.set_model_version_tag(model_name, target_ver, "promoted_from_alias", prev_alias),
            operation_name=f"tag promoted_from_alias v{target_ver}",
        )

    retry_with_backoff(
        lambda: client.set_model_version_tag(model_name, target_ver, "governance_alias", "serving"),
        operation_name=f"tag governance_alias=serving v{target_ver}",
    )
    retry_with_backoff(
        lambda: client.set_model_version_tag(model_name, target_ver, "promoted_at", timestamp_utc),
        operation_name=f"tag promoted_at v{target_ver}",
    )

    try:
        retry_with_backoff(
            lambda: client.transition_model_version_stage(
                name=model_name,
                version=target_ver,
                stage="Production",
                archive_existing_versions=False,
            ),
            operation_name=f"transition v{target_ver} to Production",
        )
    except Exception as exc:
        print(f"Warning: stage transition to Production skipped for v{target_ver}: {exc}")

    try:
        retry_with_backoff(
            lambda: client.set_registered_model_alias(model_name, "serving", target_ver),
            operation_name=f"set alias serving -> v{target_ver}",
        )
    except MlflowException as exc:
        print(f"Warning: alias endpoint skipped (tag fallback active): {exc}")

    print(f"\nSuccessfully promoted {target_scenario} (v{target_ver}) to @serving / Production!\n")
    print_status(client, model_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manage DagsHub MLflow Champion/Shadow Governance for RainfallForecastHCMC"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--status",
        action="store_true",
        help="Query DagsHub Model Registry and display all versions, checksums, aliases, and stages.",
    )
    group.add_argument(
        "--promote-scenario",
        choices=SUPPORTED_SCENARIOS,
        help="Promote the specified scenario (SC-35, SC-30, or SC-31) to @serving (Production).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview promotion changes without modifying DagsHub Model Registry.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = init_dagshub_mlflow()

    if args.status:
        print_status(client, DEFAULT_REGISTERED_MODEL_NAME)
    elif args.promote_scenario:
        promote_scenario(
            client,
            target_scenario=args.promote_scenario,
            model_name=DEFAULT_REGISTERED_MODEL_NAME,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()

