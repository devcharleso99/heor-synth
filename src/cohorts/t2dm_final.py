from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl
import yaml


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"YAML file did not parse into a dictionary: {path}")

    return data


def write_json(data: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

    return output_path


def build_t2dm_final_cohort(
    index_cohort_path: Path,
    parquet_output_path: Path,
    csv_output_path: Path | None = None,
) -> dict[str, Any]:
    index_cohort_path = Path(index_cohort_path)
    parquet_output_path = Path(parquet_output_path)

    if not index_cohort_path.exists():
        raise FileNotFoundError(
            f"Missing T2DM index cohort: {index_cohort_path}. "
            "Run the t2dm-index command first."
        )

    index_cohort = pl.read_parquet(index_cohort_path)

    required_columns = {
        "patient_id",
        "index_date",
        "age_at_index",
        "meets_adult_age_rule",
        "meets_baseline_history_rule",
        "meets_prior_t1dm_exclusion_rule",
        "meets_pregnancy_exclusion_rule",
        "meets_final_phase1_rules",
    }

    missing_columns = required_columns - set(index_cohort.columns)
    if missing_columns:
        raise ValueError(
            f"T2DM index cohort missing columns required for final cohort: {sorted(missing_columns)}"
        )

    final_cohort = (
        index_cohort
        .filter(pl.col("meets_final_phase1_rules") == True)
        .sort(["index_date", "patient_id"])
    )

    parquet_output_path.parent.mkdir(parents=True, exist_ok=True)
    final_cohort.write_parquet(parquet_output_path)

    csv_written = None
    if csv_output_path is not None:
        csv_output_path = Path(csv_output_path)
        csv_output_path.parent.mkdir(parents=True, exist_ok=True)
        final_cohort.write_csv(csv_output_path)
        csv_written = str(csv_output_path)

    age_summary = {
        "min_age_at_index": None,
        "max_age_at_index": None,
        "mean_age_at_index": None,
    }

    if final_cohort.height > 0 and "age_at_index" in final_cohort.columns:
        age_stats = final_cohort.select(
            pl.col("age_at_index").min().alias("min_age_at_index"),
            pl.col("age_at_index").max().alias("max_age_at_index"),
            pl.col("age_at_index").mean().alias("mean_age_at_index"),
        )

        age_summary = {
            "min_age_at_index": age_stats["min_age_at_index"][0],
            "max_age_at_index": age_stats["max_age_at_index"][0],
            "mean_age_at_index": age_stats["mean_age_at_index"][0],
        }

    manifest = {
        "manifest_type": "t2dm_final_cohort",
        "generated_at_utc": utc_now_iso(),
        "index_cohort_path": str(index_cohort_path),
        "parquet_output_path": str(parquet_output_path),
        "csv_output_path": csv_written,
        "summary": {
            "index_cohort_rows": index_cohort.height,
            "final_cohort_rows": final_cohort.height,
            "excluded_from_final_rows": index_cohort.height - final_cohort.height,
            **age_summary,
        },
        "output_columns": final_cohort.columns,
    }

    return manifest


def build_t2dm_final_from_scenario(
    repo_root: Path,
    scenario_path: Path | str = "configs/scenarios/default_synthea.yaml",
    parquet_output_path: Path | str | None = None,
    csv_output_path: Path | str | None = None,
    manifest_path: Path | str | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root)

    scenario_path = Path(scenario_path)
    if not scenario_path.is_absolute():
        scenario_path = repo_root / scenario_path

    scenario_config = load_yaml(scenario_path)

    gold_dir = Path(scenario_config["paths"]["gold_dir"])
    if not gold_dir.is_absolute():
        gold_dir = repo_root / gold_dir

    index_cohort_path = gold_dir / "t2dm_index_cohort.parquet"

    if parquet_output_path is None:
        parquet_output_path = gold_dir / "t2dm_final_cohort.parquet"
    else:
        parquet_output_path = Path(parquet_output_path)
        if not parquet_output_path.is_absolute():
            parquet_output_path = repo_root / parquet_output_path

    if csv_output_path is None:
        csv_output_path = repo_root / "reports" / "tables" / f"{scenario_config['scenario']['id']}_t2dm_final_cohort.csv"
    else:
        csv_output_path = Path(csv_output_path)
        if not csv_output_path.is_absolute():
            csv_output_path = repo_root / csv_output_path

    manifest = build_t2dm_final_cohort(
        index_cohort_path=index_cohort_path,
        parquet_output_path=parquet_output_path,
        csv_output_path=csv_output_path,
    )

    manifest["scenario"] = {
        "id": scenario_config["scenario"]["id"],
        "name": scenario_config["scenario"]["name"],
        "scenario_path": str(scenario_path),
    }

    manifest["synthea"] = scenario_config["synthea"]

    if manifest_path is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        manifest_path = repo_root / "reports" / "manifests" / f"t2dm_final_cohort_{timestamp}.json"
    else:
        manifest_path = Path(manifest_path)
        if not manifest_path.is_absolute():
            manifest_path = repo_root / manifest_path

    manifest["manifest_path"] = str(manifest_path)
    write_json(manifest, manifest_path)

    return manifest
