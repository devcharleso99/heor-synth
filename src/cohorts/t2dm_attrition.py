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


def build_t2dm_attrition_table(
    silver_dir: Path,
    index_cohort_path: Path,
    parquet_output_path: Path,
    csv_output_path: Path | None = None,
) -> dict[str, Any]:
    silver_dir = Path(silver_dir)
    index_cohort_path = Path(index_cohort_path)
    parquet_output_path = Path(parquet_output_path)

    patients_path = silver_dir / "patients.parquet"

    if not patients_path.exists():
        raise FileNotFoundError(f"Missing silver patients table: {patients_path}")

    if not index_cohort_path.exists():
        raise FileNotFoundError(
            f"Missing T2DM index cohort: {index_cohort_path}. "
            "Run the t2dm-index command first."
        )

    patients = pl.read_parquet(patients_path)
    index_cohort = pl.read_parquet(index_cohort_path)

    required_index_columns = {
        "patient_id",
        "index_date",
        "age_at_index",
        "meets_adult_age_rule",
        "observable_history_days_before_index",
        "meets_baseline_history_rule",
        "has_prior_t1dm_before_index",
        "has_pregnancy_window_before_or_at_index",
        "meets_final_phase1_rules",
    }

    missing_index_columns = required_index_columns - set(index_cohort.columns)
    if missing_index_columns:
        raise ValueError(
            f"T2DM index cohort missing columns: {sorted(missing_index_columns)}"
        )

    raw_patients_n = patients.select("patient_id").unique().height
    t2dm_patients_n = index_cohort.select("patient_id").unique().height

    adult_index_cohort = index_cohort.filter(pl.col("meets_adult_age_rule") == True)

    adult_t2dm_patients_n = (
        adult_index_cohort
        .select("patient_id")
        .unique()
        .height
    )

    baseline_eligible = adult_index_cohort.filter(
        pl.col("meets_baseline_history_rule") == True
    )

    baseline_eligible_patients_n = baseline_eligible.select("patient_id").unique().height

    after_t1dm_exclusion = baseline_eligible.filter(
        pl.col("has_prior_t1dm_before_index") == False
    )

    after_t1dm_exclusion_n = after_t1dm_exclusion.select("patient_id").unique().height

    after_pregnancy_exclusion = after_t1dm_exclusion.filter(
        pl.col("has_pregnancy_window_before_or_at_index") == False
    )

    after_pregnancy_exclusion_n = after_pregnancy_exclusion.select("patient_id").unique().height

    final_cohort_n = (
        index_cohort
        .filter(pl.col("meets_final_phase1_rules") == True)
        .select("patient_id")
        .unique()
        .height
    )

    rows = [
        {
            "step_number": 1,
            "step_id": "raw_synthea_population",
            "criterion": "Raw Synthea population with patient records",
            "input_n": raw_patients_n,
            "excluded_n": 0,
            "remaining_n": raw_patients_n,
            "notes": "Starting population from silver patients table.",
        },
        {
            "step_number": 2,
            "step_id": "has_t2dm_diagnosis",
            "criterion": "At least one qualifying Type 2 Diabetes Mellitus diagnosis",
            "input_n": raw_patients_n,
            "excluded_n": raw_patients_n - t2dm_patients_n,
            "remaining_n": t2dm_patients_n,
            "notes": "Uses SNOMED code 44054006 from silver conditions table.",
        },
        {
            "step_number": 3,
            "step_id": "adult_at_index",
            "criterion": "Age >= 18 years at first qualifying T2DM index date",
            "input_n": t2dm_patients_n,
            "excluded_n": t2dm_patients_n - adult_t2dm_patients_n,
            "remaining_n": adult_t2dm_patients_n,
            "notes": "Age calculated at the first qualifying T2DM diagnosis date.",
        },
        {
            "step_number": 4,
            "step_id": "baseline_history_365_days",
            "criterion": "At least 365 days of observable history before index date",
            "input_n": adult_t2dm_patients_n,
            "excluded_n": adult_t2dm_patients_n - baseline_eligible_patients_n,
            "remaining_n": baseline_eligible_patients_n,
            "notes": "Observable history currently measured from earliest encounter date to index date.",
        },
        {
            "step_number": 5,
            "step_id": "exclude_prior_type_1_diabetes",
            "criterion": "No prior Type 1 Diabetes Mellitus diagnosis before index date",
            "input_n": baseline_eligible_patients_n,
            "excluded_n": baseline_eligible_patients_n - after_t1dm_exclusion_n,
            "remaining_n": after_t1dm_exclusion_n,
            "notes": "Uses SNOMED code 46635009 before the T2DM index date.",
        },
        {
            "step_number": 6,
            "step_id": "exclude_pregnancy_window",
            "criterion": "No pregnancy or gestational-window diagnosis before or at index date",
            "input_n": after_t1dm_exclusion_n,
            "excluded_n": after_t1dm_exclusion_n - after_pregnancy_exclusion_n,
            "remaining_n": after_pregnancy_exclusion_n,
            "notes": "Uses SNOMED code 77386006 before or at the T2DM index date.",
        },
        {
            "step_number": 7,
            "step_id": "final_analytical_cohort",
            "criterion": "Final Phase 1 analytical cohort",
            "input_n": after_pregnancy_exclusion_n,
            "excluded_n": after_pregnancy_exclusion_n - final_cohort_n,
            "remaining_n": final_cohort_n,
            "notes": "Patients meeting adult age, baseline history, and exclusion rules.",
        },
    ]

    attrition = pl.DataFrame(rows)

    parquet_output_path.parent.mkdir(parents=True, exist_ok=True)
    attrition.write_parquet(parquet_output_path)

    csv_written = None
    if csv_output_path is not None:
        csv_output_path = Path(csv_output_path)
        csv_output_path.parent.mkdir(parents=True, exist_ok=True)
        attrition.write_csv(csv_output_path)
        csv_written = str(csv_output_path)

    manifest = {
        "manifest_type": "t2dm_attrition_table",
        "generated_at_utc": utc_now_iso(),
        "silver_dir": str(silver_dir),
        "index_cohort_path": str(index_cohort_path),
        "parquet_output_path": str(parquet_output_path),
        "csv_output_path": csv_written,
        "summary": {
            "raw_patients_n": raw_patients_n,
            "t2dm_patients_n": t2dm_patients_n,
            "adult_t2dm_patients_n": adult_t2dm_patients_n,
            "baseline_eligible_patients_n": baseline_eligible_patients_n,
            "after_t1dm_exclusion_n": after_t1dm_exclusion_n,
            "after_pregnancy_exclusion_n": after_pregnancy_exclusion_n,
            "final_cohort_n": final_cohort_n,
            "excluded_no_t2dm_n": raw_patients_n - t2dm_patients_n,
            "excluded_underage_at_index_n": t2dm_patients_n - adult_t2dm_patients_n,
            "excluded_insufficient_baseline_history_n": adult_t2dm_patients_n - baseline_eligible_patients_n,
            "excluded_prior_t1dm_n": baseline_eligible_patients_n - after_t1dm_exclusion_n,
            "excluded_pregnancy_window_n": after_t1dm_exclusion_n - after_pregnancy_exclusion_n,
            "attrition_step_count": attrition.height,
        },
        "output_columns": attrition.columns,
    }

    return manifest


def build_t2dm_attrition_from_scenario(
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

    silver_dir = Path(scenario_config["paths"]["silver_dir"])
    if not silver_dir.is_absolute():
        silver_dir = repo_root / silver_dir

    gold_dir = Path(scenario_config["paths"]["gold_dir"])
    if not gold_dir.is_absolute():
        gold_dir = repo_root / gold_dir

    index_cohort_path = gold_dir / "t2dm_index_cohort.parquet"

    if parquet_output_path is None:
        parquet_output_path = gold_dir / "t2dm_attrition.parquet"
    else:
        parquet_output_path = Path(parquet_output_path)
        if not parquet_output_path.is_absolute():
            parquet_output_path = repo_root / parquet_output_path

    if csv_output_path is None:
        csv_output_path = repo_root / "reports" / "tables" / f"{scenario_config['scenario']['id']}_t2dm_attrition.csv"
    else:
        csv_output_path = Path(csv_output_path)
        if not csv_output_path.is_absolute():
            csv_output_path = repo_root / csv_output_path

    manifest = build_t2dm_attrition_table(
        silver_dir=silver_dir,
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
        manifest_path = repo_root / "reports" / "manifests" / f"t2dm_attrition_{timestamp}.json"
    else:
        manifest_path = Path(manifest_path)
        if not manifest_path.is_absolute():
            manifest_path = repo_root / manifest_path

    manifest["manifest_path"] = str(manifest_path)
    write_json(manifest, manifest_path)

    return manifest
