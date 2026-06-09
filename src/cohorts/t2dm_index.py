from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl
import yaml


T2DM_CODE = "44054006"


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


def calculate_age_expr(index_col: str, birth_col: str) -> pl.Expr:
    index_date = pl.col(index_col)
    birth_date = pl.col(birth_col)

    had_birthday_this_year = (
        (index_date.dt.month() > birth_date.dt.month())
        | (
            (index_date.dt.month() == birth_date.dt.month())
            & (index_date.dt.day() >= birth_date.dt.day())
        )
    )

    return (
        index_date.dt.year()
        - birth_date.dt.year()
        - (~had_birthday_this_year).cast(pl.Int32)
    )


def build_t2dm_index_cohort(
    silver_dir: Path,
    output_path: Path,
    t2dm_code: str = T2DM_CODE,
    minimum_age_years: int = 18,
) -> dict[str, Any]:
    silver_dir = Path(silver_dir)
    output_path = Path(output_path)

    patients_path = silver_dir / "patients.parquet"
    conditions_path = silver_dir / "conditions.parquet"

    if not patients_path.exists():
        raise FileNotFoundError(f"Missing silver patients table: {patients_path}")

    if not conditions_path.exists():
        raise FileNotFoundError(f"Missing silver conditions table: {conditions_path}")

    patients = pl.read_parquet(patients_path)
    conditions = pl.read_parquet(conditions_path)

    required_patient_columns = {"patient_id", "birth_date", "death_date", "sex", "race", "ethnicity"}
    required_condition_columns = {"patient_id", "encounter_id", "condition_start_date", "code", "description"}

    missing_patient_columns = required_patient_columns - set(patients.columns)
    missing_condition_columns = required_condition_columns - set(conditions.columns)

    if missing_patient_columns:
        raise ValueError(f"patients table missing columns: {sorted(missing_patient_columns)}")

    if missing_condition_columns:
        raise ValueError(f"conditions table missing columns: {sorted(missing_condition_columns)}")

    t2dm_conditions = conditions.filter(pl.col("code").cast(pl.Utf8) == str(t2dm_code))

    first_t2dm = (
        t2dm_conditions
        .sort(["patient_id", "condition_start_date"])
        .group_by("patient_id")
        .agg(
            pl.col("condition_start_date").first().alias("index_date"),
            pl.col("encounter_id").first().alias("index_encounter_id"),
            pl.col("code").first().alias("index_condition_code"),
            pl.col("description").first().alias("index_condition_description"),
            pl.len().alias("t2dm_condition_row_count"),
        )
    )

    cohort = (
        first_t2dm
        .join(
            patients.select(
                [
                    "patient_id",
                    "birth_date",
                    "death_date",
                    "sex",
                    "race",
                    "ethnicity",
                ]
            ),
            on="patient_id",
            how="left",
        )
        .with_columns(
            calculate_age_expr("index_date", "birth_date").alias("age_at_index")
        )
        .with_columns(
            (pl.col("age_at_index") >= minimum_age_years).alias("meets_adult_age_rule"),
            pl.lit(str(t2dm_code)).alias("target_t2dm_code"),
            pl.lit("t2dm_index_cohort").alias("cohort_name"),
        )
        .sort(["index_date", "patient_id"])
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cohort.write_parquet(output_path)

    adult_count = cohort.filter(pl.col("meets_adult_age_rule")).height

    manifest = {
        "manifest_type": "t2dm_index_cohort",
        "generated_at_utc": utc_now_iso(),
        "silver_dir": str(silver_dir),
        "output_path": str(output_path),
        "t2dm_code": str(t2dm_code),
        "minimum_age_years": minimum_age_years,
        "summary": {
            "patients_total": patients.height,
            "conditions_total": conditions.height,
            "t2dm_condition_rows": t2dm_conditions.height,
            "t2dm_patient_count": first_t2dm.height,
            "index_cohort_rows": cohort.height,
            "adult_index_cohort_rows": adult_count,
            "underage_index_cohort_rows": cohort.height - adult_count,
        },
        "output_columns": cohort.columns,
    }

    return manifest


def build_t2dm_index_from_scenario(
    repo_root: Path,
    scenario_path: Path | str = "configs/scenarios/default_synthea.yaml",
    output_path: Path | str | None = None,
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

    if output_path is None:
        output_path = gold_dir / "t2dm_index_cohort.parquet"
    else:
        output_path = Path(output_path)
        if not output_path.is_absolute():
            output_path = repo_root / output_path

    manifest = build_t2dm_index_cohort(
        silver_dir=silver_dir,
        output_path=output_path,
        t2dm_code="44054006",
        minimum_age_years=18,
    )

    manifest["scenario"] = {
        "id": scenario_config["scenario"]["id"],
        "name": scenario_config["scenario"]["name"],
        "scenario_path": str(scenario_path),
    }

    manifest["synthea"] = scenario_config["synthea"]

    if manifest_path is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        manifest_path = repo_root / "reports" / "manifests" / f"t2dm_index_cohort_{timestamp}.json"
    else:
        manifest_path = Path(manifest_path)
        if not manifest_path.is_absolute():
            manifest_path = repo_root / manifest_path

    manifest["manifest_path"] = str(manifest_path)
    write_json(manifest, manifest_path)

    return manifest