from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl
import yaml


T2DM_CODE = "44054006"
T1DM_CODE = "46635009"
PREGNANCY_CODE = "77386006"


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


def empty_flag_frame(flag_name: str, first_date_name: str, count_name: str) -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "patient_id": pl.Utf8,
            flag_name: pl.Boolean,
            first_date_name: pl.Date,
            count_name: pl.UInt32,
        }
    )


def build_condition_exclusion_flag(
    conditions: pl.DataFrame,
    first_t2dm: pl.DataFrame,
    code: str,
    flag_name: str,
    first_date_name: str,
    count_name: str,
    include_index_date: bool,
) -> pl.DataFrame:
    if first_t2dm.height == 0:
        return empty_flag_frame(flag_name, first_date_name, count_name)

    candidate_rows = (
        conditions
        .filter(
            (pl.col("code").cast(pl.Utf8) == str(code))
            & pl.col("condition_start_date").is_not_null()
        )
        .join(
            first_t2dm.select(["patient_id", "index_date"]),
            on="patient_id",
            how="inner",
        )
    )

    if candidate_rows.height == 0:
        return empty_flag_frame(flag_name, first_date_name, count_name)

    if include_index_date:
        candidate_rows = candidate_rows.with_columns(
            (pl.col("condition_start_date") <= pl.col("index_date")).alias("_exclusion_flag")
        )
    else:
        candidate_rows = candidate_rows.with_columns(
            (pl.col("condition_start_date") < pl.col("index_date")).alias("_exclusion_flag")
        )

    flagged = candidate_rows.filter(pl.col("_exclusion_flag") == True)

    if flagged.height == 0:
        return empty_flag_frame(flag_name, first_date_name, count_name)

    return (
        flagged
        .group_by("patient_id")
        .agg(
            pl.col("_exclusion_flag").any().alias(flag_name),
            pl.col("condition_start_date").min().alias(first_date_name),
            pl.len().alias(count_name),
        )
    )


def build_t2dm_index_cohort(
    silver_dir: Path,
    output_path: Path,
    t2dm_code: str = T2DM_CODE,
    t1dm_code: str = T1DM_CODE,
    pregnancy_code: str = PREGNANCY_CODE,
    minimum_age_years: int = 18,
    baseline_days_before_index: int = 365,
) -> dict[str, Any]:
    silver_dir = Path(silver_dir)
    output_path = Path(output_path)

    patients_path = silver_dir / "patients.parquet"
    conditions_path = silver_dir / "conditions.parquet"
    encounters_path = silver_dir / "encounters.parquet"

    if not patients_path.exists():
        raise FileNotFoundError(f"Missing silver patients table: {patients_path}")

    if not conditions_path.exists():
        raise FileNotFoundError(f"Missing silver conditions table: {conditions_path}")

    if not encounters_path.exists():
        raise FileNotFoundError(f"Missing silver encounters table: {encounters_path}")

    patients = pl.read_parquet(patients_path)
    conditions = pl.read_parquet(conditions_path)
    encounters = pl.read_parquet(encounters_path)

    required_patient_columns = {"patient_id", "birth_date", "death_date", "sex", "race", "ethnicity"}
    required_condition_columns = {"patient_id", "encounter_id", "condition_start_date", "code", "description"}
    required_encounter_columns = {"patient_id", "encounter_start_date"}

    missing_patient_columns = required_patient_columns - set(patients.columns)
    missing_condition_columns = required_condition_columns - set(conditions.columns)
    missing_encounter_columns = required_encounter_columns - set(encounters.columns)

    if missing_patient_columns:
        raise ValueError(f"patients table missing columns: {sorted(missing_patient_columns)}")

    if missing_condition_columns:
        raise ValueError(f"conditions table missing columns: {sorted(missing_condition_columns)}")

    if missing_encounter_columns:
        raise ValueError(f"encounters table missing columns: {sorted(missing_encounter_columns)}")

    t2dm_conditions = conditions.filter(
        (pl.col("code").cast(pl.Utf8) == str(t2dm_code))
        & pl.col("condition_start_date").is_not_null()
    )

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

    first_observed = (
        encounters
        .filter(pl.col("encounter_start_date").is_not_null())
        .group_by("patient_id")
        .agg(
            pl.col("encounter_start_date").min().alias("first_observed_date")
        )
    )

    prior_t1dm_flags = build_condition_exclusion_flag(
        conditions=conditions,
        first_t2dm=first_t2dm,
        code=t1dm_code,
        flag_name="has_prior_t1dm_before_index",
        first_date_name="first_prior_t1dm_date",
        count_name="prior_t1dm_row_count",
        include_index_date=False,
    )

    pregnancy_flags = build_condition_exclusion_flag(
        conditions=conditions,
        first_t2dm=first_t2dm,
        code=pregnancy_code,
        flag_name="has_pregnancy_window_before_or_at_index",
        first_date_name="first_pregnancy_window_date",
        count_name="pregnancy_window_row_count",
        include_index_date=True,
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
        .join(first_observed, on="patient_id", how="left")
        .join(prior_t1dm_flags, on="patient_id", how="left")
        .join(pregnancy_flags, on="patient_id", how="left")
        .with_columns(
            calculate_age_expr("index_date", "birth_date").alias("age_at_index"),
            (
                pl.col("index_date") - pl.col("first_observed_date")
            ).dt.total_days().cast(pl.Int64).alias("observable_history_days_before_index"),
        )
        .with_columns(
            pl.col("has_prior_t1dm_before_index").fill_null(False),
            pl.col("has_pregnancy_window_before_or_at_index").fill_null(False),
            pl.col("prior_t1dm_row_count").fill_null(0),
            pl.col("pregnancy_window_row_count").fill_null(0),
        )
        .with_columns(
            (pl.col("age_at_index") >= minimum_age_years)
            .fill_null(False)
            .alias("meets_adult_age_rule"),

            (pl.col("observable_history_days_before_index") >= baseline_days_before_index)
            .fill_null(False)
            .alias("meets_baseline_history_rule"),

            (~pl.col("has_prior_t1dm_before_index")).alias("meets_prior_t1dm_exclusion_rule"),
            (~pl.col("has_pregnancy_window_before_or_at_index")).alias("meets_pregnancy_exclusion_rule"),

            pl.lit(str(t2dm_code)).alias("target_t2dm_code"),
            pl.lit(str(t1dm_code)).alias("prior_t1dm_exclusion_code"),
            pl.lit(str(pregnancy_code)).alias("pregnancy_exclusion_code"),
            pl.lit(baseline_days_before_index).alias("baseline_days_required"),
            pl.lit("t2dm_index_cohort").alias("cohort_name"),
        )
        .with_columns(
            (
                (pl.col("meets_adult_age_rule") == True)
                & (pl.col("meets_baseline_history_rule") == True)
                & (pl.col("meets_prior_t1dm_exclusion_rule") == True)
                & (pl.col("meets_pregnancy_exclusion_rule") == True)
            ).alias("meets_final_phase1_rules")
        )
        .sort(["index_date", "patient_id"])
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cohort.write_parquet(output_path)

    adult_count = cohort.filter(pl.col("meets_adult_age_rule")).height

    baseline_eligible_count = (
        cohort
        .filter(
            (pl.col("meets_adult_age_rule") == True)
            & (pl.col("meets_baseline_history_rule") == True)
        )
        .height
    )

    after_t1dm_exclusion_count = (
        cohort
        .filter(
            (pl.col("meets_adult_age_rule") == True)
            & (pl.col("meets_baseline_history_rule") == True)
            & (pl.col("meets_prior_t1dm_exclusion_rule") == True)
        )
        .height
    )

    final_count = cohort.filter(pl.col("meets_final_phase1_rules") == True).height

    manifest = {
        "manifest_type": "t2dm_index_cohort",
        "generated_at_utc": utc_now_iso(),
        "silver_dir": str(silver_dir),
        "output_path": str(output_path),
        "t2dm_code": str(t2dm_code),
        "minimum_age_years": minimum_age_years,
        "baseline_days_before_index": baseline_days_before_index,
        "summary": {
            "patients_total": patients.height,
            "conditions_total": conditions.height,
            "encounters_total": encounters.height,
            "t2dm_condition_rows": t2dm_conditions.height,
            "t2dm_patient_count": first_t2dm.height,
            "index_cohort_rows": cohort.height,
            "adult_index_cohort_rows": adult_count,
            "underage_index_cohort_rows": cohort.height - adult_count,
            "baseline_history_eligible_rows": baseline_eligible_count,
            "baseline_history_ineligible_rows": adult_count - baseline_eligible_count,
            "prior_t1dm_excluded_rows": baseline_eligible_count - after_t1dm_exclusion_count,
            "pregnancy_window_excluded_rows": after_t1dm_exclusion_count - final_count,
            "final_phase1_eligible_rows": final_count,
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

    study_config_path = repo_root / "configs" / "study" / "t2dm_phase1.yaml"
    study_config = load_yaml(study_config_path)

    baseline_days_before_index = int(
        study_config["windows"]["baseline"]["days_before_index"]
    )

    minimum_age_years = int(
        study_config["population"]["minimum_age_years"]
    )

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
        t2dm_code=T2DM_CODE,
        t1dm_code=T1DM_CODE,
        pregnancy_code=PREGNANCY_CODE,
        minimum_age_years=minimum_age_years,
        baseline_days_before_index=baseline_days_before_index,
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
