from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import polars as pl
import yaml


CORE_TABLES = [
    "patients",
    "encounters",
    "conditions",
    "medications",
    "observations",
    "procedures",
    "careplans",
]


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
        json.dump(data, f, indent=2)

    return output_path


def ensure_columns(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    for column in columns:
        if column not in df.columns:
            df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias(column))

    return df


def str_col(source: str, alias: str) -> pl.Expr:
    return pl.col(source).cast(pl.Utf8).alias(alias)


def date_col(source: str, alias: str) -> pl.Expr:
    return (
        pl.col(source)
        .cast(pl.Utf8)
        .str.slice(0, 10)
        .str.strptime(pl.Date, format="%Y-%m-%d", strict=False)
        .alias(alias)
    )


def float_col(source: str, alias: str) -> pl.Expr:
    return pl.col(source).cast(pl.Float64, strict=False).alias(alias)


def normalize_patients(df: pl.DataFrame) -> pl.DataFrame:
    needed = [
        "Id",
        "BIRTHDATE",
        "DEATHDATE",
        "GENDER",
        "RACE",
        "ETHNICITY",
        "MARITAL",
        "CITY",
        "STATE",
        "COUNTY",
        "ZIP",
    ]
    df = ensure_columns(df, needed)

    return (
        df.select(
            str_col("Id", "patient_id"),
            date_col("BIRTHDATE", "birth_date"),
            date_col("DEATHDATE", "death_date"),
            str_col("GENDER", "sex"),
            str_col("RACE", "race"),
            str_col("ETHNICITY", "ethnicity"),
            str_col("MARITAL", "marital_status"),
            str_col("CITY", "city"),
            str_col("STATE", "state"),
            str_col("COUNTY", "county"),
            str_col("ZIP", "zip"),
        )
        .with_columns(
            pl.col("birth_date").dt.year().alias("birth_year"),
            pl.lit("patients").alias("source_table"),
        )
    )


def normalize_encounters(df: pl.DataFrame) -> pl.DataFrame:
    needed = [
        "Id",
        "START",
        "STOP",
        "PATIENT",
        "ORGANIZATION",
        "PROVIDER",
        "PAYER",
        "ENCOUNTERCLASS",
        "CODE",
        "DESCRIPTION",
        "REASONCODE",
        "REASONDESCRIPTION",
        "BASE_ENCOUNTER_COST",
        "TOTAL_CLAIM_COST",
        "PAYER_COVERAGE",
    ]
    df = ensure_columns(df, needed)

    return df.select(
        str_col("Id", "encounter_id"),
        str_col("PATIENT", "patient_id"),
        date_col("START", "encounter_start_date"),
        date_col("STOP", "encounter_stop_date"),
        str_col("ORGANIZATION", "organization_id"),
        str_col("PROVIDER", "provider_id"),
        str_col("PAYER", "payer_id"),
        str_col("ENCOUNTERCLASS", "encounter_class"),
        str_col("CODE", "code"),
        str_col("DESCRIPTION", "description"),
        str_col("REASONCODE", "reason_code"),
        str_col("REASONDESCRIPTION", "reason_description"),
        float_col("BASE_ENCOUNTER_COST", "base_encounter_cost"),
        float_col("TOTAL_CLAIM_COST", "total_claim_cost"),
        float_col("PAYER_COVERAGE", "payer_coverage"),
        pl.lit("encounters").alias("source_table"),
    )


def normalize_conditions(df: pl.DataFrame) -> pl.DataFrame:
    needed = [
        "START",
        "STOP",
        "PATIENT",
        "ENCOUNTER",
        "CODE",
        "DESCRIPTION",
    ]
    df = ensure_columns(df, needed)

    return df.select(
        str_col("PATIENT", "patient_id"),
        str_col("ENCOUNTER", "encounter_id"),
        date_col("START", "condition_start_date"),
        date_col("STOP", "condition_stop_date"),
        str_col("CODE", "code"),
        str_col("DESCRIPTION", "description"),
        pl.lit("conditions").alias("source_table"),
    )


def normalize_medications(df: pl.DataFrame) -> pl.DataFrame:
    needed = [
        "START",
        "STOP",
        "PATIENT",
        "PAYER",
        "ENCOUNTER",
        "CODE",
        "DESCRIPTION",
        "BASE_COST",
        "PAYER_COVERAGE",
        "DISPENSES",
        "TOTALCOST",
        "REASONCODE",
        "REASONDESCRIPTION",
    ]
    df = ensure_columns(df, needed)

    return df.select(
        str_col("PATIENT", "patient_id"),
        str_col("ENCOUNTER", "encounter_id"),
        str_col("PAYER", "payer_id"),
        date_col("START", "medication_start_date"),
        date_col("STOP", "medication_stop_date"),
        str_col("CODE", "code"),
        str_col("DESCRIPTION", "description"),
        float_col("BASE_COST", "base_cost"),
        float_col("PAYER_COVERAGE", "payer_coverage"),
        float_col("DISPENSES", "dispenses"),
        float_col("TOTALCOST", "total_cost"),
        str_col("REASONCODE", "reason_code"),
        str_col("REASONDESCRIPTION", "reason_description"),
        pl.lit("medications").alias("source_table"),
    )


def normalize_observations(df: pl.DataFrame) -> pl.DataFrame:
    needed = [
        "DATE",
        "PATIENT",
        "ENCOUNTER",
        "CATEGORY",
        "CODE",
        "DESCRIPTION",
        "VALUE",
        "UNITS",
        "TYPE",
    ]
    df = ensure_columns(df, needed)

    return df.select(
        str_col("PATIENT", "patient_id"),
        str_col("ENCOUNTER", "encounter_id"),
        date_col("DATE", "observation_date"),
        str_col("CATEGORY", "category"),
        str_col("CODE", "code"),
        str_col("DESCRIPTION", "description"),
        str_col("VALUE", "value_raw"),
        float_col("VALUE", "value_numeric"),
        str_col("UNITS", "units"),
        str_col("TYPE", "value_type"),
        pl.lit("observations").alias("source_table"),
    )


def normalize_procedures(df: pl.DataFrame) -> pl.DataFrame:
    needed = [
        "START",
        "STOP",
        "PATIENT",
        "ENCOUNTER",
        "CODE",
        "DESCRIPTION",
        "BASE_COST",
        "REASONCODE",
        "REASONDESCRIPTION",
    ]
    df = ensure_columns(df, needed)

    return df.select(
        str_col("PATIENT", "patient_id"),
        str_col("ENCOUNTER", "encounter_id"),
        date_col("START", "procedure_start_date"),
        date_col("STOP", "procedure_stop_date"),
        str_col("CODE", "code"),
        str_col("DESCRIPTION", "description"),
        float_col("BASE_COST", "base_cost"),
        str_col("REASONCODE", "reason_code"),
        str_col("REASONDESCRIPTION", "reason_description"),
        pl.lit("procedures").alias("source_table"),
    )


def normalize_careplans(df: pl.DataFrame) -> pl.DataFrame:
    needed = [
        "Id",
        "START",
        "STOP",
        "PATIENT",
        "ENCOUNTER",
        "CODE",
        "DESCRIPTION",
        "REASONCODE",
        "REASONDESCRIPTION",
    ]
    df = ensure_columns(df, needed)

    return df.select(
        str_col("Id", "careplan_id"),
        str_col("PATIENT", "patient_id"),
        str_col("ENCOUNTER", "encounter_id"),
        date_col("START", "careplan_start_date"),
        date_col("STOP", "careplan_stop_date"),
        str_col("CODE", "code"),
        str_col("DESCRIPTION", "description"),
        str_col("REASONCODE", "reason_code"),
        str_col("REASONDESCRIPTION", "reason_description"),
        pl.lit("careplans").alias("source_table"),
    )


NORMALIZERS: dict[str, Callable[[pl.DataFrame], pl.DataFrame]] = {
    "patients": normalize_patients,
    "encounters": normalize_encounters,
    "conditions": normalize_conditions,
    "medications": normalize_medications,
    "observations": normalize_observations,
    "procedures": normalize_procedures,
    "careplans": normalize_careplans,
}


def normalize_table(bronze_dir: Path, silver_dir: Path, table_name: str) -> dict[str, Any]:
    bronze_path = bronze_dir / f"{table_name}.parquet"
    silver_path = silver_dir / f"{table_name}.parquet"

    if table_name not in NORMALIZERS:
        raise ValueError(f"No normalizer registered for table: {table_name}")

    if not bronze_path.exists():
        raise FileNotFoundError(f"Missing bronze table: {bronze_path}")

    df = pl.read_parquet(bronze_path)
    normalized = NORMALIZERS[table_name](df)

    silver_dir.mkdir(parents=True, exist_ok=True)
    normalized.write_parquet(silver_path)

    return {
        "table_name": table_name,
        "bronze_path": str(bronze_path),
        "silver_path": str(silver_path),
        "input_row_count": df.height,
        "output_row_count": normalized.height,
        "input_column_count": len(df.columns),
        "output_column_count": len(normalized.columns),
        "output_columns": normalized.columns,
        "status": "normalized",
    }


def normalize_bronze_to_silver(
    bronze_dir: Path,
    silver_dir: Path,
    tables: list[str] | None = None,
) -> dict[str, Any]:
    bronze_dir = Path(bronze_dir)
    silver_dir = Path(silver_dir)

    tables = tables or CORE_TABLES

    normalized = []
    failed = []
    missing = []

    for table_name in tables:
        bronze_path = bronze_dir / f"{table_name}.parquet"

        if not bronze_path.exists():
            missing.append(table_name)
            continue

        try:
            normalized.append(normalize_table(bronze_dir, silver_dir, table_name))
        except Exception as exc:
            failed.append(
                {
                    "table_name": table_name,
                    "status": "failed",
                    "error": str(exc),
                }
            )

    return {
        "manifest_type": "silver_normalization",
        "generated_at_utc": utc_now_iso(),
        "bronze_dir": str(bronze_dir),
        "silver_dir": str(silver_dir),
        "summary": {
            "requested_table_count": len(tables),
            "normalized_count": len(normalized),
            "missing_count": len(missing),
            "failed_count": len(failed),
            "all_normalized": len(missing) == 0 and len(failed) == 0,
        },
        "normalized": normalized,
        "missing": missing,
        "failed": failed,
    }


def build_silver_from_scenario(
    repo_root: Path,
    scenario_path: Path | str = "configs/scenarios/default_synthea.yaml",
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root)

    scenario_path = Path(scenario_path)
    if not scenario_path.is_absolute():
        scenario_path = repo_root / scenario_path

    scenario_config = load_yaml(scenario_path)

    bronze_dir = Path(scenario_config["paths"]["bronze_dir"])
    if not bronze_dir.is_absolute():
        bronze_dir = repo_root / bronze_dir

    silver_dir = Path(scenario_config["paths"]["silver_dir"])
    if not silver_dir.is_absolute():
        silver_dir = repo_root / silver_dir

    manifest = normalize_bronze_to_silver(
        bronze_dir=bronze_dir,
        silver_dir=silver_dir,
        tables=CORE_TABLES,
    )

    manifest["scenario"] = {
        "id": scenario_config["scenario"]["id"],
        "name": scenario_config["scenario"]["name"],
        "scenario_path": str(scenario_path),
    }

    manifest["synthea"] = scenario_config["synthea"]

    if output_path is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = repo_root / "reports" / "manifests" / f"silver_normalization_{timestamp}.json"
    else:
        output_path = Path(output_path)
        if not output_path.is_absolute():
            output_path = repo_root / output_path

    manifest["manifest_path"] = str(output_path)
    write_json(manifest, output_path)

    return manifest