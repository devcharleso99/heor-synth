from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl
import yaml


CORE_REQUIRED_COLUMNS: dict[str, list[str]] = {
    "patients": ["Id", "BIRTHDATE"],
    "encounters": ["Id", "START", "PATIENT", "CODE", "DESCRIPTION"],
    "conditions": ["START", "PATIENT", "CODE", "DESCRIPTION"],
    "medications": ["START", "PATIENT", "CODE", "DESCRIPTION"],
    "observations": ["DATE", "PATIENT", "CODE", "DESCRIPTION"],
    "procedures": ["START", "PATIENT", "CODE", "DESCRIPTION"],
    "careplans": ["Id", "START", "PATIENT", "CODE", "DESCRIPTION"],
}


TABLES_WITH_PATIENT_COLUMN = [
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


def read_parquet_if_exists(path: Path) -> pl.DataFrame | None:
    if not path.exists():
        return None

    return pl.read_parquet(path)


def inspect_bronze_table(bronze_dir: Path, table_name: str, required: bool) -> dict[str, Any]:
    path = bronze_dir / f"{table_name}.parquet"

    record: dict[str, Any] = {
        "table_name": table_name,
        "required": required,
        "exists": path.exists(),
        "path": str(path),
        "row_count": None,
        "column_count": None,
        "columns": [],
        "missing_core_columns": [],
        "status": "missing",
    }

    if not path.exists():
        return record

    df = pl.read_parquet(path)
    columns = df.columns

    required_columns = CORE_REQUIRED_COLUMNS.get(table_name, [])
    missing_core_columns = [
        column for column in required_columns if column not in columns
    ]

    record.update(
        {
            "row_count": df.height,
            "column_count": len(columns),
            "columns": columns,
            "missing_core_columns": missing_core_columns,
            "status": "present",
        }
    )

    if df.height == 0:
        record["status"] = "empty"

    if missing_core_columns:
        record["status"] = "missing_core_columns"

    return record


def count_unknown_patient_references(bronze_dir: Path, table_name: str) -> dict[str, Any]:
    patients_path = bronze_dir / "patients.parquet"
    table_path = bronze_dir / f"{table_name}.parquet"

    result: dict[str, Any] = {
        "table_name": table_name,
        "checked": False,
        "unknown_patient_reference_count": None,
        "unique_patient_reference_count": None,
        "status": "not_checked",
    }

    if not patients_path.exists() or not table_path.exists():
        result["status"] = "missing_input"
        return result

    patients = pl.read_parquet(patients_path)
    table = pl.read_parquet(table_path)

    if "Id" not in patients.columns or "PATIENT" not in table.columns:
        result["status"] = "missing_patient_key_column"
        return result

    patient_ids = patients.select(pl.col("Id").cast(pl.Utf8)).unique()

    patient_refs = (
        table
        .select(pl.col("PATIENT").cast(pl.Utf8))
        .drop_nulls()
        .unique()
    )

    unknown_refs = patient_refs.join(
        patient_ids,
        left_on="PATIENT",
        right_on="Id",
        how="anti",
    )

    result.update(
        {
            "checked": True,
            "unknown_patient_reference_count": unknown_refs.height,
            "unique_patient_reference_count": patient_refs.height,
            "status": "passed" if unknown_refs.height == 0 else "failed",
        }
    )

    return result


def count_t2dm_signal(bronze_dir: Path, t2dm_code: str = "44054006") -> dict[str, Any]:
    conditions_path = bronze_dir / "conditions.parquet"

    result: dict[str, Any] = {
        "checked": False,
        "t2dm_code": t2dm_code,
        "t2dm_condition_row_count": None,
        "t2dm_patient_count": None,
        "status": "not_checked",
    }

    if not conditions_path.exists():
        result["status"] = "conditions_missing"
        return result

    conditions = pl.read_parquet(conditions_path)

    if "CODE" not in conditions.columns:
        result["status"] = "conditions_missing_code_column"
        return result

    if "PATIENT" not in conditions.columns:
        result["status"] = "conditions_missing_patient_column"
        return result

    t2dm_rows = conditions.filter(pl.col("CODE").cast(pl.Utf8) == t2dm_code)

    result.update(
        {
            "checked": True,
            "t2dm_condition_row_count": t2dm_rows.height,
            "t2dm_patient_count": t2dm_rows.select(pl.col("PATIENT")).unique().height,
            "status": "present" if t2dm_rows.height > 0 else "absent_in_smoke_data",
        }
    )

    return result


def run_bronze_qa(
    bronze_dir: Path,
    required_tables: list[str],
    optional_tables: list[str] | None = None,
) -> dict[str, Any]:
    bronze_dir = Path(bronze_dir)
    optional_tables = optional_tables or []

    required_table_names = [Path(name).stem for name in required_tables]
    optional_table_names = [Path(name).stem for name in optional_tables]

    table_records = []

    for table_name in required_table_names:
        table_records.append(
            inspect_bronze_table(
                bronze_dir=bronze_dir,
                table_name=table_name,
                required=True,
            )
        )

    for table_name in optional_table_names:
        parquet_path = bronze_dir / f"{table_name}.parquet"
        if parquet_path.exists():
            table_records.append(
                inspect_bronze_table(
                    bronze_dir=bronze_dir,
                    table_name=table_name,
                    required=False,
                )
            )

    missing_required_tables = [
        record["table_name"]
        for record in table_records
        if record["required"] and not record["exists"]
    ]

    empty_required_tables = [
        record["table_name"]
        for record in table_records
        if record["required"] and record["status"] == "empty"
    ]

    required_tables_missing_core_columns = [
        {
            "table_name": record["table_name"],
            "missing_core_columns": record["missing_core_columns"],
        }
        for record in table_records
        if record["required"] and record["missing_core_columns"]
    ]

    patient_reference_checks = [
        count_unknown_patient_references(bronze_dir, table_name)
        for table_name in TABLES_WITH_PATIENT_COLUMN
        if table_name in required_table_names or (bronze_dir / f"{table_name}.parquet").exists()
    ]

    failed_patient_reference_tables = [
        item["table_name"]
        for item in patient_reference_checks
        if item["status"] == "failed"
    ]

    t2dm_signal = count_t2dm_signal(bronze_dir)

    structural_checks_passed = (
        len(missing_required_tables) == 0
        and len(empty_required_tables) == 0
        and len(required_tables_missing_core_columns) == 0
        and len(failed_patient_reference_tables) == 0
    )

    manifest = {
        "manifest_type": "bronze_qa",
        "generated_at_utc": utc_now_iso(),
        "bronze_dir": str(bronze_dir),
        "summary": {
            "required_table_count": len(required_table_names),
            "optional_table_count_present": len(
                [record for record in table_records if not record["required"]]
            ),
            "missing_required_table_count": len(missing_required_tables),
            "empty_required_table_count": len(empty_required_tables),
            "required_tables_missing_core_columns_count": len(required_tables_missing_core_columns),
            "failed_patient_reference_table_count": len(failed_patient_reference_tables),
            "structural_checks_passed": structural_checks_passed,
        },
        "missing_required_tables": missing_required_tables,
        "empty_required_tables": empty_required_tables,
        "required_tables_missing_core_columns": required_tables_missing_core_columns,
        "patient_reference_checks": patient_reference_checks,
        "failed_patient_reference_tables": failed_patient_reference_tables,
        "t2dm_signal": t2dm_signal,
        "tables": table_records,
    }

    return manifest


def build_bronze_qa_from_scenario(
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

    required_tables = scenario_config["expected_raw_tables"]["required"]
    optional_tables = scenario_config["expected_raw_tables"].get("optional", [])

    manifest = run_bronze_qa(
        bronze_dir=bronze_dir,
        required_tables=required_tables,
        optional_tables=optional_tables,
    )

    manifest["scenario"] = {
        "id": scenario_config["scenario"]["id"],
        "name": scenario_config["scenario"]["name"],
        "scenario_path": str(scenario_path),
    }

    manifest["synthea"] = scenario_config["synthea"]

    if output_path is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = repo_root / "reports" / "manifests" / f"bronze_qa_{timestamp}.json"
    else:
        output_path = Path(output_path)
        if not output_path.is_absolute():
            output_path = repo_root / output_path

    manifest["manifest_path"] = str(output_path)
    write_json(manifest, output_path)

    return manifest