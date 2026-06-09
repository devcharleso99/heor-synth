from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl
import yaml


SILVER_REQUIRED_COLUMNS: dict[str, list[str]] = {
    "patients": [
        "patient_id",
        "birth_date",
        "death_date",
        "sex",
        "race",
        "ethnicity",
        "birth_year",
        "source_table",
    ],
    "encounters": [
        "encounter_id",
        "patient_id",
        "encounter_start_date",
        "encounter_stop_date",
        "code",
        "description",
        "source_table",
    ],
    "conditions": [
        "patient_id",
        "encounter_id",
        "condition_start_date",
        "condition_stop_date",
        "code",
        "description",
        "source_table",
    ],
    "medications": [
        "patient_id",
        "encounter_id",
        "medication_start_date",
        "medication_stop_date",
        "code",
        "description",
        "source_table",
    ],
    "observations": [
        "patient_id",
        "encounter_id",
        "observation_date",
        "code",
        "description",
        "value_raw",
        "value_numeric",
        "units",
        "source_table",
    ],
    "procedures": [
        "patient_id",
        "encounter_id",
        "procedure_start_date",
        "procedure_stop_date",
        "code",
        "description",
        "source_table",
    ],
    "careplans": [
        "careplan_id",
        "patient_id",
        "encounter_id",
        "careplan_start_date",
        "careplan_stop_date",
        "code",
        "description",
        "source_table",
    ],
}


SILVER_DATE_COLUMNS: dict[str, list[str]] = {
    "patients": ["birth_date", "death_date"],
    "encounters": ["encounter_start_date", "encounter_stop_date"],
    "conditions": ["condition_start_date", "condition_stop_date"],
    "medications": ["medication_start_date", "medication_stop_date"],
    "observations": ["observation_date"],
    "procedures": ["procedure_start_date", "procedure_stop_date"],
    "careplans": ["careplan_start_date", "careplan_stop_date"],
}


REQUIRED_NON_NULL_DATE_COLUMNS: dict[str, list[str]] = {
    "patients": ["birth_date"],
    "encounters": ["encounter_start_date"],
    "conditions": ["condition_start_date"],
    "medications": ["medication_start_date"],
    "observations": ["observation_date"],
    "procedures": ["procedure_start_date"],
    "careplans": ["careplan_start_date"],
}


TABLES_WITH_PATIENT_ID = [
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
        json.dump(data, f, indent=2, default=str)

    return output_path


def count_nulls(df: pl.DataFrame, column: str) -> int:
    return int(df.select(pl.col(column).is_null().sum()).item())


def inspect_silver_table(silver_dir: Path, table_name: str, required: bool = True) -> dict[str, Any]:
    path = silver_dir / f"{table_name}.parquet"

    record: dict[str, Any] = {
        "table_name": table_name,
        "required": required,
        "exists": path.exists(),
        "path": str(path),
        "row_count": None,
        "column_count": None,
        "columns": [],
        "missing_required_columns": [],
        "date_column_checks": [],
        "required_date_null_counts": [],
        "status": "missing",
    }

    if not path.exists():
        return record

    df = pl.read_parquet(path)

    required_columns = SILVER_REQUIRED_COLUMNS.get(table_name, [])
    missing_required_columns = [
        column for column in required_columns if column not in df.columns
    ]

    date_checks = []
    invalid_date_columns = []

    for column in SILVER_DATE_COLUMNS.get(table_name, []):
        if column not in df.columns:
            continue

        dtype = df.schema[column]
        null_count = count_nulls(df, column)

        # Polars will represent an all-null column as dtype Null.
        # For optional/stop date columns it's acceptable to be entirely null.
        is_date = dtype == pl.Date or (dtype == pl.Null and null_count == df.height)

        date_checks.append(
            {
                "column": column,
                "dtype": str(dtype),
                "is_date": is_date,
                "null_count": null_count,
            }
        )

        if not is_date:
            invalid_date_columns.append(column)

    required_date_null_counts = []

    for column in REQUIRED_NON_NULL_DATE_COLUMNS.get(table_name, []):
        if column not in df.columns:
            continue

        null_count = count_nulls(df, column)

        required_date_null_counts.append(
            {
                "column": column,
                "null_count": null_count,
            }
        )

    record.update(
        {
            "row_count": df.height,
            "column_count": len(df.columns),
            "columns": df.columns,
            "missing_required_columns": missing_required_columns,
            "date_column_checks": date_checks,
            "required_date_null_counts": required_date_null_counts,
            "status": "present",
        }
    )

    if df.height == 0:
        record["status"] = "empty"

    if missing_required_columns:
        record["status"] = "missing_required_columns"

    if invalid_date_columns:
        record["status"] = "invalid_date_dtypes"
        record["invalid_date_columns"] = invalid_date_columns

    return record


def count_unknown_patient_references(silver_dir: Path, table_name: str) -> dict[str, Any]:
    patients_path = silver_dir / "patients.parquet"
    table_path = silver_dir / f"{table_name}.parquet"

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

    if "patient_id" not in patients.columns or "patient_id" not in table.columns:
        result["status"] = "missing_patient_id_column"
        return result

    patient_ids = patients.select(pl.col("patient_id").cast(pl.Utf8)).drop_nulls().unique()
    patient_refs = table.select(pl.col("patient_id").cast(pl.Utf8)).drop_nulls().unique()

    unknown_refs = patient_refs.join(
        patient_ids,
        on="patient_id",
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


def count_t2dm_signal(silver_dir: Path, t2dm_code: str = "44054006") -> dict[str, Any]:
    conditions_path = silver_dir / "conditions.parquet"

    result: dict[str, Any] = {
        "checked": False,
        "t2dm_code": t2dm_code,
        "t2dm_condition_row_count": None,
        "t2dm_patient_count": None,
        "earliest_t2dm_date": None,
        "latest_t2dm_date": None,
        "status": "not_checked",
    }

    if not conditions_path.exists():
        result["status"] = "conditions_missing"
        return result

    conditions = pl.read_parquet(conditions_path)

    required_columns = {"patient_id", "code", "condition_start_date"}
    missing = required_columns - set(conditions.columns)

    if missing:
        result["status"] = f"conditions_missing_columns: {sorted(missing)}"
        return result

    t2dm_rows = conditions.filter(pl.col("code").cast(pl.Utf8) == t2dm_code)

    if t2dm_rows.height == 0:
        result.update(
            {
                "checked": True,
                "t2dm_condition_row_count": 0,
                "t2dm_patient_count": 0,
                "status": "absent",
            }
        )
        return result

    date_summary = t2dm_rows.select(
        pl.col("condition_start_date").min().alias("earliest"),
        pl.col("condition_start_date").max().alias("latest"),
    )

    result.update(
        {
            "checked": True,
            "t2dm_condition_row_count": t2dm_rows.height,
            "t2dm_patient_count": t2dm_rows.select("patient_id").unique().height,
            "earliest_t2dm_date": str(date_summary["earliest"][0]),
            "latest_t2dm_date": str(date_summary["latest"][0]),
            "status": "present",
        }
    )

    return result


def run_silver_qa(
    silver_dir: Path,
    required_tables: list[str],
) -> dict[str, Any]:
    silver_dir = Path(silver_dir)

    required_table_names = [Path(name).stem for name in required_tables]

    table_records = [
        inspect_silver_table(silver_dir=silver_dir, table_name=table_name, required=True)
        for table_name in required_table_names
    ]

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

    tables_missing_required_columns = [
        {
            "table_name": record["table_name"],
            "missing_required_columns": record["missing_required_columns"],
        }
        for record in table_records
        if record["missing_required_columns"]
    ]

    tables_with_invalid_date_dtypes = [
        {
            "table_name": record["table_name"],
            "date_column_checks": record["date_column_checks"],
        }
        for record in table_records
        if record["status"] == "invalid_date_dtypes"
    ]

    patient_reference_checks = [
        count_unknown_patient_references(silver_dir, table_name)
        for table_name in TABLES_WITH_PATIENT_ID
        if table_name in required_table_names
    ]

    failed_patient_reference_tables = [
        item["table_name"]
        for item in patient_reference_checks
        if item["status"] == "failed"
    ]

    t2dm_signal = count_t2dm_signal(silver_dir)

    structural_checks_passed = (
        len(missing_required_tables) == 0
        and len(empty_required_tables) == 0
        and len(tables_missing_required_columns) == 0
        and len(tables_with_invalid_date_dtypes) == 0
        and len(failed_patient_reference_tables) == 0
    )

    return {
        "manifest_type": "silver_qa",
        "generated_at_utc": utc_now_iso(),
        "silver_dir": str(silver_dir),
        "summary": {
            "required_table_count": len(required_table_names),
            "missing_required_table_count": len(missing_required_tables),
            "empty_required_table_count": len(empty_required_tables),
            "tables_missing_required_columns_count": len(tables_missing_required_columns),
            "tables_with_invalid_date_dtypes_count": len(tables_with_invalid_date_dtypes),
            "failed_patient_reference_table_count": len(failed_patient_reference_tables),
            "structural_checks_passed": structural_checks_passed,
        },
        "missing_required_tables": missing_required_tables,
        "empty_required_tables": empty_required_tables,
        "tables_missing_required_columns": tables_missing_required_columns,
        "tables_with_invalid_date_dtypes": tables_with_invalid_date_dtypes,
        "patient_reference_checks": patient_reference_checks,
        "failed_patient_reference_tables": failed_patient_reference_tables,
        "t2dm_signal": t2dm_signal,
        "tables": table_records,
    }


def build_silver_qa_from_scenario(
    repo_root: Path,
    scenario_path: Path | str = "configs/scenarios/default_synthea.yaml",
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root)

    scenario_path = Path(scenario_path)
    if not scenario_path.is_absolute():
        scenario_path = repo_root / scenario_path

    scenario_config = load_yaml(scenario_path)

    silver_dir = Path(scenario_config["paths"]["silver_dir"])
    if not silver_dir.is_absolute():
        silver_dir = repo_root / silver_dir

    required_tables = scenario_config["expected_raw_tables"]["required"]

    manifest = run_silver_qa(
        silver_dir=silver_dir,
        required_tables=required_tables,
    )

    manifest["scenario"] = {
        "id": scenario_config["scenario"]["id"],
        "name": scenario_config["scenario"]["name"],
        "scenario_path": str(scenario_path),
    }

    manifest["synthea"] = scenario_config["synthea"]

    if output_path is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = repo_root / "reports" / "manifests" / f"silver_qa_{timestamp}.json"
    else:
        output_path = Path(output_path)
        if not output_path.is_absolute():
            output_path = repo_root / output_path

    manifest["manifest_path"] = str(output_path)
    write_json(manifest, output_path)

    return manifest