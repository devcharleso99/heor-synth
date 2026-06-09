from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl
import yaml


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def read_csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        try:
            return next(reader)
        except StopIteration:
            return []


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


def convert_csv_to_bronze_parquet(csv_path: Path, bronze_dir: Path) -> dict[str, Any]:
    csv_path = Path(csv_path)
    bronze_dir = Path(bronze_dir)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    if csv_path.suffix.lower() != ".csv":
        raise ValueError(f"Expected CSV file, got: {csv_path}")

    bronze_dir.mkdir(parents=True, exist_ok=True)
    output_path = bronze_dir / f"{csv_path.stem}.parquet"

    header = read_csv_header(csv_path)

    # Bronze layer rule:
    # Preserve raw values safely. Do not force clinical/date typing here.
    # Silver layer will handle date parsing, normalization, and table-specific schemas.
    df = pl.read_csv(
        csv_path,
        infer_schema_length=0,
        try_parse_dates=False,
        ignore_errors=False,
        null_values=[""],
    )

    df.write_parquet(output_path)

    return {
        "source_csv": str(csv_path),
        "bronze_parquet": str(output_path),
        "table_name": csv_path.stem,
        "source_size_bytes": csv_path.stat().st_size,
        "source_sha256": sha256_file(csv_path),
        "columns": header,
        "column_count": len(header),
        "row_count": df.height,
        "status": "converted",
    }


def convert_raw_dir_to_bronze(
    raw_dir: Path,
    bronze_dir: Path,
    expected_tables: list[str] | None = None,
) -> dict[str, Any]:
    raw_dir = Path(raw_dir)
    bronze_dir = Path(bronze_dir)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    csv_files = sorted(raw_dir.glob("*.csv"))

    if expected_tables:
        expected_names = set(expected_tables)
        csv_files = [path for path in csv_files if path.name in expected_names]

    converted = []
    failed = []

    for csv_path in csv_files:
        try:
            converted.append(convert_csv_to_bronze_parquet(csv_path, bronze_dir))
        except Exception as exc:
            failed.append(
                {
                    "source_csv": str(csv_path),
                    "table_name": csv_path.stem,
                    "status": "failed",
                    "error": str(exc),
                }
            )

    return {
        "manifest_type": "bronze_conversion",
        "generated_at_utc": utc_now_iso(),
        "raw_dir": str(raw_dir),
        "bronze_dir": str(bronze_dir),
        "summary": {
            "csv_files_seen": len(csv_files),
            "converted_count": len(converted),
            "failed_count": len(failed),
            "all_converted": len(failed) == 0,
        },
        "converted": converted,
        "failed": failed,
    }


def build_bronze_from_scenario(
    repo_root: Path,
    scenario_path: Path | str = "configs/scenarios/default_synthea.yaml",
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root)

    scenario_path = Path(scenario_path)
    if not scenario_path.is_absolute():
        scenario_path = repo_root / scenario_path

    scenario_config = load_yaml(scenario_path)

    raw_dir = Path(scenario_config["paths"]["raw_dir"])
    if not raw_dir.is_absolute():
        raw_dir = repo_root / raw_dir

    bronze_dir = Path(scenario_config["paths"]["bronze_dir"])
    if not bronze_dir.is_absolute():
        bronze_dir = repo_root / bronze_dir

    required_tables = scenario_config["expected_raw_tables"]["required"]
    optional_tables = scenario_config["expected_raw_tables"].get("optional", [])
    expected_tables = required_tables + optional_tables

    manifest = convert_raw_dir_to_bronze(
        raw_dir=raw_dir,
        bronze_dir=bronze_dir,
        expected_tables=expected_tables,
    )

    manifest["scenario"] = {
        "id": scenario_config["scenario"]["id"],
        "name": scenario_config["scenario"]["name"],
        "scenario_path": str(scenario_path),
    }

    manifest["synthea"] = scenario_config["synthea"]

    if output_path is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = repo_root / "reports" / "manifests" / f"bronze_conversion_{timestamp}.json"
    else:
        output_path = Path(output_path)
        if not output_path.is_absolute():
            output_path = repo_root / output_path

    manifest["manifest_path"] = str(output_path)
    write_json(manifest, output_path)

    return manifest
