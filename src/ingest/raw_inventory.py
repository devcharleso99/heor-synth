from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def read_csv_header_and_row_count(path: Path) -> tuple[list[str], int]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)

        try:
            header = next(reader)
        except StopIteration:
            return [], 0

        row_count = sum(1 for _ in reader)

    return header, row_count


def inspect_csv_file(raw_dir: Path, filename: str, required: bool, expected: bool) -> dict[str, Any]:
    path = raw_dir / filename

    record: dict[str, Any] = {
        "filename": filename,
        "required": required,
        "expected": expected,
        "exists": path.exists(),
        "path": str(path),
        "size_bytes": None,
        "sha256": None,
        "header": [],
        "row_count": None,
        "status": "missing",
    }

    if not path.exists():
        return record

    if not path.is_file():
        record["status"] = "not_a_file"
        return record

    header, row_count = read_csv_header_and_row_count(path)

    record.update(
        {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "header": header,
            "row_count": row_count,
            "status": "present",
        }
    )

    return record


def inventory_raw_dir(
    raw_dir: Path,
    required_tables: list[str],
    optional_tables: list[str] | None = None,
) -> dict[str, Any]:
    optional_tables = optional_tables or []
    raw_dir = Path(raw_dir)

    expected_records: list[dict[str, Any]] = []

    for filename in required_tables:
        expected_records.append(
            inspect_csv_file(
                raw_dir=raw_dir,
                filename=filename,
                required=True,
                expected=True,
            )
        )

    for filename in optional_tables:
        expected_records.append(
            inspect_csv_file(
                raw_dir=raw_dir,
                filename=filename,
                required=False,
                expected=True,
            )
        )

    expected_names = set(required_tables) | set(optional_tables)

    discovered_csv_files = []
    if raw_dir.exists():
        discovered_csv_files = sorted(path.name for path in raw_dir.glob("*.csv"))

    unexpected_csv_files = [
        filename for filename in discovered_csv_files if filename not in expected_names
    ]

    unexpected_records = [
        inspect_csv_file(
            raw_dir=raw_dir,
            filename=filename,
            required=False,
            expected=False,
        )
        for filename in unexpected_csv_files
    ]

    all_records = expected_records + unexpected_records

    missing_required = [
        record["filename"]
        for record in expected_records
        if record["required"] and not record["exists"]
    ]

    present_required = [
        record["filename"]
        for record in expected_records
        if record["required"] and record["exists"]
    ]

    manifest = {
        "manifest_type": "raw_inventory",
        "generated_at_utc": utc_now_iso(),
        "raw_dir": str(raw_dir),
        "raw_dir_exists": raw_dir.exists(),
        "summary": {
            "required_table_count": len(required_tables),
            "optional_table_count": len(optional_tables),
            "present_required_count": len(present_required),
            "missing_required_count": len(missing_required),
            "unexpected_csv_count": len(unexpected_csv_files),
            "all_required_present": len(missing_required) == 0,
        },
        "missing_required_tables": missing_required,
        "present_required_tables": present_required,
        "unexpected_csv_files": unexpected_csv_files,
        "files": all_records,
    }

    return manifest


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"YAML file did not parse into a dictionary: {path}")

    return data


def write_manifest(manifest: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return output_path


def build_raw_inventory_from_scenario(
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

    required_tables = scenario_config["expected_raw_tables"]["required"]
    optional_tables = scenario_config["expected_raw_tables"].get("optional", [])

    manifest = inventory_raw_dir(
        raw_dir=raw_dir,
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
        output_path = repo_root / "reports" / "manifests" / f"raw_inventory_{timestamp}.json"
    else:
        output_path = Path(output_path)
        if not output_path.is_absolute():
            output_path = repo_root / output_path

    manifest["manifest_path"] = str(output_path)
    write_manifest(manifest, output_path)

    return manifest