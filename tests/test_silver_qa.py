from datetime import date
from pathlib import Path

import sys


# Ensure repo root is importable so `import src...` works when running tests
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import polars as pl

from src.normalize.silver_qa import count_t2dm_signal, run_silver_qa


def write_parquet(path: Path, rows: list[dict]):
    df = pl.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def test_silver_qa_passes_on_minimal_valid_tables(tmp_path: Path):
    silver_dir = tmp_path / "silver"

    write_parquet(
        silver_dir / "patients.parquet",
        [
            {
                "patient_id": "p1",
                "birth_date": date(1970, 1, 1),
                "death_date": None,
                "sex": "M",
                "race": "white",
                "ethnicity": "nonhispanic",
                "birth_year": 1970,
                "source_table": "patients",
            }
        ],
    )

    write_parquet(
        silver_dir / "conditions.parquet",
        [
            {
                "patient_id": "p1",
                "encounter_id": "e1",
                "condition_start_date": date(2020, 1, 1),
                "condition_stop_date": None,
                "code": "44054006",
                "description": "Diabetes mellitus type 2",
                "source_table": "conditions",
            }
        ],
    )

    manifest = run_silver_qa(
        silver_dir=silver_dir,
        required_tables=["patients.csv", "conditions.csv"],
    )

    assert manifest["summary"]["structural_checks_passed"] is True
    assert manifest["summary"]["missing_required_table_count"] == 0
    assert manifest["summary"]["tables_with_invalid_date_dtypes_count"] == 0
    assert manifest["summary"]["failed_patient_reference_table_count"] == 0
    assert manifest["t2dm_signal"]["status"] == "present"
    assert manifest["t2dm_signal"]["t2dm_patient_count"] == 1


def test_silver_qa_fails_invalid_date_dtype(tmp_path: Path):
    silver_dir = tmp_path / "silver"

    write_parquet(
        silver_dir / "patients.parquet",
        [
            {
                "patient_id": "p1",
                "birth_date": "1970-01-01",
                "death_date": None,
                "sex": "M",
                "race": "white",
                "ethnicity": "nonhispanic",
                "birth_year": 1970,
                "source_table": "patients",
            }
        ],
    )

    manifest = run_silver_qa(
        silver_dir=silver_dir,
        required_tables=["patients.csv"],
    )

    assert manifest["summary"]["structural_checks_passed"] is False
    assert manifest["summary"]["tables_with_invalid_date_dtypes_count"] == 1


def test_silver_qa_fails_unknown_patient_reference(tmp_path: Path):
    silver_dir = tmp_path / "silver"

    write_parquet(
        silver_dir / "patients.parquet",
        [
            {
                "patient_id": "p1",
                "birth_date": date(1970, 1, 1),
                "death_date": None,
                "sex": "M",
                "race": "white",
                "ethnicity": "nonhispanic",
                "birth_year": 1970,
                "source_table": "patients",
            }
        ],
    )

    write_parquet(
        silver_dir / "conditions.parquet",
        [
            {
                "patient_id": "p999",
                "encounter_id": "e1",
                "condition_start_date": date(2020, 1, 1),
                "condition_stop_date": None,
                "code": "44054006",
                "description": "Diabetes mellitus type 2",
                "source_table": "conditions",
            }
        ],
    )

    manifest = run_silver_qa(
        silver_dir=silver_dir,
        required_tables=["patients.csv", "conditions.csv"],
    )

    assert manifest["summary"]["structural_checks_passed"] is False
    assert manifest["summary"]["failed_patient_reference_table_count"] == 1
    assert "conditions" in manifest["failed_patient_reference_tables"]


def test_t2dm_signal_absent_when_no_matching_code(tmp_path: Path):
    silver_dir = tmp_path / "silver"

    write_parquet(
        silver_dir / "conditions.parquet",
        [
            {
                "patient_id": "p1",
                "encounter_id": "e1",
                "condition_start_date": date(2020, 1, 1),
                "condition_stop_date": None,
                "code": "123",
                "description": "Other condition",
                "source_table": "conditions",
            }
        ],
    )

    result = count_t2dm_signal(silver_dir)

    assert result["checked"] is True
    assert result["status"] == "absent"
    assert result["t2dm_condition_row_count"] == 0
    assert result["t2dm_patient_count"] == 0