from pathlib import Path
import sys


# Ensure repo root is importable so `import src...` works when running tests
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import polars as pl

from src.ingest.bronze_qa import count_t2dm_signal, run_bronze_qa


def write_parquet(path: Path, rows: list[dict]):
    df = pl.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def test_bronze_qa_passes_on_minimal_valid_tables(tmp_path: Path):
    bronze_dir = tmp_path / "bronze"

    write_parquet(
        bronze_dir / "patients.parquet",
        [{"Id": "p1", "BIRTHDATE": "1970-01-01"}],
    )

    write_parquet(
        bronze_dir / "encounters.parquet",
        [{"Id": "e1", "START": "2020-01-01", "PATIENT": "p1", "CODE": "x", "DESCRIPTION": "encounter"}],
    )

    write_parquet(
        bronze_dir / "conditions.parquet",
        [{"START": "2020-01-01", "PATIENT": "p1", "CODE": "44054006", "DESCRIPTION": "Diabetes mellitus type 2"}],
    )

    write_parquet(
        bronze_dir / "medications.parquet",
        [{"START": "2020-01-01", "PATIENT": "p1", "CODE": "860974", "DESCRIPTION": "Metformin"}],
    )

    write_parquet(
        bronze_dir / "observations.parquet",
        [{"DATE": "2020-01-01", "PATIENT": "p1", "CODE": "x", "DESCRIPTION": "A1c"}],
    )

    write_parquet(
        bronze_dir / "procedures.parquet",
        [{"START": "2020-01-01", "PATIENT": "p1", "CODE": "x", "DESCRIPTION": "Procedure"}],
    )

    write_parquet(
        bronze_dir / "careplans.parquet",
        [{"Id": "c1", "START": "2020-01-01", "PATIENT": "p1", "CODE": "x", "DESCRIPTION": "Care plan"}],
    )

    manifest = run_bronze_qa(
        bronze_dir=bronze_dir,
        required_tables=[
            "patients.csv",
            "encounters.csv",
            "conditions.csv",
            "medications.csv",
            "observations.csv",
            "procedures.csv",
            "careplans.csv",
        ],
    )

    assert manifest["summary"]["structural_checks_passed"] is True
    assert manifest["summary"]["missing_required_table_count"] == 0
    assert manifest["summary"]["empty_required_table_count"] == 0
    assert manifest["summary"]["failed_patient_reference_table_count"] == 0
    assert manifest["t2dm_signal"]["t2dm_condition_row_count"] == 1
    assert manifest["t2dm_signal"]["t2dm_patient_count"] == 1


def test_bronze_qa_fails_unknown_patient_reference(tmp_path: Path):
    bronze_dir = tmp_path / "bronze"

    write_parquet(
        bronze_dir / "patients.parquet",
        [{"Id": "p1", "BIRTHDATE": "1970-01-01"}],
    )

    write_parquet(
        bronze_dir / "conditions.parquet",
        [{"START": "2020-01-01", "PATIENT": "p999", "CODE": "44054006", "DESCRIPTION": "Diabetes mellitus type 2"}],
    )

    manifest = run_bronze_qa(
        bronze_dir=bronze_dir,
        required_tables=[
            "patients.csv",
            "conditions.csv",
        ],
    )

    assert manifest["summary"]["structural_checks_passed"] is False
    assert manifest["summary"]["failed_patient_reference_table_count"] == 1
    assert "conditions" in manifest["failed_patient_reference_tables"]


def test_t2dm_signal_reports_absent_code(tmp_path: Path):
    bronze_dir = tmp_path / "bronze"

    write_parquet(
        bronze_dir / "conditions.parquet",
        [{"START": "2020-01-01", "PATIENT": "p1", "CODE": "123", "DESCRIPTION": "Other condition"}],
    )

    result = count_t2dm_signal(bronze_dir)

    assert result["checked"] is True
    assert result["t2dm_condition_row_count"] == 0
    assert result["t2dm_patient_count"] == 0
    assert result["status"] == "absent_in_smoke_data"