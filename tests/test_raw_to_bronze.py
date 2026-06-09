from pathlib import Path
import sys


# Ensure repo root is importable so `import src...` works when running tests
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import polars as pl

from src.ingest.raw_to_bronze import convert_csv_to_bronze_parquet, convert_raw_dir_to_bronze


def test_convert_csv_to_bronze_parquet(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    bronze_dir = tmp_path / "bronze"
    raw_dir.mkdir()

    csv_path = raw_dir / "patients.csv"
    csv_path.write_text(
        "Id,BIRTHDATE,DEATHDATE\n"
        "p1,1970-01-01,\n"
        "p2,1980-01-01,\n",
        encoding="utf-8",
    )

    result = convert_csv_to_bronze_parquet(csv_path, bronze_dir)

    parquet_path = Path(result["bronze_parquet"])

    assert parquet_path.exists()
    assert result["table_name"] == "patients"
    assert result["row_count"] == 2
    assert result["column_count"] == 3
    assert result["status"] == "converted"

    df = pl.read_parquet(parquet_path)

    assert df.height == 2
    assert df.columns == ["Id", "BIRTHDATE", "DEATHDATE"]


def test_convert_raw_dir_to_bronze_converts_expected_tables(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    bronze_dir = tmp_path / "bronze"
    raw_dir.mkdir()

    (raw_dir / "patients.csv").write_text(
        "Id,BIRTHDATE\np1,1970-01-01\n",
        encoding="utf-8",
    )

    (raw_dir / "conditions.csv").write_text(
        "START,STOP,PATIENT,CODE,DESCRIPTION\n2020-01-01,,p1,44054006,Diabetes mellitus type 2\n",
        encoding="utf-8",
    )

    manifest = convert_raw_dir_to_bronze(
        raw_dir=raw_dir,
        bronze_dir=bronze_dir,
        expected_tables=["patients.csv", "conditions.csv"],
    )

    assert manifest["summary"]["converted_count"] == 2
    assert manifest["summary"]["failed_count"] == 0
    assert manifest["summary"]["all_converted"] is True

    assert (bronze_dir / "patients.parquet").exists()
    assert (bronze_dir / "conditions.parquet").exists()


def test_convert_raw_dir_to_bronze_ignores_unexpected_when_expected_tables_given(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    bronze_dir = tmp_path / "bronze"
    raw_dir.mkdir()

    (raw_dir / "patients.csv").write_text(
        "Id,BIRTHDATE\np1,1970-01-01\n",
        encoding="utf-8",
    )

    (raw_dir / "unexpected.csv").write_text(
        "A,B\n1,2\n",
        encoding="utf-8",
    )

    manifest = convert_raw_dir_to_bronze(
        raw_dir=raw_dir,
        bronze_dir=bronze_dir,
        expected_tables=["patients.csv"],
    )

    assert manifest["summary"]["converted_count"] == 1
    assert (bronze_dir / "patients.parquet").exists()
    assert not (bronze_dir / "unexpected.parquet").exists()