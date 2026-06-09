from pathlib import Path

import polars as pl

from src.normalize.bronze_to_silver import normalize_bronze_to_silver, normalize_table


def write_parquet(path: Path, rows: list[dict]):
    df = pl.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def test_normalize_patients_table(tmp_path: Path):
    bronze_dir = tmp_path / "bronze"
    silver_dir = tmp_path / "silver"

    write_parquet(
        bronze_dir / "patients.parquet",
        [
            {
                "Id": "p1",
                "BIRTHDATE": "1970-01-01",
                "DEATHDATE": "",
                "GENDER": "M",
                "RACE": "white",
                "ETHNICITY": "nonhispanic",
            }
        ],
    )

    result = normalize_table(bronze_dir, silver_dir, "patients")

    assert result["status"] == "normalized"
    assert result["input_row_count"] == 1
    assert result["output_row_count"] == 1

    df = pl.read_parquet(silver_dir / "patients.parquet")

    assert "patient_id" in df.columns
    assert "birth_date" in df.columns
    assert "birth_year" in df.columns
    assert df["patient_id"][0] == "p1"
    assert df["birth_year"][0] == 1970


def test_normalize_conditions_table_preserves_t2dm_code(tmp_path: Path):
    bronze_dir = tmp_path / "bronze"
    silver_dir = tmp_path / "silver"

    write_parquet(
        bronze_dir / "conditions.parquet",
        [
            {
                "START": "2020-01-01",
                "STOP": "",
                "PATIENT": "p1",
                "ENCOUNTER": "e1",
                "CODE": "44054006",
                "DESCRIPTION": "Diabetes mellitus type 2",
            }
        ],
    )

    result = normalize_table(bronze_dir, silver_dir, "conditions")

    assert result["status"] == "normalized"

    df = pl.read_parquet(silver_dir / "conditions.parquet")

    assert df["patient_id"][0] == "p1"
    assert df["encounter_id"][0] == "e1"
    assert df["code"][0] == "44054006"
    assert df["description"][0] == "Diabetes mellitus type 2"
    assert "condition_start_date" in df.columns


def test_normalize_bronze_to_silver_core_tables(tmp_path: Path):
    bronze_dir = tmp_path / "bronze"
    silver_dir = tmp_path / "silver"

    write_parquet(
        bronze_dir / "patients.parquet",
        [{"Id": "p1", "BIRTHDATE": "1970-01-01", "DEATHDATE": "", "GENDER": "F"}],
    )

    write_parquet(
        bronze_dir / "encounters.parquet",
        [{"Id": "e1", "START": "2020-01-01", "STOP": "", "PATIENT": "p1", "CODE": "x", "DESCRIPTION": "Encounter"}],
    )

    write_parquet(
        bronze_dir / "conditions.parquet",
        [{"START": "2020-01-01", "STOP": "", "PATIENT": "p1", "ENCOUNTER": "e1", "CODE": "44054006", "DESCRIPTION": "Diabetes mellitus type 2"}],
    )

    manifest = normalize_bronze_to_silver(
        bronze_dir=bronze_dir,
        silver_dir=silver_dir,
        tables=["patients", "encounters", "conditions"],
    )

    assert manifest["summary"]["all_normalized"] is True
    assert manifest["summary"]["normalized_count"] == 3
    assert manifest["summary"]["missing_count"] == 0
    assert manifest["summary"]["failed_count"] == 0

    assert (silver_dir / "patients.parquet").exists()
    assert (silver_dir / "encounters.parquet").exists()
    assert (silver_dir / "conditions.parquet").exists()


def test_normalize_bronze_to_silver_reports_missing_tables(tmp_path: Path):
    bronze_dir = tmp_path / "bronze"
    silver_dir = tmp_path / "silver"
    bronze_dir.mkdir()

    manifest = normalize_bronze_to_silver(
        bronze_dir=bronze_dir,
        silver_dir=silver_dir,
        tables=["patients"],
    )

    assert manifest["summary"]["all_normalized"] is False
    assert manifest["summary"]["missing_count"] == 1
    assert manifest["missing"] == ["patients"]