import json
import sys
from pathlib import Path


# Ensure repo root is importable so `import src...` works when running tests
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ingest.raw_inventory import inventory_raw_dir, write_manifest


def test_inventory_reports_missing_required_tables(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    manifest = inventory_raw_dir(
        raw_dir=raw_dir,
        required_tables=["patients.csv", "conditions.csv"],
        optional_tables=["allergies.csv"],
    )

    assert manifest["summary"]["required_table_count"] == 2
    assert manifest["summary"]["missing_required_count"] == 2
    assert manifest["summary"]["all_required_present"] is False
    assert "patients.csv" in manifest["missing_required_tables"]
    assert "conditions.csv" in manifest["missing_required_tables"]


def test_inventory_hashes_and_counts_existing_csv(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    patients_csv = raw_dir / "patients.csv"
    patients_csv.write_text(
        "Id,BIRTHDATE,DEATHDATE\n"
        "p1,1970-01-01,\n"
        "p2,1980-01-01,\n",
        encoding="utf-8",
    )

    manifest = inventory_raw_dir(
        raw_dir=raw_dir,
        required_tables=["patients.csv"],
        optional_tables=[],
    )

    file_record = manifest["files"][0]

    assert manifest["summary"]["all_required_present"] is True
    assert file_record["exists"] is True
    assert file_record["status"] == "present"
    assert file_record["row_count"] == 2
    assert file_record["header"] == ["Id", "BIRTHDATE", "DEATHDATE"]
    assert isinstance(file_record["sha256"], str)
    assert len(file_record["sha256"]) == 64


def test_inventory_detects_unexpected_csv_files(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    extra_csv = raw_dir / "extra_table.csv"
    extra_csv.write_text("a,b\n1,2\n", encoding="utf-8")

    manifest = inventory_raw_dir(
        raw_dir=raw_dir,
        required_tables=[],
        optional_tables=[],
    )

    assert manifest["summary"]["unexpected_csv_count"] == 1
    assert manifest["unexpected_csv_files"] == ["extra_table.csv"]


def test_write_manifest_creates_json_file(tmp_path: Path):
    manifest = {
        "manifest_type": "raw_inventory",
        "summary": {
            "all_required_present": False,
        },
    }

    output_path = tmp_path / "manifest.json"
    written_path = write_manifest(manifest, output_path)

    assert written_path.exists()

    loaded = json.loads(written_path.read_text(encoding="utf-8"))
    assert loaded["manifest_type"] == "raw_inventory"