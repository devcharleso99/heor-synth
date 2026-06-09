from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(relative_path: str):
    path = REPO_ROOT / relative_path
    assert path.exists(), f"Missing file: {relative_path}"

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert isinstance(data, dict), f"YAML did not parse into a dictionary: {relative_path}"
    return data


def test_required_config_files_exist():
    required_files = [
        "configs/study/t2dm_phase1.yaml",
        "configs/concepts/t2dm_concepts.yaml",
        "configs/scenarios/default_synthea.yaml",
    ]

    for relative_path in required_files:
        assert (REPO_ROOT / relative_path).exists(), f"Missing required file: {relative_path}"


def test_study_config_core_rules():
    data = load_yaml("configs/study/t2dm_phase1.yaml")

    assert data["study"]["id"] == "t2dm_phase1"
    assert data["population"]["minimum_age_years"] == 18
    assert data["windows"]["baseline"]["days_before_index"] == 365
    assert data["windows"]["follow_up"]["max_months_after_index"] == 24

    assert len(data["inclusion_criteria"]) >= 4
    assert len(data["exclusion_criteria"]) >= 2


def test_concept_config_has_required_t2dm_codes():
    data = load_yaml("configs/concepts/t2dm_concepts.yaml")
    concept_sets = data["concept_sets"]

    assert "type_2_diabetes_mellitus" in concept_sets
    assert "type_1_diabetes_mellitus" in concept_sets
    assert "pregnancy" in concept_sets
    assert "metformin" in concept_sets
    assert "sglt2_inhibitors" in concept_sets
    assert "glp1_receptor_agonists" in concept_sets

    t2dm_codes = concept_sets["type_2_diabetes_mellitus"]["codes"]
    assert any(str(item["code"]) == "44054006" for item in t2dm_codes)


def test_default_synthea_scenario_is_reproducible():
    data = load_yaml("configs/scenarios/default_synthea.yaml")

    assert data["scenario"]["id"] == "default_synthea"
    assert data["synthea"]["seed"] is not None
    assert data["synthea"]["reference_date"] is not None
    assert data["synthea"]["export_format"] == "csv"

    required_tables = data["expected_raw_tables"]["required"]

    for table in [
        "patients.csv",
        "encounters.csv",
        "conditions.csv",
        "medications.csv",
        "observations.csv",
        "procedures.csv",
        "careplans.csv",
    ]:
        assert table in required_tables