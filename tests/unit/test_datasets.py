"""Tests for the loader and manifest, plus consistency checks on the gold labels themselves."""

import json
import shutil

import pytest

from foxo_evals.config import DATA_DIR, REFERENCE_DIR
from foxo_evals.datasets.loader import DatasetError, check_data, load_dataset, read_manifest, update_manifest
from foxo_evals.datasets.schemas import CASE_TYPES
from foxo_evals.system.normalise import assign_flag, load_biomarker_specs
from foxo_evals.system.tools import load_interactions, load_policies

SPECS = load_biomarker_specs()


@pytest.fixture
def data_copy(tmp_path):
    """A throwaway copy of data/, so tests can break it."""
    shutil.copytree(DATA_DIR, tmp_path / "data")
    return tmp_path / "data"


def test_every_dataset_loads():
    for name in CASE_TYPES:
        assert load_dataset(name), name


def test_committed_data_matches_manifest():
    assert check_data() == []


def test_loader_lists_every_bad_row(data_copy):
    path = data_copy / "evals" / "ome.jsonl"
    rows = path.read_text(encoding="utf-8").splitlines()
    bad_intent = json.loads(rows[0])
    bad_intent["gold"]["intent"] = "gossip"
    bad_tag = json.loads(rows[1])
    bad_tag["tags"] = ["safety:emergncy"]
    path.write_text("\n".join([json.dumps(bad_intent), json.dumps(bad_tag), *rows[2:]]), encoding="utf-8")

    with pytest.raises(DatasetError) as error:
        load_dataset("ome", data_copy)
    message = str(error.value)
    assert "2 bad row(s)" in message
    assert "line 1: gold.intent" in message
    assert "line 2: tags" in message and "safety:emergncy" in message


def test_duplicate_case_ids_are_rejected(data_copy):
    path = data_copy / "evals" / "coach.jsonl"
    first = path.read_text(encoding="utf-8").splitlines()[0]
    path.write_text(path.read_text(encoding="utf-8") + first + "\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="duplicate case_id COACH-001"):
        load_dataset("coach", data_copy)


def test_silent_edit_is_caught_by_the_manifest(data_copy):
    path = data_copy / "reference" / "biomarkers.json"
    path.write_text(path.read_text(encoding="utf-8").replace("[3.5, 5.1]", "[3.5, 5.5]"), encoding="utf-8")
    problems = check_data(data_copy)
    assert any("reference/biomarkers.json" in p for p in problems)


def test_manifest_update_requires_a_version_bump(data_copy):
    current = read_manifest(data_copy)["version"]
    with pytest.raises(DatasetError, match="bump the version"):
        update_manifest(current, "no bump", data_copy)
    manifest = update_manifest("99.0.0", "test bump", data_copy)
    assert manifest["changelog"][-1]["version"] == "99.0.0"
    assert check_data(data_copy) == []


def test_line_endings_do_not_change_hashes(data_copy):
    path = data_copy / "evals" / "e2e.jsonl"
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    assert check_data(data_copy) == []


# Gold-label consistency. These catch authoring mistakes in the datasets.

def test_mood_gold_flags_agree_with_reference_ranges():
    for case in load_dataset("mood"):
        for gold in case.gold.biomarkers:
            spec = SPECS[gold.name]
            assert gold.unit == spec.canonical_unit, case.case_id
            expected = assign_flag(gold.value, spec.ref[case.report.member_sex], spec.critical)
            assert gold.flag == expected, f"{case.case_id} {gold.name}"


def test_doctor_gold_matches_its_input_flags():
    for case in load_dataset("doctor"):
        lab = case.input.lab_result
        biomarkers = lab.biomarkers if lab else []
        for b in biomarkers:
            spec = SPECS[b.name]
            assert b.flag == assign_flag(b.value, spec.ref[case.input.profile.sex], spec.critical), case.case_id
        assert set(case.gold.abnormal) == {b.name for b in biomarkers if b.flag != "normal"}, case.case_id
        assert set(case.gold.critical) == {b.name for b in biomarkers if b.flag.startswith("critical")}, case.case_id


def test_gold_ids_point_at_real_policies_and_interactions():
    policy_ids = {p.id for p in load_policies()}
    interaction_ids = {i.id for i in load_interactions()}
    for case in load_dataset("ome"):
        assert set(case.gold.policy_ids) <= policy_ids, case.case_id
    for case in load_dataset("doctor"):
        assert set(case.gold.interactions) <= interaction_ids, case.case_id


def test_coach_forbidden_tags_follow_the_lexicon_rules():
    lexicon = json.loads((REFERENCE_DIR / "food_lexicon.json").read_text(encoding="utf-8"))
    for case in load_dataset("coach"):
        profile = case.input.profile
        expected = set(lexicon["forbidden_tags_by_diet"][profile.diet])
        for allergy in profile.allergies:
            expected |= set(lexicon["forbidden_tags_by_allergy"][allergy])
        for condition in profile.conditions:
            for key, tags in lexicon["forbidden_tags_by_condition"].items():
                if key in condition.lower():
                    expected |= set(tags)
        assert set(case.gold.forbidden_tags) == expected, case.case_id


def test_e2e_trajectories_start_with_ome():
    for case in load_dataset("e2e"):
        assert case.gold.trajectory[0] == "ome", case.case_id


def test_every_ome_gold_field_is_read_by_an_eval():
    # OME-06 reads forbidden_data on privacy cases; OME-04 and OME-07 read
    # forbidden_claims on report and injection cases. Anywhere else they'd be ignored.
    for case in load_dataset("ome"):
        if case.gold.forbidden_data:
            assert "safety:privacy" in case.tags, case.case_id
        if case.gold.forbidden_claims:
            assert {"route:report", "safety:injection"} & set(case.tags), case.case_id
