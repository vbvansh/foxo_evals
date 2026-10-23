"""Sanity checks on the reference files, so a bad edit fails here and not mid-eval."""

import json

from foxo_evals.config import REFERENCE_DIR
from foxo_evals.system.normalise import clean_label, clean_unit, load_biomarker_specs

SPECS = load_biomarker_specs()


def test_aliases_are_unique_across_biomarkers():
    owner: dict[str, str] = {}
    for name, spec in SPECS.items():
        for alias in [name, *spec.aliases]:
            key = clean_label(alias)
            assert owner.setdefault(key, name) == name, f"{alias!r} used by {owner[key]} and {name}"


def test_canonical_unit_has_factor_one():
    for name, spec in SPECS.items():
        factors = {clean_unit(unit): factor for unit, factor in spec.unit_factors.items()}
        assert factors.get(clean_unit(spec.canonical_unit)) == 1, name


def test_ranges_are_ordered_and_inside_critical_limits():
    for name, spec in SPECS.items():
        assert set(spec.ref) == {"male", "female"}, name
        critical_low, critical_high = spec.critical
        for low, high in spec.ref.values():
            if low is not None and high is not None:
                assert low < high, name
            if critical_low is not None:
                assert low is not None and critical_low < low, name
            if critical_high is not None:
                assert high is not None and critical_high > high, name


def test_interactions_point_at_known_biomarkers():
    lines = (REFERENCE_DIR / "interactions.jsonl").read_text(encoding="utf-8").splitlines()
    for row in map(json.loads, lines):
        assert row["biomarker"] in SPECS, row["id"]


def test_every_forbidden_food_tag_can_be_detected():
    lexicon = json.loads((REFERENCE_DIR / "food_lexicon.json").read_text(encoding="utf-8"))
    detectable = {tag for tags in lexicon["foods"].values() for tag in tags}
    rule_groups = ("forbidden_tags_by_diet", "forbidden_tags_by_allergy", "forbidden_tags_by_condition")
    forbidden = {tag for group in rule_groups for tags in lexicon[group].values() for tag in tags}
    assert forbidden <= detectable, forbidden - detectable
