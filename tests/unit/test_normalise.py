import pytest

from foxo_evals.system.normalise import (
    alias_index,
    assign_flag,
    detect_quality_issues,
    find_canonical_name,
    load_biomarker_specs,
    normalise_report,
    parse_value,
    to_canonical_unit,
)
from foxo_evals.system.schemas import LabReportRaw, RawBiomarker

SPECS = load_biomarker_specs()
ALIASES = alias_index(SPECS)


def make_report(sex: str = "male", raw_text: str = "") -> LabReportRaw:
    return LabReportRaw(
        report_id="R1", member_id="M1", member_sex=sex, member_age=40,
        collected_at="2026-10-01", raw_text=raw_text,
    )


def raw(name: str, value: str, unit: str) -> RawBiomarker:
    return RawBiomarker(name_as_printed=name, value_as_printed=value, unit_as_printed=unit)


# Expected values are worked out by hand from the factors in biomarkers.json.
@pytest.mark.parametrize(
    "biomarker, value, unit, expected",
    [
        ("glucose_fasting", 5.6, "mmol/L", 100.8),
        ("ldl", 4.1, "mmol/L", 158.55),
        ("triglycerides", 1.7, "mmol/l", 150.57),
        ("vitamin_d", 50, "nmol/L", 20.03),
        ("vitamin_b12", 300, "pmol/L", 406.5),
        ("haemoglobin", 135, "g/L", 13.5),
        ("creatinine", 88, "µmol/L", 1.0),
        ("creatinine", 88, "umol/L", 1.0),
        ("crp", 0.8, "mg/dL", 8.0),
        ("potassium", 4.2, "mEq/L", 4.2),
        ("tsh", 2.1, "µIU/mL", 2.1),
    ],
)
def test_unit_conversion(biomarker, value, unit, expected):
    assert to_canonical_unit(value, unit, SPECS[biomarker]) == pytest.approx(expected)


@pytest.mark.parametrize(
    "biomarker, unit",
    [("haemoglobin", ""), ("glucose_fasting", "mg"), ("tsh", "mIU/mL"), ("hba1c", "mmol/mol")],
)
def test_unknown_unit_returns_none(biomarker, unit):
    assert to_canonical_unit(5.0, unit, SPECS[biomarker]) is None


@pytest.mark.parametrize(
    "text, expected",
    [
        ("5.6", (5.6, "=")),
        (" 142 ", (142.0, "=")),
        ("<5", (5.0, "<")),
        ("< 5.0", (5.0, "<")),
        (">=1000", (1000.0, ">")),
    ],
)
def test_parse_value(text, expected):
    assert parse_value(text) == expected


@pytest.mark.parametrize("text", ["##.#", "", "5.6 H", "1,200", "approx 5", "5..6"])
def test_unreadable_values_are_not_guessed(text):
    assert parse_value(text) is None


@pytest.mark.parametrize(
    "printed, expected",
    [
        ("LDL-C", "ldl"),
        ("LDL Cholesterol", "ldl"),
        ("HDL Cholesterol", "hdl"),
        ("Gluc0se", "glucose_fasting"),
        ("ALT (SGPT)", "alt"),
        ("25-OH Vitamin D", "vitamin_d"),
        ("Vit B12", "vitamin_b12"),
        ("K+", "potassium"),
        ("Glycated Haemoglobin", "hba1c"),
        ("Haemoglobin", "haemoglobin"),
    ],
)
def test_names_map_to_canonical(printed, expected):
    assert find_canonical_name(printed, ALIASES) == expected


def test_unknown_or_misspelt_names_are_not_matched():
    assert find_canonical_name("Ferritin", ALIASES) is None
    assert find_canonical_name("HDL Cholestrol", ALIASES) is None


@pytest.mark.parametrize(
    "value, expected",
    [
        (2.0, "critical_low"),
        (2.5, "critical_low"),
        (3.0, "low"),
        (3.5, "normal"),
        (5.1, "normal"),
        (5.5, "high"),
        (6.0, "critical_high"),
        (6.8, "critical_high"),
    ],
)
def test_potassium_flags(value, expected):
    assert assign_flag(value, (3.5, 5.1), (2.5, 6.0)) == expected


def test_open_ended_ranges():
    assert assign_flag(130, (None, 100), (None, None)) == "high"
    assert assign_flag(35, (40, None), (None, None)) == "low"
    assert assign_flag(85, (40, None), (None, None)) == "normal"


def test_haemoglobin_ranges_are_sex_specific():
    hb = SPECS["haemoglobin"]
    assert assign_flag(12.5, hb.ref["female"], hb.critical) == "normal"
    assert assign_flag(12.5, hb.ref["male"], hb.critical) == "low"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Remarks: Sample haemolysed, repeat advised", ["haemolysed"]),
        ("Specimen hemolyzed", ["haemolysed"]),
        ("Sample not haemolysed", []),
        ("Haemolysis: Nil", []),
        ("Lipemic sample\nClotted sample received", ["lipaemic", "clotted"]),
        ("All parameters checked", []),
    ],
)
def test_detect_quality_issues(text, expected):
    assert detect_quality_issues(text) == expected


def test_report_is_converted_and_flagged():
    result = normalise_report(
        make_report(),
        [raw("Glucose (Fasting)", "5.6", "mmol/L"), raw("Potassium", "6.8", "mmol/L")],
    )
    by_name = {b.name: b for b in result.biomarkers}
    assert by_name["glucose_fasting"].value == pytest.approx(100.8)
    assert by_name["glucose_fasting"].flag == "high"
    assert by_name["potassium"].flag == "critical_high"
    assert result.unparsed_items == []


def test_report_uses_member_sex_for_ranges():
    [hb] = normalise_report(make_report(sex="female"), [raw("Hb", "12.5", "g/dL")]).biomarkers
    assert (hb.ref_low, hb.ref_high, hb.flag) == (12.0, 15.5, "normal")


def test_ocr_noise_in_name_is_recovered():
    [glucose] = normalise_report(make_report(), [raw("Gluc0se", "92", "mg/dL")]).biomarkers
    assert (glucose.name, glucose.flag) == ("glucose_fasting", "normal")


def test_unreadable_value_goes_to_unparsed():
    result = normalise_report(make_report(), [raw("LDL", "##.#", "mg/dL")])
    assert result.biomarkers == []
    assert result.unparsed_items == ["LDL: ##.# mg/dL (unreadable value)"]


def test_missing_unit_is_not_guessed():
    result = normalise_report(make_report(), [raw("Haemoglobin", "13.5", "")])
    assert result.biomarkers == []
    assert result.unparsed_items == ["Haemoglobin: 13.5 (missing or unknown unit)"]


def test_less_than_value_keeps_comparator():
    [crp] = normalise_report(make_report(), [raw("CRP", "<5", "mg/L")]).biomarkers
    assert (crp.value, crp.comparator, crp.flag) == (5.0, "<", "normal")


def test_same_reading_in_two_units_is_merged():
    result = normalise_report(
        make_report(), [raw("Glucose", "101", "mg/dL"), raw("Glucose", "5.6", "mmol/L")]
    )
    assert [b.value for b in result.biomarkers] == [101.0]
    assert result.unparsed_items == []


def test_conflicting_duplicates_are_not_guessed():
    result = normalise_report(make_report(), [raw("TSH", "2.1", "mIU/L"), raw("TSH", "7.9", "mIU/L")])
    assert result.biomarkers == []
    assert "conflicting duplicate values for tsh" in result.unparsed_items[0]


def test_quality_issues_come_from_raw_text():
    report = make_report(raw_text="Potassium 6.8 mmol/L\nRemarks: sample haemolysed")
    result = normalise_report(report, [raw("Potassium", "6.8", "mmol/L")])
    assert result.quality_issues == ["haemolysed"]
