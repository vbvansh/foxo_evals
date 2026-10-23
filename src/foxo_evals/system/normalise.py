"""Deterministic half of the Mood agent: names, units, reference ranges and flags.

The LLM only reads the report and copies what is printed. Anything with a single
right answer (unit conversion, range lookup, flagging) happens here, in code that
can be unit tested. An LLM doing arithmetic is a confident wrong answer waiting
to happen.
"""

import json
import re
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel

from foxo_evals.config import REFERENCE_DIR
from foxo_evals.system.schemas import (
    Biomarker,
    Comparator,
    Flag,
    LabReportRaw,
    LabResult,
    RawBiomarker,
    Sex,
)

Bounds = tuple[float | None, float | None]


class BiomarkerSpec(BaseModel):
    display_name: str
    canonical_unit: str
    aliases: list[str]
    unit_factors: dict[str, float]
    ref: dict[Sex, Bounds]
    critical: Bounds = (None, None)


@lru_cache
def load_biomarker_specs(path: Path = REFERENCE_DIR / "biomarkers.json") -> dict[str, BiomarkerSpec]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {name: BiomarkerSpec.model_validate(spec) for name, spec in data["biomarkers"].items()}


def clean_label(text: str) -> str:
    """Lowercase and drop punctuation so 'LDL-C' and 'ldl c' compare equal."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def clean_unit(text: str) -> str:
    return text.strip().lower().replace("µ", "u").replace("μ", "u").replace(" ", "")


def alias_index(specs: dict[str, BiomarkerSpec]) -> dict[str, str]:
    """Map every cleaned alias to its canonical biomarker name."""
    index = {}
    for name, spec in specs.items():
        for alias in [name, *spec.aliases]:
            index[clean_label(alias)] = name
    return index


# Characters OCR commonly swaps for letters, e.g. 'Gluc0se'.
OCR_FIXES = str.maketrans({"0": "o", "1": "l"})


def find_canonical_name(name_as_printed: str, aliases: dict[str, str]) -> str | None:
    """Exact alias match, then one retry with OCR digit-for-letter fixes.

    No fuzzy matching on purpose: 'LDL cholesterol' and 'HDL cholesterol' are one
    letter apart, and mixing them up is exactly the error we can't afford.
    """
    label = clean_label(name_as_printed)
    return aliases.get(label) or aliases.get(label.translate(OCR_FIXES))


VALUE_PATTERN = re.compile(r"^(<=?|>=?|≤|≥)?\s*(\d+(?:\.\d+)?)$")
COMPARATORS: dict[str, Comparator] = {"<": "<", "<=": "<", "≤": "<", ">": ">", ">=": ">", "≥": ">"}


def parse_value(text: str) -> tuple[float, Comparator] | None:
    """Parse '5.6', '<5' or '> 1000'. Anything else returns None: we never guess a value."""
    match = VALUE_PATTERN.match(text.strip())
    if not match:
        return None
    return float(match.group(2)), COMPARATORS.get(match.group(1) or "", "=")


def to_canonical_unit(value: float, unit_as_printed: str, spec: BiomarkerSpec) -> float | None:
    factors = {clean_unit(unit): factor for unit, factor in spec.unit_factors.items()}
    factor = factors.get(clean_unit(unit_as_printed))
    if factor is None:
        return None
    return round(value * factor, 2)


def assign_flag(value: float, ref: Bounds, critical: Bounds) -> Flag:
    """Critical thresholds are checked first, so a critical value is never reported as just 'high'.

    Reference bounds are inclusive (a value equal to ref_high is normal); critical
    thresholds trigger at or beyond the threshold. A value printed as '<5' is
    flagged using 5 itself.
    """
    ref_low, ref_high = ref
    critical_low, critical_high = critical
    if critical_low is not None and value <= critical_low:
        return "critical_low"
    if critical_high is not None and value >= critical_high:
        return "critical_high"
    if ref_low is not None and value < ref_low:
        return "low"
    if ref_high is not None and value > ref_high:
        return "high"
    return "normal"


QUALITY_KEYWORDS = {
    "haemolysed": ("haemoly", "hemoly"),
    "lipaemic": ("lipaem", "lipem"),
    "clotted": ("clotted",),
    "insufficient_sample": ("insufficient sample", "quantity not sufficient", "qns"),
    "delayed_processing": ("delayed processing", "processed after"),
}
NEGATION = re.compile(r"\b(no|not|nil|none|absent|without)\b")


def detect_quality_issues(raw_text: str) -> list[str]:
    """Keyword scan per line. Lines like 'Sample not haemolysed' or 'Haemolysis: Nil' are skipped."""
    issues = []
    for line in raw_text.lower().splitlines():
        if NEGATION.search(line):
            continue
        for issue, keywords in QUALITY_KEYWORDS.items():
            if issue not in issues and any(keyword in line for keyword in keywords):
                issues.append(issue)
    return issues


def describe(item: RawBiomarker) -> str:
    return f"{item.name_as_printed}: {item.value_as_printed} {item.unit_as_printed}".strip()


def readings_agree(readings: list[Biomarker]) -> bool:
    """Duplicates agree if within 2%, e.g. glucose printed as 101 mg/dL and as 5.6 mmol/L."""
    values = [b.value for b in readings]
    same_comparator = len({b.comparator for b in readings}) == 1
    return same_comparator and max(values) - min(values) <= 0.02 * max(values)


def normalise_report(
    report: LabReportRaw,
    raw_items: list[RawBiomarker],
    specs: dict[str, BiomarkerSpec] | None = None,
) -> LabResult:
    """Turn the LLM's raw readings into a LabResult.

    Anything we can't read with certainty goes to unparsed_items with a reason:
    unknown names, unreadable values, unknown units, and markers printed twice
    with different values (we won't pick one).
    """
    specs = specs or load_biomarker_specs()
    aliases = alias_index(specs)
    readings: dict[str, list[Biomarker]] = {}
    unparsed: list[str] = []

    for item in raw_items:
        source = describe(item)
        name = find_canonical_name(item.name_as_printed, aliases)
        if name is None:
            unparsed.append(f"{source} (unrecognised biomarker)")
            continue
        parsed = parse_value(item.value_as_printed)
        if parsed is None:
            unparsed.append(f"{source} (unreadable value)")
            continue
        value, comparator = parsed
        spec = specs[name]
        canonical_value = to_canonical_unit(value, item.unit_as_printed, spec)
        if canonical_value is None:
            unparsed.append(f"{source} (missing or unknown unit)")
            continue
        ref = spec.ref[report.member_sex]
        readings.setdefault(name, []).append(
            Biomarker(
                name=name,
                value=canonical_value,
                comparator=comparator,
                unit=spec.canonical_unit,
                ref_low=ref[0],
                ref_high=ref[1],
                flag=assign_flag(canonical_value, ref, spec.critical),
                source=item,
            )
        )

    biomarkers = []
    for name, found in readings.items():
        if readings_agree(found):
            biomarkers.append(found[0])
        else:
            sources = "; ".join(describe(b.source) for b in found)
            unparsed.append(f"{sources} (conflicting duplicate values for {name})")

    return LabResult(
        report_id=report.report_id,
        biomarkers=biomarkers,
        quality_issues=detect_quality_issues(report.raw_text),
        unparsed_items=unparsed,
    )
