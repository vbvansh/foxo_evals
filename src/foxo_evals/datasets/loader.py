"""Load eval datasets and guard them with a manifest of hashes.

Every eval run records the dataset version and hashes. check_data() fails if a
file changed without a manifest update, so a dataset edit can never silently
change what an eval means.
"""

import hashlib
import json
import re
from datetime import date
from pathlib import Path

from pydantic import ValidationError

from foxo_evals.config import DATA_DIR
from foxo_evals.datasets.schemas import CASE_TYPES, Case

# Reference data and the KB are hashed too: changing a reference range changes
# what "correct" means just as much as editing a gold label does.
TRACKED_FILES = [f"evals/{name}.jsonl" for name in CASE_TYPES] + [
    "reference/biomarkers.json",
    "reference/interactions.jsonl",
    "reference/food_lexicon.json",
    "kb/policies.jsonl",
]


class DatasetError(Exception):
    pass


def load_dataset(name: str, data_dir: Path = DATA_DIR) -> list[Case]:
    """Validate every row and fail with a list of all bad rows, not just the first."""
    schema = CASE_TYPES[name]
    path = data_dir / "evals" / f"{name}.jsonl"
    cases: list[Case] = []
    errors: list[str] = []
    seen: set[str] = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            case = schema.model_validate_json(line)
        except ValidationError as error:
            details = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors())
            errors.append(f"  line {line_no}: {details}")
            continue
        if case.case_id in seen:
            errors.append(f"  line {line_no}: duplicate case_id {case.case_id}")
        seen.add(case.case_id)
        cases.append(case)
    if errors:
        raise DatasetError(f"{path.name} has {len(errors)} bad row(s):\n" + "\n".join(errors))
    return cases


def file_sha256(path: Path) -> str:
    # Hash with LF line endings so Windows and Linux checkouts agree.
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def read_manifest(data_dir: Path = DATA_DIR) -> dict:
    return json.loads((data_dir / "evals" / "MANIFEST.json").read_text(encoding="utf-8"))


def check_data(data_dir: Path = DATA_DIR) -> list[str]:
    """Return every problem found: invalid rows, wrong case counts, changed files."""
    problems = []
    manifest = read_manifest(data_dir)
    for name in CASE_TYPES:
        try:
            count = len(load_dataset(name, data_dir))
        except DatasetError as error:
            problems.append(str(error))
            continue
        if manifest["case_counts"].get(name) != count:
            problems.append(f"{name}: {count} cases, manifest says {manifest['case_counts'].get(name)}")
    for relative in TRACKED_FILES:
        actual = file_sha256(data_dir / relative)
        if manifest["files"].get(relative) != actual:
            problems.append(f"{relative}: contents changed since manifest version {manifest['version']}")
    if problems:
        problems.append("If the change is intended, run update-manifest with a new version and a note.")
    return problems


def update_manifest(version: str, note: str, data_dir: Path = DATA_DIR) -> dict:
    """Record the current hashes under a new version, after an intended data change."""
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise DatasetError(f"version must look like 1.2.0, got {version!r}")
    path = data_dir / "evals" / "MANIFEST.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"changelog": []}
    if old.get("version") == version:
        raise DatasetError(f"manifest is already at {version}; bump the version for a data change")
    manifest = {
        "version": version,
        "files": {relative: file_sha256(data_dir / relative) for relative in TRACKED_FILES},
        "case_counts": {name: len(load_dataset(name, data_dir)) for name in CASE_TYPES},
        "changelog": old["changelog"] + [{"version": version, "date": str(date.today()), "note": note}],
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
