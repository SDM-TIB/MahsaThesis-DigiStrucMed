"""Quoted, atomic CSV exports with streaming validation of records and labels."""
import csv
import json
import os
import shutil
from pathlib import Path
import tempfile


def audit_dataset_csv(path, *, expected_frame=None):
    """Check record structure and labels, optionally verifying source text verbatim."""
    csv.field_size_limit(2**31 - 1)
    counts = {"rows": 0, "yes": 0, "no": 0}
    with open(path, encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        headers = next(reader)
        label_index = headers.index("Label")
        if len(headers) != len(set(headers)):
            raise ValueError(f"{path}: duplicate column names")
        expected_rows = None
        if expected_frame is not None:
            if headers != list(expected_frame.columns):
                raise ValueError(f"{path}: CSV round-trip changed column names")
            expected_rows = expected_frame.itertuples(index=False, name=None)
        for number, row in enumerate(reader, 2):
            if len(row) != len(headers):
                raise ValueError(f"{path}: record {number} has {len(row)} fields; expected {len(headers)}")
            label = row[label_index]
            if label not in ("0", "1"):
                raise ValueError(f"{path}: record {number} has invalid Label {label!r}")
            if expected_rows is not None:
                expected = next(expected_rows, None)
                if expected is None:
                    raise ValueError(f"{path}: CSV round-trip added records")
                for column, value, original in zip(headers, row, expected):
                    if isinstance(original, str) and value != original:
                        raise ValueError(
                            f"{path}: record {number}, column {column!r}: "
                            "CSV round-trip changed text"
                        )
            counts["rows"] += 1
            counts["yes" if label == "1" else "no"] += 1
        if expected_rows is not None and next(expected_rows, None) is not None:
            raise ValueError(f"{path}: CSV round-trip lost records")
    counts["columns"] = headers
    return counts


def write_dataset_csv(frame, path):
    """Preserve commas/newlines, validate the completed file, then replace target."""
    path = Path(path)
    if "Label" not in frame or not frame["Label"].isin([0, 1]).all():
        raise ValueError(f"{path}: all records must have a binary Label")
    expected = {"rows": len(frame), "yes": int((frame["Label"] == 1).sum()),
                "no": int((frame["Label"] == 0).sum())}
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8",
                     quoting=csv.QUOTE_ALL, doublequote=True, lineterminator="\n")
        # Counts alone cannot detect shifted or altered text fields. Compare
        # parsed strings with their source, including quotes, commas and newlines.
        actual = audit_dataset_csv(temporary, expected_frame=frame)
        if any(actual[key] != value for key, value in expected.items()):
            raise ValueError(f"{path}: CSV round-trip changed row or label counts")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    with open(str(path) + ".audit.json", "w", encoding="utf-8") as stream:
        json.dump(actual, stream, indent=2)
    print(f"  Verified {path.name}: {actual['rows']:,} records, "
          f"yes={actual['yes']:,}, no={actual['no']:,}")
    return actual


def copy_dataset_csv(source, destination):
    """Validate before restoring/backing up; publish only a complete copy."""
    audit_dataset_csv(source)
    destination = Path(destination)
    descriptor, temporary = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=destination.parent)
    os.close(descriptor)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
