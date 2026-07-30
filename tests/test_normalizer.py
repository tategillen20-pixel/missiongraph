"""Tests for award data normalization."""

from datetime import datetime

from data.normalizer import normalize_award_records


def test_normalizes_complete_record() -> None:
    """A complete raw award is converted without losing source identifiers."""
    raw_record = {
        "Award ID": "W91TEST-25-C-0001",
        "generated_internal_id": "CONT_AWD_W91TEST25C0001_9700",
        "Recipient Name": "  Example Systems, Inc.  ",
        "Award Amount": "125000.50",
        "Awarding Agency": "  Department of Defense  ",
        "Awarding Sub Agency": " Department of the Army ",
        "Start Date": "2025-01-01",
        "End Date": "2025-12-31",
        "Description": "Test support services",
    }

    records, errors = normalize_award_records([raw_record])

    assert errors == []
    assert len(records) == 1
    record = records[0]
    assert record["source_award_id"] == "W91TEST-25-C-0001"
    assert record["generated_internal_id"] == (
        "CONT_AWD_W91TEST25C0001_9700"
    )
    assert record["recipient_name"] == "Example Systems, Inc."
    assert record["original_recipient_name"] == (
        "  Example Systems, Inc.  "
    )
    assert record["award_amount"] == 125000.50
    assert record["original_award_amount"] == "125000.50"
    assert record["awarding_agency"] == "Department of Defense"
    assert record["awarding_sub_agency"] == "Department of the Army"
    assert record["description"] == "Test support services"
    assert record["original_award_description"] == "Test support services"
    assert record["original_start_date"] == "2025-01-01"
    assert record["original_end_date"] == "2025-12-31"
    assert record["source"] == "USAspending"
    assert record["evidence_type"] == "direct"
    retrieved_at = datetime.fromisoformat(record["retrieved_at"])
    assert retrieved_at.utcoffset().total_seconds() == 0
    assert raw_record["Recipient Name"] == "  Example Systems, Inc.  "


def test_normalizes_record_with_missing_optional_fields() -> None:
    """Missing optional values receive stable defaults."""
    raw_record = {
        "Award ID": "W91TEST-25-C-0002",
        "Recipient Name": "Example Labs",
        "Award Amount": 50,
    }

    records, errors = normalize_award_records([raw_record])

    assert errors == []
    assert records[0]["award_amount"] == 50.0
    assert records[0]["original_award_amount"] == 50
    assert records[0]["description"] == ""
    assert records[0]["original_award_description"] is None
    assert records[0]["awarding_agency"] == ""
    assert records[0]["awarding_sub_agency"] == ""
    assert records[0]["generated_internal_id"] is None
    assert records[0]["start_date"] is None
    assert records[0]["end_date"] is None


def test_returns_validation_error_for_invalid_amount() -> None:
    """An invalid amount is reported along with its source record."""
    raw_record = {
        "Award ID": "W91TEST-25-C-0003",
        "Recipient Name": "Example Analytics",
        "Award Amount": "not-a-number",
    }

    records, errors = normalize_award_records([raw_record])

    assert records == []
    assert errors == [
        {
            "index": 0,
            "errors": ["Award Amount must be a numeric value."],
            "record": raw_record,
        }
    ]


def test_returns_validation_error_for_empty_recipient_name() -> None:
    """Whitespace-only recipient names are invalid after normalization."""
    raw_record = {
        "Award ID": "W91TEST-25-C-0004",
        "Recipient Name": "   ",
        "Award Amount": 1000,
    }

    records, errors = normalize_award_records([raw_record])

    assert records == []
    assert errors == [
        {
            "index": 0,
            "errors": ["Recipient Name must not be empty."],
            "record": raw_record,
        }
    ]
