"""Convert raw USAspending awards into MissionGraph's internal format."""

from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
import math
from typing import Any


def _organization_name(value: Any) -> str:
    """Return a trimmed organization name, or an empty string when missing."""
    return "" if value is None else str(value).strip()


def _award_amount(value: Any) -> float:
    """Convert an API award amount to a finite floating-point value."""
    if isinstance(value, bool) or value is None:
        raise ValueError("Award Amount must be a numeric value.")

    try:
        amount = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Award Amount must be a numeric value.") from exc

    if not math.isfinite(amount):
        raise ValueError("Award Amount must be a finite numeric value.")
    return amount


def normalize_award_records(
    raw_records: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Normalize USAspending award records without mutating the source data.

    Organization names are stripped of surrounding whitespace, while the
    recipient's unmodified name is retained in ``original_recipient_name``.
    Records that cannot be normalized are returned in the validation error
    list rather than being silently omitted.

    Args:
        raw_records: Raw result dictionaries from the USAspending API.

    Returns:
        A pair containing normalized records and validation errors. Each
        validation error includes the input index, all detected messages, and
        the original record.
    """
    valid_records: list[dict[str, Any]] = []
    validation_errors: list[dict[str, Any]] = []
    retrieved_at = datetime.now(timezone.utc).isoformat()

    for index, raw_record in enumerate(raw_records):
        if not isinstance(raw_record, Mapping):
            validation_errors.append(
                {
                    "index": index,
                    "errors": ["Record must be a dictionary-like object."],
                    "record": raw_record,
                }
            )
            continue

        errors: list[str] = []
        source_award_id = raw_record.get("Award ID")
        if source_award_id is None or not str(source_award_id).strip():
            errors.append("Award ID is required.")

        original_recipient_name = raw_record.get("Recipient Name")
        recipient_name = _organization_name(original_recipient_name)
        if not recipient_name:
            errors.append("Recipient Name must not be empty.")

        try:
            award_amount = _award_amount(raw_record.get("Award Amount"))
        except ValueError as exc:
            errors.append(str(exc))
            award_amount = None

        if errors:
            validation_errors.append(
                {
                    "index": index,
                    "errors": errors,
                    "record": raw_record,
                }
            )
            continue

        description = raw_record.get("Description")
        generated_internal_id = raw_record.get("generated_internal_id")
        source_url = (
            "https://www.usaspending.gov/award/"
            f"{generated_internal_id}/"
            if generated_internal_id
            else "https://www.usaspending.gov/"
        )
        valid_records.append(
            {
                "source_award_id": source_award_id,
                "generated_internal_id": generated_internal_id,
                "source_url": source_url,
                "recipient_name": recipient_name,
                "original_recipient_name": original_recipient_name,
                "award_amount": award_amount,
                "original_award_amount": raw_record.get("Award Amount"),
                "awarding_agency": _organization_name(
                    raw_record.get("Awarding Agency")
                ),
                "awarding_sub_agency": _organization_name(
                    raw_record.get("Awarding Sub Agency")
                ),
                "start_date": raw_record.get("Start Date"),
                "end_date": raw_record.get("End Date"),
                "original_start_date": raw_record.get("Start Date"),
                "original_end_date": raw_record.get("End Date"),
                "description": "" if description is None else str(description),
                "original_award_description": description,
                "source": "USAspending",
                "evidence_type": "direct",
                "retrieved_at": retrieved_at,
            }
        )

    return valid_records, validation_errors
