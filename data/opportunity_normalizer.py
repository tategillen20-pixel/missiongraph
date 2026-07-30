"""Normalize raw SAM.gov opportunities for MissionGraph."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def _source_url(record: dict[str, Any]) -> str:
    """Return SAM.gov's original UI or self URL when available."""
    ui_link = record.get("uiLink")
    if isinstance(ui_link, str) and ui_link.strip():
        return ui_link

    links = record.get("links")
    if isinstance(links, list):
        for link in links:
            if (
                isinstance(link, dict)
                and str(link.get("rel", "")).casefold() == "self"
                and isinstance(link.get("href"), str)
            ):
                return link["href"]
    return ""


def normalize_opportunity(
    record: dict[str, Any],
    description_text: str = "",
) -> dict[str, Any]:
    """Convert one raw SAM.gov opportunity into the internal schema.

    Args:
        record: One object from SAM.gov's ``opportunitiesData`` list.
        description_text: Text retrieved from the record's description URL.

    Returns:
        A normalized opportunity with direct-evidence metadata.

    Raises:
        ValueError: If the record has no SAM.gov notice identifier.
    """
    notice_id = str(record.get("noticeId", "")).strip()

    if not notice_id:
        raise ValueError("Opportunity is missing its notice ID.")

    parent_path = str(record.get("fullParentPathName", "")).strip()
    parent_parts = [part.strip() for part in parent_path.split(".") if part.strip()]

    response_deadline = (
        record.get("responseDeadLine") or record.get("reponseDeadLine") or ""
    )

    point_of_contact = record.get("pointOfContact") or []
    description_url = record.get("description")
    source_url = _source_url(record)

    return {
        "opportunity_id": notice_id,
        "notice_id": notice_id,
        "title": str(record.get("title", "")).strip(),
        "solicitation_number": str(record.get("solicitationNumber", "")).strip(),
        "organization_path": parent_path,
        "organization_id": str(
            record.get("fullParentPathCode", "")
        ).strip(),
        "department": parent_parts[0] if parent_parts else "",
        "issuing_office": parent_parts[-1] if parent_parts else "",
        "posted_date": record.get("postedDate"),
        "response_deadline": response_deadline,
        "opportunity_type": record.get("type"),
        "base_type": record.get("baseType"),
        "naics_code": record.get("naicsCode"),
        "classification_code": record.get("classificationCode"),
        "set_aside": (
            record.get("typeOfSetAsideDescription")
            or record.get("setAside")
            or ""
        ),
        "active": record.get("active"),
        "description": description_text.strip(),
        "description_url": description_url,
        "original_description_url": description_url,
        "source_url": source_url,
        "sam_url": source_url,
        "point_of_contact": point_of_contact,
        "place_of_performance": record.get("placeOfPerformance"),
        "source": "SAM.gov",
        "source_record_id": notice_id,
        "evidence_type": "direct",
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
    }
