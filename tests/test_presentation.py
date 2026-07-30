"""Tests for final presentation formatting helpers."""

from app import (
    _award_dropdown_label,
    _build_sam_graph_demo,
    _confidence_label,
    _format_currency,
    _format_date,
    _opportunity_dropdown_label,
    _shorten,
    _why_award_matched,
    _why_opportunity_matched,
    truncate_at_word_boundary,
)


def test_formats_currency_and_dates_for_display() -> None:
    """Raw numeric and ISO values become readable presentation strings."""
    assert _format_currency("1234567.5") == "$1,234,567.50"
    assert _format_date("2026-07-29", False) == "Jul 29, 2026"
    assert _format_date(
        "2026-07-30T13:00:00-04:00",
        None,
    ).startswith("Jul 30, 2026 at 1:00 PM")


def test_confidence_uses_qualitative_bands() -> None:
    """Numeric model scores map to cautious human-readable bands."""
    assert _confidence_label(0.90) == "High"
    assert _confidence_label(0.70) == "Moderate"
    assert _confidence_label(0.30) == "Low"


def test_table_description_preview_is_shortened() -> None:
    """Tables receive a preview while details can retain complete text."""
    description = "A" * 200

    preview = _shorten(description, limit=40)

    assert len(preview) == 40
    assert preview.endswith("…")


def test_match_explanations_use_visible_fields_and_ai_classification() -> None:
    """Match context identifies direct fields before AI classification."""
    award = {
        "description": "Cybersecurity engineering support",
        "awarding_agency": "Department of Defense",
        "awarding_sub_agency": "Department of the Army",
    }
    opportunity = {
        "title": "Enterprise software support",
        "description": "",
        "issuing_office": "Army Contracting Command",
        "department": "Department of Defense",
        "organization_path": "",
    }
    extraction = {
        "capabilities": [
            {"name": "Digital engineering", "category": "digital_engineering"}
        ]
    }

    assert _why_award_matched(award, "cybersecurity") == "Description"
    assert _why_opportunity_matched(
        opportunity, "software", extraction
    ) == "Title"
    assert _why_opportunity_matched(
        opportunity, "digital", extraction
    ) == "AI classification"


def test_dropdown_labels_hide_ids_and_truncate_at_word_boundaries() -> None:
    """Choice labels remain readable without exposing stable source IDs."""
    award = {
        "source_award_id": "CONT_AWD_123456789",
        "recipient_name": "EMPOWER AI, INC.",
        "description": "Cybersecurity requirements and engineering support",
        "awarding_sub_agency": "Army Contracting Command",
    }
    opportunity = {
        "source_record_id": "e467249347954101b4c0da45abdefd3f",
        "title": (
            "Enterprise software engineering and professional services "
            "integration support for mission operations worldwide"
        ),
        "issuing_office": "Defense Health Agency HCD West",
    }

    award_label = _award_dropdown_label(award, {"empower ai, inc."})
    opportunity_label = _opportunity_dropdown_label(opportunity)

    assert award_label.startswith("EMPOWER AI, INC. — Cybersecurity")
    assert award["source_award_id"] not in award_label
    assert opportunity["source_record_id"] not in opportunity_label
    assert opportunity_label.endswith("— Defense Health Agency HCD West")
    visible_title = opportunity_label.split(" — ", 1)[0]
    assert len(visible_title) <= 75
    assert visible_title.endswith("…")
    assert truncate_at_word_boundary("alpha beta gamma", 12) == "alpha beta…"


def test_unique_award_contractor_does_not_need_extra_context() -> None:
    """A unique contractor gets the shortest useful dropdown label."""
    record = {
        "recipient_name": "Example Systems LLC",
        "description": "Software engineering",
    }

    assert _award_dropdown_label(record, set()) == "Example Systems LLC"


def test_sam_graph_demo_is_local_and_supports_multiple_capabilities() -> None:
    """The visual demo uses local records and the production graph model."""
    graph = _build_sam_graph_demo(6)

    node_types = [
        attributes["node_type"] for _, attributes in graph.nodes(data=True)
    ]
    assert node_types.count("agency") == 1
    assert node_types.count("opportunity") == 1
    assert node_types.count("capability") == 6
    assert graph.number_of_edges() == 7
    assert all(
        attributes["source_record_id"] == "local-demo-opportunity"
        for _, _, attributes in graph.edges(data=True)
    )
