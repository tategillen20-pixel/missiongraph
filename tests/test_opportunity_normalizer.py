"""Tests for SAM.gov normalization and opportunity graph construction."""

from datetime import datetime

from data.opportunity_normalizer import normalize_opportunity
from graph.builder import (
    add_extracted_capabilities_to_graph,
    build_opportunity_graph,
)


def test_normalizes_direct_samgov_evidence() -> None:
    """Normalization preserves identifiers, URLs, and evidence metadata."""
    raw_record = {
        "noticeId": "NOTICE-123",
        "title": "Cybersecurity support",
        "solicitationNumber": "SOL-456",
        "fullParentPathName": "DEPARTMENT OF DEFENSE.ARMY.OFFICE",
        "fullParentPathCode": "DOD.ARMY.OFFICE",
        "postedDate": "2025-02-03 10:00:00",
        "reponseDeadLine": "2025-03-04 17:00:00",
        "type": "Solicitation",
        "active": "Yes",
        "description": "https://api.sam.gov/description/NOTICE-123",
        "uiLink": "https://sam.gov/opp/NOTICE-123/view",
    }

    opportunity = normalize_opportunity(raw_record, "<p>Original text</p>")

    assert opportunity["notice_id"] == "NOTICE-123"
    assert opportunity["source_record_id"] == "NOTICE-123"
    assert opportunity["source_url"] == "https://sam.gov/opp/NOTICE-123/view"
    assert opportunity["original_description_url"] == (
        "https://api.sam.gov/description/NOTICE-123"
    )
    assert opportunity["evidence_type"] == "direct"
    assert opportunity["source"] == "SAM.gov"
    retrieved_at = datetime.fromisoformat(opportunity["retrieved_at"])
    assert retrieved_at.utcoffset().total_seconds() == 0


def test_builds_agency_published_opportunity_relationship() -> None:
    """A normalized record creates a direct PUBLISHED evidence edge."""
    opportunity = normalize_opportunity(
        {
            "noticeId": "NOTICE-123",
            "title": "Cybersecurity support",
            "fullParentPathName": "DEPARTMENT OF DEFENSE.ARMY.OFFICE",
            "fullParentPathCode": "DOD.ARMY.OFFICE",
            "uiLink": "https://sam.gov/opp/NOTICE-123/view",
        }
    )

    graph = build_opportunity_graph([opportunity])

    assert graph.number_of_nodes() == 2
    assert graph.number_of_edges() == 1
    _, _, edge = next(iter(graph.edges(data=True)))
    assert edge["relationship_type"] == "PUBLISHED"
    assert edge["source_record_id"] == "NOTICE-123"
    assert edge["evidence_type"] == "direct"
    assert edge["source_url"] == "https://sam.gov/opp/NOTICE-123/view"
    assert edge["retrieved_at"] == opportunity["retrieved_at"]


def test_adds_only_opportunity_requires_capability_relationship() -> None:
    """AI extraction adds capability evidence without adding companies."""
    opportunity = normalize_opportunity(
        {
            "noticeId": "NOTICE-123",
            "title": "Cybersecurity support",
            "fullParentPathName": "DEPARTMENT OF DEFENSE.ARMY.OFFICE",
            "fullParentPathCode": "DOD.ARMY.OFFICE",
        }
    )
    graph = build_opportunity_graph([opportunity])
    extraction = {
        "source_record_id": "NOTICE-123",
        "evidence_type": "ai_extracted",
        "model": "test-structured-model",
        "extracted_at": "2025-01-01T00:00:00+00:00",
        "capabilities": [
            {
                "name": "Zero Trust Architecture",
                "category": "cybersecurity",
                "evidence_quote": "provide zero trust architecture",
                "confidence": 0.91,
            }
        ],
    }

    add_extracted_capabilities_to_graph(graph, opportunity, extraction)

    capability_nodes = [
        data
        for _, data in graph.nodes(data=True)
        if data["node_type"] == "capability"
    ]
    company_nodes = [
        data
        for _, data in graph.nodes(data=True)
        if data["node_type"] == "company"
    ]
    capability_edges = [
        data
        for _, _, data in graph.edges(data=True)
        if data["relationship_type"] == "REQUIRES_CAPABILITY"
    ]
    assert len(capability_nodes) == 1
    assert company_nodes == []
    assert capability_edges == [
        {
            "relationship_type": "REQUIRES_CAPABILITY",
            "evidence_type": "ai_extracted",
            "evidence_quote": "provide zero trust architecture",
            "confidence": 0.91,
            "source": "SAM.gov",
            "source_record_id": "NOTICE-123",
            "model": "test-structured-model",
            "extracted_at": "2025-01-01T00:00:00+00:00",
        }
    ]
