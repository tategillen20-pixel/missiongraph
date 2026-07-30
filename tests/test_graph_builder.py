"""Tests for graph construction."""

from graph.builder import build_award_graph


def _award(
    award_id: str = "W91TEST-25-C-0001",
    recipient_name: str = "Example Systems, Inc.",
) -> dict:
    """Create a minimal normalized award record for graph tests."""
    return {
        "source_award_id": award_id,
        "recipient_name": recipient_name,
        "award_amount": 1000.0,
        "awarding_agency": "Department of Defense",
        "awarding_sub_agency": "Department of the Army",
        "description": "Test services",
        "start_date": "2025-01-01",
        "end_date": "2025-12-31",
        "source": "USAspending",
        "evidence_type": "direct",
    }


def test_one_award_creates_three_nodes() -> None:
    """An award creates one agency, one award, and one company node."""
    graph = build_award_graph([_award()])

    assert graph.number_of_nodes() == 3
    assert {attributes["node_type"] for _, attributes in graph.nodes(data=True)} == {
        "agency",
        "award",
        "company",
    }


def test_one_award_creates_two_directed_edges() -> None:
    """An award creates the two required directed relationships."""
    graph = build_award_graph([_award()])

    assert graph.number_of_edges() == 2
    assert {
        attributes["relationship_type"]
        for _, _, attributes in graph.edges(data=True)
    } == {"ISSUED", "AWARDED_TO"}


def test_repeated_company_names_share_one_company_node() -> None:
    """Repeated canonical company names do not create duplicate companies."""
    graph = build_award_graph(
        [
            _award("W91TEST-25-C-0001", "Example Systems, Inc."),
            _award("W91TEST-25-C-0002", "Example Systems, Inc."),
        ]
    )

    company_nodes = [
        node_id
        for node_id, attributes in graph.nodes(data=True)
        if attributes["node_type"] == "company"
    ]
    assert len(company_nodes) == 1
    assert graph.number_of_nodes() == 4


def test_every_edge_contains_evidence_information() -> None:
    """Every graph relationship points back to direct source evidence."""
    graph = build_award_graph([_award()])

    for _, _, attributes in graph.edges(data=True):
        assert attributes["source_record_id"] == "W91TEST-25-C-0001"
        assert attributes["evidence_type"] == "direct"
