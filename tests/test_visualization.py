"""Tests for the restrained relationship graph renderer."""

import networkx as nx
import pytest

from graph.visualization import graph_to_dot


def test_graph_to_dot_distinguishes_direct_and_ai_edges() -> None:
    """Direct and AI-extracted relationships receive different styles."""
    graph = nx.DiGraph()
    graph.add_node("agency:1", node_type="agency", name="Army")
    graph.add_node("opportunity:1", node_type="opportunity", name="Notice")
    graph.add_node("capability:1", node_type="capability", name="Autonomy")
    graph.add_edge(
        "agency:1",
        "opportunity:1",
        relationship_type="PUBLISHED",
        evidence_type="direct",
    )
    graph.add_edge(
        "opportunity:1",
        "capability:1",
        relationship_type="REQUIRES_CAPABILITY",
        evidence_type="ai_extracted",
    )

    dot = graph_to_dot(graph)

    assert 'label="Published", style="solid"' in dot
    assert 'label="Requires capability", style="dashed"' in dot
    assert "agency:1" in dot
    assert "Autonomy" in dot


def test_graph_to_dot_wraps_long_node_labels() -> None:
    """Long labels are wrapped and capped before Graphviz sizes the node."""
    graph = nx.DiGraph()
    graph.add_node(
        "opportunity:1",
        node_type="opportunity",
        name=(
            "Enterprise software engineering and professional services "
            "integration support requirement"
        ),
    )

    dot = graph_to_dot(graph)

    assert "\\n" in dot
    assert "…" in dot


@pytest.mark.parametrize("capability_count", [1, 3])
def test_opportunity_layout_is_compact_and_preserves_full_title_metadata(
    capability_count: int,
) -> None:
    """The SAM.gov layout shortens display text but retains the source title."""
    graph = nx.DiGraph()
    full_title = (
        "Omnicell maintenance and upgrades for multiple defense health "
        "facilities and associated support services"
    )
    graph.add_node("agency:1", node_type="agency", name="Defense Health Agency")
    graph.add_node(
        "opportunity:1",
        node_type="opportunity",
        name=full_title,
        source_record_id="notice-1",
    )
    capabilities = [
        "Systems integration",
        "Cybersecurity engineering",
        "Sustainment",
    ]
    for index, capability in enumerate(capabilities[:capability_count]):
        graph.add_node(
            f"capability:{index}",
            node_type="capability",
            name=capability,
        )
        graph.add_edge(
            "opportunity:1",
            f"capability:{index}",
            relationship_type="REQUIRES_CAPABILITY",
            evidence_type="ai_extracted",
        )
    graph.add_edge(
        "agency:1",
        "opportunity:1",
        relationship_type="PUBLISHED",
        evidence_type="direct",
    )

    dot = graph_to_dot(graph, opportunity_layout=True)

    assert 'pad="0.08", nodesep="0.20", ranksep="0.42"' in dot
    assert 'width="1.45", height="0.36"' in dot
    assert "Opportunity · Omnicell" in dot
    assert f'tooltip="{full_title}"' in dot
    assert "{ rank=same;" in dot
    assert 'label="Published", style="solid"' in dot
    assert 'label="Requires capability", style="dashed"' in dot
