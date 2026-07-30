"""Build an evidence-backed NetworkX graph from normalized awards."""
from __future__ import annotations 

from collections.abc import Iterable, Mapping
from typing import Any, Optional
from uuid import NAMESPACE_URL, uuid5

import networkx as nx


def add_extracted_capabilities_to_graph(
    graph: nx.DiGraph,
    opportunity: Mapping[str, Any],
    extraction: Mapping[str, Any],
) -> None:
    """Add only AI-extracted capability edges for one opportunity."""
    capabilities = extraction.get("capabilities", [])
    if not capabilities:
        return

    source_record_id = _external_identifier(
        opportunity,
        ("source_record_id", "opportunity_id", "notice_id"),
    )
    source = _external_identifier(opportunity, ("source",))
    extraction_record_id = _external_identifier(
        extraction, ("source_record_id", "notice_id")
    )
    model = _external_identifier(extraction, ("model",))
    extracted_at = _external_identifier(extraction, ("extracted_at",))
    if not all((source_record_id, source, model, extracted_at)):
        raise ValueError(
            "Capability graph data requires source record ID, source, model, "
            "and extraction time."
        )
    if extraction_record_id != source_record_id:
        raise ValueError(
            "Capability extraction source record ID does not match opportunity."
        )

    opportunity_node_id = _opportunity_node_id(source, source_record_id)
    if opportunity_node_id not in graph:
        raise ValueError(
            "Opportunity node must exist before adding extracted capabilities."
        )

    for capability in capabilities:
        capability_name = str(capability["name"]).strip()
        category = str(capability["category"]).strip()
        evidence_quote = str(capability["evidence_quote"])
        confidence = float(capability["confidence"])
        if not capability_name or not evidence_quote:
            continue
        capability_id = _entity_node_id(
            "capability",
            "MissionGraph",
            capability_name,
            f"{category}:{capability_name.casefold()}",
        )

        graph.add_node(
            capability_id,
            node_type="capability",
            name=capability_name,
            label=capability_name,
            category=category,
        )

        graph.add_edge(
            opportunity_node_id,
            capability_id,
            relationship_type="REQUIRES_CAPABILITY",
            evidence_type="ai_extracted",
            evidence_quote=evidence_quote,
            confidence=confidence,
            source=source,
            source_record_id=source_record_id,
            model=model,
            extracted_at=extracted_at,
        )

def _external_identifier(
    record: Mapping[str, Any], field_names: tuple[str, ...]
) -> Optional[str]:
    """Return the first non-empty external identifier in a record."""
    for field_name in field_names:
        value = record.get(field_name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _entity_node_id(
    node_type: str,
    source: str,
    display_name: str,
    external_id: str | None = None,
) -> str:
    """Create a stable, namespaced node ID for an organization."""
    identity = external_id or display_name.casefold()
    stable_uuid = uuid5(
        NAMESPACE_URL,
        f"missiongraph:{source.casefold()}:{node_type}:{identity}",
    )
    return f"{node_type}:{stable_uuid}"


def _opportunity_node_id(source: str, source_record_id: str) -> str:
    """Return the canonical node ID for a SAM.gov opportunity."""
    return f"opportunity:{source.casefold()}:{source_record_id}"


def build_award_graph(
    award_records: Iterable[Mapping[str, Any]],
) -> nx.DiGraph:
    """Create a directed graph from normalized USAspending award records.

    Only direct relationships represented by the normalized source records are
    created: an agency issues an award and that award is awarded to a company.
    External entity identifiers are preferred when supplied. Otherwise,
    deterministic UUIDs scoped by source and node type provide stable fallback
    IDs without using display labels directly as graph keys.

    Args:
        award_records: Valid records produced by
            :func:`data.normalizer.normalize_award_records`.

    Returns:
        A directed NetworkX graph containing agency, award, and company nodes.

    Raises:
        ValueError: If a record lacks an award ID, agency name, recipient name,
            source, or evidence type.
    """
    graph = nx.DiGraph()

    for record in award_records:
        source_record_id = _external_identifier(
            record, ("source_award_id", "generated_internal_id")
        )
        source = _external_identifier(record, ("source",))
        evidence_type = _external_identifier(record, ("evidence_type",))
        agency_name = _external_identifier(
            record, ("awarding_sub_agency", "awarding_agency")
        )
        company_name = _external_identifier(record, ("recipient_name",))

        missing_fields = [
            name
            for name, value in (
                ("source_award_id", source_record_id),
                ("source", source),
                ("evidence_type", evidence_type),
                ("awarding agency", agency_name),
                ("recipient_name", company_name),
            )
            if value is None
        ]
        if missing_fields:
            raise ValueError(
                "Cannot build graph; missing required fields: "
                + ", ".join(missing_fields)
            )

        agency_external_id = _external_identifier(
            record,
            (
                "awarding_sub_agency_id",
                "awarding_sub_agency_code",
                "awarding_agency_id",
                "awarding_agency_code",
            ),
        )
        company_external_id = _external_identifier(
            record, ("recipient_id", "recipient_uei")
        )

        agency_id = _entity_node_id(
            "agency",
            source,
            agency_name,
            agency_external_id,
        )
        award_id = f"award:{source.casefold()}:{source_record_id}"
        company_id = _entity_node_id(
            "company",
            source,
            company_name,
            company_external_id,
        )

        graph.add_node(
            agency_id,
            node_type="agency",
            name=agency_name,
            external_id=agency_external_id,
        )
        graph.add_node(
            award_id,
            node_type="award",
            source_record_id=source_record_id,
            award_amount=record.get("award_amount"),
            description=record.get("description", ""),
            start_date=record.get("start_date"),
            end_date=record.get("end_date"),
        )
        graph.add_node(
            company_id,
            node_type="company",
            name=company_name,
            external_id=company_external_id,
        )

        evidence = {
            "source_record_id": source_record_id,
            "evidence_type": evidence_type,
        }
        graph.add_edge(
            agency_id,
            award_id,
            relationship_type="ISSUED",
            **evidence,
        )
        graph.add_edge(
            award_id,
            company_id,
            relationship_type="AWARDED_TO",
            **evidence,
        )

    return graph

def add_opportunity_to_graph(
    graph: nx.DiGraph,
    opportunity: Mapping[str, Any],
) -> None:
    """Add one normalized SAM.gov opportunity to an existing graph."""

    source_record_id = _external_identifier(
        opportunity,
        (
            "source_record_id",
            "opportunity_id",
            "notice_id",
        ),
    )
    source = _external_identifier(
        opportunity,
        ("source",),
    )
    evidence_type = _external_identifier(
        opportunity,
        ("evidence_type",),
    )
    agency_name = _external_identifier(
        opportunity,
        (
            "issuing_office",
            "department",
            "organization_path",
        ),
    )
    opportunity_title = _external_identifier(
        opportunity,
        ("title",),
    )

    missing_fields = [
        name
        for name, value in (
            ("source_record_id", source_record_id),
            ("source", source),
            ("evidence_type", evidence_type),
            ("issuing organization", agency_name),
            ("title", opportunity_title),
        )
        if value is None
    ]

    if missing_fields:
        raise ValueError(
            "Cannot add opportunity to graph; missing required fields: "
            + ", ".join(missing_fields)
        )

    # The organization path helps distinguish similarly named offices.
    agency_external_id = _external_identifier(
        opportunity,
        (
            "organization_id",
            "organization_path",
        ),
    )

    agency_id = _entity_node_id(
        node_type="agency",
        source=source,
        display_name=agency_name,
        external_id=agency_external_id,
    )

    opportunity_id = _opportunity_node_id(source, source_record_id)

    graph.add_node(
        agency_id,
        node_type="agency",
        name=agency_name,
        label=agency_name,
        external_id=agency_external_id,
    )

    graph.add_node(
        opportunity_id,
        node_type="opportunity",
        name=opportunity_title,
        label=opportunity_title,
        source_record_id=source_record_id,
        solicitation_number=opportunity.get(
            "solicitation_number",
            "",
        ),
        description=opportunity.get("description", ""),
        posted_date=opportunity.get("posted_date"),
        response_deadline=opportunity.get(
            "response_deadline",
        ),
        opportunity_type=opportunity.get(
            "opportunity_type",
        ),
        naics_code=opportunity.get("naics_code"),
        set_aside=opportunity.get("set_aside", ""),
        source=source,
        source_url=opportunity.get(
            "source_url",
            opportunity.get("sam_url", ""),
        ),
        retrieved_at=opportunity.get("retrieved_at"),
    )

    graph.add_edge(
        agency_id,
        opportunity_id,
        relationship_type="PUBLISHED",
        source=source,
        source_record_id=source_record_id,
        evidence_type=evidence_type,
        source_url=opportunity.get(
            "source_url",
            opportunity.get("sam_url", ""),
        ),
        retrieved_at=opportunity.get("retrieved_at"),
    )


def build_opportunity_graph(
    opportunity_records: Iterable[Mapping[str, Any]],
) -> nx.DiGraph:
    """Create a directed graph from normalized SAM.gov opportunities."""

    graph = nx.DiGraph()

    for opportunity in opportunity_records:
        add_opportunity_to_graph(
            graph,
            opportunity,
        )

    return graph


def build_mission_graph(
    award_records: Iterable[Mapping[str, Any]],
    opportunity_records: Iterable[Mapping[str, Any]],
) -> nx.DiGraph:
    """Build one graph containing awards and opportunities."""

    graph = build_award_graph(award_records)

    for opportunity in opportunity_records:
        add_opportunity_to_graph(
            graph,
            opportunity,
        )

    return graph
