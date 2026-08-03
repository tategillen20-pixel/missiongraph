"""Render MissionGraph NetworkX graphs with restrained Graphviz styling."""

from __future__ import annotations

import textwrap
from typing import Any

import networkx as nx

NODE_STYLES = {
    "agency": ("box", "#18324A", "#FFFFFF"),
    "award": ("box", "#415A77", "#FFFFFF"),
    "company": ("ellipse", "#067647", "#FFFFFF"),
    "opportunity": ("box", "#415A77", "#FFFFFF"),
    "capability": ("ellipse", "#B54708", "#FFFFFF"),
}

RELATIONSHIP_LABELS = {
    "ISSUED": "Issued",
    "AWARDED_TO": "Awarded to",
    "PUBLISHED": "Published",
    "REQUIRES_CAPABILITY": "Requires capability",
}


def _dot_escape(value: Any) -> str:
    """Escape a value for a quoted Graphviz attribute."""
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
    )


def _wrapped_label(
    value: Any,
    width: int = 28,
    max_lines: int = 3,
    max_characters: int | None = None,
) -> str:
    """Wrap and cap long graph labels to keep nodes readable."""
    text = " ".join(str(value).split())
    if max_characters is not None and len(text) > max_characters:
        candidate = text[: max(1, max_characters - 1)]
        if candidate and not candidate[-1].isspace():
            candidate = candidate.rsplit(" ", 1)[0]
        candidate = candidate.rstrip()
        text = (candidate or text[: max_characters - 1].rstrip()) + "…"
    lines = textwrap.wrap(
        text,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    ) or [""]
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][: max(1, width - 1)].rstrip() + "…"
    return "\n".join(lines)


def _node_label(
    node_id: str,
    attributes: dict[str, Any],
    *,
    opportunity_layout: bool = False,
) -> str:
    """Return a concise display label without exposing internal node IDs."""
    node_type = attributes.get("node_type", "entity")
    label = (
        attributes.get("label")
        or attributes.get("name")
        or attributes.get("source_record_id")
        or node_id
    )
    prefix = f"{str(node_type).title()} · "
    if opportunity_layout and node_type == "opportunity":
        short_title = _wrapped_label(
            label,
            width=25,
            max_lines=2,
            max_characters=45,
        )
        return prefix + short_title
    width = 23 if opportunity_layout else 28
    return _wrapped_label(
        f"{prefix}{label}",
        width=width,
        max_lines=3,
        max_characters=68 if opportunity_layout else None,
    )


def graph_to_dot(
    graph: nx.DiGraph,
    *,
    opportunity_layout: bool = False,
) -> str:
    """Convert a directed MissionGraph graph into styled Graphviz DOT."""
    graph_spacing = (
        'pad="0.08", nodesep="0.20", ranksep="0.42"'
        if opportunity_layout
        else 'pad="0.25", nodesep="0.55", ranksep="0.8"'
    )
    default_node_size = (
        'fontsize="8.5", margin="0.07,0.04", width="1.75", height="0.42"'
        if opportunity_layout
        else 'fontsize="10", margin="0.14,0.08", width="2.45", height="0.72"'
    )
    edge_font_size = "8" if opportunity_layout else "9"
    lines = [
        "digraph MissionGraph {",
        f'  graph [rankdir="LR", bgcolor="transparent", {graph_spacing}];',
        f'  node [fontname="Arial", {default_node_size}, '
        'style="rounded,filled", penwidth="1.0", fixedsize="false"];',
        f'  edge [fontname="Arial", fontsize="{edge_font_size}", '
        'color="#667085", '
        'fontcolor="#415A77", arrowsize="0.7", penwidth="1.1"];',
    ]

    for node_id, attributes in graph.nodes(data=True):
        node_type = str(attributes.get("node_type", "entity"))
        shape, fillcolor, fontcolor = NODE_STYLES.get(
            node_type,
            ("box", "#6B7C8F", "#FFFFFF"),
        )
        label = _dot_escape(
            _node_label(
                str(node_id),
                attributes,
                opportunity_layout=opportunity_layout,
            )
        )
        escaped_id = _dot_escape(node_id)
        full_label = (
            attributes.get("label")
            or attributes.get("name")
            or attributes.get("source_record_id")
            or node_id
        )
        tooltip = _dot_escape(full_label)
        size = ""
        if opportunity_layout and node_type == "opportunity":
            size = ', width="2.25", height="0.48"'
        elif opportunity_layout and node_type == "capability":
            size = ', width="1.45", height="0.36", margin="0.06,0.03"'
        lines.append(
            f'  "{escaped_id}" [label="{label}", shape="{shape}", '
            f'fillcolor="{fillcolor}", color="{fillcolor}", '
            f'fontcolor="{fontcolor}", tooltip="{tooltip}"{size}];'
        )

    for source, target, attributes in graph.edges(data=True):
        relationship_type = str(
            attributes.get("relationship_type", "RELATED_TO")
        )
        relationship = _dot_escape(
            RELATIONSHIP_LABELS.get(
                relationship_type,
                relationship_type.replace("_", " ").title(),
            )
        )
        evidence_type = attributes.get("evidence_type")
        style = "dashed" if evidence_type == "ai_extracted" else "solid"
        color = "#B54708" if evidence_type == "ai_extracted" else "#667085"
        lines.append(
            f'  "{_dot_escape(source)}" -> "{_dot_escape(target)}" '
            f'[label="{relationship}", style="{style}", color="{color}"];'
        )

    if opportunity_layout:
        capability_ids = [
            _dot_escape(node_id)
            for node_id, attributes in graph.nodes(data=True)
            if attributes.get("node_type") == "capability"
        ]
        if capability_ids:
            members = "; ".join(f'"{node_id}"' for node_id in capability_ids)
            lines.append(f"  {{ rank=same; {members}; }}")

    lines.append("}")
    return "\n".join(lines)
