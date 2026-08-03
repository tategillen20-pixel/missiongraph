"""Professional Streamlit interface for MissionGraph."""

from __future__ import annotations

from collections import Counter
from datetime import date
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

st.set_page_config(
    page_title="MissionGraph",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Load local, git-ignored credentials while preserving any values already
# supplied by the deployment environment.
load_dotenv(override=False)

from ai.capability_extractor import (
    CapabilityExtractionError,
    extract_capabilities,
)
from data.normalizer import normalize_award_records
from data.opportunity_normalizer import normalize_opportunity
from graph.builder import (
    add_extracted_capabilities_to_graph,
    build_award_graph,
    build_opportunity_graph,
)
from graph.visualization import graph_to_dot
from services.samgov import (
    SamGovError,
    fetch_opportunity_description,
    search_opportunities,
)
from services.usaspending import USAspendingAPIError, search_contract_awards

PUBLIC_DATA_CACHE_TTL = 900


def inject_custom_css() -> None:
    """Apply MissionGraph's centralized, responsive visual system."""
    st.markdown(
        """
        <style>
        :root {
            --mg-navy: #18324A; --mg-blue-gray: #415A77;
            --mg-text: #17212B; --mg-secondary: #667085;
            --mg-muted: #98A2B3; --mg-border: #E4E7EC;
            --mg-surface: #FFFFFF; --mg-page: #F6F8FB;
        }
        .stApp { background: var(--mg-page); color: var(--mg-text); }
        .block-container {
            max-width: 1480px; padding: 3.75rem 2rem 2.5rem;
        }
        h1, h2, h3 { color: var(--mg-navy); letter-spacing: -0.02em; }
        h2 { font-size: 1.35rem !important; }
        h3 { font-size: 1.08rem !important; margin-top: .35rem !important; }
        p, label { color: var(--mg-text); }
        [data-testid="stCaptionContainer"] { color: var(--mg-secondary); }
        .mg-header { padding: .15rem 0 1rem; border-bottom: 1px solid var(--mg-border); }
        .mg-title { color: var(--mg-navy); font-size: 2.15rem; font-weight: 750;
            letter-spacing: -.04em; line-height: 1.05; margin: 0 0 .35rem; }
        .mg-subtitle { color: var(--mg-blue-gray); font-size: 1.02rem;
            font-weight: 650; margin-bottom: .25rem; }
        .mg-support { color: var(--mg-secondary); font-size: .92rem; max-width: 840px; }
        .mg-badges { display: flex; flex-wrap: wrap; gap: .4rem; margin-top: .7rem; }
        .mg-badge { background: #EEF4FF; border: 1px solid #D8E5FF;
            border-radius: 999px; color: var(--mg-blue-gray); font-size: .72rem;
            font-weight: 650; padding: .18rem .55rem; }
        .mg-empty { background: #FFF; border: 1px dashed #CFD6E0;
            border-radius: 12px; color: var(--mg-secondary); margin: .75rem 0;
            padding: 1rem 1.15rem; }
        div[data-testid="stForm"], div[data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--mg-surface); border-color: var(--mg-border) !important;
            border-radius: 12px !important; box-shadow: 0 1px 2px rgba(16,24,40,.035);
        }
        div[data-testid="stMetric"] { background: #FFF; border: 1px solid var(--mg-border);
            border-radius: 12px; padding: .85rem 1rem; min-height: 98px; }
        div[data-testid="stMetricLabel"] { color: var(--mg-secondary); }
        div[data-testid="stMetricValue"] { color: var(--mg-navy); font-size: 1.55rem; }
        .stTabs [data-baseweb="tab-list"] { gap: 1.1rem; border-bottom: 1px solid var(--mg-border); }
        .stTabs [data-baseweb="tab"] { color: var(--mg-secondary); font-weight: 600;
            height: 3rem; padding: 0 .1rem; }
        .stTabs [aria-selected="true"] { color: var(--mg-navy) !important; }
        .stTabs [data-baseweb="tab-highlight"] { background: var(--mg-navy); }
        .stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"],
        .stLinkButton > a { border-radius: 8px; font-weight: 650; min-height: 2.5rem; }
        .stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
            background: var(--mg-navy); border-color: var(--mg-navy); }
        .stButton > button[kind="primary"]:hover,
        .stFormSubmitButton > button[kind="primary"]:hover { background: #244762; }
        [data-testid="stDataFrame"] { border: 1px solid var(--mg-border);
            border-radius: 10px; overflow: hidden; }
        [data-testid="stAlert"] { border-radius: 9px; padding: .65rem .85rem; }
        [data-testid="stGraphVizChart"] { display: flex; justify-content: center;
            overflow-x: auto; padding: .35rem 0; }
        details { background: #FFF; border-color: var(--mg-border) !important; border-radius: 9px; }
        @media (max-width: 760px) {
            .block-container { padding: 3.5rem .9rem 2rem; }
            .mg-title { font-size: 1.8rem; }
            div[data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
            div[data-testid="column"] { min-width: min(100%, 260px); flex: 1 1 260px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_app_header() -> None:
    """Render the compact product identity and source badges."""
    st.markdown(
        """
        <header class="mg-header">
          <div class="mg-title">MissionGraph</div>
          <div class="mg-subtitle">Public federal contracting intelligence</div>
          <div class="mg-support">Connect historical awards, active opportunities,
          organizations, contractors, and technical capabilities using traceable public data.</div>
          <div class="mg-badges" aria-label="MissionGraph data sources">
            <span class="mg-badge">USAspending</span><span class="mg-badge">SAM.gov</span>
            <span class="mg-badge">Evidence-backed</span>
          </div>
        </header>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state(message: str) -> None:
    """Render quiet, friendly guidance for an incomplete workflow."""
    st.markdown(f'<div class="mg-empty">{message}</div>', unsafe_allow_html=True)


@st.cache_data(ttl=PUBLIC_DATA_CACHE_TTL, show_spinner=False)
def _cached_award_search(
    keyword: str,
    start_date: str,
    end_date: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Cache identical public USAspending searches for 15 minutes."""
    return search_contract_awards(keyword, start_date, end_date, limit)


@st.cache_data(ttl=PUBLIC_DATA_CACHE_TTL, show_spinner=False)
def _cached_opportunity_search(
    start_date: str,
    end_date: str,
    title: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Cache identical public SAM.gov searches for 15 minutes."""
    return search_opportunities(
        start_date=start_date,
        end_date=end_date,
        title=title or None,
        page_size=limit,
        max_pages=1,
    )


def _initialize_state() -> None:
    """Initialize persistent UI state without overwriting prior searches."""
    defaults = {
        "award_records": None,
        "award_validation_errors": [],
        "award_selection": None,
        "award_query": "",
        "opportunity_records": None,
        "opportunity_raw_by_id": {},
        "opportunity_validation_errors": [],
        "opportunity_selection": None,
        "opportunity_query": "",
        "description_retrieved": set(),
        "capability_extractions": {},
        "show_sam_graph_demo": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _safe_public_url(value: Any) -> str:
    """Remove API-key query parameters before displaying a public source URL."""
    if not isinstance(value, str) or not value.strip():
        return ""
    parts = urlsplit(value.strip())
    if parts.scheme not in {"http", "https"}:
        return ""
    query = [
        (key, item)
        for key, item in parse_qsl(parts.query, keep_blank_values=True)
        if key.casefold() != "api_key"
    ]
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(query), "")
    )


def _display_value(value: Any, fallback: str = "Not provided") -> str:
    """Return a readable value for optional public-data fields."""
    if value is None or str(value).strip() == "":
        return fallback
    return str(value)


def _format_currency(value: Any) -> str:
    """Format a numeric award value as US currency."""
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "Not provided"


def _format_date(value: Any, include_time: bool | None = None) -> str:
    """Format common API date strings for readable display."""
    if value is None or not str(value).strip():
        return "Not provided"
    text = str(value).strip()
    try:
        timestamp = pd.Timestamp(text)
    except (TypeError, ValueError):
        return text
    if pd.isna(timestamp):
        return "Not provided"

    date_label = timestamp.strftime("%b %d, %Y").replace(" 0", " ")
    show_time = include_time
    if show_time is None:
        show_time = "T" in text or " " in text.strip()
    if not show_time:
        return date_label

    time_label = timestamp.strftime("%I:%M %p").lstrip("0")
    timezone_label = timestamp.strftime("%Z") or timestamp.strftime("%z")
    suffix = f" {timezone_label}" if timezone_label else ""
    return f"{date_label} at {time_label}{suffix}"


def _shorten(value: Any, limit: int = 140) -> str:
    """Return a compact one-line preview without losing the full source text."""
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def truncate_at_word_boundary(value: str, max_length: int = 75) -> str:
    """Shorten display text without splitting the final visible word."""
    text = " ".join(str(value or "").split())
    if len(text) <= max_length:
        return text
    if max_length <= 1:
        return "…"[:max_length]
    candidate = text[: max_length - 1]
    if candidate and not candidate[-1].isspace():
        candidate = candidate.rsplit(" ", 1)[0]
    candidate = candidate.rstrip()
    return (candidate or text[: max_length - 1].rstrip()) + "…"


def _award_dropdown_label(
    record: dict[str, Any],
    duplicate_contractor_names: set[str],
) -> str:
    """Format an award choice while keeping its source ID out of view."""
    contractor = _display_value(record.get("recipient_name"))
    if contractor.casefold() not in duplicate_contractor_names:
        return contractor
    context = record.get("description") or (
        record.get("awarding_sub_agency") or record.get("awarding_agency")
    )
    context_label = truncate_at_word_boundary(str(context or ""), 58)
    return f"{contractor} — {context_label}" if context_label else contractor


def _opportunity_dropdown_label(record: dict[str, Any]) -> str:
    """Format an opportunity choice without exposing its notice ID."""
    title = truncate_at_word_boundary(str(record.get("title") or ""), 75)
    organization = (
        record.get("issuing_office")
        or record.get("department")
        or record.get("organization_path")
        or "Organization not provided"
    )
    return f"{title or 'Untitled opportunity'} — {organization}"


def _confidence_label(score: float) -> str:
    """Translate model confidence into a cautious qualitative band."""
    if score >= 0.80:
        return "High"
    if score >= 0.55:
        return "Moderate"
    return "Low"


def _humanize_category(value: Any) -> str:
    """Convert an extraction category key into a readable label."""
    return str(value).replace("_", " ").title()


def _why_award_matched(record: dict[str, Any], query: str) -> str:
    """Explain which visible award field contains the submitted keyword."""
    term = query.strip().casefold()
    if not term:
        return "No keyword was retained for this search."
    if term in str(record.get("description", "")).casefold():
        return "Description"
    agency = " ".join(
        [
            str(record.get("awarding_agency", "")),
            str(record.get("awarding_sub_agency", "")),
        ]
    ).casefold()
    if term in agency:
        return "Agency"
    return "USAspending keyword index; the matching field is not returned."


def _why_opportunity_matched(
    record: dict[str, Any],
    query: str,
    extraction: dict[str, Any] | None,
) -> str:
    """Explain which visible opportunity field contains the submitted term."""
    term = query.strip().casefold()
    if not term:
        return "Active-notice search; no title keyword was supplied."
    if term in str(record.get("title", "")).casefold():
        return "Title"
    if term in str(record.get("description", "")).casefold():
        return "Description"
    agency = " ".join(
        [
            str(record.get("issuing_office", "")),
            str(record.get("department", "")),
            str(record.get("organization_path", "")),
        ]
    ).casefold()
    if term in agency:
        return "Agency"
    if extraction and any(
        term in (
            f"{capability['name']} {capability['category']}"
        ).casefold()
        for capability in extraction["capabilities"]
    ):
        return "AI classification"
    return "SAM.gov title search; the exact matching token is not visible."


def _section_heading(title: str, description: str | None = None) -> None:
    """Render a consistent section heading and optional supporting text."""
    st.subheader(title)
    if description:
        st.caption(description)


def _render_graph(
    graph: Any,
    empty_message: str,
    *,
    opportunity_layout: bool = False,
) -> None:
    """Render a relationship graph or a polished unavailable state."""
    with st.container(border=True):
        if graph.number_of_nodes() == 0:
            render_empty_state(empty_message)
            return
        chart_width: int | str = 1050 if opportunity_layout else "stretch"
        st.graphviz_chart(
            graph_to_dot(graph, opportunity_layout=opportunity_layout),
            width=chart_width,
        )
        st.caption(
            "Evidence key: solid lines are direct source relationships; "
            "dashed lines are post-validated AI extraction."
        )


def _build_sam_graph_demo(capability_count: int) -> Any:
    """Build an in-memory SAM.gov-style graph without external API calls."""
    opportunity = {
        "source_record_id": "local-demo-opportunity",
        "source": "Local demonstration data",
        "evidence_type": "direct",
        "issuing_office": "FBI-JEH",
        "organization_id": "local-demo-agency",
        "organization_path": "Federal Bureau of Investigation.FBI-JEH",
        "title": "Enterprise GPU Servers for Artificial Intelligence",
        "solicitation_number": "DEMO-ONLY",
        "description": "Local presentation data used only to preview the graph.",
        "posted_date": "2026-07-29",
        "response_deadline": "2026-08-29",
        "opportunity_type": "Sources Sought",
        "naics_code": "541512",
        "set_aside": "",
        "source_url": "",
        "retrieved_at": "local-demo",
    }
    capability_names = [
        "AI systems",
        "Pod-scale AI systems",
        "AI inference servers and expansion hardware",
        "High-speed networking components",
        "OEM software and licenses",
        "OEM warranty and technical support",
    ]
    extraction = {
        "source_record_id": opportunity["source_record_id"],
        "model": "local-demo-no-model",
        "extracted_at": "local-demo",
        "capabilities": [
            {
                "name": name,
                "category": "local_demo",
                "evidence_quote": "Local demonstration evidence.",
                "confidence": 1.0,
            }
            for name in capability_names[:capability_count]
        ],
    }
    graph = build_opportunity_graph([opportunity])
    add_extracted_capabilities_to_graph(graph, opportunity, extraction)
    return graph


def _show_sam_graph_demo() -> None:
    """Render an optional local graph preview that never mutates search state."""
    with st.expander("Preview graph layout without using an API key"):
        st.caption(
            "This uses local demonstration data only. It does not call SAM.gov "
            "or OpenAI and does not replace your saved search results."
        )
        show_demo = st.toggle(
            "Show local graph demo",
            key="show_sam_graph_demo",
        )
        if not show_demo:
            st.write(
                "Turn on the demo to inspect the SAM.gov graph layout. Turn it "
                "off again to return to the normal tab."
            )
            return
        capability_count = st.select_slider(
            "Demo capability nodes",
            options=list(range(1, 7)),
            value=6,
            help="Try different counts to confirm the graph remains readable.",
        )
        st.info(
            "Demo mode is active for this preview only. All names and evidence "
            "below are local sample data."
        )
        _render_graph(
            _build_sam_graph_demo(capability_count),
            "The local demo graph could not be created.",
            opportunity_layout=True,
        )


def _award_table(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Build the primary contract-award table."""
    return pd.DataFrame(
        [
            {
                "Contractor": record["recipient_name"],
                "Awarding office": (
                    record["awarding_sub_agency"]
                    or record["awarding_agency"]
                    or "Not provided"
                ),
                "Award amount": _format_currency(record["award_amount"]),
                "Start date": _format_date(record["start_date"], False),
                "Description": _shorten(record["description"]),
                "Source": _safe_public_url(record.get("source_url")),
            }
            for record in records
        ]
    )


def _opportunity_status(record: dict[str, Any]) -> str:
    """Convert SAM.gov's active flag into a concise status."""
    active = record.get("active")
    if isinstance(active, bool):
        return "Active" if active else "Inactive"
    if str(active).strip().casefold() in {"yes", "true", "active", "1"}:
        return "Active"
    if str(active).strip():
        return str(active)
    return "Active"


def _opportunity_table(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Build the primary SAM.gov opportunity table."""
    return pd.DataFrame(
        [
            {
                "Opportunity title": record["title"],
                "Organization": (
                    record["issuing_office"]
                    or record["department"]
                    or "Not provided"
                ),
                "Posted date": _format_date(record["posted_date"], False),
                "Response deadline": _format_date(
                    record["response_deadline"],
                    None,
                ),
                "Opportunity type": record["opportunity_type"],
                "Status": _opportunity_status(record),
                "Source": _safe_public_url(record.get("source_url")),
            }
            for record in records
        ]
    )


def _open_deadline_count(records: list[dict[str, Any]]) -> int:
    """Count response deadlines that have not yet passed."""
    values = pd.to_datetime(
        [record.get("response_deadline") for record in records],
        errors="coerce",
        utc=True,
    )
    now = pd.Timestamp.now(tz="UTC")
    return sum(not pd.isna(value) and value >= now for value in values)


def _show_award_metrics(records: list[dict[str, Any]]) -> None:
    """Render the four contract-award summary metrics."""
    total_value = sum(record["award_amount"] for record in records)
    contractors = {
        record["recipient_name"].casefold()
        for record in records
        if record["recipient_name"]
    }
    agencies = {
        (
            record["awarding_sub_agency"]
            or record["awarding_agency"]
        ).casefold()
        for record in records
        if record["awarding_sub_agency"] or record["awarding_agency"]
    }
    columns = st.columns(4)
    columns[0].metric("Awards found", len(records))
    columns[1].metric("Total award value", _format_currency(total_value))
    columns[2].metric("Unique contractors", len(contractors))
    columns[3].metric("Unique agencies", len(agencies))


def _show_opportunity_metrics(records: list[dict[str, Any]]) -> None:
    """Render the four opportunity summary metrics."""
    agencies = {
        (
            record["issuing_office"]
            or record["department"]
            or record["organization_path"]
        ).casefold()
        for record in records
        if (
            record["issuing_office"]
            or record["department"]
            or record["organization_path"]
        )
    }
    types = [
        str(record["opportunity_type"])
        for record in records
        if record.get("opportunity_type")
    ]
    common_type = Counter(types).most_common(1)[0][0] if types else "Not provided"
    columns = st.columns(4)
    columns[0].metric("Opportunities found", len(records))
    columns[1].metric("Open deadlines", _open_deadline_count(records))
    columns[2].metric("Unique agencies", len(agencies))
    columns[3].metric("Most common type", common_type)


def _show_award_details(record: dict[str, Any], query: str) -> None:
    """Render selected-award details without hiding source values."""
    _section_heading(
        "Selected award details",
        "The values below come from the selected normalized USAspending record.",
    )
    with st.container(border=True):
        left, right = st.columns(2)
        with left:
            st.markdown(f"**{record['recipient_name']}**")
            st.write(
                _display_value(
                    record["awarding_sub_agency"]
                    or record["awarding_agency"]
                )
            )
            st.write(f"**Award value:** {_format_currency(record['award_amount'])}")
        with right:
            st.write(
                f"**Start date:** {_format_date(record['start_date'], False)}"
            )
            st.write(
                f"**End date:** {_format_date(record['end_date'], False)}"
            )
            st.write(
                f"**Award ID:** {_display_value(record['source_award_id'])}"
            )
        st.markdown("**Description**")
        st.write(record["description"] or "No description was provided.")
        st.markdown("**What this record tells you**")
        office = (
            record["awarding_sub_agency"]
            or record["awarding_agency"]
            or "The awarding office"
        )
        st.write(
            f"{office} awarded {_format_currency(record['award_amount'])} "
            f"to {record['recipient_name']} under award "
            f"{record['source_award_id']}."
        )
        st.write(f"**Why this matched:** {_why_award_matched(record, query)}")


def _show_award_evidence(record: dict[str, Any]) -> None:
    """Render provenance for the selected award."""
    _section_heading(
        "Evidence and source information",
        "Direct source fields remain separate from any later interpretation.",
    )
    with st.container(border=True):
        direct, ai_data, inferred = st.columns(3)
        with direct:
            st.markdown("🟢 **Direct source data**")
            st.caption("USAspending · Direct evidence")
            source_url = _safe_public_url(record.get("source_url"))
            if source_url:
                st.link_button("Open USAspending source", source_url)
        with ai_data:
            st.markdown("🟠 **AI-extracted information**")
            st.caption("None for this record")
        with inferred:
            st.markdown("○ **Inferred information**")
            st.caption("None")
        with st.expander("Technical provenance details"):
            st.write("Source record ID")
            st.code(str(record["source_award_id"]), language=None)
            st.write("Generated internal ID")
            st.code(
                _display_value(record["generated_internal_id"]),
                language=None,
            )
            st.write("Original recipient name")
            st.code(str(record["original_recipient_name"]), language=None)
            st.write("Raw retrieval timestamp")
            st.code(str(record["retrieved_at"]), language=None)
            st.write("Original amount value")
            st.code(
                _format_currency(record["original_award_amount"]),
                language=None,
            )


def _show_capabilities(
    extraction: dict[str, Any] | None,
) -> None:
    """Render post-validated AI capability extraction results."""
    _section_heading(
        "AI-extracted capabilities",
        "Capabilities appear only after the selected notice is analyzed.",
    )
    if extraction is None:
        render_empty_state(
            "No AI analysis has been run for this opportunity yet. Analyze "
            "the selected notice when you are ready to review extracted capabilities."
        )
        return
    st.warning(
        "Review AI-extracted fields against the quoted source text. "
        "MissionGraph does not predict bidders, winners, or outcomes."
    )
    capabilities = extraction["capabilities"]
    if not capabilities:
        st.info(
            "No capability with an exact supporting quotation was retained."
        )
    for capability in capabilities:
        with st.container(border=True):
            heading, confidence = st.columns([3, 1])
            heading.markdown(f"**{capability['name']}**")
            confidence.metric(
                "Model confidence",
                _confidence_label(capability["confidence"]),
            )
            st.caption(
                f"Category: {_humanize_category(capability['category'])} · "
                "Evidence type: ai_extracted"
            )
            st.markdown("**Exact supporting quotation**")
            st.info(capability["evidence_quote"])
            st.caption(
                f"Source record ID: {extraction['source_record_id']}"
            )
            with st.expander("Model confidence details"):
                st.write(
                    f"Numeric model score: "
                    f"{capability['confidence']:.2f} "
                    f"({capability['confidence']:.0%})"
                )
    if extraction.get("model"):
        with st.expander("Technical provenance details"):
            st.write(f"Model: {extraction['model']}")
            st.write(f"Raw extraction timestamp: {extraction['extracted_at']}")
            st.write(f"Source record ID: {extraction['source_record_id']}")


def _show_opportunity_details(
    record: dict[str, Any],
    query: str,
    extraction: dict[str, Any] | None,
) -> None:
    """Render selected-opportunity details."""
    _section_heading(
        "Selected opportunity details",
        "Direct fields from the selected SAM.gov notice.",
    )
    with st.container(border=True):
        st.markdown(f"**{record['title']}**")
        left, right = st.columns(2)
        with left:
            st.write(
                "**Organization:** "
                + _display_value(
                    record["issuing_office"]
                    or record["department"]
                    or record["organization_path"]
                )
            )
            st.write(
                "**Solicitation number:** "
                + _display_value(record["solicitation_number"])
            )
            st.write(
                "**Opportunity type:** "
                + _display_value(record["opportunity_type"])
            )
        with right:
            st.write(
                f"**Posted date:** {_format_date(record['posted_date'], False)}"
            )
            st.write(
                "**Response deadline:** "
                + _format_date(record["response_deadline"], None)
            )
            st.write(f"**Status:** {_opportunity_status(record)}")
        if record["description"]:
            with st.expander("View retrieved source description"):
                st.write(record["description"])
        else:
            st.caption(
                "The full description will be retrieved only if this "
                "opportunity is analyzed."
            )
        st.markdown("**What this record tells you**")
        organization = (
            record["issuing_office"]
            or record["department"]
            or record["organization_path"]
            or "The issuing organization"
        )
        deadline = _format_date(record["response_deadline"], None)
        st.write(
            f"{organization} published this "
            f"{_display_value(record['opportunity_type']).lower()} notice"
            + (
                f" with a response deadline of {deadline}."
                if deadline != "Not provided"
                else "."
            )
        )
        st.write(
            "**Why this matched:** "
            + _why_opportunity_matched(record, query, extraction)
        )


def _show_opportunity_evidence(
    record: dict[str, Any],
    extraction: dict[str, Any] | None,
) -> None:
    """Render direct and AI evidence classes for one opportunity."""
    _section_heading(
        "Evidence and source information",
        "MissionGraph keeps direct, AI-extracted, and inferred information "
        "visibly distinct.",
    )
    with st.container(border=True):
        direct, ai_data, inferred = st.columns(3)
        with direct:
            st.markdown("🟢 **Direct source data**")
            st.caption("SAM.gov · Direct evidence")
            source_url = _safe_public_url(record.get("source_url"))
            if source_url:
                st.link_button("Open SAM.gov source", source_url)
        with ai_data:
            st.markdown("🟠 **AI-extracted information**")
            if extraction is None:
                st.caption("Not analyzed")
            else:
                st.caption(
                    f"{len(extraction['capabilities'])} post-validated relationship(s)"
                )
        with inferred:
            st.markdown("○ **Inferred information**")
            st.caption("None · Outcomes are not inferred")
        with st.expander("Technical provenance details"):
            st.write("Notice ID / source record ID")
            st.code(str(record["source_record_id"]), language=None)
            st.write("Raw retrieval timestamp")
            st.code(str(record["retrieved_at"]), language=None)
            st.write("Original description URL")
            st.code(
                _safe_public_url(record.get("original_description_url"))
                or "Not provided",
                language=None,
            )
            if extraction is not None:
                st.write("Extraction model")
                st.code(
                    str(extraction["model"] or "Not called"),
                    language=None,
                )
                st.write("Raw extraction timestamp")
                st.code(str(extraction["extracted_at"]), language=None)


def _show_award_tab() -> None:
    """Render the complete contract-award workflow."""
    with st.form("award_search_form", border=True):
        st.markdown("### Search contract awards")
        st.caption(
            "Find Department of the Army prime awards and inspect traceable "
            "USAspending evidence."
        )
        keyword = st.text_input(
            "Contract keyword",
            placeholder="e.g., cybersecurity",
            help="Searches USAspending award text for a required keyword.",
        )
        date_left, date_right, limit_column = st.columns([1, 1, 1])
        with date_left:
            start_date = st.date_input(
                "Start date",
                value=date(date.today().year - 1, 1, 1),
                key="award_start_date",
            )
        with date_right:
            end_date = st.date_input(
                "End date",
                value=date.today(),
                key="award_end_date",
            )
        with limit_column:
            limit = st.selectbox(
                "Maximum results",
                options=list(range(5, 51, 5)),
                index=4,
                help="Returns the first page of matching awards.",
            )
        submitted = st.form_submit_button(
            "Search contract awards",
            type="primary",
            width="content",
        )

    if submitted:
        if not keyword.strip():
            st.error("Enter a contract keyword before searching.")
        elif start_date > end_date:
            st.error("Start date must be on or before end date.")
        else:
            try:
                with st.spinner("Searching USAspending..."):
                    raw_records = _cached_award_search(
                        keyword.strip(),
                        start_date.isoformat(),
                        end_date.isoformat(),
                        limit,
                    )
                    records, errors = normalize_award_records(raw_records)
            except (ValueError, USAspendingAPIError):
                st.error(
                    "USAspending could not complete this search. Check the "
                    "filters and try again shortly."
                )
            else:
                st.session_state["award_records"] = records
                st.session_state["award_validation_errors"] = errors
                st.session_state["award_query"] = keyword.strip()
                st.session_state["award_selection"] = (
                    records[0]["source_award_id"] if records else None
                )

    records = st.session_state["award_records"]
    if records is None:
        render_empty_state(
            "Search federal contract awards to inspect funding records, "
            "contractors, and source-linked evidence."
        )
        return
    errors = st.session_state["award_validation_errors"]
    if errors:
        st.warning(
            f"{len(errors)} malformed source record(s) were excluded. "
            "Their validation details remain available below."
        )
    if not records:
        render_empty_state(
            "No matching records were found. Try adjusting the keyword, "
            "date range, or result limit."
        )
        if errors:
            with st.expander("View validation details"):
                st.json(errors)
        return

    _show_award_metrics(records)
    _section_heading(
        "Contract award results",
        "Review the result set, then choose one award for a source-level inspection.",
    )
    st.dataframe(
        _award_table(records),
        hide_index=True,
        width="stretch",
        height=420,
        column_config={
            "Contractor": st.column_config.TextColumn(width="medium"),
            "Awarding office": st.column_config.TextColumn(width="medium"),
            "Award amount": st.column_config.TextColumn(width="small"),
            "Start date": st.column_config.TextColumn(width="small"),
            "Description": st.column_config.TextColumn(width="large"),
            "Source": st.column_config.LinkColumn(
                display_text="View source",
                width="small",
            ),
        },
    )

    record_by_id = {
        str(record["source_award_id"]): record for record in records
    }
    contractor_counts = Counter(
        str(record.get("recipient_name") or "").casefold()
        for record in records
    )
    duplicate_contractor_names = {
        name for name, count in contractor_counts.items() if name and count > 1
    }
    if st.session_state["award_selection"] not in record_by_id:
        st.session_state["award_selection"] = next(iter(record_by_id))
    selected_id = st.selectbox(
        "Select an award for details",
        options=list(record_by_id),
        format_func=lambda record_id: _award_dropdown_label(
            record_by_id[record_id],
            duplicate_contractor_names,
        ),
        key="award_selection",
    )
    selected = record_by_id[selected_id]
    _show_award_details(selected, st.session_state["award_query"])

    _section_heading("Relationship graph")
    try:
        award_graph = build_award_graph([selected])
    except ValueError:
        st.info(
            "A relationship graph is unavailable because this source record "
            "does not include all required organization fields."
        )
    else:
        _render_graph(award_graph, "No relationships are available.")
    _show_award_evidence(selected)

    if errors:
        with st.expander("Excluded-record validation details"):
            st.json(errors)


def _normalize_opportunities(
    raw_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Normalize SAM.gov results while retaining malformed-record errors."""
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for index, raw_record in enumerate(raw_records):
        try:
            records.append(normalize_opportunity(raw_record))
        except ValueError as exc:
            errors.append(
                {
                    "index": index,
                    "errors": [str(exc)],
                    "record": raw_record,
                }
            )
    return records, errors


def _analyze_selected_opportunity(
    selected: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Retrieve one description once, then run selected-only extraction."""
    record_id = selected["source_record_id"]
    if record_id not in st.session_state["description_retrieved"]:
        raw_record = st.session_state["opportunity_raw_by_id"][record_id]
        description = fetch_opportunity_description(
            str(selected.get("description_url") or "")
        )
        selected = normalize_opportunity(raw_record, description)
        st.session_state["description_retrieved"].add(record_id)

    extraction = extract_capabilities(
        title=selected["title"],
        description=selected["description"],
        notice_id=record_id,
    )
    return selected, extraction


def _show_opportunity_tab() -> None:
    """Render the complete SAM.gov opportunity workflow."""
    _show_sam_graph_demo()

    with st.form("opportunity_search_form", border=True):
        st.markdown("### Search SAM.gov opportunities")
        st.caption(
            "Find active federal notices and inspect direct source fields "
            "before optionally running AI extraction."
        )
        title = st.text_input(
            "Opportunity title contains",
            placeholder="e.g., software",
            help="Leave blank to search all supported active notice types.",
        )
        date_left, date_right, limit_column = st.columns([1, 1, 1])
        with date_left:
            posted_from = st.date_input(
                "Posted from",
                value=date(date.today().year, 1, 1),
                key="sam_posted_from",
            )
        with date_right:
            posted_to = st.date_input(
                "Posted to",
                value=date.today(),
                key="sam_posted_to",
            )
        with limit_column:
            limit = st.selectbox(
                "Maximum results",
                options=list(range(5, 51, 5)),
                index=1,
                key="sam_limit",
                help="Returns one page of active SAM.gov notices.",
            )
        submitted = st.form_submit_button(
            "Search SAM.gov opportunities",
            type="primary",
            width="content",
        )

    if submitted:
        if posted_from > posted_to:
            st.error("Posted-from date must be on or before posted-to date.")
        else:
            try:
                with st.spinner("Searching SAM.gov..."):
                    raw_records = _cached_opportunity_search(
                        posted_from.isoformat(),
                        posted_to.isoformat(),
                        title.strip(),
                        limit,
                    )
                    records, errors = _normalize_opportunities(raw_records)
            except (ValueError, SamGovError):
                st.error(
                    "SAM.gov could not complete this search. Confirm that "
                    "SAM_API_KEY is configured, then try again."
                )
            else:
                st.session_state["opportunity_records"] = records
                st.session_state["opportunity_raw_by_id"] = {
                    str(record.get("noticeId")): record
                    for record in raw_records
                    if str(record.get("noticeId", "")).strip()
                }
                st.session_state["opportunity_validation_errors"] = errors
                st.session_state["opportunity_query"] = title.strip()
                st.session_state["opportunity_selection"] = (
                    records[0]["source_record_id"] if records else None
                )
                st.session_state["description_retrieved"] = set()
                st.session_state["capability_extractions"] = {}

    records = st.session_state["opportunity_records"]
    if records is None:
        render_empty_state(
            "Search active federal opportunities and inspect direct notice "
            "fields before optionally running AI extraction."
        )
        return
    errors = st.session_state["opportunity_validation_errors"]
    if errors:
        st.warning(
            f"{len(errors)} malformed source record(s) were excluded from "
            "the results."
        )
    if not records:
        render_empty_state(
            "No matching records were found. Try adjusting the title, date "
            "range, or result limit."
        )
        if errors:
            with st.expander("View validation details"):
                st.json(errors)
        return

    _show_opportunity_metrics(records)
    _section_heading(
        "SAM.gov opportunity results",
        "Review active notices, then choose one for a source-level inspection.",
    )
    st.dataframe(
        _opportunity_table(records),
        hide_index=True,
        width="stretch",
        height=420,
        column_config={
            "Opportunity title": st.column_config.TextColumn(width="large"),
            "Organization": st.column_config.TextColumn(width="medium"),
            "Posted date": st.column_config.TextColumn(width="small"),
            "Response deadline": st.column_config.TextColumn(width="small"),
            "Opportunity type": st.column_config.TextColumn(width="medium"),
            "Status": st.column_config.TextColumn(width="small"),
            "Source": st.column_config.LinkColumn(
                display_text="View source",
                width="small",
            ),
        },
    )

    record_by_id = {
        str(record["source_record_id"]): record for record in records
    }
    if st.session_state["opportunity_selection"] not in record_by_id:
        st.session_state["opportunity_selection"] = next(iter(record_by_id))
    selected_id = st.selectbox(
        "Select an opportunity for details",
        options=list(record_by_id),
        format_func=lambda record_id: _opportunity_dropdown_label(
            record_by_id[record_id]
        ),
        key="opportunity_selection",
    )
    selected = record_by_id[selected_id]
    existing_extraction = st.session_state["capability_extractions"].get(
        selected_id
    )
    _show_opportunity_details(
        selected,
        st.session_state["opportunity_query"],
        existing_extraction,
    )
    button_label = (
        "Re-analyze selected opportunity"
        if existing_extraction
        else "Analyze selected opportunity"
    )
    if st.button(
        button_label,
        type="primary",
        help="Only the selected notice is sent for AI extraction.",
    ):
        try:
            with st.spinner(
                "Retrieving source text and extracting capabilities..."
            ):
                selected, extraction = _analyze_selected_opportunity(selected)
        except SamGovError:
            st.error(
                "The selected SAM.gov description could not be retrieved. "
                "Try again shortly."
            )
        except CapabilityExtractionError:
            st.error(
                "AI analysis could not be completed. Verify OPENAI_API_KEY, "
                "OPENAI_MODEL, API access, and billing, then try again."
            )
        except ValueError:
            st.error(
                "The selected notice does not contain enough valid source "
                "information for analysis."
            )
        else:
            for index, record in enumerate(records):
                if record["source_record_id"] == selected_id:
                    records[index] = selected
                    break
            st.session_state["opportunity_records"] = records
            st.session_state["capability_extractions"][selected_id] = extraction
            existing_extraction = extraction

    _section_heading(
        "Relationship graph",
        "A focused view of the selected notice and its evidence-backed relationships.",
    )
    try:
        opportunity_graph = build_opportunity_graph([selected])
        if existing_extraction is not None:
            add_extracted_capabilities_to_graph(
                opportunity_graph,
                selected,
                existing_extraction,
            )
    except ValueError:
        st.info(
            "A relationship graph is unavailable because this notice lacks "
            "required organization fields."
        )
    else:
        _render_graph(
            opportunity_graph,
            "No relationships are available.",
            opportunity_layout=True,
        )

    _show_opportunity_evidence(selected, existing_extraction)
    _show_capabilities(existing_extraction)
    if errors:
        with st.expander("Excluded-record validation details"):
            st.json(errors)


def _show_about() -> None:
    """Render a compact project and affiliation statement."""
    with st.expander("About MissionGraph"):
        st.write(
            "MissionGraph is an independent portfolio project built using "
            "public USAspending and SAM.gov data. It demonstrates API "
            "ingestion, data normalization, knowledge-graph modeling, "
            "AI-supported extraction, and evidence-backed reasoning. It is "
            "not affiliated with Torch.AI or the U.S. government."
        )


def main() -> None:
    """Render the MissionGraph portfolio application."""
    inject_custom_css()
    _initialize_state()
    render_app_header()

    award_tab, opportunity_tab = st.tabs(
        ["Contract Awards", "SAM.gov Opportunities"]
    )
    with award_tab:
        _show_award_tab()
    with opportunity_tab:
        _show_opportunity_tab()

    st.divider()
    _show_about()


if __name__ == "__main__":
    main()
