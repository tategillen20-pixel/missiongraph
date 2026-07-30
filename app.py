"""Streamlit entry point for the MissionGraph application."""

from datetime import date
from typing import Any

import pandas as pd
import streamlit as st

from ai.capability_extractor import (
    CapabilityExtractionError,
    extract_capabilities,
)
from data.normalizer import normalize_award_records
from data.opportunity_normalizer import normalize_opportunity
from graph.builder import (
    add_extracted_capabilities_to_graph,
    build_opportunity_graph,
)
from services.samgov import (
    SamGovError,
    fetch_opportunity_description,
    search_opportunities,
)
from services.usaspending import USAspendingAPIError, search_contract_awards


def _results_table(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Create the user-facing results table from normalized award records."""
    return pd.DataFrame(
        [
            {
                "Recipient": record["recipient_name"],
                "Award Amount": record["award_amount"],
                "Agency": (
                    record["awarding_sub_agency"]
                    or record["awarding_agency"]
                ),
                "Start Date": record["start_date"],
                "End Date": record["end_date"],
                "Description": record["description"],
            }
            for record in records
        ]
    )


def _show_results(
    records: list[dict[str, Any]],
    validation_errors: list[dict[str, Any]],
) -> None:
    """Render normalized awards, summary metrics, and validation warnings."""
    if validation_errors:
        st.warning(
            f"{len(validation_errors)} record(s) could not be normalized and "
            "were excluded from the table."
        )
        with st.expander("View record validation errors"):
            st.json(validation_errors)

    if not records:
        st.info("No valid awards were found for the selected search.")
        return

    total_award_value = sum(record["award_amount"] for record in records)
    unique_recipients = len(
        {record["recipient_name"].casefold() for record in records}
    )

    award_metric, value_metric, recipient_metric = st.columns(3)
    award_metric.metric("Number of awards", len(records))
    value_metric.metric("Total award value", f"${total_award_value:,.2f}")
    recipient_metric.metric("Unique recipients", unique_recipients)

    st.subheader("Contract awards")
    st.dataframe(
        _results_table(records),
        hide_index=True,
        use_container_width=True,
        column_config={
            "Award Amount": st.column_config.NumberColumn(
                "Award Amount",
                format="$%.2f",
            )
        },
    )


def _show_evidence_inspector(records: list[dict[str, Any]]) -> None:
    """Render direct source evidence for a user-selected award."""
    st.subheader("Evidence Inspector")
    st.caption(
        "Claims shown here are assembled directly from USAspending fields. "
        "No inferred or language-model-generated conclusions are included."
    )

    selected_index = st.selectbox(
        "Select an award",
        options=range(len(records)),
        format_func=lambda index: (
            f"{records[index]['source_award_id']} — "
            f"{records[index]['original_recipient_name']}"
        ),
    )
    record = records[selected_index]
    agency = record["awarding_sub_agency"] or record["awarding_agency"]
    original_recipient = record["original_recipient_name"]

    with st.container(border=True):
        st.markdown("**Human-readable direct claim**")
        st.write(
            f"USAspending reports that {agency} issued award "
            f"{record['source_award_id']} to {original_recipient} for "
            f"${record['award_amount']:,.2f}."
        )

        st.markdown("**Direct USAspending fields**")
        source_left, source_right = st.columns(2)
        with source_left:
            st.write("Source record ID")
            st.code(str(record["source_award_id"]), language=None)
            st.write("Original recipient name")
            st.code(str(original_recipient), language=None)
            st.write("Original award amount")
            st.code(str(record["original_award_amount"]), language=None)
        with source_right:
            st.write("Original start date")
            st.code(str(record["original_start_date"]), language=None)
            st.write("Original end date")
            st.code(str(record["original_end_date"]), language=None)
            st.write("Original award description")
            original_description = record["original_award_description"]
            st.code(
                "" if original_description is None else str(original_description),
                language=None,
            )

        st.markdown("**MissionGraph evidence metadata**")
        metadata_left, metadata_right, timestamp_column = st.columns(3)
        metadata_left.metric("Evidence type", record["evidence_type"])
        metadata_right.metric("Source", record["source"])
        timestamp_column.write("Retrieved at (UTC)")
        timestamp_column.code(record["retrieved_at"], language=None)


def _show_award_search() -> None:
    """Render the USAspending award-search interface."""
    st.write(
        "Explore Department of the Army contract awards using public federal "
        "spending data from USAspending."
    )

    keyword = st.text_input(
        "Keyword",
        placeholder="For example: cybersecurity",
    )

    start_column, end_column = st.columns(2)
    with start_column:
        start_date = st.date_input(
            "Start date",
            value=date(date.today().year, 1, 1),
        )
    with end_column:
        end_date = st.date_input("End date", value=date.today())

    limit = st.slider(
        "Maximum results",
        min_value=5,
        max_value=50,
        value=25,
        step=5,
    )

    if "award_records" not in st.session_state:
        st.session_state["award_records"] = None
        st.session_state["validation_errors"] = []

    if st.button("Search Awards", type="primary"):
        if not keyword.strip():
            st.error("Enter a keyword before searching.")
            return
        if start_date > end_date:
            st.error("The start date must be on or before the end date.")
            return

        try:
            with st.spinner("Retrieving USAspending contract awards..."):
                raw_records = search_contract_awards(
                    keyword=keyword,
                    start_date=start_date.isoformat(),
                    end_date=end_date.isoformat(),
                    limit=limit,
                )
                normalized_records, validation_errors = (
                    normalize_award_records(raw_records)
                )
        except ValueError as exc:
            st.error(f"Invalid search input: {exc}")
            return
        except USAspendingAPIError as exc:
            st.error(
                "USAspending could not complete the search. "
                f"Please try again later. Details: {exc}"
            )
            return

        st.session_state["award_records"] = normalized_records
        st.session_state["validation_errors"] = validation_errors

    current_records = st.session_state["award_records"]
    if current_records is None:
        return
    if not current_records and not st.session_state["validation_errors"]:
        st.info("No awards matched the keyword and date range.")
        return

    _show_results(
        current_records,
        st.session_state["validation_errors"],
    )
    if current_records:
        _show_evidence_inspector(current_records)


def _show_capability_results(extraction: dict[str, Any]) -> None:
    """Render AI-extracted capabilities with their supporting quotations."""
    st.subheader("AI-extracted capabilities")
    st.caption(
        "These relationships are AI-extracted, not direct SAM.gov facts. "
        "Every displayed quote passed exact-substring validation."
    )
    capabilities = extraction["capabilities"]
    if not capabilities:
        st.info(
            "No capability with an exact supporting quotation was extracted."
        )
        return

    for capability in capabilities:
        with st.expander(capability["name"]):
            st.write(f"Category: {capability['category']}")
            st.write(f"Confidence: {capability['confidence']:.0%}")
            st.write("Exact SAM.gov evidence quote:")
            st.info(capability["evidence_quote"])

    st.caption(
        f"Model: {extraction['model']} · "
        f"Extracted at: {extraction['extracted_at']}"
    )


def _show_opportunity_analysis() -> None:
    """Search SAM.gov and analyze only the opportunity selected by the user."""
    st.write(
        "Search active SAM.gov opportunities, then optionally extract "
        "evidence-backed capabilities from one selected notice."
    )

    date_left, date_right = st.columns(2)
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
    title = st.text_input(
        "Opportunity title contains",
        key="sam_title",
    )
    limit = st.slider(
        "Maximum SAM.gov results",
        min_value=5,
        max_value=50,
        value=10,
        step=5,
    )

    if "opportunity_raw_records" not in st.session_state:
        st.session_state["opportunity_raw_records"] = None
        st.session_state["opportunity_records"] = []
        st.session_state["capability_extractions"] = {}

    if st.button("Search SAM.gov Opportunities", type="primary"):
        try:
            with st.spinner("Retrieving active SAM.gov opportunities..."):
                raw_records = search_opportunities(
                    start_date=posted_from.isoformat(),
                    end_date=posted_to.isoformat(),
                    title=title or None,
                    page_size=limit,
                    max_pages=1,
                )
                opportunities = [
                    normalize_opportunity(record) for record in raw_records
                ]
        except (ValueError, SamGovError) as exc:
            st.error(f"SAM.gov search failed: {exc}")
        else:
            st.session_state["opportunity_raw_records"] = raw_records
            st.session_state["opportunity_records"] = opportunities
            st.session_state["capability_extractions"] = {}
            st.session_state.pop("selected_opportunity_index", None)

    raw_records = st.session_state["opportunity_raw_records"]
    opportunities = st.session_state["opportunity_records"]
    if raw_records is None:
        return
    if not opportunities:
        st.info("No active opportunities matched the search.")
        return

    selected_index = st.selectbox(
        "Select one opportunity to analyze",
        options=range(len(opportunities)),
        format_func=lambda index: (
            f"{opportunities[index]['notice_id']} — "
            f"{opportunities[index]['title']}"
        ),
        key="selected_opportunity_index",
    )
    selected = opportunities[selected_index]
    st.write(selected["title"])
    st.caption(
        f"SAM.gov notice {selected['source_record_id']} · "
        f"Evidence type: {selected['evidence_type']}"
    )

    if st.button("Analyze Selected Opportunity"):
        try:
            with st.spinner(
                "Retrieving its description and extracting capabilities..."
            ):
                description = fetch_opportunity_description(
                    str(selected["description_url"] or "")
                )
                selected = normalize_opportunity(
                    raw_records[selected_index],
                    description_text=description,
                )
                extraction = extract_capabilities(
                    title=selected["title"],
                    description=selected["description"],
                    notice_id=selected["source_record_id"],
                )
                graph = build_opportunity_graph([selected])
                add_extracted_capabilities_to_graph(
                    graph,
                    selected,
                    extraction,
                )
        except (ValueError, SamGovError, CapabilityExtractionError) as exc:
            st.error(f"Selected opportunity analysis failed: {exc}")
        else:
            opportunities[selected_index] = selected
            st.session_state["opportunity_records"] = opportunities
            st.session_state["capability_extractions"][
                selected["source_record_id"]
            ] = extraction
            st.session_state["selected_opportunity_graph"] = graph

    existing_extraction = st.session_state["capability_extractions"].get(
        selected["source_record_id"]
    )
    if existing_extraction:
        _show_capability_results(existing_extraction)


def main() -> None:
    """Render the MissionGraph application."""
    st.set_page_config(page_title="MissionGraph", layout="wide")
    st.title("MissionGraph")
    award_tab, opportunity_tab = st.tabs(
        ["Contract Awards", "SAM.gov Opportunities"]
    )
    with award_tab:
        _show_award_search()
    with opportunity_tab:
        _show_opportunity_analysis()


if __name__ == "__main__":
    main()
