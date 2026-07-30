"""Manual live smoke test for SAM.gov opportunity ingestion.

This file intentionally performs work only when run directly, so pytest can
collect the repository without making live API requests.
"""

from datetime import date, timedelta

from data.opportunity_normalizer import normalize_opportunity
from graph.builder import build_opportunity_graph
from services.samgov import (
    fetch_opportunity_description,
    search_opportunities,
)


def main() -> None:
    """Run one small live search, description fetch, and graph build."""
    end_date = date.today()
    start_date = end_date - timedelta(days=30)
    results = search_opportunities(
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        page_size=5,
        max_pages=1,
    )
    print(f"Found {len(results)} active opportunities")
    if not results:
        return

    raw_record = results[0]
    description = fetch_opportunity_description(
        str(raw_record.get("description", ""))
    )
    opportunity = normalize_opportunity(raw_record, description)
    graph = build_opportunity_graph([opportunity])
    print(
        opportunity["notice_id"],
        opportunity["title"],
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )


if __name__ == "__main__":
    main()
