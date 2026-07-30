"""Manual live smoke test for one SAM.gov/OpenAI capability extraction."""

from datetime import date, timedelta

from ai.capability_extractor import extract_capabilities
from data.opportunity_normalizer import normalize_opportunity
from services.samgov import (
    fetch_opportunity_description,
    search_opportunities,
)


def main() -> None:
    """Search SAM.gov and analyze only the first returned opportunity."""
    posted_to = date.today()
    posted_from = posted_to - timedelta(days=30)
    records = search_opportunities(
        start_date=posted_from.isoformat(),
        end_date=posted_to.isoformat(),
        page_size=5,
        max_pages=1,
    )
    if not records:
        print("No active SAM.gov opportunities were returned.")
        return

    raw_record = records[0]
    description = fetch_opportunity_description(
        str(raw_record.get("description", ""))
    )
    opportunity = normalize_opportunity(raw_record, description)
    extraction = extract_capabilities(
        title=opportunity["title"],
        description=opportunity["description"],
        notice_id=opportunity["source_record_id"],
    )

    print("Notice:", extraction["source_record_id"])
    print("Model:", extraction["model"])
    print("Capabilities:", len(extraction["capabilities"]))
    for capability in extraction["capabilities"]:
        print(
            capability["name"],
            capability["category"],
            capability["confidence"],
            repr(capability["evidence_quote"]),
        )


if __name__ == "__main__":
    main()
