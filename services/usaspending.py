"""Client functions for retrieving award data from the USAspending API."""

from datetime import datetime
from typing import Any

import requests

USASPENDING_API_URL = (
    "https://api.usaspending.gov/api/v2/search/spending_by_award/"
)
REQUEST_TIMEOUT = (5, 30)
CONTRACT_AWARD_TYPE_CODES = ("A", "B", "C", "D")


class USAspendingAPIError(RuntimeError):
    """Raised when a USAspending search cannot be completed or decoded."""


def _parse_date(value: str, field_name: str) -> datetime:
    """Parse and validate an ISO ``YYYY-MM-DD`` date."""
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string in YYYY-MM-DD format.")

    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must be a valid date in YYYY-MM-DD format."
        ) from exc


def search_contract_awards(
    keyword: str,
    start_date: str,
    end_date: str,
    limit: int = 25,
) -> list[dict[str, Any]]:
    """Search Department of the Army contract awards.

    Args:
        keyword: Non-empty term to search across USAspending award data.
        start_date: Beginning of the search period in ``YYYY-MM-DD`` format.
        end_date: End of the search period in ``YYYY-MM-DD`` format.
        limit: Maximum number of raw award records to return.

    Returns:
        The raw dictionaries from the API response's ``results`` list.

    Raises:
        ValueError: If the keyword, dates, or limit are invalid.
        USAspendingAPIError: If the request fails, the API returns an HTTP
            error, or the response does not contain the expected JSON shape.
    """
    if not isinstance(keyword, str) or not keyword.strip():
        raise ValueError("keyword must be a non-empty string.")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ValueError("limit must be a positive integer.")

    parsed_start_date = _parse_date(start_date, "start_date")
    parsed_end_date = _parse_date(end_date, "end_date")
    if parsed_start_date > parsed_end_date:
        raise ValueError("start_date must be on or before end_date.")

    payload = {
        "filters": {
            "award_type_codes": list(CONTRACT_AWARD_TYPE_CODES),
            "agencies": [
                {
                    "type": "awarding",
                    "tier": "subtier",
                    "name": "Department of the Army",
                    "toptier_name": "Department of Defense",
                }
            ],
            "time_period": [
                {
                    "start_date": start_date,
                    "end_date": end_date,
                }
            ],
            "keywords": [keyword.strip()],
        },
        "fields": [
            "Award ID",
            "Recipient Name",
            "Award Amount",
            "Awarding Agency",
            "Awarding Sub Agency",
            "Start Date",
            "End Date",
            "Description",
            "generated_internal_id",
        ],
        "limit": limit,
        "page": 1,
        "subawards": False,
    }

    try:
        response = requests.post(
            USASPENDING_API_URL,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        status = getattr(exc.response, "status_code", None)
        status_message = f" (HTTP {status})" if status is not None else ""
        raise USAspendingAPIError(
            f"USAspending contract award search failed{status_message}."
        ) from exc

    try:
        response_data = response.json()
    except (requests.exceptions.JSONDecodeError, ValueError) as exc:
        raise USAspendingAPIError(
            "USAspending returned a response that was not valid JSON."
        ) from exc

    if not isinstance(response_data, dict):
        raise USAspendingAPIError(
            "USAspending returned an unexpected response format."
        )

    results = response_data.get("results")
    if not isinstance(results, list) or not all(
        isinstance(record, dict) for record in results
    ):
        raise USAspendingAPIError(
            "USAspending response did not contain a valid results list."
        )

    return results
