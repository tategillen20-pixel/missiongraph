"""Retrieve public contract opportunities from the SAM.gov API."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime
import os
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from bs4 import BeautifulSoup
import requests

SAM_SEARCH_URL = "https://api.sam.gov/opportunities/v2/search"
REQUEST_TIMEOUT = (5, 30)

DEFAULT_PROCUREMENT_TYPES = (
    "p",  # Presolicitation
    "r",  # Sources sought
    "o",  # Solicitation
    "k",  # Combined synopsis/solicitation
    "s",  # Special notice
)
SUPPORTED_PROCUREMENT_TYPES = frozenset(DEFAULT_PROCUREMENT_TYPES)


class SamGovError(RuntimeError):
    """Raised when a SAM.gov request cannot be completed safely."""


def _parse_date(value: str, field_name: str) -> date:
    """Parse an ISO date without accepting alternate formats."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} is required in YYYY-MM-DD format.")
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must be a valid date in YYYY-MM-DD format."
        ) from exc


def _to_sam_date(value: str, field_name: str) -> str:
    """Convert a required ``YYYY-MM-DD`` value to ``MM/DD/YYYY``."""
    return _parse_date(value, field_name).strftime("%m/%d/%Y")


def _one_year_after(value: date) -> date:
    """Return the corresponding date one calendar year later."""
    try:
        return value.replace(year=value.year + 1)
    except ValueError:
        return value.replace(year=value.year + 1, day=28)


def _validate_date_range(posted_from: str, posted_to: str) -> None:
    """Validate SAM.gov's required, maximum-one-year posting window."""
    start = _parse_date(posted_from, "postedFrom")
    end = _parse_date(posted_to, "postedTo")
    if start > end:
        raise ValueError("postedFrom must be on or before postedTo.")
    if end > _one_year_after(start):
        raise ValueError("postedFrom and postedTo cannot be over one year apart.")


def _get_api_key() -> str:
    """Read the SAM.gov API key exclusively from the process environment."""
    api_key = os.environ.get("SAM_API_KEY", "").strip()
    if not api_key:
        raise SamGovError("SAM_API_KEY is not set in the environment.")
    return api_key


def _procurement_types(values: Iterable[str]) -> tuple[str, ...]:
    """Validate and deduplicate supported procurement-type codes."""
    types = tuple(dict.fromkeys(str(value).strip().lower() for value in values))
    if not types:
        raise ValueError("At least one procurement type is required.")
    unsupported = sorted(set(types) - SUPPORTED_PROCUREMENT_TYPES)
    if unsupported:
        raise ValueError(
            "Unsupported procurement type code(s): " + ", ".join(unsupported)
        )
    return types


def _is_active(record: dict[str, Any]) -> bool:
    """Interpret SAM.gov's documented active flag."""
    value = record.get("active")
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"yes", "true", "active", "1"}


def search_opportunities(
    start_date: str,
    end_date: str,
    organization_name: str | None = None,
    procurement_types: Iterable[str] = DEFAULT_PROCUREMENT_TYPES,
    title: str | None = None,
    page_size: int = 100,
    max_pages: int = 3,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Search active SAM.gov opportunities using limit/offset pagination.

    Args:
        start_date: Required posted-from date in ``YYYY-MM-DD`` format.
        end_date: Required posted-to date in ``YYYY-MM-DD`` format.
        organization_name: Optional federal organization search text.
        procurement_types: Notice codes to include. Only presolicitations,
            sources sought, solicitations, combined synopsis/solicitations,
            and special notices are supported.
        title: Optional title search text.
        page_size: SAM.gov ``limit`` value for each page, from 1 through 1000.
        max_pages: Maximum number of pages to request.
        offset: Initial zero-based SAM.gov page index.

    Returns:
        Raw, active SAM.gov opportunity dictionaries, deduplicated by notice ID.

    Raises:
        ValueError: If search arguments are invalid.
        SamGovError: If authentication, HTTP, or response decoding fails.
    """
    _validate_date_range(start_date, end_date)
    types = _procurement_types(procurement_types)
    if (
        not isinstance(page_size, int)
        or isinstance(page_size, bool)
        or not 1 <= page_size <= 1000
    ):
        raise ValueError("page_size must be an integer between 1 and 1000.")
    if not isinstance(max_pages, int) or isinstance(max_pages, bool) or max_pages < 1:
        raise ValueError("max_pages must be a positive integer.")
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise ValueError("offset must be a non-negative integer.")

    api_key = _get_api_key()
    records_by_notice_id: dict[str, dict[str, Any]] = {}
    records_without_notice_id: list[dict[str, Any]] = []

    with requests.Session() as session:
        for page_index in range(offset, offset + max_pages):
            params: list[tuple[str, Any]] = [
                ("api_key", api_key),
                ("postedFrom", _to_sam_date(start_date, "postedFrom")),
                ("postedTo", _to_sam_date(end_date, "postedTo")),
                ("limit", page_size),
                ("offset", page_index),
            ]
            params.extend(("ptype", notice_type) for notice_type in types)
            if organization_name and organization_name.strip():
                params.append(("organizationName", organization_name.strip()))
            if title and title.strip():
                params.append(("title", title.strip()))

            try:
                response = session.get(
                    SAM_SEARCH_URL,
                    params=params,
                    timeout=REQUEST_TIMEOUT,
                )
                response.raise_for_status()
            except requests.RequestException as exc:
                status = getattr(exc.response, "status_code", None)
                suffix = f" (HTTP {status})" if status is not None else ""
                raise SamGovError(f"SAM.gov search request failed{suffix}.") from exc

            try:
                payload = response.json()
            except ValueError as exc:
                raise SamGovError(
                    "SAM.gov returned a response that was not valid JSON."
                ) from exc
            if not isinstance(payload, dict):
                raise SamGovError("SAM.gov returned an unexpected response format.")

            page_records = payload.get("opportunitiesData")
            if not isinstance(page_records, list):
                raise SamGovError(
                    "SAM.gov response did not contain an opportunitiesData list."
                )

            for record in page_records:
                if not isinstance(record, dict):
                    raise SamGovError(
                        "SAM.gov returned a non-object opportunity record."
                    )
                if not _is_active(record):
                    continue
                notice_id = str(record.get("noticeId", "")).strip()
                if notice_id:
                    records_by_notice_id.setdefault(notice_id, record)
                else:
                    records_without_notice_id.append(record)

            total_records = payload.get("totalRecords")
            pages_consumed = page_index - offset + 1
            if not page_records or len(page_records) < page_size:
                break
            if isinstance(total_records, int) and pages_consumed * page_size >= total_records:
                break

    return [*records_by_notice_id.values(), *records_without_notice_id]


def _safe_description_url(description_url: str) -> tuple[str, dict[str, str]]:
    """Validate a SAM.gov URL and replace any existing API-key parameter."""
    parts = urlsplit(description_url)
    if parts.scheme != "https" or parts.hostname != "api.sam.gov":
        raise ValueError("description_url must be an HTTPS URL on api.sam.gov.")

    query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key.casefold() != "api_key"
    ]
    sanitized_url = urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(query), "")
    )
    return sanitized_url, {"api_key": _get_api_key()}


def fetch_opportunity_description(description_url: str) -> str:
    """Download plain description text from a SAM.gov description URL.

    The URL is restricted to HTTPS on ``api.sam.gov``. Any existing API-key
    parameter is removed before the environment key is safely supplied through
    Requests query parameters.
    """
    if not isinstance(description_url, str) or not description_url.strip():
        return ""
    if description_url.strip().casefold() == "null":
        return ""

    url, key_param = _safe_description_url(description_url.strip())
    try:
        response = requests.get(
            url,
            params=key_param,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        status = getattr(exc.response, "status_code", None)
        suffix = f" (HTTP {status})" if status is not None else ""
        raise SamGovError(
            f"SAM.gov description request failed{suffix}."
        ) from exc

    content_type = response.headers.get("Content-Type", "").casefold()
    if "application/json" in content_type:
        try:
            payload = response.json()
        except ValueError as exc:
            raise SamGovError(
                "SAM.gov description response was not valid JSON."
            ) from exc
        if isinstance(payload, dict):
            for field in ("description", "content", "text"):
                value = payload.get(field)
                if isinstance(value, str):
                    return value.strip()
        return ""

    return BeautifulSoup(response.text, "html.parser").get_text(
        separator=" ", strip=True
    )
