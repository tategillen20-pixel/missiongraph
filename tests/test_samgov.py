"""Unit tests for the SAM.gov service client."""

from unittest.mock import Mock, patch

import pytest
import requests

from services.samgov import (
    SamGovError,
    fetch_opportunity_description,
    search_opportunities,
)


def _response(payload: dict) -> Mock:
    """Create a successful mocked Requests response."""
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = payload
    return response


def test_search_converts_dates_filters_types_and_paginates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Search sends SAM.gov dates, supported types, limit, and page offsets."""
    monkeypatch.setenv("SAM_API_KEY", "secret-test-key")
    first_page = _response(
        {
            "totalRecords": 4,
            "opportunitiesData": [
                {"noticeId": "one", "active": "Yes"},
                {"noticeId": "archived", "active": "No"},
            ],
        }
    )
    second_page = _response(
        {
            "totalRecords": 4,
            "opportunitiesData": [
                {"noticeId": "two", "active": True},
                {"noticeId": "one", "active": "Yes"},
            ],
        }
    )
    session = Mock()
    session.get.side_effect = [first_page, second_page]
    session_context = Mock()
    session_context.__enter__ = Mock(return_value=session)
    session_context.__exit__ = Mock(return_value=False)

    with patch("services.samgov.requests.Session", return_value=session_context):
        records = search_opportunities(
            "2025-01-02",
            "2025-02-03",
            page_size=2,
            max_pages=2,
            offset=3,
        )

    assert [record["noticeId"] for record in records] == ["one", "two"]
    first_params = session.get.call_args_list[0].kwargs["params"]
    second_params = session.get.call_args_list[1].kwargs["params"]
    assert ("postedFrom", "01/02/2025") in first_params
    assert ("postedTo", "02/03/2025") in first_params
    assert ("limit", 2) in first_params
    assert ("offset", 3) in first_params
    assert ("offset", 4) in second_params
    assert [value for key, value in first_params if key == "ptype"] == [
        "p",
        "r",
        "o",
        "k",
        "s",
    ]


@pytest.mark.parametrize(
    ("start_date", "end_date"),
    [
        ("", "2025-01-01"),
        ("2025-01-01", ""),
        ("2025-02-01", "2025-01-01"),
        ("2025-01-01", "2026-01-02"),
    ],
)
def test_search_rejects_invalid_date_ranges(
    monkeypatch: pytest.MonkeyPatch,
    start_date: str,
    end_date: str,
) -> None:
    """Both dates are required and cannot span over one calendar year."""
    monkeypatch.setenv("SAM_API_KEY", "secret-test-key")

    with pytest.raises(ValueError):
        search_opportunities(start_date, end_date)


def test_search_reads_key_only_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing process environment key produces a safe custom error."""
    monkeypatch.delenv("SAM_API_KEY", raising=False)

    with pytest.raises(SamGovError, match="not set"):
        search_opportunities("2025-01-01", "2025-01-02")


def test_description_safely_replaces_existing_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Description retrieval strips URL keys and supplies the environment key."""
    monkeypatch.setenv("SAM_API_KEY", "secret-test-key")
    response = Mock()
    response.raise_for_status.return_value = None
    response.headers = {"Content-Type": "text/html"}
    response.text = "<p>Direct <strong>description</strong></p>"

    with patch("services.samgov.requests.get", return_value=response) as get:
        text = fetch_opportunity_description(
            "https://api.sam.gov/example?noticeid=123&api_key=old-key"
        )

    assert text == "Direct description"
    assert get.call_args.args[0] == "https://api.sam.gov/example?noticeid=123"
    assert get.call_args.kwargs["params"] == {"api_key": "secret-test-key"}


def test_request_error_does_not_expose_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Custom errors never interpolate a Requests URL containing the key."""
    monkeypatch.setenv("SAM_API_KEY", "secret-test-key")
    response = Mock(status_code=403)
    request_error = requests.HTTPError(
        "403 for https://api.sam.gov/?api_key=secret-test-key",
        response=response,
    )
    session = Mock()
    session.get.side_effect = request_error
    session_context = Mock()
    session_context.__enter__ = Mock(return_value=session)
    session_context.__exit__ = Mock(return_value=False)

    with patch("services.samgov.requests.Session", return_value=session_context):
        with pytest.raises(SamGovError) as caught:
            search_opportunities("2025-01-01", "2025-01-02")

    assert "secret-test-key" not in str(caught.value)
