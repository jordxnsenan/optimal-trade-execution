import json
import unittest

from datetime import datetime, timezone
from io import BytesIO
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse

from execai.data.alpaca import fetch_spy_bars


class AlpacaDataTests(unittest.TestCase):

    def setUp(self):
        self.start_time = datetime(
            2025, 10, 15, 13, 30, tzinfo=timezone.utc
        )

        self.end_time = datetime(
            2025, 10, 15, 19, 59, tzinfo=timezone.utc
        )

    def make_response(self, bars, next_page_token=None):
        data = {
            "bars": bars,
            "next_page_token": next_page_token,
        }

        return BytesIO(json.dumps(data).encode("utf-8"))

    @patch("execai.data.alpaca.urlopen")
    def test_fetches_all_pages(self, mock_urlopen):
        first_bar = {
            "t": "2025-10-15T13:30:00Z",
            "c": 100.0,
        }

        second_bar = {
            "t": "2025-10-15T13:35:00Z",
            "c": 100.5,
        }

        mock_urlopen.side_effect = [
            self.make_response([first_bar], "next-page"),
            self.make_response([second_bar]),
        ]

        bars = fetch_spy_bars(
            self.start_time,
            self.end_time,
            "test-key",
            "test-secret",
        )

        self.assertEqual(bars, [first_bar, second_bar])
        self.assertEqual(mock_urlopen.call_count, 2)

        first_request = mock_urlopen.call_args_list[0].args[0]
        second_request = mock_urlopen.call_args_list[1].args[0]

        first_query = parse_qs(urlparse(first_request.full_url).query)
        second_query = parse_qs(urlparse(second_request.full_url).query)

        self.assertEqual(first_query["timeframe"], ["5Min"])
        self.assertEqual(first_query["feed"], ["sip"])
        self.assertEqual(
            first_query["start"],
            ["2025-10-15T13:30:00Z"],
        )

        self.assertNotIn("page_token", first_query)
        self.assertEqual(second_query["page_token"], ["next-page"])

    @patch("execai.data.alpaca.urlopen")
    def test_empty_response(self, mock_urlopen):
        mock_urlopen.return_value = self.make_response([])

        bars = fetch_spy_bars(
            self.start_time,
            self.end_time,
            "test-key",
            "test-secret",
        )

        self.assertEqual(bars, [])

    def test_rejects_invalid_time_range(self):
        with self.assertRaises(ValueError):
            fetch_spy_bars(
                self.end_time,
                self.start_time,
                "test-key",
                "test-secret",
            )

    def test_rejects_missing_timezone(self):
        start_time = datetime(2025, 10, 15, 13, 30)

        with self.assertRaises(ValueError):
            fetch_spy_bars(
                start_time,
                self.end_time,
                "test-key",
                "test-secret",
            )

    @patch("execai.data.alpaca.urlopen")
    def test_rejects_repeated_page_token(self, mock_urlopen):
        mock_urlopen.side_effect = [
            self.make_response([], "same-token"),
            self.make_response([], "same-token"),
        ]

        with self.assertRaisesRegex(
            RuntimeError, "repeated page token"
        ):
            fetch_spy_bars(
                self.start_time,
                self.end_time,
                "test-key",
                "test-secret",
            )

    @patch("execai.data.alpaca.urlopen")
    def test_reports_http_error(self, mock_urlopen):
        mock_urlopen.side_effect = HTTPError(
            "https://data.alpaca.markets",
            429,
            "Too Many Requests",
            None,
            None,
        )

        with self.assertRaisesRegex(RuntimeError, "HTTP 429"):
            fetch_spy_bars(
                self.start_time,
                self.end_time,
                "test-key",
                "test-secret",
            )


if __name__ == "__main__":
    unittest.main()