import unittest

from datetime import date, timezone

from scripts.check_alpaca_access import get_regular_session_bounds


class MarketSessionTests(unittest.TestCase):

    def test_regular_session_during_standard_time(self):
        session_start, session_end = get_regular_session_bounds(
            date(2023, 2, 15)
        )

        self.assertEqual(
            session_start.astimezone(timezone.utc).strftime("%H:%M:%S"),
            "14:30:00",
        )

        self.assertEqual(
            session_end.astimezone(timezone.utc).strftime("%H:%M:%S"),
            "20:59:59",
        )

    def test_regular_session_during_daylight_saving_time(self):
        session_start, session_end = get_regular_session_bounds(
            date(2024, 7, 17)
        )

        self.assertEqual(
            session_start.astimezone(timezone.utc).strftime("%H:%M:%S"),
            "13:30:00",
        )

        self.assertEqual(
            session_end.astimezone(timezone.utc).strftime("%H:%M:%S"),
            "19:59:59",
        )


if __name__ == "__main__":
    unittest.main()