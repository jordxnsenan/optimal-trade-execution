import unittest

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.evaluate_dqn_final import (
    load_evaluation_sessions,
    run_final_evaluation,
)


class FinalDQNEvaluationTests(unittest.TestCase):

    @patch("scripts.evaluate_dqn_final.load_dataset_split")
    def test_validation_does_not_require_confirmation(self, mock_load):
        mock_load.return_value = {index: [] for index in range(250)}

        sessions = load_evaluation_sessions("validation")

        self.assertEqual(len(sessions), 250)

        mock_load.assert_called_once()
        self.assertEqual(mock_load.call_args.args[1], "validation")

    @patch("scripts.evaluate_dqn_final.load_dataset_split")
    def test_held_out_requires_confirmation(self, mock_load):
        with self.assertRaisesRegex(
            PermissionError,
            "explicit confirmation",
        ):
            load_evaluation_sessions("test")

        # ensures the test dataset is never be loaded without confirmation
        mock_load.assert_not_called()

    @patch("scripts.evaluate_dqn_final.load_dataset_split")
    def test_confirmed_held_out_uses_test_split(self, mock_load):
        mock_load.return_value = {index: [] for index in range(187)}

        sessions = load_evaluation_sessions(
            "test",
            confirm_held_out=True,
        )

        self.assertEqual(len(sessions), 187)
        self.assertEqual(mock_load.call_args.args[1], "test")

    @patch("scripts.evaluate_dqn_final.load_dataset_split")
    def test_rejects_incomplete_dataset(self, mock_load):
        mock_load.return_value = {index: [] for index in range(100)}

        with self.assertRaisesRegex(
            ValueError,
            "Unexpected number",
        ):
            load_evaluation_sessions("validation")

    @patch("scripts.evaluate_dqn_final.load_dataset_split")
    def test_rejects_unknown_split(self, mock_load):
        with self.assertRaisesRegex(
            ValueError,
            "Unsupported evaluation split",
        ):
            load_evaluation_sessions("training")

        mock_load.assert_not_called()

    @patch("scripts.evaluate_dqn_final.load_evaluation_sessions")
    def test_does_not_overwrite_held_out_results(self, mock_load):
        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "final_results.csv"
            output_path.write_text("Existing results")

            with self.assertRaises(FileExistsError):
                run_final_evaluation(
                    "test",
                    output_path,
                    confirm_held_out=True,
                )

        mock_load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
