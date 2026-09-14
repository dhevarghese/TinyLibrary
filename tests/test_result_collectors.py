"""Fixture tests for experimental-run parsing and result collation."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
TRAINING = REPOSITORY / "training"


def add_report(root, run_name, task, dataset, score):
    report = (
        root
        / run_name
        / "main"
        / "zero_shot"
        / "causal"
        / task
        / dataset
        / "best_temperature_report.txt"
    )
    report.parent.mkdir(parents=True)
    report.write_text(f"### AVERAGE ACCURACY\n{score}\n", encoding="utf-8")
    return report


def run_collector(script, results, *arguments):
    return subprocess.run(
        [sys.executable, str(TRAINING / script), "--results_dir", str(results), *arguments],
        cwd=TRAINING,
        capture_output=True,
        text=True,
    )


class CollectorTests(unittest.TestCase):
    def test_zero_shot_keeps_training_budgets_separate(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            results = Path(temporary_directory)
            add_report(
                results, "tl_random_staged_ep5_scratchlm_mm_seed0_first",
                "blimp", "blimp_filtered", 50.0,
            )
            add_report(
                results, "tl_random_staged_ep10_scratchlm_mm_seed0_second",
                "blimp", "blimp_filtered", 70.0,
            )

            completed = run_collector("collect_zeroshot.py", results)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("ep5/scratchlm/mm/random/staged | 50.00", completed.stdout)
            self.assertIn("ep10/scratchlm/mm/random/staged | 70.00", completed.stdout)

    def test_zero_shot_rejects_ambiguous_reruns(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            results = Path(temporary_directory)
            for run_id, score in (("first", 50.0), ("rerun", 90.0)):
                add_report(
                    results, f"tl_random_staged_ep10_scratchlm_mm_seed0_{run_id}",
                    "blimp", "blimp_filtered", score,
                )

            completed = run_collector("collect_zeroshot.py", results, "--epochs", "10")
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("Multiple results match", completed.stderr)

    def test_multimodal_keeps_modalities_separate(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            results = Path(temporary_directory)
            add_report(
                results, "tl_random_staged_ep10_scratchlm_mm_seed0_first",
                "vqa", "vqa_filtered", 22.0,
            )
            add_report(
                results, "tl_random_staged_ep10_scratchlm_textonly_seed0_second",
                "vqa", "vqa_filtered", 44.0,
            )

            completed = run_collector("collect_multimodal.py", results)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("scratchlm/mm/random/staged | 22.00", completed.stdout)
            self.assertIn("scratchlm/textonly/random/staged | 44.00", completed.stdout)


if __name__ == "__main__":
    unittest.main()
