"""Checks for the release package's recorded revisions, pins, and notices."""

import importlib.util
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]


def load_environment_checker():
    path = REPOSITORY / "environment" / "check.py"
    spec = importlib.util.spec_from_file_location("environment_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EnvironmentMetadataTests(unittest.TestCase):
    def test_external_revisions_are_exact_commits(self):
        checker = load_environment_checker()
        self.assertEqual(
            checker.EXTERNAL_REPOSITORY_REVISIONS,
            {
                "nanoVLM": "4e0c0961846135c2217f95e54cb4c2d66eb55e42",
                "babylm-eval": "02b56cbc8185de1462da195b54877b4be153fbfe",
            },
        )

    def test_requirement_profiles_are_exact_and_distinct(self):
        checker = load_environment_checker()
        construction = checker.load_expected_packages("construction")
        training = checker.load_expected_packages("training")
        self.assertEqual(construction["transformers"], "4.43.4")
        self.assertEqual(training["transformers"], "4.57.6")
        self.assertIn("torchvision", construction)
        self.assertIn("torchvision", training)
        self.assertIn("einops", construction)
        self.assertIn("einops", training)


class ReleaseComplianceTests(unittest.TestCase):
    def test_citation_has_year_only_inside_preferred_citation(self):
        citation = (REPOSITORY / "CITATION.cff").read_text(encoding="utf-8")
        top_level_years = [line for line in citation.splitlines() if line.startswith("year:")]
        self.assertEqual(top_level_years, [])
        self.assertIn("\n  year: 2026\n", citation)

    def test_vendored_code_has_scoped_license_notice_and_full_license(self):
        notices = (REPOSITORY / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
        source = (REPOSITORY / "pipeline/llava_format/sanity_check.py").read_text(
            encoding="utf-8"
        )
        license_text = (
            REPOSITORY / "licenses/TinyLLaVA-Apache-2.0.txt"
        ).read_text(encoding="utf-8")
        self.assertIn("pipeline/llava_format/sanity_check.py", notices)
        self.assertIn("Licensed under the Apache License, Version 2.0", source)
        self.assertIn("Apache License\n                           Version 2.0", license_text)
        self.assertIn("END OF TERMS AND CONDITIONS", license_text)

    def test_release_tree_contains_no_macos_metadata(self):
        self.assertEqual(list(REPOSITORY.rglob(".DS_Store")), [])


if __name__ == "__main__":
    unittest.main()
