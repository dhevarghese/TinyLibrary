"""Small fixture tests for construction-pipeline handoffs and resume logic."""

import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, REPOSITORY / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AgeImageJoinTests(unittest.TestCase):
    def test_flat_images_are_retained_and_missing_images_are_reported(self):
        age_filter = load_module("age_filter", "pipeline/age/filter.py")
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            captions = root / "captions"
            images = root / "images"
            output = root / "output"
            captions.mkdir()
            images.mkdir()
            (images / "book-a_0001.jpg").touch()
            records = [
                {"book_name": "book-a", "page_number": "0001", "caption": "kept"},
                {"book_name": "book-b", "page_number": "0002", "caption": "missing"},
                {"book_name": "book-c", "page_number": "0003", "caption": "unlabeled"},
            ]
            with (captions / "library-000000.jsonl").open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")

            messages = io.StringIO()
            with contextlib.redirect_stdout(messages):
                age_filter.filter_captions_by_age_group(
                    {"book-a": "3-5", "book-b": "6-8"},
                    captions,
                    images,
                    output,
                )

            retained = (output / "library-3-5.jsonl").read_text(encoding="utf-8")
            self.assertEqual(json.loads(retained)["caption"], "kept")
            self.assertFalse((output / "library-6-8.jsonl").exists())
            self.assertIn("wrote 1/3 descriptions", messages.getvalue())
            self.assertIn("missing age label=1", messages.getvalue())
            self.assertIn("missing retained image=1", messages.getvalue())
            self.assertIn("Missing-age-label examples: book-c/0003", messages.getvalue())

    def test_age_label_book_names_accept_filename_or_plain_id(self):
        age_filter = load_module("age_filter_names", "pipeline/age/filter.py")
        with tempfile.TemporaryDirectory() as temporary_directory:
            labels = Path(temporary_directory) / "labels.jsonl"
            labels.write_text(
                json.dumps({"book_name": "book-a.jsonl", "age-group": "3-5"})
                + "\n"
                + json.dumps({"book_name": "book-b", "age-group": "6-8"})
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(
                age_filter.load_age_group_data(labels),
                {"book-a": "3-5", "book-b": "6-8"},
            )


class AnnotationResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.resume = load_module("image_resume", "pipeline/image/resume.py")

    def test_partial_batch_resume_selects_every_unfinished_page(self):
        keys = [f"book/{page}" for page in range(30)]
        images = [f"image-{page}" for page in range(30)]
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "descriptions.jsonl"
            with output.open("w", encoding="utf-8") as handle:
                for page in range(8):
                    handle.write(json.dumps({
                        "book_name": "book", "page_number": str(page), "caption": "done"
                    }) + "\n")
            processed = self.resume.get_processed_keys(output)

            pending = []
            for start in range(0, len(keys), 10):
                pending.extend(
                    self.resume.pending_items(
                        keys[start:start + 10], images[start:start + 10], processed
                    )
                )

        self.assertEqual([key for key, _ in pending], keys[8:])

    def test_annotation_matrix_covers_all_tasks_and_age_bands(self):
        runner = REPOSITORY / "pipeline/llava_format/generate_all_responses.py"
        completed = subprocess.run(
            [
                sys.executable,
                str(runner),
                "--descriptions_dir", "/tmp/descriptions",
                "--output_dir", "/tmp/responses",
                "--dry_run",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        commands = completed.stdout.splitlines()
        self.assertEqual(len(commands), 12)
        for task in ("caption", "vqa", "reasoning"):
            for band in ("3_5", "6_8", "9_12", "12"):
                self.assertTrue(
                    any(f"--output_file {task}_{band}" in command for command in commands),
                    f"missing {task}/{band}",
                )
        self.assertTrue(all("meta-llama/Meta-Llama-3-8B-Instruct" in c for c in commands))
        self.assertTrue(all("e1945c40cd546c78e41f1151f4db032b271faeaa" in c for c in commands))

    def test_existing_output_rejects_duplicate_page_keys(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "descriptions.jsonl"
            duplicate = {"book_name": "book", "page_number": "7", "caption": "text"}
            output.write_text(
                json.dumps(duplicate) + "\n" + json.dumps(duplicate) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Duplicate page book/7"):
                self.resume.get_processed_keys(output)


if __name__ == "__main__":
    unittest.main()
