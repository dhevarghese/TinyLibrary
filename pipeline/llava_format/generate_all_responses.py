#!/usr/bin/env python3
"""Run the 12 age-band/task annotation jobs used to build TinyLibrary."""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_REVISION = "e1945c40cd546c78e41f1151f4db032b271faeaa"
AGE_BANDS = (
    ("3-5", "3_5"),
    ("6-8", "6_8"),
    ("9-12", "9_12"),
    ("12+", "12"),
)
TASKS = (
    ("caption", "captions"),
    ("vqa", "conversations"),
    ("reasoning", "reasoning"),
)


def commands(args):
    script_dir = Path(__file__).resolve().parent
    prompt_root = script_dir.parents[1] / "playground" / "prompts"
    generator = script_dir / "generate_responses.py"
    for task_name, prompt_name in TASKS:
        task_output = Path(args.output_dir) / task_name
        for source_band, file_band in AGE_BANDS:
            prompt_dir = prompt_root / prompt_name / file_band
            yield [
                sys.executable,
                str(generator),
                "--library_file", str(Path(args.descriptions_dir) / f"library-{source_band}.jsonl"),
                "--output_path", str(task_output),
                "--output_file", f"{task_name}_{file_band}",
                "--system_message_file", str(prompt_dir / "system_message.txt"),
                "--samples_file", str(prompt_dir / "samples.txt"),
                "--log_file", str(task_output / f"{task_name}_{file_band}.log"),
                "--model_name", args.model_name,
                "--model_revision", args.model_revision,
                "--max_batch_size", str(args.max_batch_size),
            ]


def main():
    parser = argparse.ArgumentParser(
        description="Generate caption, VQA, and reasoning annotations for all four age bands."
    )
    parser.add_argument(
        "--descriptions_dir",
        required=True,
        help="Directory containing library-3-5.jsonl, library-6-8.jsonl, "
             "library-9-12.jsonl, and library-12+.jsonl",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--model_revision", default=MODEL_REVISION)
    parser.add_argument("--max_batch_size", type=int, default=5000)
    parser.add_argument(
        "--dry_run", action="store_true", help="Print the 12 commands without running them."
    )
    args = parser.parse_args()

    for command in commands(args):
        if args.dry_run:
            print(shlex.join(command))
            continue
        Path(command[command.index("--output_path") + 1]).mkdir(parents=True, exist_ok=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
