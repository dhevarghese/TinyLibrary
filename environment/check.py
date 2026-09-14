#!/usr/bin/env python3
"""Check TinyLibrary package pins, data resources, and upstream revisions."""

import argparse
import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path


EXTERNAL_REPOSITORY_REVISIONS = {
    "nanoVLM": "4e0c0961846135c2217f95e54cb4c2d66eb55e42",
    "babylm-eval": "02b56cbc8185de1462da195b54877b4be153fbfe",
}


def load_expected_packages(profile):
    requirements = Path(__file__).with_name(f"requirements-{profile}.txt")
    expected = {}
    for line in requirements.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        package, wanted = line.split("==", 1)
        expected[package] = wanted
    return expected


def package_errors(expected):
    errors = []
    for package, wanted in expected.items():
        try:
            installed = metadata.version(package)
        except metadata.PackageNotFoundError:
            errors.append(f"missing package: {package}=={wanted}")
            continue
        if installed.split("+", 1)[0] != wanted:
            errors.append(f"{package}: expected {wanted}, found {installed}")
    return errors


def git_revision(path):
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def repository_errors(name, path):
    if path is None:
        return [f"missing path for {name}"]
    path = Path(path).expanduser().resolve()
    if not (path / ".git").is_dir():
        return [f"{name}: not a Git checkout: {path}"]
    try:
        installed = git_revision(path)
    except (OSError, subprocess.CalledProcessError) as exc:
        return [f"{name}: cannot read revision at {path}: {exc}"]
    expected = EXTERNAL_REPOSITORY_REVISIONS[name]
    if installed != expected:
        return [f"{name}: expected {expected}, found {installed}"]
    return []


def nltk_errors():
    try:
        import nltk.data
    except ImportError:
        return ["cannot import nltk"]
    resources = ("corpora/words", "tokenizers/punkt", "tokenizers/punkt_tab")
    errors = []
    for resource in resources:
        try:
            nltk.data.find(resource)
        except LookupError:
            errors.append(
                f"missing NLTK resource: {resource} "
                "(run: python -m nltk.downloader words punkt punkt_tab)"
            )
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", choices=("construction", "training"))
    parser.add_argument("--nanovlm-dir", default=os.getenv("NANOVLM_DIR"))
    parser.add_argument("--babylm-eval-dir", default=os.getenv("BABYLM_EVAL_DIR"))
    args = parser.parse_args()

    errors = []
    if sys.version_info[:2] != (3, 11):
        errors.append(
            f"Python: expected 3.11, found {sys.version_info.major}.{sys.version_info.minor}"
        )

    if args.profile == "construction":
        errors.extend(package_errors(load_expected_packages("construction")))
        errors.extend(nltk_errors())
    else:
        errors.extend(package_errors(load_expected_packages("training")))
        errors.extend(repository_errors("nanoVLM", args.nanovlm_dir))
        errors.extend(repository_errors("babylm-eval", args.babylm_eval_dir))

    if errors:
        print("Environment check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        raise SystemExit(1)
    print(f"TinyLibrary {args.profile} environment check passed.")


if __name__ == "__main__":
    main()
