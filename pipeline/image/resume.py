"""Dependency-light helpers for resuming image-description generation."""

import json


def split_key(key):
    """Split a WebDataset page key into its book and page components."""
    if isinstance(key, bytes):
        key = key.decode("utf-8")
    book_name, page_number = key.rsplit("/", 1)
    return book_name, page_number


def pending_items(keys, images, processed_keys):
    """Return all batch items whose page identity is not already recorded."""
    return [
        (key, image)
        for key, image in zip(keys, images)
        if split_key(key) not in processed_keys
    ]


def get_processed_keys(output_name):
    """Load completed page identities, rejecting ambiguous duplicate records."""
    processed_keys = set()
    with open(output_name, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            key = (str(item["book_name"]), str(item["page_number"]))
            if key in processed_keys:
                raise ValueError(
                    f"Duplicate page {key[0]}/{key[1]} in {output_name} "
                    f"(line {line_number})"
                )
            processed_keys.add(key)
    return processed_keys
