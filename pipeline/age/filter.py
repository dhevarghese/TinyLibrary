import argparse
import glob
import json
import os


def read_jsonl(file_path):
    with open(file_path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def normalize_book_name(value):
    """Match postprocessed label filenames to description/shard book IDs."""
    book_name = os.path.basename(str(value))
    for suffix in (".jsonl", ".json"):
        if book_name.endswith(suffix):
            return book_name[:-len(suffix)]
    return book_name


def load_age_group_data(file_path):
    age_group_data = {}
    for obj in read_jsonl(file_path):
        age_group_data[normalize_book_name(obj["book_name"])] = obj["age-group"]
    return age_group_data

def page_candidates(library_dir, book_name, page_number):
    """Return supported locations for one retained page, flat layout first."""
    page_stem = os.path.splitext(str(page_number))[0]
    return (
        os.path.join(library_dir, f"{book_name}_{page_stem}.jpg"),
        os.path.join(library_dir, book_name, f"{page_stem}.jpg"),
    )


def filter_captions_by_age_group(age_group_data, captions_folder, library_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    total = written = missing_age = missing_image = 0
    missing_age_examples = []
    missing_examples = []
    output_handles = {}
    try:
        for jsonl_file in sorted(glob.glob(os.path.join(captions_folder, "*.jsonl"))):
            for line in read_jsonl(jsonl_file):
                total += 1
                book_name = line["book_name"]
                page_number = line["page_number"]
                age_group = age_group_data.get(book_name)
                if age_group is None:
                    missing_age += 1
                    if len(missing_age_examples) < 10:
                        missing_age_examples.append(f"{book_name}/{page_number}")
                    continue
                if not any(os.path.isfile(path) for path in page_candidates(
                        library_dir, book_name, page_number)):
                    missing_image += 1
                    if len(missing_examples) < 10:
                        missing_examples.append(f"{book_name}/{page_number}")
                    continue
                if age_group not in output_handles:
                    output_path = os.path.join(save_dir, f"library-{age_group}.jsonl")
                    output_handles[age_group] = open(output_path, "a", encoding="utf-8")
                json.dump(line, output_handles[age_group], ensure_ascii=False)
                output_handles[age_group].write("\n")
                written += 1
    finally:
        for handle in output_handles.values():
            handle.close()

    print(
        f"Age-label join: wrote {written}/{total} descriptions; "
        f"missing age label={missing_age}, missing retained image={missing_image}."
    )
    if missing_examples:
        print("Missing-image examples: " + ", ".join(missing_examples))
    if missing_age_examples:
        print("Missing-age-label examples: " + ", ".join(missing_age_examples))

def main():
    parser = argparse.ArgumentParser(description="Retrieve captions / pages present both in library and age group data.")
    parser.add_argument('--age_group_file', type=str, default='age-category.jsonl',
                        help='The file containing age group data')
    parser.add_argument('--captions_folder', '--descriptions_dir', dest='captions_folder',
                        type=str, default='captions/',
                        help='Directory containing page-description JSONL shards')
    parser.add_argument('--library_dir', type=str, default='library',
                        help='Directory of retained images (flat book_page.jpg layout; nested layout is also accepted)')
    parser.add_argument('--save_dir', type=str, default='filtered-captions',
                        help='The directory to save the filtered captions')
    args = parser.parse_args()

    age_group_data = load_age_group_data(args.age_group_file)
    filter_captions_by_age_group(age_group_data, args.captions_folder, args.library_dir, args.save_dir)

if __name__ == "__main__":
    main()
