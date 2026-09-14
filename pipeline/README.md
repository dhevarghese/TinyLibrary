# TinyLibrary construction pipeline

Use the separate Python 3.11 construction environment described in [`environment/README.md`](../environment/README.md). Its Transformers version is pinned for CogVLM2 compatibility and must not be replaced with the training/evaluation environment.

The pipeline begins with source files obtained independently: one directory of page images per book and OCR/PaddleOCR annotations for those pages. It intentionally contains no ICDL downloader.

The construction stages used for TinyLibrary are:

1. `text/clean_library.py` cleans OCR text with Mistral-7B-Instruct, using the examples in `playground/prompts/cleaning/`.
2. `age/categorize_library.py` assigns book-text chunks to age bands, using `playground/prompts/age-group/`; `age/postprocess.py` takes the majority vote for each book.
3. `image/filter_pages.py` applies the text-area and edge-content filters reported in the paper. It copies retained original images to a flat output directory; the text-masked images are used only during filtering.
4. `generate_shards.py` packages retained images as WebDataset shards, and `image/generate_detailed_captions.py` generates CogVLM2 page descriptions.
5. `age/filter.py` joins the page descriptions to the inferred book-level age labels and verifies that the corresponding retained images exist.
6. `llava_format/generate_responses.py` turns the age-grouped descriptions into age-conditioned captioning, VQA, or reasoning examples using `meta-llama/Meta-Llama-3-8B-Instruct`. The default revision is frozen at `e1945c40cd546c78e41f1151f4db032b271faeaa`; access to this gated model must be configured with Hugging Face before generation. The original jobs did not log their resolved Hugging Face commit; this was the repository state current when the annotation script was added and supplies an immutable reproduction target.
7. The scripts in `llava_format/` clean the generations and convert them to the LLaVA-style JSON consumed by `training/prepare_manifest.py`.

Every script exposes its paths and model choices through command-line options; run a script with `--help` for its exact interface. Large-model inference in the original pipeline uses DeepSpeed-MII. The prompt files are part of the experimental record and should not be changed when reproducing the published corpus.

Join the generated page descriptions to the book-level age labels and the flat image directory with:

```bash
python pipeline/age/filter.py \
    --age_group_file /path/to/age-category.jsonl \
    --descriptions_dir /path/to/page-descriptions \
    --library_dir /path/to/retained-images \
    --save_dir /path/to/filtered-captions
```

The command reports descriptions missing either an age label or a retained image and writes `library-3-5.jsonl`, `library-6-8.jsonl`, `library-9-12.jsonl`, and `library-12+.jsonl` as applicable. Generate all three tasks for all four age bands with:

```bash
python pipeline/llava_format/generate_all_responses.py \
    --descriptions_dir /path/to/filtered-captions \
    --output_dir /path/to/generated-responses
```

This creates the complete matrix below. The output stems are preserved by the conversion scripts and are the filenames expected by `training/prepare_manifest.py`.

| Task | Prompt directory | Output stems |
|---|---|---|
| Caption | `captions/` | `caption_{3_5,6_8,9_12,12}` |
| VQA | `conversations/` | `vqa_{3_5,6_8,9_12,12}` |
| Reasoning | `reasoning/` | `reasoning_{3_5,6_8,9_12,12}` |

Use `--dry_run` to print the 12 fully expanded generation commands. Convert their JSONL outputs into one manifest-ready directory as follows:

```bash
python pipeline/llava_format/convert_captions_to_llava.py \
    --dataset_dir /path/to/generated-responses/caption \
    --clean_dir /path/to/generated-json \
    --patterns_file pipeline/llava_format/caption_patterns.txt \
    --errors_file pipeline/llava_format/err_patterns.txt \
    --instructs_file pipeline/llava_format/caption_instructs.txt

python pipeline/llava_format/convert_to_llava.py \
    --dataset_dir /path/to/generated-responses/vqa \
    --clean_dir /path/to/generated-json \
    --patterns_file pipeline/llava_format/vqa_patterns.txt \
    --errors_file pipeline/llava_format/err_patterns.txt

python pipeline/llava_format/convert_to_llava.py \
    --dataset_dir /path/to/generated-responses/reasoning \
    --clean_dir /path/to/generated-json \
    --patterns_file pipeline/llava_format/reasoning_patterns.txt \
    --errors_file pipeline/llava_format/err_patterns.txt

python training/prepare_manifest.py \
    --json_dir /path/to/generated-json \
    --image_root /path/to/retained-images \
    --out /path/to/manifest.jsonl
```

OCR cleaning uses NLTK's `words`, `punkt`, and `punkt_tab` resources. Install them once with `python -m nltk.downloader words punkt punkt_tab` before running `clean_library.py`.

The page-filtering thresholds are the values reported in the paper: detected text may cover at most 30% of a page, the initial normalized Canny-edge score must be at least 0.2, and the score after masking detected text must be at least 0.3.