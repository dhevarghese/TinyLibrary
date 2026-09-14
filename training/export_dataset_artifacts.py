"""Export scores, mappings, and verified pre-packing curriculum sequences.

Requires NumPy, but not Torch, model weights, images, or GPU access. Outputs
contain no annotation text. Orders are reconstructed, not training traces.
The output directory must be empty; existing artifacts are never overwritten.
"""
import argparse
import ast
import hashlib
import json
import math
import platform
import re
from pathlib import Path

import numpy as np

from curriculum import build_order

TASKS = ('caption', 'vqa', 'reasoning')
BANDS = ('3_5', '6_8', '9_12', '12')
SCORE_FIELDS = ('fk_grade', 'n_words', 'n_tokens', 'nll', 'nll_qwen', 'nll_gemma')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require(test, message):
    if not test:
        raise ValueError(message)


def code_ast(path, function=None):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    if function:
        tree = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.body and isinstance(node.body[0], ast.Expr):
                value = node.body[0].value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    node.body.pop(0)
    return ast.dump(tree, include_attributes=False)


def split_records(records):
    # Load only the released split function, avoiding the adapter's Torch/PIL
    # imports. The export checks this function against the preserved source.
    path = Path(__file__).with_name('tinylibrary_dataset.py')
    tree = ast.parse(path.read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'split_train_val')
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['split_train_val'](records, val_fraction=0.02, seed=42)


def id_hash(records):
    return hashlib.sha256('\n'.join(r['id'] for r in records).encode()).hexdigest()


def order_specs():
    for epochs in (5, 10):
        for seed in (0, 1, 2):
            for order in ('developmental', 'antidevelopmental', 'readability', 'perplexity', 'random'):
                for schedule in (('staged',) if order == 'random' else ('staged', 'competence')):
                    yield order, schedule, epochs, seed
    for order in ('randomblock', 'qwenppl', 'gemmappl'):
        for seed in (0, 1, 2):
            yield order, 'staged', 10, seed


def validate_records(records):
    seen = set()
    for rec in records:
        require(isinstance(rec.get('id'), str) and bool(rec['id']) and rec['id'] not in seen, 'Missing or duplicate sample ID')
        seen.add(rec['id'])
        require(rec.get('task') in TASKS and rec.get('age_band') in BANDS, f"Invalid task/band: {rec['id']}")
        for field in SCORE_FIELDS:
            value = rec.get(field)
            require(type(value) in (int, float) and math.isfinite(value), f"Invalid {field}: {rec['id']}")
            if field.startswith('nll'):
                require(value >= 0, f"Negative NLL: {rec['id']}")
        for field in ('n_words', 'n_tokens'):
            require(type(rec[field]) is int and rec[field] > 0, f"Invalid count {field}: {rec['id']}")


def validate_annotations(records, annotation_dir):
    """Join by unique composite ID and compare all normalized conversation text."""
    by_id = {r['id']: r for r in records}
    locations = {}
    checksums = {}
    ordered_ids = []
    for task in TASKS:
        for band in BANDS:
            name = f'{task}_{band}.json'
            path = Path(annotation_dir) / name
            checksums[name] = sha256(path)
            with path.open(encoding='utf-8') as handle:
                data = json.load(handle)
            for row, annotation in enumerate(data):
                key = f"{task}_{band}_{annotation['id']}"
                require(key not in locations and key in by_id, f'Unexpected/duplicate annotation: {key}')
                rec = by_id[key]
                require(rec['image'] == annotation['image'], f'Image reference mismatch: {key}')
                require(rec['task'] == task and rec['age_band'] == band, f'Label mismatch: {key}')
                convs = annotation['conversations']
                require(convs and len(convs) % 2 == 0, f'Invalid conversation: {key}')
                turns = []
                for human, assistant in zip(convs[::2], convs[1::2]):
                    require(human['from'] == 'human' and assistant['from'] == 'gpt', f'Invalid roles: {key}')
                    user = re.sub(r'\s*<image>\s*', ' ', human['value']).strip()
                    answer = assistant['value'].strip()
                    require(user and answer, f'Empty conversation text: {key}')
                    turns.append({'user': user, 'assistant': answer})
                require(turns == rec['texts'], f'Annotation/manifest text mismatch: {key}')
                text = ' '.join(t['user'] + ' ' + t['assistant'] for t in turns)
                require(len(text.split()) == rec['n_words'], f'Word-count mismatch: {key}')
                locations[key] = {'annotation_file': name, 'annotation_index': row, 'source_id': annotation['id']}
                ordered_ids.append(key)
    require(ordered_ids == [r['id'] for r in records], 'Annotation order/membership differs from manifest')
    return locations, checksums


def sample_mapping(records, train, validation, locations):
    splits = {r['id']: ('train', i) for i, r in enumerate(train)}
    splits.update({r['id']: ('validation', i) for i, r in enumerate(validation)})
    require(len(splits) == len(records), 'Split membership is not exhaustive and disjoint')
    for i, rec in enumerate(records):
        split, split_index = splits[rec['id']]
        yield {'sample_id': rec['id'], 'manifest_index': i, **locations[rec['id']],
               'image': rec['image'], 'task': rec['task'], 'age_band': rec['age_band'],
               'split': split, 'split_index': split_index}


def validate_sequence(sequence, n_train, epochs, order, schedule):
    require(sequence.ndim == 1 and sequence.dtype.kind in 'iu', 'Sequence must be a 1D integer array')
    require(len(sequence) == n_train * epochs, 'Wrong sequence length')
    require(len(sequence) > 0 and int(sequence.min()) >= 0 and int(sequence.max()) < n_train,
            'Out-of-range sequence index')
    if order == 'random' or schedule == 'staged':
        for epoch in sequence.reshape(epochs, n_train):
            require(np.all(np.bincount(epoch.astype(np.int64), minlength=n_train) == 1),
                    'Staged/random epoch is not a complete permutation')
    counts = np.bincount(sequence.astype(np.int64), minlength=n_train)
    unique = int(np.count_nonzero(counts))
    return {'n_draws': len(sequence), 'n_unique_samples': unique,
            'n_omitted_samples': n_train - unique, 'n_repeated_draws': len(sequence) - unique,
            'min_sample_exposure': int(counts.min()), 'max_sample_exposure': int(counts.max())}


def write_json(path, data):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def write_jsonl(path, rows):
    with Path(path).open('x', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def export(args):
    out = Path(args.output)
    require(not out.exists() or not any(out.iterdir()), 'Output directory must be empty')
    reference = json.loads(Path(args.reference).read_text())
    reference = reference.get('source_order_reference', reference)
    manifest_hash = sha256(args.manifest)
    require(manifest_hash == reference['manifest_sha256'], 'Manifest differs from source reference')
    local = Path(__file__).parent
    source_ast_verified = None
    if args.source_code_dir:
        source = Path(args.source_code_dir)
        require(sha256(source / 'curriculum.py') == reference['curriculum_sha256'], 'Original curriculum hash mismatch')
        require(sha256(source / 'tinycomics_dataset.py') == reference['dataset_code_sha256'], 'Original split source hash mismatch')
        require(code_ast(local / 'curriculum.py') == code_ast(source / 'curriculum.py'), 'Curriculum code has behavioral changes')
        require(code_ast(local / 'tinylibrary_dataset.py', 'split_train_val') ==
                code_ast(source / 'tinycomics_dataset.py', 'split_train_val'), 'Split code has behavioral changes')
        source_ast_verified = True
    with Path(args.manifest).open(encoding='utf-8') as handle:
        records = [json.loads(line) for line in handle]
    require(len(records) == reference['n_samples'] == 128823, 'Not the paper-version manifest')
    validate_records(records)
    locations, annotation_checksums = validate_annotations(records, args.annotations)
    train, validation = split_records(records)
    require((len(train), len(validation)) == (126247, 2576), 'Unexpected split sizes')
    require(id_hash(train) == reference['train_ids_sha256'], 'Training index mapping differs from reference')
    require(id_hash(validation) == reference['validation_ids_sha256'], 'Validation index mapping differs from reference')
    references = {(r['order'], r['schedule'], r['epochs'], r['seed']): r for r in reference['orders']}
    require(len(references) == len(reference['orders']) == 63 and set(references) == set(order_specs()),
            'Reference must contain all 63 unique configurations')
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / 'difficulty_scores.jsonl',
                ({'sample_id': r['id'], **{k: r[k] for k in SCORE_FIELDS}} for r in records))
    write_jsonl(out / 'sample_mapping.jsonl', sample_mapping(records, train, validation, locations))
    orders = []
    for order, schedule, epochs, seed in order_specs():
        sequence = build_order(train, order, schedule, epochs, seed, competence_c0=0.1)
        summary = validate_sequence(sequence, len(train), epochs, order, schedule)
        canonical = np.asarray(sequence, dtype='<u4')
        raw_hash = hashlib.sha256(canonical.tobytes()).hexdigest()
        ref = references[(order, schedule, epochs, seed)]
        require(raw_hash == ref['sha256_indices_u32le'] and len(sequence) == ref['n_draws'],
                f'Source-environment mismatch: {order}/{schedule}/ep{epochs}/seed{seed}')
        name = f"{order}_{schedule}_seed{seed}.npy" if order != 'random' else f'random_seed{seed}.npy'
        relative = Path('curriculum_orders') / f'ep{epochs}' / name
        (out / relative).parent.mkdir(parents=True, exist_ok=True)
        with (out / relative).open('xb') as handle:
            np.save(handle, canonical, allow_pickle=False)
        reloaded = np.load(out / relative, allow_pickle=False)
        require(np.array_equal(canonical, reloaded), 'Order file round-trip mismatch')
        modalities = ['multimodal']
        if epochs == 10 and schedule == 'staged' and order in ('random', 'developmental', 'perplexity'):
            modalities.append('text_only')
        orders.append({'file': relative.as_posix(), 'order': order,
                       'schedule': None if order == 'random' else schedule,
                       'builder_schedule_argument': schedule, 'epochs': epochs, 'seed': seed,
                       'competence_c0': 0.1 if schedule == 'competence' else None,
                       'applies_to': modalities, **summary,
                       'sha256_indices_u32le': raw_hash, 'sha256_file': sha256(out / relative)})
        print(f'Verified {relative}', flush=True)
    write_json(out / 'curriculum_orders/metadata.json', {
        'format_version': 1, 'provenance': 'reconstructed_from_preserved_code_not_saved_training_trace',
        'index_space': 'train rows of sample_mapping.jsonl ordered by split_index (zero-based)',
        'array_dtype': '<u4', 'sequence_stage': 'before dataloader sharding, batching, packing, and stopping',
        'n_unique_sequences': len(orders), 'n_training_configurations_covered': sum(len(r['applies_to']) for r in orders),
        'orders': orders,
    })
    source_environment = {k: reference[k] for k in ('python', 'numpy', 'curriculum_sha256', 'dataset_code_sha256',
                                                   'train_ids_sha256', 'validation_ids_sha256')}
    write_json(out / 'export_metadata.json', {
        'format_version': 1, 'n_samples': len(records), 'n_train': len(train), 'n_validation': len(validation),
        'split_seed': 42, 'validation_fraction': 0.02,
        'source_manifest': {'filename': Path(args.manifest).name, 'sha256': manifest_hash},
        'annotation_sha256': annotation_checksums, 'source_environment': source_environment,
        'export_environment': {'python': platform.python_version(), 'numpy': np.__version__},
        'verification': {'normalized_annotation_text_matches_manifest': True, 'score_values_copied_without_rescoring': True,
                         'curriculum_and_split_executable_ast_match_source': source_ast_verified,
                         'all_63_sequence_hashes_match_source_environment': True},
        'exporter_sha256': sha256(__file__), 'released_curriculum_sha256': sha256(local / 'curriculum.py'),
        'source_order_reference': reference,
        'score_semantics': {'fk_grade': 'joined prompt and response text; image placeholder removed',
                            'nll': 'SmolLM2-135M; initial prompt masked; later user turns included',
                            'nll_qwen': 'Qwen3-0.6B; same scoring convention',
                            'nll_gemma': 'Gemma-3-270M; same scoring convention',
                            'n_tokens': 'primary-teacher non-padding tokens after truncation, including initial prompt'},
    })
    files = sorted(p for p in out.rglob('*') if p.is_file())
    with (out / 'artifacts.sha256').open('x', encoding='utf-8') as handle:
        for path in files:
            handle.write(f'{sha256(path)}  {path.relative_to(out).as_posix()}\n')
    print(f'Export complete: {len(records)} scores, {len(records)} mappings, {len(orders)} sequences.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--source-code-dir', help='Optional preserved source directory for additional executable-code comparison')
    parser.add_argument('--reference', required=True, help='Fingerprints produced in the preserved source environment')
    parser.add_argument('--output', required=True, help='New or empty staging directory')
    export(parser.parse_args())


if __name__ == '__main__':
    main()
