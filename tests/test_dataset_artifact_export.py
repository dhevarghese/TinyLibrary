"""Dependency-light checks for the dataset release exporter (NumPy required)."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

TRAINING = Path(__file__).resolve().parents[1] / 'training'
sys.path.insert(0, str(TRAINING))
spec = importlib.util.spec_from_file_location('dataset_export', TRAINING / 'export_dataset_artifacts.py')
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def records(n=100):
    return [{'id': f'caption_3_5_page{i}', 'image': f'page{i}.jpg', 'task': 'caption',
             'age_band': exporter.BANDS[i % 4], 'texts': [{'user': 'Describe.', 'assistant': 'A cat.'}],
             'n_words': 3, 'n_tokens': 10, 'fk_grade': float(i), 'nll': 1.0 + i / n,
             'nll_qwen': 2.0 + i / n, 'nll_gemma': 3.0 + i / n} for i in range(n)]


class ScoreValidationTests(unittest.TestCase):
    def test_valid_scores_and_negative_readability(self):
        rows = records()
        rows[0]['fk_grade'] = -2.5
        exporter.validate_records(rows)

    def test_missing_nonfinite_negative_and_duplicate_values_fail(self):
        for value in (None, float('nan'), float('inf'), -1):
            rows = records()
            rows[0]['nll'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                exporter.validate_records(rows)
        with self.assertRaises(ValueError):
            exporter.validate_records([records()[0], records()[0]])


class MappingTests(unittest.TestCase):
    def test_split_is_deterministic_disjoint_and_preserves_manifest_order(self):
        rows = records()
        train, val = exporter.split_records(rows)
        self.assertEqual((len(train), len(val)), (98, 2))
        self.assertEqual(exporter.split_records(rows), (train, val))
        locations = {r['id']: {'annotation_file': 'fixture.json', 'annotation_index': i,
                                'source_id': f'page{i}'} for i, r in enumerate(rows)}
        mapping = list(exporter.sample_mapping(rows, train, val, locations))
        self.assertEqual([m['manifest_index'] for m in mapping], list(range(100)))
        for m in mapping:
            subset = train if m['split'] == 'train' else val
            self.assertEqual(subset[m['split_index']]['id'], m['sample_id'])

    def test_annotation_join_checks_text_and_file_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for task in exporter.TASKS:
                for band in exporter.BANDS:
                    item = {'id': 'page', 'image': 'page.jpg', 'conversations': [
                        {'from': 'human', 'value': '<image>\nDescribe.'},
                        {'from': 'gpt', 'value': 'A cat.'}]}
                    (root / f'{task}_{band}.json').write_text(json.dumps([item]))
                    rec = records(1)[0]
                    rec.update(id=f'{task}_{band}_page', image='page.jpg', task=task, age_band=band)
                    rows.append(rec)
            locations, hashes = exporter.validate_annotations(rows, root)
            self.assertEqual((len(locations), len(hashes)), (12, 12))
            rows[0]['texts'][0]['assistant'] = 'A dog.'
            with self.assertRaisesRegex(ValueError, 'text mismatch'):
                exporter.validate_annotations(rows, root)


class OrderTests(unittest.TestCase):
    def test_grid_has_63_sequences_and_72_model_configurations(self):
        specs = list(exporter.order_specs())
        self.assertEqual(len(specs), 63)
        self.assertEqual(len(set(specs)), 63)
        text_only = [x for x in specs if x[2] == 10 and x[1] == 'staged'
                     and x[0] in ('random', 'developmental', 'perplexity')]
        self.assertEqual(len(specs) + len(text_only), 72)

    def test_every_order_is_deterministic_and_has_valid_indices(self):
        rows = records()
        for order, schedule, _, seed in exporter.order_specs():
            with self.subTest(order=order, schedule=schedule, seed=seed):
                seq = exporter.build_order(rows, order, schedule, 2, seed)
                self.assertTrue(np.array_equal(seq, exporter.build_order(rows, order, schedule, 2, seed)))
                summary = exporter.validate_sequence(seq, len(rows), 2, order, schedule)
                self.assertEqual(summary['n_draws'], 200)
                self.assertEqual(summary['n_unique_samples'] + summary['n_omitted_samples'], 100)
                if schedule == 'staged':
                    self.assertEqual(summary['min_sample_exposure'], 2)
                    self.assertEqual(summary['max_sample_exposure'], 2)

    def test_invalid_sequences_are_rejected(self):
        with self.assertRaises(ValueError):
            exporter.validate_sequence(np.array([0, 0]), 2, 1, 'random', 'staged')
        with self.assertRaises(ValueError):
            exporter.validate_sequence(np.array([0, 2]), 2, 1, 'perplexity', 'competence')

    def test_existing_json_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'data.json'
            exporter.write_json(path, {'original': True})
            with self.assertRaises(FileExistsError):
                exporter.write_json(path, {'original': False})
            self.assertEqual(json.loads(path.read_text()), {'original': True})


if __name__ == '__main__':
    unittest.main()
