"""Runtime asset paths stay out of the shared training recipe."""
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from hand_kd.train import load_training_config


class TrainingConfigTests(unittest.TestCase):
    def test_resolves_cli_assets_and_detects_changed_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recipe = root/'train.json'
            recipe.write_text(json.dumps(dict(seed=7, epochs=38, batch_size=32, workers=0,
                lr=.001, crop_base_scale=1.5, feature_weight=1., checkpoint_interval_batches=10)))
            manifest = root/'manifest.json'; manifest.write_text('{"upstream_revision":"abc"}')
            selection = root/'selection.json'; selection.write_text('{}')
            teacher = root/'teacher.pt'; teacher.write_bytes(b'first')
            validation = root/'validation.json'; validation.write_text('{}')
            cache = root/'cache'; cache.mkdir()
            (cache/'index.json').write_text('{"regressor_sha256":"reg"}')
            template = root/'external/mobrecon/template'; template.mkdir(parents=True)
            for name in ['right_faces.npy', 'transform.pkl']:
                (template/name).write_bytes(b'template')
            args = Namespace(config=recipe, dataset_manifest=manifest, selection=selection,
                             teacher=teacher, validation_manifest=validation, validation_cache=cache)
            first = load_training_config(args, root=root)
            self.assertEqual(first['seed'], 7)
            self.assertEqual(first['validation']['manifest'], str(validation))
            self.assertEqual(first['teacher_checkpoint'], str(teacher))
            teacher.write_bytes(b'changed')
            second = load_training_config(args, root=root)
            self.assertNotEqual(first['teacher_checkpoint_sha256'], second['teacher_checkpoint_sha256'])
            self.assertNotIn('dataset_manifest', json.loads(recipe.read_text()))

    def test_rejects_unknown_or_invalid_training_parameters_before_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            recipe = Path(tmp)/'train.json'
            recipe.write_text('{"authorization":"unused descriptive field"}')
            with self.assertRaises(ValueError):
                load_training_config(Namespace(config=recipe))


if __name__ == '__main__':
    unittest.main()
