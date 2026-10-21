"""Filesystem-only smoke/production isolation tests; no GPU or CFD data needed."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

if __package__:
    from . import base_comparison as base, run_sensors as runner
else:
    import base_comparison as base
    import run_sensors as runner


class OutputSafetyTests(unittest.TestCase):
    @staticmethod
    def arguments(root, *, smoke, task=0):
        return SimpleNamespace(root=str(root), source='unused', checkpoints='unused',
            split_dir='unused', task=task, smoke=smoke)

    def test_all_tasks_have_disjoint_smoke_and_production_paths(self):
        production, smoke = set(), set()
        for task in range(36):
            for is_smoke, target in ((False, production), (True, smoke)):
                with patch.object(base, 'evaluate') as evaluate:
                    runner.evaluate(self.arguments(Path('unused_root'), smoke=is_smoke, task=task))
                self.assertEqual(evaluate.call_count, 3)
                target.update(Path(call.args[0].output_dir) for call in evaluate.call_args_list)
        self.assertEqual(len(production), 108)
        self.assertEqual(len(smoke), 108)
        self.assertFalse(production & smoke)

    def test_existing_and_completed_conditions_are_not_overwritten(self):
        with TemporaryDirectory(prefix='fno_sensor_safety_') as temporary:
            output = Path(temporary)/'results'/'condition'
            base.claim_output_directory(output)
            with self.assertRaisesRegex(FileExistsError, 'existing or in-progress'):
                base.claim_output_directory(output)
            (output/'COMPLETE.json').write_text('{"n_targets":145}', encoding='utf-8')
            (output/'forward.csv').write_text('preserved production result', encoding='utf-8')
            with self.assertRaisesRegex(FileExistsError, 'completed'):
                base.evaluate(SimpleNamespace(output_dir=str(output)))
            self.assertEqual((output/'forward.csv').read_text(encoding='utf-8'), 'preserved production result')

    def test_smoke_after_completed_production_leaves_production_untouched(self):
        with TemporaryDirectory(prefix='fno_sensor_safety_') as temporary:
            root = Path(temporary)
            production = root/'results'/'layout0_n4_seed7_noise0'
            base.claim_output_directory(production)
            sentinel = production/'COMPLETE.json'
            sentinel.write_text('preserved production marker', encoding='utf-8')
            claimed = []

            def fake_evaluation(arguments):
                claimed.append(base.claim_output_directory(arguments.output_dir))

            with patch.object(base, 'evaluate', side_effect=fake_evaluation):
                runner.evaluate(self.arguments(root, smoke=True))
            self.assertEqual(len(claimed), 3)
            self.assertTrue(all(path.is_relative_to(root/'smoke'/'task0') for path in claimed))
            self.assertEqual(sentinel.read_text(encoding='utf-8'), 'preserved production marker')

    def test_parallel_workers_cannot_claim_the_same_condition(self):
        with TemporaryDirectory(prefix='fno_sensor_safety_') as temporary:
            output = Path(temporary)/'results'/'condition'

            def claim(_):
                try:
                    base.claim_output_directory(output)
                    return True
                except FileExistsError:
                    return False

            with ThreadPoolExecutor(max_workers=8) as executor:
                results = list(executor.map(claim, range(8)))
            self.assertEqual(sum(results), 1)


if __name__ == '__main__':
    unittest.main()
