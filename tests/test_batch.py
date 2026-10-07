"""Exercise orchestration with NumPy/SciPy standing in for unavailable CUDA.

This checks the real AgentSystem and runner, but is NOT GPU validation.
"""
import argparse
import csv
import importlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]


class BatchTests(unittest.TestCase):
    def setUp(self):
        cp = types.ModuleType('cupy')
        cp.__dict__.update({k: getattr(np, k) for k in dir(np) if not k.startswith('__')})
        cp.__version__ = 'numpy-test-double'
        cp.asnumpy = np.asarray
        cp.cuda = types.SimpleNamespace(
            Device=lambda: types.SimpleNamespace(id=0),
            runtime=types.SimpleNamespace(runtimeGetVersion=lambda: 0, driverGetVersion=lambda: 0,
                                         getDeviceProperties=lambda _: {'name': b'CPU test double'}))
        self.modules = patch.dict(sys.modules, {'cupy': cp, 'cupyx': types.ModuleType('cupyx'),
                                 'cupyx.scipy': types.ModuleType('cupyx.scipy'),
                                 'cupyx.scipy.sparse': scipy.sparse, 'networkx': types.ModuleType('networkx')})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        for name in ('algorithm', 'initial_conditions', 'measurements', 'networks', 'simulation'):
            sys.modules.pop(name, None)
        self.simulation = importlib.import_module('simulation')
        self.batch = importlib.import_module('batch')
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.config = self.root / 'config.json'
        self.config.write_text(json.dumps({'numAgents': 8, 'bankBins': 3}))
        self.args = argparse.Namespace(config=self.config, out=self.root / 'run', steps=7, seed=51,
                                       sample_every=3, snapshot_every=2, temporal=True)

    def tearDown(self):
        self.tmp.cleanup()
        for name in ('algorithm', 'initial_conditions', 'measurements', 'networks', 'simulation'):
            sys.modules.pop(name, None)
        self.modules.stop()

    def test_run_matches_direct_and_plots_without_gpu(self):
        from entropy import state_labels, measure
        np.random.seed(self.args.seed)
        direct = self.simulation.Simulation({'numAgents': 8}, visualize=False)
        expected = {}
        for step in range(1, 8):
            system = direct.agent_system
            previous = state_labels(system.agent_strategies, system.agent_bank_values)
            direct.step()
            if step in (3, 6, 7):
                labels = state_labels(system.agent_strategies, system.agent_bank_values)
                expected[step] = measure(labels, previous=previous)[0]
        self.assertIsNone(direct.visualizer)
        self.assertNotIn('visualization', sys.modules)
        self.assertEqual(self.batch.run(self.args), 0)
        final = np.load(self.args.out / 'final.npz')
        np.testing.assert_array_equal(final['strategies'], direct.agent_system.agent_strategies.argmax(axis=0))
        np.testing.assert_array_equal(final['bank'], direct.agent_system.agent_bank_values)
        with (self.args.out / 'measurements.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([int(r['step']) for r in rows], [0, 3, 6, 7])
        for row in rows[1:]:
            self.assertAlmostEqual(float(row['temporal_joint_bits']), expected[int(row['step'])]['temporal_joint_bits'])
        meta = json.loads((self.args.out / 'metadata.json').read_text())
        self.assertEqual(meta['status'], 'complete')
        self.assertEqual(len(list(self.args.out.glob('state_*.npz'))), 3)
        # New process: importing plotting must not load CuPy, Flask, or codecs.
        result = subprocess.run([sys.executable, str(ROOT / 'batch.py'), 'plot', str(self.args.out),
                                 '--out', str(self.root / 'plots')], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(list((self.root / 'plots').glob('*.png'))), 5)
        with self.assertRaises(FileExistsError):
            self.batch.run(self.args)

    def test_failure_is_recorded(self):
        with patch.object(self.simulation.Simulation, 'step', side_effect=RuntimeError('test failure')):
            with self.assertRaises(RuntimeError):
                self.batch.run(self.args)
        self.assertEqual(json.loads((self.args.out / 'metadata.json').read_text())['status'], 'failed')
        self.assertTrue((self.args.out / 'initial.npz').exists())

    def test_graceful_interrupt(self):
        original = self.simulation.Simulation.step
        def interrupted(sim):
            original(sim)
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        with patch.object(self.simulation.Simulation, 'step', interrupted):
            self.assertEqual(self.batch.run(self.args), 130)
        meta = json.loads((self.args.out / 'metadata.json').read_text())
        self.assertEqual(meta['status'], 'interrupted')
        self.assertEqual(meta['steps_completed'], 1)
        self.assertEqual(int(np.load(self.args.out / 'final.npz')['step']), 1)


if __name__ == '__main__':
    unittest.main()
