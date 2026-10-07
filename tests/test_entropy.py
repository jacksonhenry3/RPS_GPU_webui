import sys
from pathlib import Path
import unittest

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from entropy import bank_labels, state_labels, measure


class EntropyTests(unittest.TestCase):
    def test_all_nine_states_uniform(self):
        metrics, dist = measure(np.arange(9))
        self.assertAlmostEqual(dist['joint'].sum(), 1)
        self.assertAlmostEqual(metrics['product_bits'], np.log2(9))
        self.assertAlmostEqual(metrics['joint_normalized'], 1)
        self.assertAlmostEqual(metrics['strategy_bank_mi_bits'], 0)

    def test_correlated_product_is_not_joint(self):
        metrics, _ = measure(np.array([0, 4, 8]))
        self.assertAlmostEqual(metrics['product_bits'], np.log2(9))
        self.assertAlmostEqual(metrics['joint_bits'], np.log2(3))
        self.assertAlmostEqual(metrics['strategy_bank_mi_bits'], np.log2(3))

    def test_constant_and_tied_banks(self):
        np.testing.assert_array_equal(bank_labels(np.ones(12)), 0)
        labels = bank_labels(np.array([0, 0, 1, 1, 2, 2]), bins=3)
        np.testing.assert_array_equal(labels, [0, 0, 1, 1, 2, 2])
        metrics, _ = measure(state_labels(np.eye(3), np.ones(3)))
        self.assertEqual(metrics['bank_bits'], 0)

    def test_fixed_thresholds_and_validation(self):
        np.testing.assert_array_equal(bank_labels(np.array([-100, 0, 1, 100]), 3, [0, 1]), [0, 1, 2, 2])
        for bins, edges in ((0, None), (3, [1]), (3, [1, 1]), (3, [np.nan, 2])):
            with self.assertRaises(ValueError):
                bank_labels(np.ones(3), bins, edges)
        with self.assertRaises(ValueError):
            bank_labels(np.array([np.nan]))

    def test_temporal_identity_and_independence(self):
        states = np.arange(9)
        metrics, _ = measure(states, previous=states)
        self.assertAlmostEqual(metrics['next_given_current_bits'], 0)
        self.assertAlmostEqual(metrics['temporal_mi_bits'], np.log2(9))
        metrics, dist = measure(np.tile(states, 9), previous=np.repeat(states, 9))
        self.assertAlmostEqual(metrics['next_given_current_bits'], np.log2(9))
        self.assertAlmostEqual(metrics['temporal_mi_bits'], 0)
        self.assertAlmostEqual(metrics['temporal_joint_normalized'], 1)
        self.assertEqual(len(dist['transition_ids']), 81)

    def test_single_bank_bin(self):
        metrics, _ = measure(np.arange(3), bins=1)
        self.assertAlmostEqual(metrics['joint_normalized'], 1)
        self.assertEqual(metrics['bank_bits'], 0)

    def test_cupy_parity_when_available(self):
        try:
            import cupy as cp
            cp.zeros(1)
        except (ImportError, RuntimeError):
            self.skipTest('CUDA/CuPy unavailable')
        labels = np.array([0, 4, 8, 0, 0, 3])
        cpu, _ = measure(labels, previous=labels[::-1])
        gpu, _ = measure(cp.asarray(labels), previous=cp.asarray(labels[::-1]), xp=cp)
        for key in cpu:
            self.assertAlmostEqual(cpu[key], gpu[key])


if __name__ == '__main__':
    unittest.main()
