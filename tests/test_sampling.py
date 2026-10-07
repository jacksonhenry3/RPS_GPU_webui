"""Deterministic endpoint regressions; NumPy stands in for CuPy on the CPU."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))


def load_algorithm():
    spec = importlib.util.spec_from_file_location('sampling_algorithm', SRC / 'algorithm.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SamplingTests(unittest.TestCase):
    def setUp(self):
        # Restore imports after each test; do not masquerade as real GPU tests.
        self.modules = patch.dict(sys.modules, {'cupy': np})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        for name in ('initial_conditions', 'measurements'):
            sys.modules.pop(name, None)
        self.algorithm = load_algorithm()

    def sample(self, probabilities, draws):
        probabilities = np.asarray(probabilities, dtype=np.float32)
        system = self.algorithm.AgentSystem.__new__(self.algorithm.AgentSystem)
        system.N = probabilities.shape[1]
        with patch.object(self.algorithm.cp.random, 'rand', return_value=np.asarray(draws, dtype=np.float64)):
            system._sample_strategies(probabilities)
        self.assertEqual(system.agent_strategies.dtype, np.float32)
        return system.agent_strategies.argmax(axis=0)

    def test_draw_rounding_to_one_cannot_select_rock(self):
        draw = 1.0 - 2**-26
        self.assertEqual(np.float32(draw), 1.0)  # reproduces the original bug
        np.testing.assert_array_equal(self.sample([[0, 0], [1, 0], [0, 1]], [draw, draw]), [1, 2])

    def test_cdf_shortfall_and_zero_probability_last_strategy(self):
        # Both float32 and float64 accumulation give a sum below one here.
        # Setting only the final CDF entry to 1 would incorrectly select S.
        probs = np.array([[0], [np.nextafter(np.float32(1), np.float32(0))], [0]])
        np.testing.assert_array_equal(self.sample(probs, [np.nextafter(1., 0.)]), [1])

    def test_strict_boundaries_skip_zero_probability_intervals(self):
        np.testing.assert_array_equal(self.sample([[0, .5, .5], [1, 0, .5], [0, .5, 0]], [0, .5, .5]), [1, 2, 1])

    def test_known_categorical_frequencies(self):
        # Stratified draws avoid a flaky probabilistic tolerance.
        draws = (np.arange(10000) + .5) / 10000
        probs = np.repeat(np.array([[.125], [.375], [.5]]), len(draws), axis=1)
        np.testing.assert_array_equal(np.bincount(self.sample(probs, draws)), [1250, 3750, 5000])

    def test_both_stochastic_selection_modes_use_safe_sampler(self):
        for mode in ('choose_new_strats_local_choice', 'choose_new_strats_random'):
            with self.subTest(mode=mode):
                system = self.algorithm.AgentSystem.__new__(self.algorithm.AgentSystem)
                system.N, system.beta, system.precision = 2, 1., np.float32
                system._worst_exp_arg = None
                system.agent_strategies = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
                system._neighbor_strategy_counts = np.ones((3, 2), dtype=np.float32)
                system._calculate_neighborhood_scores = lambda: np.array([[-1000, -1000], [0, -1000], [-1000, 0]], dtype=np.float32)
                with patch.object(self.algorithm.cp.random, 'rand', return_value=np.full(2, 1.-2**-26)):
                    getattr(system, mode)()
                np.testing.assert_array_equal(system.agent_strategies.argmax(axis=0), [1, 2])


if __name__ == '__main__':
    unittest.main()
