#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Unit tests for the partitioning code.

Authors: 
* Jean-Gabriel Young <jean-gabriel.young@uvm.edu>
* Alice Patania <alice.patania@uvm.edu>
"""
import unittest
from dyvider import utilities
from dyvider.objectives import Partitioning
from dyvider.layers import Layers
from dyvider import algorithms
import networkx as nx

class TestPartition(unittest.TestCase):
    """Test suite the partitioning functions."""

    def _test_algorithm_w_partitioning(self, algo):
        """Generic test of an algorithm with the Partioning objective."""

        # create graph
        g = nx.Graph()
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        g.add_edge(0, 3)
        g.add_edge(1, 2)
        scores = {0: 5.0, 1: 4.0, 2: 3.0, 3: 2.0}
        nx.set_node_attributes(g, scores, 'score')
        g = utilities.preprocess(g)

        # test small penalty
        objective = Partitioning(penalty=0.1)
        layers, Q = algo(g, objective)
        target_layers = Layers.from_sets([{0, 1, 2, 3}])
        self.assertEqual(layers, target_layers)
        self.assertAlmostEqual(Q, 2.4)

        # test mild penalty
        # (multiple solution, so we only test the objective's value)
        objective = Partitioning(penalty=1/3)
        layers, Q = algo(g, objective)
        self.assertAlmostEqual(Q, 3 - 10 / 3)

        # test large penalty
        objective = Partitioning(penalty=1)
        layers, Q = algo(g, objective)
        target_layers = Layers.from_sets([{0}, {1}, {2}, {3}])
        self.assertEqual(layers, target_layers)
        self.assertAlmostEqual(Q, -4)

    def test_naive_dp(self):
        """Test that the naive dynamical program works."""
        self._test_algorithm_w_partitioning(algorithms.naive_dp)

    def test_pair_sum_dp(self):
        """Test that the "pair-sum" dynamical program works."""
        self._test_algorithm_w_partitioning(algorithms.pair_sum_dp)

    def test_auto_algo(self):
        """Test that the high-level interface works."""
        self._test_algorithm_w_partitioning(algorithms.run)

    def test_brute_force(self):
        """Test that the brute force algorithm works."""
        # run the standard test
        self._test_algorithm_w_partitioning(algorithms.brute_force)

        # test the parameter controlling the number of layers
        g = nx.Graph()
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        g.add_edge(0, 3)
        g.add_edge(1, 2)
        scores = {0: 5.0, 1: 4.0, 2: 3.0, 3: 2.0}
        nx.set_node_attributes(g, scores, 'score')
        g = utilities.preprocess(g)
        objective = Partitioning(penalty=1/3)

        layers, Q = algorithms.brute_force(g, objective, num_layers=2)
        target_layers = Layers.from_sets([{0, 1, 2}, {3}])
        self.assertEqual(layers, target_layers)

        layers, Q = algorithms.brute_force(g, objective, num_layers=3)
        self.assertAlmostEqual(Q, -1.0)

    def test_greedy(self):
        """Test that the greedy algorithm works."""
        g = nx.Graph()
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        g.add_edge(0, 3)
        g.add_edge(1, 2)
        scores = {0: 5.0, 1: 4.0, 2: 3.0, 3: 2.0}
        nx.set_node_attributes(g, scores, 'score')
        g = utilities.preprocess(g)
        objective = Partitioning(penalty=1/3)
        layers, Q = algorithms.greedy(g, objective)
        # the only thing we can guarantee is that the solution will be optimal
        # at best, but probably suboptimal
        self.assertTrue(Q <= 3 - 10/3 + 1e-14)

    def test_heap_merge(self):
        """Test the heap_merge algorithm."""
        g = nx.Graph()
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        g.add_edge(0, 3)
        g.add_edge(1, 2)
        scores = {0: 5.0, 1: 4.0, 2: 3.0, 3: 2.0}
        nx.set_node_attributes(g, scores, 'score')
        g = utilities.preprocess(g)
        objective = Partitioning(penalty=1/3)
        layers, Q, Q_hist = algorithms.heap_merge(g, objective, True)
        # we can guarantee that the solution will be optimal at best, but
        # probably suboptimal
        self.assertTrue(Q <= 3 - 10/3 + 1e-14)
        # we can also test the number of merges, which should be equal to the
        # number of nodes
        self.assertTrue(len(Q_hist) == 4)
        # assert that direct evaluation of the objective is the same as the
        # iteratively determined objective
        self.assertAlmostEqual(Q, objective.eval(g, layers))

    def test_pair_sum_heap_merge(self):
        """Test the the pair sum version of the heap_merge algorithm."""
        g = nx.Graph()
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        g.add_edge(0, 3)
        g.add_edge(1, 2)
        scores = {0: 5.0, 1: 4.0, 2: 3.0, 3: 2.0}
        nx.set_node_attributes(g, scores, 'score')
        g = utilities.preprocess(g)
        objective = Partitioning(penalty=1/3)
        layers, Q, Q_hist = algorithms.pair_sum_heap_merge(g, objective, True)
        # we can guarantee that the solution will be optimal at best, but
        # probably suboptimal
        self.assertTrue(Q <= 3 - 10/3 + 1e-14)
        # we can also test the number of merges, which should be equal to the
        # number of nodes
        self.assertTrue(len(Q_hist) == 4)
        # assert that direct evaluation of the objective is the same as the
        # iteratively determined objective
        self.assertAlmostEqual(Q, objective.eval(g, layers))

    def test_critical_gap(self):
        """Test the critical gap algorithm."""
        g = nx.Graph()
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        g.add_edge(0, 3)
        g.add_edge(1, 2)
        scores = {0: 5.0, 1: 4.0, 2: 3.0, 3: 2.0}
        nx.set_node_attributes(g, scores, 'score')
        g = utilities.preprocess(g)
        objective = Partitioning(penalty=1/3)
        layers, Q, Q_hist = algorithms.critical_gap(g, objective, True)
        # we can guarantee that the solution will be optimal at best, but
        # probably suboptimal
        self.assertTrue(Q <= 3 - 10/3 + 1e-14)
        # we can also test the number of merges, which should be equal to the
        # number of nodes
        self.assertTrue(len(Q_hist) == 4)
        # assert that direct evaluation of the objective is the same as the
        # iteratively determined objective
        self.assertAlmostEqual(Q, objective.eval(g, layers))

    def test_pair_sum_critical_gap(self):
        """Test the pair sum version of the critical gap algorithm."""
        g = nx.Graph()
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        g.add_edge(0, 3)
        g.add_edge(1, 2)
        scores = {0: 5.0, 1: 4.0, 2: 3.0, 3: 2.0}
        nx.set_node_attributes(g, scores, 'score')
        g = utilities.preprocess(g)
        objective = Partitioning(penalty=1/3)
        layers, Q, Q_hist = algorithms.pair_sum_critical_gap(g, objective, True)
        # we can guarantee that the solution will be optimal at best, but
        # probably suboptimal
        self.assertTrue(Q <= 3 - 10/3 + 1e-14)
        # we can also test the number of merges, which should be equal to the
        # number of nodes
        self.assertTrue(len(Q_hist) == 4)
        # assert that direct evaluation of the objective is the same as the
        # iteratively determined objective
        self.assertAlmostEqual(Q, objective.eval(g, layers))


if __name__ == '__main__':
    unittest.main()