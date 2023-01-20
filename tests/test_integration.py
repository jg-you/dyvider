#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Integration tests.

Authors: 
* Jean-Gabriel Young <jean-gabriel.young@uvm.edu>
* Alice Patania <alice.patania@uvm.edu>
"""
import unittest
from dyvider import utilities
from dyvider import algorithms
from dyvider import objectives
from dyvider.layers import Layers
import networkx as nx
import numpy as np

class IntegrationTests(unittest.TestCase):
    """Test suite of larger portions of the code."""

    def test_consistency_dp(self):
        """Test whether dynamical program returns optimal solutions"""
        g = nx.generators.barbell_graph(5, 5)
        scores = {v: np.random.rand() for v in g.nodes()}
        nx.set_node_attributes(g, scores, 'score')
        g = utilities.preprocess(g)

        # Brute force should find the same thing as DP
        objective = objectives.Partitioning(penalty=1/3)
        bf_output, bf_Q = algorithms.brute_force(g, objective)

        dp_naive_output, dp_naive_Q = algorithms.naive_dp(g, objective)
        dp_ps_output, dp_ps_Q = algorithms.pair_sum_dp(g, objective)

        # test on objective function's value since multiple optima may exist
        self.assertAlmostEqual(bf_Q, dp_naive_Q)
        self.assertAlmostEqual(bf_Q, dp_ps_Q)
        self.assertAlmostEqual(dp_naive_Q, dp_ps_Q)

        # we can test the partition themselves for the two dynamical program
        # since they should only differ in the way updates are calculated.
        self.assertEqual(dp_naive_output, dp_ps_output)

    def test_consistency_heap_merge(self):
        """Test whether all heap merge algorithms return the same solution."""
        g = nx.generators.barbell_graph(5, 5)
        scores = {v: np.random.rand() for v in g.nodes()}
        nx.set_node_attributes(g, scores, 'score')
        g = utilities.preprocess(g)

        # Brute force should find the same thing as DP
        objective = objectives.Partitioning(penalty=1/3)
        hm_output, hm_Q = algorithms.heap_merge(g, objective)
        hm_ps_output, hm_ps_Q = algorithms.pair_sum_heap_merge(g, objective)

        # we can test the partition themselves since the two merging algorithm
        # they should only differ in the way updates are calculated.
        self.assertEqual(hm_output, hm_ps_output)
        self.assertAlmostEqual(hm_ps_Q, hm_Q)

    def test_consistency_critical_gap(self):
        """Test whether all critical gap algorithms return the same solution."""
        g = nx.generators.barbell_graph(5, 5)
        scores = {v: np.random.rand() for v in g.nodes()}
        nx.set_node_attributes(g, scores, 'score')
        g = utilities.preprocess(g)

        # Brute force should find the same thing as DP
        objective = objectives.Partitioning(penalty=1/3)
        cgm_output, cgm_Q = algorithms.critical_gap(g, objective)
        cgm_ps_output, cgm_ps_Q = algorithms.pair_sum_critical_gap(g, objective)

        # we can test the partition themselves since the two merging algorithm
        # they should only differ in the way updates are calculated.
        self.assertEqual(cgm_output, cgm_ps_output)
        self.assertAlmostEqual(cgm_ps_Q, cgm_Q)

    def test_consistency_partitioning(self):
        """Test whether mathematical equivalencies of algorithms are respected."""
        g = nx.generators.barbell_graph(3, 3)
        scores = {v: np.random.rand() for v in g.nodes()}
        nx.set_node_attributes(g, scores, 'score')
        g = utilities.preprocess(g)

        # The partitioning objective with a particular weight should return
        # the same quality as the modularity with an ER null

        penalty = nx.density(g)
        p_obj = objectives.Partitioning(penalty=nx.density(g)/2)
        m_obj = objectives.Modularity(null_model='ER')

        n = g.number_of_nodes()
        partition = Layers(range(n))
        for v in range(n):
            partition.move_up(v)
            self.assertAlmostEqual(p_obj.eval(g, partition), m_obj.eval(g, partition))


if __name__ == '__main__':
    unittest.main()