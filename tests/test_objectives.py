#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Unit tests for the objective functions code.

Authors: 
* Jean-Gabriel Young <jean-gabriel.young@uvm.edu>
* Alice Patania <alice.patania@uvm.edu>
"""
import unittest
import networkx as nx
from inspect import getmembers, isclass
from dyvider import objectives
from dyvider.layers import Layers

class TestObjectives(unittest.TestCase):
    """Test suite for all the objective functions."""

    def test_attribute_existence(self):
        """Check that all objective functions have a fast_update attribute."""
        for name, obj in getmembers(objectives, isclass):
            self.assertTrue(isinstance(obj().is_pair_sum, bool))

    def test_fast_update(self):
        """Check that the PairSumObjectives flag is truthful."""
        # create test graph
        g = nx.random_geometric_graph(100, 0.05, dim=1)
        for v in g.nodes():
            g.nodes[v]['score'] = g.nodes[v].pop('pos')[0]

        # for each PairSumObjectives, test the udpate equation
        for name, obj in getmembers(objectives, isclass):
            if name == 'PairSumObjectives':
                continue
            objective_function = obj()
            if objective_function.is_pair_sum:
                f_Lkj = objective_function.eval_layer(g, 10, 15)
                f_Lkjm1 = objective_function.eval_layer(g, 10, 14)
                f_Lkp1j = objective_function.eval_layer(g, 11, 15)
                f_Lkp1jm1 = objective_function.eval_layer(g, 11, 14)
                f_kj = objective_function.eval_pair(g, 10, 15)
                self.assertAlmostEqual(f_Lkj, f_kj + f_Lkjm1 + f_Lkp1j - f_Lkp1jm1)

    def test_evals(self):
        """Check that all eval functions are consistent with one another."""
        g = nx.Graph()
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        g.add_edge(0, 3)
        g.add_edge(1, 2)
        obj = objectives.Partitioning(penalty=0.42)
        
        # global evals is the same as layer evals
        Q1 = obj.eval(g, Layers.from_sets([{0, 1, 2}, {3}]))
        Q2 = obj.eval_layer(g, 0, 2) + obj.eval_layer(g, 3, 3)
        self.assertAlmostEqual(Q1, Q2)
        
        # layer eval is the same as sum of pair eval
        # (for a PairSum objective)
        f1 = obj.eval_layer(g, 0, 2) 
        f2 = obj.eval_pair(g, 0, 0) + obj.eval_pair(g, 1, 1) + obj.eval_pair(g, 2, 2) +\
          obj.eval_pair(g, 0, 1) + obj.eval_pair(g, 0, 2) + obj.eval_pair(g, 1, 2)
        self.assertAlmostEqual(f1, f2)

    def test_partitioning(self):
        """Test numerical value of the partitioning objective."""
        # create graph and instantiate objective function
        g = nx.Graph()
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        g.add_edge(0, 3)
        g.add_edge(1, 2)
        obj = objectives.Partitioning()
        # test individual terms
        self.assertEqual(obj.eval_pair(g, 0, 0), - 1)
        self.assertEqual(obj.eval_pair(g, 1, 2), 1 - 2)
        self.assertEqual(obj.eval_pair(g, 1, 3), 0 - 2)
        # test overall layer quality
        self.assertEqual(obj.eval_layer(g, 0, 1), 1 - 4)
        self.assertEqual(obj.eval_layer(g, 0, 2), 3 - 9)
        self.assertEqual(obj.eval_layer(g, 0, 3), 4 - 16)
        self.assertEqual(obj.eval_layer(g, 1, 3), 1 - 9)
        self.assertEqual(obj.eval_layer(g, 2, 3), 0 - 4)
        self.assertEqual(obj.eval_layer(g, 1, 2), 1 - 4)

    def test_egalitarian(self):
        """Test numerical value of the Egalitarian objective."""
        # create graph and instantiate objective function
        g = nx.DiGraph()
        g.add_edge(0, 1)
        g.add_edge(1, 0)
        g.add_edge(2, 3)
        g.add_edge(3, 2)
        g.add_edge(1, 2)
        obj = objectives.Egalitarian(g)
        # test individual terms
        self.assertAlmostEqual(obj.eval_pair(g, 0, 0), 0 - 1/5)
        # test overall layer quality
        self.assertAlmostEqual(obj.eval_layer(g, 0, 1), 1 - 3/5)
        self.assertAlmostEqual(obj.eval_layer(g, 0, 2), 1 - 6/5)
        self.assertAlmostEqual(obj.eval_layer(g, 0, 3), 2 - 10/5)
        self.assertAlmostEqual(obj.eval_layer(g, 1, 3), 1 - 6/5)
        self.assertAlmostEqual(obj.eval_layer(g, 2, 3), 1 - 3/5)
        self.assertAlmostEqual(obj.eval_layer(g, 1, 2), 0 - 3/5)

    def test_dominance(self):
        """Test numerical value of the Egalitarian objective."""
        # create graph and instantiate objective function
        g = nx.DiGraph()
        g.add_edge(0, 1)
        g.add_edge(1, 0)
        g.add_edge(2, 3)
        g.add_edge(3, 2)
        g.add_edge(0, 2)
        g.add_edge(1, 3)
        obj = objectives.Dominance(g)
    
        # test individual terms
        self.assertAlmostEqual(obj.eval_pair(g, 0, 0), 0 + 1/5)
        # test overall layer quality
        self.assertAlmostEqual(obj.eval_layer(g, 0, 1), 0 + 3/5)
        self.assertAlmostEqual(obj.eval_layer(g, 0, 2), -1 + 6/5)
        self.assertAlmostEqual(obj.eval_layer(g, 0, 3), -2 + 10/5)
        self.assertAlmostEqual(obj.eval_layer(g, 1, 3), -1 + 6/5)
        self.assertAlmostEqual(obj.eval_layer(g, 2, 3), 0 + 3/5)
        self.assertAlmostEqual(obj.eval_layer(g, 1, 2), 0 + 3/5)

    def test_modularity_CM_null(self):
        """Test numerical value of the modularity objective with a CM null."""
        # create graph and instantiate objective function
        g = nx.Graph()
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        g.add_edge(0, 3)
        g.add_edge(1, 2)
        obj = objectives.Modularity()
        # test individual terms
        self.assertAlmostEqual(obj.eval_pair(g, 0, 0), - 9 / 16)
        self.assertAlmostEqual(obj.eval_pair(g, 1, 2), 1 - 1 / 2)
        self.assertAlmostEqual(obj.eval_pair(g, 1, 3), 0 - 1 / 4)
        # test overall layer quality
        self.assertAlmostEqual(obj.eval_layer(g, 0, 1), 1 - 25 / 16)
        self.assertAlmostEqual(obj.eval_layer(g, 0, 2), 3 - 49 / 16)
        self.assertAlmostEqual(obj.eval_layer(g, 0, 3), 0.0)
        self.assertAlmostEqual(obj.eval_layer(g, 1, 3), 1 - 25 / 16)
        self.assertAlmostEqual(obj.eval_layer(g, 2, 3), 0 - 9 / 16)
        self.assertAlmostEqual(obj.eval_layer(g, 1, 2), 0.0)

    def test_modularity_ER_null(self):
        """Test numerical value of the modularity objective with an ER null."""
        # create graph and instantiate objective function
        g = nx.Graph()
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        g.add_edge(0, 3)
        g.add_edge(1, 2)
        obj = objectives.Modularity(null_model='ER')
        # test individual terms
        rho = 4 / 6
        self.assertAlmostEqual(obj.eval_pair(g, 0, 0), -rho / 2)
        self.assertAlmostEqual(obj.eval_pair(g, 1, 2), 1 - rho)
        self.assertAlmostEqual(obj.eval_pair(g, 1, 3), 0 - rho)
        # test overall layer quality
        self.assertAlmostEqual(obj.eval_layer(g, 0, 1), 1 - 2 * rho)
        self.assertAlmostEqual(obj.eval_layer(g, 0, 2), 3 - 9 * rho / 2)
        self.assertAlmostEqual(obj.eval_layer(g, 0, 3), 4 - 8 * rho)
        self.assertAlmostEqual(obj.eval_layer(g, 1, 3), 1 - 9 * rho / 2)
        self.assertAlmostEqual(obj.eval_layer(g, 2, 3), 0 - 2 * rho)
        self.assertAlmostEqual(obj.eval_layer(g, 1, 2), 1 - 2 * rho)


if __name__ == '__main__':
    unittest.main()