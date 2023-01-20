#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Unit tests for the utility functions.

Authors:
* Jean-Gabriel Young <jean-gabriel.young@uvm.edu>
* Alice Patania <alice.patania@uvm.edu>
"""
import unittest
from dyvider import utilities
import networkx as nx

class TestUtilities(unittest.TestCase):
    """Test suite for all utility functions."""

    def test_preprocess_basic(self):
        """Test sorting and relabeling capabilities of the pre-processing function."""

        # create graph
        edges = [('1', '2'),
                 ('2', '3'),
                 ('1', '3'),
                 ('5', '6'),
                 ('5', '7'),
                 ('5', '8'),
                 ('6', '7'),
                 ('6', '8'),
                 ('7', '8'),
                 ('6', '2')]
        scores = {'1': -1.0,
                  '2': -3,
                  '3':  0.5,
                  '5': 2,
                  '6': 5.1,
                  '7': 4,
                  '8': 3.0}
        g = nx.Graph()
        g.add_edges_from(edges)
        nx.set_node_attributes(g, scores, 'score')

        # preprocess
        g = utilities.preprocess(g)

        # test sorting
        sorted_scores = [5.1, 4, 3.0, 2, 0.5, -1.0, -3]
        for v in range(7):
            self.assertAlmostEqual(sorted_scores[v], g.nodes[v]['score'])
 
        # test structure
        self.assertEqual(g.degree(0), 4)  # node with highest score has degree 4

    def test_preprocess_equalities(self):
        """Test that the pre-processing function correctly handles equalities."""

        # create graog
        edges = [(0, 1),
                 (0, 2),
                 (1, 2),
                 (1, 3),
                 (2, 3),
                 (0, 3)]
        scores = {0: 3.0,
                  1: 2.1,
                  2: 2.1,
                  3: 0}
        g = nx.Graph()
        g.add_edges_from(edges)
        nx.set_node_attributes(g, scores, 'score')

        # preprocess
        g = utilities.preprocess(g)

        # test equalities
        self.assertListEqual(g.nodes[1]['node_mapping'], [1, 2])
        # test structure
        self.assertListEqual([g.degree(0), g.degree(1), g.degree(2)],
                             [3, 6, 3])

if __name__ == '__main__':
    unittest.main()