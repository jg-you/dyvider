#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Unit tests for the Layers class.

Authors: 
* Jean-Gabriel Young <jean-gabriel.young@uvm.edu>
* Alice Patania <alice.patania@uvm.edu>
"""
import unittest
import networkx as nx
from copy import copy
from dyvider.layers import Layers

class TestLayers(unittest.TestCase):
    """Test suite for the Layers class."""

    def test_init(self):
        """Test all the init methods."""
        # test explicit class method
        l1 = Layers([3, 5, 6, 9])
        l2 = Layers.from_cutpoints([3, 5, 6, 9])
        self.assertEqual(l1, l2)

        # test dict constructor
        l3 = Layers.from_dict({0: 0, 1: 0, 2: 0, 3: 0,
                               4: 1, 5: 1,
                               6: 2,
                               7: 3, 8: 3, 9: 3})
        self.assertEqual(l1, l3)

        # test sets constructor
        l4 = Layers.from_sets([{0, 1, 2, 3}, {4, 5}, {6}, {7, 8, 9}])
        self.assertEqual(l1, l4)

        # test that method doesn't depend on choice of layer idx
        l5 = Layers.from_dict({0: 0, 1: 0, 2: 0, 3: 0,
                               4: 1, 5: 1,
                               6: 3,
                               7: 2, 8: 2, 9: 2})
        self.assertEqual(l1, l5)

    def test_iteration(self):
        """Test that iteration over Layers work."""
        layer_data = [frozenset({0, 1, 2, 3}),
                      frozenset({4, 5}),
                      frozenset({6}),
                      frozenset({7, 8, 9})]
        layers = Layers.from_sets(layer_data)
        for l in layers:
            self.assertTrue(l in layers)

    def test_length(self):
        """Test that Layers have the proper `len` method."""
        layer_data = [frozenset({0, 1, 2, 3}),
                      frozenset({4, 5}),
                      frozenset({6}),
                      frozenset({7, 8, 9})]
        layers = Layers.from_sets(layer_data)
        self.assertEqual(4, len(layers))

    def test_moves(self):
        """Test methods moving the nodes."""
        # move down
        l1 = Layers([3, 5])
        l2 = Layers([2, 5])
        l1.move_dn(3)
        self.assertEqual(l1, l2)

        # move up
        l1 = Layers([3, 5])
        l2 = Layers([2, 5])
        l2.move_up(3)
        self.assertEqual(l1, l2)

        # there and back again
        l1 = Layers([3, 5])
        l2 = copy(l1)
        l1.move_dn(3)
        l1.move_up(3)
        self.assertEqual(l1, l2)

    def test_move_edgecases(self):
        """Test edge cases for the move methods."""
        # moving the first node up shouldn't do anything
        l1 = Layers([3, 5])
        l2 = copy(l1)
        l1.move_up(0)
        self.assertEqual(l1, l2)

        # moving the last node down shouldn't do anything
        l1 = Layers([3, 5])
        l2 = copy(l1)
        l1.move_dn(5)
        self.assertEqual(l1, l2)

        # moving a node in the middle of a layer shouldn't do anything
        l1 = Layers([5, 10])
        l2 = copy(l1)
        l1.move_up(7)
        self.assertEqual(l1, l2)

        l1 = Layers([5, 10])
        l2 = copy(l1)
        l1.move_dn(7)
        self.assertEqual(l1, l2)

    def test_destructive_moves(self):
        """Test moves that will remove layers."""
        l1 = Layers([1, 2, 5])
        l2 = Layers([2, 5])
        l1.move_up(2)
        self.assertEqual(l1, l2)

        l1 = Layers([1, 2, 5])
        l2 = Layers([2, 5])
        l1.move_dn(1)
        l1.move_dn(0)
        self.assertEqual(l1, l2)

        l1 = Layers([1, 2, 3, 5])
        l2 = Layers([1, 3, 5])
        l1.move_dn(2)
        self.assertEqual(l1, l2)

    def test_split(self):
        """Test layer splitting methods."""
        l1 = Layers([2, 5])
        l2 = Layers([2, 3, 5])
        l1.split_above(4)
        self.assertEqual(l1, l2)

        l1 = Layers([2, 5])
        l2 = Layers([2, 3, 5])
        l1.split_below(3)
        self.assertEqual(l1, l2)

    def test_split_direction(self):
        """Test that the splitting method work in both 'directions'."""
        # Note: Some implementations of Layers could get a decent speedup by
        # splitting a layer from the top or bottom, depending on which is
        # closest to the edge.
        l1 = Layers([2, 10, 100])
        l2 = Layers([2, 90, 100])
        
        l3 = Layers([2, 100])
        l3.split_above(11)
        self.assertEqual(l1, l3)

        l3 = Layers([2, 100])
        l3.split_above(91)
        self.assertEqual(l2, l3)

        l3 = Layers([2, 100])
        l3.split_below(10)
        self.assertEqual(l1, l3)

        l3 = Layers([2, 100])
        l3.split_below(90)
        self.assertEqual(l2, l3)

    def test_split_edge(self):
        """Test that splitting methods do not do anything at the edge of layers."""
        l1 = Layers([2, 5])
        l2 = copy(l1)
        l1.split_above(0)
        self.assertEqual(l1, l2)

        l1 = Layers([2, 5])
        l2 = copy(l1)
        l1.split_below(5)
        self.assertEqual(l1, l2)

        l1 = Layers([2, 5])
        l2 = copy(l1)
        l1.split_below(2)
        self.assertEqual(l1, l2)

        l1 = Layers([2, 5])
        l2 = copy(l1)
        l1.split_above(3)
        self.assertEqual(l1, l2)

    def test_merge_range(self):
        """Test the merging method."""
        # test merging without cutting
        #   no border nodes
        l1 = Layers([2, 5, 6, 8, 9])
        l1.merge_range(1, 4)
        l2 = Layers([5, 6, 8, 9])
        self.assertEqual(l1, l2)
        #   top bordering node
        l1 = Layers([2, 5, 6, 8, 9])
        l1.merge_range(1, 5)
        l2 = Layers([5, 6, 8, 9])
        self.assertEqual(l1, l2)
        #   bottom bordering node
        l1 = Layers([2, 5, 6, 8, 9])
        l1.merge_range(3, 5)
        l2 = Layers([2, 5, 6, 8, 9])
        self.assertEqual(l1, l2)

        # test merging with cutting
        #   no border nodes
        l1 = Layers([2, 5, 6, 8, 9])
        l1.merge_range(1, 4, cut=True)
        l2 = Layers([0, 4, 5, 6, 8, 9])
        self.assertEqual(l1, l2)
        #   top bordering node
        l1 = Layers([2, 5, 6, 8, 9])
        l1.merge_range(1, 5, cut=True)
        l2 = Layers([0, 5, 6, 8, 9])
        self.assertEqual(l1, l2)
        #   bottom bordering node
        l1 = Layers([2, 5, 6, 8, 9])
        l1.merge_range(3, 5)
        l2 = Layers([2, 5, 6, 8, 9])
        self.assertEqual(l1, l2)


    def test_as_set(self):
        """Test the conversion method return the layers as sets."""
        l1 = Layers([3, 5, 6, 9])
        l1_sets = l1.as_sets()
        self.assertTrue(frozenset({0, 1, 2, 3}) in l1_sets)
        self.assertTrue(frozenset({4, 5}) in l1_sets)
        self.assertTrue(frozenset({6}) in l1_sets)
        self.assertTrue(frozenset({7, 8, 9}) in l1_sets)
        self.assertEqual(len(l1_sets), 4)

if __name__ == '__main__':
    unittest.main()