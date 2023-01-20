# -*- coding: utf-8 -*-
"""
Class for storing and handling partitions in layers.

Authors:
Alice Patania <alice.patania@uvm.edu>
Jean-Gabriel Young <jean-gabriel.young@uvm.edu>
"""
from collections import deque

class Layers(object):
    """Class representing partition of the nodes in layers."""

    # =========================================================================
    # Constructors
    # =========================================================================
    def __init__(self, cutpoints):
        """Initialize layers with cutpoints."""
        self._n = cutpoints[-1]
        self._max_idx = len(cutpoints)
        self._free_idx = set()
        self.layer_of = dict()
        self.layers = dict()
        u = -1
        for layer_idx, v in enumerate(cutpoints):
            self.layers[layer_idx] = deque(range(u + 1, v + 1))
            for j in range(u + 1, v + 1):
                self.layer_of[j] = layer_idx
            u = v

    @classmethod
    def from_cutpoints(cls, cutpoints):
        """Initialize layers with cutpoints.
        
        A  list of cutpoints contains the identifier of nodes at the top of
        every layer, in ascending order of identifier (i.e., descending order.
        of ranks).

        Parameters
        ----------
        cutpoints : list of int
        """
        return cls(cutpoints)

    @classmethod
    def from_dict(cls, layers_dict):
        """Initialize layers with a dictionary giving the ID of each layer.
        
        Assumes that the dictionary is well-formatted, in the sense that only
        contiguous nodes are placed in the same layer.

        Parameters
        ----------
        layers_dict: dict
            Keys are node identifiers (contiguous integers) and values are
            layer indexes.
        """
        tmp = sorted([(node, layer) for node, layer in layers_dict.items()],
                     key= lambda x: x[0])
        cutpoints = []
        current_layer = tmp[0][1]
        for node_idx, layer_idx in tmp:
            if layer_idx != current_layer:
                current_layer = layer_idx
                cutpoints.append(node_idx - 1)
        cutpoints.append(node_idx)
        return cls(cutpoints)

    @classmethod
    def from_sets(cls, layers_sets):
        """Initialize layers with a lsit of sets.

        Assumes that the list is well-formatted, in the sense that only layers
        appear in descending order with contiguous nodes placed in the same
        layer.

        Parameters
        ----------
        layers: list of sets
            List of layers encoded as sets.
        """
        cutpoints = []
        for l in layers_sets:
            cutpoints.append(max(l))
        return cls(cutpoints)

    # =========================================================================
    # Reserved methods
    # =========================================================================
    def __repr__(self):
        """Return repr(self)."""
        rep = '[' + "; ".join([str(list(v)) for l, v in self.layers.items()]) + ']'
        return rep

    def __eq__(self, other):
        """Return self==value."""
        if not isinstance(other, Layers):
            return NotImplemented
        for v in range(self._n):
            if self.layers[self.layer_of[v]] != other.layers[other.layer_of[v]]:
                return False
        return True

    def __iter__(self):
        return iter(self.as_sets())

    def __len__(self):
        return len(self.as_sets())

    # =========================================================================
    # Public methods
    # =========================================================================
    def as_sets(self):
        """Return layers as sets, in increasing order of identifiers."""
        # sort layers by node indexes
        tmp = sorted([(k, v) for (k, v) in self.layers.items()],
                     key=lambda x: x[1][0])
        # return as sets
        return [frozenset(v) for (k, v) in tmp]

    def move_up(self, v):
        """Move node v to the layer above."""
        # test that the move is possible: 
        # 1. `v` must be at the top of its layer
        # 2. v!=0
        l_v   = self.layer_of[v]
        if self.layers[l_v][0] == v and v != 0:
            l_vm1 = self.layer_of[v - 1]
            # pop out of layer and append to above layer
            self.layers[l_vm1].append(self.layers[l_v].popleft())
            # update layers
            self.layer_of[v] = l_vm1
            # check for empty layers:
            if len(self.layers[l_v]) == 0:
                del self.layers[l_v]
                self._free_idx.add(l_v)

    def move_dn(self, v):
        """Move node v to the layer below."""
        # test that the move is possible: 
        # 1. `v` must be at the bottom of its layer
        # 2. v!= n - 1
        l_v   = self.layer_of[v]
        if self.layers[l_v][-1] == v and v != self._n:
            l_vp1 = self.layer_of[v + 1]
            # pop out of layer and append to above layer
            self.layers[l_vp1].appendleft(self.layers[l_v].pop())
            # update layers
            self.layer_of[v] = l_vp1
            # check for empty layers:
            if len(self.layers[l_v]) == 0:
                del self.layers[l_v]
                self._free_idx.add(l_v)

    def split_above(self, v):
        """Split the a layer right above node v."""
        l_v   = self.layer_of[v]
        if self.layers[l_v][0] == v:
            # nothing to do
            pass
        else:
            target = self._get_free_idx()
            self.layers[target] = deque()
            if (v - self.layers[l_v][0]) < (self.layers[l_v][-1] - v):
                # fastest split is from the top
                u = None
                while u != v - 1:
                    u = self.layers[l_v].popleft()
                    self.layers[target].append(u)
                    self.layer_of[u] = target
            else:
                # fastest split is from the bottom
                u = None
                while u != v:
                    u = self.layers[l_v].pop()
                    self.layers[target].appendleft(u)
                    self.layer_of[u] = target

    def split_below(self, v):
        """Split the a layer right below node v."""
        l_v   = self.layer_of[v]
        if self.layers[l_v][-1] == v:
            # nothing to do
            pass
        else:
            target = self._get_free_idx()
            self.layers[target] = deque()
            if (v - self.layers[l_v][0]) < (self.layers[l_v][-1] - v):
                # fastest split is from the top
                u = None
                while u != v:
                    u = self.layers[l_v].popleft()
                    self.layers[target].append(u)
                    self.layer_of[u] = target
            else:
                # fastest split is from the bottom
                u = None
                while u != v + 1:
                    u = self.layers[l_v].pop()
                    self.layers[target].appendleft(u)
                    self.layer_of[u] = target

    def merge_range(self, u, v, cut=False):
        """Merge all nodes in the specified range, inclusively.

        Parameters
        ----------
        u, v: int
            Two end points of the new layer (inclusive).
        cut: bool
            If true, cut the specified range at its end to form a separated
            layers. If false, merge layers in which the endpoints are already
            included.
        """
        # index layer above and below
        if v < u:
            top = v
            bot = u
        else:
            top = u
            bot = v
        if top > 0:
            above_bot = top - 1
            above_top = min(self.layers[self.layer_of[above_bot]])
        if bot < self._n - 1:
            below_top = bot + 1
            below_bot = max(self.layers[self.layer_of[below_top]])
        # check if cuts are needed:
        if cut:
            self.split_above(top)
            self.split_below(bot)
        # merge
        start_point = min(self.layers[self.layer_of[top]]) + 1
        end_point = max(self.layers[self.layer_of[bot]])
        for v in range(start_point, end_point + 1):
            self.move_up(v)

    # =========================================================================
    # Private methods
    # =========================================================================
    def _get_free_idx(self):
        """Index manager for adding new layers."""
        if len(self._free_idx) > 0:
            return self._free_idx.pop()
        else:
            self._max_idx += 1
            return self._max_idx
