# -*- coding: utf-8 -*-
"""
Partitioning in one dimension.

Authors:
Alice Patania <alice.patania@uvm.edu>
Jean-Gabriel Young <jean-gabriel.young@uvm.edu>
"""
import networkx as nx
import numpy as np
import itertools as it

class PairSumObjectives(object):
    """The abstract class of quality functions calculated as sum over pairs of nodes."""
    def __init__(self):
        self.is_pair_sum = True

    def eval_layer(self, g, upper, lower):
        """Evaluate the quality of a layer.

        Parameters
        ----------
        g : networkx DiGraph, Graph, MulDiGraph or MultiGraph
            Input graph.
        upper, lower: int
            Identifiers of the nodes with the highest and lowest rank in the
            community.
        """
        f = 0
        for u, v in it.combinations_with_replacement(range(upper, lower + 1), 2):
            f += self.eval_pair(g, u, v)
        return f

    def eval(self, g, layers):
        """Evaluate the quality of a partition in layers.

        Parameters
        ----------
        g : networkx DiGraph, Graph, MulDiGraph or MultiGraph
            Input graph.
        layers: 
            Either a collection of sets, in which case the sets are interpreted as
            the nodes of a layer.
        """
        Q = 0
        for L in layers:
            Q += self.eval_layer(g, min(L), max(L))
        return Q


class ArbitraryPairSumObjectives(PairSumObjectives):
    """Arbitrary PairSumObjectives quality function.

    Quality of putting pairs of nodes together specified by a matrix

    Parameters
    ----------
    f : np.array
        Quality increment array. Assumed symmetric.
    """
    def __init__(self, f=None):
        super().__init__()
        self.f = f

    def eval_pair(self, g, u, v):
        """Calculate the contribution of a pair of nodes.

        Parameters
        ----------
        g : networkx DiGraph or MulDiGraph
            Input graph.
        u, v: int
            Node identifiers.
        """
        # overwrite the function as soon as epsilon is set
        return self.f[u, v]


class Partitioning(PairSumObjectives):
    """Partitioning objective.
    
    Increases with the number of internal edges in layers.

    Parameters
    ----------
    penalty : float
        Strength of the penalty term.
    """
    def __init__(self, penalty=1.0):
        super().__init__()
        self.penalty = penalty

    def eval_pair(self, g, u, v):
        """Calculate the contribution of a pair of nodes.

        Parameters
        ----------
        g : networkx DiGraph, Graph, MulDiGraph or MultiGraph
            Input graph.
        u, v: int
            Node identifiers.
        """
        return g.number_of_edges(u, v) - self.penalty * ((u != v) + 1)


class Modularity(PairSumObjectives):
    """Modularity objective.
    
    Sum of the difference of the realized and expected number of edges in layers.

    Parameters
    ----------
    null_model : string
        Null model to use in the calculation.
        Options are:
            * 'CM': Configuration model
            * 'ER': Erdős–Rényi
    g : networkx DiGraph, Graph, MulDiGraph or MultiGraph (optional)
        If passed at construction, will pre-compute quantities to speed up
        calculation.
    
    Notes
    -----
    The numerical value is not normalized by the number of edges.

    """
    def __init__(self, null_model='CM', g=None):
        super().__init__()
        if null_model == 'CM':
            if g is not None:
                self.k = {u: g.degree(u) for u in g.nodes()}
                self.m = g.number_of_edges()
                self.eval_pair = self._eval_pair_CM_precompute
            else:
                self.eval_pair = self._eval_pair_CM
        if null_model == 'ER':
            self.eval_pair = self._eval_pair_ER

    def _eval_pair_CM(self, g, u, v):
        """Calculate the contribution of a pair of nodes.

        Parameters
        ----------
        g : networkx DiGraph, Graph, MulDiGraph or MultiGraph
            Input graph.
        u, v: int
            Node identifiers.
        """
        k_u = g.degree(u)
        k_v = g.degree(v)
        m = g.number_of_edges()
        return (g.number_of_edges(u, v) - k_u * k_v / (2 * m)) / ((u == v) + 1)

    def _eval_pair_CM_precompute(self, g, u, v):
        """Calculate the contribution of a pair of nodes.

        Parameters
        ----------
        g : networkx DiGraph, Graph, MulDiGraph or MultiGraph
            Input graph.
        u, v: int
            Node identifiers.
        """
        return (g.number_of_edges(u, v) - self.k[u] * self.k[v] / (2 * self.m)) / ((u == v) + 1)

    def _eval_pair_ER(self, g, u, v):
        """Calculate the contribution of a pair of nodes.

        Parameters
        ----------
        g : networkx DiGraph, Graph, MulDiGraph or MultiGraph
            Input graph.
        u, v: int
            Node identifiers.
        """
        return (g.number_of_edges(u, v) - nx.density(g)) / ((u == v) + 1)


class Egalitarian(PairSumObjectives):
    """Egalitarian objective.

    Increases with the number of reciprocated edges within layers.

    Parameters
    ----------
    g : networkx DiGraph, Graph, MulDiGraph or MultiGraph
        Input graph used for precomputing correction terms.
    """
    def __init__(self, g=None):
        super().__init__()
        if g is None:
            self.epsilon = None
        else:
            self._set_epsilon(g)

    def _set_epsilon(self, g):
        mr = 0
        for u, v in it.combinations(g.nodes(), 2):
            if g.has_edge(u, v):
                if g.has_edge(v, u):
                    mr += 1
        n = g.number_of_nodes()
        self.epsilon = mr / (n * (n + 1) / 2)

    def eval_pair(self, g, u, v):
        """Calculate the contribution of a pair of nodes.

        Parameters
        ----------
        g : networkx DiGraph or MulDiGraph
            Input graph.
        u, v: int
            Node identifiers.
        """
        # overwrite the function as soon as epsilon is set
        if self.epsilon is None:
            self._set_epsilon(g)
        a_uv = g.number_of_edges(u, v)
        a_vu = g.number_of_edges(v, u)
        return a_uv * a_vu - self.epsilon


class Dominance(PairSumObjectives):
    """Dominance objective.

    Decreases with the number of unreciprocated edges within layers.

    Parameters
    ----------
    g : networkx DiGraph, Graph, MulDiGraph or MultiGraph
        Input graph used for precomputing correction terms.
    """
    def __init__(self, g=None):
        super().__init__()
        if g is None:
            self.epsilon = None
        else:
            self._set_epsilon(g)

    def _set_epsilon(self, g):
        mu = 0
        for u, v in it.combinations(g.nodes(), 2):
            if g.has_edge(u, v):
                if not g.has_edge(v, u):
                    mu += 1
            elif g.has_edge(v, u):
                mu += 1 
        n = g.number_of_nodes()
        self.epsilon = mu / (n * (n + 1) / 2)

    def eval_pair(self, g, u, v):
        """Calculate the contribution of a pair of nodes.

        Parameters
        ----------
        g : networkx DiGraph or MulDiGraph
            Input graph.
        u, v: int
            Node identifiers.
        """
        # overwrite the function as soon as epsilon is set
        if self.epsilon is None:
            self._set_epsilon(g)
        a_uv = g.number_of_edges(u, v)
        a_vu = g.number_of_edges(v, u)
        return 2 * a_uv * a_vu - (a_uv + a_vu) + self.epsilon
