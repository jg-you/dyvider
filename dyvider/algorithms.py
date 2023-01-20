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
import heapq
from copy import deepcopy
from collections import defaultdict
from dyvider.layers import Layers
# from layers import Layers


# =====================================
# Private functions for heap management 
# =====================================
_REMOVED = (-np.inf, -np.inf, -np.inf, -np.inf)

def _add_move(move_heap, move_finder, move, quality=0):
    if move in move_finder:
        remove_move(move)
    entry = [quality, move]
    move_finder[move] = entry
    heapq.heappush(move_heap, entry)

def _remove_move(move_heap, move_finder, move):
    entry = move_finder.pop(move)
    quality = entry[0]
    entry[-1] = _REMOVED
    return quality

def _pop_move(move_heap, move_finder):
    while move_heap:
        quality, move = heapq.heappop(move_heap)
        if move is not _REMOVED:
            del move_finder[move]
            return quality, move

# ====================================
# Dynamical programs
# ====================================
def run(g, objective):
    """High-level wrapper for the partitioning algorithms.

    Parameters
    ----------
    g : networkx DiGraph or Graph
        Annotated graph structure. Nodes should have a "score" attribute
        corresponding to their embedding position.
    objective : objective function object
        Will be used to asses the quality of a partition in layers.

    See Also
    --------
    naive_dp, pair_sum_dp
    """
    try:
        if objective.is_pair_sum:
            return pair_sum_dp(g, objective)
        else:
            return naive_dp(g, objective)
    except AttributeError as e:
        return naive_dp(g, objective)

def naive_dp(g, objective):
    """Partition using an additive objective with arbitrary layer quality functions.

    Parameters
    ----------
    g : networkx DiGraph or Graph
        Annotated graph structure. Nodes should have a "score" attribute
        corresponding to their embedding position.
    objective : objective function object
        Will be used to asses the quality of a partition in layers. The method
        `eval_layer(g, upper, lower) -> float` should be defined and compute
        the  quality of a layer comprising nodes `upper` through `lower`
        inclusively.

    Warnings
    --------
    Assumes that the graphs has been pre-processed, see utilities.py.

    See Also
    --------
    pair_sum_dp
    """
    n = g.number_of_nodes()
    Q = np.zeros(n + 1) # extra entry so that the base case is Q[-1] = 0
    optimal_cutpoint = np.zeros(n, dtype=int) # maximizing indices

    # unfolded recursion
    for j in range(0, n):
        Q[j]  = -np.inf
        for k in reversed(range(0, j + 1)):
            quality_of_split = Q[k - 1] + objective.eval_layer(g, k, j)
            if quality_of_split >  Q[j]:
                Q[j] = quality_of_split
                optimal_cutpoint[j] = k

    # retrieve cuts
    cuts = [optimal_cutpoint[-1]]
    while cuts[-1] != 0:
        cuts.append(optimal_cutpoint[cuts[-1] - 1])
    cuts = [v - 1 for v in (cuts[::-1] + [n])[1:]]

    return Layers(cuts), Q[-2]

def pair_sum_dp(g, objective):
    """Partition using an objective function with "pair-sum" quality functions.

    Parameters
    ----------
    g : networkx DiGraph or Graph
        Annotated graph structure. Nodes should have a "score" attribute
        corresponding to their embedding position.
    objective : objective function object
        Will be used to asses the quality of a partition in layers. The method
        `eval_pair(g, u, v) -> float` should be defined and compute the quality
        increment associated with putting nodes u and v in the same layer.

    Warnings
    --------
    Assumes that the graphs has been pre-processed, see utilities.py.

    See Also
    --------
    naive_dp
    """
    n = g.number_of_nodes()
    Q = np.zeros(n + 1) # extra entry so that the base case is Q[-1] = 0
    optimal_cutpoint = np.zeros(n, dtype=int) # maximizing indices

    # create storage for layer qualities
    # implemented as a dict with default value of 0 to handle
    # boundary conditions, e.g., calls like layer_quality[(1,0)]
    # which are nonsensical and should return 0
    layer_quality = defaultdict(int)

    # unfolded recursion
    for j in range(0, n):
        Q[j]  = -np.inf
        for k in reversed(range(0, j + 1)):
            layer_quality[(k, j)] =\
                layer_quality[(k, j - 1)] +\
                layer_quality[(k + 1, j)] -\
                layer_quality[(k + 1, j - 1)] +\
                objective.eval_pair(g, k, j)
            quality_of_split = Q[k - 1] + layer_quality[(k, j)]
            if quality_of_split >  Q[j]:
                Q[j] = quality_of_split
                optimal_cutpoint[j] = k

    # retrieve top node of each layer
    bounds = [n - 1, optimal_cutpoint[-1] - 1]
    while bounds[-1] != -1:
        bounds.append(optimal_cutpoint[bounds[-1]] - 1)
    bounds = list(reversed(bounds[:-1]))  # clip end condition and reverse

    return Layers(bounds), Q[-2]

# ====================================
# Heuristics / slow programs
# ====================================

def brute_force(g, objective, num_layers=None):
    """Brute force search for the best partition.

    Parameters
    ----------
    g : networkx DiGraph or Graph
        Annotated graph structure. Nodes should have a "score" attribute
        corresponding to their embedding position.
    objective : objective function object
        Will be used to asses the quality of a partition in layers. The method
        `eval_layer(g, upper, lower) -> float` should be defined and compute
        the  quality of a layer comprising nodes `upper` through `lower`
        inclusively.
    num_layers : int or None
        Desired number of layers. If None, this quantity is determined
        automatically by searching over all possible number of layers.

    Warnings
    --------
    * Assumes that the graphs has been pre-processed, see utilities.py.
    * This algorithm is not designed to be efficient; it should only used as
      a benchmark.
    """
    n = g.number_of_nodes()
    # create list of all the numbers of layers that will be tried
    if num_layers is None:
        num_layers = list(range(2, n + 1))
        optimal_Q = objective.eval_layer(g, 0, n - 1)
        optimal_layers = [n - 1]
    else:
        num_layers = [num_layers]
        optimal_Q = -np.inf
        optimal_layers = None

    # search
    for k in num_layers:
        # loop over the number of layers
        # in each iteration of the loop, get boundaries of layers as
        # combinations of the node indexes
        for boundaries in it.combinations(range(n - 1), k - 1):
            Q = 0
            upper = -1
            for lower in list(boundaries) + [n - 1]:
                Q += objective.eval_layer(g, upper + 1, lower)
                upper = lower
            if Q > optimal_Q:
                optimal_Q = Q
                optimal_layers = list(boundaries) + [n - 1]
    return Layers(optimal_layers), optimal_Q


def greedy(g, objective):
    """Repeatedly apply the best local improvement starting from isolated layers.

    Parameters
    ----------
    g : networkx DiGraph or Graph
        Annotated graph structure. Nodes should have a "score" attribute
        corresponding to their embedding position.
    objective : objective function object
        Should have the methods:
        * eval_layer(g, upper, lower) -> float
            This functions should compute the quality of a layer comprising
            nodes `upper` through `lower` inclusively.

    Warnings
    --------
    Assumes that the graphs has been pre-processed, see utilities.py.

    """
    # initialize each node in its own communities
    n = g.number_of_nodes()
    solution = Layers.from_dict({v: v for v in range(n)})
    iteration_order = list(range(n))

    made_move = True
    while made_move:
        np.random.shuffle(iteration_order)
        made_move = False
        # iterate on nodes
        for v in iteration_order:
            # check where the node sits (top, bottom, middle of layer?)
            L = solution.layers[solution.layer_of[v]]
            top = min(L)
            bot = max(L)
            delta_Q_up = 0
            delta_Q_down = 0
            if v == top and v != 0:
                # compute the quality of moving the top node to the layer above
                above_bot = top - 1
                above_top = min(solution.layers[solution.layer_of[above_bot]])
                delta_Q_up =\
                    objective.eval_layer(g, above_top, v) -\
                    objective.eval_layer(g, above_top, above_bot) +\
                    objective.eval_layer(g, v + 1, bot) -\
                    objective.eval_layer(g, top, bot)
            if v == bot and v != n - 1:
                # compute the quality of moving the botton node to the layer below
                below_top = bot + 1
                below_bot = max(solution.layers[solution.layer_of[below_top]])
                delta_Q_down =\
                    objective.eval_layer(g, top, v - 1) -\
                    objective.eval_layer(g, top, bot) +\
                    objective.eval_layer(g, v, below_bot) -\
                    objective.eval_layer(g, below_top, below_bot)

            #  check which move is best
            merge_target = np.argmax([0, delta_Q_up, delta_Q_down])
            if merge_target == 0:
                # do nothing
                continue
            if merge_target == 1:
                # better to move the node to the layer above
                made_move = True
                solution.move_up(v)
            if merge_target == 2:
                # better to move the node to the layer below
                made_move = True
                solution.move_dn(v)

    return solution, objective.eval(g, solution)


def heap_merge(g, objective, return_trace=False):
    """Greedy algorithm inspired by Clauset, Newman and Moore (2004).

    Start each node in its own layer, and iteratively merge neighboring layers
    that would most increase (or least decrease) the objective function.
    Return the best partition encoutered during the process.

    Parameters
    ----------
    g : networkx DiGraph or Graph
        Annotated graph structure. Nodes should have a "score" attribute
        corresponding to their embedding position.
    objective : objective function object
        Should have the methods:
        * eval_layer(g, upper, lower) -> float
            This functions should compute the quality of a layer comprising
            nodes `upper` through `lower` inclusively.
    return_history : bool
        If true, also return the evolution of the objective and of the solution.

    Warnings
    --------
    Assumes that the graphs has been pre-processed, see utilities.py.

    See Also
    --------
    pairsum_heap_merge
    """
    # initialize each node in its own layer
    n = g.number_of_nodes()
    tmp_layers = Layers.from_dict({v: v for v in range(n)})
    # initialize tracking of the best solution
    solution = deepcopy(tmp_layers)
    Q_history = [objective.eval(g, solution)]
    Q_max = Q_history[-1]
    
    move_heap = []
    move_finder = {}

    # initialize heap
    for v in range(1, n):
        above_bot = v - 1
        above_top = min(tmp_layers.layers[tmp_layers.layer_of[above_bot]])
        delta_Q =\
            objective.eval_layer(g, v - 1, v) -\
            objective.eval_layer(g, v - 1, v - 1) -\
            objective.eval_layer(g, v, v)
        _add_move(move_heap, move_finder, (v - 1, v - 1, v, v), -delta_Q)

    # run until we have a single layer
    while len(tmp_layers) > 1:
        # apply move
        (merge_delta_Q, (target_top, target_bot, top, bot)) = _pop_move(move_heap, move_finder)
        tmp_layers.merge_range(target_top, bot)

        # update solutions
        Q_history.append(Q_history[-1] - merge_delta_Q)
        if Q_history[-1] > Q_max:
            Q_max = Q_history[-1]
            solution = deepcopy(tmp_layers)

        # update merge with the layer below
        if bot < n - 1:
            below_top = bot + 1
            below_bot = max(tmp_layers.layers[tmp_layers.layer_of[below_top]])
            delta_Q =\
                objective.eval_layer(g, target_top, below_bot) -\
                objective.eval_layer(g, below_top, below_bot) -\
                objective.eval_layer(g, target_top, bot)
            _remove_move(move_heap, move_finder, (top, bot, below_top, below_bot))
            _add_move(move_heap, move_finder, (target_top, bot, below_top, below_bot), -delta_Q)
        # update merge with the layer above
        if target_top != 0:
            above_bot = target_top - 1
            above_top = min(tmp_layers.layers[tmp_layers.layer_of[above_bot]])
            delta_Q =\
                objective.eval_layer(g, above_top, bot) -\
                objective.eval_layer(g, above_top, above_bot) -\
                objective.eval_layer(g, target_top, bot)
            _remove_move(move_heap, move_finder, (above_top, above_bot, target_top, target_bot))
            _add_move(move_heap, move_finder, (above_top, above_bot, target_top, bot), -delta_Q)

    #  return
    if return_trace:
        return solution, Q_max, Q_history
    else:
        return solution, Q_max


def pair_sum_heap_merge(g, objective, return_trace=False):
    """Greedy algorithm inspired by Clauset, Newman and Moore (2004).

    Start each node in its own layer, and iteratively merge neighboring layers
    that would most increase (or least decrease) the objective function.
    Return the best partition encoutered during the process. This version of
    the function implements a speed-up specific to pair-sum objectives.

    Parameters
    ----------
    g : networkx DiGraph or Graph
        Annotated graph structure. Nodes should have a "score" attribute
        corresponding to their embedding position.
    objective : objective function object
        Should have the methods:
        * eval_layer(g, upper, lower) -> float
            This functions should compute the quality of a layer comprising
            nodes `upper` through `lower` inclusively.
    return_history : bool
        If true, also return the evolution of the objective and of the solution.

    Warnings
    --------
    Assumes that the graphs has been pre-processed, see utilities.py.

    See Also
    --------
    heap_merge
    """
    # initialize each node in its own layer
    n = g.number_of_nodes()
    tmp_layers = Layers.from_dict({v: v for v in range(n)})
    # initialize tracking of the best solution
    solution = deepcopy(tmp_layers)
    Q_history = [objective.eval(g, solution)]
    Q_max = Q_history[-1]
    
    move_heap = []
    move_finder = {}

    # initialize heap and dictionary
    for v in range(1, n):
        above_bot = v - 1
        above_top = min(tmp_layers.layers[tmp_layers.layer_of[above_bot]])
        delta_Q =\
            objective.eval_layer(g, v - 1, v) -\
            objective.eval_layer(g, v - 1, v - 1) -\
            objective.eval_layer(g, v, v)
        _add_move(move_heap, move_finder, (v - 1, v - 1, v, v), -delta_Q)

    # run until we have a single layer
    while len(tmp_layers) > 1:
        # apply move
        (merge_delta_Q, (target_top, target_bot, top, bot)) = _pop_move(move_heap, move_finder)
        tmp_layers.merge_range(target_top, bot)

        # update solutions
        Q_history.append(Q_history[-1] - merge_delta_Q)
        if Q_history[-1] > Q_max:
            Q_max = Q_history[-1]
            solution = deepcopy(tmp_layers)

        # update merge with the layer below
        if bot < n - 1:
            below_top = bot + 1
            below_bot = max(tmp_layers.layers[tmp_layers.layer_of[below_top]])
            delta_Q = -_remove_move(move_heap, move_finder, (top, bot, below_top, below_bot))
            for u, v in it.product(range(below_top, below_bot + 1), range(target_top, target_bot + 1)):
                delta_Q += objective.eval_pair(g, u, v)
            _add_move(move_heap, move_finder, (target_top, bot, below_top, below_bot), -delta_Q)
        # update merge with the layer above
        if target_top != 0:
            above_bot = target_top - 1
            above_top = min(tmp_layers.layers[tmp_layers.layer_of[above_bot]])
            delta_Q = -_remove_move(move_heap, move_finder, (above_top, above_bot, target_top, target_bot))
            for u, v in it.product(range(top, bot + 1), range(above_top, above_bot + 1)):
                delta_Q += objective.eval_pair(g, u, v)
            _add_move(move_heap, move_finder, (above_top, above_bot, target_top, bot), -delta_Q)

    #  return
    if return_trace:
        return solution, Q_max, Q_history
    else:
        return solution, Q_max


def critical_gap(g, objective, return_trace=False):
    """Clusters nodes greedily based on proximity in the embedding space.

    Parameters
    ----------
    g : networkx DiGraph or Graph
        Annotated graph structure. Nodes should have a "score" attribute
        corresponding to their embedding position.
    objective : objective function object
        Should have the methods:
        * eval_layer(g, upper, lower) -> float
            This functions should compute the quality of a layer comprising
            nodes `upper` through `lower` inclusively.
    return_history : bool
        If true, also return the evolution of the objective and of the solution.

    Notes
    -----
    Proceeds by merging the closest pairs of nodes into layers instead of by
    literally changing the gap. This is equivalent but much faster to the
    gap-variation method.

    Warnings
    --------
    Assumes that the graphs has been pre-processed, see utilities.py.

    See Also
    --------
    pair_sum_critical_gap
    """
    # initialize each node in its own layer
    n = g.number_of_nodes()
    tmp_layers = Layers.from_dict({v: v for v in range(n)})
    # initialize tracking of the best solution
    solution = deepcopy(tmp_layers)
    Q_history = [objective.eval(g, solution)]
    Q_max = Q_history[-1]

    move_heap = []
    move_finder = {}

    # initialize heap
    for v in range(1, n):
        above_bot = v - 1
        above_top = min(tmp_layers.layers[tmp_layers.layer_of[above_bot]])
        delta_Q =\
            objective.eval_layer(g, v - 1, v) -\
            objective.eval_layer(g, v - 1, v - 1) -\
            objective.eval_layer(g, v, v)
        distance = abs(g.nodes[v]['score'] - g.nodes[v-1]['score'])
        _add_move(move_heap, move_finder, (v - 1, v - 1, v, v), (distance, -delta_Q))

    # run until we have a single layer
    while len(tmp_layers) > 1:
        # apply move
        ((distance, merge_delta_Q), (target_top, target_bot, top, bot)) = _pop_move(move_heap, move_finder)
        tmp_layers.merge_range(target_top, bot)

        # update solutions
        Q_history.append(Q_history[-1] - merge_delta_Q)
        if Q_history[-1] > Q_max:
            Q_max = Q_history[-1]
            solution = deepcopy(tmp_layers)

        # update merge with the layer below
        if bot < n - 1:
            below_top = bot + 1
            below_bot = max(tmp_layers.layers[tmp_layers.layer_of[below_top]])
            delta_Q =\
                objective.eval_layer(g, target_top, below_bot) -\
                objective.eval_layer(g, below_top, below_bot) -\
                objective.eval_layer(g, target_top, bot)
            distance = abs(g.nodes[bot]['score'] - g.nodes[below_top]['score'])
            _remove_move(move_heap, move_finder, (top, bot, below_top, below_bot))
            _add_move(move_heap, move_finder, (target_top, bot, below_top, below_bot), (distance, -delta_Q))
        # update merge with the layer above
        if target_top != 0:
            above_bot = target_top - 1
            above_top = min(tmp_layers.layers[tmp_layers.layer_of[above_bot]])
            delta_Q =\
                objective.eval_layer(g, above_top, bot) -\
                objective.eval_layer(g, above_top, above_bot) -\
                objective.eval_layer(g, target_top, bot)
            distance = abs(g.nodes[above_bot]['score'] - g.nodes[target_top]['score'])
            _remove_move(move_heap, move_finder, (above_top, above_bot, target_top, target_bot))
            _add_move(move_heap, move_finder, (above_top, above_bot, target_top, bot), (distance, -delta_Q))

    #  return
    if return_trace:
        return solution, Q_max, Q_history
    else:
        return solution, Q_max


def pair_sum_critical_gap(g, objective, return_trace=False):
    """Clusters nodes greedily based on proximity in the embedding space.

    This version of the function implements a speed up specific to pair-sum
    objectives.

    Parameters
    ----------
    g : networkx DiGraph or Graph
        Annotated graph structure. Nodes should have a "score" attribute
        corresponding to their embedding position.
    objective : objective function object
        Should have the methods:
        * eval_layer(g, upper, lower) -> float
            This functions should compute the quality of a layer comprising
            nodes `upper` through `lower` inclusively.
    return_history : bool
        If true, also return the evolution of the objective and of the solution.

    Notes
    -----
    Proceeds by merging the closest pairs of nodes into layers instead of by
    literally changing the gap. This is equivalent but much faster to the
    gap-variation method.

    Warnings
    --------
    Assumes that the graphs has been pre-processed, see utilities.py.

    See Also
    --------
    critical_gap
    """
    # initialize each node in its own layer
    n = g.number_of_nodes()
    tmp_layers = Layers.from_dict({v: v for v in range(n)})
    # initialize tracking of the best solution
    solution = deepcopy(tmp_layers)
    Q_history = [objective.eval(g, solution)]
    Q_max = Q_history[-1]

    move_heap = []
    move_finder = {}

    # initialize heap
    for v in range(1, n):
        above_bot = v - 1
        above_top = min(tmp_layers.layers[tmp_layers.layer_of[above_bot]])
        delta_Q =\
            objective.eval_layer(g, v - 1, v) -\
            objective.eval_layer(g, v - 1, v - 1) -\
            objective.eval_layer(g, v, v)
        distance = abs(g.nodes[v]['score'] - g.nodes[v-1]['score'])
        _add_move(move_heap, move_finder, (v - 1, v - 1, v, v), (distance, -delta_Q))

    # run until we have a single layer
    while len(tmp_layers) > 1:
        # apply move
        ((distance, merge_delta_Q), (target_top, target_bot, top, bot)) = _pop_move(move_heap, move_finder)
        tmp_layers.merge_range(target_top, bot)

        # update solutions
        Q_history.append(Q_history[-1] - merge_delta_Q)
        if Q_history[-1] > Q_max:
            Q_max = Q_history[-1]
            solution = deepcopy(tmp_layers)

        # update merge with the layer below
        if bot < n - 1:
            below_top = bot + 1
            below_bot = max(tmp_layers.layers[tmp_layers.layer_of[below_top]])

            (_, priority) = _remove_move(move_heap, move_finder, (top, bot, below_top, below_bot))
            delta_Q = -priority
            for u, v in it.product(range(below_top, below_bot + 1), range(target_top, target_bot + 1)):
                delta_Q += objective.eval_pair(g, u, v)
            distance = abs(g.nodes[bot]['score'] - g.nodes[below_top]['score'])
            _add_move(move_heap, move_finder, (target_top, bot, below_top, below_bot), (distance, -delta_Q))
        # update merge with the layer above
        if target_top != 0:
            above_bot = target_top - 1
            above_top = min(tmp_layers.layers[tmp_layers.layer_of[above_bot]])
            (_, priority) = _remove_move(move_heap, move_finder, (above_top, above_bot, target_top, target_bot))
            delta_Q = -priority
            for u, v in it.product(range(top, bot + 1), range(above_top, above_bot + 1)):
                delta_Q += objective.eval_pair(g, u, v)
            distance = abs(g.nodes[above_bot]['score'] - g.nodes[target_top]['score'])
            _add_move(move_heap, move_finder, (above_top, above_bot, target_top, bot), (distance, -delta_Q))

    #  return
    if return_trace:
        return solution, Q_max, Q_history
    else:
        return solution, Q_max
