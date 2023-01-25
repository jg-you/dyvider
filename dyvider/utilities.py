# -*- coding: utf-8 -*-
"""
Utility functinos for the partitioner package.

Authors:
* Jean-Gabriel Young <jean-gabriel.young@uvm.edu>
* Alice Patania <alice.patania@uvm.edu>
"""
import networkx as nx

def preprocess(g):
    """Pre-process graph and scores.

    The nodes of the returned graph will be labeled with consecutive integers,
    in decreasing score order, and stored as a MultiGraph with collapsed nodes
    to account for equal scores.

    Parameters
    ----------
    g : networkx DiGraph, Graph, MulDiGraph or MultiGraph
        Annotated graph structure. Nodes should have a "score" attribute
        corresponding to their embedding position.
    """
    sorted_scores = sorted(nx.get_node_attributes(g, 'score').items(),
                           key=lambda x: x[1],
                           reverse=True)

    # separate nodes by equivalence classes with respect to scores
    has_equal_scores = False
    scores_to_supernode = dict()
    node_mapping = dict()
    curr_supernode = 0
    for (node_id, score) in sorted_scores:
        if score not in scores_to_supernode:
            scores_to_supernode[score] = curr_supernode
            node_mapping[curr_supernode] = [node_id]
            curr_supernode +=1
        else:
            node_mapping[scores_to_supernode[score]] += [node_id]
            has_equal_scores = True

    # create inverse mappings 
    supernode_to_scores = dict(map(reversed, scores_to_supernode.items()))
    inv_node_mapping = dict()
    for v in node_mapping:
        for u in node_mapping[v]:
            inv_node_mapping[u] = v

    # create processed graph
    if g.is_directed():
        g_prime = nx.MultiDiGraph()
    else:
        g_prime = nx.MultiGraph()
    # add nodes
    g_prime.add_nodes_from(range(len(node_mapping)))
    nx.set_node_attributes(g_prime, node_mapping, 'node_mapping')
    nx.set_node_attributes(g_prime, supernode_to_scores, 'score')
    # add edges
    for e in g.edges():
        g_prime.add_edge(inv_node_mapping[e[0]], inv_node_mapping[e[1]])

    # raise Warnings if needed:
    if has_equal_scores:
        raise RuntimeWarning("Nodes with identical scores have been collapsed "
                             "into  super-nodes. The default objective "
                             "functions don't account for super-nodes; use "
                             "custom objectives.")

    return g_prime

