#!/usr/bin/python

class Graph:
    def __init__(self, nodes=None, edges=None):
        """Initialize a graph object.
        Args:
            nodes:  Iterator of nodes. Each node is an object.
            edges:  Iterator of edges. Each edge is a tuple of 2 nodes.
        """
        self.nodes, self.adj = [], {}
        if nodes != None:
            self.add_nodes_from(nodes)
        if edges != None:
            self.add_edges_from(edges)

    def __len__(self):
        """Returns the number of nodes in the graph.

        >>> g = Graph(nodes=[x for x in range(7)])
        >>> len(g)
        7
        """
        return len(self.nodes)

    def __contains__(self, x):
        """Return true if a node x is in the graph.

        >>> g = Graph(nodes=[x for x in range(7)])
        >>> 6 in g
        True
        >>> 7 in g
        False
        """
        return x in self.nodes

    def __iter__(self):
        """Iterate over the nodes in the graph.

        >>> g = Graph(nodes=[x for x in range(7)])
        >>> [x * 2 for x in g]
        [0, 2, 4, 6, 8, 10, 12]
        """
        return iter(self.nodes)

    def __getitem__(self, x):
        """Returns the iterator over the adjacent nodes of x.

        >>> g = Graph(nodes=[x for x in range(7)], edges=[(1,0), (1,2), (1,3)])
        >>> [x for x in g[1]]
        [0, 2, 3]
        """
        return iter(self.adj[x])

    def __str__(self):
        return 'V: %s\nE: %s' % (self.nodes, self.adj)

    def add_node(self, n):
        if n not in self.nodes:
            self.nodes.append(n)
            self.adj[n] = []

    def add_nodes_from(self, i):
        for n in i:
            self.add_node(n)

    def add_edge(self, u, v):   # undirected unweighted graph
        self.adj[u] = self.adj.get(u, []) + [v]
        self.adj[v] = self.adj.get(v, []) + [u]

    def add_edges_from(self, i):
        for n in i:
            self.add_edge(*n)

    def number_of_nodes(self):
        return len(self.nodes)

    def number_of_edges(self):
        return sum(len(l) for _, l in self.adj.items()) // 2

#     add_nodes(x)
#     add_nodes_from(iterator)
#     add_edge(u,v,object=x)
#     add_edges_from(iterator, each edge as a tuple)
#     remove_node(x)
#     remove_edge(u,v)
#     clear()
#     number_of_nodes()
#     number_of_edges()
#     nodes()
#     edges()
#     neighbors()
#     edges_iter()

# node x in G __contains__
# len(G) = number of nodes
# for node x in G ... __iter__
# G[x] __getitem__


class DGraph(Graph):
    def add_edge(self, u, v):
        self.adj[u] = self.adj.get(u, []) + [v]


class WGraph(Graph):
    def __init__(self, nodes=None, edges=None):
        """Initialize a graph object.
        Args:
            nodes:  Iterator of nodes. Each node is an object.
            edges:  Iterator of edges. Each edge is a tuple of 2 nodes and a weight.
        """
        self.nodes, self.adj, self.weight = [], {}, {}
        if nodes != None:
            self.add_nodes_from(nodes)
        if edges != None:
            self.add_edges_from(edges)

    def add_edge(self, u, v, w):
        self.adj[u] = self.adj.get(u, []) + [v]
        self.adj[v] = self.adj.get(v, []) + [u]
        self.weight[(u,v)] = w
        self.weight[(v,u)] = w

    def get_weight(self, u, v):
        return self.weight[(u,v)]

class DWGraph(WGraph):
    def add_edge(self, u, v, w):
        self.adj[u] = self.adj.get(u, []) + [v]
        self.weight[(u,v)] = w

if __name__=='__main__':
    import doctest
    doctest.testmod()