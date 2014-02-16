#!/usr/bin/python
from pygraph import Graph, DGraph, WGraph
from collections import deque
import heapq
import itertools

# UnionFind for kruskal's MST algorihtm.
class UnionFind:
    """Union-find data structure.

    Each unionFind instance X maintains a family of disjoint sets of
    hashable objects, supporting the following two methods:

    - X[item] returns a name for the set containing the given item.
      Each set is named by an arbitrarily-chosen one of its members; as
      long as the set remains unchanged it will keep the same name. If
      the item is not yet part of a set in X, a new singleton set is
      created for it.

    - X.union(item1, item2, ...) merges the sets containing each item
      into a single larger set.  If any item is not yet part of a set
      in X, it is added to X as one of the members of the merged set.

    - len(X) returns the number of disjoint sets currently in the data structure.
    """

    def __init__(self):
        """Create a new empty union-find structure."""
        self.weights = {}
        self.parents = {}

    def __getitem__(self, object):
        """Find and return the name of the set containing the object."""

        # check for previously unknown object
        if object not in self.parents:
            self.parents[object] = object
            self.weights[object] = 1
            return object

        # find path of objects leading to the root
        path = [object]
        root = self.parents[object]
        while root != path[-1]:
            path.append(root)
            root = self.parents[root]

        # compress the path and return
        for ancestor in path:
            self.parents[ancestor] = root
        return root
        
    def __iter__(self):
        """Iterate through all items ever found or unioned by this structure."""
        return iter(self.parents)

    def union(self, *objects):
        """Find the sets containing the objects and merge them all."""
        roots = [self[x] for x in objects]
        heaviest = max([(self.weights[r],r) for r in roots])[1]
        for r in roots:
            if r != heaviest:
                self.weights[heaviest] += self.weights[r]
                self.parents[r] = heaviest
    
    def numberOfSets(self):
        S = set()
        for k, v in self.parents.items():
            if v not in S:
                S.add(v)
        return len(S)

    def __len__(self):
        return self.numberOfSets()

def dfs(g, src):
    """Initialize a depth first search from vertex src on graph g.

    Args:
        g:      Graph to perform depth first search on.
        src:    Source vertex.
    Returns:
        Dictionary of paths.
    Raises:
        KeyError:   When source vertex is not in the graph.

    >>> V = [x for x in range(8)]
    >>> E = [(0,1), (0,2), (0,5), (0,6), (5,3), (5,4), (3,4), (6,4), (3,7)]
    >>> g = Graph(nodes=V, edges=E)
    >>> print(g)
    V: [0, 1, 2, 3, 4, 5, 6, 7]
    E: {0: [1, 2, 5, 6], 1: [0], 2: [0], 3: [5, 4, 7], 4: [5, 3, 6], 5: [0, 3, 4], 6: [0, 4], 7: [3]}
    >>> print(dfs(g, 0)[7])
    [0, 5, 3, 7]
    """
    mark = {}
    prev = {}
    for node in g:
        mark[node] = False
        prev[node] = None

    __dfs(g, src, mark, prev)

    paths = {}
    for n in g:
        if not mark[n]:     # n cannot be reached
            paths[n] = None
            break
        path, dst = [], n   # find path from src to n
        while dst != src:
            path.append(dst)
            dst = prev[dst]
        path.append(src)
        paths[n] = list(reversed(path))
    return paths

def __dfs(g, node, mark, prev):
    """Perform a depth first search based on the given graph and source vertex.
    Called when object is initialized.

    Args:
        node:   Node to recursively visit.
    Raises:
        KeyError:   When node is not in the graph.
    """
    mark[node] = True
    for n in g[node]:
        if not mark[n]:
            prev[n] = node
            __dfs(g, n, mark, prev)

def bfs(g, src):
    """Initialize a breadth first search on graph g from node src to other nodes.

    Args:
        g:      Undirected graph. Weighted or unweighted.
        src:    Source vertex.
    Returns:
        A dictionary of paths from src vertex.
    Raises:
        KeyError:   When source vertex is not in the graph.

    >>> V = [x for x in range(8)]
    >>> E = [(0,1), (0,2), (0,5), (0,6), (5,3), (5,4), (3,4), (6,4), (3,7)]
    >>> g = Graph(nodes=V, edges=E)
    >>> print(g)
    V: [0, 1, 2, 3, 4, 5, 6, 7]
    E: {0: [1, 2, 5, 6], 1: [0], 2: [0], 3: [5, 4, 7], 4: [5, 3, 6], 5: [0, 3, 4], 6: [0, 4], 7: [3]}
    >>> paths = bfs(g, 0)
    >>> print(paths[6])
    [0, 6]
    >>> print(paths[7])
    [0, 5, 3, 7]
    """
    mark, prev, dist = {}, {}, {}
    for node in g:
        mark[node] = False
        prev[node] = None
        dist[node] = -1

    dq = deque([src])
    dist[src] = 0

    while len(dq) > 0:
        n = dq.pop()
        mark[n] = True
        for nbr_of_n in g[n]:
            if not mark[nbr_of_n]:
                dq.appendleft(nbr_of_n)
                prev[nbr_of_n] = n
                dist[nbr_of_n] = dist[n] + 1

    paths = {}
    for n in g:
        if not mark[n]:     # n cannot be reached
            paths[n] = None
            continue
        path, dst = [], n   # find path from src to n
        while dst != src:
            path.append(dst)
            dst = prev[dst]
        path.append(src)
        paths[n] = list(reversed(path))
    return paths

def topo(g):
    """Topological sort on graph g.

    Args:
        g:  Directed acyclic graph. Weighted or unweighted.
    Returns:
        List of nodes in topological order.

    >>> V = [x for x in range(8)]
    >>> E = [(0, 2), (1, 2), (2, 4), (3, 4), (3, 5), (4, 6), (5, 6), (6, 7)]
    >>> dg = DGraph(nodes=V, edges=E)
    >>> print(topo(dg))
    [3, 5, 1, 0, 2, 4, 6, 7]

    >>> V = [x for x in range(7)]
    >>> E = [(0, 1), (0, 2), (0, 4), (0, 3), (2, 5), (4, 5), (5, 3), (3, 6)]
    >>> dg = DGraph(nodes=V, edges=E)
    >>> print(topo(dg))
    [0, 4, 2, 5, 3, 6, 1]
    """
    mark, l = {}, []
    for n in g:
        mark[n] = False
    for n in g:
        __dfs_topo(g, n, mark, l)
    return list(reversed(l))

def __dfs_topo(g, node, mark, l):
    """Perform a depth first search based on the given graph and source vertex.
    Called when object is initialized.

    Args:
        node:   Node to recursively visit.
    Raises:
        KeyError:   When node is not in the graph.
    """
    if mark[node]:
        return
    mark[node] = True
    for n in g[node]:
        if not mark[n]:
            __dfs_topo(g, n, mark, l)
    l.append(node)

def dijkstra(g, src):
    """Perform a Dijkstra shortest path search on directed weighted graph g from src.

    Args:
        g:      Weighted graph, directed or undirected.
        src:    Source vertex to perform search.
    Returns:
        A tuple of 2 dicts:
            paths: key = node, value = list of nodes from src to node, inclusive.
            dist: key = node, value = int, shortest distance from src to node.
    Raises:
        KeyError

    # WGraph
    # Sample graph from CS 3345.HON F13 graphs, dijkstra's algorithm.
    >>> V = [chr(ord('a') + x) for x in range(13) if x != 8]
    >>> E = [('a', 'b', 6),('a', 'd', 9),('b', 'd', 8),('b', 'e', 7), \
    ('b', 'f', 10),('b', 'c', 4),('c', 'f', 2),('c', 'g', 3),('d', 'h', 9),\
    ('d', 'e', 5),('e', 'h', 12), ('e', 'j', 13),('e', 'f', 7),('f', 'j', 6),\
    ('f', 'k', 8), ('f', 'g', 11),('g', 'k', 1),('h', 'l', 1),('h', 'j', 2),\
    ('j', 'l', 3),('j', 'm', 15),('j', 'k', 5),('k', 'm', 6),('l', 'm', 2)]
    >>> wg = WGraph(nodes=V, edges=E)
    >>> paths, dist = dijkstra(wg, 'a')
    >>> for n in wg:
    ...     print('%s: %s, %s' % (str(n), str(paths[n]), str(dist[n])))
    a: ['a', 'a'], 0
    b: ['a', 'b'], 6
    c: ['a', 'b', 'c'], 10
    d: ['a', 'd'], 9
    e: ['a', 'b', 'e'], 13
    f: ['a', 'b', 'c', 'f'], 12
    g: ['a', 'b', 'c', 'g'], 13
    h: ['a', 'd', 'h'], 18
    j: ['a', 'b', 'c', 'f', 'j'], 18
    k: ['a', 'b', 'c', 'g', 'k'], 14
    l: ['a', 'd', 'h', 'l'], 19
    m: ['a', 'b', 'c', 'g', 'k', 'm'], 20
    """
    # mark[x]: True if node x is in the known set (marked).
    # dist[x]: Current shortest distance to x.
    # prev[x]: Previous node that points to x along the current shortest path.
    mark, dist, prev = {}, {}, {}
    for n in g:
        mark[n] = False
        dist[n] = float('inf')
        prev[n] = None
    prev[src] = src
    dist[src] = 0

    # make a priority heap based on the distance.
    # float('inf') = inf (infinity)
    heap = [(float('inf'), x) for x in g if x != src]
    heap.append((0, src))
    heapq.heapify(heap)    
    while len(heap) > 0:
        _,u = heapq.heappop(heap)
        if mark[u]:     # u has been popped out before
            continue
        mark[u] = True  # mark u as known
        for v in g[u]:
            if dist.get(v, float('inf')) > dist.get(u, float('inf')) + g.get_weight(u,v):
                dist[v] = dist[u] + g.get_weight(u,v)
                # Update dist of v in the heap.
                # It is not necessarily required to change the key (dist) of v
                # in the heap. Instead, a new tuple of new dist and v is inserted
                # into the heap. If v is popped out the first time, relax its
                # edges. Else if v has been popped out before, just ignore it.
                # https://github.com/wlxiong/k_shortest_bus_routes/wiki/Dijkstra's-algorithm-and-priority-queue
                heapq.heappush(heap, (dist[v], v))
                prev[v] = u

    # return shortest paths
    paths = {}
    for n in g:
        if dist[n] == float('inf'):     # n cannot be reached
            paths[n] = None
            continue
        path, dst = [], n   # find path from src to n
        while True:
            path.append(dst)
            dst = prev[dst]
            if dst == src:
                break
        path.append(src)
        paths[n] = list(reversed(path))
    return (paths, dist)

def bellman_ford(g, src):
    """Perform a Bellman Ford shortest path search on directed weighted graph g from src.
    Bellman Ford is slower than Dijkstra's but works with negative weight edges.
    If a graph has a negative cost cycle, it doesn't have shortest paths.
    Negative weight cycle detection is not implemented. BellmanFord will be stuck
    in an infinite loop if given a graph containing a negative weight cycle.

    Args:
        g:      Weighted graph, directed or undirected.
        src:    Source vertex to perform search.
    Returns:
        A tuple of 2 dicts:
            paths: key = node, value = list of nodes from src to node, inclusive.
            dist: key = node, value = int, shortest distance from src to node.
    Raises:
        KeyError
    
    # Sample graph from CS 3345.HON F13 graphs, dijkstra's algorithm.
    >>> V = [chr(ord('a') + x) for x in range(13) if x != 8]
    >>> E = [('a', 'b', 6), ('a', 'd', 9), ('b', 'd', 8), ('b', 'e', 7), \
    ('b', 'f', 10), ('b', 'c', 4), ('c', 'f', 2), ('c', 'g', 3), ('d', 'h', 9),\
    ('d', 'e', 5), ('e', 'h', 12), ('e', 'j', 13), ('e', 'f', 7), ('f', 'j', 6),\
    ('f', 'k', 8), ('f', 'g', 11), ('g', 'k', 1), ('h', 'l', 1), ('h', 'j', 2),\
    ('j', 'l', 3), ('j', 'm', 15), ('j', 'k', 5), ('k', 'm', 6), ('l', 'm', 2)]
    >>> wg = WGraph(nodes=V, edges=E)
    >>> paths, dist = bellman_ford(wg, 'a')
    >>> for n in wg:
    ...     print('%s: %s, %s' % (str(n), str(paths[n]), str(dist[n])))
    a: ['a', 'a'], 0
    b: ['a', 'b'], 6
    c: ['a', 'b', 'c'], 10
    d: ['a', 'd'], 9
    e: ['a', 'b', 'e'], 13
    f: ['a', 'b', 'c', 'f'], 12
    g: ['a', 'b', 'c', 'g'], 13
    h: ['a', 'd', 'h'], 18
    j: ['a', 'b', 'c', 'f', 'j'], 18
    k: ['a', 'b', 'c', 'g', 'k'], 14
    l: ['a', 'd', 'h', 'l'], 19
    m: ['a', 'b', 'c', 'g', 'k', 'm'], 20
    """
    q = deque()
    prev, dist = {}, {}
    for v in g:
        dist[v] = float('inf')
        prev[v] = None
    dist[src] = 0
    prev[src] = src
    q.appendleft(src)
    while len(q) > 0:
        v = q.pop()
        for w in g[v]:
            if dist[w] > dist[v] + g.get_weight(v,w):
                dist[w] = dist[v] + g.get_weight(v,w)
                prev[w] = v
                q.appendleft(w)

    # return shortest paths
    paths = {}
    for n in g:
        if dist[n] == float('inf'):     # n cannot be reached
            paths[n] = None
            continue
        path, dst = [], n   # find path from src to n
        while True:
            path.append(dst)
            dst = prev[dst]
            if dst == src:
                break
        path.append(src)
        paths[n] = list(reversed(path))
    return (paths, dist)

def kruskal(g):
    """Minimum spanning tree using Kruskal's algorithm.

    Args:
        g:  Undirected, weighted graph.
    Returns:
        Sorted list of edges in the MST. Each edge is a tuple (start, end, weight).

    >>> V = [0,1,2,3,4,5,6,7]
    >>> E = [(4, 5, .35), (4, 7, .37), (5, 7, .28), (0, 7, .16), (1, 5, .32),\
    (0, 4, .38), (2, 3, .17), (1, 7, .19), (0, 2, .26), (1, 2, .36),\
    (1, 3, .28), (2, 7, .34), (6, 2, .40), (3, 6, .52), (6, 0, .58), (6, 4, .93)]
    >>> wg = WGraph(nodes=V, edges=E)
    >>> print(kruskal(wg))
    [(0, 2, 0.26), (0, 7, 0.16), (1, 7, 0.19), (2, 3, 0.17), (2, 6, 0.4), (4, 5, 0.35), (5, 7, 0.28)]
    """
    edges = []
    MSTedges = []
    uf = UnionFind()
    for u in g:
        for v in g[u]:
            edges.append((g.get_weight(u,v), u, v))
    edges.sort()
    for w, u, v in edges:
        if len(edges) == len(g) - 1:
            break
        if uf[u] != uf[v]:
            uf.union(u,v)
            MSTedges.append((u,v,w))
    MSTedges.sort()
    return MSTedges

def tarjan(g):
    """Tarjan's strongly connected components and articulation points (cut vertices).

    Args:
        g:  Undirected graph, weighted or unweighted.
    Returns:
        A list of articulation points.

    >>> V = [1,2,3,4,5,6,7]
    >>> E = [(1,2),(1,4),(2,3),(3,4),(3,7),(4,5),(5,6),(4,6)]
    >>> print(tarjan(Graph(nodes=V, edges=E)))
    [4, 3]
    >>> V = [1,2,3,4,5,6]
    >>> E = [(1,2),(1,4),(2,3),(3,4),(1,5),(5,6)]
    >>> print(tarjan(Graph(nodes=V, edges=E)))
    [5, 1]
    >>> V = [1,2,3,4,5,6]
    >>> E = [(1,2),(1,4),(2,3),(3,4),(1,5),(1,6),(5,6)]
    >>> print(tarjan(Graph(nodes=V, edges=E)))
    [1]
    """
    '''
    For more information, read: http://en.wikipedia.org/wiki/Biconnected_component
    root:       Any random point from which dfs starts. Here root is the first 
                    vertex from g.

    depth:      The order in which a vertex is visited. Must be distinct. 
                    Hence the depth = [1] line below.

    forward low:    Lowpoint from one child of a vertex u. Minimum of u's depth,
                    depths of u's back vertices (neighbors of u other than its
                    direct parent in the dfs tree), lowpoints of u's children.

    num:    synonymous to "depth" of a vertex

    num back:   Depth of a neighbor of v. This neighbor must not be v's direct 
                    parent in the dfs tree.

    back edge:  An edge connecting a vertex u back to a vertex v higher in the 
                    dfs tree. v is a neighbor of u but v is not u's direct parent
                    in the dfs tree.
    '''
    root = next(g.__iter__())   # choose root to be the first vertex in the graph
    num = {}
    depth = [1]                 # put depth in a list so that it retains its value across calls
    points = []                 # list of articulation points
    __dfs_tarjan(g, root, None, root, depth, num, points)
    # print(num)
    return points

def __dfs_tarjan(g, u, parent, root, depth, num, points):
    # num
    if u not in num:
        num[u] = depth[0]
    else:
        return -num[u]      # u is in num -> back edge, u is back vertex of caller -> return u's depth
                            # The - sign for distinguishing between NumBack and FwdLow

    d = depth[0]
    numbacks = []
    fwdlows = []
    depth[0] = depth[0] + 1

    # recursive call
    for v in g[u]:
        if v != parent:     # prevent forming back edge to direct parent
            ret = __dfs_tarjan(g, v, u, root, depth, num, points)
            if ret < 0:
                # collect depths from u's neighbors other than its direct parent
                numbacks.append(-ret)
            else:
                # collect lowpoints from u's children
                fwdlows.append(ret)

    # if u is root, detect if it has multiple lows returned (= multiple children)
    if u == root:
        if len(fwdlows) > 1:
            points.append(root)

    # or if u is not root, after receiving fwdlows from children, detect if any of fwdlow >= depth(u)
    else:
        for l in fwdlows:
            if l >= d:
                points.append(u)
                break

    # calculate and return low
    return min([d] + numbacks + fwdlows)

def hierholzer(g):
    """Find an Euler circuit on the given undirected graph if one exists.

    Args:
        g:  Undirected graph.
    Returns:
        List of vertices on the circuit or None if a circuit does not exist.

    # http://www.austincc.edu/powens/+Topics/HTML/06-1/ham2.gif
    # two triangles ABC and CDE
    >>> V = ['A', 'B', 'C', 'D', 'E']
    >>> E = [('A', 'B'), ('A', 'C'), ('B', 'C'), ('C','D'), ('C', 'E'), ('D', 'E')]
    >>> g = Graph(nodes=V, edges=E)
    >>> print(hierholzer(g))
    ['A', 'B', 'C', 'D', 'E', 'C', 'A']

    # V shape graph
    >>> print(hierholzer(Graph(nodes=[1,2,3], edges=[(1,2), (2,3)])))
    None
    """
    # Check if the graph has an Euler circuit: All vertices have even degrees.
    for u in g:
        if len(list(g[u])) % 2 == 1:
            return None

    # Create necessary data structures.
    start = next(g.__iter__())  # choose the start vertex to be the first vertex in the graph
    circuit = [start]           # can use a linked list for better performance here
    traversed = {}
    ptr = 0
    while len(traversed) // 2 < g.number_of_edges() and ptr < len(circuit):
        subpath = []            # vertices on subpath
        __dfs_hierholzer(g, circuit[ptr], circuit[ptr], subpath, traversed)
        if len(subpath) != 0:   # insert subpath vertices into circuit
            circuit = list(itertools.chain(circuit[:ptr+1], subpath, circuit[ptr+1:]))
        ptr += 1

    return circuit


def __dfs_hierholzer(g, u, root, subpath, traversed):
    """Dfs on vertex u until get back to u. The argument vertices is a list and
    contains the vertices traversed. If all adjacent edges of starting vertex
    are already traversed, 'vertices' is empty after the call.
    """
    for v in g[u]:
        if (u,v) not in traversed or (v,u) not in traversed:
            traversed[(u,v)] = traversed[(v,u)] = True
            subpath.append(v)
            if v == root:
                return
            else:
                __dfs_hierholzer(g, v, root, subpath, traversed)


if __name__=='__main__':
    import doctest
    doctest.testmod()
