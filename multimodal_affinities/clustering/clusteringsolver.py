import networkx as nx
from networkx import DiGraph
from networkx.algorithms import topological_sort
from sklearn.cluster import DBSCAN
import heapq
import numpy as np
from multimodal_affinities.clustering.rect_distance_nearest_neighbors import get_isolated_nodes


class ClusteringSolver:
    '''
    Performs clustering for entities using configurable algorithms
    '''

    def __init__(self):
        self.G = DiGraph()

    def populate(self, nodes, edges=None):
        """ Initializes the clustering algorithm with entities to be clustered.
        :param nodes Entities can be any arbitrary class..
        """
        self.G.clear()
        self.G.add_nodes_from(nodes)
        if edges is not None and len(edges) > 0:
            self.G.add_edges_from(ebunch_to_add=edges, weight=0.0)

    def dbscan_cluster(self, predicate, **kwargs):
        """
        Performs clustering using DBScan
        :param predicate: A function that receives 2 nodes and determines the distance between them
        :param kwargs: DBScan hyperparameters. See scikit documentation.
        :return: List of clusters: each cluster is a list of nodes
        """
        nodes = self.G.nodes
        dist = np.empty([len(nodes), len(nodes)])
        for i, u in enumerate(nodes):
            for j, v in enumerate(nodes):
                weight = predicate(u, v)
                dist[i, j] = weight

        labels = DBSCAN(metric='precomputed', **kwargs).fit_predict(dist)

        clusters = [list() for _ in range(max(labels)+1)]
        for i, node in enumerate(nodes):
            cluster_idx = labels[i]
            if cluster_idx == -1:
                continue
            clusters[cluster_idx].append(node)
        return clusters

    def graph_cluster(self, predicate, **kwargs):
        """
        Performs simple clustering using connected components.
        :param predicate: A function that receives 2 nodes and determines the distance between them.
        The distance is expected to be 0.0 <= distance
        :param kwargs Graph clustering hyper parameters:
            - 'threshold': Minimum value of predicate indicating that nodes should be in same cluster.
        :return: List of clusters: each cluster is a list of nodes
        """
        threshold = kwargs.get('threshold', 0.0)  # Minimum predicate value before disconnecting nodes

        for u in self.G.nodes:
            for v in self.G.nodes:
                if self.G.has_edge(u, v) or self.G.has_edge(v, u):  # Protect against cycles
                    continue
                pred_u_v = predicate(u, v)
                if u != v and pred_u_v < threshold:
                    self.G.add_edge(u, v, weight=pred_u_v)

        # Calc connected components
        subgraphs = [component for component in nx.weakly_connected_component_subgraphs(self.G)]
        clusters = []
        is_sort_clusters = None
        try:
            nx.find_cycle(self.G)
            is_sort_clusters = False
        except nx.NetworkXNoCycle:
            is_sort_clusters = True
        for connected_component_graph in subgraphs:  # Topological sort each CC so entities are grouped in order
            if is_sort_clusters:
                cluster = [w for w in topological_sort(connected_component_graph)]
            else:
                cluster = [w for w in connected_component_graph]
            clusters.append(cluster)
        return clusters

    def minimum_spanning_tree_isolatees(self, predicate, **kwargs):

        threshold = kwargs.get('mst_threshold', 0.25)  # Maximum predicate value before disconnecting nodes for good
        max_island_cluster_size = kwargs.get('max_island_cluster_size', 2)  # Max size of small cluster to attach to MST

        # Distance to closest neighbor should be > mean * ratio for a node to be considered isolated
        ratio_to_mean_dist_to_isolate = kwargs.get('ratio_to_mean_dist_to_isolate', 2.0)

        remote_isolated_nodes = get_isolated_nodes(entities=list(self.G.nodes), threshold=ratio_to_mean_dist_to_isolate)
        detached_island_nodes = lambda graph: [node
                                               for component in nx.weakly_connected_component_subgraphs(graph)
                                               for node in component.nodes
                                               if len(component.nodes) <= max_island_cluster_size and
                                               node not in remote_isolated_nodes]

        heap = []
        for u in detached_island_nodes(self.G):
            for v in [node for node in self.G.nodes if node not in detached_island_nodes(self.G)]:
                if self.G.has_edge(u, v) or self.G.has_edge(v, u):  # Protect against cycles
                    continue
                pred_u_v = predicate(u, v)
                if u != v and pred_u_v < threshold:
                    edge = (u, v)
                    # Push len(heap) as a tie breaker - if 2 elements have the same threshold, the first will
                    # be arbitrarily picked
                    heapq.heappush(heap, (pred_u_v, len(heap), edge))

        isolated_set = set(detached_island_nodes(self.G))
        # Get most confident edge and attach to mst
        while len(heap) > 0:
            pred_u_v, _, edge = heapq.heappop(heap)    # Pop smallest
            if edge[0] in isolated_set or edge[1] in isolated_set:
                self.G.add_edge(edge[0], edge[1], weight=pred_u_v)
                isolated_set = set(detached_island_nodes(self.G))
                if len(isolated_set) == 0:
                    break

        # Calc connected components
        subgraphs = [component for component in nx.weakly_connected_component_subgraphs(self.G)]
        clusters = []
        is_sort_clusters = None
        try:
            nx.find_cycle(self.G)
            is_sort_clusters = False
        except nx.NetworkXNoCycle:
            is_sort_clusters = True
        for connected_component_graph in subgraphs:  # Topological sort each CC so entities are grouped in order
            if is_sort_clusters:
                cluster = [w for w in topological_sort(connected_component_graph)]
            else:
                cluster = [w for w in connected_component_graph]
            clusters.append(cluster)

        return clusters

    def solve(self, predicate, algorithm='graph', **kwargs):
        '''
        Runs the clustering algorithm
        :param algorithm One of:
            - 'graph' - Graph based clustering, using connected components and a threshold
            - 'dbscan' - Clustering based on 'dbscan'
        :param predicate: A function that receives 2 nodes and determines the distance between them.
        :param kwargs is a dictionary of hyperparameters.
            - 'graph':
                - 'threshold' - float ~ (0.0 ~ inf).
            - 'dbscan':
                - See scikit.
        :return: List of clusters in the graph. Each cluster is a list of nodes.
        '''
        if algorithm == 'graph':
            return self.graph_cluster(predicate, **kwargs)
        elif algorithm == 'dbscan':
            return self.dbscan_cluster(predicate, **kwargs)
        else:
            raise ValueError('Unsupported clustering algorithm: %r' % algorithm)
