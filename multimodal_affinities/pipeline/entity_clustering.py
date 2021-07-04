import itertools
from itertools import chain
from multimodal_affinities.clustering.clusteringsolver import ClusteringSolver
from multimodal_affinities.blocks.cluster import Cluster
from multimodal_affinities.clustering.distance_metrics.neural_distances import embedding_distance, \
    neural_phrase_affinity_distance


class EntityClustering:
    '''
    Performs clustering logic -
    combines words / phrases to clusters with similar semantic meaning according to embedding space
    '''

    def __init__(self):
        self.clustering_solver = ClusteringSolver()
        self.distance_metrics = {
            'neural_affinity': (neural_phrase_affinity_distance, 'phrases'),
            'embedding_distance': (embedding_distance, 'phrases')
        }

    def get_distance_metric(self, document, algorithm_params):
        distance_metric_title = algorithm_params.get('distance_metric', 'neural_affinity')
        height_ratio_cutoff_threshold = algorithm_params.get('height_ratio_cutoff_threshold', 0.15)
        reduce = algorithm_params.get('reduce', 'mean')
        distance_metric_generator, entities_level = self.distance_metrics[distance_metric_title]
        distance_metric = distance_metric_generator(document, height_ratio_cutoff_threshold, reduce)

        if 'distance_metric' in algorithm_params:
            del algorithm_params['distance_metric']
        if 'height_ratio_cutoff_threshold' in algorithm_params:
            del algorithm_params['height_ratio_cutoff_threshold']

        clustering_entities = document.get_words() if entities_level == 'words' else document.phrases

        return distance_metric, clustering_entities, entities_level

    @staticmethod
    def _calculate_must_link_edges(document):
        """ Calculate from user constraints """
        words_to_phrases = document.words_to_phrases()
        must_link_edges = set()

        for constraint in document.mustlink_constraints:

            interconnected_phrases = set()

            for word in constraint:
                word = document.get_words()[word]
                phrase = words_to_phrases[word]
                interconnected_phrases.add(phrase)

            for phrases_pair in itertools.product(interconnected_phrases, interconnected_phrases):
                if phrases_pair[0] != phrases_pair[1]:
                    must_link_edges.add(frozenset(phrases_pair))

        return must_link_edges

    def cluster(self, document, algorithm, **kwargs):
        '''
        Performs phrase clustering by embedding distance.
        :param document: Document containing entities and possibly other extra information needed by
        clustering algorithms.
        :param algorithm: The algorithm in use, e.g: 'graph, 'dbscan'..
        :param kwargs: Additional hyperparameters for the clustering algorithm.
        :return: List of clusters (each cluster is an object containing a list of phrases).
        '''
        distance_metric, clustering_entities, entities_level = self.get_distance_metric(document, kwargs)
        must_link_edges = self._calculate_must_link_edges(document)
        self.clustering_solver.populate(nodes=clustering_entities, edges=must_link_edges)
        clustering_results = self.clustering_solver.solve(predicate=distance_metric,
                                                          algorithm=algorithm,
                                                          **kwargs)
        if kwargs.get('mst_threshold', 0.0) > 0.0:
            clustering_results = self.clustering_solver.minimum_spanning_tree_isolatees(predicate=distance_metric,
                                                                                        **kwargs)

        if entities_level == 'words':
            clusters = [Cluster(entities_list) for entities_list in clustering_results]
        else:
            # Break phrases to words
            clusters = []
            for phrases_list in clustering_results:
                words_list = list(chain(*[phrase.words for phrase in phrases_list]))
                clusters.append(Cluster(words_list))

        return clusters
