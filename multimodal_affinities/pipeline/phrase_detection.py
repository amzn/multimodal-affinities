import itertools
from collections import defaultdict
from multimodal_affinities.clustering.clusteringsolver import ClusteringSolver
from multimodal_affinities.clustering.distance_metrics.euclidean_v3 import *
from multimodal_affinities.blocks.phrase import Phrase


class PhraseDetector:
    '''
    Performs clustering to combine ocr words to phrases
    '''

    def __init__(self):
        # Distance metrics available
        self.dist_metrics = {
            ('graph', 'horizontal'): horizontal_metric,
            ('graph', 'vertical'): vertical_metric,
            ('dbscan', 'horizontal'): horizontal_symmetric_metric,
            ('dbscan', 'vertical'): vertical_symmetric_metric
        }

        self.clustering_solver = ClusteringSolver()

    def _get_distance_metrics(self, algorithm, direction, words, document):
        '''
        Returns the distance metric by the clustering algorithm & the direction of the clustering pass.
        The distance metric may also pack some global statistics about the words, to be used when calculating
        distances.
        :param algorithm: The algorithm in use, e.g: 'graph, 'dbscan'..
        :param direction: Direction of clustering pass: 'horizontal', 'vertical'..
        :param words: List of Word objects to cluster.
        :param document: Document object, containing additional global information
        :return: A distance metric callable function, that receives a pair of DocumentEntity objects and returns their
        normalized distance ~ [0.0, 1.0]
        '''
        dist_metric = self.dist_metrics.get((algorithm, direction))
        assert dist_metric, 'Unsupported PhraseDetection scheme: %r - %r' % (algorithm, direction)
        distance_metric_for_words = dist_metric(words, document)  # Global statistics about the words are accumulated here

        return distance_metric_for_words

    def _run_clustering_pass(self, algorithm, direction, atomic_units,
                             document, post_processing_args, **kwargs):
        '''
        Runs a single clustering pass.
        :param algorithm: The algorithm in use, e.g: 'graph, 'dbscan'..
        :param direction: Direction of clustering pass: 'horizontal', 'vertical'..
        :param atomic_units: List of Word or Phrase (sub-phrases) objects to cluster.
        :param document: Document object, containing additional global information
        :param kwargs: Additional hyperparameters for the clustering algorithm.
        :return: List of Phrases or sub-phrases (also represented as Phrases).
        '''
        self.clustering_solver.populate(atomic_units)
        distance_metric = self._get_distance_metrics(algorithm, direction, atomic_units, document)
        clusters = self.clustering_solver.solve(predicate=distance_metric, algorithm=algorithm, **kwargs)
        # clusters = self.post_process(clusters, post_processing_args)
        # clusters = self.break_phrases_according_to_user_constraints(clusters, atomic_units,
        #                                                             document.cannotlink_constraints)
        phrases = [Phrase(cluster) for cluster in clusters]

        return phrases

    def post_process(self, clusters, post_processing_args):

        dist_threshold = post_processing_args['post_processing_ordinals_threshold']
        is_break_ordinals_at_start = True
        is_break_ordinals_at_end = True
        if dist_threshold < 1e-6:   # if dist_threshold -> 0, don't post process
            return clusters

        updated_clusters = []

        for cluster in clusters:

            if len(cluster) < 3:
                updated_clusters.append(cluster)
                continue

            is_nonalpha = lambda w: all(not c.isalpha() for c in w.text)
            dists_within_cluster = [b.geometry.left - (a.geometry.left + a.geometry.width)
                                    for a, b in zip(cluster[:-1], cluster[1:])]
            dists_sum = sum(dists_within_cluster)
            avg_without_prefix = (dists_sum - dists_within_cluster[0]) / (len(dists_within_cluster) - 1)
            avg_without_suffix = (dists_sum - dists_within_cluster[-1]) / (len(dists_within_cluster) - 1)

            if is_break_ordinals_at_start and is_nonalpha(cluster[0]) and \
                    dists_within_cluster[0] >= avg_without_prefix * dist_threshold:
                updated_clusters.append([cluster[0]])
                cluster = cluster[1:] if len(cluster) > 1 else None

            if is_break_ordinals_at_end and is_nonalpha(cluster[-1]) and \
                    dists_within_cluster[-1] >= avg_without_suffix * dist_threshold:
                updated_clusters.append([cluster[-1]])
                cluster = cluster[:-1] if len(cluster) > 1 else None

            if cluster is not None:
                updated_clusters.append(cluster)

        return updated_clusters

    def break_phrases_according_to_user_constraints(self, clusters, atomic_units, cannotlink_constraints):

        clusters_to_break = defaultdict(set)

        for nonlink_constraint in cannotlink_constraints:
            for constraint_pair in itertools.product(nonlink_constraint[0], nonlink_constraint[1]):
                for cluster_idx, cluster in enumerate(clusters):
                    word_pair = (atomic_units[constraint_pair[0]], atomic_units[constraint_pair[1]])
                    if word_pair[0] in cluster and word_pair[1] in cluster:
                        if (word_pair[1], word_pair[0]) not in clusters_to_break[cluster_idx]:
                            clusters_to_break[cluster_idx].add(word_pair)

        result_clusters = [cluster for cluster_idx, cluster in enumerate(clusters)
                           if cluster_idx not in clusters_to_break]

        # Handle phrase break here
        for cluster_idx, anchor_pairs in clusters_to_break.items():
            cluster = clusters[cluster_idx]
            print(cluster)
            print(anchor_pairs)

        return result_clusters

    @staticmethod
    def _extract_post_processing_args(**kwargs):
        post_processing_ordinals_threshold = kwargs.get('post_processing_ordinals_threshold', 0.0)
        return dict(post_processing_ordinals_threshold=post_processing_ordinals_threshold)

    def detect(self, document, algorithm, **kwargs):
        '''
        Performs phrase detection by running 2 clustering passes: horizontal and vertical.
        :param document: Document, containing list of Word objects to group into phrases.
        :param algorithm: The algorithm in use, e.g: 'graph, 'dbscan'..
        :param kwargs: Additional hyperparameters for the clustering algorithm.
        :return: List of phrases.
        '''
        words = document.get_words()
        post_processing_args = self._extract_post_processing_args(**kwargs)

        horizontal_kwargs = self._filter_dict(kwargs, 'horizontal')
        sub_phrases = self._run_clustering_pass(algorithm=algorithm, direction='horizontal',
                                                atomic_units=words, document=document,
                                                post_processing_args=post_processing_args,
                                                **horizontal_kwargs)
        vertical_kwargs = self._filter_dict(kwargs, 'vertical')
        phrases = self._run_clustering_pass(algorithm=algorithm, direction='vertical',
                                            atomic_units=sub_phrases, document=document,
                                            post_processing_args=post_processing_args,
                                            **vertical_kwargs)

        return phrases

    @staticmethod
    def _filter_dict(dictionary, subword):
        """
        Filters dictionary by only keeping key:value pairs where key is of the form '<subword>_...'.
        For such pairs, it removes the <subword>_ suffix from the key.
        :param dictionary: a dictionary where the keys are strings.
        :param subword: a string.
        """
        start_point = len(subword) + 1
        filtered_dictionary = {key[start_point:] : val for key, val in dictionary.items() if key.startswith(subword)}
        return filtered_dictionary