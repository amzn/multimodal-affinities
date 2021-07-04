import itertools
from multimodal_affinities.blocks.document_entity import DocumentEntity
from multimodal_affinities.blocks.cluster import Cluster
from multimodal_affinities.clustering.clusteringsolver import ClusteringSolver
from multimodal_affinities.evaluation.analysis.clustering_measurements import doc_to_labels


def _constraints_to_edges(document):
    entity_pairs = {}
    entities = document.get_words()

    must_link_groups = document.get_constraints()['must_link']
    for group in must_link_groups:
        all_entity_indices_pairs = list(itertools.combinations(group, 2))
        for idx1, idx2 in all_entity_indices_pairs:
            entity1 = entities[idx1]
            entity2 = entities[idx2]
            pair_key = _key(entity1, entity2)
            entity_pairs[pair_key] = 1

    must_not_link_groups = document.get_constraints()['must_not_link']
    for pair_of_must_not_link in must_not_link_groups:
        group1 = pair_of_must_not_link[0]
        group2 = pair_of_must_not_link[1]
        must_not_link_entity_pairs = list(itertools.product(group1, group2))
        for idx1, idx2 in must_not_link_entity_pairs:
            entity1 = entities[idx1]
            entity2 = entities[idx2]
            pair_key = _key(entity1, entity2)
            entity_pairs[pair_key] = 0

    return entity_pairs


def _key(*args):
    """
    Returns a key for unordered pair of entities
    :param args: Tuple of entities or 2 document entities
    :return: Unique key, invariant to order of entities in pair
    """
    if len(args) == 1 and isinstance(args[0], tuple):
        entities_tuple = args[0]
        return frozenset(entities_tuple)
    elif len(args) == 2 and isinstance(args[0], DocumentEntity) and isinstance(args[1], DocumentEntity):
        entity1 = args[0]
        entity2 = args[1]
        return frozenset((entity1, entity2))
    else:
        raise ValueError('Unsupported key type in DocumentEntityPairsDataset')


def gt_from_user_constraints(document):
    edges = _constraints_to_edges(document)
    clustering_solver = ClusteringSolver()
    clustering_solver.populate(nodes=document.get_words())

    def _hard_distance(w1, w2):
        if w1 != w2 and edges.get(_key(w1, w2), 0) == 1:
            dist = 0.1
        else:
            dist = 1
        return dist

    graph_clustering_params = {
        'threshold': 0.5
    }
    clustering_results = clustering_solver.solve(predicate=_hard_distance,
                                                 algorithm='graph',
                                                 **graph_clustering_params)
    clusters = [Cluster(entities_list) for entities_list in clustering_results]
    document.set_clusters(clusters)

    gt_labels = doc_to_labels(document)
    return gt_labels
