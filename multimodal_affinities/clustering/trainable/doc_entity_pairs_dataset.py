# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-4.0

import itertools
import torch
from torch.utils.data.dataset import Dataset
from multimodal_affinities.blocks.document_entity import DocumentEntity
from multimodal_affinities.clustering.trainable.auto_constraints import AutoConstraints
import numpy as np

class DocumentEntityPairsDataset(Dataset):

    # Predefined confidence constants
    default_nonlink_confidence = 0.6
    phrase_detection_must_link_confidence = 0.9
    manual_constraints_confidence = 1.5
    knn_must_link_confidence = 0.8

    def __init__(self, document, embedding_level='words', mode=None,
                 constraints=None, constraints_types=None,
                 downsample_default=True, downsample_user=True, ratio_must_to_cannot=2,
                 font_word_mknn=3, max_mustlink=400):
        """
        Creates a dataset of pairs of document entities with label 1 if they're linked and 0 if not.
        :param document: Document object
        :param embedding_level: Level of embeddings: 'words' or 'phrases'
        :param mode: Which pairs should be included, any combination of: {'must_link', 'non_linked'}.
        :param constraints: Which constraints should be included, any combination of:
        {'default-push', 'inter-phrase', 'mutual-knn', 'manual'}.
        Use different modes to train on "pull" / "push" losses (or both)
        """
        # Defaults
        np.random.seed(0)

        self.mutual_knn_params = {
            'font_word': (font_word_mknn, 'cosine'),
            'font_phrase': (2, 'cosine'),
            'nlp_word': (2, 'cosine'),
            'nlp_phrase': (2, 'cosine'),
            'geometry_word': (0, 'cosine'),
            'geometry_phrase': (2, 'cosine')
        }

        if mode is None:
            mode = ['must_link', 'non_linked']
        if constraints is None:
            constraints = ['inter-phrase', 'mutual-knn', 'push-mutual-knn' 'manual']
        if constraints_types is None:
            constraints_types = ['font', 'nlp-ner', 'geometry']

        self.constraints_types = constraints_types
        entity_pairs_intra_phrase = self._gather_intra_phrase_constraints(document)
        num_of_intra_phrase_constraints = len(entity_pairs_intra_phrase) #assumes default push is off
        entity_pairs_manual, num_of_mustlink_manual, num_of_cannotlink_manual, must_link_indices, cannot_link_indices = self._apply_manual_constraints(document,embedding_level,['manual'],{})
        entity_pairs_knn_mustlink, num_of_knn_mustlink_constraints = self._apply_knn_auto_constraints(document, constraints, {}, w_multiplier=1, filter_list = cannot_link_indices)
        # use k = 3 for cannot link
        self._update_mknn_param_value(3)
        entity_pair_knn_cannotlink, num_of_knn_cannotlink_constraints = self._apply_knn_auto_constraints(document, constraints, {}, w_multiplier=-1, filter_list = must_link_indices)


        num_of_mustlink = num_of_intra_phrase_constraints + num_of_knn_mustlink_constraints + num_of_mustlink_manual
        num_of_cannotlink = num_of_knn_cannotlink_constraints + num_of_cannotlink_manual

        print("Before removal: total mustlink {}, total cannotlink {}".format(num_of_mustlink, num_of_cannotlink))

        if num_of_mustlink > max_mustlink:
            num_of_mustlink_intra_phrase_to_remove_ = min(num_of_mustlink - max_mustlink, num_of_intra_phrase_constraints)
            num_of_mustlink_knn_to_remove_ = max(num_of_mustlink - max_mustlink - num_of_mustlink_intra_phrase_to_remove_, 0)
            num_of_mustlink = max_mustlink
        else:
            num_of_mustlink_intra_phrase_to_remove_ = 0
            num_of_mustlink_knn_to_remove_ = 0


        if num_of_mustlink > ratio_must_to_cannot * num_of_cannotlink > 0:
            # first sample from intra phrase constraints
            num_of_mustlink_intra_phrase_to_remove = num_of_mustlink_intra_phrase_to_remove_ + min(num_of_mustlink - ratio_must_to_cannot * num_of_cannotlink, num_of_intra_phrase_constraints)
            num_of_mustlink_knn_to_remove = num_of_mustlink_knn_to_remove_ + max(num_of_mustlink - ratio_must_to_cannot * num_of_cannotlink - num_of_mustlink_intra_phrase_to_remove, 0)
            num_of_cannotlink_to_remove = 0
        else:
            num_of_mustlink_intra_phrase_to_remove = num_of_mustlink_intra_phrase_to_remove_
            num_of_mustlink_knn_to_remove = num_of_mustlink_knn_to_remove_
            num_of_cannotlink_to_remove = min(num_of_cannotlink - int(num_of_mustlink / ratio_must_to_cannot), num_of_cannotlink)
            num_of_cannotlink_to_remove = max(num_of_cannotlink_to_remove, 0)

        self.linked_data = []
        self.non_linked_data = []
        self.add_to_linked_data(entity_pairs_intra_phrase, num_of_mustlink_intra_phrase_to_remove)
        self.add_to_linked_data(entity_pairs_knn_mustlink, num_of_mustlink_knn_to_remove)
        self.add_to_linked_data(entity_pair_knn_cannotlink, num_of_cannotlink_to_remove)
        self.add_to_linked_data(entity_pairs_manual, 0)

        print("total mustlink {}, total cannotlink {}, defined ratio {}".format(num_of_mustlink, num_of_cannotlink, ratio_must_to_cannot))
        print("removed {}, num_of_removed_mustlink_intra_phrase".format(num_of_mustlink_intra_phrase_to_remove))
        print("removed {}, num_of_removed_mustlink_mknn".format(num_of_mustlink_knn_to_remove))
        print("removed {}, num_of_removed_cannotlink_mknn".format(num_of_cannotlink_to_remove))
        self.mode = mode

    def add_to_linked_data(self, entity_pairs, num_of_pairs_to_remove):
        num_total = len(entity_pairs)
        num_of_pairs_to_remove = int(num_of_pairs_to_remove)
        indices_to_remove = np.random.permutation(num_total)[:min(num_of_pairs_to_remove,num_total)]
        counter = -1
        for pair_set, label in entity_pairs.items():
            counter = counter + 1
            if counter in indices_to_remove:
                continue
            is_connected = label[0]
            confidence = label[1]
            # downsample default non link constraints
            if is_connected == 1:
                self.linked_data.append((pair_set, confidence))
            else:
                self.non_linked_data.append((pair_set, confidence))

    @staticmethod
    def _gather_intra_phrase_constraints(document):
        data = {}
        DocumentEntityPairsDataset._apply_phrase_detection_constraints(document, data)
        print("number of intra phrase constraints: {}".format(len(data)))
        return data

    @staticmethod
    def _gather_entities(document, embedding_level, constraints):
        """
        Creates a dict of pairs that must / must not be linked according to phrase detection results.
        :param document: Document object, containing phrases from latest phrase detection run.
        :param embedding_level: Level of document entities pairs to work with: 'words' or 'phrases'
        :param constraints: Which constraints should be included, any combination of:
        {'default-push', 'inter-phrase', 'mutual-knn', 'manual'}.
        :return: dict of (word, word) -> bool, float   or  (phrase, phrase) -> bool, float where 1 signifies the entities should
        be linked and 0 means the entities shouldn't be linked. The second float param is the confidence.
        """

        data = {}
        default_confidence = DocumentEntityPairsDataset.default_nonlink_confidence

        if embedding_level == 'words':
            if 'default-push' in constraints:
                print("--- setting cannot link constraints between all pairs ---")
                entities = document.get_words()
                pairs = list(itertools.combinations(entities, r=2))
                for pair in pairs:
                    data[_key(pair)] = 0, default_confidence
            if 'inter-phrase' in constraints:
                DocumentEntityPairsDataset._apply_phrase_detection_constraints(document, data)
        elif embedding_level == 'phrases':
            if 'default-push' in constraints:
                print("--- setting cannot link constraints between all pairs ---")
                entities = document.get_phrases()
                pairs = list(itertools.combinations(entities, r=2))
                for pair in pairs:
                    data[_key(pair)] = 0, default_confidence
        else:
            raise ValueError('Incompatible embedding_level arg given to DocumentEntityPairsDataset')


        print("number of default constraints: {}".format(len(data)))

        return data

    @staticmethod
    def _apply_phrase_detection_constraints(document, data):
        """
        Applies phrase-detection results constraints.
        All words grouped in the same phrase are automatically tagged as "must link"
        :param document: Document object, containing phrases (results from phrase detection phase)
        :param data: The dict of (word, word) -> 0 or 1, confidence - defining links between words.
        :return: data is updated in place
        """
        for phrase in document.get_phrases():
            all_word_pairs = list(itertools.combinations(phrase.words, r=2))
            for word_pair in all_word_pairs:
                data[_key(word_pair)] = 1, DocumentEntityPairsDataset.phrase_detection_must_link_confidence

    def _update_mknn_param_value(self, new_val):
        self.mutual_knn_params['font_word'] = (new_val, 'cosine')

    def _apply_knn_auto_constraints(self, document, constraints, entity_pairs, w_multiplier, filter_list):
        """
        Applies automatic k-nearest-neighbours constraints on top of entity_pairs.
        Entities that belong to the same "k-clique" automatically gain a must-link label.
        :param document: Document object, containing phrases (results from phrase detection phase)
        :param constraints: Which constraints should be included, any combination of:
        {'default-push', 'inter-phrase', 'mutual-knn', 'manual'}.
        :param entity_pairs: Dictionary containing all pairs of entities with a label 1 for link and 0 for no-link.
        :return: data is updated in place
        """
        if 'mutual-knn' not in constraints:
            return entity_pairs, 0
        print("--- setting Auto constraints ---")
        auto_const_calculator = AutoConstraints()
        auto_constraints = []
        doc_entities = document.get_words()
        entity_embedding_sample = doc_entities[0].embedding
        # entity embeddings is a dict of multi-modal embeddings
        if isinstance(entity_embedding_sample, dict):
            num_of_embeddings = len(entity_embedding_sample['embeddings'])

            is_font_constraints_enabled = False
            is_nlp_constraints_enabled = False
            is_geometry_constraints_enabled = False
            for embedding_idx in range(num_of_embeddings):
                embedding_type = entity_embedding_sample['embeddings_desc'][embedding_idx][0]
                if embedding_type.startswith('font') and 'font' in self.constraints_types:
                    is_font_constraints_enabled = True
                elif embedding_type.startswith('nlp') and 'nlp-ner' in self.constraints_types:
                    is_nlp_constraints_enabled = True
                elif embedding_type.startswith('geometry') and 'geometry' in self.constraints_types:
                    is_geometry_constraints_enabled = True

            for embedding_idx in range(num_of_embeddings):
                embeddings = [entity.embedding['embeddings'][embedding_idx] for entity in doc_entities]
                embedding_type = entity_embedding_sample['embeddings_desc'][embedding_idx][0]
                auto_constraints_per_embedding = None
                if embedding_type.startswith('font') and 'font' in self.constraints_types:
                    n_neighbors, dist_measure = self.mutual_knn_params[embedding_type]
                    auto_constraints_per_embedding = auto_const_calculator.generate_auto_must_link_const_from_embeddings(
                        embeddings=embeddings,
                        n_neighbors=n_neighbors,
                        dist_meas=dist_measure,
                        w_multiplier=w_multiplier)

                    def _filter_height(pair):
                        if not is_geometry_constraints_enabled:
                            return True
                        word_a_idx, word_b_idx = pair
                        word_a_height = doc_entities[word_a_idx].geometry.height
                        word_b_height = doc_entities[word_b_idx].geometry.height
                        word_ratio = float(word_a_height) / word_b_height
                        return word_ratio < 1.5 and word_ratio > 1 / 1.5

                    def _filter_caps(pair):
                        if not is_nlp_constraints_enabled:
                            return True
                        word_a_idx, word_b_idx = pair
                        word_a = doc_entities[word_a_idx].text
                        word_b = doc_entities[word_b_idx].text
                        word_a_type = 0 if word_a.isupper() else 1 if word_a.islower() else 2
                        word_b_type = 0 if word_b.isupper() else 1 if word_b.islower() else 2
                        return word_a_type == word_b_type

                    def _filter_ner(pair):
                        if not is_nlp_constraints_enabled:
                            return True
                        word_a_idx, word_b_idx = pair
                        word_a_tag = doc_entities[word_a_idx].ner_tag
                        word_a_tag = word_a_tag if word_a_tag is not None else -1
                        word_b_tag = doc_entities[word_b_idx].ner_tag
                        word_b_tag = word_b_tag if word_b_tag is not None else -1
                        ner_unknown = word_a_tag == -1 or word_b_tag == -1
                        return ner_unknown or word_a_tag == word_b_tag

                    def _filter_short(pair, allow_if_ner = False):
                        if not is_nlp_constraints_enabled:
                            return True
                        word_a_idx, word_b_idx = pair
                        word_a = doc_entities[word_a_idx].text
                        word_b = doc_entities[word_b_idx].text
                        # allow to pass if contains tag
                        if allow_if_ner:
                            word_a_tag = doc_entities[word_a_idx].ner_tag
                            word_a_tag = word_a_tag if word_a_tag is not None else -1
                            word_b_tag = doc_entities[word_b_idx].ner_tag
                            word_b_tag = word_b_tag if word_b_tag is not None else -1
                        else:
                            word_a_tag = -1
                            word_b_tag = -1
                        len_word_a = len(word_a)
                        len_word_b = len(word_b)
                        return (len_word_a >= 3 or word_a_tag >= 0) and (len_word_b >= 3 or word_b_tag >= 0)

                    if w_multiplier > 0: # do some filtering for mustlink
                        auto_constraints_per_embedding = [constraint for constraint in auto_constraints_per_embedding if
                                                          _filter_height(constraint) and _filter_caps(constraint)
                                                          and _filter_ner(constraint) and _filter_short(constraint, True)]
                    else:
                        auto_constraints_per_embedding = [constraint for constraint in auto_constraints_per_embedding if
                                                          _filter_short(constraint, False)]

                elif embedding_type.startswith('geometry') and 'geometry' in self.constraints_types:
                    if w_multiplier < 0:
                        # add geometric cannot link constraints
                        auto_constraints_per_embedding = auto_const_calculator.generate_auto_cannot_link_const_from_geometric_embeddings(
                            embeddings=embeddings,
                            ratio_threshold = 1.5)

                if auto_constraints_per_embedding is not None:
                    auto_constraints.extend(auto_constraints_per_embedding)

            if 'nlp-ner' in self.constraints_types:
                if w_multiplier < 0:
                    # add geometric cannot link constraints
                    auto_constraints_per_embedding = auto_const_calculator.generate_auto_cannot_link_const_from_ner_tags(doc_entities)
                    if auto_constraints_per_embedding is not None:
                        auto_constraints.extend(auto_constraints_per_embedding)


        def _filter_from_filter_list(pair, filter_list = []):
            a_idx, b_idx = pair
            found = [1 for filter_pair in filter_list if (filter_pair[0] == a_idx and filter_pair[1] == b_idx) or
                     (filter_pair[1] == a_idx and filter_pair[0] == b_idx)]
            return len(found) == 0

        auto_constraints = [constraint for constraint in auto_constraints if
                            _filter_from_filter_list(constraint, filter_list)]
        if w_multiplier > 0:
            document.auto_mustlink_constraints = auto_constraints
        else:
            document.auto_cannotlink_constraints = auto_constraints
        benefit = 0
        fails = 0

        for mutually_knn_pair in auto_constraints:
            idx1, idx2 = mutually_knn_pair
            entity1 = doc_entities[idx1]
            entity2 = doc_entities[idx2]
            pair_key = _key(entity1, entity2)
            if entity1 == entity2:
                fails += 1
                continue
            elif pair_key not in entity_pairs or entity_pairs[pair_key][0] != 1:
                benefit += 1

            if w_multiplier > 0:
                entity_pairs[pair_key] = 1, DocumentEntityPairsDataset.knn_must_link_confidence
            else:
                entity_pairs[pair_key] = -1, DocumentEntityPairsDataset.knn_must_link_confidence

        print('Auto constraint have contributed %r new must-links and wasted %r opportunities' % (benefit, fails))
        return entity_pairs, benefit

    def _apply_manual_constraints(self, document, embedding_level, constraints, entity_pairs):
        """
        Adds user constraints on top of the entity_pairs dictionary.
        :param document: Document containing constraints in a form of a dict.
        :param embedding_level: Are entities treated at word, phrase level, etc.
        :param constraints: Which constraints should be included, any combination of:
        {'default-push', 'inter-phrase', 'mutual-knn', 'manual'}.
        :param entity_pairs: Dictionary containing all pairs of entities with a label 1 for link and 0 for no-link.
        :return: Updated entity_pairs with labels from the constraints.
        """
        if 'manual' not in constraints:
            return entity_pairs, 0, 0
        print("--- setting user constraints ---")
        manual_constraints_confidence = DocumentEntityPairsDataset.manual_constraints_confidence
        entities = document.get_words() if embedding_level == 'words' else document.get_phrases()

        num_of_must_link = 0
        must_link_groups = document.get_constraints()['must_link']
        must_link_indices = []
        for group in must_link_groups:
            all_entity_indices_pairs = list(itertools.combinations(group, 2))
            for idx1, idx2 in all_entity_indices_pairs:
                entity1 = entities[idx1]
                entity2 = entities[idx2]
                pair_key = _key(entity1, entity2)
                entity_pairs[pair_key] = 1, manual_constraints_confidence
                num_of_must_link += 1
                must_link_indices.append([idx1, idx2])

        num_of_cannot_link = 0
        must_not_link_groups = document.get_constraints()['must_not_link']
        must_not_link_indices = []
        for pair_of_must_not_link in must_not_link_groups:
            group1 = pair_of_must_not_link[0]
            group2 = pair_of_must_not_link[1]
            must_not_link_entity_pairs = list(itertools.product(group1, group2))
            for idx1, idx2 in must_not_link_entity_pairs:
                entity1 = entities[idx1]
                entity2 = entities[idx2]
                pair_key = _key(entity1, entity2)
                entity_pairs[pair_key] = 0, manual_constraints_confidence
                num_of_cannot_link += 1
                must_not_link_indices.append([idx1,idx2])


        print("num of user must-link constraints: {}, cannot link {}".format(num_of_must_link, num_of_cannot_link))

        return entity_pairs, num_of_must_link, num_of_cannot_link, must_link_indices, must_not_link_indices

    def __len__(self):
        total = 0
        if 'must_link' in self.mode:
            total += len(self.linked_data)
        if 'non_linked' in self.mode:
            total += len(self.non_linked_data)
        return total

    @staticmethod
    def _get_embeddings(entity):
        """ Extract embedding from entity whether it's a multi-modal tuple or a single tensor """
        if isinstance(entity.embedding, dict):
            return [torch.tensor(feature.detach().cpu().numpy()).squeeze() for feature in entity.embedding['embeddings']]
        else:
            return entity.embedding

    def __getitem__(self, idx):
        if len(self.mode) == 1:
            pairs_list = self.linked_data if 'must_link' in self.mode else self.non_linked_data
            y = 1 if 'must_link' in self.mode else 0
            sample_idx = idx
        else:
            pairs_list = self.linked_data if idx < len(self.linked_data) else self.non_linked_data
            y = 1 if idx < len(self.linked_data) else 0
            sample_idx = idx if pairs_list == self.linked_data else idx - len(self.linked_data)

        pair, confidence = pairs_list[sample_idx]
        x1, x2 = pair
        x1_embedding, x2_embedding = self._get_embeddings(x1), self._get_embeddings(x2)
        return x1_embedding, x2_embedding, y, confidence


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
