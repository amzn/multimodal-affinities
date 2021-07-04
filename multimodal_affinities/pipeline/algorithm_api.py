import json
import torch
from multimodal_affinities.blocks.document import Document
from multimodal_affinities.pipeline.ocr_preprocessor import OCRPreprocessor
from multimodal_affinities.pipeline.phrase_detection import PhraseDetector
from multimodal_affinities.pipeline.entity_clustering import EntityClustering
from multimodal_affinities.pipeline.font_embeddings import FontEmbeddings
from multimodal_affinities.pipeline.nlp_embeddings import NLPEmbedding
from multimodal_affinities.pipeline.geometry_embeddings import GeometryEmbeddings
from multimodal_affinities.pipeline.combine_embeddings import CombineEmbeddings
from multimodal_affinities.clustering.trainable.ae_embeddings_trainer import AEEmbeddingsTrainer
import pickle

NER_DICT = {'PERSON' : 0, 'NORP' : 1, 'FACILITY' : 2, 'ORGANIZATION' : 3, 'ORG' : 3,
            'GPE' : 4, 'LOCATION' : 5, 'PRODUCT' : 6, 'EVENT' : 7, 'WORK OF ART' : 8,
            'LAW' : 9, 'LANGUAGE' : 10, 'DATE' : 11, 'TIME' : 12, 'PERCENT' : 12,
            'QUANTITY' : 15, 'ORDINAL' : 12, 'CARDINAL' : 12, 'MONEY' : 13, 'MIX': 14, 'LOC' : 16}

class CoreLogic:
    """
     =============================================
      Gateway module for accessing the app's logic
     =============================================
    """

    def __init__(self, logger):
        self.logger = logger
        self.trainer = None  # May be used repeatedly for further refinements

    @staticmethod
    def load_from_pickle(pickle_path):
        with open(pickle_path, 'rb') as pickleFile:
            core_logic = pickle.load(pickleFile)
        return core_logic

    def reset(self):
        """ Resets the current CoreLogic session data (e.g: network training, etc) """
        self.trainer = None


    @staticmethod
    def get_ner_tag(ner_string):
        def is_number(s):
            try:
                float(s)
                return True
            except ValueError:
                return False

        words = ner_string.split(' ')
        tags = [word[3:-1] for word in words if len(word) > 4 and word[0] == '<' and word[-1] == '>']

        if (len(tags) == 0 and len(words) == 1) or (len(tags) == 1 and len(words) == 2):
            if words[0].startswith('$') and is_number(words[0][1:]):
                return NER_DICT['MONEY']
            elif is_number(words[0][:-2]) and words[0].endswith('g'):
                return NER_DICT['QUANTITY']
            elif words[0] == 'MAY' or words[0] == 'May':
                return NER_DICT['DATE']

        if len(tags) == 1 and len(words) == 2:
            if tags[0] in NER_DICT:
                return NER_DICT[tags[0]]
            else:
                print('%s not in NER_DICT!' % tags[0])
                return -1
        elif len(tags) >= 3:
            return NER_DICT['MIX']
        else:
            return -1

    @staticmethod
    def load_document(doc_ocr_json_path, doc_img_path):
        '''
        Creates a new Document object out of json / img paths.
        :param doc_img_path:
        :param doc_ocr_json_path:
        :return: Document
        '''
        with open(doc_ocr_json_path) as f:
            gt_dict = json.load(f)

        ocr_preprocessor = OCRPreprocessor()
        words = ocr_preprocessor(gt_dict)
        doc = Document(words, doc_img_path)
        return doc

    def detect_phrases(self, document, algorithm, **kwargs):
        '''
        Detects phrases in the document, using ocr words as input.
        :param document: Document object.
        :param algorithm: Algorithm to use for words clustering: 'graph', 'dbscan'..
        :param kwargs: Additional hyperparameters for the clustering algorithm
        :return: List of Phrase objects
        '''
        phrase_detector = PhraseDetector()

        self.logger.info('[SpatialProximityAssociator] :: Performing phrase detection..')
        phrases = phrase_detector.detect(document=document, algorithm=algorithm, **kwargs)
        self.logger.info('[SpatialProximityAssociator] :: Phrase detection completed successfully.')
        return phrases

    def _get_font_embedder(self, embedding_params, embedding_level):
        if 'font_embeddings' not in embedding_params or len(embedding_params['font_embeddings']) == 0:
            return None
        if not embedding_params['font_embeddings'].get(embedding_level, False):
            return None
        trained_model_path = embedding_params['font_embeddings'].get('trained_model_file', None)
        font_classes_count = embedding_params['font_embeddings'].get('num_classes', 118)
        font_embedder = FontEmbeddings(trained_model_path=trained_model_path, num_classes=font_classes_count)
        return font_embedder

    def _get_nlp_embedder(self, embedding_params, embedding_level):
        params_key = 'nlp_word_embeddings' if embedding_level == 'word_level_embedding' else 'nlp_phrase_embeddings'
        if params_key not in embedding_params or len(embedding_params[params_key]) == 0:
            return None
        nlp_embedding_params = embedding_params[params_key]
        character_embeddings = nlp_embedding_params.get('character_embeddings', False)
        word_embedding_types = nlp_embedding_params.get('word_embeddings', [])
        flair_embedding_sources = nlp_embedding_params.get('contextual_flair_embeddings', [])
        elmo_embedding_sources = nlp_embedding_params.get('contextual_elmo_embeddings', [])
        bert_embedding_sources = nlp_embedding_params.get('contextual_bert_embeddings', [])

        if not character_embeddings and len(word_embedding_types) == 0 and len(flair_embedding_sources) == 0 and \
                len(elmo_embedding_sources) == 0 and len(bert_embedding_sources) == 0:
            return None
        if embedding_level == 'word_level_embedding':
            sequence_encoder = 'stacked'
        else:
            sequence_encoder = nlp_embedding_params.get('sequence_encoder', 'mean')

        nlp_embedder = NLPEmbedding(use_character_embeddings=character_embeddings,
                                    word_embedding_types=word_embedding_types,
                                    flair_embedding_sources=flair_embedding_sources,
                                    elmo_embedding_sources=elmo_embedding_sources,
                                    bert_embedding_sources=bert_embedding_sources,
                                    sequence_encoder=sequence_encoder)
        return nlp_embedder

    def _get_geometry_embedder(self, embedding_params, embedding_level):
        if 'geometry_embeddings' not in embedding_params or len(embedding_params['geometry_embeddings']) == 0:
            return None
        if not embedding_params['geometry_embeddings'].get(embedding_level, False):
            return None
        geom_embedder = GeometryEmbeddings()
        return geom_embedder

    def _get_combined_embedder(self, embedding_params):
        if 'combined_embeddings' not in embedding_params:
            embeddings_mixer = CombineEmbeddings(strategy='concat')
        else:
            combined_params = embedding_params['combined_embeddings']
            embeddings_mixer = CombineEmbeddings(strategy=combined_params['strategy'],
                                                 **combined_params['strategy_params'])
        return embeddings_mixer

    @staticmethod
    def _generate_embeddings_descriptor(document, is_word_level_embeddings,
                                        font_embedder, nlp_embedder, geom_embedder):
        """
        Generates a list of tuples, where is tuple is of (embedding_type_str, feature_dim).
        The tuples describe the embeddings contained in each word.
        A call to this function generates an embedding descriptor for word level OR phrase level.
        :param document: Document, containing word entities
        :param is_word_level_embeddings: True for word embeddings, false for phrase levle embeddings
        :param font_embedder: Embedder for extracting font embeddings
        :param nlp_embedder: Embedder for extracting NLP embeddings
        :param geom_embedder: Embedder for extracting geometry embeddings
        :return: A list of tuples, where is tuple is of (embedding_type_str, feature_dim).
        """
        # For words - start counting the embeddings from beginning to end
        # For phrases - count how many phrase embeddings we have (non-None embedders), and
        # iterate from -num_of_embedders to end
        embedding_sample = document.get_words()[0].embedding  # All words have same feature dims
        embedding_type_idx = 0 if is_word_level_embeddings \
            else -sum(x is not None for x in [font_embedder, nlp_embedder, geom_embedder])
        embeddings_descriptor = []
        if nlp_embedder:
            embedding_dim = embedding_sample[embedding_type_idx].shape[1]
            embedding_type_idx += 1
            embeddings_descriptor.append(('nlp', embedding_dim))
        if font_embedder:
            embedding_dim = embedding_sample[embedding_type_idx].shape[1]
            embedding_type_idx += 1
            embeddings_descriptor.append(('font', embedding_dim))
        if geom_embedder:
            embedding_dim = embedding_sample[embedding_type_idx].shape[1]
            embedding_type_idx += 1
            embeddings_descriptor.append(('geometry', embedding_dim))

        return embeddings_descriptor

    def _extract_multi_modal_embeddings(self, document, embedding_params, embedding_level):
        """
        Extracts multi-modal embeddings for each document entity according to configuration
        :param document: Document containing entities
        :param embedding_params: Configuration for each embedding type
        :param embedding_level: word_level_embedding or phrase_level_embedding
        :return: Document entities are updated in place with multi-modal iterable of embeddings.
        An embedding descriptor in the form of a list of tuples (embedding_type, dim) is returned.
        """
        self.logger.info('[ExtractingEmbeddings] :: Starting...')
        is_word_level_embeddings = embedding_level == 'word_level_embedding'
        is_phrase_level_embeddings = embedding_level == 'phrase_level_embedding'
        clustering_entities = document.get_words() if is_word_level_embeddings else document.phrases
        entities_to_crops = document.generate_crops(doc_entities=clustering_entities)
        font_embedder = self._get_font_embedder(embedding_params, embedding_level)
        nlp_embedder = self._get_nlp_embedder(embedding_params, embedding_level)
        geom_embedder = self._get_geometry_embedder(embedding_params, embedding_level)

        no_computed_embedding = False
        if font_embedder:
            crops = [entities_to_crops[entity] for entity in clustering_entities]
            entities_font_embedding = font_embedder.forward(crops)
        if nlp_embedder:
            self.logger.info('[EntityClustering] :: Extracting NLP embeddings..')
            # NLPEmbedder tokenizes and works at the word (token) level anyway.
            # For phrases, pooling is finally applied on all tokens
            for phrase in document.get_phrases():
                nlp_embedding, ner_string = nlp_embedder.forward(phrase.get_text())
                # print(ner_string)
                if is_phrase_level_embeddings:  # NLP Phrase level embeddings
                    entity_embedding = nlp_embedding
                    for word in phrase.words:
                        word.embedding.extend([entity_embedding.clone()])
                        word.ner_tag = CoreLogic.get_ner_tag(ner_string)
                        # print(word.ner_tag)
                else:  # NLP Word level embeddings
                    for word in phrase.words:
                        entity_embedding = nlp_embedding[word.text]
                        entity_embedding = entity_embedding.unsqueeze(0)
                        word.embedding.extend([entity_embedding])
        for i, entity in enumerate(clustering_entities):
            embeddings = []
            if font_embedder:
                self.logger.info('[EntityClustering] :: Extracting Font embeddings..')
                entity_font_embedding = entities_font_embedding[i,:].unsqueeze_(0)
                embeddings.append(entity_font_embedding)
            if geom_embedder:
                self.logger.info('[PhraseClustering] :: Extracting Geometry embeddings..')
                phrase_geom_embedding = geom_embedder.forward(entity)
                embeddings.append(phrase_geom_embedding)
            if len(embeddings) == 0:
                no_computed_embedding = True
                break

            if is_word_level_embeddings:
                entity.embedding.extend(embeddings)
            else:
                for word in entity.words:
                    word.embedding.extend(embeddings)

        if no_computed_embedding and not nlp_embedder:
            self.logger.info('[EntityClustering] :: Finished with no embeddings..')
            embeddings_descriptor = []
        else:
            embeddings_descriptor = self._generate_embeddings_descriptor(document, is_word_level_embeddings,
                                                                         font_embedder, nlp_embedder, geom_embedder)
        return embeddings_descriptor

    def _combine_embeddings(self, document, embeddings_desc, embedding_params):
        """
        Combines multi-modal embeddings into a unified representation according to embedding_params
        :param document: Document containing entities
        :param embeddings_desc: A descriptor containing the list of multi-modal types the embeddings
        are made of, and each corresponding feature dimension size.
        :param embedding_params: Configuration for each embedding type
        :return: Document entities are updated in place with multi-modal iterable of embeddings or a single tensor
        """
        embeddings_mixer = self._get_combined_embedder(embedding_params)

        entities_embeddings = [tuple(word.embedding) for word in document.get_words()]

        self.logger.info('[EntityClustering] :: Projecting embeddings together..')

        if embedding_params['combined_embeddings']['strategy'] == 'kernel_pca':
            if torch.cuda.is_available():
                embeddings_gpu = [tuple(emb.cuda() for emb in entry) for entry in entities_embeddings]
            else:
                embeddings_gpu = [tuple(emb for emb in entry) for entry in entities_embeddings]

            # concated_embeddings = embeddings_mixer.squeeze_and_concat(embeddings_gpu)
            for idx, word in enumerate(document.get_words()):
                word.unprojected_embedding = {'embeddings': embeddings_gpu[idx]}

        combined_embeddings = embeddings_mixer.forward(entities_embeddings, embeddings_desc)
        for idx, word in enumerate(document.get_words()):
            word.embedding = combined_embeddings[idx]

    def extract_embeddings(self, document, embedding_params):
        """
        Extracts embeddings for each document entity and sets them within the document entities "embedding" property.
        The exact embedding is determined according to embedding_params.
        :param document: Document containing words and / or phrases.
        :param embedding_params: Multi-modal embeddings configuration to extract.
        :return: Document entities are updated in place with extracted embeddings.
        Old embeddings, if exist, are overridden.
        """
        # Reset embeddings
        for word in document.get_words():
            word.embedding = []

        word_embeddings_desc = self._extract_multi_modal_embeddings(document=document,
                                                                    embedding_params=embedding_params,
                                                                    embedding_level='word_level_embedding')
        if document.get_phrases():
            phrase_embeddings_desc = self._extract_multi_modal_embeddings(document=document,
                                                                          embedding_params=embedding_params,
                                                                          embedding_level='phrase_level_embedding')
        embeddings_desc = []
        embeddings_desc.extend([(embedding_type + '_word', dim) for embedding_type, dim in word_embeddings_desc])
        embeddings_desc.extend([(embedding_type + '_phrase', dim) for embedding_type, dim in phrase_embeddings_desc])

        self._combine_embeddings(document, embeddings_desc, embedding_params)

    def refine_embeddings(self, document, save_training_progress=False, constraints=None, **kwargs):
        """
        Refines the embeddings by training with push / pull lose & siamese network
        :param document: Document containing words and / or phrases with embeddings.
        :param save_training_progress: Bool flag - indicating whether the embedding projections
        of each epoch should be stored in the trainer. When true - validation phase is slower
        but yields more debug information.
        :param constraints: Which constraints should the network use as labels. Any subset of:
        ['default-push', 'inter-phrase', 'mutual-knn', 'manual'].
        By default, all will be used.
        :param kwargs: Training hyperparameters
        """
        words = document.get_words()
        if len(words) == 0:
            return
        if self.trainer is None:
            # All words use the same feature dim for embeddings
            sample_embedding = words[0].embedding
            if isinstance(sample_embedding, dict) and 'projections' in sample_embedding:
                input_embedding_dim = sample_embedding['projections']
            else:
                input_embedding_dim = sample_embedding.shape[-1]
            self.trainer = AEEmbeddingsTrainer(self.logger, input_embedding_dim, **kwargs)
        else:

            # Restore unprojected embeddings for further refinement (we keep training the net and then recalculate them)
            # If the 'unprojected_embedding' attribute doesn't exist, the trainer have never been run on this document
            # before, so using the original 'embedding' attribute is the correct flow.
            for word in words:
                if hasattr(word, 'unprojected_embedding'):
                    word.embedding = word.unprojected_embedding
        self.trainer.train_embeddings(document, save_training_progress=save_training_progress,
                                      constraints=constraints)

    def cluster_entities(self, document, clustering_algorithm, **kwargs):
        """
        Clusters phrases (containing words with embeddings) together.
        The number of clusters is initially unknown.
        :param document: Document containing words and / or phrases with embeddings.
        :param clustering_algorithm: Algorithm to use for words clustering: 'graph', 'dbscan'..
        :param kwargs: Additional hyperparameters for the clustering algorithm
        :return: List of clusters
        """
        self.logger.info('[PhraseClustering] :: Performing phrase clustering..')
        entity_clustering = EntityClustering()
        clusters = entity_clustering.cluster(document, clustering_algorithm, **kwargs)
        self.logger.info('[EntityClustering] :: Phrase clustering completed successfully.')
        return clusters

    def run_full_pipeline(self, doc_ocr_json_path, doc_img_path, algorithm_config):
        """
        Runs the full pipeline of the algorithm:
        1) Loads the document
        2) Applies user constraints, if any exist
        3) Detects phrases
        4) Extracts embeddings
        5) Refines embeddings with push / pull loss
        6) Performs clustering on refined embeddings
        :param doc_ocr_json_path: Path to json file containing OCR results of the document image
        :param doc_img_path: Path to image file of the document
        :param algorithm_config: Dictionary containing configuration for the algorithm steps
        :return: Document object containing phrase & clusters as predicted by the algorithm.
        """
        document = self.load_document(doc_ocr_json_path, doc_img_path)

        # Load user constraints, if there are any
        if 'user_constraints' in algorithm_config:
            constraints = algorithm_config['user_constraints']
            document.set_mustlink_constraints(constraints.get('must_link', []))
            document.set_cannotlink_constraints(constraints.get('must_not_link', []))

        # Phrase Detection
        phrase_detection_algorithm = algorithm_config['phrase_detection']['algorithm']
        phrase_detection_params = algorithm_config['phrase_detection']['parameters']
        phrases = self.detect_phrases(document=document, algorithm=phrase_detection_algorithm, **phrase_detection_params)
        document.set_phrases(phrases)

        # Extract embeddings
        embeddings_config = algorithm_config['embeddings_initialization']
        self.extract_embeddings(document=document, embedding_params=embeddings_config)

        # Refine embeddings
        embeddings_refinement_config = algorithm_config['embeddings_refinement']
        constraints = algorithm_config['embeddings_refinement'].get('constraints', None)
        if constraints is not None:
            del embeddings_refinement_config['constraints']
        self.refine_embeddings(document=document, save_training_progress=True,
                               constraints=constraints, **embeddings_refinement_config)  # Invoke training

        # Cluster words & phrases
        clustering_algorithm = algorithm_config['clustering']['algorithm']
        clustering_params = algorithm_config['clustering']['parameters']
        clusters = self.cluster_entities(document=document,
                                         clustering_algorithm=clustering_algorithm, **clustering_params)
        document.set_clusters(clusters)

        # Clusters & phrases are stored within the document
        return document

    def __getstate__(self):
        """
        :return: For pickling - avoid storing the logger state.
        """
        return dict((k, v) for (k, v) in self.__dict__.items() if k != 'logger')
