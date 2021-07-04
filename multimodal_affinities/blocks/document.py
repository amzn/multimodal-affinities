import os, sys
import cv2
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(SRC_DIR)

from PIL import Image
from multimodal_affinities.blocks.word import Word
from multimodal_affinities.visualization.image_utils import resize_image
import copy
import PIL
import pickle

class Document(object):
    """ Represents a document - a pair of image + ocr_words dictionary """

    def __init__(self, words, image_fn):
        self.image, self.image_path = self.load_image(image_fn)
        self.basename = os.path.splitext(os.path.basename(image_fn))[0]
        img_size = self.image.shape
        self.height = img_size[0]
        self.width = img_size[1]
        self.words = words
        # entities to be computed at run-time
        self.phrases = []
        self.clusters = []
        self.mustlink_constraints = []
        self.auto_mustlink_constraints = []
        self.auto_cannotlink_constraints = []
        self.cannotlink_constraints = []
        self.aggregated_non_link = []

    def get_words(self):
        return self.words

    def get_phrases(self):
        return self.phrases

    def get_clusters(self):
        return self.clusters

    def get_constraints(self):
        return {
            'must_link': self.mustlink_constraints,
            'must_not_link': self.cannotlink_constraints
        }

    @staticmethod
    def load_document_pickle(document_path):
        with open(document_path, 'rb') as pickleFile:
            doc = pickle.load(pickleFile)
        return doc

    @staticmethod
    def load_image(image_fn):
        image = cv2.imread(image_fn, cv2.IMREAD_COLOR)

        if image is None:
            supported_formats = ['.png', '.jpg', '.jpeg', '.bmp']
            for format in supported_formats:
                image_fn = image_fn[:-4] + format
                image = cv2.imread(image_fn, cv2.IMREAD_COLOR)
                if image is not None:
                    break
        if image is None:
            raise ValueError('Document image does not exist: %r', image_fn)
        return image, image_fn

    def generate_crops_simple_resize(self, doc_entities, percent_pad=0.05,crop_width=224, crop_height=224):
        crops = []
        for i,entity in enumerate(doc_entities):
            bbox = entity.get_bbox()  # left, top, width, height
            y_min = int(round(bbox[1] * self.height))
            y_max = int(round((bbox[1] + bbox[3]) * self.height))
            x_min = int(round(bbox[0] * self.width))
            x_max = int(round((bbox[0] + bbox[2]) * self.width))
            y_padding = int(percent_pad * (y_max - y_min))
            x_padding = y_padding
            image_of_crop = self.image[max(0,y_min - y_padding):min(y_max + y_padding,self.height),
                                    max(0,x_min - x_padding):min(x_max + x_padding,self.width), :]


            image_of_crop = cv2.resize(image_of_crop, (crop_width, crop_height))
            crops.append(pil_image)
        return crops

    def generate_crops(self, doc_entities, crop_width=224, crop_height=224, percent_pad=0.1):
        crops = {}
        for i,entity in enumerate(doc_entities):
            bbox = entity.get_bbox()  # left, top, width, height
            y_min = int(round(bbox[1] * self.height))
            y_max = int(round((bbox[1] + bbox[3]) * self.height))
            x_min = int(round(bbox[0] * self.width))
            x_max = int(round((bbox[0] + bbox[2]) * self.width))
            y_padding = int(percent_pad * (y_max - y_min))
            x_padding = y_padding if (y_max - y_min) < (x_max - x_min) else 3*(y_max - y_min)
            image_of_crop = self.image[max(0,y_min - y_padding):min(y_max + y_padding,self.height),
                                    max(0,x_min - x_padding):min(x_max + x_padding,self.width), :]
            image_of_crop = resize_image(image_of_crop, [crop_width, crop_height])

            pil_image = Image.fromarray(image_of_crop)
            pil_image = pil_image.convert('RGB')
            crops[entity] = pil_image
        return crops

    def set_phrases(self, phrases):
        self.phrases = phrases

    def set_clusters(self, clusters):
        self.clusters = clusters

    def add_must_link_constraints(self, constraints):
        self.mustlink_constraints.append(constraints)

    def aggregate_cannot_link_constraints(self, constraints):
        """ Add another group of entities to the set of unlinked entities. """
        self.aggregated_non_link.append(constraints)

    def submit_cannot_link_constraints(self):
        """ Convert all aggregated document entities to a constraint on non-linked.
            Expects two groups where each a in A and b in B will realize a non-link constraint between them.
        """
        self.cannotlink_constraints.append(self.aggregated_non_link)
        self.aggregated_non_link = []

    def set_mustlink_constraints(self, constraints):
        """ Used for loading cached constraints """
        self.mustlink_constraints = copy.deepcopy(constraints)

    def set_cannotlink_constraints(self, constraints):
        """ Used for loading cached constraints """
        self.cannotlink_constraints = copy.deepcopy(constraints)

    def remove_all_must_link_constraints(self):
        self.mustlink_constraints.clear()

    def remove_all_cannot_link_constraints(self):
        self.cannotlink_constraints.clear()

    def remove_last_must_link_constraints(self):
        self.mustlink_constraints.remove(self.mustlink_constraints[-1])

    def remove_last_cannot_link_constraints(self):
        self.cannotlink_constraints.remove(self.cannotlink_constraints[-1])

    def words_to_phrases(self):
        """ Returns a mapping from each word to it's corresponding phrase """
        word_to_phrases = {}

        for phrase in self.get_phrases():
            for word in phrase.words:
                word_to_phrases[word] = phrase

        return word_to_phrases

    def words_to_clusters(self):
        """ Returns a mapping from each word to it's corresponding cluster """
        word_to_clusters = {}

        for cluster in self.get_clusters():
            for word in cluster.words:
                word_to_clusters[word] = cluster

        # Words without clusters
        for word in self.get_words():
            if word not in word_to_clusters:
                word_to_clusters[word] = None

        return word_to_clusters

    def jsonify(self):
        """ Converts document analysis results to dict containing words & phrases with their clustering ids.
            If phrases or clustering are missing, partial information will be dumped.
        """
        words_to_phrases = self.words_to_phrases()
        word_to_clusters = self.words_to_clusters()
        phrase_to_id = {}
        cluster_to_id = {}
        curr_phrase_idx = 0
        curr_cluster_idx = 0

        words_info = []
        phrases_info = []

        for word in self.get_words():
            if len(self.get_phrases()) == 0:
                phrase_id = -1
            else:
                phrase = words_to_phrases[word]
                if phrase not in phrase_to_id:
                    phrase_to_id[phrase] = curr_phrase_idx
                    curr_phrase_idx += 1
                phrase_id = phrase_to_id[phrase]

            if len(self.get_clusters()) == 0:
                cluster_id = -1
            else:
                cluster = word_to_clusters[word]
                if cluster not in cluster_to_id:
                    cluster_to_id[cluster] = curr_cluster_idx
                    curr_cluster_idx += 1
                cluster_id = cluster_to_id[cluster]

            word_dict = {
                "Geometry": {
                    "BoundingBox": word.geometry.bb_obj,
                    "Polygon": word.geometry.polygon
                },
                "DetectedText": word.text,
                "Type": "WORD",
                "PhraseId": phrase_id,
                "ClusterId": cluster_id
            }
            words_info.append(word_dict)

        for phrase in self.get_phrases():
            phrase_dict = {
                "Geometry": {
                    "BoundingBox": phrase.geometry.bb_obj,
                    "Polygon": phrase.geometry.polygon
                },
                "DetectedText": phrase.text,
                "Type": "PHRASE"
            }
            phrases_info.append(phrase_dict)

        doc_json = {
            "Words": words_info,
            "Phrases": phrases_info
        }

        return doc_json
