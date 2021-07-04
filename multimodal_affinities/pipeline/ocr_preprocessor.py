from multimodal_affinities.blocks.word import Word
from multimodal_affinities.blocks.geometry import Geometry
from shapely.geometry import Polygon
import string
import numpy as np
import logging


class OCRPreprocessor:

    def __init__(self, height_width_ratio_threshold_high=5, height_width_ratio_threshold_low=2,
                 overlap_threshold=0.7, min_line_confidence=0.8, min_word_confidence=0.1, consider_words_with_lines_only_flag = True):
        """
        Post processing for ocr results
        :param height_width_ratio_threshold: Filter out if word height > width * ratio
        :param overlap_threshold: Filter out if 2 words' overlap > threshold
        """
        self.logger = logging
        self.height_width_ratio_threshold_high = height_width_ratio_threshold_high
        self.height_width_ratio_threshold_low = height_width_ratio_threshold_low
        self.overlap_threshold = overlap_threshold
        self.min_line_confidence = min_line_confidence
        self.min_word_confidence = min_word_confidence
        self.marked_for_removal = set()
        self._allowed_chars = string.ascii_letters + string.digits
        self.area_threshold_for_ratio = 0
        self.consider_words_with_lines_only_flag = consider_words_with_lines_only_flag

    @staticmethod
    def measure_overlap(word1, word2):
        """ Computes the overlap coefficient:
        https://en.wikipedia.org/wiki/Overlap_coefficient
        This metric is different from IoU as if one box is a subset of the other this metric equals to 1.0
        (in other words: IoU gets low score on composition if one box is significantly larger, while this metric
        scores high).
        :param word1: The first bbox to compute the overlap between the pair
        :param word2: The second bbox to compute the overlap between the pair
        :return: The overlap value between 0 to 1
        """
        # Must convert to format: [(x0, y0), (x1, y1), (x2, y2), (x3, y3)]
        bbox1 = word1.geometry.get_bb_corners()
        bbox2 = word2.geometry.get_bb_corners()
        polygon1 = Polygon(bbox1)
        polygon2 = Polygon(bbox2)
        overlap_area = polygon1.intersection(polygon2).area
        min_area = min(polygon1.area, polygon2.area)
        # avoid divide by zero
        if min_area == 0:
            return 0
        return overlap_area / min_area

    def _filter_by_threshold(self, word):
        area = Polygon(word.geometry.get_bb_corners()).area
        if area > self.area_threshold_for_ratio:
            return not (word.geometry.width * self.height_width_ratio_threshold_low < word.geometry.height)
        else:
            return not (word.geometry.width * self.height_width_ratio_threshold_high < word.geometry.height)

    def _filter_by_text(self, word):
        # contains_valid = any(char in self._allowed_chars for char in word.text)
        return not len(word.text) == 0 #and contains_valid

    def _single_word_filters(self, word):
        return self._filter_by_threshold(word) and self._filter_by_text(word)

    def _filter_by_overlap(self, word_tuple):

        word = word_tuple[0]
        all_words = word_tuple[1]
        for other_word in all_words:
            if word == other_word:
                continue
            if self.measure_overlap(word, other_word) > self.overlap_threshold and \
                word.geometry.area() >= other_word.geometry.area() and \
                    other_word not in self.marked_for_removal:
                        self.marked_for_removal.add(word)
                        return False

        return True

    @staticmethod
    def convert_to_old_engine_geometry_format(left, top, right, bottom):
        geometry_dict = {
            "BoundingBox": {
                "Width": right - left,
                "Top": top,
                "Height": bottom - top,
                "Left": left
            },
            "Polygon": [
                {
                    "Y": top,
                    "X": left
                },
                {
                    "Y": top,
                    "X": right
                },
                {
                    "Y": bottom,
                    "X": right
                },
                {
                    "Y": bottom,
                    "X": left
                }
            ]
        }
        return geometry_dict

    def handle_new_ocr_format(self, doc_dict):
        doc_objects = doc_dict['objects']

        words = []
        word_idx_to_line = {}

        for entity in doc_objects:
            if entity['type'] == 'Word':
                word_dict = entity
                if word_dict['score'] < self.min_word_confidence:
                    continue
                left = min([entry['x'] for entry in word_dict['bbox']['vertices']]) / doc_dict['imageSize']['width']
                right = max([entry['x'] for entry in word_dict['bbox']['vertices']]) / doc_dict['imageSize']['width']
                top = min([entry['y'] for entry in word_dict['bbox']['vertices']]) / doc_dict['imageSize']['height']
                bottom = max([entry['y'] for entry in word_dict['bbox']['vertices']]) / doc_dict['imageSize']['height']
                word_dict['Geometry'] = self.convert_to_old_engine_geometry_format(left, top, right, bottom)
                if word_dict.get('text'):
                    word_dict['DetectedText'] = word_dict['text']['content']
                    word = Word(word_dict)
                    words.append(word)
                    word_idx_to_line[word] = (None, self.overlap_threshold)

        for entity in doc_objects:
            if entity['type'] == 'Line':
                line = entity
                if line['score'] < self.min_line_confidence:
                    continue
                left = min([entry['x'] for entry in line['bbox']['vertices']]) / doc_dict['imageSize']['width']
                right = max([entry['x'] for entry in line['bbox']['vertices']]) / doc_dict['imageSize']['width']
                top = min([entry['y'] for entry in line['bbox']['vertices']]) / doc_dict['imageSize']['height']
                bottom = max([entry['y'] for entry in line['bbox']['vertices']]) / doc_dict['imageSize']['height']
                line_geometry = self.convert_to_old_engine_geometry_format(left, top, right, bottom)
                line = Word(dict(Geometry=line_geometry, DetectedText='dummy'))
                for word in words:
                    overlap = self.measure_overlap(word, line)
                    best_overlap = word_idx_to_line[word][1]
                    if overlap > best_overlap:
                        word_idx_to_line[word] = (line, overlap)

        words_with_lines = []

        for word, overlap_entry in word_idx_to_line.items():
            line = overlap_entry[0]
            if line is None:
                continue
            left = word.geometry.left
            right = word.geometry.left + word.geometry.width
            top = line.geometry.top
            bottom = line.geometry.top + line.geometry.height
            word.geometry = Geometry(self.convert_to_old_engine_geometry_format(left, top, right, bottom))
            words_with_lines.append(word)
        if self.consider_words_with_lines_only_flag:
            return words_with_lines
        else:
            return words

    def process_ocr_words(self, doc_dict):
        """
        Loads words from the document dictionary and potentially filters out bad ocr results
        :param doc_dict: Loaded json document with ocr words
        :return: List of Word objects
        """
        # Support OCR engine 2.0
        if 'objects' in doc_dict:
            self.logger.info('[OCRPreprocess] :: New OCR (2.0) entry found for file..')
            words = self.handle_new_ocr_format(doc_dict)
            words = [word for word in words if self._filter_by_text(word)]
        else: # Support OCR engine 1.0
            self.logger.info('[OCRPreprocess] :: Old OCR (1.0) entry found for file..')
            words = [Word(word_dict) for word_dict in doc_dict]
        area_all = np.array([Polygon(word.geometry.get_bb_corners()).area for word in words])
        self.area_threshold_for_ratio = np.percentile(area_all, 20)
        # Erase boxes where:
        # 1) The height is bigger than the width / ratio
        # 2) Empty text
        words = [word for word in words if self._single_word_filters(word)]
        word_against_all_pairs = [(word, words) for word in words]
        word_pairs = filter(self._filter_by_overlap, word_against_all_pairs)
        word_pairs = list(word_pairs)
        words = [entry[0] for entry in word_pairs]

        return words

    def __call__(self, doc_dict):
        return self.process_ocr_words(doc_dict)
