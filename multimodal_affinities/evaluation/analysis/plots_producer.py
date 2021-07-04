import os
import cv2
from collections import namedtuple
import imageio
from PIL import Image
from random import randrange
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
import torch
import matplotlib
matplotlib.use('Agg')   # Required for gif animations
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as image
import matplotlib.patches as patches
from multimodal_affinities.visualization.vis_handler import VisHandler
from multimodal_affinities.visualization.image_utils import resize_image
from multimodal_affinities.visualization.colors_util import rgb_hex_to_tuple

class PlotsProducer:

    def __init__(self, document, output_path):
        # Load background image
        self.image_path = document.image_path
        self.img = plt.imread(self.image_path)
        self.img_opencv = cv2.imread(self.image_path)

        dpi = 120
        mpl.rcParams['figure.dpi'] = dpi
        height = self.img.shape[0]
        width = self.img.shape[1]
        self.figsize = width / float(dpi), height / float(dpi)   # Fig size in inches

        self.document = document
        self.output_path = output_path
        if not os.path.exists(output_path):
            os.makedirs(output_path)

    def plot_word_boxes_on_image(self):
        set_of_words = [[word] for word in self.document.get_words()]  # list of singleton word lists
        fig, ax = plt.subplots(1, figsize=self.figsize)
        monochrome_colors_list = ['#5a5d8f' for _ in self.document.get_words()]
        self._draw_entity_bounding_boxes(fig=fig, ax=ax, bg_img=self.img,
                                         title='',
                                         entity_sets=set_of_words,
                                         colors_list=monochrome_colors_list)
        fig.savefig(os.path.join(self.output_path, self.document.basename + '_word_boxes.png'))
        plt.close(fig)

    def save_phrase_detection_results(self):
        set_of_phrases = [[phrase] for phrase in self.document.get_phrases()]  # list of singleton phrase lists
        fig, ax = plt.subplots(1, figsize=self.figsize)
        self._draw_entity_bounding_boxes(fig=fig, ax=ax, bg_img=self.img,
                                         title='Phrase Detection', entity_sets=set_of_phrases)
        fig.savefig(os.path.join(self.output_path, self.document.basename + '_phrase_detection.png'))
        plt.close(fig)

    def save_clustering_results(self, with_title=True, colors_list=None):
        set_of_clusters = [cluster.words for cluster in self.document.get_clusters()]  # list of list of words (clusters)
        self._save_set_of_clusters(set_of_clusters, with_title, colors_list)

    def save_clustering_labels(self, clustering_labels, colors_list=None):
        cluster_ids = np.unique(np.array(clustering_labels))
        cluster_id_to_cluster_idx = {cluster_id: idx for idx, cluster_id in enumerate(cluster_ids)}

        # Converts from list of labels to list of list of words (clusters)
        set_of_clusters = [[] for _ in range(len(cluster_ids))]
        for word_idx, word in enumerate(self.document.get_words()):
            cluster_id = clustering_labels[word_idx]
            if cluster_id == -1:    # Ignore non-clustered words
                continue
            cluster_idx = cluster_id_to_cluster_idx[cluster_id]
            set_of_clusters[cluster_idx].append(word)

        self._save_set_of_clusters(set_of_clusters, colors_list)

    def _save_set_of_clusters(self, set_of_clusters, with_title=True, colors_list=None):
        """
        :param document:
        :param set_of_clusters: list of list of words (clusters)
        :return:
        """
        output_img = self._draw_entity_bounding_boxes_opencv(bg_img=self.img_opencv,
                                                             entity_sets=set_of_clusters,
                                                             colors_list=colors_list)
        cv2.imwrite(os.path.join(self.output_path, self.document.basename + '_clustering.png'), output_img)

    @staticmethod
    def _draw_entity_bounding_boxes_opencv(bg_img, entity_sets, colors_list=None):

        img_height = bg_img.shape[0]
        img_width = bg_img.shape[1]

        if colors_list is None:
            colors_list = VisHandler.generate_colors_list(amount=len(entity_sets))

        face_colors = colors_list
        edge_colors = VisHandler.generate_darker_palette(colors_list)
        output_img = bg_img.copy()
        alpha = 0.8
        for set_idx, entities_set in enumerate(entity_sets):
            face_color = face_colors[set_idx]
            edge_color = edge_colors[set_idx]
            for entity in entities_set:
                x = entity.geometry.left * img_width
                y = entity.geometry.top * img_height
                width = entity.geometry.width * img_width
                height = entity.geometry.height * img_height
                # writing the text onto the image and returning it
                rgb_color = rgb_hex_to_tuple(face_color)
                cv2.rectangle(output_img, (int(x), int(y)), (int(x + width), int(y + height)),
                              (rgb_color[2], rgb_color[1], rgb_color[0]), cv2.FILLED)

        output_img = cv2.addWeighted(output_img, alpha, bg_img, 1 - alpha, 0)
        return output_img


    @staticmethod
    def _draw_entity_bounding_boxes(fig, ax, bg_img, title, entity_sets, colors_list=None):
        ax.set_title(title)

        plt.tick_params(axis='both', which='both',
                        bottom='off', top='off', labelbottom='off', right='off', left='off',
                        labelleft='off')

        plt.imshow(bg_img)
        img_height = bg_img.shape[0]
        img_width = bg_img.shape[1]

        if colors_list is None:
            colors_list = VisHandler.generate_colors_list(amount=len(entity_sets))

        face_colors = colors_list
        edge_colors = VisHandler.generate_darker_palette(colors_list)

        for set_idx, entities_set in enumerate(entity_sets):
            face_color = face_colors[set_idx]
            edge_color = edge_colors[set_idx]
            for entity in entities_set:
                x = entity.geometry.left * img_width
                y = entity.geometry.top * img_height
                width = entity.geometry.width * img_width
                height = entity.geometry.height * img_height
                rect = patches.Rectangle((x, y), width, height,
                                         linewidth=2,
                                         edgecolor=edge_color,
                                         facecolor=face_color,
                                         alpha=0.4)
                ax.add_patch(rect)

    @staticmethod
    def plot_pca_embedding_space_for_clusters(document, output_path,
                                              embedding_property='embedding',
                                              title=''):
        """
        Plot 2d PCA visualization of the embedding space according to cluster colors.
        :param document: Document with clustering results
        :param embedding_property: Embedding property of words - normally 'embedding' or 'unprojected_embedding'
        :return:
        """
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        words = document.get_words()
        clusters = document.get_clusters()

        if len(words) == 0 or getattr(words[0], embedding_property) is None:
            return
        if embedding_property == 'unprojected_embedding':
            embeddings = []
            for word in words:
                unprojected_embedding = torch.cat(word.unprojected_embedding['embeddings'], dim=1)
                unprojected_embedding = unprojected_embedding.detach().cpu().numpy()
                embeddings.append(unprojected_embedding)
        else:
            embeddings = [getattr(word, embedding_property).detach().cpu().numpy() for word in words]
        colors_palette = VisHandler.generate_colors_list(amount=len(clusters))
        word_to_color = {word: colors_palette[cluster_idx]
                         for cluster_idx, cluster in enumerate(clusters)
                         for word in cluster.words}
        colors = [word_to_color[word] for word in words]

        embeddings_array = np.array(embeddings).squeeze()
        num_pca_comp = 2
        embeddings_2d = PCA(n_components=num_pca_comp).fit_transform(embeddings_array)
        x_list = [embeddings_2d[i, 0] for i in range(embeddings_2d.shape[0])]
        y_list = [embeddings_2d[i, 1] for i in range(embeddings_2d.shape[0])]

        fig, ax = plt.subplots(1)
        plot_title = embedding_property
        if plot_title != '':
            plot_title += ': ' + title
        plt.title(plot_title)
        plt.scatter(x_list, y_list, c=colors, s=1, alpha=0.8)

        fig.tight_layout()
        fig.savefig(os.path.join(output_path, document.basename + '_' + embedding_property + '_pca.png'))
        plt.close(fig)

    @staticmethod
    def _find_k_furthest_words_per_cluster(document, embeddings_2d, k=3):
        """ Greedy approximation algorithm for finding k furthest neighbour words per cluster.
            k is expected to be relatively small (< 100)
        """
        words = document.get_words()
        word_to_embedding_2d_idx = {word: idx for idx, word in enumerate(words)}
        clusters = document.get_clusters()
        solution_per_cluster = {}
        ClusterSolution = namedtuple('ClusterSolution', ['word_indices', 'words'])
        for cluster in clusters:
            # Generate cluster pairwise distances matrix
            all_cluster_embeddings_indices = [word_to_embedding_2d_idx[word] for word in cluster.words]
            all_cluster_embeddings = np.take(embeddings_2d, all_cluster_embeddings_indices, axis=0)
            pairwise_distances = pdist(all_cluster_embeddings, metric='euclidean')
            distances_matrix = squareform(pairwise_distances)

            # Total distance from selected set so far
            distances_accumulator = np.zeros(len(cluster.words))

            # Sample first point
            random_index = randrange(len(cluster.words))

            # Indices of selected points
            selected_points = [random_index]

            # How many points we need to add
            points_to_calc_count = min(k - 1, len(words) - 1)

            for _ in range(points_to_calc_count):
                last_point_selected = selected_points[-1]

                # Update accumulator with distance collected from last point
                distances_accumulator += distances_matrix[last_point_selected]

                # Eliminate last point selected from distance matrix & accumulator
                distances_matrix[:, random_index] = 0
                distances_matrix[random_index, :] = 0

                furthrest_point_from_set = np.argmax(distances_accumulator, axis=0)
                selected_points.append(furthrest_point_from_set)

            selected_words = [cluster.words[point] for point in selected_points]
            selected_word_indices = [word_to_embedding_2d_idx[word] for word in selected_words]
            solution_per_cluster[cluster] = ClusterSolution(word_indices=selected_word_indices, words=selected_words)

        return solution_per_cluster

    @staticmethod
    def _extract_crops_per_cluster_solution(document, solution_per_cluster):
        """
        Extracts crops for each selected word in k-furthest neighbours solution
        :param document:
        :param solution_per_cluster: Solution of k-furthest neighbours
        :return:
        """
        word_indices_to_crops = {}
        for cluster, cluster_solution in solution_per_cluster.items():
            for word_index, word in zip(cluster_solution.word_indices, cluster_solution.words):
                bbox = word.get_bbox()  # left, top, width, height
                y_min = int(round(bbox[1] * document.height))
                y_max = int(round((bbox[1] + bbox[3]) * document.height))
                x_min = int(round(bbox[0] * document.width))
                x_max = int(round((bbox[0] + bbox[2]) * document.width))
                image_of_crop = document.image[max(0, y_min):min(y_max, document.height),
                                max(0, x_min):min(x_max, document.width), :]

                pil_image = Image.fromarray(image_of_crop[...,::-1])    # BGR to RGB
                pil_image = pil_image.convert('RGB')
                word_indices_to_crops[word_index] = pil_image
        return word_indices_to_crops

    @staticmethod
    def _space_out_crops(indices_to_crops, words, x_list, y_list, dist_from_pt=0.01, height=0.02):
        """
        Calculates the positions and dimensions of crop images on the embedding space plot.
        Makes sure crops don't overlay each other.
        This method assumes a small number of crops (< 1000) and performs a naive linear comparison for each crop.
        :param indices_to_crops: dict of word index (by order in doc) to PIL crop
        :param words: List of words
        :param x_list: List of corresponding pt x positions
        :param y_list: List of corresponding pt y positions
        :param dist_from_pt: How far in (x-y) coords the crop should be placed from the plot
        :param height: Height of the crop, in figure axes dimensions (note: for normalized pca space: -1 to 1)
        :return: indices_to_extents: dict of word index to extens describing position and dimensions of each crop.
        Crops are shifted so they don't cover each other,
        """
        indices_to_extents = {}
        MatplotExtent = namedtuple('matplot_extent', ['left', 'right', 'bottom', 'top'])
        is_extent_x_intersect = lambda e1, e2: not (e1.right < e2.left or e1.left > e2.right)
        is_extent_y_intersect = lambda e1, e2: not (e1.top > e2.bottom or e1.bottom < e2.top)
        is_extent_intersect = lambda e1, e2: is_extent_x_intersect(e1, e2) and is_extent_y_intersect(e1, e2)

        min_x, max_x = min(x_list), max(x_list)
        min_y, max_y = min(y_list), max(y_list)
        height = (max_y - min_y) * height
        dist_from_pt = min(max_y - min_y, max_x - min_x) * dist_from_pt

        for point_index, crop in indices_to_crops.items():
            word_aspect_ratio = words[point_index].geometry.width / words[point_index].geometry.height
            axis_ratio = (max_x-min_x) / (max_y-min_y) / 2
            width = height * word_aspect_ratio * axis_ratio
            left, right = x_list[point_index] + dist_from_pt, x_list[point_index] + dist_from_pt + width
            bottom, top = y_list[point_index] + dist_from_pt + height, y_list[point_index] + dist_from_pt

            overlap = True
            while overlap:
                overlap = False
                extent = MatplotExtent(left, right, bottom, top)
                for other_crop_extent in indices_to_extents.values():
                    other_left, other_right, other_bottom, other_top = other_crop_extent
                    spaceout_margin = dist_from_pt / 2
                    if is_extent_intersect(extent, other_crop_extent):
                        overlap = True
                        # shift below
                        if other_bottom <= top <= other_top:
                            top = other_bottom + spaceout_margin
                            bottom = top + height
                        else:   # shift above
                            bottom = other_top - spaceout_margin
                            top = bottom - height
                        continue

            indices_to_extents[point_index] = extent
        return indices_to_extents

    def plot_clusters_and_embedding_space_with_crops(self, document, output_path, crops_per_cluster=3,
                                                     embedding_properties=['embedding', 'unprojected_embedding'],
                                                     unprojected_caption=None):
        """
        Plot 2d PCA visualization of the embedding space according to cluster colors.
        :param document: Document with clustering results
        :param embedding_property: Embedding property of words - normally 'embedding' or 'unprojected_embedding'
        :return:
        """
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        words = document.get_words()
        clusters = document.get_clusters()

        if len(words) == 0 or \
                all([getattr(words[0], embedding_property) is None for embedding_property in embedding_properties]):
                    return

        colors_palette = VisHandler.generate_colors_list(amount=len(clusters))
        word_to_color = {word: colors_palette[cluster_idx]
                         for cluster_idx, cluster in enumerate(clusters)
                         for word in cluster.words}
        colors = [word_to_color[word] for word in words]

        # Initially empty, the first embedding property we process will set those for all figures
        selected_word_crops_per_cluster = None
        indices_to_crops = None

        for embedding_property in embedding_properties:
            if embedding_property == 'unprojected_embedding':   # Can't handle tuples, concat them
                embeddings = []
                for word in words:
                    unprojected_embedding = torch.cat(word.unprojected_embedding['embeddings'], dim=1)
                    unprojected_embedding = unprojected_embedding.detach().cpu().numpy()
                    embeddings.append(unprojected_embedding)
            else:
                embeddings = [getattr(word, embedding_property).detach().cpu().numpy() for word in words]

            embeddings_array = np.array(embeddings).squeeze()
            num_pca_comp = 2
            embeddings_2d = PCA(n_components=num_pca_comp).fit_transform(embeddings_array)
            x_list = [embeddings_2d[i, 0] for i in range(embeddings_2d.shape[0])]
            y_list = [embeddings_2d[i, 1] for i in range(embeddings_2d.shape[0])]

            fig, ax = plt.subplots(1)
            if crops_per_cluster > 0:
                if selected_word_crops_per_cluster is None and indices_to_crops is None:   # Calculate per first attribute
                    selected_word_crops_per_cluster = PlotsProducer._find_k_furthest_words_per_cluster(document, embeddings_2d, k=crops_per_cluster)
                    indices_to_crops = PlotsProducer._extract_crops_per_cluster_solution(document, selected_word_crops_per_cluster)
                indices_to_extents = PlotsProducer._space_out_crops(indices_to_crops, words,
                                                                    x_list, y_list, dist_from_pt=0.02, height=0.04)

                # Plot crop images
                for point_index, crop in indices_to_crops.items():
                    extent = indices_to_extents[point_index]
                    rect = patches.Rectangle((extent.left, extent.top), extent.right-extent.left, extent.bottom-extent.top,
                                             linewidth=0.5,
                                             edgecolor="black",
                                             facecolor="none",
                                             zorder=5)
                    ax.imshow(crop, aspect='auto', alpha=0.65, extent=extent, zorder=4)
                    ax.add_patch(rect)

            # Plot points
            if embedding_property == 'unprojected_embedding':
                plot_title = 'Initial unprojected embeddings, pre training (PCA)'
            else:
                if unprojected_caption is None:
                    plot_title = 'Projected embeddings, post training (PCA)'
                else:
                    plot_title = unprojected_caption
            plt.title(plot_title)
            plt.scatter(x_list, y_list, c=colors, s=18, alpha=1.0, edgecolors='black', linewidth=1.0, zorder=3)
            plt.tick_params(axis='both', which='both',
                            bottom='off', top='off', labelbottom='off', right='off', left='off',
                            labelleft='off')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            fig.tight_layout()
            fig.savefig(os.path.join(output_path, document.basename + '_' + embedding_property + '_pca.png'))
            plt.close(fig)

        # Finally plot clusters on original image
        self.save_clustering_results(with_title=False, colors_list=colors_palette)

        return colors_palette

    @staticmethod
    def animate_pca_embedding_space_for_clusters(document, output_path, embeddings_history, colors_palette=None):
        """
        Plot 2d PCA visualization of the embedding space according to cluster colors.
        :param document: Document with clustering results
        :param embedding_property: Embedding property of words - normally 'embedding' or 'unprojected_embedding'
        :return:
        """
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        words = document.get_words()
        clusters = document.get_clusters()

        if len(words) == 0 or embeddings_history is None or len(embeddings_history) == 0:
            return

        if colors_palette is None:
            colors_palette = VisHandler.generate_colors_list(amount=len(clusters))
        word_to_color = {word: colors_palette[cluster_idx]
                         for cluster_idx, cluster in enumerate(clusters)
                         for word in cluster.words}
        colors = [word_to_color[word] for word in words]
        scatter_data = []

        for state_idx, embeddings_state in enumerate(embeddings_history):
            epoch = state_idx + 1
            normalized_embeddings_dict = embeddings_state['normalized']
            unnormalized_embeddings_dict = embeddings_state['unnormalized']
            if len(normalized_embeddings_dict) > 0:
                normalized_embeddings = [normalized_embeddings_dict[word].detach().cpu().numpy() for word in words]
                chosen_embedding = normalized_embeddings
            elif len(unnormalized_embeddings_dict) > 0:
                unnormalized_embeddings = [unnormalized_embeddings_dict[word].detach().cpu().numpy() for word in words]
                chosen_embedding = unnormalized_embeddings
            else:
                return

            embeddings_array = np.array(chosen_embedding).squeeze()
            num_pca_comp = 2
            embeddings_2d = PCA(n_components=num_pca_comp).fit_transform(embeddings_array)
            x_list = [embeddings_2d[i, 0] for i in range(embeddings_2d.shape[0])]
            y_list = [embeddings_2d[i, 1] for i in range(embeddings_2d.shape[0])]
            push_pull_ratio = embeddings_state['push_pull_ratio']
            scatter_data.append((epoch, x_list, y_list, push_pull_ratio))

        min_x = min(min(scatter_data, key=lambda entry: min(entry[1]))[1])
        max_x = max(max(scatter_data, key=lambda entry: max(entry[1]))[1])
        min_y = min(min(scatter_data, key=lambda entry: min(entry[2]))[2])
        max_y = max(max(scatter_data, key=lambda entry: max(entry[2]))[2])
        padding_factor = 0.1
        min_x -= (max_x - min_x) * padding_factor
        max_x += (max_x - min_x) * padding_factor
        min_y -= (max_y - min_y) * padding_factor
        max_y += (max_y - min_y) * padding_factor

        frames = []
        for epoch, x_list, y_list, push_pull_ratio in scatter_data:
            fig, ax = plt.subplots(1)
            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_y, max_y)
            plot_title = 'Projected embeddings at epoch #' + str(epoch) + ' (PCA)'
            plt.title(plot_title)

            plt.scatter(x_list, y_list, c=colors, s=18, alpha=1.0, edgecolors='black', linewidth=1.0, zorder=3)
            plt.tick_params(axis='both', which='both',
                            bottom='off', top='off', labelbottom='off', right='off', left='off',
                            labelleft='off')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # Used to return the plot as an image rray
            fig.tight_layout()
            fig.canvas.draw()  # draw the canvas, cache the renderer
            output_frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            output_frame = output_frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(output_frame)

        imageio.mimsave(os.path.join(output_path, document.basename + '_embeddings_history.gif'), frames, fps=2)

