# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-4.0

import itertools
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from multimodal_affinities.visualization.image_utils import ImageUtils
from multimodal_affinities.visualization.vis_handler import VisHandler
from multimodal_affinities.blocks.phrase import Phrase
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
import numpy as np
from sklearn.decomposition import PCA
import torch


class DocumentFigure:
    """ A class for managing the document figure in bokeh, containing the image as a background and plots of information
        on top.
    """
    def __init__(self, image_path="", image_resize_height=800, figure_name="", x_range=[0,1], y_range=[1, 0]):
        # Load background image
        input_image, image_size = ImageUtils.read_image_for_bokeh(image_path, resize_height=image_resize_height)
        input_image_width, input_image_height = image_size
        aspect_ratio = float(input_image_height) / input_image_width

        if not figure_name.endswith("Vis"):
            tools = "lasso_select, reset, save"
        else:
            tools = "reset, save, wheel_zoom, box_zoom, pan"

        # Set up the main figure
        self.doc_figure = figure(plot_height=image_resize_height,
                                 plot_width=int(image_resize_height / aspect_ratio),
                                 title=figure_name,
                                 tools=tools,
                                 x_range=x_range, y_range=y_range,
                                 output_backend="webgl")

        if not figure_name.endswith("Vis"):
           # doc_figure the image on the background
            self.doc_figure.image_rgba(image=[input_image], x=0, y=1, dw=1, dh=1)

        self.plots = []

    def draw_document_entities_colors_plot(self, doc_entities, plot_name, hide_plots=True):
        if type(doc_entities[0]) == Phrase:
            # get phrases as singleton lists
            entities_list = [[phrase] for phrase in doc_entities]
        else:
            # get words from clusters
            entities_list = [cluster.words for cluster in doc_entities]
        colors_list = VisHandler.generate_colors_list(amount=len(entities_list))
        x_list_all = []
        y_list_all = []
        width_list_all = []
        height_list_all = []
        color_list_all = []
        for i,phrase_list in enumerate(entities_list):
            x_list, y_list, width_list, height_list = VisHandler.get_word_bboxes_bokeh_x_y_w_h(phrase_list)
            x_list_all.append(x_list)
            y_list_all.append(y_list)
            width_list_all.append(width_list)
            height_list_all.append(height_list)
            color_list_all.append([colors_list[i]] * len(x_list))
        source = ColumnDataSource(data=dict(x=list(itertools.chain.from_iterable(x_list_all)),
                                            y=list(itertools.chain.from_iterable(y_list_all)),
                                            width=list(itertools.chain.from_iterable(width_list_all)),
                                            height=list(itertools.chain.from_iterable(height_list_all)),
                                            colors=list(itertools.chain.from_iterable(color_list_all))))
        entities_plot = self.doc_figure.rect('x', 'y', 'width', 'height', source=source,
                                             line_color='colors', fill_color='colors',
                                             angle=0, fill_alpha=0.9,
                                             line_width=2, line_alpha=0.7,
                                             name=plot_name)
        if hide_plots:
            self.hide_all_plots()
        self.plots.append(entities_plot)
        return entities_plot

    def draw_document_entities_circles(self, doc_entities, plot_name, source, hide_plots=True):
        x_list, y_list, width_list, height_list = VisHandler.get_word_bboxes_bokeh_x_y_w_h(doc_entities)
        source.data = dict(x=x_list, y=y_list)
        entities_plot = self.doc_figure.circle('x', 'y', source=source,alpha=0.6, size=5, color="white")

        if hide_plots:
            self.hide_all_plots()
        self.plots.append(entities_plot)
        return entities_plot

    def draw_document_entities_grayscale_plot(self, doc_entities=None, plot_name="no_name", source=None, hide_plots=True, fill_color="#cab2d6",fill_alpha=0.4):
        if doc_entities:
            x_list, y_list, width_list, height_list = VisHandler.get_word_bboxes_bokeh_x_y_w_h(doc_entities)

        if doc_entities and source:
            source.data = dict(x=x_list, y=y_list, width=width_list, height=height_list)
        elif doc_entities and not source:
            source = ColumnDataSource(data=dict(x=x_list, y=y_list, width=width_list, height=height_list))
        else:
            source = ColumnDataSource(data=source.data)

        if fill_color == "random":
            num_of_colors = 20
            cmap = plt.get_cmap("tab20", num_of_colors)
            rand_ind = np.random.randint(num_of_colors)
            rgb = cmap(rand_ind)[:3]
            hex = rgb2hex(rgb)
            fill_color = hex

        entities_plot = self.doc_figure.rect('x', 'y', 'width', 'height', source=source,
                                             line_color="black", fill_color=fill_color,
                                             angle=0, fill_alpha=fill_alpha,
                                             line_width=1.5, line_alpha=0.7,
                                             name=plot_name)
        if hide_plots:
            self.hide_all_plots()
        self.plots.append(entities_plot)
        return entities_plot

    def show_pca_visualization(self, doc_entities, plot_name, num_pca_comp = 3, hide_plots = True):
        embeddings = [entity.embedding.detach().cpu().numpy() for entity in doc_entities]
        embedding_pseudo_colors = VisHandler.get_pseudo_colors(embeddings, num_pca_comp)
        x_list, y_list, width_list, height_list = VisHandler.get_word_bboxes_bokeh_x_y_w_h(doc_entities)
        source = ColumnDataSource(data=dict(x=x_list,
                                            y=y_list,
                                            width=width_list,
                                            height=height_list,
                                            colors=embedding_pseudo_colors))
        entities_plot = self.doc_figure.rect('x', 'y', 'width', 'height', source=source,
                                             line_color='black', fill_color='colors',
                                             angle=0, fill_alpha=0.9,
                                             line_width=2, line_alpha=0.7,
                                             name=plot_name)
        if hide_plots:
            self.hide_all_plots()
        self.plots.append(entities_plot)
        return entities_plot

    def show_pca_visualization_embedding_space(self, doc_entities, source_1, source_2=None, deep=True):
        self.hide_all_plots()

        if deep:
            if doc_entities[0].embedding is None:
                return
            embeddings = [entity.embedding.detach().cpu().numpy() for entity in doc_entities]
        else:
            if doc_entities[0].unprojected_embedding is None:
                return
            embeddings = []
            for entity in doc_entities:
                unprojected_embedding = torch.cat(entity.unprojected_embedding['embeddings'], dim=1)
                unprojected_embedding = unprojected_embedding.detach().cpu().numpy()
                embeddings.append(unprojected_embedding)

        embeddings_array = np.array(embeddings).squeeze().transpose()
        num_pca_comp = 2
        Y = PCA(n_components=num_pca_comp).fit_transform(embeddings_array.transpose())
        embeddings_2d = Y.transpose()
        x_list = [embeddings_2d[0 , i] for i in range(len(embeddings_2d[0]))]
        y_list = [embeddings_2d[1 ,i] for i in range(len(embeddings_2d[0]))]

        # color_list = ["blue" for _ in range(len(embeddings_2d[0]))]
        # print(x_list)
        # print(y_list)

        source_1.data = dict(x=x_list, y=y_list)

        entities_plot2 = self.doc_figure.circle('x', 'y', color='red', source=source_2,alpha=0.4, size=10)
        self.plots.append(entities_plot2)

        entities_plot = self.doc_figure.circle('x', 'y', color='blue', source=source_1, alpha=0.6, size=5)
        self.plots.append(entities_plot)

        # Arrange axes range
        self.doc_figure.x_range.start = min(x_list)
        self.doc_figure.x_range.end = max(x_list)
        self.doc_figure.y_range.start = min(y_list)
        self.doc_figure.y_range.end = max(y_list)
        print("x_range {:.2f}:{:.2f}, y_range {:.2f}:{:.2f}, deep={}, len(x_list):{}, len(y_list):{}".format(
            min(x_list), max(x_list), min(y_list), max(y_list), deep, len(x_list), len(y_list)))

    def draw_loss_plot(self, source):
        loss_plot = self.doc_figure.line('x', 'y', source=source)
        self.plots.append(loss_plot)
        if len(source.data["y"]) > 0:
            self.doc_figure.x_range.start = min(source.data["x"])
            self.doc_figure.x_range.end = max(source.data["x"])
            self.doc_figure.y_range.start = min(source.data["y"])
            self.doc_figure.y_range.end = max(source.data["y"])

    def hide_all_plots(self):
        for fig_plot in self.plots:
            fig_plot.visible = False

    def get_figure(self):
        return self.doc_figure

    def get_plots(self):
        return self.plots

