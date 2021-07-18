# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-4.0

from bokeh.layouts import column, widgetbox
from bokeh.models.widgets import RadioGroup
import matplotlib.colors as matplot_colors
import numpy as np

class VisualizationWidget:


    def __init__(self, widget_title, doc_figure = None, doc_obj = None):
        self.labels = ['input', 'phrases', 'clusters', 'pca_1d', 'pca_3d', 'auto_cannot_link_sample', 'auto_must_link_inter_phrase', 'must_link', 'cannot_link', 'ner']
        self.words = None
        self.phrases = None
        self.clusters = None
        self.doc_figure = doc_figure
        self.doc_obj = doc_obj
        self.visualization_radio_buttons = self.init_visualization_radio_buttons(self.labels)

        def on_change_selected_algo_radio_button(attrname, old, new):
            self.update_visualization()

        self.visualization_radio_buttons.on_change('active', on_change_selected_algo_radio_button)

        self.widget_component = column(self.visualization_radio_buttons)

    @staticmethod
    def init_visualization_radio_buttons(init_labels):
        labels = list(init_labels)
        active_idx = labels.index('clusters')
        return RadioGroup(labels=labels, active=active_idx)

    def widget(self):
        return self.widget_component

    def get_selected_visualization(self):
        active_idx = self.visualization_radio_buttons.active
        selected_widget_key = list(self.labels)[active_idx]
        return selected_widget_key

    def update_visualization(self):
        selected_widget_key = self.get_selected_visualization()
        self.doc_figure.hide_all_plots()
        if selected_widget_key == 'phrases':
            if self.doc_obj.phrases is not None:
                plot_name = 'phrases_plot_' + str(len(self.doc_figure.plots))
                self.doc_figure.draw_document_entities_colors_plot(doc_entities=self.doc_obj.phrases, plot_name=plot_name)
        elif selected_widget_key == 'clusters':
            if self.doc_obj.clusters is not None:
                plot_name = 'clusters_plot_' + str(len(self.doc_figure.plots))
                self.doc_figure.draw_document_entities_colors_plot(doc_entities=self.doc_obj.clusters, plot_name=plot_name)
        elif selected_widget_key == 'pca_1d':
            if self.doc_obj.phrases is not None:
                plot_name = 'visualize_plot_1d_' + str(len(self.doc_figure.plots))
                if hasattr(self.doc_obj.phrases[0], 'embedding') and self.doc_obj.phrases[0].embedding is not None:
                    self.doc_figure.show_pca_visualization(doc_entities=self.doc_obj.phrases, plot_name=plot_name,
                                                           num_pca_comp=1)
                else:
                    self.doc_figure.show_pca_visualization(doc_entities=self.doc_obj.words, plot_name=plot_name,
                                                           num_pca_comp=1)
        elif selected_widget_key == 'pca_3d':
            if self.doc_obj.phrases is not None:
                plot_name = 'visualize_plot_3d_' + str(len(self.doc_figure.plots))
                if hasattr(self.doc_obj.phrases[0], 'embedding') and self.doc_obj.phrases[0].embedding is not None:
                    self.doc_figure.show_pca_visualization(doc_entities=self.doc_obj.phrases, plot_name=plot_name,
                                                           num_pca_comp=3)
                else:
                    self.doc_figure.show_pca_visualization(doc_entities=self.doc_obj.words, plot_name=plot_name,
                                                           num_pca_comp=3)
        elif selected_widget_key == 'must_link':
            for constraints in self.doc_obj.mustlink_constraints:
                doc_entities = [self.doc_obj.get_words()[i] for i in constraints]
                self.doc_figure.draw_document_entities_grayscale_plot(doc_entities=doc_entities,
                                                                      plot_name='words_plot',
                                                                      hide_plots=False,
                                                                      fill_color="random",
                                                                      fill_alpha=0.85)
        elif selected_widget_key == 'auto_cannot_link_sample':
            print("--- auto_cannot_link_sample vis ---")
            constraints_length = len(self.doc_obj.auto_cannotlink_constraints)
            print("num of auto cannot link constrains: {}".format(constraints_length))
            ind = np.random.choice(range(constraints_length), size=10, replace=False)
            constraints_sample = [self.doc_obj.auto_cannotlink_constraints[i] for i in ind]

            for constraints in constraints_sample:
                doc_entities = [self.doc_obj.get_words()[i] for i in constraints]
                self.doc_figure.draw_document_entities_grayscale_plot(doc_entities=doc_entities,
                                                                      plot_name='words_plot',
                                                                      hide_plots=False,
                                                                      fill_color="random",
                                                                      fill_alpha=0.85)
        elif selected_widget_key == 'auto_must_link_inter_phrase':
            print("--- auto_must_link_inter_phrase vis ---")
            constraints_length = len(self.doc_obj.auto_mustlink_constraints)
            print("num of auto mknn mustlink constrains: {}".format(constraints_length))
            ind = np.random.choice(range(constraints_length), size=10, replace=False)
            constraints_sample = [self.doc_obj.auto_mustlink_constraints[i] for i in ind]

            for constraints in constraints_sample:
                doc_entities = [self.doc_obj.get_words()[i] for i in constraints]
                self.doc_figure.draw_document_entities_grayscale_plot(doc_entities=doc_entities,
                                                                      plot_name='words_plot',
                                                                      hide_plots=False,
                                                                      fill_color="random",
                                                                      fill_alpha=0.85)


        elif selected_widget_key == 'cannot_link':
            print("--- cannot_link vis ---")
            for constraints in self.doc_obj.cannotlink_constraints:
                first_nonlink_group = [self.doc_obj.get_words()[i] for i in constraints[0]]
                second_nonlink_group = [self.doc_obj.get_words()[i] for i in constraints[1]]
                doc_entities = first_nonlink_group + second_nonlink_group
                self.doc_figure.draw_document_entities_grayscale_plot(doc_entities=doc_entities,
                                                                      plot_name='words_plot',
                                                                      hide_plots=False,
                                                                      fill_color="random",
                                                                      fill_alpha=0.85)
        elif selected_widget_key == 'ner':
            print("--- ner vis ---")
            words = self.doc_obj.get_words()
            tags = [word.ner_tag for word in words]
            for tag_i in range(17):
                words_i = [word for word, tag in zip(words, tags) if tag == tag_i]
                if len(words_i) > 0:
                    self.doc_figure.draw_document_entities_grayscale_plot(doc_entities=words_i,
                                                                          plot_name='words_plot',
                                                                          hide_plots=False,
                                                                          fill_color="random",
                                                                      fill_alpha=0.85)

    def update_active_button(self, new_label):
        self.visualization_radio_buttons.active = self.labels.index(new_label)




