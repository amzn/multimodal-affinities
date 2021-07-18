# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-4.0

from bokeh.layouts import column, widgetbox
from bokeh.models.widgets import Slider
from bokeh.models.widgets import CheckboxGroup


class EmbeddingsWidget:

    SLIDEBAR_ACTIVE_COLOR = '#415ea2'
    SLIDEBAR_INACTIVE_COLOR = '#e6e6e6'

    def __init__(self, widget_title, embedding_setup_config):
        if not embedding_setup_config['is_editable']:
            active = []
        else:
            active = [0]
        self.activate_embedding_widget_checkbox = CheckboxGroup(labels=[widget_title], active=active)

        self.embeddings_params_config_widgets = self.create_embedding_config_boxes(embedding_setup_config['parameters'])

        for clustering_algo_widget in self.embeddings_params_config_widgets.values():
            self.enable_widgetbox(clustering_algo_widget)

        def on_click_activate_widget_checkbox_group(attrname, old, new):
            if 0 in self.activate_embedding_widget_checkbox.active:
                for clustering_algo_widget in self.embeddings_params_config_widgets.values():
                    self.enable_widgetbox(clustering_algo_widget)
            else:
                for clustering_algo_widget in self.embeddings_params_config_widgets.values():
                    self.disable_widgetbox(clustering_algo_widget)

        self.activate_embedding_widget_checkbox.on_change('active', on_click_activate_widget_checkbox_group)

        self.widget_component = column(self.activate_embedding_widget_checkbox,
                                       *self.embeddings_params_config_widgets.values())

        if not embedding_setup_config['is_editable']:
            for clustering_algo_widget in self.embeddings_params_config_widgets.values():
                self.disable_widgetbox(clustering_algo_widget)

    def create_embedding_config_boxes(self, embedding_params):
        param_components = {}
        for param_name, single_param in embedding_params.items():
            component_class = single_param['type']
            component_params = dict(single_param)
            del component_params['type']
            gui_component = component_class(**component_params)
            param_components[param_name] = widgetbox(gui_component)
        return param_components

    @staticmethod
    def enable_widgetbox(component):
        for child_comp in component.children:
            child_comp.disabled = False
            if isinstance(child_comp, Slider):
                child_comp.bar_color = EmbeddingsWidget.SLIDEBAR_ACTIVE_COLOR

    @staticmethod
    def disable_widgetbox(component):
        for child_comp in component.children:
            if isinstance(child_comp, Slider):
                child_comp.bar_color = EmbeddingsWidget.SLIDEBAR_INACTIVE_COLOR
            child_comp.disabled = True

    def widget(self):
        return self.widget_component

    @staticmethod
    def _get_value_from_gui_comp(gui_comp):
        if isinstance(gui_comp, CheckboxGroup):
            return 0 in gui_comp.active
        else:
            return gui_comp.value

    @staticmethod
    def _get_title_from_gui_comp(gui_comp):
        if isinstance(gui_comp, CheckboxGroup):
            return gui_comp.labels[0]
        else:
            return gui_comp.title

    def hyperparams(self):
        hyperparams = {}
        for child_comp in self.embeddings_params_config_widgets.values():
            inner_comp = child_comp.children[0]
            hyperparams[self._get_title_from_gui_comp(inner_comp)] = self._get_value_from_gui_comp(inner_comp)
        return hyperparams

    def is_active(self):
        return 0 in self.activate_embedding_widget_checkbox.active
