# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-4.0

from bokeh.layouts import column, widgetbox
from bokeh.models.widgets import Slider
from bokeh.models.widgets import CheckboxGroup, RadioGroup


class ClusteringWidget:

    SLIDEBAR_ACTIVE_COLOR = '#415ea2'
    SLIDEBAR_INACTIVE_COLOR = '#e6e6e6'

    def __init__(self, widget_title, clustering_setup_config, default_algo):
        self.activate_clustering_widget_checkbox = CheckboxGroup(labels=[widget_title], active=[0])
        self.clustering_algos_radio_buttons = self.init_clustering_radio_buttons(clustering_setup_config, default_algo)
        self.clustering_algos_config_widgets = self.create_algo_config_boxes(clustering_setup_config)

        for clustering_algo_widget in self.clustering_algos_config_widgets.values():
            self.disable_widgetbox(clustering_algo_widget)

        self.enable_widgetbox(self.clustering_algos_config_widgets[default_algo])

        def on_click_activate_widget_checkbox_group(attrname, old, new):
            if 0 in self.activate_clustering_widget_checkbox.active:
                self.clustering_algos_radio_buttons.disabled = False
                self.toggle_config_widget()
            else:
                self.clustering_algos_radio_buttons.disabled = True
                for clustering_algo_widget in self.clustering_algos_config_widgets.values():
                    self.disable_widgetbox(clustering_algo_widget)

        self.activate_clustering_widget_checkbox.on_change('active', on_click_activate_widget_checkbox_group)

        def on_change_selected_algo_radio_button(attrname, old, new):
            self.toggle_config_widget()

        self.clustering_algos_radio_buttons.on_change('active', on_change_selected_algo_radio_button)

        self.widget_component = column(self.activate_clustering_widget_checkbox, self.clustering_algos_radio_buttons,
                                       *self.clustering_algos_config_widgets.values())

    @staticmethod
    def init_clustering_radio_buttons(clustering_algos, default_algo):
        labels = list(clustering_algos.keys())
        active_idx = labels.index(default_algo)
        return RadioGroup(labels=labels, active=active_idx)

    def create_algo_config_boxes(self, clustering_algos):
        algo_config_boxes = {}
        for algo_name, algo_hyperparams in clustering_algos.items():
            algo_components = []
            for single_param in algo_hyperparams.values():
                component_class = single_param['type']
                component_params = dict(single_param)
                del component_params['type']
                gui_component = component_class(**component_params)
                algo_components.append(gui_component)

            algo_config_boxes[algo_name] = widgetbox(algo_components)
        return algo_config_boxes

    @staticmethod
    def enable_widgetbox(component):
        for child_comp in component.children:
            child_comp.disabled = False
            if isinstance(child_comp, Slider):
                child_comp.bar_color = ClusteringWidget.SLIDEBAR_ACTIVE_COLOR

    @staticmethod
    def disable_widgetbox(component):
        for child_comp in component.children:
            if isinstance(child_comp, Slider):
                child_comp.bar_color = ClusteringWidget.SLIDEBAR_INACTIVE_COLOR
            child_comp.disabled = True

    def get_select_clustering_algo_config(self):
        active_idx = self.clustering_algos_radio_buttons.active
        selected_widget_key = list(self.clustering_algos_config_widgets.keys())[active_idx]
        return self.clustering_algos_config_widgets[selected_widget_key]

    def toggle_config_widget(self):
        selected_widget = self.get_select_clustering_algo_config()
        for clustering_algo_widget in self.clustering_algos_config_widgets.values():
            self.disable_widgetbox(clustering_algo_widget)
        self.enable_widgetbox(selected_widget)

    def widget(self):
        return self.widget_component

    def selected_algorithm(self):
        active_idx = self.clustering_algos_radio_buttons.active
        selected_widget_key = list(self.clustering_algos_config_widgets.keys())[active_idx]
        return selected_widget_key

    def hyperparams(self):
        hyperparams = {}
        for child_comp in self.get_select_clustering_algo_config().children:
            hyperparams[child_comp.title] = child_comp.value
        return hyperparams

    def is_active(self):
        return 0 in self.activate_clustering_widget_checkbox.active
