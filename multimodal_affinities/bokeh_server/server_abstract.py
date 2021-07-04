import os, sys

def _assert_file_path(file_path):
    assert os.path.isfile(file_path), "no such file: {}".format(file_path)

try:
    import numpy as np
    import torch
    # =============================================
    # Bokeh server - main runnable script
    # =============================================

    SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    sys.path.append(SRC_DIR)

    from multimodal_affinities import common_config

    CACHE_DIR = common_config.CACHE_DIR
    USER_STUDY_ROOT = common_config.USER_STUDY_ROOT
    APP_DIR = common_config.CACHE_DIR
    UPDATE_VISUALIZATIONS_ON_CHANGE = False

    import matplotlib
    matplotlib.use('Agg')   # Required for gif animations
    from multimodal_affinities.pipeline.algorithm_api import CoreLogic
    from multimodal_affinities.bokeh_server.bokeh_logger import BokehLogger
    from multimodal_affinities.bokeh_server.clustering_widget import ClusteringWidget
    from multimodal_affinities.bokeh_server.embeddings_widget import EmbeddingsWidget
    from multimodal_affinities.bokeh_server.visualization_widget import VisualizationWidget
    from multimodal_affinities.bokeh_server.document_figure import DocumentFigure
    from multimodal_affinities.bokeh_server.config.input_config import input_config
    from multimodal_affinities.bokeh_server.config.phrase_detector_config import phrase_detector_config
    from multimodal_affinities.bokeh_server.config.cluster_entities_config import cluster_entities_config
    from multimodal_affinities.bokeh_server.config.font_embeddings_config import font_embeddings_config
    from multimodal_affinities.bokeh_server.config.nlp_embeddings_word_level_config import nlp_word_embeddings_config
    from multimodal_affinities.bokeh_server.config.nlp_embeddings_phrase_level_config import nlp_phrase_embeddings_config
    from multimodal_affinities.bokeh_server.config.geometry_embeddings_config import geometry_embeddings_config
    from multimodal_affinities.bokeh_server.config.combined_embeddings_config import combined_embeddings_config
    from multimodal_affinities.bokeh_server.config.embeddings_training_config import embeddings_training_config
    from multimodal_affinities.evaluation.analysis.clustering_measurements import doc_to_labels, measure_scores, cluster_accuracy

    from multimodal_affinities.evaluation.analysis.ground_truth_generator import gt_from_user_constraints
    from multimodal_affinities.evaluation.analysis.plots_producer import PlotsProducer
    from bokeh.io import curdoc
    from bokeh.layouts import row, column, widgetbox
    from bokeh.models.widgets import Button, PreText, Panel, Tabs, CheckboxGroup
    from bokeh.models import CustomJS, ColumnDataSource, TextInput
    import json
    import pickle
    import copy
    import random
    # Set up data


    logger = BokehLogger()
    core_logic = CoreLogic(logger)
    test_folder_path = input_config['data_folder']
    file_basename = random.choice(input_config['file_basename'])
    image_file_path = os.path.join(test_folder_path, file_basename + ".png")
    json_file_path = os.path.join(test_folder_path, file_basename + ".json")
    _assert_file_path(json_file_path)
    doc = core_logic.load_document(doc_ocr_json_path=json_file_path, doc_img_path=image_file_path)
    initial_doc = None
    if input_config['is_user_study']:
        results_root_path = os.path.join(USER_STUDY_ROOT, 'pipeline_results')
        try:
            doc_pickle_path = os.path.join(results_root_path, doc.basename + '_clustered_document.pkl')
            print("loading clustered document pickle: {}".format(doc_pickle_path))
            _assert_file_path(doc_pickle_path)
            doc = doc.load_document_pickle(doc_pickle_path)
            initial_doc = copy.deepcopy(doc)
        except:
            logger.error("ERROR: could not load saved document pickle")
            raise ValueError('Invalid argument for user study')

        try:
            core_logic_path = os.path.join(results_root_path, doc.basename + '_corelogic.pkl')
            if os.path.isfile(core_logic_path):
                print("loading core logic (trainer) pickle: {}".format(core_logic_path))
                core_logic = CoreLogic.load_from_pickle(core_logic_path)
                core_logic.logger = logger
                core_logic.trainer.logger = logger
        except:
            logger.error("ERROR: could not load saved core logic (trainer) pickle")
            raise ValueError('Invalid argument for user study')

        # try:
        #     algorithm_config_path = os.path.join(USER_STUDY_ROOT, 'config', 'algorithm_config.json')
        #     print("loading algorithm config json: {}".format(algorithm_config_path))
        #     with open(algorithm_config_path, 'r') as json_file:
        #         algorithm_config = json.load(json_file)
        # except:
        #     logger.error("ERROR: missing algorithm configuration file")
        #     raise ValueError('Invalid argument for user study')
        input_config['is_user_study'] = False

    image_file_path = doc.image_path    # Might have been updated due to different image format

    # Set up widgets
    document_figure = DocumentFigure(image_path=image_file_path, image_resize_height=800, figure_name="Phrase Groups")
    document_word_groups_figure = DocumentFigure(image_path=image_file_path, image_resize_height=800, figure_name="Select Must/ Must Not Link")

    document_2d_embedding_vis = DocumentFigure(image_path=image_file_path, image_resize_height=800, figure_name="Embeddings Vis",
                                               x_range=[-0.5, 0.5], y_range=[-0.5, 0.5])
    document_2d_embedding_vis_deep = DocumentFigure(image_path=image_file_path, image_resize_height=800, figure_name="Deep Embeddings Vis",
                                                    x_range=[-0.5, 0.5], y_range=[-0.5, 0.5])
    document_loss_plot = DocumentFigure(image_path=image_file_path, image_resize_height=800, figure_name="Loss Plot Vis",
                                        x_range=[0, 100], y_range=[0, 1])

    phrase_detector_widget = ClusteringWidget(widget_title="Phrase Detection",
                                              clustering_setup_config=phrase_detector_config,
                                              default_algo="Graph Clustering")
    clustering_phrases_widget = ClusteringWidget(widget_title="Cluster Phrases",
                                                 clustering_setup_config=cluster_entities_config,
                                                 default_algo="Graph Clustering")
    visualization_widget = VisualizationWidget(widget_title="Visualization", doc_figure=document_figure, doc_obj = doc)

    font_embeddings_widget = EmbeddingsWidget(widget_title='Font embeddings',
                                              embedding_setup_config=font_embeddings_config)
    nlp_word_embeddings_widget = EmbeddingsWidget(widget_title='NLP embeddings (word)',
                                                  embedding_setup_config=nlp_word_embeddings_config)
    nlp_phrase_embeddings_widget = EmbeddingsWidget(widget_title='NLP embeddings (phrase)',
                                                    embedding_setup_config=nlp_phrase_embeddings_config)
    geom_embeddings_widget = EmbeddingsWidget(widget_title='Geometry embeddings',
                                              embedding_setup_config=geometry_embeddings_config)
    combined_embeddings_widget = ClusteringWidget(widget_title='Combined embeddings',
                                                  clustering_setup_config=combined_embeddings_config,
                                                  default_algo="mlp")
    embeddings_training_widget = EmbeddingsWidget(widget_title='Embeddings Training',
                                                  embedding_setup_config=embeddings_training_config)

    run_cluster_button = Button(label="Run Pipeline", button_type="success")
    rerun_cluster_button = Button(label="ReRun Clustering", button_type="success")
    refine_button = Button(label="Refine Results", button_type="success")
    export_configuration_button = Button(label="Export Configuration", button_type="warning")
    export_results_button = Button(label="Export Results", button_type="warning")
    export_gt_button = Button(label="Export Ground Truth from User Const.", button_type="warning")
    export_user_study_results_button = Button(label="Export User Study Results", button_type="warning")
    instructions_must_link_button = PreText(
    text="""Use the selection tool to select\na "must-link" / "cannot-link" group\nthen register them by pressing the button\nbelow """)

    must_link_button = Button(label="Save Must-Link", button_type="primary")
    must_not_link_button = Button(label="Save Cannot-Link", button_type="primary")
    delete_last_mustlink_group_button = Button(label="Delete Last Must-Link Group", button_type="primary")
    delete_last_cannotlink_group_button = Button(label="Delete Last Cannot-Link Group", button_type="primary")
    save_constraints_button = Button(label="Save Constraints To Cache", button_type="primary")
    load_constraints_button = Button(label="Load Constraints From Cache", button_type="primary")

    toggle_word_boxes_checkbox = CheckboxGroup(labels=["Toggle Word boxes"], active=[])

    s2 = ColumnDataSource(data=dict(x=[], y=[]))  # for all the word boxes
    words_plot = document_figure.draw_document_entities_grayscale_plot(doc_entities=doc.get_words(), plot_name='words_plot',
                                                                       source=s2)
    words_plot.visible = False

    s1 = ColumnDataSource(data=dict(x=[], y=[]))  # for selected circles callback
    s3 = ColumnDataSource(data=dict(x=[], y=[], width=[], height=[]))  # for drawing the boxes
    s1_embd_vis = ColumnDataSource(dict(x=[], y=[]))
    s2_embd_vis = ColumnDataSource(dict(x=[], y=[]))
    s1_deep_embd_vis = ColumnDataSource(dict(x=[], y=[]))
    s2_deep_embd_vis = ColumnDataSource(dict(x=[], y=[]))
    loss_data_source = ColumnDataSource(data=dict(x=[], y=[]))

    document_word_groups_figure.draw_document_entities_grayscale_plot(doc_entities=doc.get_words(), plot_name='words_plot',
                                                                      source=s3, hide_plots=False)

    document_word_groups_figure.draw_document_entities_circles(doc_entities=doc.get_words(), plot_name='words_plot',
                                                               source=s1, hide_plots=False)

    selected_indices = TextInput(title="Selected data indices", value='', disabled=True)
    num_selected_indices = TextInput(title="Number of selected data indices", value='', disabled=True)

    s1.selected.js_on_change('indices', CustomJS(args=dict(s1=s1,
                                                           s2=s2,
                                                           s3=s3,
                                                           s1_deep_embd_vis=s1_deep_embd_vis,
                                                           s2_deep_embd_vis=s2_deep_embd_vis,
                                                           s1_embd_vis=s1_embd_vis,
                                                           s2_embd_vis=s2_embd_vis,
                                                           selected_indices=selected_indices,
                                                           num_selected_indices=num_selected_indices), code="""
            var inds = cb_obj.indices;
            var d2 = s2.data;
            var d3 = s3.data;
            var d1_d_embed = s1_deep_embd_vis.data;
            var d2_d_embed = s2_deep_embd_vis.data;
            var d1_embed = s1_embd_vis.data;
            var d2_embed = s2_embd_vis.data;
            d3['x'] = []
            d3['y'] = []
            d3['width'] = []
            d3['height'] = []
            
            d2_d_embed['x'] = []
            d2_d_embed['y'] = []
            d2_embed['x'] = []
            d2_embed['y'] = []
            
            
            for (var i = 0; i < inds.length; i++) {
                d3['x'].push(d2['x'][inds[i]])
                d3['y'].push(d2['y'][inds[i]])
                d3['width'].push(d2['width'][inds[i]])
                d3['height'].push(d2['height'][inds[i]])
                
                d2_embed['x'].push(d1_embed['x'][inds[i]])
                d2_embed['y'].push(d1_embed['y'][inds[i]])
                d2_d_embed['x'].push(d1_d_embed['x'][inds[i]])
    -           d2_d_embed['y'].push(d1_d_embed['y'][inds[i]])
            }
            selected_indices.value = inds.toString(); 
            num_selected_indices.value = Object.keys(inds).length.toString();
            s3.change.emit();
            s2_embd_vis.change.emit();
            s2_deep_embd_vis.change.emit();
        """)
                             )

    algo_gui_to_logic_names = {
        'Graph Clustering': 'graph',
        'DBScan': 'dbscan'
    }


    """
        Set callbacks
    """

    def get_embedding_params():
        embedding_params = {}
        if nlp_word_embeddings_widget.is_active():
            embedding_params['nlp_word_embeddings'] = nlp_word_embeddings_widget.hyperparams()
        if nlp_phrase_embeddings_widget.is_active():
            embedding_params['nlp_phrase_embeddings'] = nlp_phrase_embeddings_widget.hyperparams()
        if font_embeddings_widget.is_active():
            embedding_params['font_embeddings'] = font_embeddings_widget.hyperparams()
        if geom_embeddings_widget.is_active():
            embedding_params['geometry_embeddings'] = geom_embeddings_widget.hyperparams()
        if combined_embeddings_widget.is_active():
            embedding_params['combined_embeddings'] = {
                'strategy': combined_embeddings_widget.selected_algorithm(),
                'strategy_params': combined_embeddings_widget.hyperparams()
            }

            # Fix some parameters not exposed in bokeh, but expected by the logic.
            combined_params = embedding_params['combined_embeddings']['strategy_params']
            if 'font_dims' in combined_params:
                font_dim = combined_params['font_dims']
                combined_params['font_words_dims'] = font_dim
                combined_params['font_phrases_dims'] = font_dim
                del combined_params['font_dims']
            if 'geometry_dims' in combined_params:
                geometry_dim = combined_params['geometry_dims']
                combined_params['geometry_words_dims'] = geometry_dim
                combined_params['geometry_phrases_dims'] = geometry_dim
                del combined_params['geometry_dims']
        return embedding_params


    def get_training_params():
        training_params = embeddings_training_widget.hyperparams()
        ae_dims = [
            '-1' if not training_params['Autoenc. Layer 1'].isdigit() else training_params['Autoenc. Layer 1'],
            '-1' if not training_params['Autoenc. Layer 2'].isdigit() else training_params['Autoenc. Layer 2'],
            '-1' if not training_params['Autoenc. Layer 3'].isdigit() else training_params['Autoenc. Layer 3'],
            '-1' if not training_params['Autoenc. Layer 4'].isdigit() else training_params['Autoenc. Layer 4'],
        ]
        del training_params['Autoenc. Layer 1']
        del training_params['Autoenc. Layer 2']
        del training_params['Autoenc. Layer 3']
        del training_params['Autoenc. Layer 4']
        ae_dims = [int(dim) for dim in ae_dims]
        training_params['ae_dims'] = list(filter(lambda dim: dim > 0, ae_dims))
        if len(training_params['ae_dims']) == 0:
            del training_params['ae_dims']

        training_params['batch_size'] = int(training_params['batch_size'])
        training_params['learning_rate'] = float(training_params['learning_rate'])

        training_params['must_cannot_ratio'] = float(training_params['must_cannot_ratio'])
        training_params['font_word_mknn'] = int(training_params['font_word_mknn'])
        training_params['max_mustlink'] = int(training_params['max_mustlink'])

        if not embeddings_training_widget.is_active():
            training_params['epochs'] = 0

        return training_params


    def run_phrase_detection():
        print("---- running phrase detection ----")
        algorithm = algo_gui_to_logic_names[phrase_detector_widget.selected_algorithm()]
        hyperparams = phrase_detector_widget.hyperparams()

        print("Clustering params: " + str(hyperparams))
        phrases = core_logic.detect_phrases(document=doc, algorithm=algorithm, **hyperparams)
        doc.set_phrases(phrases)
        visualization_widget.update_active_button('phrases')
        return phrases


    def run_phrase_clustering(phrases):
        if not input_config['is_user_study']:
            algorithm = algo_gui_to_logic_names[clustering_phrases_widget.selected_algorithm()]
            clustering_hyperparams = clustering_phrases_widget.hyperparams()
        else:
            algorithm = algorithm_config["clustering"]["algorithm"]
            clustering_hyperparams = algorithm_config["clustering"]["parameters"]

        clusters = core_logic.cluster_entities(document=doc, clustering_algorithm=algorithm, **clustering_hyperparams)
        doc.set_clusters(clusters)
        print("--- show_pca_visualization_embedding_space ---")

        document_2d_embedding_vis_deep.show_pca_visualization_embedding_space(doc_entities=doc.get_words(),
                                                                              source_1=s1_deep_embd_vis,
                                                                              source_2=s2_deep_embd_vis,
                                                                              deep=True)
        # run again because of javascript callback
        document_2d_embedding_vis_deep.show_pca_visualization_embedding_space(doc_entities=doc.get_words(),
                                                                              source_1=s1_deep_embd_vis,
                                                                              source_2=s2_deep_embd_vis,
                                                                              deep=True)

        document_2d_embedding_vis.show_pca_visualization_embedding_space(doc_entities=doc.get_words(),
                                                                         source_1=s1_embd_vis,
                                                                         source_2=s2_embd_vis,
                                                                         deep=False)
        # run again because of javascript callback
        document_2d_embedding_vis.show_pca_visualization_embedding_space(doc_entities=doc.get_words(),
                                                                         source_1=s1_embd_vis,
                                                                         source_2=s2_embd_vis,
                                                                         deep=False)

        words_plot.visible = False
        return clusters


    def on_click_must_link():
        print("must link group: {}".format(selected_indices.value))
        inds_split_string = selected_indices.value.split(",")
        inds = [int(x) for x in inds_split_string]
        doc.add_must_link_constraints(inds)
        visualization_widget.update_active_button('must_link')
        if UPDATE_VISUALIZATIONS_ON_CHANGE:
            visualization_widget.update_visualization()

    must_link_button.on_click(on_click_must_link)


    non_link_selection = None
    def on_click_cannot_link():
        is_one_group_selected = must_not_link_button.label == 'Choose 2nd group'
        if not is_one_group_selected:
            print("must not link group: {}".format(selected_indices.value))
            inds_split_string = selected_indices.value.split(",")
            non_link_selection = [int(x) for x in inds_split_string]
            doc.aggregate_cannot_link_constraints(non_link_selection)
            must_not_link_button.label = 'Choose 2nd group'
        else:
            inds_split_string = selected_indices.value.split(",")
            non_link_selection = [int(x) for x in inds_split_string]
            doc.aggregate_cannot_link_constraints(non_link_selection)
            doc.submit_cannot_link_constraints()
            must_not_link_button.label = 'Save Cannot-Link'
        visualization_widget.update_active_button('cannot_link')
        if UPDATE_VISUALIZATIONS_ON_CHANGE:
            visualization_widget.update_visualization()

    must_not_link_button.on_click(on_click_cannot_link)

    def on_click_delete_last_mustlink_group():
        print("---- deleting last mustlink group ----")
        doc.remove_last_must_link_constraints()
        visualization_widget.update_active_button('must_link')
        if UPDATE_VISUALIZATIONS_ON_CHANGE:
            visualization_widget.update_visualization()


    delete_last_mustlink_group_button.on_click(on_click_delete_last_mustlink_group)

    def on_click_delete_last_cannotlink_group():
        print("---- deleting last cannot group ----")
        doc.remove_last_cannot_link_constraints()
        visualization_widget.update_active_button('cannot_link')
        if UPDATE_VISUALIZATIONS_ON_CHANGE:
            visualization_widget.update_visualization()


    delete_last_cannotlink_group_button.on_click(on_click_delete_last_cannotlink_group)

    def on_click_save_constraints_to_cache_group():
        print("---- saving constraints to cache ----")
        constraints = doc.get_constraints()
        constraints_root_path = os.path.join(CACHE_DIR, 'user_constraints')
        if not os.path.exists(constraints_root_path):
            os.makedirs(constraints_root_path)
        cache_fn = os.path.join(constraints_root_path, doc.basename + '.json')
        with open(cache_fn, 'w') as file_ptr:
            json.dump(constraints, file_ptr, indent=4)
        logger.info('Constraints saved successfully to %r.' % constraints_root_path)

    save_constraints_button.on_click(on_click_save_constraints_to_cache_group)

    def on_click_load_constraints_from_cache_group():
        print("---- loading constraints from cache ----")
        constraints_root_path = os.path.join(CACHE_DIR, 'user_constraints')
        cache_fn = os.path.join(constraints_root_path, doc.basename + '.json')
        if os.path.exists(cache_fn):
            with open(cache_fn, 'r') as file_ptr:
                constraints = json.load(file_ptr)
                doc.set_mustlink_constraints(constraints.get('must_link', []))
                doc.set_cannotlink_constraints(constraints.get('must_not_link', []))
        logger.info('Constraints loaded successfully.')
        print('Constraints loaded successfully.')


    load_constraints_button.on_click(on_click_load_constraints_from_cache_group)


    def on_click_run_cluster_button():
        print("---- running clustering ----")
        try:
            core_logic.reset()

            if phrase_detector_widget.is_active():
                phrases = run_phrase_detection()
            else:
                phrases = doc.get_words()

            core_logic.extract_embeddings(document=doc, embedding_params=get_embedding_params())
            core_logic.refine_embeddings(document=doc,
                                         save_training_progress=False,
                                         **get_training_params())  # Invoke training

            if clustering_phrases_widget.is_active():
                clusters = run_phrase_clustering(phrases)

            embeddings_history = core_logic.trainer.embeddings_history
            animation_path = os.path.join(CACHE_DIR, 'plots', 'experiments_animations', doc.basename)
            PlotsProducer.animate_pca_embedding_space_for_clusters(document=doc, output_path=animation_path, embeddings_history=embeddings_history)

            full_plots_path = os.path.join(CACHE_DIR, 'plots', 'plots_with_crops', doc.basename)
            plotter = PlotsProducer(document=doc, output_path=full_plots_path)
            plotter.plot_clusters_and_embedding_space_with_crops(document=doc, output_path=full_plots_path, crops_per_cluster=4,)

            loss_history = core_logic.trainer.loss_history
            loss_data_source.data = {"x": range(len(loss_history)), "y": loss_history}
            document_loss_plot.draw_loss_plot(source=loss_data_source)
            logger.info('Ready!')

        except Exception as e:
            exc_info = sys.exc_info()
            logger.error('Error :: ' + str(exc_info[1]))
            raise exc_info[0].with_traceback(exc_info[1], exc_info[2])


    run_cluster_button.on_click(on_click_run_cluster_button)

    def on_click_refine_button():
        print("---- refining clustering ----")
        try:

            # Invoke training
            if not input_config['is_user_study']:
                core_logic.refine_embeddings(document=doc, **get_training_params())   # bokeh params
            else:
                core_logic.extract_embeddings(document=doc, embedding_params=algorithm_config['embeddings_initialization'])
                # refine_epochs = get_training_params()["epochs"]
                refine_epochs = 2
                core_logic.refine_embeddings(document=doc, refine_epochs=refine_epochs, **algorithm_config['embeddings_refinement'])  # loaded config

            if clustering_phrases_widget.is_active():
                clusters = run_phrase_clustering(doc.get_words())

            visualization_widget.update_active_button('clusters')
            visualization_widget.update_visualization()

            loss_history = core_logic.trainer.loss_history
            loss_data_source.data = {"x": range(len(loss_history)), "y": loss_history}
            document_loss_plot.draw_loss_plot(source=loss_data_source)
            logger.info('Ready!')

        except Exception as e:
            exc_info = sys.exc_info()
            logger.error('Error :: ' + str(exc_info[1]))
            raise exc_info[0].with_traceback(exc_info[1], exc_info[2])

    refine_button.on_click(on_click_refine_button)

    def on_update_clustering_threshold():
        try:
            if doc.clusters is None or len(doc.clusters) == 0:
                logger.warning('Warning :: Nothing to cluster. Run pipeline first..')
                return  # Nothing to refine
            if clustering_phrases_widget.is_active():
                logger.info('Re-running clustering phase')
                clusters = run_phrase_clustering(doc.get_words())

            visualization_widget.update_active_button('clusters')
            visualization_widget.update_visualization()

        except Exception as e:
            exc_info = sys.exc_info()
            logger.error('Error :: ' + str(exc_info[1]))
            raise exc_info[0].with_traceback(exc_info[1], exc_info[2])

    # -- Slider update mode --
    # threshold_slider = clustering_phrases_widget.clustering_algos_config_widgets['Graph Clustering'].children[0]
    # threshold_slider.callback_throttle = 1000
    # threshold_slider.on_change('value', on_update_clustering_threshold_slider)
    # -- Button update mode --
    rerun_cluster_button.on_click(on_update_clustering_threshold)

    def on_click_export_configuration_button():
        print("---- exporting configuration ----")
        try:
            config_root_path = os.path.join(CACHE_DIR, 'configurations')
            if not os.path.exists(config_root_path):
                os.makedirs(config_root_path)

            configuration = {
                'phrase_detection': {
                    'algorithm': algo_gui_to_logic_names[phrase_detector_widget.selected_algorithm()],
                    'parameters': phrase_detector_widget.hyperparams()
                },
                'embeddings_initialization': get_embedding_params(),
                'embeddings_refinement': get_training_params(),
                'clustering': {
                    'algorithm': algo_gui_to_logic_names[clustering_phrases_widget.selected_algorithm()],
                    'parameters': clustering_phrases_widget.hyperparams()
                },
                'user_constraints': doc.get_constraints()
            }

            config_path = os.path.join(config_root_path, doc.basename + '_config.json')
            with open(config_path, 'w') as file_ptr:
                json.dump(configuration, file_ptr, indent=4)
            logger.info('Configuration exported successfully to %r.' % config_path)

        except Exception as e:
            exc_info = sys.exc_info()
            logger.error('Error :: ' + str(exc_info[1]))
            raise exc_info[0].with_traceback(exc_info[1], exc_info[2])


    export_configuration_button.on_click(on_click_export_configuration_button)

    def on_click_export_gt_button():
        print("---- exporting gt ----")
        try:
            gt_root_path = os.path.join(CACHE_DIR, 'gt_root')
            if not os.path.exists(gt_root_path):
                os.makedirs(gt_root_path)

            clustering_labels = gt_from_user_constraints(doc)

            labels_path = os.path.join(gt_root_path, doc.basename + '_gt_labels.pkl')
            with open(labels_path, 'wb') as output:
                pickle.dump(clustering_labels, output, pickle.HIGHEST_PROTOCOL)

            plots_path = os.path.join(CACHE_DIR, 'plots', 'gt', doc.basename)
            pp = PlotsProducer(document=doc,
                               output_path=plots_path)
            pp.save_clustering_labels(clustering_labels)

            logger.info('GT Labels results exported successfully to %r. \n'
                        'Plot available at %r' % (labels_path, plots_path))
        except Exception as e:
            exc_info = sys.exc_info()
            logger.error('Error :: ' + str(exc_info[1]))
            raise exc_info[0].with_traceback(exc_info[1], exc_info[2])

    export_gt_button.on_click(on_click_export_gt_button)


    def on_click_export_results_button():
        print("---- exporting results ----")
        try:
            results_root_path = os.path.join(CACHE_DIR, 'pipeline_results')
            if not os.path.exists(results_root_path):
                os.makedirs(results_root_path)

            analysis_results = doc.jsonify()

            doc_json_path = os.path.join(results_root_path, doc.basename + '_results.json')
            with open(doc_json_path, 'w') as file_ptr:
                json.dump(analysis_results, file_ptr, indent=4)

            output_doc_path = os.path.join(results_root_path, doc.basename + '_clustered_document.pkl')
            with open(output_doc_path, 'wb') as output:
                pickle.dump(doc, output, pickle.HIGHEST_PROTOCOL)

            output_logic_path = os.path.join(results_root_path, doc.basename + '_corelogic.pkl')
            with open(output_logic_path, 'wb') as output:
                pickle.dump(core_logic, output, pickle.HIGHEST_PROTOCOL)

            logger.info('Document json exported successfully to %r. \n'
                        'Document pickle exported successfully to %r. \n' 
                        'CoreLogic state exported successfully to %r' %
                        (doc_json_path, output_doc_path, output_logic_path))

        except Exception as e:
            exc_info = sys.exc_info()
            logger.error('Error :: ' + str(exc_info[1]))
            raise exc_info[0].with_traceback(exc_info[1], exc_info[2])

    export_results_button.on_click(on_click_export_results_button)

    def on_click_export_user_study_results_button():
        print("---- exporting user study results ----")
        if input_config['is_user_study']:
            try:
                results_dict = dict()
                results_dict["initial_clusters"] = doc_to_labels(initial_doc)
                results_dict["user_clusters"] = doc_to_labels(doc)
                acc, ami, nmi = measure_scores(initial_doc, doc)
                results_dict["ami"] = ami
                results_dict["nmi"] = nmi
                results_dict["acc"] = acc
                results_dict["user_constraints"] = doc.get_constraints()

                # save results
                file_base_name_w_rand_suffix = "{:s}_user_study_{:07d}".format(doc.basename, np.random.randint(10**7))

                output_folder = input_config['user_study_output_dir']
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                results_path = os.path.join(output_folder, file_base_name_w_rand_suffix)
                with open(results_path, 'w') as file_ptr:
                    json.dump(results_dict, file_ptr, indent=4)

                logger.info('Results exported successfully - please refresh the page')
                print("---- user study results: \n ami {}, ami {}, acc {}".format(ami, nmi, acc))
                print("---- exporting user study results - DONE ----")

            except Exception as e:
                exc_info = sys.exc_info()
                logger.error('Error :: ' + str(exc_info[1]))
                raise exc_info[0].with_traceback(exc_info[1], exc_info[2])

    export_user_study_results_button.on_click(on_click_export_user_study_results_button)


    def on_click_checkbox_button_group(attrname):
        if 0 in toggle_word_boxes_checkbox.active:
            words_plot.visible = True
        else:
            words_plot.visible = False


    toggle_word_boxes_checkbox.on_click(on_click_checkbox_button_group)

    """
        Set up widget box layouts and create document
    """
    if not input_config['is_user_study']:
        default_menu_widgets = widgetbox(run_cluster_button, rerun_cluster_button,
                                         export_configuration_button, export_results_button, export_gt_button)
        right_menu = column(instructions_must_link_button,
                            must_link_button,
                            must_not_link_button,
                            delete_last_mustlink_group_button,
                            delete_last_cannotlink_group_button,
                            save_constraints_button,
                            load_constraints_button,
                            selected_indices,
                            num_selected_indices,
                            refine_button,
                            export_user_study_results_button)

        font_tab = Panel(child=font_embeddings_widget.widget(), title="Font")
        nlp_word_tab = Panel(child=nlp_word_embeddings_widget.widget(), title="NLP (Word)")
        nlp_phrase_tab = Panel(child=nlp_phrase_embeddings_widget.widget(), title="NLP (Phrase)")
        geometry_tab = Panel(child=geom_embeddings_widget.widget(), title="Geom.")
        combined_tab = Panel(child=combined_embeddings_widget.widget(), title="Mix")
        embeddings_menu = Tabs(tabs=[font_tab, geometry_tab, nlp_word_tab, nlp_phrase_tab, combined_tab], height=1100)


        left_menu = column(visualization_widget.widget(),
                           default_menu_widgets,
                           phrase_detector_widget.widget(),
                           embeddings_menu)
        left_menu = row(left_menu, column(clustering_phrases_widget.widget(), embeddings_training_widget.widget()))
        logger_widget = logger.widget()

        analysis_tabs = column(Tabs(tabs=[
            Panel(child=document_2d_embedding_vis.get_figure(), title="Original Embeddings"),
            Panel(child=document_2d_embedding_vis_deep.get_figure(), title="Trained Embeddings"),
            Panel(child=document_loss_plot.get_figure(), title="Loss Curve")
        ]), width=document_2d_embedding_vis.get_figure().plot_width)


        main_layout = column(logger_widget,
                             row(left_menu,
                                 document_figure.get_figure(),
                                 document_word_groups_figure.get_figure(),
                                 analysis_tabs,
                                 right_menu,
                                 width=1600))

    else:
        default_menu_widgets = widgetbox(refine_button,
                                         instructions_must_link_button,
                                         must_link_button,
                                         must_not_link_button,
                                         delete_last_mustlink_group_button,
                                         delete_last_cannotlink_group_button,
                                         # save_constraints_button,
                                         # load_constraints_button,
                                         selected_indices,
                                         export_user_study_results_button)

        left_menu = column(visualization_widget.widget(),
                           default_menu_widgets)
        left_menu = row(left_menu)
        logger_widget = logger.widget()

        analysis_tabs = column(Tabs(tabs=[
            Panel(child=document_2d_embedding_vis.get_figure(), title="Original Embeddings"),
            Panel(child=document_2d_embedding_vis_deep.get_figure(), title="Trained Embeddings"),
            Panel(child=document_loss_plot.get_figure(), title="Loss Curve")
        ]), width=document_2d_embedding_vis.get_figure().plot_width)

        main_layout = column(logger_widget,
                             row(left_menu,
                                 document_figure.get_figure(),
                                 document_word_groups_figure.get_figure(),
                                 analysis_tabs,
                                 width=1600))

    curdoc().add_root(main_layout)
    curdoc().title = "Interactive ML"

    print('Server ready.')

except Exception as e:
    exc_info = sys.exc_info()
    print('Error :: ' + str(exc_info[1]))
    raise exc_info[0].with_traceback(exc_info[1], exc_info[2])
