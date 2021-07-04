import logging
import json
import argparse
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
from multimodal_affinities.pipeline.algorithm_api import CoreLogic
from multimodal_affinities.evaluation.analysis.plots_producer import PlotsProducer


def run_algorithm(doc_ocr_json_path, doc_img_path, algorithm_config):
    logger = logging.getLogger('multimodal')
    core_logic = CoreLogic(logger)
    document = core_logic.run_full_pipeline(doc_ocr_json_path=doc_ocr_json_path,
                                            doc_img_path=doc_img_path,
                                            algorithm_config=algorithm_config)
    return document, core_logic


def plot_results(document, output_path, core_logic, is_plot_clusters, is_plot_animation):

    if is_plot_clusters:
        logging.info('Generating result plots..')
        os.makedirs(name=output_path, exist_ok=True)
        plotter = PlotsProducer(document=document, output_path=output_path)

        colors_palette = \
            plotter.plot_clusters_and_embedding_space_with_crops(document=document, output_path=output_path,
                                                                 crops_per_cluster=0)

        logging.info('Generating optimization animation..')
        if is_plot_animation:
            embeddings_history = core_logic.trainer.embeddings_history
            PlotsProducer.animate_pca_embedding_space_for_clusters(document=document, output_path=output_path,
                                                                   embeddings_history=embeddings_history,
                                                                   colors_palette=colors_palette)

    logging.info(f'Plotting completed, available at {output_path}')


def load_config():
    algorithm_config_path = args.config
    with open(algorithm_config_path) as f:
        algorithm_config = json.load(f)
    return algorithm_config


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--img',
                        help='Path of the input image to process',
                        required=True)
    parser.add_argument('-c', '--ocr',
                        help='Path of the ocr engine json output, over the input image. Defaults to image name with json suffix',
                        required=False)
    parser.add_argument('-g', '--config',
                        help='Path of config files, with hyperparameters for algorithm',
                        required=False,
                        default='config/algorithm_config.json')
    parser.add_argument('-o', '--output',
                        help='Output root - result plots will be placed here. Defaults to working directory.',
                        required=False,
                        default='output')
    parser.add_argument('-np', '--noplot',
                        help='When specified, no result plots or animations will be generated under output root',
                        action='store_true',
                        default=False)
    parser.add_argument('-na', '--noanimation',
                        help='When specified, no optimization animations will be generated under output output.',
                        action='store_true',
                        default=False)
    args = parser.parse_args()
    if args.ocr is None:
        ocr_path = Path(args.img)
        ocr_path = os.path.join(str(ocr_path.parent), str(ocr_path.stem)) + '.json'
        args.ocr = ocr_path

    return args


if __name__ == '__main__':
    args = parse_arguments()
    algorithm_config = load_config()
    logging.info(f'Inference script running over image: {args.img} with ocr results {args.ocr}, using config at {args.config}')
    document, core_logic = run_algorithm(args.ocr, args.img, algorithm_config)

    clusters = document.get_clusters()
    logging.info(f'{len(clusters)} clusters found.')
    for idx, cluster in enumerate(clusters):
        logging.info(f'Cluster #{idx} with {len(cluster.text)} words: {cluster.text}')

    plot_results(document=document, output_path=args.output, core_logic=core_logic,
                 is_plot_clusters=not args.noplot, is_plot_animation=not args.noanimation)

    logging.info('Done.')