from matplotlib import pyplot as plt
import matplotlib.colors as matplot_colors
from matplotlib.patches import Rectangle
from sklearn.decomposition import PCA
import numpy as np
from multimodal_affinities.visualization.colors_util import colors_sampler, rgb_tuple_to_hex
import numpy as np

class VisHandler(object):
    def __init__(self):
        pass

    def add_word_bbox_to_ax(self, ax, image_width, image_height, words_list, labels = []):

        if len(labels) > 0 :
            u, labels = np.unique(labels, return_inverse=True)
            colors_generator = VisHandler.generate_colors_list(max(labels)+1)
        else:
            colors_generator = colors_sampler(colors_count=len(words_list))

        for i,word in enumerate(words_list):
            # if word.get('Geometry'):
            #     x0 = word['Geometry']['BoundingBox']['Left']
            #     y0 = word['Geometry']['BoundingBox']['Top']
            #     width = word['Geometry']['BoundingBox']['Width']
            #     height = word['Geometry']['BoundingBox']['Height']
            # else:
            x0, y0, width, height = word.get_bbox()
            y0 = int(y0 * image_height)
            x0 = int(x0 * image_width)
            height = int(height * image_height)
            width = int(width * image_width)
            if len(labels) > 0:
                facecolor = colors_generator[labels[i]]
                edgecolor = "black"
            else:
                facecolor = "white"
                edgecolor = next(colors_generator)
            rect = Rectangle((x0, y0), width, height,
                             facecolor=facecolor,
                             alpha=0.6,
                             edgecolor=edgecolor,
                             linewidth=0.5)

            ax.add_patch(rect)

    @staticmethod
    def get_word_bboxes_bokeh_x_y_w_h(words_list):
        x_list = list()
        y_list = list()
        width_list = list()
        height_list = list()
        for word in words_list:
            x0, y0, width, height = word.get_bbox()
            x_list.append(x0 + width / 2)
            y_list.append(y0 + height / 2)
            width_list.append(width)
            height_list.append(height)

        return x_list, y_list, width_list, height_list

    @staticmethod
    def generate_colors_list(amount):
        '''
        Generate a list of colors in str #RRGGBB format.
        #amount amount of colors is sampled from the RGB space, and sampled are spaced as much as possible.
        :param amount: Amount of colors to sample
        :return: List of str representing colors in ##RRGGBB format
        '''
        colors_generator = colors_sampler(colors_count=amount, min_color_val=0.02, max_color_val=0.98)
        colors = list(map(rgb_tuple_to_hex, colors_generator))  # Exhaust generator with all available colors

        if len(colors) > amount:
            colors = colors[:amount]

        return colors

    @staticmethod
    def generate_darker_palette(colors, min_color_val=0.02, max_color_val=0.98):
        """
        Generates another set of colors, slightly darker than the input
        :param colors:
        :param min_color_val:
        :param max_color_val:
        :return:
        """
        darker_colors = []
        dark_amount = (max_color_val - min_color_val) * 0.3
        for color in colors:
            hex_values = color.lstrip('#')
            r,g,b = tuple(int(hex_values[i:i + 2], 16) * 1.0 / 255 for i in (0, 2, 4))

            r = max(min_color_val, r - dark_amount)
            g = max(min_color_val, g - dark_amount)
            b = max(min_color_val, b - dark_amount)
            darker_colors.append(rgb_tuple_to_hex((r,g,b)))
        return darker_colors

    @staticmethod
    def get_pseudo_colors(embeddings, num_pca_comp):
        '''
        Gets pseudo colors by mapping PCA embedding on cool-warm colorbar
        :param embeddings:
        :return:
        '''
        print("--- get pseudo colors ---")
        embeddings_array = np.array(embeddings).squeeze().transpose()
        num_pca_comp = 3 if num_pca_comp > 1 else 1
        embeddings_1d = PCA(n_components=num_pca_comp).fit_transform(embeddings_array.transpose())
        embeddings_1d = embeddings_1d.transpose()
        min_1d = np.min(embeddings_1d)
        max_1d = np.max(embeddings_1d)
        cmap = plt.cm.get_cmap('coolwarm')
        colors = cmap(np.arange(cmap.N))
        colors_arr = []
        for i in range(embeddings_1d.shape[1]):
            norm_embedding = (embeddings_1d[:,i] - min_1d) / (max_1d - min_1d)
            if num_pca_comp > 1:
                colors_arr.append(matplot_colors.rgb2hex(norm_embedding))
            else:
                ind_selected = int(np.floor(norm_embedding * (cmap.N-1)))
                colors_arr.append(matplot_colors.rgb2hex(colors[ind_selected,:3]))
        return colors_arr

    @staticmethod
    def generate_default_colors_palette():
        '''
        Generate a predefined list of common colors in str #RRGGBB format.
        '''
        return [
            '#D93613',  # Red
            '#4FD913',  # Green
            '#1358D9',  # Blue
            '#D9A013',  # Yellow
            '#D913D9',  # Pink
            '#13CDD9',  # Teal
            '#7013D9',  # Purple
            '#6F1919',  # Brown
            '#6A7C80',  # Gray
            '#220428'   # Black
        ]
