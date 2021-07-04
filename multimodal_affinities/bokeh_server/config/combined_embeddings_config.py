from bokeh.models.widgets import Select, Slider

combined_embeddings_config = {
    'concat': {
    },
    'kernel_pca': {
        'combine_kernelpca_n_components': {
            'type': Slider,
            'title': 'combine_kernelpca_n_components',
            'value': 10,
            'start': 2,
            'end': 100,
            'step': 1
        },
        'combine_kernelpca_kernel': {
            'type': Select,
            'title': 'combine_kernelpca_kernel',
            'options': ["linear", "poly", "rbf", "sigmoid", "cosine", "precomputed"],
            'value': "linear"
        }
    },
    'mlp': {
        'font': {
            'type': Slider,
            'title': 'font_dims',
            'value': 512,
            'start': 1,
            'end': 512,
            'step': 1
        },
        'nlp_words': {
            'type': Slider,
            'title': 'nlp_words_dims',
            'value': 4096,
            'start': 1,
            'end': 4096,
            'step': 1
        },
        'nlp_phrases': {
            'type': Slider,
            'title': 'nlp_phrases_dims',
            'value': 4096,
            'start': 1,
            'end': 4096,
            'step': 1
        },
        'geometry': {
            'type': Slider,
            'title': 'geometry_dims',
            'value': 4,
            'start': 1,
            'end': 20,
            'step': 1
        },
    }

}
