from bokeh.models.widgets import MultiSelect, CheckboxGroup, Select

geometry_embeddings_config = {

    "embedding_name": 'Geometry Embeddings',
    "is_editable": True,
    "parameters": {
        'word_level_embedding': {
            'type': CheckboxGroup,
            'labels': ['word_level_embedding'],
            'active': []
        },
        'phrase_level_embedding': {
            'type': CheckboxGroup,
            'labels': ['phrase_level_embedding'],
            'active': [0]
        }
    }
}
